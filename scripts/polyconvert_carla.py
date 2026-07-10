"""Generate a SUMO polygon file by scanning every streamed CARLA map region."""

from __future__ import annotations

import argparse
import collections
import hashlib
import logging
import math
import os
import re
import xml.etree.ElementTree as XmlTree
from typing import Any, Sequence


DEFAULT_LABELS = "Buildings,Walls,Vegetation,Foliage"
DEFAULT_CARLA_HOST = "localhost"
DEFAULT_CARLA_PORT = 2000
DEFAULT_CARLA_TIMEOUT = 30.0
DEFAULT_CARLA_MAP_NAME: str | None = None
DEFAULT_BOUNDS: tuple[float, float, float, float] | None = None
DEFAULT_BOUNDS_MARGIN = 0.0
DEFAULT_BOUNDARY_MARGIN = 150.0
DEFAULT_MAX_AREA = 5000.0
DEFAULT_MIN_HEIGHT = 1.0
DEFAULT_MAX_WALL_THICKNESS = 8.0
DEFAULT_STREAM_DISTANCE = 2000.0
DEFAULT_SCAN_STEP = 1800.0
DEFAULT_SETTLE_TICKS = 2
DEFAULT_SCAN_ALTITUDE = 500.0
DEFAULT_PRETTY_PRINT = False
DEFAULT_VERBOSE = False

OPENDRIVE_PARSE_CHUNK_SIZE = 65536
MIN_MAP_SPAN = 1.0
MIN_SETTLE_TICKS = 1
BBOX_EXTENT_SCALE = 2.0
MIN_BBOX_DIMENSION = 0.01
SIGNATURE_PRECISION = 3
OBJECT_HASH_LENGTH = 16
MIN_POLYGON_AREA = 0.01
COORDINATE_PRECISION = 2
HEIGHT_PRECISION = 2
POLY_FILL = "1"
OBJECT_ID_PATTERN = r"[^A-Za-z0-9_.-]+"
OBJECT_ID_REPLACEMENT = "_"
FALLBACK_OBJECT_ID = "object"
HASH_ENCODING = "ascii"
XML_ENCODING = "UTF-8"
XML_DECLARATION = True
XML_SCHEMA_INSTANCE_NS = "http://www.w3.org/2001/XMLSchema-instance"
SUMO_ADDITIONAL_SCHEMA = "http://sumo.dlr.de/xsd/additional_file.xsd"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
CARLA_STREAM_SETTING_NAMES = (
    "spectator_as_ego",
    "tile_stream_distance",
    "actor_active_distance",
)

LABEL_ALIASES = {
    "building": ("Buildings",),
    "buildings": ("Buildings",),
    "wall": ("Walls",),
    "walls": ("Walls",),
    "vegetation": ("Vegetation", "Foliage"),
    "foliage": ("Foliage", "Vegetation"),
    "tree": ("Vegetation", "Foliage"),
    "trees": ("Vegetation", "Foliage"),
    "park": ("Vegetation", "Foliage"),
    "parks": ("Vegetation", "Foliage"),
}

POLY_STYLE = {
    "Buildings": ("building", "120,150,180", "1.00"),
    "Walls": ("wall", "160,160,160", "1.00"),
    "Vegetation": ("forest", "green", "0.00"),
    "Foliage": ("forest", "green", "0.00"),
}


def _parse_bounds(value: str) -> tuple[float, float, float, float]:
    """Parse a CARLA bounding rectangle.

    Parameters
    ----------
    value : str
        Rectangle in ``min_x,min_y,max_x,max_y`` form.

    Returns
    -------
    tuple[float, float, float, float]
        Parsed rectangle coordinates.
    """
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bounds must be min_x,min_y,max_x,max_y")
    min_x, min_y, max_x, max_y = parts
    if min_x >= max_x or min_y >= max_y:
        raise argparse.ArgumentTypeError("bounds minimums must be smaller than maximums")
    return min_x, min_y, max_x, max_y


def _sumo_net_location(
    net_file: str,
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    """Read coordinate metadata from a SUMO network.

    Parameters
    ----------
    net_file : str
        Path to the SUMO network XML file.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float, float, float]]
        Network offset and conversion boundary.
    """
    if not os.path.isfile(net_file):
        raise FileNotFoundError(f"SUMO net file does not exist: {net_file}")
    for event in XmlTree.iterparse(net_file, events=("start",)):
        element = event[1]
        if element.tag.rsplit("}", 1)[-1] != "location":
            continue
        offset = tuple(float(value) for value in element.attrib["netOffset"].split(","))
        boundary = tuple(float(value) for value in element.attrib["convBoundary"].split(","))
        if len(offset) == 2 and len(boundary) == 4:
            return (offset[0], offset[1]), (boundary[0], boundary[1], boundary[2], boundary[3])
        break
    raise ValueError(f"SUMO net {net_file} has no valid location element")


def _prepare_output(output: str) -> None:
    """Validate the output path and create its parent directory.

    Parameters
    ----------
    output : str
        Destination polygon XML path.
    """
    output_path = os.path.abspath(output)
    if os.path.isdir(output_path):
        raise IsADirectoryError(f"poly output is a directory: {output}")
    output_dir = os.path.dirname(output_path)
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise NotADirectoryError(f"poly output parent is not a directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"poly output directory is not writable: {output_dir}")


def _opendrive_bounds(xodr: str) -> tuple[float, float, float, float]:
    """Extract CARLA-coordinate bounds from OpenDRIVE XML.

    Parameters
    ----------
    xodr : str
        OpenDRIVE XML document.

    Returns
    -------
    tuple[float, float, float, float]
        Minimum and maximum CARLA X/Y coordinates.
    """
    parser: XmlTree.XMLPullParser[XmlTree.Element] = XmlTree.XMLPullParser(events=("start",))
    for offset in range(0, len(xodr), OPENDRIVE_PARSE_CHUNK_SIZE):
        parser.feed(xodr[offset : offset + OPENDRIVE_PARSE_CHUNK_SIZE])
        for event in parser.read_events():
            parsed_event: Any = event
            element = parsed_event[1]
            if element.tag.rsplit("}", 1)[-1] != "header":
                continue
            west = float(element.attrib["west"])
            east = float(element.attrib["east"])
            south = float(element.attrib["south"])
            north = float(element.attrib["north"])
            return min(west, east), min(-north, -south), max(west, east), max(-north, -south)
    raise ValueError("OpenDRIVE data has no header with map bounds")


def _map_bounds(carla_map: Any) -> tuple[float, float, float, float]:
    """Determine complete CARLA map bounds.

    Parameters
    ----------
    carla_map : Any
        CARLA map object.

    Returns
    -------
    tuple[float, float, float, float]
        Minimum and maximum CARLA X/Y coordinates.
    """
    try:
        bounds = _opendrive_bounds(str(carla_map.to_opendrive()))
        if bounds[2] - bounds[0] > MIN_MAP_SPAN and bounds[3] - bounds[1] > MIN_MAP_SPAN:
            return bounds
    except (KeyError, TypeError, ValueError, XmlTree.ParseError) as error:
        logging.warning("Could not read OpenDRIVE bounds: %s. Falling back to spawn points.", error)
    points = [transform.location for transform in carla_map.get_spawn_points()]
    if not points:
        raise ValueError("map has neither usable OpenDRIVE bounds nor spawn points")
    return (
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def _axis_positions(minimum: float, maximum: float, step: float) -> list[float]:
    """Build scan coordinates that include both axis boundaries.

    Parameters
    ----------
    minimum : float
        Lower axis boundary.
    maximum : float
        Upper axis boundary.
    step : float
        Maximum spacing between positions.

    Returns
    -------
    list[float]
        Evenly spaced scan coordinates.
    """
    if step <= 0:
        raise ValueError("scan step must be positive")
    span = maximum - minimum
    segment_count = max(1, math.ceil(span / step))
    return [minimum + span * index / segment_count for index in range(segment_count + 1)]


def _scan_positions(bounds: tuple[float, float, float, float], step: float) -> list[tuple[float, float]]:
    """Build a serpentine scan path over a rectangle.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        Minimum and maximum X/Y coordinates.
    step : float
        Maximum spacing between adjacent positions.

    Returns
    -------
    list[tuple[float, float]]
        Ordered CARLA X/Y scan positions.
    """
    min_x, min_y, max_x, max_y = bounds
    x_positions = _axis_positions(min_x, max_x, step)
    y_positions = _axis_positions(min_y, max_y, step)
    positions: list[tuple[float, float]] = []
    for row, y in enumerate(y_positions):
        row_x = x_positions if row % 2 == 0 else reversed(x_positions)
        positions.extend((x, y) for x in row_x)
    return positions


def _resolve_labels(carla_module: Any, labels: str) -> list[tuple[str, Any]]:
    """Resolve labels against the installed CARLA API.

    Parameters
    ----------
    carla_module : Any
        Imported CARLA module.
    labels : str
        Comma-separated label names or aliases.

    Returns
    -------
    list[tuple[str, Any]]
        Canonical names paired with CARLA enum values.
    """
    resolved: list[tuple[str, Any]] = []
    seen_values: set[str] = set()
    for requested in (part.strip() for part in labels.split(",")):
        if not requested:
            continue
        candidates = LABEL_ALIASES.get(requested.lower(), (requested,))
        selected = next(
            ((name, getattr(carla_module.CityObjectLabel, name)) for name in candidates if hasattr(carla_module.CityObjectLabel, name)),
            None,
        )
        if selected is None:
            logging.warning("CARLA CityObjectLabel %s is unavailable; skipping it", requested)
            continue
        name, value = selected
        value_key = str(value)
        if value_key not in seen_values:
            seen_values.add(value_key)
            resolved.append((name, value))
    if not resolved:
        raise ValueError("none of the requested CityObjectLabel values is available")
    return resolved


def _map_identifier(map_name: str) -> str:
    """Return a normalized map identifier.

    Parameters
    ----------
    map_name : str
        CARLA map name or path.

    Returns
    -------
    str
        Case-folded basename without an extension.
    """
    normalized = map_name.replace("\\", "/").rstrip("/").split("/")[-1]
    return os.path.splitext(normalized)[0].casefold()


def _get_world(client: Any, requested_map_name: str | None) -> Any:
    """Get the current world and load a map only when necessary.

    Parameters
    ----------
    client : Any
        CARLA client.
    requested_map_name : str or None
        Requested map, or ``None`` to keep the current one.

    Returns
    -------
    Any
        Selected CARLA world.
    """
    world = client.get_world()
    if not requested_map_name:
        return world
    current_map_name = str(world.get_map().name)
    if _map_identifier(current_map_name) != _map_identifier(requested_map_name):
        logging.info("Loading CARLA map %s (current map: %s)", requested_map_name, current_map_name)
        return client.load_world(requested_map_name)
    logging.info("CARLA map %s is already loaded; keeping the current world", current_map_name)
    return world


def _box_vertices(box: Any, carla_module: Any) -> list[tuple[float, float, float]]:
    """Return world-space vertices for a CARLA bounding box."""
    return [(float(point.x), float(point.y), float(point.z)) for point in box.get_world_vertices(carla_module.Transform())]


def _box_passes_filters(
    box: Any,
    label: str,
    min_height: float,
    max_area: float,
    max_wall_thickness: float,
) -> bool:
    """Check whether a CARLA bounding box passes size filters."""
    width = BBOX_EXTENT_SCALE * float(box.extent.x)
    length = BBOX_EXTENT_SCALE * float(box.extent.y)
    height = BBOX_EXTENT_SCALE * float(box.extent.z)
    if height < min_height:
        return False
    if max_area > 0 and width * length > max_area:
        return False
    if label == "Walls" and max_wall_thickness > 0 and min(width, length) > max_wall_thickness:
        return False
    return width > MIN_BBOX_DIMENSION and length > MIN_BBOX_DIMENSION


def _collect_objects(
    world: Any,
    carla_module: Any,
    labels: str,
    bounds: tuple[float, float, float, float] | None,
    bounds_margin: float,
    stream_distance: float,
    scan_step: float,
    settle_ticks: int,
    timeout: float,
    scan_altitude: float,
    min_height: float,
    max_area: float,
    max_wall_thickness: float,
) -> list[dict[str, Any]]:
    """Scan streamed CARLA tiles and collect unique object boxes.

    Parameters
    ----------
    world : Any
        CARLA world to scan.
    carla_module : Any
        Imported CARLA module.
    labels : str
        Comma-separated semantic labels.
    bounds : tuple[float, float, float, float] or None
        Explicit scan bounds, or ``None`` to detect map bounds.
    bounds_margin : float
        Extra margin around scan bounds.
    stream_distance : float
        CARLA tile streaming distance.
    scan_step : float
        Maximum distance between scan positions.
    settle_ticks : int
        Ticks to wait after moving the spectator.
    timeout : float
        Timeout for CARLA tick operations.
    scan_altitude : float
        Spectator altitude during scanning.
    min_height : float
        Minimum object height.
    max_area : float
        Maximum footprint area; non-positive disables the limit.
    max_wall_thickness : float
        Maximum wall thickness; non-positive disables the limit.

    Returns
    -------
    list[dict[str, Any]]
        Unique objects with labels, identifiers, and box vertices.
    """
    if scan_step > stream_distance:
        logging.warning(
            "scan step %.1f exceeds stream distance %.1f; unloaded gaps may remain",
            scan_step,
            stream_distance,
        )
    selected_bounds = bounds or _map_bounds(world.get_map())
    min_x, min_y, max_x, max_y = selected_bounds
    scan_bounds = (
        min_x - bounds_margin,
        min_y - bounds_margin,
        max_x + bounds_margin,
        max_y + bounds_margin,
    )
    positions = _scan_positions(scan_bounds, scan_step)
    resolved_labels = _resolve_labels(carla_module, labels)
    spectator = world.get_spectator()
    original_transform = spectator.get_transform()
    original_settings = world.get_settings()
    original_values = {name: getattr(original_settings, name) for name in CARLA_STREAM_SETTING_NAMES if hasattr(original_settings, name)}
    retained: dict[tuple[str, tuple[tuple[float, float, float], ...]], dict[str, Any]] = {}
    try:
        settings = world.get_settings()
        if hasattr(settings, "spectator_as_ego"):
            settings.spectator_as_ego = True
        if hasattr(settings, "tile_stream_distance"):
            settings.tile_stream_distance = stream_distance
        if hasattr(settings, "actor_active_distance"):
            settings.actor_active_distance = max(stream_distance, float(settings.actor_active_distance))
        world.apply_settings(settings)
        synchronous_mode = bool(settings.synchronous_mode)
        for index, (x, y) in enumerate(positions, start=1):
            location = carla_module.Location(x=x, y=y, z=scan_altitude)
            spectator.set_transform(carla_module.Transform(location, original_transform.rotation))
            for _ in range(max(MIN_SETTLE_TICKS, settle_ticks)):
                if synchronous_mode:
                    world.tick(timeout)
                else:
                    world.wait_for_tick(timeout)
            added: collections.Counter[str] = collections.Counter()
            for label_name, label_value in resolved_labels:
                for box in world.get_level_bbs(label_value):
                    if not _box_passes_filters(box, label_name, min_height, max_area, max_wall_thickness):
                        continue
                    vertices = _box_vertices(box, carla_module)
                    signature_vertices = tuple(
                        sorted(
                            (
                                round(px, SIGNATURE_PRECISION),
                                round(py, SIGNATURE_PRECISION),
                                round(pz, SIGNATURE_PRECISION),
                            )
                            for px, py, pz in vertices
                        )
                    )
                    signature = (label_name, signature_vertices)
                    if signature in retained:
                        continue
                    digest = hashlib.sha1(repr(signature).encode(HASH_ENCODING)).hexdigest()[:OBJECT_HASH_LENGTH]
                    retained[signature] = {
                        "id": f"{label_name.lower()}-{digest}",
                        "label": label_name,
                        "vertices": [[px, py, pz] for px, py, pz in vertices],
                    }
                    added[label_name] += 1
            logging.info(
                "CARLA map scan %s/%s at (%.1f, %.1f): added=%s total=%s",
                index,
                len(positions),
                x,
                y,
                dict(added),
                len(retained),
            )
    finally:
        spectator.set_transform(original_transform)
        settings = world.get_settings()
        for name, value in original_values.items():
            setattr(settings, name, value)
        world.apply_settings(settings)
    return sorted(retained.values(), key=lambda item: (item["label"], item["id"]))


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute a two-dimensional convex hull."""
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _polygon_area(points: list[tuple[float, float]]) -> float:
    """Return the absolute area of a polygon."""
    if len(points) < 3:
        return 0.0
    return (
        abs(
            sum(
                points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1]
                for index in range(len(points))
            )
        )
        * 0.5
    )


def _minimum_edge_length(points: list[tuple[float, float]]) -> float:
    """Return the shortest polygon edge length."""
    return min(
        math.hypot(
            points[(index + 1) % len(points)][0] - point[0],
            points[(index + 1) % len(points)][1] - point[1],
        )
        for index, point in enumerate(points)
    )


def _footprint(item: dict[str, Any], offset: tuple[float, float]) -> tuple[list[tuple[float, float]], float]:
    """Convert CARLA box vertices into a SUMO footprint and height."""
    vertices = [(float(x), float(y), float(z)) for x, y, z in item["vertices"]]
    height = max(vertex[2] for vertex in vertices) - min(vertex[2] for vertex in vertices)
    points = [(vertex[0] + offset[0], -vertex[1] + offset[1]) for vertex in vertices]
    return _convex_hull(points), height


def _intersects_boundary(
    points: list[tuple[float, float]],
    boundary: tuple[float, float, float, float],
    margin: float,
) -> bool:
    """Check whether a polygon intersects the expanded SUMO boundary."""
    min_x, min_y, max_x, max_y = boundary
    return not (
        max(point[0] for point in points) < min_x - margin
        or min(point[0] for point in points) > max_x + margin
        or max(point[1] for point in points) < min_y - margin
        or min(point[1] for point in points) > max_y + margin
    )


def _write_poly(
    objects: Sequence[dict[str, Any]],
    output: str,
    offset: tuple[float, float],
    boundary: tuple[float, float, float, float],
    labels: str,
    boundary_margin: float,
    max_area: float,
    min_height: float,
    max_wall_thickness: float,
) -> str:
    """Write collected objects as a SUMO polygon XML file.

    Parameters
    ----------
    objects : Sequence[dict[str, Any]]
        Collected CARLA objects.
    output : str
        Destination polygon XML path.
    offset : tuple[float, float]
        SUMO coordinate offset.
    boundary : tuple[float, float, float, float]
        SUMO conversion boundary.
    labels : str
        Comma-separated labels to include.
    boundary_margin : float
        Margin around the SUMO boundary.
    max_area : float
        Maximum polygon area.
    min_height : float
        Minimum object height.
    max_wall_thickness : float
        Maximum wall thickness.

    Returns
    -------
    str
        Output path.
    """
    from lxml import etree

    root = etree.Element(
        "additional",
        {f"{{{XML_SCHEMA_INSTANCE_NS}}}noNamespaceSchemaLocation": SUMO_ADDITIONAL_SCHEMA},
    )
    requested_labels = {
        next((name for name in LABEL_ALIASES.get(label.strip().lower(), (label.strip(),)) if name in POLY_STYLE), label.strip())
        for label in labels.split(",")
        if label.strip()
    }
    indices: collections.Counter[str] = collections.Counter()
    retained_count = 0
    for item in objects:
        label = str(item["label"])
        if label not in requested_labels:
            continue
        footprint, height = _footprint(item, offset)
        area = _polygon_area(footprint)
        if len(footprint) < 3 or area <= MIN_POLYGON_AREA or height < min_height:
            continue
        if max_area > 0 and area > max_area:
            continue
        if label == "Walls" and max_wall_thickness > 0 and _minimum_edge_length(footprint) > max_wall_thickness:
            continue
        if not _intersects_boundary(footprint, boundary, boundary_margin):
            continue
        poly_type, color, layer = POLY_STYLE[label]
        source_id = re.sub(OBJECT_ID_PATTERN, OBJECT_ID_REPLACEMENT, str(item["id"])).strip("._-") or FALLBACK_OBJECT_ID
        shape = " ".join(f"{point[0]:.{COORDINATE_PRECISION}f},{point[1]:.{COORDINATE_PRECISION}f}" for point in footprint)
        poly = etree.SubElement(
            root,
            "poly",
            {
                "id": f"{poly_type}_{indices[label]}_{source_id}",
                "type": poly_type,
                "color": color,
                "fill": POLY_FILL,
                "layer": layer,
                "shape": shape,
            },
        )
        indices[label] += 1
        retained_count += 1
        etree.SubElement(poly, "param", {"key": "carlaLabel", "value": label})
        etree.SubElement(poly, "param", {"key": "carlaSourceId", "value": str(item["id"])})
        etree.SubElement(
            poly,
            "param",
            {"key": "carlaHeight", "value": f"{height:.{HEIGHT_PRECISION}f}"},
        )
    etree.ElementTree(root).write(
        output,
        pretty_print=DEFAULT_PRETTY_PRINT,
        encoding=XML_ENCODING,
        xml_declaration=XML_DECLARATION,
    )
    logging.info("Wrote %s CARLA environment polygons to %s", retained_count, output)
    return output


def generate_poly(
    net_file: str,
    output: str,
    carla_host: str = DEFAULT_CARLA_HOST,
    carla_port: int = DEFAULT_CARLA_PORT,
    carla_timeout: float = DEFAULT_CARLA_TIMEOUT,
    carla_map_name: str | None = DEFAULT_CARLA_MAP_NAME,
    labels: str = DEFAULT_LABELS,
    bounds: tuple[float, float, float, float] | None = DEFAULT_BOUNDS,
    bounds_margin: float = DEFAULT_BOUNDS_MARGIN,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
    stream_distance: float = DEFAULT_STREAM_DISTANCE,
    scan_step: float = DEFAULT_SCAN_STEP,
    settle_ticks: int = DEFAULT_SETTLE_TICKS,
    scan_altitude: float = DEFAULT_SCAN_ALTITUDE,
    min_height: float = DEFAULT_MIN_HEIGHT,
    max_area: float = DEFAULT_MAX_AREA,
    max_wall_thickness: float = DEFAULT_MAX_WALL_THICKNESS,
) -> str:
    """Generate SUMO polygons by scanning a running CARLA world.

    Parameters
    ----------
    net_file : str
        SUMO network used for coordinate conversion and clipping.
    output : str
        Destination polygon XML path.
    carla_host : str
        CARLA server host.
    carla_port : int
        CARLA server port.
    carla_timeout : float
        CARLA API timeout in seconds.
    carla_map_name : str or None
        Map to load when the current map differs.
    labels : str
        Comma-separated CARLA semantic labels.
    bounds : tuple[float, float, float, float] or None
        Explicit CARLA scan bounds.
    bounds_margin : float
        Extra margin around scan bounds.
    boundary_margin : float
        Extra margin around the SUMO boundary.
    stream_distance : float
        CARLA tile streaming distance.
    scan_step : float
        Maximum distance between scan positions.
    settle_ticks : int
        Ticks to wait after each spectator move.
    scan_altitude : float
        Spectator altitude during scanning.
    min_height : float
        Minimum object height.
    max_area : float
        Maximum footprint area.
    max_wall_thickness : float
        Maximum wall thickness.

    Returns
    -------
    str
        Output path.
    """
    # Validate every filesystem dependency before connecting to CARLA or scanning.
    offset, boundary = _sumo_net_location(net_file)
    _prepare_output(output)

    import carla

    client = carla.Client(carla_host, carla_port)
    client.set_timeout(carla_timeout)
    world = _get_world(client, carla_map_name)
    objects = _collect_objects(
        world,
        carla,
        labels,
        bounds,
        bounds_margin,
        stream_distance,
        scan_step,
        settle_ticks,
        carla_timeout,
        scan_altitude,
        min_height,
        max_area,
        max_wall_thickness,
    )
    return _write_poly(
        objects,
        output,
        offset,
        boundary,
        labels,
        boundary_margin,
        max_area,
        min_height,
        max_wall_thickness,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-file", required=True, help="existing SUMO net file")
    parser.add_argument("--output", "--poly-output", "-o", required=True, help="output SUMO poly file")
    parser.add_argument("--carla-host", default=DEFAULT_CARLA_HOST)
    parser.add_argument("--carla-port", type=int, default=DEFAULT_CARLA_PORT)
    parser.add_argument("--carla-timeout", type=float, default=DEFAULT_CARLA_TIMEOUT)
    parser.add_argument("--carla-map", default=DEFAULT_CARLA_MAP_NAME)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--bounds", type=_parse_bounds, default=DEFAULT_BOUNDS)
    parser.add_argument("--bounds-margin", type=float, default=DEFAULT_BOUNDS_MARGIN)
    parser.add_argument("--boundary-margin", type=float, default=DEFAULT_BOUNDARY_MARGIN)
    parser.add_argument("--stream-distance", type=float, default=DEFAULT_STREAM_DISTANCE)
    parser.add_argument("--scan-step", type=float, default=DEFAULT_SCAN_STEP)
    parser.add_argument("--settle-ticks", type=int, default=DEFAULT_SETTLE_TICKS)
    parser.add_argument("--scan-altitude", type=float, default=DEFAULT_SCAN_ALTITUDE)
    parser.add_argument("--min-height", type=float, default=DEFAULT_MIN_HEIGHT)
    parser.add_argument("--max-area", type=float, default=DEFAULT_MAX_AREA)
    parser.add_argument("--max-wall-thickness", type=float, default=DEFAULT_MAX_WALL_THICKNESS)
    parser.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE)
    return parser


def main() -> None:
    """Run the command-line polygon converter."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format=LOG_FORMAT,
    )
    try:
        generate_poly(
            args.net_file,
            args.output,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            carla_timeout=args.carla_timeout,
            carla_map_name=args.carla_map,
            labels=args.labels,
            bounds=args.bounds,
            bounds_margin=args.bounds_margin,
            boundary_margin=args.boundary_margin,
            stream_distance=args.stream_distance,
            scan_step=args.scan_step,
            settle_ticks=args.settle_ticks,
            scan_altitude=args.scan_altitude,
            min_height=args.min_height,
            max_area=args.max_area,
            max_wall_thickness=args.max_wall_thickness,
        )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, ValueError, XmlTree.ParseError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
