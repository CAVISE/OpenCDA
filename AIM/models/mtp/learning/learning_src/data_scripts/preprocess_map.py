import os
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from PIL import Image, ImageDraw
import math
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

from .data_config import config
from .generate_csv_utils import (
    get_map_bounding,
    get_entry_exit_edges,
    get_shortest_path,
    get_connection_via,
    get_connection_priority,
    get_connection_lanes,
)
from .preprocess_utils import normalize_coords, transform_coords


MAP_LANES_DTYPE = [("img", "O"), ("shape_points", "O"), ("map_boundings", "f4", (1,))]
MAP_MAP_DTYPE = [("img", "O"), ("map_boundings", "f4", (1,))]
MAP_OBJECT_DTYPE = [("lanes", "O"), ("map_boundings", "f4", (1,)), ("lanes_yaw", "O")]


def _get_by_id(root: Element, target_id: str) -> Optional[Element]:
    """
    find element by id in xml tree

    :param root: root element of xml tree
    :param target_id: target element id

    :return: element with matching id or None if not found
    """
    for elem in root.iter():
        if elem.get("id") == target_id:
            return elem
    return None


def _point2pixel(point: Tuple[float, float], min: float, max: float) -> Tuple[float, float]:
    """
    convert point coordinates to pixel coordinates

    :param point: point coordinates (x, y)
    :param min: minimum coordinate value
    :param max: maximum coordinate value

    :return: pixel coordinates (px, py)
    """
    scale = config.image_map.img_shape / (max - min)
    px = point[0] * scale
    py = point[1] * scale
    return (px + config.image_map.img_shape / 2, -py + config.image_map.img_shape / 2)


def _get_lane_by_id(edge_obj: Element, lane_id: str):
    if lane_id is None:
        return None

    for lane in edge_obj.findall("lane"):
        if lane.get("index") == lane_id:
            return lane

    return None


def _points_equal(p1: Tuple[float, float], p2: Tuple[float, float], eps: float = 1e-3) -> bool:
    return abs(p1[0] - p2[0]) <= eps and abs(p1[1] - p2[1]) <= eps


def _suffix_prefix_overlap_len(first_hop: List[Tuple[float, float]], second_hop: List[Tuple[float, float]]) -> int:
    """
    Largest m such that the last m points of first_hop equal the first m points of second_hop (pairwise, eps)
    """
    if not first_hop or not second_hop:
        return 0

    max_m = min(len(first_hop), len(second_hop))
    for m in range(max_m, 0, -1):
        ok = True

        for j in range(m):
            if not _points_equal(first_hop[len(first_hop) - m + j], second_hop[j]):
                ok = False
                break
        if ok:
            return m

    return 0


def _orient_segment_to_endpoint(
    segment: List[Tuple[float, float]], endpoint: Tuple[float, float], allow_reverse: bool = True
) -> Optional[List[Tuple[float, float]]]:
    """
    Check if segment has endpoint on its start or end points
    """
    if not segment:
        return None

    if _points_equal(segment[0], endpoint):
        return segment

    if allow_reverse and _points_equal(segment[-1], endpoint):
        return list(reversed(segment))

    return None


def _append_unique(base: List[Tuple[float, float]], segment: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not base:
        return list(segment)

    merged = list(base)
    if segment and _points_equal(merged[-1], segment[0]):
        merged.extend(segment[1:])
    else:
        merged.extend(segment)

    return merged


def _append_stitch_overlapping(base: List[Tuple[float, float]], segment: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Concatenate hop polylines: drop longest shared suffix/prefix run (may be multiple points)
    """
    if not base:
        return list(segment)

    if not segment:
        return list(base)

    m = _suffix_prefix_overlap_len(base, segment)
    merged = list(base)
    merged.extend(segment[m:])
    return merged


def _build_trajectory(
    from_points: List[Tuple[float, float]],
    to_points: List[Tuple[float, float]],
    via_segments: List[List[Tuple[float, float]]],
) -> Optional[List[Tuple[float, float]]]:
    if not from_points or not to_points:
        return None

    def _dfs(current: List[Tuple[float, float]], remaining: List[List[Tuple[float, float]]]) -> Optional[Tuple[List[Tuple[float, float]], int]]:
        current_end = current[-1]

        to_oriented = _orient_segment_to_endpoint(to_points, current_end, allow_reverse=True)
        if to_oriented is not None:
            finished = _append_unique(current, to_oriented)
            return finished, len(via_segments) - len(remaining)

        best_result = None
        best_used = -1
        for i, segment in enumerate(remaining):
            oriented = _orient_segment_to_endpoint(segment, current_end)
            if oriented is None:
                continue

            next_current = _append_unique(current, oriented)
            next_remaining = remaining[:i] + remaining[i + 1 :]
            result = _dfs(next_current, next_remaining)
            if result is None:
                continue

            candidate_path, used_count = result
            if used_count > best_used:
                best_result = candidate_path
                best_used = used_count

        if best_result is None:
            return None
        return best_result, best_used

    from_variants = [from_points, list(reversed(from_points))]
    best_path = None
    best_used = -1

    for start_variant in from_variants:
        result = _dfs(list(start_variant), via_segments)

        if result is None:
            continue

        candidate_path, used_count = result
        if used_count > best_used:
            best_path = candidate_path
            best_used = used_count

    return best_path


def _orient_next_to_stitch(prev_traj: List[Tuple[float, float]], next_traj: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    """
    Align next segment so it continues from prev_traj: pick forward vs reversed by
    longest suffix/prefix overlap (shared run can be one point or many)
    """
    if not prev_traj or not next_traj:
        return None

    forward = next_traj
    reversed_seg = list(reversed(next_traj))
    m_f = _suffix_prefix_overlap_len(prev_traj, forward)
    m_r = _suffix_prefix_overlap_len(prev_traj, reversed_seg)

    if m_f == 0 and m_r == 0:
        return None

    if m_r > m_f:
        return reversed_seg

    return forward


def _stitch_hop_trajectories(
    segments: List[List[List[Tuple[float, float]]]],
) -> List[List[Tuple[float, float]]]:
    """
    Chain trajectories across hops when hop lists have different lengths
    Next segment is chosen by longest suffix/prefix overlap of polylines (with optional reverse)

    :param segments: outer index = hop along path_edges; inner = all final_lane_trajectories for that hop
    """
    if not segments or any(len(h) == 0 for h in segments):
        return []

    if len(segments) == 1:
        return [list(t) for t in segments[0] if len(t) > 1]

    def _dfs(hop_idx: int, current: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        if hop_idx == len(segments) - 1:
            return [current] if len(current) > 1 else []
        out = []

        for nxt in segments[hop_idx + 1]:
            oriented = _orient_next_to_stitch(current, nxt)
            if oriented is None:
                continue

            merged = _append_stitch_overlapping(current, oriented)
            out.extend(_dfs(hop_idx + 1, merged))
        return out

    full_paths = []
    for t0 in segments[0]:
        if len(t0) <= 1:
            continue
        full_paths.extend(_dfs(0, list(t0)))

    seen = set()
    unique = []
    for path in full_paths:
        key = tuple(path)

        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


def preprocess_object_map_util(net_file_path: str) -> Tuple[List[np.ndarray], List]:
    """
    parsing lanes information from net.xml file

    :param net_file_path: path to sumo network xml file

    :return: list of lane objects and lane priorities
    """
    tree = ET.parse(net_file_path)
    root = tree.getroot()

    map_lanes_data_shapes = []
    map_lanes_data_priorities = []
    start_edges = get_entry_exit_edges(net_file_path)

    for start_edge, end_edges in start_edges.items():
        for end_edge in end_edges:
            done_edges = []
            path_str = get_shortest_path(net_file_path, start_edge, end_edge)
            path_edges = path_str.split(sep=" ")
            hop_segments = []

            for i, edge in enumerate(path_edges):
                if i == len(path_edges) - 1:
                    continue

                edge_elem_start = _get_by_id(root, edge)
                edge_elem_end = _get_by_id(root, path_edges[i + 1])
                from_lane_ids, to_lane_ids = get_connection_lanes(root, edge, path_edges[i + 1])

                from_lanes = [_get_lane_by_id(edge_elem_start, from_lane_id) for from_lane_id in from_lane_ids]
                to_lanes = [_get_lane_by_id(edge_elem_end, to_lane_id) for to_lane_id in to_lane_ids]

                done_edges.append(edge)
                done_edges.append(path_edges[i + 1])

                all_from_lane_points = []
                for lane in from_lanes:
                    shape_str = lane.get("shape", None)

                    if shape_str:
                        points = []
                        for p in shape_str.split():
                            x, y = map(float, p.split(","))
                            points.append((x, y))
                        all_from_lane_points.append(points)

                all_to_lane_points = []
                for lane in to_lanes:
                    shape_str = lane.get("shape", None)

                    if shape_str:
                        points = []
                        for p in shape_str.split():
                            x, y = map(float, p.split(","))
                            points.append((x, y))
                        all_to_lane_points.append(points)

                all_via_points = []
                if i < len(path_edges) - 1:
                    next_edge = path_edges[i + 1]
                    via_edge_ids = get_connection_via(root, edge, next_edge)

                    for via_edge_id in via_edge_ids:
                        if via_edge_id is None:
                            continue

                        via_edge = _get_by_id(root, via_edge_id)

                        if via_edge is None:
                            continue

                        for lane in via_edge.findall("lane"):
                            shape_str = lane.get("shape")

                            if shape_str:
                                via_points = []
                                for p in shape_str.split():
                                    x, y = map(float, p.split(","))
                                    via_points.append((x, y))
                                all_via_points.append(via_points)

                final_lane_trajectories = []
                for from_points, to_points in zip(all_from_lane_points, all_to_lane_points):
                    trajectory = _build_trajectory(from_points, to_points, all_via_points)
                    if trajectory is not None:
                        final_lane_trajectories.append(trajectory)

                hop_segments.append(final_lane_trajectories)

            stitched_paths = _stitch_hop_trajectories(hop_segments)
            for stitched in stitched_paths:
                if len(stitched) > 1:
                    map_lanes_data_shapes.append(np.array(stitched, dtype=np.float32))
                    connection_priority = math.prod(
                        [get_connection_priority(root, path_edges[i], path_edges[i + 1]) for i in range(len(path_edges) - 1)]
                    )
                    map_lanes_data_priorities.append(connection_priority)

    return map_lanes_data_shapes, map_lanes_data_priorities


def nolinear_time_function(t: np.ndarray):
    return t * (1 - t) * (0.5 - t)


def preprocess_object_map(net_file_path: str, output_dir: Optional[str] = None, save_png: bool = False) -> np.ndarray:
    """
    preprocess sumo network map and generate map representations at lanes level in image based encoding in object format

    :param net_file_path: path to sumo network xml file
    :param output_dir: directory to save preprocessed map data (optional)

    :return: lane_level_data
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    lane_object_representations = None
    lane_object_representations_yaw = None
    map_bounding = get_map_bounding(net_file_path)
    map_lanes_data_shapes, map_lanes_data_priorities = preprocess_object_map_util(net_file_path)

    for i, map_lanes_data_shape in enumerate(map_lanes_data_shapes):
        map_lanes_data_shape_masked = map_lanes_data_shape.copy()
        dots_dists = np.linalg.norm(map_lanes_data_shape_masked[:-1, :] - map_lanes_data_shape_masked[1:, :], axis=-1)
        dots_dists_cumsums = np.cumsum(dots_dists)
        t = dots_dists_cumsums / dots_dists_cumsums[-1]
        t = np.insert(t, 0, 0)

        map_lanes_data_shape_masked = transform_coords(map_lanes_data_shape_masked)
        interpolation_curve = make_interp_spline(t, map_lanes_data_shape_masked, k=1)

        t_new_linear = np.linspace(0, 1, config.object_map.n_lane_samples)
        # non-linear function adding to make shure that in the middle of lane there are enougth dots
        t_new = t_new_linear + config.object_map.non_linear_coeff * nolinear_time_function(t_new_linear)

        lane_points = interpolation_curve(t_new)
        lane_points_movment_orient = interpolation_curve(t_new + float(config.object_map.dt)) - lane_points
        lane_points_movment_orient_norms = np.linalg.norm(lane_points_movment_orient, axis=-1, keepdims=True)
        lane_points_movment_orient = lane_points_movment_orient / lane_points_movment_orient_norms

        lane_points_normalized = lane_points.copy()
        normalize_coords(lane_points_normalized, map_bounding)
        lane_object_representation = np.concatenate(
            [
                lane_points_normalized,
                lane_points_movment_orient,
                map_lanes_data_priorities[i] * np.ones((config.object_map.n_lane_samples, 1)),
            ],
            axis=-1,
        )
        lane_object_representation = np.expand_dims(lane_object_representation, 0)
        if lane_object_representations is None:
            lane_object_representations = lane_object_representation.copy()
        else:
            lane_object_representations = np.concatenate([lane_object_representations, lane_object_representation], axis=0)

        t_new_yaw_linear = np.linspace(0, 1, config.object_map.n_lane_samples_yaw)
        t_new_yaw = t_new_yaw_linear + config.object_map.non_linear_coeff * nolinear_time_function(t_new_yaw_linear)
        lane_points_yaw = interpolation_curve(t_new_yaw)
        lane_points_yaw_normalized = lane_points_yaw.copy()
        normalize_coords(lane_points_yaw_normalized, map_bounding)

        lane_object_representation_yaw = np.expand_dims(lane_points_yaw_normalized, 0)
        if lane_object_representations_yaw is None:
            lane_object_representations_yaw = lane_object_representation_yaw.copy()
        else:
            lane_object_representations_yaw = np.concatenate([lane_object_representations_yaw, lane_object_representation_yaw], axis=0)

    if output_dir and save_png:
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        for i, lane_object_representation in enumerate(lane_object_representations):
            ax.plot(
                lane_object_representation[:, 0],
                lane_object_representation[:, 1],
                color="red" if np.all(lane_object_representation[:, 4]) else "blue",
            )
            ax.quiver(
                lane_object_representation[:, 0],
                lane_object_representation[:, 1],
                lane_object_representation[:, 2] * 0.05,
                lane_object_representation[:, 3] * 0.05,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="black",
            )
            plt.savefig(os.path.join(output_dir, f"map_image_{i}.png"))

    lane_level_object_data = np.array([(lane_object_representations, map_bounding, lane_object_representations_yaw)], dtype=MAP_OBJECT_DTYPE)
    if output_dir is not None:
        np.save(
            os.path.join(output_dir, f"{config.image_map.map_lane_name}_object"),
            lane_level_object_data,
        )
    return lane_level_object_data, lane_object_representations_yaw


def preprocess_image_map(net_file_path: str, output_dir: Optional[str] = None, save_png: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    preprocess sumo network map and generate map representations at different levels in image based encoding in image format

    :param net_file_path: path to sumo network xml file
    :param output_dir: directory to save preprocessed map data (optional)
    :param save_png: flag to save png images of map parts and lanes

    :return: tuple of (part_level_data, lane_level_data, map_level_data)
    """
    map_lanes_data_shapes, map_lanes_data_priorities = preprocess_object_map_util(net_file_path)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    imgs = []
    map_bounding = get_map_bounding(net_file_path)

    main_img = Image.new("1", (config.image_map.img_shape, config.image_map.img_shape), config.image_map.bg_color)
    main_draw = ImageDraw.Draw(main_img)

    for i, map_lanes_data_shape in enumerate(map_lanes_data_shapes):
        img = Image.new("1", (config.image_map.img_shape, config.image_map.img_shape), config.image_map.bg_color)
        draw = ImageDraw.Draw(img)

        pixel_points = list(
            map(
                lambda point: _point2pixel(point, -map_bounding, map_bounding),
                map_lanes_data_shape,
            )
        )
        draw.line(pixel_points, fill=config.image_map.road_color, width=config.image_map.road_width_px, joint="curve")
        main_draw.line(pixel_points, fill=config.image_map.road_color, width=config.image_map.road_width_px, joint="curve")
        imgs.append(img)

        if save_png:
            img.save(os.path.join(output_dir, f"{config.image_map.map_img_name}_{i}.png"))

    map_level_data = np.array([(main_img, map_bounding)], dtype=MAP_MAP_DTYPE)
    if output_dir is not None:
        np.save(os.path.join(output_dir, config.image_map.map_img_name), map_level_data)

        if save_png:
            main_img.save(os.path.join(output_dir, f"{config.image_map.map_img_name}.png"))

    lane_level_data = np.array([(np.array(imgs), map_lanes_data_shapes, map_bounding)], dtype=MAP_LANES_DTYPE)
    if output_dir is not None:
        np.save(
            os.path.join(output_dir, config.image_map.map_lane_name),
            lane_level_data,
        )

    return lane_level_data, map_level_data
