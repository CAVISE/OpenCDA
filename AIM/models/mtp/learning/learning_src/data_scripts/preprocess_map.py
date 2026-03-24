from PIL import Image, ImageDraw
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from xml.etree.ElementTree import Element

from .data_config import config
from .generate_csv_utils import get_map_bounding, get_entry_exit_edges, get_shortest_path, get_connection_via


MAP_PARTS_DTYPE = [("img", "O"), ("shape_points", "O"), ("center", "O"), ("map_boundings", "f4", (1,))]
MAP_LANES_DTYPE = [("img", "O"), ("shape_points", "O"), ("map_boundings", "f4", (1,))]
MAP_MAP_DTYPE = [("img", "O"), ("map_boundings", "f4", (1,))]


def get_by_id(root: Element, target_id: str) -> Optional[Element]:
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


def point2pixel(point: tuple[float, float], min: float, max: float) -> tuple[float, float]:
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
    return (px + config.image_map.img_shape / 2, py + config.image_map.img_shape / 2)


def preprocess_map(net_file_path: str, output_dir: Optional[str] = None, save_png: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    preprocess sumo network map and generate map representations at different levels in image based encoding

    :param net_file_path: path to sumo network xml file
    :param output_dir: directory to save preprocessed map data (optional)
    :param save_png: flag to save png images of map parts and lanes

    :return: tuple of (part_level_data, lane_level_data, map_level_data)
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    map_bounding = get_map_bounding(net_file_path)
    tree = ET.parse(net_file_path)
    root = tree.getroot()

    main_img = Image.new("1", (config.image_map.img_shape, config.image_map.img_shape), config.image_map.bg_color)
    main_draw = ImageDraw.Draw(main_img)

    map_parts_data_cannels = []
    map_parts_data_shapes = []
    map_parts_data_centers = []
    idx = 0
    for edge in root.findall("edge"):
        for lane in edge.findall("lane"):
            shape_str = lane.get("shape")

            if shape_str:
                points = []
                for p in shape_str.split():
                    x, y = map(float, p.split(","))
                    points.append((x, y))

                if len(set(points)) > 1:
                    img = Image.new("1", (config.image_map.img_shape, config.image_map.img_shape), config.image_map.bg_color)
                    draw = ImageDraw.Draw(img)

                    pixel_points = list(
                        map(
                            lambda point: point2pixel(
                                point, -map_bounding * config.vehicle.collect_data_radius, map_bounding * config.vehicle.collect_data_radius
                            ),
                            points,
                        )
                    )
                    draw.line(pixel_points, fill=config.image_map.road_color, width=config.image_map.road_width_px, joint="curve")
                    main_draw.line(pixel_points, fill=config.image_map.road_color, width=config.image_map.road_width_px, joint="curve")

                    if output_dir is not None and save_png:
                        img.save(os.path.join(output_dir, f"part_{idx}.png"))

                    np_img = np.array(img, dtype=np.float32)
                    shape_points = np.array(points, dtype=np.float32)
                    map_parts_data_cannels.append(np_img)
                    map_parts_data_shapes.append(shape_points)
                    map_parts_data_centers.append(shape_points.mean(axis=0))
                    idx += 1

    part_level_data = np.array(
        [(np.array(map_parts_data_cannels), map_parts_data_shapes, map_parts_data_centers, map_bounding)], dtype=MAP_PARTS_DTYPE
    )

    if output_dir is not None:
        np.save(
            os.path.join(output_dir, config.image_map.map_parts_name),
            part_level_data,
        )

        if save_png:
            main_img.save(os.path.join(output_dir, f"{config.image_map.map_img_name}.png"))

    map_level_data = np.array([(main_img, map_bounding)], dtype=MAP_MAP_DTYPE)
    if output_dir is not None:
        np.save(os.path.join(output_dir, config.image_map.map_img_name), map_level_data)

    map_lanes_data_cannels = []
    map_lanes_data_shapes = []
    idx = 0
    start_edges = get_entry_exit_edges(net_file_path)
    for start_edge, end_edges in start_edges.items():
        for end_edge in end_edges:
            path_str = get_shortest_path(net_file_path, start_edge, end_edge)
            path_edges = path_str.split(sep=" ")

            img = Image.new("1", (config.image_map.img_shape, config.image_map.img_shape), config.image_map.bg_color)
            draw = ImageDraw.Draw(img)
            all_points = []

            for i, edge in enumerate(path_edges):
                edge_elem = get_by_id(root, edge)

                for lane in edge_elem.findall("lane"):
                    shape_str = lane.get("shape")

                    if shape_str:
                        points = []
                        for p in shape_str.split():
                            x, y = map(float, p.split(","))
                            points.append((x, y))

                        if len(set(points)) > 1:
                            pixel_points = []
                            for point in points:
                                pixel_point = point2pixel(
                                    point, -map_bounding * config.vehicle.collect_data_radius, map_bounding * config.vehicle.collect_data_radius
                                )
                                pixel_points.append(pixel_point)

                                if len(all_points) == 0 or ((all_points[-1][0] != point[0]) or (all_points[-1][1] != point[1])):
                                    all_points.append(point)
                            draw.line(pixel_points, fill=config.image_map.road_color, width=config.image_map.road_width_px, joint="curve")

                if i < len(path_edges) - 1:
                    next_edge = path_edges[i + 1]
                    via_edge_id = get_connection_via(root, edge, next_edge)
                    via_edge = get_by_id(root, via_edge_id)

                    if via_edge is not None:
                        for lane in via_edge.findall("lane"):
                            shape_str = lane.get("shape")
                            if shape_str:
                                via_points = []
                                for p in shape_str.split():
                                    x, y = map(float, p.split(","))
                                    via_points.append((x, y))

                                if len(set(via_points)) > 1:
                                    via_pixel_points = []

                                    for point in via_points:
                                        pixel_point = point2pixel(
                                            point,
                                            -map_bounding * config.vehicle.collect_data_radius,
                                            map_bounding * config.vehicle.collect_data_radius,
                                        )
                                        via_pixel_points.append(pixel_point)
                                        if len(all_points) == 0 or ((all_points[-1][0] != point[0]) or (all_points[-1][1] != point[1])):
                                            all_points.append(point)
                                    draw.line(via_pixel_points, fill=config.image_map.road_color, width=config.image_map.road_width_px, joint="curve")

            if output_dir is not None and save_png:
                img.save(os.path.join(output_dir, f"lane_{idx}.png"))

            np_img = np.array(img, dtype=np.float32)
            shape_points = np.array(all_points, dtype=np.float32) if len(all_points) > 0 else np.array([], dtype=np.float32)
            map_lanes_data_cannels.append(np_img)
            map_lanes_data_shapes.append(shape_points)
            idx += 1

    lane_level_data = np.array([(np.array(map_lanes_data_cannels), map_lanes_data_shapes, map_bounding)], dtype=MAP_LANES_DTYPE)
    if output_dir is not None:
        np.save(
            os.path.join(output_dir, config.image_map.map_lane_name),
            lane_level_data,
        )
    return part_level_data, lane_level_data, map_level_data
