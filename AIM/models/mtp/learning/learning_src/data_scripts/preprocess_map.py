from PIL import Image, ImageDraw
import numpy as np
import os
import xml.etree.ElementTree as ET

from .data_config import (
    IMG_SHAPE,
    BG_COLOR,
    ROAD_COLOR,
    ROAD_WIDTH_PX,
    COLLECT_DATA_RADIUS,
    MAP_IMG_NAME,
    MAP_LANE_NAME,
    MAP_PARTS_NAME,
)
from .generate_csv_utils import get_map_bounding, get_entry_exit_edges, get_shortest_path, get_connection_via


MAP_PARTS_DTYPE = [("img", "O"), ("shape_points", "O"), ("center", "O"), ("map_boundings", "f4", (1,))]
MAP_LANES_DTYPE = [("img", "O"), ("shape_points", "O"), ("map_boundings", "f4", (1,))]
MAP_MAP_DTYPE = [("img", "O"), ("map_boundings", "f4", (1,))]


def get_by_id(root, target_id: str):
    for elem in root.iter():
        if elem.get("id") == target_id:
            return elem
    return None


def point2pixel(point, min, max):
    scale = IMG_SHAPE / (max - min)
    px = point[0] * scale
    py = point[1] * scale
    return (px + IMG_SHAPE / 2, py + IMG_SHAPE / 2)


def preprocess_map(net_file_path: str, output_dir: str = None, save_png=False):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    map_bounding = get_map_bounding(net_file_path)
    tree = ET.parse(net_file_path)
    root = tree.getroot()

    main_img = Image.new("1", (IMG_SHAPE, IMG_SHAPE), BG_COLOR)
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
                    img = Image.new("1", (IMG_SHAPE, IMG_SHAPE), BG_COLOR)
                    draw = ImageDraw.Draw(img)

                    pixel_points = list(
                        map(lambda point: point2pixel(point, -map_bounding * COLLECT_DATA_RADIUS, map_bounding * COLLECT_DATA_RADIUS), points)
                    )
                    draw.line(pixel_points, fill=ROAD_COLOR, width=ROAD_WIDTH_PX, joint="curve")
                    main_draw.line(pixel_points, fill=ROAD_COLOR, width=ROAD_WIDTH_PX, joint="curve")

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
            os.path.join(output_dir, MAP_PARTS_NAME),
            part_level_data,
        )

        if save_png:
            main_img.save(os.path.join(output_dir, f"{MAP_IMG_NAME}.png"))

    map_level_data = np.array([(main_img, map_bounding)], dtype=MAP_MAP_DTYPE)
    if output_dir is not None:
        np.save(os.path.join(output_dir, MAP_IMG_NAME), map_level_data)

    map_lanes_data_cannels = []
    map_lanes_data_shapes = []
    idx = 0
    start_edges = get_entry_exit_edges(net_file_path)
    for start_edge, end_edges in start_edges.items():
        for end_edge in end_edges:
            path_str = get_shortest_path(net_file_path, start_edge, end_edge)
            path_edges = path_str.split(sep=" ")

            img = Image.new("1", (IMG_SHAPE, IMG_SHAPE), BG_COLOR)
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
                                pixel_point = point2pixel(point, -map_bounding * COLLECT_DATA_RADIUS, map_bounding * COLLECT_DATA_RADIUS)
                                pixel_points.append(pixel_point)

                                if len(all_points) == 0 or ((all_points[-1][0] != point[0]) or (all_points[-1][1] != point[1])):
                                    all_points.append(point)
                            draw.line(pixel_points, fill=ROAD_COLOR, width=ROAD_WIDTH_PX, joint="curve")

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
                                        pixel_point = point2pixel(point, -map_bounding * COLLECT_DATA_RADIUS, map_bounding * COLLECT_DATA_RADIUS)
                                        via_pixel_points.append(pixel_point)
                                        if len(all_points) == 0 or ((all_points[-1][0] != point[0]) or (all_points[-1][1] != point[1])):
                                            all_points.append(point)
                                    draw.line(via_pixel_points, fill=ROAD_COLOR, width=ROAD_WIDTH_PX, joint="curve")

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
            os.path.join(output_dir, MAP_LANE_NAME),
            lane_level_data,
        )
    return part_level_data, lane_level_data, map_level_data
