from __future__ import annotations

import xml.etree.ElementTree as XmlTree
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import polyconvert_carla as converter


def test_generate_poly_validates_net_before_importing_carla(tmp_path: Path) -> None:
    output = tmp_path / "map.poly.xml"

    with pytest.raises(FileNotFoundError, match="SUMO net file does not exist"):
        converter.generate_poly(str(tmp_path / "missing.net.xml"), str(output))

    assert not output.exists()


def test_sumo_net_location_reads_offset_and_boundary(tmp_path: Path) -> None:
    net_file = tmp_path / "map.net.xml"
    net_file.write_text(
        '<net><location netOffset="10.5,-20" convBoundary="0,1,100,200"/></net>',
        encoding="utf-8",
    )

    offset, boundary = converter._sumo_net_location(str(net_file))

    assert offset == (10.5, -20.0)
    assert boundary == (0.0, 1.0, 100.0, 200.0)


def test_scan_positions_cover_bounds_in_serpentine_order() -> None:
    positions = converter._scan_positions((0.0, 0.0, 4.0, 2.0), step=2.0)

    assert positions == [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (4.0, 2.0),
        (2.0, 2.0),
        (0.0, 2.0),
    ]


class _FakeClient:
    def __init__(self, current_name: str) -> None:
        self.current_world = SimpleNamespace(get_map=lambda: SimpleNamespace(name=current_name))
        self.loaded_maps: list[str] = []

    def get_world(self) -> object:
        return self.current_world

    def load_world(self, map_name: str) -> object:
        self.loaded_maps.append(map_name)
        return SimpleNamespace(get_map=lambda: SimpleNamespace(name=map_name))


def test_get_world_keeps_equivalent_current_map() -> None:
    client = _FakeClient("/Game/Carla/Maps/Town13.Town13")

    world = converter._get_world(client, "Town13")

    assert world is client.current_world
    assert client.loaded_maps == []


def test_get_world_loads_different_map() -> None:
    client = _FakeClient("Town12")

    converter._get_world(client, "Town13")

    assert client.loaded_maps == ["Town13"]


def test_write_poly_transforms_coordinates_and_uses_compact_xml(tmp_path: Path) -> None:
    output = tmp_path / "map.poly.xml"
    objects = [
        {
            "id": "house 1",
            "label": "Buildings",
            "vertices": [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 3.0, 2.0],
                [0.0, 3.0, 2.0],
            ],
        }
    ]

    converter._write_poly(
        objects,
        str(output),
        offset=(10.0, 20.0),
        boundary=(0.0, 0.0, 100.0, 100.0),
        labels="Buildings",
        boundary_margin=0.0,
        max_area=100.0,
        min_height=1.0,
        max_wall_thickness=8.0,
    )

    content = output.read_text(encoding="utf-8")
    poly = XmlTree.parse(output).getroot().find("poly")
    assert poly is not None
    assert poly.attrib["type"] == "building"
    assert poly.attrib["shape"] == "10.00,17.00 12.00,17.00 12.00,20.00 10.00,20.00"
    assert "\n  <poly" not in content


def test_argument_parser_defaults_use_module_constants() -> None:
    parser = converter._build_argument_parser()
    args = parser.parse_args(["--net-file", "map.net.xml", "--output", "map.poly.xml"])

    assert args.carla_host == converter.DEFAULT_CARLA_HOST
    assert args.carla_port == converter.DEFAULT_CARLA_PORT
    assert args.carla_timeout == converter.DEFAULT_CARLA_TIMEOUT
    assert args.carla_map == converter.DEFAULT_CARLA_MAP_NAME
    assert args.labels == converter.DEFAULT_LABELS
    assert args.bounds == converter.DEFAULT_BOUNDS
    assert args.stream_distance == converter.DEFAULT_STREAM_DISTANCE
    assert args.scan_step == converter.DEFAULT_SCAN_STEP
    assert args.settle_ticks == converter.DEFAULT_SETTLE_TICKS
    assert args.scan_altitude == converter.DEFAULT_SCAN_ALTITUDE
    assert args.min_height == converter.DEFAULT_MIN_HEIGHT
    assert args.max_area == converter.DEFAULT_MAX_AREA
    assert args.max_wall_thickness == converter.DEFAULT_MAX_WALL_THICKNESS
    assert args.verbose is converter.DEFAULT_VERBOSE
