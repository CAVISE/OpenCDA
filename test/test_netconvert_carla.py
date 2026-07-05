from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import netconvert_carla as converter


def test_netconvert_validates_input_before_loading_dependencies(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OpenDRIVE file does not exist"):
        converter.netconvert_carla(str(tmp_path / "missing.xodr"), str(tmp_path / "map.net.xml"))


def test_netconvert_rejects_non_xodr_input(tmp_path: Path) -> None:
    input_file = tmp_path / "map.xml"
    input_file.write_text("<OpenDRIVE/>", encoding="utf-8")

    with pytest.raises(ValueError, match="must have the .xodr extension"):
        converter.netconvert_carla(str(input_file), str(tmp_path / "map.net.xml"))


class _FakeMap:
    def __init__(self, name: str, xodr: str = "<OpenDRIVE/>") -> None:
        self.name = name
        self.xodr = xodr

    def to_opendrive(self) -> str:
        return self.xodr


class _FakeWorld:
    def __init__(self, map_name: str) -> None:
        self.map = _FakeMap(map_name)

    def get_map(self) -> _FakeMap:
        return self.map


class _FakeClient:
    instance: "_FakeClient"

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.timeout: float | None = None
        self.world = _FakeWorld("/Game/Carla/Maps/Town13.Town13")
        self.loaded_maps: list[str] = []
        _FakeClient.instance = self

    def set_timeout(self, timeout: float) -> None:
        self.timeout = timeout

    def get_world(self) -> _FakeWorld:
        return self.world

    def load_world(self, map_name: str) -> _FakeWorld:
        self.loaded_maps.append(map_name)
        self.world = _FakeWorld(map_name)
        return self.world


@pytest.mark.parametrize(
    ("requested_map", "expected_loads"),
    [("Town13", []), ("Town12", ["Town12"])],
)
def test_from_carla_loads_map_only_when_different(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_map: str,
    expected_loads: list[str],
) -> None:
    calls: list[tuple[str, str, str, bool]] = []

    def fake_impl(xodr_file: str, output: str, tmpdir: str, guess_tls: bool) -> None:
        calls.append((Path(xodr_file).read_text(encoding="utf-8"), output, tmpdir, guess_tls))

    monkeypatch.setattr(converter, "_load_carla", lambda: SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(converter, "_netconvert_carla_impl", fake_impl)
    output = tmp_path / "map.net.xml"

    result = converter.netconvert_carla_from_carla(
        str(output),
        carla_map_name=requested_map,
    )

    assert result == str(output)
    assert _FakeClient.instance.loaded_maps == expected_loads
    assert calls[0][0] == "<OpenDRIVE/>"
    assert calls[0][3] is converter.DEFAULT_GUESS_TLS


def test_argument_parser_defaults_use_module_constants() -> None:
    parser = converter._build_argument_parser()
    args = parser.parse_args(["map.xodr"])

    assert args.output == converter.DEFAULT_OUTPUT_FILE
    assert args.guess_tls is converter.DEFAULT_GUESS_TLS
    assert args.carla_host == converter.DEFAULT_CARLA_HOST
    assert args.carla_port == converter.DEFAULT_CARLA_PORT
    assert args.carla_timeout == converter.DEFAULT_CARLA_TIMEOUT
    assert args.carla_map == converter.DEFAULT_CARLA_MAP_NAME
