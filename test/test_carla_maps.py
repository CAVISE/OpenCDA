"""Map resolution across the CARLA UE4 and UE5 packages."""

import pytest

from opencda.scenario_testing.utils.carla_maps import resolve_carla_map


def test_classic_town_uses_advertised_pack_path():
    assert resolve_carla_map("Town06", ["/AdditionalMaps/Maps/Town06_Opt"]) == "/AdditionalMaps/Maps/Town06_Opt"


def test_original_town_has_priority_over_layered_variant():
    maps = ["/Game/Carla/Maps/Town06", "/AdditionalMaps/Maps/Town06_Opt"]
    assert resolve_carla_map("Town06", maps) == maps[0]


def test_full_path_is_preserved():
    maps = ["/Game/Maps/Town06_Opt", "/AdditionalMaps/Maps/Town06_Opt"]
    assert resolve_carla_map(maps[1], maps) == maps[1]


@pytest.mark.parametrize("town,maps", [("Town06", []), ("Town12", ["/Game/Maps/Town12_Opt"]), ("Town06", ["/A/Town06_Opt", "/B/Town06_Opt"])])
def test_missing_or_ambiguous_maps_fail(town, maps):
    with pytest.raises(RuntimeError):
        resolve_carla_map(town, maps)
