"""Live-release metadata coverage and the real SUMO bridge's class conversion."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opencda.scenario_testing.utils.sim_api import multi_class_vehicle_blueprint_filter

ROOT = Path(__file__).resolve().parents[1]
# IDs/base_type attributes were reported by the running CARLA 1.0 release.
# Preserve the established SUV/electric categories where they are more specific.
RELEASE_CLASSES = {
    "vehicle.ambulance.ford": ("truck", "emergency"),
    "vehicle.bh.crossbike": ("bicycle", "bicycle"),
    "vehicle.carlacola.actors": ("truck", "truck"),
    "vehicle.diamondback.century": ("bicycle", "bicycle"),
    "vehicle.dodge.charger": ("car", "passenger"),
    "vehicle.dodgecop.charger": ("car", "authority"),
    "vehicle.firetruck.actors": ("truck", "emergency"),
    "vehicle.fuso.mitsubishi": ("bus", "bus"),
    "vehicle.gazelle.omafiets": ("bicycle", "bicycle"),
    "vehicle.harley-davidson.low_rider": ("motorcycle", "motorcycle"),
    "vehicle.kawasaki.ninja": ("motorcycle", "motorcycle"),
    "vehicle.kia.niro": ("car", "passenger"),
    "vehicle.lincoln.mkz": ("car", "passenger"),
    "vehicle.mini.cooper": ("car", "passenger"),
    "vehicle.nissan.patrol": ("suv", "passenger"),
    "vehicle.sprinter.mercedes": ("van", "passenger"),
    "vehicle.taxi.ford": ("car", "taxi"),
    "vehicle.tesla.model3": ("car", "evehicle"),
    "vehicle.ue4.audi.tt": ("car", "passenger"),
    "vehicle.ue4.chevrolet.impala": ("car", "passenger"),
    "vehicle.ue4.ford.crown": ("car", "passenger"),
    "vehicle.ue4.ford.mustang": ("car", "passenger"),
    "vehicle.ue4.mercedes.ccc": ("car", "passenger"),
    "vehicle.vespa.zx125": ("motorcycle", "motorcycle"),
    "vehicle.yamaha.yzf": ("motorcycle", "motorcycle"),
}


class ReleaseBlueprint:
    def __init__(self, model_id):
        self.id = model_id
        self.attributes = {}

    def has_attribute(self, _name):
        return False

    def set_attribute(self, name, value):
        self.attributes[name] = value


class ReleaseLibrary(list):
    def find(self, model_id):
        for blueprint in self:
            if blueprint.id == model_id:
                return blueprint
        raise IndexError(f"blueprint {model_id!r} not found")

    def filter(self, model_id):
        return [blueprint for blueprint in self if blueprint.id == model_id]


@pytest.fixture
def bridge_helper(monkeypatch):
    # The suite's conftest supplies a placeholder BridgeHelper. Load the real
    # bridge and its signal enum without changing other tests' module bindings.
    modules = {}
    for name in ("sumo_simulation", "bridge_helper"):
        path = ROOT / "opencda/co_simulation/sumo_integration" / f"{name}.py"
        alias = f"_metadata_real_{name}"
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, alias, module)
        spec.loader.exec_module(module)
        modules[name] = module
        if name == "sumo_simulation":
            monkeypatch.setitem(sys.modules, "opencda.co_simulation.sumo_integration.sumo_simulation", module)
    return modules["bridge_helper"]


@pytest.fixture
def release_library():
    return ReleaseLibrary(ReleaseBlueprint(model_id) for model_id in RELEASE_CLASSES)


@pytest.fixture
def blueprint_metadata():
    return json.loads((ROOT / "opencda/assets/blueprint_meta/bbx_stats_0915.json").read_text())


def test_live_release_vehicle_is_classified_in_both_metadata_tables(blueprint_metadata, bridge_helper):
    for model_id, classes in RELEASE_CLASSES.items():
        assert blueprint_metadata[model_id]["class"] == classes[0]
        assert bridge_helper.BridgeHelper._VTYPES[model_id]["vClass"] == classes[1]


def test_all_configured_traffic_classes_remain_available_with_live_release(release_library, blueprint_metadata):
    weights = {"car": 0.5, "truck": 0.1, "bus": 0.1, "bicycle": 0.1, "motorcycle": 0.1}
    pools = {label: multi_class_vehicle_blueprint_filter(label, release_library, blueprint_metadata) for label in weights}
    assert all(pools.values())
    assert {blueprint.id for blueprint in pools["bus"]} == {"vehicle.fuso.mitsubishi"}
    assert {blueprint.id for blueprint in pools["truck"]} == {"vehicle.ambulance.ford", "vehicle.carlacola.actors", "vehicle.firetruck.actors"}
    assert "vehicle.sprinter.mercedes" not in {blueprint.id for pool in pools.values() for blueprint in pool}


@pytest.mark.parametrize("vclass,expected_id", [("bus", "vehicle.fuso.mitsubishi"), ("truck", "vehicle.carlacola.actors")])
def test_sumo_legacy_vtype_maps_to_available_release_vehicle_of_same_class(vclass, expected_id, release_library, monkeypatch, bridge_helper):
    monkeypatch.setattr(bridge_helper.BridgeHelper, "blueprint_library", release_library)
    sumo_actor = SimpleNamespace(type_id=f"legacy.unavailable.{vclass}", vclass=SimpleNamespace(value=vclass))
    selected = bridge_helper.BridgeHelper.get_carla_blueprint(sumo_actor)
    assert selected.id == expected_id
    assert selected.attributes["role_name"] == "sumo_driver"


@pytest.mark.parametrize("model_id,vclass", [("vehicle.fuso.mitsubishi", "bus"), ("vehicle.carlacola.actors", "truck")])
def test_carla_release_vehicle_creates_sumo_type_with_class_and_physical_dimensions(model_id, vclass, monkeypatch, bridge_helper):
    vehicle_type = Mock()
    vehicle_type.getIDList.return_value = []
    monkeypatch.setattr(bridge_helper, "traci", SimpleNamespace(vehicletype=vehicle_type))
    actor = SimpleNamespace(
        type_id=model_id,
        attributes={"number_of_wheels": "4"},
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=3.0, y=1.0, z=1.5)),
    )
    assert bridge_helper.BridgeHelper.get_sumo_vtype(actor) == model_id
    vehicle_type.copy.assert_called_once_with("DEFAULT_VEHTYPE", model_id)
    vehicle_type.setVehicleClass.assert_called_once_with(model_id, vclass)
    vehicle_type.setLength.assert_called_once_with(model_id, 6.0)
    vehicle_type.setWidth.assert_called_once_with(model_id, 2.0)
    vehicle_type.setHeight.assert_called_once_with(model_id, 3.0)


def test_existing_legacy_and_special_categories_are_preserved(blueprint_metadata, bridge_helper):
    assert blueprint_metadata["vehicle.lincoln.mkz_2017"]["class"] == "car"
    assert blueprint_metadata["vehicle.lincoln.mkz_2020"]["class"] == "car"
    assert blueprint_metadata["vehicle.nissan.patrol"]["class"] == "suv"
    assert bridge_helper.BridgeHelper._VTYPES["vehicle.lincoln.mkz2017"]["vClass"] == "passenger"
    assert bridge_helper.BridgeHelper._VTYPES["vehicle.tesla.model3"]["vClass"] == "evehicle"
