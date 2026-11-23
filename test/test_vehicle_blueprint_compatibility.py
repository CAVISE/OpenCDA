"""Missing blueprint handling with a strict simulation of the CARLA library."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opencda.scenario_testing.utils.sim_api import (
    ScenarioManager,
    _available_traffic_classes,
    car_blueprint_filter,
    multi_class_vehicle_blueprint_filter,
)


class StrictBlueprintLibrary:
    def __init__(self, model_ids):
        self.blueprints = {model_id: SimpleNamespace(id=model_id) for model_id in model_ids}
        self.requested_ids = []

    def find(self, model_id):
        self.requested_ids.append(model_id)
        try:
            return self.blueprints[model_id]
        except KeyError:
            raise IndexError(f"blueprint {model_id!r} not found") from None


def test_preferred_traffic_filter_skips_missing_ids_without_changing_existing_weights():
    library = StrictBlueprintLibrary(["vehicle.audi.tt", "vehicle.nissan.micra"])
    selected = car_blueprint_filter(library)
    # The previous preferred list includes the Micra twice; preserve that weighting.
    assert [blueprint.id for blueprint in selected] == ["vehicle.audi.tt", "vehicle.nissan.micra", "vehicle.nissan.micra"]
    assert "vehicle.audi.a2" in library.requested_ids


def test_preferred_traffic_filter_rejects_a_completely_unavailable_list():
    library = StrictBlueprintLibrary(["vehicle.unrelated"])
    with pytest.raises(RuntimeError, match="preferred traffic vehicle blueprints"):
        car_blueprint_filter(library)


def test_class_filter_ignores_removed_models_and_preserves_requested_class():
    metadata = {
        "vehicle.removed": {"class": "car"},
        "vehicle.car": {"class": "car"},
        "vehicle.truck": {"class": "truck"},
    }
    library = StrictBlueprintLibrary(["vehicle.car", "vehicle.truck"])
    assert [bp.id for bp in multi_class_vehicle_blueprint_filter("car", library, metadata)] == ["vehicle.car"]
    assert library.requested_ids == ["vehicle.removed", "vehicle.car"]


def test_missing_classes_are_excluded_and_remaining_probabilities_are_renormalized():
    metadata = {
        "vehicle.car": {"class": "car"},
        "vehicle.truck": {"class": "truck"},
        "vehicle.bus": {"class": "bus"},
    }
    library = StrictBlueprintLibrary(["vehicle.car", "vehicle.truck"])
    pools, probabilities = _available_traffic_classes(library, metadata, {"car": 0.2, "truck": 0.6, "bus": 0.2})
    assert list(pools) == ["car", "truck"]
    assert probabilities == pytest.approx([0.25, 0.75])
    assert [pool[0].id for pool in pools.values()] == ["vehicle.car", "vehicle.truck"]


def test_zero_probability_classes_are_never_selected_or_resolved():
    metadata = {"vehicle.car": {"class": "car"}, "vehicle.bus": {"class": "bus"}}
    library = StrictBlueprintLibrary(["vehicle.car", "vehicle.bus"])
    pools, probabilities = _available_traffic_classes(library, metadata, {"car": 1.0, "bus": 0.0})
    assert list(pools) == ["car"]
    assert probabilities == [1.0]
    assert library.requested_ids == ["vehicle.car"]


@pytest.mark.parametrize("probability", [-0.1, float("inf"), float("nan")])
def test_invalid_class_probabilities_are_rejected(probability):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _available_traffic_classes(StrictBlueprintLibrary([]), {}, {"car": probability})


@pytest.mark.parametrize("weights", [{"car": 1.0}, {"car": 0.0}, {}])
def test_no_usable_class_has_an_actionable_error(weights):
    with pytest.raises(RuntimeError, match="Update blueprint metadata"):
        _available_traffic_classes(StrictBlueprintLibrary([]), {"vehicle.removed": {"class": "car"}}, weights)


def test_explicit_unavailable_actor_model_is_not_replaced():
    manager = ScenarioManager.__new__(ScenarioManager)
    library = StrictBlueprintLibrary(["vehicle.available"])
    with pytest.raises(IndexError, match="vehicle.requested"):
        manager.prepare_actor_blueprint({"model": "vehicle.requested"}, "vehicle.available", library)
    assert library.requested_ids == ["vehicle.requested"]


@pytest.mark.parametrize("method,spawn_key", [("spawn_vehicles_by_list", "vehicle_list"), ("spawn_vehicle_by_range", "range")])
def test_random_traffic_does_not_resolve_an_unused_fixed_model(method, spawn_key, mocker):
    manager = ScenarioManager.__new__(ScenarioManager)
    manager.use_multi_class_bp = True
    manager.bp_meta = {"vehicle.audi.tt": {"class": "car"}}
    manager.bp_class_sample_prob = {"car": 1.0}
    library = StrictBlueprintLibrary(["vehicle.audi.tt"])
    manager.world = SimpleNamespace(get_blueprint_library=lambda: library)
    manager.cav_world = Mock()
    # Empty requests avoid any actor spawning. The invalid fixed mkz_2020 must not
    # be consulted merely to initialize a random-traffic generator.
    config = {"random": True, spawn_key: []}
    assert getattr(manager, method)(Mock(), config, []) == []
    assert "vehicle.lincoln.mkz_2020" not in library.requested_ids


# Catalogue reported by a running CARLA 1.0 extended server.
CARLA_1_0_VEHICLE_IDS = (
    "vehicle.ambulance.ford",
    "vehicle.bh.crossbike",
    "vehicle.carlacola.actors",
    "vehicle.diamondback.century",
    "vehicle.dodge.charger",
    "vehicle.dodgecop.charger",
    "vehicle.firetruck.actors",
    "vehicle.fuso.mitsubishi",
    "vehicle.gazelle.omafiets",
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.kia.niro",
    "vehicle.lincoln.mkz",
    "vehicle.mini.cooper",
    "vehicle.nissan.patrol",
    "vehicle.sprinter.mercedes",
    "vehicle.taxi.ford",
    "vehicle.tesla.model3",
    "vehicle.ue4.audi.tt",
    "vehicle.ue4.chevrolet.impala",
    "vehicle.ue4.ford.crown",
    "vehicle.ue4.ford.mustang",
    "vehicle.ue4.mercedes.ccc",
    "vehicle.vespa.zx125",
    "vehicle.yamaha.yzf",
)


@pytest.mark.parametrize("legacy_default", ["vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020"])
def test_implicit_legacy_mkz_uses_confirmed_model_from_real_release_catalogue(legacy_default, caplog):
    manager = ScenarioManager.__new__(ScenarioManager)
    library = StrictBlueprintLibrary(CARLA_1_0_VEHICLE_IDS)
    selected = manager.prepare_actor_blueprint({}, legacy_default, library)
    assert selected.id == "vehicle.lincoln.mkz"
    assert library.requested_ids == [legacy_default, "vehicle.lincoln.mkz"]
    assert "using installed blueprint vehicle.lincoln.mkz" in caplog.text


@pytest.mark.parametrize("legacy_default", ["vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020"])
def test_implicit_legacy_default_has_priority_when_it_is_still_available(legacy_default):
    manager = ScenarioManager.__new__(ScenarioManager)
    library = StrictBlueprintLibrary([legacy_default, *CARLA_1_0_VEHICLE_IDS])
    selected = manager.prepare_actor_blueprint({}, legacy_default, library)
    assert selected.id == legacy_default
    assert library.requested_ids == [legacy_default]


@pytest.mark.parametrize("explicit_model", ["vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020", "vehicle.unknown"])
def test_explicit_missing_model_is_not_aliased_despite_an_available_release_mkz(explicit_model):
    manager = ScenarioManager.__new__(ScenarioManager)
    library = StrictBlueprintLibrary(CARLA_1_0_VEHICLE_IDS)
    with pytest.raises(IndexError, match="not found"):
        manager.prepare_actor_blueprint({"model": explicit_model}, "vehicle.lincoln.mkz_2017", library)
    assert library.requested_ids == [explicit_model]


@pytest.mark.parametrize("legacy_default", ["vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020"])
def test_missing_legacy_and_release_defaults_require_an_explicit_model(legacy_default):
    manager = ScenarioManager.__new__(ScenarioManager)
    library = StrictBlueprintLibrary(["vehicle.tesla.model3"])
    with pytest.raises(IndexError, match="Set an available 'model'"):
        manager.prepare_actor_blueprint({}, legacy_default, library)
    assert library.requested_ids == [legacy_default, "vehicle.lincoln.mkz"]


def test_unrelated_missing_default_is_not_replaced_with_a_release_mkz():
    manager = ScenarioManager.__new__(ScenarioManager)
    library = StrictBlueprintLibrary(CARLA_1_0_VEHICLE_IDS)
    with pytest.raises(IndexError, match="vehicle.unrelated"):
        manager.prepare_actor_blueprint({}, "vehicle.unrelated", library)
    assert library.requested_ids == ["vehicle.unrelated"]


@pytest.mark.parametrize("method,spawn_key", [("spawn_vehicles_by_list", "vehicle_list"), ("spawn_vehicle_by_range", "range")])
def test_nonrandom_traffic_initializes_on_real_release_catalogue(method, spawn_key):
    manager = ScenarioManager.__new__(ScenarioManager)
    manager.use_multi_class_bp = False
    library = StrictBlueprintLibrary(CARLA_1_0_VEHICLE_IDS)
    manager.world = SimpleNamespace(get_blueprint_library=lambda: library)
    manager.cav_world = Mock()
    assert getattr(manager, method)(Mock(), {"random": False, spawn_key: []}, []) == []
    assert library.requested_ids == ["vehicle.lincoln.mkz_2020", "vehicle.lincoln.mkz"]
