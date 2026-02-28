"""Unit tests for opencda.core.common.rsu_manager.RSUManager.

Covers:
- ID parsing/autogeneration and duplicates (including autogen-on-duplicate)
- Invalid ID types (with and without autogen)
- DataDumper creation based on data_dumping
- update_info/run_step/destroy call chains
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest


def _patch_rsu_manager_deps(mocker):
    localizer = Mock()
    perception = Mock()
    dumper = Mock()

    mocker.patch("opencda.core.common.rsu_manager.LocalizationManager", return_value=localizer)
    mocker.patch("opencda.core.common.rsu_manager.PerceptionManager", return_value=perception)
    mocker.patch("opencda.core.common.rsu_manager.DataDumper", return_value=dumper)

    return {"localizer": localizer, "perception": perception, "dumper": dumper}


def test_valid_id_from_config(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    cfg = {**minimal_rsu_config, "id": 3}

    rsu = RSUManager(Mock(), cfg, Mock(), mock_cav_world, data_dumping=False)
    assert rsu.rid == "rsu-3"
    mock_cav_world.update_rsu_manager.assert_called_once_with(rsu)


def test_duplicate_id_raises_when_autogen_disabled(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    cfg = {**minimal_rsu_config, "id": 3}

    RSUManager(Mock(), cfg, Mock(), mock_cav_world, data_dumping=False)
    with pytest.raises(ValueError, match="Duplicate RSU ID"):
        RSUManager(Mock(), cfg, Mock(), mock_cav_world, data_dumping=False, autogenerate_id_on_failure=False)


def test_duplicate_id_with_autogen_generates_new(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    cfg = {**minimal_rsu_config, "id": 3}

    rsu1 = RSUManager(Mock(), cfg, Mock(), mock_cav_world, data_dumping=False, autogenerate_id_on_failure=True)
    rsu2 = RSUManager(Mock(), cfg, Mock(), mock_cav_world, data_dumping=False, autogenerate_id_on_failure=True)

    assert rsu1.rid == "rsu-3"
    assert rsu2.rid == "rsu-1"
    assert rsu1.rid != rsu2.rid


def test_negative_id_with_autogen(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    cfg = {**minimal_rsu_config, "id": -1}

    rsu = RSUManager(Mock(), cfg, Mock(), mock_cav_world, data_dumping=False)
    assert rsu.rid == "rsu-1"


def test_invalid_id_type_with_autogen(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    cfg = {**minimal_rsu_config, "id": "not_a_number"}

    rsu = RSUManager(Mock(), cfg, Mock(), mock_cav_world, data_dumping=False, autogenerate_id_on_failure=True)
    assert rsu.rid == "rsu-1"


def test_invalid_id_type_without_autogen(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    cfg = {**minimal_rsu_config, "id": "not_a_number"}

    with pytest.raises(ValueError):
        RSUManager(Mock(), cfg, Mock(), mock_cav_world, data_dumping=False, autogenerate_id_on_failure=False)


def test_missing_id_with_autogen(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False)
    assert rsu.rid == "rsu-1"


def test_missing_id_without_autogen(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    with pytest.raises(ValueError, match="No RSU ID specified"):
        RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False, autogenerate_id_on_failure=False)


def test_data_dumper_created_when_enabled(mocker, minimal_rsu_config, mock_cav_world):
    deps = _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, current_time="t0", data_dumping=True)
    assert rsu.data_dumper is deps["dumper"]


def test_data_dumper_none_when_disabled(mocker, minimal_rsu_config, mock_cav_world):
    _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False)
    assert rsu.data_dumper is None


def test_update_info_calls_localize_and_detect(mocker, minimal_rsu_config, mock_cav_world):
    deps = _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False)

    ego_pos = Mock()
    deps["localizer"].get_ego_pos.return_value = ego_pos

    rsu.update_info()

    deps["localizer"].localize.assert_called_once_with()
    deps["perception"].detect.assert_called_once_with(ego_pos)


def test_run_step_with_dumper(mocker, minimal_rsu_config, mock_cav_world):
    deps = _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=True)

    rsu.run_step()
    deps["dumper"].run_step.assert_called_once_with(deps["perception"], deps["localizer"], None)


def test_run_step_without_dumper(mocker, minimal_rsu_config, mock_cav_world):
    deps = _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False)

    rsu.run_step()
    deps["dumper"].run_step.assert_not_called()


def test_destroy_calls_both_destroy(mocker, minimal_rsu_config, mock_cav_world):
    deps = _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False)

    rsu.destroy()
    deps["perception"].destroy.assert_called_once_with()
    deps["localizer"].destroy.assert_called_once_with()


def test_update_info_v2x_does_not_raise_and_has_no_side_effects(mocker, minimal_rsu_config, mock_cav_world):
    """update_info_v2x() is currently a no-op: should not raise and must not trigger side effects."""
    deps = _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False)

    # Guard against accidental future logic being added to update_info_v2x().
    deps["localizer"].reset_mock()
    deps["perception"].reset_mock()

    assert rsu.update_info_v2x() is None
    assert deps["localizer"].mock_calls == []
    assert deps["perception"].mock_calls == []


def test_update_info_localizer_failure_propagates_and_stops_chain(mocker, minimal_rsu_config, mock_cav_world):
    """If localizer.localize() fails, update_info() should propagate and not call perception."""
    deps = _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False)
    deps["localizer"].localize.side_effect = RuntimeError("localize failed")

    with pytest.raises(RuntimeError, match="localize failed"):
        rsu.update_info()

    deps["perception"].detect.assert_not_called()


def test_update_info_perception_failure_propagates(mocker, minimal_rsu_config, mock_cav_world):
    """If perception.detect() fails, update_info() should propagate the error."""
    deps = _patch_rsu_manager_deps(mocker)
    from opencda.core.common.rsu_manager import RSUManager

    rsu = RSUManager(Mock(), minimal_rsu_config, Mock(), mock_cav_world, data_dumping=False)

    ego_pos = Mock()
    deps["localizer"].get_ego_pos.return_value = ego_pos
    deps["perception"].detect.side_effect = RuntimeError("detect failed")

    with pytest.raises(RuntimeError, match="detect failed"):
        rsu.update_info()

    deps["localizer"].localize.assert_called_once_with()
    deps["perception"].detect.assert_called_once_with(ego_pos)
