"""Replay CLI and output preservation regressions."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from opencda.scenario_testing.utils import run_metadata


@pytest.fixture
def entrypoint():
    """Load the scenario CLI without importing the simulation runtime.

    Returns
    -------
    types.ModuleType
        Runner exposing the real parser and replay argument handling.
    """
    spec = importlib.util.spec_from_file_location("opencda_cli_regressions", Path(__file__).resolve().parents[1] / "opencda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flag", ["--model-dir", "--test-scenario", "--log-file", "--models-root", "--advcp-config", "--carla-host"])
@pytest.mark.parametrize("value", ["-weights", "--weights", "-", "-weights with spaces"])
def test_leading_dash_values_round_trip_through_replay(entrypoint, tmp_path, monkeypatch, flag, value):
    """Preserve valid leading-dash values in recorded and replayed commands.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real runner module loaded by the fixture.
    tmp_path : pathlib.Path
        Directory containing the saved manifest.
    monkeypatch : pytest.MonkeyPatch
        Fixture selecting the working directory for relative model paths.
    flag, value : str
        Value-taking option and argument that must retain their meaning.
    """
    arguments = ["-t", "demo", f"{flag}={value}"]
    if flag == "--model-dir":
        (tmp_path / value).mkdir()
        monkeypatch.chdir(tmp_path)
        arguments.append("--with-coperception")
    original, command, _ = entrypoint.resolve_launch_arguments(arguments)
    run_metadata.write_json(tmp_path / "run_manifest.json", {"launch": {"argv": command}})
    override = [f"{flag}={value}"] if flag == "--log-file" else []
    restored, replay_command, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), *override])
    destination = flag[2:].replace("-", "_")
    assert getattr(original, destination) == getattr(restored, destination) == value
    assert getattr(entrypoint.arg_parse(replay_command), destination) == value
    if flag == "--model-dir":
        entrypoint.model_resolver.resolve_runtime_models(original)
        entrypoint.model_resolver.resolve_runtime_models(restored)
        assert original.model_dir == restored.model_dir == str(tmp_path / value)


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("scenario", ["demo", "demo.v1"])
@pytest.mark.parametrize("as_path", [False, True])
def test_same_second_launch_preserves_original_run_artifacts(entrypoint, tmp_path, monkeypatch, replay, scenario, as_path):
    """Reserve a distinct output path before opening a same-second run log.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real runner module loaded by the fixture.
    tmp_path : pathlib.Path
        Isolated working directory containing the scenario and run artifacts.
    monkeypatch : pytest.MonkeyPatch
        Fixture freezing the clock and replacing the simulation runtime.
    replay : bool
        Whether the second invocation replays or independently reruns the scenario.
    scenario : str
        Scenario name, including a valid name containing dots.
    as_path : bool
        Whether the original launch selects a YAML path or a scenario name.
    """
    monkeypatch.chdir(tmp_path)
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime(2026, 11, 11, 0, 0, 0)
    monkeypatch.setattr(entrypoint, "datetime", clock)
    scenario_module = ModuleType("opencda.scenario_testing.scenario")
    scenario_module.run_scenario = Mock()
    monkeypatch.setitem(sys.modules, scenario_module.__name__, scenario_module)
    monkeypatch.setattr(entrypoint, "install_traceback_handler", Mock())
    monkeypatch.setattr(entrypoint, "show_performance_stats", Mock())
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {})
    configs = tmp_path / "opencda/scenario_testing/config_yaml"
    configs.mkdir(parents=True)
    (configs / "default.yaml").write_text("world:\n  seed: 11\n")
    scenario_config = configs / f"{scenario}.yaml"
    scenario_config.write_text("scenario: {}\n")
    argument = str(scenario_config) if as_path else scenario
    original_dir = tmp_path / "simulation_output/evaluation_outputs" / f"{scenario}_2026_11_11_00_00_00"
    reserved_dir = original_dir.with_name(f"{original_dir.name}_1")
    logger = entrypoint.logging.getLogger("cavise.opencda")
    previous_handlers = set(logger.handlers)
    snapshots = {}
    for arguments in (["-t", argument], ["--replay-from", str(original_dir)] if replay else ["-t", argument]):
        monkeypatch.setattr(sys, "argv", ["opencda.py", *arguments])
        try:
            entrypoint.main()
        finally:
            for handler in tuple(logger.handlers):
                if handler not in previous_handlers:
                    logger.removeHandler(handler)
                    handler.close()
        if not snapshots:
            snapshots = {path: path.read_bytes() for path in original_dir.rglob("*") if path.is_file()}
            reserved_dir.mkdir()
            (reserved_dir / "existing.txt").write_text("keep this run")
    assert snapshots
    assert all(path.read_bytes() == contents for path, contents in snapshots.items())
    assert (reserved_dir / "existing.txt").read_text() == "keep this run"
    new_dir = original_dir.with_name(f"{original_dir.name}_2")
    manifest = run_metadata.load_run_manifest(new_dir)
    assert manifest["status"] == "completed"
    assert manifest["scenario"] == scenario
    assert scenario_module.run_scenario.call_args.kwargs["current_time"] == "2026_11_11_00_00_00_2"
    assert (new_dir / entrypoint.DEFAULT_LOG_FILENAME).is_file()


def test_replay_uses_saved_configuration_after_source_changes(entrypoint, tmp_path, monkeypatch):
    """Pass the same resolved scenario to the runtime after source YAML changes.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real runner with a stubbed simulation runtime.
    tmp_path : pathlib.Path
        Workspace for original configuration and experiment outputs.
    monkeypatch : pytest.MonkeyPatch
        Fixture selecting paths and observing scenario execution.
    """
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / "opencda/scenario_testing/config_yaml"
    configs.mkdir(parents=True)
    (configs / "default.yaml").write_text("world:\n  seed: 11\n  client_port: 2000\n")
    (configs / "demo.yaml").write_text("world:\n  client_port: 3000\n")
    runtime = ModuleType("opencda.scenario_testing.scenario")
    runtime.run_scenario = Mock()
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(entrypoint, "create_logger", Mock())
    monkeypatch.setattr(entrypoint, "install_traceback_handler", Mock())
    monkeypatch.setattr(entrypoint, "show_performance_stats", Mock())
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {})
    original = tmp_path / "original"
    monkeypatch.setattr(entrypoint, "get_default_output_path", lambda *args: original)
    monkeypatch.setattr(sys, "argv", ["opencda.py", "-t", "demo", "--ticks", "50", "--save-vis"])
    entrypoint.main()
    original_config = runtime.run_scenario.call_args.args[1]
    (configs / "default.yaml").write_text("world:\n  seed: 999\n  added: true\n")
    (configs / "demo.yaml").unlink()
    monkeypatch.setattr(entrypoint, "get_default_output_path", lambda *args: tmp_path / "replay")
    monkeypatch.setattr(sys, "argv", ["opencda.py", "--replay-from", str(original)])
    entrypoint.main()
    options, replay_config = runtime.run_scenario.call_args.args
    assert replay_config == original_config
    assert replay_config.world.seed == 11 and replay_config.world.client_port == 3000
    assert "added" not in replay_config.world
    assert options.ticks == 50 and options.save_vis


def test_legacy_manifest_ignores_removed_capture_options(entrypoint, tmp_path):
    """Replay an earlier manifest using its resolved local model directory.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Runner exposing replay argument handling.
    tmp_path : pathlib.Path
        Directory holding an earlier manifest.
    """
    run_metadata.write_json(
        tmp_path / "run_manifest.json",
        {
            "launch": {"argv": ["-t", "demo", "--with-coperception", "--model-id", "model", "--save-predictions", "--auto-fetch-models"]},
            "checkpoint": {"local_dir": "/saved/model"},
        },
    )
    options, command, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path)])
    assert options.model_dir == "/saved/model" and options.model_id is None
    assert "--save-predictions" not in command and "--auto-fetch-models" not in command
