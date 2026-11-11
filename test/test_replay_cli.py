"""Replay CLI and output preservation regressions."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

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


@pytest.mark.parametrize("selection", [None, "name", "path"])
def test_replay_uses_saved_configuration_after_source_changes(entrypoint, tmp_path, monkeypatch, selection):
    """Use the snapshot by default and merge an explicitly selected scenario.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real runner with a stubbed simulation runtime.
    tmp_path : pathlib.Path
        Workspace for original configuration and experiment outputs.
    monkeypatch : pytest.MonkeyPatch
        Fixture selecting paths and observing scenario execution.
    selection : str or None
        Explicit scenario name/path, or None to retain the saved configuration.
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
    arguments = ["opencda.py", "--replay-from", str(original)]
    if selection:
        other = configs / "other.yaml"
        other.write_text("world:\n  client_port: 4000\n")
        arguments += ["-t", "other" if selection == "name" else str(other), "--cosim"]
    monkeypatch.setattr(sys, "argv", arguments)
    entrypoint.main()
    options, replay_config = runtime.run_scenario.call_args.args
    if selection:
        assert options.test_scenario == "other" and options.cosim
        assert replay_config.world.seed == 999 and replay_config.world.client_port == 4000
        assert replay_config.world.added
        assert run_metadata.load_run_manifest(tmp_path / "replay")["scenario"] == "other"
        return
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


@pytest.mark.parametrize("override", [None, "--models-root", "--advcp-assets-id", "--models-repository", "--models-ref"])
@pytest.mark.parametrize("environment", [None, "different-checkout"])
def test_advcp_replay_preserves_resolved_assets(entrypoint, tmp_path, monkeypatch, override, environment):
    """Keep saved AdvCP assets unless an asset selector is explicitly overridden.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real scenario entrypoint and model resolver.
    tmp_path : pathlib.Path
        Workspace for original models, saved inputs and alternative assets.
    monkeypatch : pytest.MonkeyPatch
        Fixture controlling environment defaults and bundle resolution.
    override : str or None
        Explicit selector that should release the saved asset directory.
    environment : str or None
        Changed or removed models-root environment variable.
    """
    model = tmp_path / "perception"
    assets = tmp_path / "original-assets"
    alternative = tmp_path / "alternative-assets"
    for directory in (model, assets, alternative):
        directory.mkdir()
    config = tmp_path / "attack.yaml"
    config.write_text("mode: spoofing\n")
    opt, argv, _ = entrypoint.resolve_launch_arguments(
        ["-t", "demo", "--with-coperception", "--model-dir", str(model), "--with-advcp", "--advcp-config", str(config)]
    )
    monkeypatch.setenv("CAVISE_MODELS_ROOT", str(tmp_path / "original-checkout"))
    resolver = Mock(return_value=assets)
    monkeypatch.setattr(entrypoint.model_resolver, "resolve_bundle", resolver)
    entrypoint.model_resolver.resolve_runtime_models(opt)
    original = tmp_path / "run"
    run_metadata.write_run_artifacts(
        original, opt=opt, argv=argv, merged_config=OmegaConf.create({"world": {"seed": 11}}), input_files={"advcp_config": config}
    )
    assert run_metadata.load_run_manifest(original)["model"]["advcp_assets_dir"] == str(assets)
    if environment is None:
        monkeypatch.delenv("CAVISE_MODELS_ROOT")
    else:
        monkeypatch.setenv("CAVISE_MODELS_ROOT", str(tmp_path / environment))
    resolver.reset_mock()
    resolver.return_value = alternative
    arguments = ["--replay-from", str(original)]
    if override:
        arguments += [override, "explicit-value"]
    replay, _, _ = entrypoint.resolve_launch_arguments(arguments)
    entrypoint.model_resolver.resolve_runtime_models(replay)
    assert replay.advcp_assets_dir == str(alternative if override else assets)
    assert resolver.call_count == (1 if override else 0)
    if not override:
        assets.rmdir()
        with pytest.raises(entrypoint.model_resolver.ModelResolutionError, match="Saved AdvCP assets directory"):
            entrypoint.model_resolver.resolve_runtime_models(replay)
        resolver.assert_not_called()


@pytest.mark.parametrize("selector", ["--models-root", "--models-repository", "--models-ref"])
@pytest.mark.parametrize("logical_model", [False, True])
def test_replay_model_checkout_overrides(entrypoint, tmp_path, monkeypatch, selector, logical_model):
    """Honor checkout overrides for logical IDs while retaining explicit model paths.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real entrypoint and runtime resolver.
    tmp_path : pathlib.Path
        Workspace for model directories and the saved manifest.
    monkeypatch : pytest.MonkeyPatch
        Fixture observing bundle resolution without network access.
    selector : str
        Explicit checkout option supplied during replay.
    logical_model : bool
        Whether the original command selected a logical ID or a local directory.
    """
    old = tmp_path / "original-model"
    new = tmp_path / "new-model"
    old.mkdir()
    new.mkdir()
    arguments = ["-t", "demo", "--with-coperception"]
    arguments += ["--model-id", "example"] if logical_model else ["--model-dir", str(old)]
    run_metadata.write_json(tmp_path / "run_manifest.json", {"launch": {"argv": arguments}, "model": {"directory": str(old)}})
    resolver = Mock(return_value=new)
    monkeypatch.setattr(entrypoint.model_resolver, "resolve_bundle", resolver)
    options, _, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), selector, "override"])
    entrypoint.model_resolver.resolve_runtime_models(options)
    assert options.model_dir == str(new if logical_model else old)
    if logical_model:
        assert options.model_id == "example"
        expected_key = {"--models-root": "models_root", "--models-repository": "repository_url", "--models-ref": "ref"}[selector]
        assert str(resolver.call_args.kwargs[expected_key]) == "override"
    else:
        resolver.assert_not_called()
