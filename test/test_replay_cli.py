"""Replay CLI and output preservation regressions."""

from __future__ import annotations

import importlib.util
import hashlib
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
    module.import_runtime_libs()
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
    restored, replay_command, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path)])
    destination = flag[2:].replace("-", "_")
    assert getattr(original, destination) == value
    assert getattr(restored, destination) == (None if flag == "--log-file" else value)
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
    """Use the saved configuration even when the source changes.

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
    arguments = ["opencda.py", "--replay-from", str(original)]
    monkeypatch.setattr(sys, "argv", arguments)
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


@pytest.mark.parametrize("environment", [None, "different-checkout"])
def test_advcp_replay_preserves_resolved_assets(entrypoint, tmp_path, monkeypatch, environment):
    """Keep saved AdvCP assets when the environment changes.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real scenario entrypoint and model resolver.
    tmp_path : pathlib.Path
        Workspace for original models, saved inputs and alternative assets.
    monkeypatch : pytest.MonkeyPatch
        Fixture controlling environment defaults and bundle resolution.
    environment : str or None
        Changed or removed models-root environment variable.
    """
    model = tmp_path / "perception"
    assets = tmp_path / "original-assets"
    alternative = tmp_path / "alternative-assets"
    for directory in (model, assets, alternative):
        directory.mkdir()
    (assets / "mesh.ply").write_bytes(b"mesh")
    run_metadata.write_json(
        assets / "meta.yaml",
        {
            "schema_version": 1,
            "id": "base-car",
            "kind": "advcp-assets",
            "artifacts": [{"path": "mesh.ply", "size": 4, "sha256": hashlib.sha256(b"mesh").hexdigest()}],
        },
    )
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
    replay, _, _ = entrypoint.resolve_launch_arguments(arguments)
    entrypoint.model_resolver.resolve_runtime_models(replay)
    assert replay.advcp_assets_dir == str(assets)
    resolver.assert_not_called()
    (assets / "mesh.ply").unlink()
    (assets / "meta.yaml").unlink()
    assets.rmdir()
    with pytest.raises(entrypoint.model_resolver.ModelResolutionError, match="Saved AdvCP assets directory"):
        entrypoint.model_resolver.resolve_runtime_models(replay)
    resolver.assert_not_called()


@pytest.mark.parametrize("change", [None, "modify", "delete", "wrong-id"])
def test_saved_advcp_bundle_is_validated(entrypoint, tmp_path, monkeypatch, change):
    """Validate the saved bundle without fetching a replacement on failure.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real entrypoint and bundle validator.
    tmp_path : pathlib.Path
        Workspace for model, assets and saved run.
    monkeypatch : pytest.MonkeyPatch
        Fixture observing unexpected bundle fetching.
    change : str or None
        Same-size corruption, missing artifact, wrong identity or no change.
    """
    model = tmp_path / "model"
    model.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    mesh = assets / "mesh.ply"
    mesh.write_bytes(b"mesh")
    run_metadata.write_json(
        assets / "meta.yaml",
        {
            "schema_version": 1,
            "id": "other" if change == "wrong-id" else "base-car",
            "kind": "advcp-assets",
            "artifacts": [{"path": "mesh.ply", "size": 4, "sha256": hashlib.sha256(b"mesh").hexdigest()}],
        },
    )
    run_metadata.write_json(
        tmp_path / "run_manifest.json",
        {
            "launch": {"argv": ["-t", "demo", "--with-coperception", "--model-dir", str(model), "--with-advcp"]},
            "model": {"directory": str(model), "advcp_assets_dir": str(assets)},
        },
    )
    if change == "modify":
        mesh.write_bytes(b"oops")
    elif change == "delete":
        mesh.unlink()
    fetch = Mock()
    monkeypatch.setattr(entrypoint.model_resolver, "resolve_bundle", fetch)
    options, _, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path)])
    if change:
        message = {"modify": "checksum", "delete": "missing", "wrong-id": "ID does not match"}[change]
        with pytest.raises(entrypoint.model_resolver.ModelResolutionError, match=message):
            entrypoint.model_resolver.resolve_runtime_models(options)
    else:
        entrypoint.model_resolver.resolve_runtime_models(options)
        assert options.advcp_assets_dir == str(assets)
    fetch.assert_not_called()


@pytest.mark.parametrize("logical", [False, True])
@pytest.mark.parametrize("change", [None, "modify", "delete", "wrong-id"])
def test_perception_identity_and_validation_survive_replays(entrypoint, tmp_path, monkeypatch, logical, change):
    """Retain logical identity and validation across three successive replays.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real parser, replay handling and model resolver.
    tmp_path : pathlib.Path
        Workspace for model artifacts and successive manifests.
    monkeypatch : pytest.MonkeyPatch
        Fixture isolating the replay environment.
    logical : bool
        Whether the original launch uses a logical ID or explicit directory.
    change : str or None
        Mutation to the pinned checkpoint after successful replays.
    """
    directory = tmp_path / "model"
    directory.mkdir()
    weights = directory / "weights.pth"
    weights.write_bytes(b"data")
    metadata = {
        "schema_version": 1,
        "id": "example",
        "kind": "coperception-checkpoint",
        "artifacts": [{"path": "weights.pth", "size": 4, "sha256": hashlib.sha256(b"data").hexdigest()}],
    }
    run_metadata.write_json(directory / "meta.yaml", metadata)
    arguments = ["-t", "demo", "--with-coperception"] + (["--model-id", "example"] if logical else ["--model-dir", str(directory)])
    options, command, _ = entrypoint.resolve_launch_arguments(arguments)
    options.model_dir = str(directory)
    run = tmp_path / "original"
    config = OmegaConf.create({"world": {"seed": 11}})
    for index in range(3):
        run_metadata.write_run_artifacts(run, opt=options, argv=command, merged_config=config)
        assert run_metadata.load_run_manifest(run)["model"]["id"] == ("example" if logical else None)
        options, command, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(run)])
        entrypoint.model_resolver.resolve_runtime_models(options)
        previous = run
        run = tmp_path / f"replay-{index}"
    if change == "modify":
        weights.write_bytes(b"oops")
    elif change == "delete":
        weights.unlink()
    elif change == "wrong-id":
        metadata["id"] = "other"
        run_metadata.write_json(directory / "meta.yaml", metadata)
    options, _, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(previous)])
    if logical and change:
        with pytest.raises(entrypoint.model_resolver.ModelResolutionError):
            entrypoint.model_resolver.resolve_runtime_models(options)
    else:
        entrypoint.model_resolver.resolve_runtime_models(options)


@pytest.mark.parametrize("replay_form", ["split", "equals"])
@pytest.mark.parametrize(
    "extra",
    [
        ["--ticks", "50"],
        ["-t", "other"],
        ["--model-id", "other"],
        ["--model-dir=/new/model"],
        ["--models-root", "/new"],
        ["--models-ref", "other"],
        ["--models-repository", "other"],
        ["--advcp-assets-id", "other"],
        ["--advcp-config", "other"],
        ["--no-with-coperception"],
        ["--with-advcp"],
        ["--carla-host", "carla"],
        ["--verbose=3"],
        ["--log-file", "other.json"],
    ],
)
def test_replay_rejects_experiment_parameters(entrypoint, tmp_path, capsys, replay_form, extra):
    """Reject overrides before reading a manifest or starting a simulation.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Runner exposing the CLI parser.
    tmp_path : pathlib.Path
        Directory with no manifest, to verify validation happens first.
    capsys : pytest.CaptureFixture
        Captured parser error output.
    replay_form : str
        Split or equals spelling of the replay option.
    extra : list of str
        Experiment parameters that cannot accompany replay.
    """
    replay = ["--replay-from", str(tmp_path)] if replay_form == "split" else [f"--replay-from={tmp_path}"]
    with pytest.raises(SystemExit) as error:
        entrypoint.resolve_launch_arguments([*extra, *replay])
    assert error.value.code == 2
    assert "--replay-from must be used alone" in capsys.readouterr().err


@pytest.mark.parametrize("replay_form", ["split", "equals"])
def test_replay_retains_original_arguments(entrypoint, tmp_path, replay_form):
    """Preserve the saved command across successive replays.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Runner exposing the CLI parser.
    tmp_path : pathlib.Path
        Directory containing the original and replay manifests.
    replay_form : str
        Split or equals spelling of the replay option.
    """
    original = ["-t", "demo", "--ticks=50", "--save-vis", "--no-record", "--verbose", "1"]
    options, command, _ = entrypoint.resolve_launch_arguments(original)
    assert command == original
    for index in range(3):
        run = tmp_path / str(index)
        run.mkdir()
        run_metadata.write_json(run / "run_manifest.json", {"launch": {"argv": command}})
        replay = ["--replay-from", str(run)] if replay_form == "split" else [f"--replay-from={run}"]
        options, command, _ = entrypoint.resolve_launch_arguments(replay)
        assert command == original
        assert options.ticks == 50 and options.save_vis and not options.record
        assert options.verbose == 1
