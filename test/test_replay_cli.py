"""Regression tests for replay argument and model-fetch policy round trips."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
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


@pytest.mark.parametrize("with_exception", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "https://review-user:TEST_LOG_PASSWORD@example.com/models?api_key=TEST_LOG_KEY",
        "https://example.com/models?api_key=TEST_LOG_KEY",
        "https://example.com/models#access_token=TEST_LOG_KEY",
    ],
)
def test_persisted_json_logs_redact_url_credentials(entrypoint, with_exception, url):
    """Keep credential-bearing diagnostics out of the shareable JSON log.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Runner exposing the production JSON log formatter.
    with_exception : bool
        Whether the record includes an exception with a credential-bearing URL.
    url : str
        Credentials carried by userinfo, query parameters or an OAuth fragment.
    """
    exc_info = None
    if with_exception:
        try:
            raise RuntimeError(f"Unable to fetch {url}")
        except RuntimeError:
            exc_info = sys.exc_info()
    record = logging.LogRecord("cavise.opencda.models", logging.WARNING, __file__, 1, "Fetching model from %s", (url,), exc_info)
    serialized = entrypoint.JsonFormatter().format(record)
    assert "TEST_LOG_PASSWORD" not in serialized
    assert "TEST_LOG_KEY" not in serialized
    assert "review-user" not in serialized
    payload = json.loads(serialized)
    assert payload["level"] == "WARNING"
    assert "example.com/models" in payload["message"]
    if with_exception:
        assert "RuntimeError" in payload["exception"]


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


@pytest.mark.parametrize("original_flag", [None, "--auto-fetch-models", "--no-auto-fetch-models"])
@pytest.mark.parametrize("override", [None, "--auto-fetch-models", "--no-auto-fetch-models"])
@pytest.mark.parametrize("environment", ["0", "1"])
def test_fetch_policy_survives_replay_and_environment_defaults(entrypoint, tmp_path, monkeypatch, original_flag, override, environment):
    """Honor explicit fetch choices while preserving environment defaults.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real runner module loaded by the fixture.
    tmp_path : pathlib.Path
        Directory containing the saved manifest and test model location.
    monkeypatch : pytest.MonkeyPatch
        Fixture controlling the environment and observing resolver arguments.
    original_flag, override : str or None
        Saved and replay fetch choices; None leaves the policy unspecified.
    environment : str
        Environment default, enabled or disabled.
    """
    monkeypatch.setenv("CAVISE_MODELS_AUTO_FETCH", environment)
    arguments = ["-t", "demo", "--with-coperception", "--model-id", "missing", "--with-advcp"]
    if original_flag:
        arguments.append(original_flag)
    _, saved_command, _ = entrypoint.resolve_launch_arguments(arguments)
    run_metadata.write_json(tmp_path / "run_manifest.json", {"launch": {"argv": saved_command}})
    replay_arguments = ["--replay-from", str(tmp_path)]
    if override:
        replay_arguments.append(override)
    restored, command, _ = entrypoint.resolve_launch_arguments(replay_arguments)
    selected = override or original_flag
    expected = None if selected is None else selected == "--no-auto-fetch-models"
    assert restored.no_auto_fetch_models is expected
    assert entrypoint.arg_parse(command).no_auto_fetch_models is expected
    resolve_bundle = Mock(return_value=tmp_path / "resolved")
    monkeypatch.setattr(entrypoint.model_resolver, "resolve_bundle", resolve_bundle)
    entrypoint.model_resolver.resolve_runtime_models(restored)
    assert [call.kwargs["auto_fetch"] for call in resolve_bundle.call_args_list] == [None if expected is None else not expected] * 2


def test_explicit_fetch_clones_missing_bundle_despite_disabled_environment(entrypoint, tmp_path, monkeypatch):
    """Fetch a real local Git bundle when the explicit CLI enables fetching.

    Parameters
    ----------
    entrypoint : types.ModuleType
        Real runner module loaded by the fixture.
    tmp_path : pathlib.Path
        Workspace for the source Git repository and sparse destination.
    monkeypatch : pytest.MonkeyPatch
        Fixture disabling the default automatic-fetch policy.
    """
    monkeypatch.setenv("CAVISE_MODELS_AUTO_FETCH", "0")
    source = tmp_path / "source"
    bundle = source / "coperception/test-model"
    bundle.mkdir(parents=True)
    (bundle / "artifact.bin").write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    (bundle / "meta.yaml").write_text(
        f"schema_version: 1\nid: test-model\nkind: coperception-checkpoint\nartifacts:\n  - path: artifact.bin\n    size: 5\n    sha256: {digest}\n"
    )
    subprocess.run(["git", "init", "--initial-branch=main", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=OpenCDA tests", "-c", "user.email=tests@example.invalid", "commit", "-m", "Add test model"],
        check=True,
        capture_output=True,
    )
    options, _, _ = entrypoint.resolve_launch_arguments(
        [
            "-t",
            "demo",
            "--with-coperception",
            "--model-id",
            "test-model",
            "--models-root",
            str(tmp_path / "checkout"),
            "--models-repository",
            source.as_uri(),
            "--auto-fetch-models",
        ]
    )
    entrypoint.model_resolver.resolve_runtime_models(options)
    assert (Path(options.model_dir) / "artifact.bin").read_bytes() == b"model"


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
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {"opencda": {"git": None}})
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
    assert manifest["current_time"] == "2026_11_11_00_00_00_2"
    assert scenario_module.run_scenario.call_args.kwargs["current_time"] == manifest["current_time"]
    assert (new_dir / entrypoint.DEFAULT_LOG_FILENAME).is_file()
