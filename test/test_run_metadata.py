"""Regression coverage for scenario snapshots and replay behavior."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import subprocess
import types
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import Mock
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from opencda.scenario_testing.utils import run_metadata


def _make_opt(**overrides):
    values = {"test_scenario": "demo", "with_coperception": False, "with_advcp": False, "model_dir": None, "model_id": None}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def merged_config():
    return OmegaConf.merge(OmegaConf.create({"world": {"seed": 11}, "ticks": 10}), {"ticks": 50, "derived": "${world.seed}"})


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("opencda_entrypoint", Path(__file__).resolve().parents[1] / "opencda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.import_runtime_libs()
    return module


def test_scenario_or_replay_is_required():
    entrypoint = _load_entrypoint()
    with pytest.raises(SystemExit):
        entrypoint.arg_parse([])
    assert entrypoint.arg_parse(["-t", "demo"]).test_scenario == "demo"


def test_manifest_is_failed_when_scenario_import_fails(tmp_path: Path, monkeypatch):
    """Failures after manifest creation must update the run status."""
    entrypoint = _load_entrypoint()
    run_dir = tmp_path / "run"
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["opencda.py", "-t", "single_town06_carla"])
    monkeypatch.setattr(entrypoint, "get_default_output_path", lambda scenario, current_time: run_dir)
    monkeypatch.setattr(entrypoint, "create_logger", lambda level, filename: entrypoint.logging.getLogger("test"))
    monkeypatch.setattr(entrypoint, "install_traceback_handler", lambda verbose: None)
    # Importing a mock scenario without run_scenario should fail.
    monkeypatch.setitem(sys.modules, "opencda.scenario_testing.scenario", types.ModuleType("opencda.scenario_testing.scenario"))

    with pytest.raises(ImportError):
        entrypoint.main()

    manifest = run_metadata.load_run_manifest(run_dir)
    assert manifest["status"] == "failed"
    assert "ImportError" in manifest["error"]


def _replay_main(entrypoint, monkeypatch, run_dir: Path, argv: list[str]) -> dict:
    """Run main with a failing scenario stub and return the replay manifest."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["opencda.py", *argv])
    monkeypatch.setattr(entrypoint, "get_default_output_path", lambda scenario, current_time: run_dir)
    monkeypatch.setattr(entrypoint, "create_logger", lambda level, filename: entrypoint.logging.getLogger("test"))
    monkeypatch.setattr(entrypoint, "install_traceback_handler", lambda verbose: None)
    monkeypatch.setattr(
        "opencda.core.plan.model_resolver.resolve_runtime_models",
        lambda opt: setattr(opt, "model_dir", opt.model_dir or str(run_dir.parent / "model")),
    )
    monkeypatch.setitem(sys.modules, "opencda.scenario_testing.scenario", types.ModuleType("opencda.scenario_testing.scenario"))
    with pytest.raises(ImportError):
        entrypoint.main()
    return run_metadata.load_run_manifest(run_dir)


def test_replay_uses_saved_advcp_config(tmp_path: Path, monkeypatch, merged_config):
    entrypoint = _load_entrypoint()
    repo = Path(__file__).resolve().parents[1]
    original_dir = tmp_path / "original"
    run_metadata.write_run_artifacts(
        original_dir,
        opt=_make_opt(),
        argv=[
            "--test-scenario",
            "single_town06_carla",
            "--with-coperception",
            "--model-dir",
            str(tmp_path / "model"),
            "--with-advcp",
            "--advcp-config",
            "example",
        ],
        merged_config=merged_config,
        input_files={"advcp_config": repo / "opencda/scenario_testing/config_yaml/advcp-configs/example.yaml"},
    )

    manifest = _replay_main(entrypoint, monkeypatch, tmp_path / "replay", ["--replay-from", str(original_dir)])

    assert manifest["status"] == "failed"
    assert manifest["inputs"]["advcp_config"]["source"].endswith("reproducibility/inputs/advcp_config.yaml")


def test_replay_does_not_reuse_original_log_file(tmp_path: Path, merged_config):
    """Replay must not overwrite the original run log."""
    entrypoint = _load_entrypoint()
    run_metadata.write_run_artifacts(
        tmp_path,
        opt=_make_opt(),
        argv=["--test-scenario", "demo", "--log-file", str(tmp_path / "original.log.json")],
        merged_config=merged_config,
    )

    opt, argv, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path)])
    assert opt.log_file is None
    assert "--log-file" in argv  # Preserve launch history, but use a fresh output path at runtime.


def test_saved_advcp_yml_replay_uses_manifest_path(tmp_path: Path, monkeypatch, merged_config):
    entrypoint = _load_entrypoint()
    source = tmp_path / "original-config.yml"
    source.write_text("mode: spoofing\n", encoding="utf-8")
    original = tmp_path / "original"
    run_metadata.write_run_artifacts(
        original,
        opt=_make_opt(),
        argv=["-t", "single_town06_carla", "--with-coperception", "--with-advcp", "--with-advcp", "--advcp-config", str(source)],
        merged_config=merged_config,
        input_files={"advcp_config": source},
    )
    saved = original / "reproducibility/inputs/advcp_config.yml"
    assert saved.is_file()
    source.unlink()

    replay = _replay_main(entrypoint, monkeypatch, tmp_path / "replay", ["--replay-from", str(original)])
    assert replay["inputs"]["advcp_config"]["source"] == str(saved.resolve())


def test_replay_rejects_advcp_copy_path_traversal(tmp_path: Path, monkeypatch, merged_config):
    entrypoint = _load_entrypoint()
    original = tmp_path / "original"
    source = tmp_path / "attack.yaml"
    source.write_text("mode: spoofing\n", encoding="utf-8")
    run_metadata.write_run_artifacts(
        original,
        opt=_make_opt(),
        argv=["-t", "single_town06_carla", "--with-coperception", "--with-advcp", "--with-advcp", "--advcp-config", str(source)],
        merged_config=merged_config,
        input_files={"advcp_config": source},
    )
    manifest_path = original / "reproducibility/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["advcp_config"]["copy"] = "../../attack.yaml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["opencda.py", "--replay-from", str(original)])
    with pytest.raises(ValueError, match="outside the run inputs directory"):
        entrypoint.main()


@pytest.mark.parametrize("failure_stage", [None, "start", "scenario", "stop"])
def test_managed_components_finalize_manifest(tmp_path, monkeypatch, failure_stage):
    entrypoint = _load_entrypoint()
    run_dir = tmp_path / "run"
    events = []
    failure_url = "http://adapter.example/v1/status"

    @contextmanager
    def managed_components(opt):
        assert opt.manage_components
        events.append("start")
        if failure_stage == "start":
            raise RuntimeError(f"start failed: {failure_url}")
        try:
            yield
        finally:
            events.append("stop")
            if failure_stage == "stop":
                raise RuntimeError(f"stop failed: {failure_url}")

    def run_scenario(*args, **kwargs):
        events.append("scenario")
        if failure_stage == "scenario":
            raise RuntimeError(f"scenario failed: {failure_url}")

    scenario = ModuleType("opencda.scenario_testing.scenario")
    scenario.run_scenario = run_scenario
    lifecycle = ModuleType("opencda.scenario_testing.component_lifecycle")
    lifecycle.ManagedComponents = managed_components
    monkeypatch.setitem(sys.modules, scenario.__name__, scenario)
    monkeypatch.setattr(entrypoint, "component_lifecycle", lifecycle)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["opencda.py", "-t", "single_town06_carla", "--manage-components"])
    monkeypatch.setattr(entrypoint, "get_default_output_path", lambda *args: run_dir)
    monkeypatch.setattr(entrypoint, "create_logger", lambda *args, **kwargs: entrypoint.logging.getLogger("test"))
    monkeypatch.setattr(entrypoint, "install_traceback_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(entrypoint, "show_performance_stats", lambda stats: None)
    if failure_stage:
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            entrypoint.main()
    else:
        entrypoint.main()
    manifest = run_metadata.load_run_manifest(run_dir)
    assert manifest["status"] == ("failed" if failure_stage else "completed")
    assert events == (["start"] if failure_stage == "start" else ["start", "scenario", "stop"])
    if failure_stage:
        assert f"{failure_stage} failed" in manifest["error"]
        assert "adapter.example/v1/status" in manifest["error"]


@pytest.mark.parametrize("exit_code", [None, 0, 1, "scenario failed"])
@pytest.mark.parametrize("profile", [False, True])
def test_system_exit_finalizes_manifest(tmp_path, monkeypatch, exit_code, profile):
    """Record explicit scenario exits while preserving their exit semantics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated output directory for the run artifacts.
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing the simulator and configuring the CLI.
    exit_code : int, str or None
        Successful or failing SystemExit payload raised by the scenario.
    profile : bool
        Whether the profiled scenario execution path is exercised.
    """
    entrypoint = _load_entrypoint()
    run_dir = tmp_path / "run"
    exit_error = SystemExit(exit_code)

    scenario = ModuleType("opencda.scenario_testing.scenario")
    scenario.run_scenario = Mock(side_effect=exit_error)
    monkeypatch.setitem(sys.modules, scenario.__name__, scenario)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    arguments = ["opencda.py", "-t", "single_town06_carla"]
    if profile:
        arguments.append("--profile")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(entrypoint, "get_default_output_path", lambda *args: run_dir)
    monkeypatch.setattr(entrypoint, "create_logger", lambda *args, **kwargs: entrypoint.logging.getLogger("test"))
    monkeypatch.setattr(entrypoint, "install_traceback_handler", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit) as caught:
        entrypoint.main()
    assert caught.value is exit_error
    manifest = run_metadata.load_run_manifest(run_dir)
    successful = exit_code is None or exit_code == 0
    assert manifest["status"] == ("completed" if successful else "failed")
    assert manifest["error"] == (None if successful else repr(exit_error))
    if profile:
        assert (run_dir / "profile_output.prof").is_file()


def test_snapshot_contains_only_experiment_inputs(tmp_path, merged_config):
    """Retain resolved configuration and experiment inputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory supplied by pytest.
    merged_config : omegaconf.DictConfig
        Merged configuration containing an interpolation.
    """
    manifest = run_metadata.write_run_artifacts(tmp_path, opt=_make_opt(), argv=["-t", "demo"], merged_config=merged_config)
    assert set(manifest) == {"schema_version", "scenario", "launch", "model", "inputs", "status"}
    saved = run_metadata.reproducibility_dir(tmp_path)
    assert set(path.name for path in saved.iterdir()) == {"config.yaml", "run_manifest.json"}
    config = OmegaConf.load(saved / "config.yaml")
    assert config.ticks == 50 and config.derived == 11
    assert "${" not in (saved / "config.yaml").read_text()
    run_metadata.finalize_run_manifest(tmp_path, status="completed")
    assert run_metadata.load_run_manifest(saved)["status"] == "completed"


def test_selected_model_manifest_is_copied_without_loading_tagged_config(tmp_path, merged_config):
    """Copy existing model metadata without loading weights or Python YAML tags.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Workspace for the model and experiment.
    merged_config : omegaconf.DictConfig
        Final scenario configuration.
    """
    model = tmp_path / "model"
    model.mkdir()
    content = b"numpy_value: !!python/object/apply:numpy.core.multiarray._reconstruct []\n"
    (model / "config.yaml").write_bytes(content)
    (model / "meta.yaml").write_text("id: example-model\nartifacts: []\n")
    options = _make_opt(with_coperception=True, model_dir=str(model), model_id="example-model")
    run = tmp_path / "run"
    manifest = run_metadata.write_run_artifacts(
        run, opt=options, argv=["-t", "demo", "--with-coperception", "--model-id", "example-model"], merged_config=merged_config
    )
    saved = run_metadata.reproducibility_dir(run)
    assert (saved / manifest["inputs"]["model_config"]["copy"]).read_bytes() == content
    assert (saved / manifest["inputs"]["model_manifest"]["copy"]).read_bytes() == (model / "meta.yaml").read_bytes()
    assert manifest["model"] == {"id": "example-model", "directory": str(model)}
    replay, _, _ = _load_entrypoint().resolve_launch_arguments(["--replay-from", str(run)])
    assert replay.model_id is None and replay.model_dir == str(model)


@pytest.mark.parametrize("key", run_metadata.ADVCP_PATH_KEYS)
def test_advcp_paths_use_symlink_target_directory(tmp_path, merged_config, key):
    """Keep canonical and legacy AdvCP assets anchored to the actual YAML.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Workspace for the source YAML, symlink and run.
    merged_config : omegaconf.DictConfig
        Final scenario configuration.
    key : str
        Canonical or legacy asset path key.
    """
    assets = tmp_path / "assets"
    assets.mkdir()
    source = assets / "attack.yaml"
    source.write_text(f"{key}: mesh.obj\n")
    link = tmp_path / "attack.yaml"
    link.symlink_to(source)
    manifest = run_metadata.write_run_artifacts(
        tmp_path / "run", opt=_make_opt(), argv=[], merged_config=merged_config, input_files={"advcp_config": link}
    )
    saved = run_metadata.reproducibility_dir(tmp_path / "run") / manifest["inputs"]["advcp_config"]["copy"]
    assert OmegaConf.load(saved)[key] == str(assets / "mesh.obj")
    assert source.read_text() == f"{key}: mesh.obj\n"


@pytest.mark.parametrize("key", run_metadata.ADVCP_PATH_KEYS)
def test_advcp_absolute_symlink_name_is_preserved(tmp_path, merged_config, key):
    """Preserve absolute symlink names used to select attack reference dimensions.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Workspace for the mesh, absolute symlink and saved configuration.
    merged_config : omegaconf.DictConfig
        Final scenario configuration.
    key : str
        Canonical or legacy AdvCP asset path key.
    """
    mesh = tmp_path / "mesh.ply"
    mesh.write_bytes(b"mesh")
    link = tmp_path / "car_000000.ply"
    link.symlink_to(mesh)
    source = tmp_path / "attack.yaml"
    source.write_text(f"{key}: {link}\n")
    run = tmp_path / "run"
    manifest = run_metadata.write_run_artifacts(run, opt=_make_opt(), argv=[], merged_config=merged_config, input_files={"advcp_config": source})
    saved = run_metadata.reproducibility_dir(run) / manifest["inputs"]["advcp_config"]["copy"]
    assert OmegaConf.load(saved)[key] == str(link)
    assert Path(OmegaConf.load(saved)[key]).stem == "car_000000"
    assert source.read_text() == f"{key}: {link}\n"


@pytest.mark.parametrize("recorded_snapshot", [False, True])
def test_advcp_replay_distinguishes_uncaptured_and_missing_inputs(tmp_path, monkeypatch, merged_config, recorded_snapshot):
    """Use the CLI config when uncaptured, but reject a lost recorded snapshot.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Workspace for original and replayed runs.
    monkeypatch : pytest.MonkeyPatch
        Fixture substituting the simulation runtime.
    merged_config : omegaconf.DictConfig
        Saved final scenario configuration.
    recorded_snapshot : bool
        Whether a snapshot was recorded and subsequently lost.
    """
    source = tmp_path / "attack.yaml"
    source.write_text("mode: spoofing\n")
    original = tmp_path / "original"
    manifest = run_metadata.write_run_artifacts(
        original,
        opt=_make_opt(),
        merged_config=merged_config,
        argv=[
            "-t",
            "single_town06_carla",
            "--with-coperception",
            "--model-dir",
            str(tmp_path / "model"),
            "--with-advcp",
            "--advcp-config",
            str(source),
        ],
        input_files={"advcp_config": source} if recorded_snapshot else {},
    )
    if recorded_snapshot:
        (run_metadata.reproducibility_dir(original) / manifest["inputs"]["advcp_config"]["copy"]).unlink()
        with pytest.raises(FileNotFoundError, match="Saved AdvCP configuration"):
            _replay_main(_load_entrypoint(), monkeypatch, tmp_path / "replay", ["--replay-from", str(original)])
    else:
        replay = _replay_main(_load_entrypoint(), monkeypatch, tmp_path / "replay", ["--replay-from", str(original)])
        assert replay["inputs"]["advcp_config"]["source"] == str(source)


@pytest.mark.parametrize("key", run_metadata.ADVCP_PATH_KEYS)
def test_home_relative_advcp_paths_are_frozen(tmp_path, monkeypatch, merged_config, key):
    """Expand home-relative assets once while preserving absolute symlink names.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Workspace for home directories and the configuration snapshot.
    monkeypatch : pytest.MonkeyPatch
        Fixture changing the runtime home directory.
    merged_config : omegaconf.DictConfig
        Final scenario configuration.
    key : str
        Canonical or legacy asset path key.
    """
    home = tmp_path / "original-home"
    home.mkdir()
    (home / "mesh.ply").write_bytes(b"mesh")
    (home / "car_000000.ply").symlink_to(home / "mesh.ply")
    monkeypatch.setenv("HOME", str(home))
    source = tmp_path / "attack.yaml"
    source.write_text(f"{key}: ~/car_000000.ply\n")
    run = tmp_path / "run"
    manifest = run_metadata.write_run_artifacts(run, opt=_make_opt(), argv=[], merged_config=merged_config, input_files={"advcp_config": source})
    monkeypatch.setenv("HOME", str(tmp_path / "different-home"))
    copy = run_metadata.reproducibility_dir(run) / manifest["inputs"]["advcp_config"]["copy"]
    assert OmegaConf.load(copy)[key] == str(home / "car_000000.ply")


@pytest.mark.parametrize("flag, expected", [("--help", "usage:"), ("-h", "usage:"), ("--version", "OpenCDA")])
def test_cli_information_without_runtime_dependencies(flag, expected):
    """Run informational commands with site-packages disabled.

    Parameters
    ----------
    flag : str
        Help or version option that must exit before runtime imports.
    expected : str
        Text expected in the command output.
    """
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", str(root / "opencda.py"), flag],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
