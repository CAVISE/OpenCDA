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
    return module


def test_command_is_rebuilt_from_options():
    entrypoint = _load_entrypoint()
    opt = entrypoint.arg_parse(["--ticks=5", "-t", "demo", "-c", "--verbose", "1", "--carla-host", "carla"])

    argv = entrypoint.build_command_from_options(opt)

    # Check the canonical command without default options.
    assert argv == ["--test-scenario", "demo", "--cosim", "--verbose", "1", "--ticks", "5"]
    assert entrypoint.arg_parse(argv) == opt


def test_scenario_or_replay_is_required():
    entrypoint = _load_entrypoint()
    with pytest.raises(SystemExit):
        entrypoint.arg_parse([])
    assert entrypoint.arg_parse(["-t", "demo"]).test_scenario == "demo"


def test_replay_cli_model_dir_replaces_saved_model_id(tmp_path: Path, merged_config):
    entrypoint = _load_entrypoint()
    run_metadata.write_run_artifacts(
        tmp_path,
        opt=_make_opt(),
        argv=["--test-scenario", "demo", "--with-coperception", "--model-id", "pointpillar-late-opv2v-30"],
        merged_config=merged_config,
    )

    opt, argv, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), "--model-dir", "/models/local"])

    assert opt.model_dir == "/models/local"
    assert opt.model_id is None
    assert "--model-id" not in argv


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


@pytest.mark.parametrize(
    ("override", "expected_suffix"),
    [([], "reproducibility/inputs/advcp_config.yaml"), (["--advcp-config", "spoofing_forward"], "advcp-configs/spoofing_forward.yaml")],
)
def test_replay_uses_saved_advcp_config_unless_overridden(tmp_path: Path, monkeypatch, merged_config, override, expected_suffix):
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

    manifest = _replay_main(entrypoint, monkeypatch, tmp_path / "replay", ["--replay-from", str(original_dir), *override])

    assert manifest["status"] == "failed"
    assert manifest["inputs"]["advcp_config"]["source"].endswith(expected_suffix)


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
    assert "--log-file" not in argv

    opt, _, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), "--log-file", "replay.log.json"])
    assert opt.log_file == "replay.log.json"


def test_saved_advcp_yml_replay_uses_manifest_path(tmp_path: Path, monkeypatch, merged_config):
    entrypoint = _load_entrypoint()
    source = tmp_path / "original-config.yml"
    source.write_text("mode: spoofing\n", encoding="utf-8")
    original = tmp_path / "original"
    run_metadata.write_run_artifacts(
        original,
        opt=_make_opt(),
        argv=["-t", "single_town06_carla", "--with-coperception", "--with-advcp", "--advcp-config", str(source)],
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
        argv=["-t", "single_town06_carla", "--with-coperception", "--with-advcp", "--advcp-config", str(source)],
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


def test_replay_parses_cli_once_and_applies_explicit_defaults(tmp_path, monkeypatch):
    entrypoint = _load_entrypoint()
    run_metadata.write_json(
        tmp_path / "run_manifest.json",
        {
            "launch": {"argv": ["-t", "demo", "--verbose", "1", "--carla-host", "original", "--model-id", "original-model"]},
        },
    )
    parser = entrypoint.build_parser()
    parse_args = Mock(wraps=parser.parse_args)
    monkeypatch.setattr(parser, "parse_args", parse_args)
    build_parser = Mock(return_value=parser)
    monkeypatch.setattr(entrypoint, "build_parser", build_parser)
    cli = ["--replay-from", str(tmp_path), "--verbose=3", "--carla-host", "carla", "--model-dir=/new/model"]
    opt, command, _ = entrypoint.resolve_launch_arguments(cli)
    assert opt.verbose == entrypoint.VerbosityLevel.FULL
    assert opt.carla_host == "carla"
    assert opt.model_dir == "/new/model" and opt.model_id is None
    assert "--verbose" not in command and "--carla-host" not in command
    assert build_parser.call_count == 1
    assert parse_args.call_count == 2
    assert parse_args.call_args_list[0].args == (cli,)


@pytest.mark.parametrize(
    "flag",
    [
        "record",
        "free-spectator",
        "xodr",
        "cosim",
        "with-capi",
        "with-coperception",
        "show-video-vis",
        "save-vis",
        "save-npy",
        "with-advcp",
        "manage-components",
        "profile",
    ],
)
def test_replay_inverse_boolean_overrides_saved_true(tmp_path, flag):
    """Disable saved side effects and retain that choice in the next replay."""
    entrypoint = _load_entrypoint()
    assert getattr(entrypoint.arg_parse(["-t", "demo"]), flag.replace("-", "_")) is False
    run_metadata.write_json(tmp_path / "run_manifest.json", {"launch": {"argv": ["-t", "demo", f"--{flag}"]}})

    opt, command, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), f"--no-{flag}"])

    assert getattr(opt, flag.replace("-", "_")) is False
    assert f"--{flag}" not in command
    assert getattr(entrypoint.arg_parse(command), flag.replace("-", "_")) is False
    enabled = entrypoint.arg_parse(["-t", "demo", f"--{flag}"])
    canonical = entrypoint.build_command_from_options(enabled)
    assert f"--{flag}" in canonical
    assert entrypoint.arg_parse(canonical) == enabled


@pytest.mark.parametrize(
    "selectors", [["--model-id", "saved-model"], ["--model-dir", "/saved/model"], ["--model-id", "saved-model", "--model-dir", "/saved/model"]]
)
def test_replay_disabling_coperception_clears_saved_model_selectors(tmp_path, selectors):
    """Disable replay perception without resolving the saved model bundle.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    selectors : list of str
        Saved model arguments, including legacy manifests with both selectors.
    """
    entrypoint = _load_entrypoint()
    run_metadata.write_json(tmp_path / "run_manifest.json", {"launch": {"argv": ["-t", "demo", "--with-coperception", *selectors]}})

    opt, command, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), "--no-with-coperception"])

    assert opt.with_coperception is False
    assert opt.model_id is None and opt.model_dir is None
    assert "--model-id" not in command and "--model-dir" not in command
    entrypoint.model_resolver.resolve_runtime_models(opt)
    restored = entrypoint.arg_parse(command)
    entrypoint.model_resolver.resolve_runtime_models(restored)
    assert restored.with_coperception is False
    assert restored.model_id is None and restored.model_dir is None


@pytest.mark.parametrize("selector", ["--model-id", "--model-dir"])
def test_replay_disabling_coperception_preserves_explicit_selector_validation(tmp_path, selector):
    """Retain validation for an explicitly requested conflicting model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    selector : str
        Model selector explicitly supplied alongside disabled perception.
    """
    entrypoint = _load_entrypoint()
    run_metadata.write_json(tmp_path / "run_manifest.json", {"launch": {"argv": ["-t", "demo", "--with-coperception", "--model-id", "saved-model"]}})

    opt, _, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), "--no-with-coperception", selector, "explicit-model"])

    assert getattr(opt, selector.removeprefix("--").replace("-", "_")) == "explicit-model"
    with pytest.raises(entrypoint.model_resolver.ModelResolutionError, match="require --with-coperception"):
        entrypoint.model_resolver.resolve_runtime_models(opt)


def test_snapshot_contains_only_experiment_inputs(tmp_path, merged_config, monkeypatch):
    """Retain resolved configuration and commits without environment inventories.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory supplied by pytest.
    merged_config : omegaconf.DictConfig
        Merged configuration containing an interpolation.
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing local component discovery.
    """
    code = {"opencda": {"commit": "abc123", "dirty": True}}
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: code)
    manifest = run_metadata.write_run_artifacts(tmp_path, opt=_make_opt(), argv=["-t", "demo"], merged_config=merged_config)
    assert manifest["code"] == code
    assert set(manifest) == {"schema_version", "scenario", "launch", "code", "model", "inputs", "status"}
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
    overridden, _, _ = _load_entrypoint().resolve_launch_arguments(["--replay-from", str(run), "--model-id", "other"])
    assert overridden.model_id == "other" and overridden.model_dir is None


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


def test_component_revisions_from_real_checkout_and_explicit_fallback(tmp_path, monkeypatch):
    """Read local Git metadata and avoid reporting a parent as a missing child.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory outside a Git checkout.
    monkeypatch : pytest.MonkeyPatch
        Fixture configuring component locations and fallback revisions.
    """
    root = Path(__file__).resolve().parents[1]
    for name in ("CAVISE", "OPENCOOD", "SUMO", "ARTERY"):
        monkeypatch.setenv(f"{name}_WORKSPACE", str(tmp_path / name))
        monkeypatch.setenv(f"{name}_COMMIT", f"{name.lower()}-build-revision")
    monkeypatch.setenv("OPENCDA_WORKSPACE", str(root))
    monkeypatch.setenv("OPENCDA_COMMIT", "outdated-build-value")
    versions = run_metadata.collect_code_versions()
    expected = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert versions["opencda"]["commit"] == expected
    for name in ("cavise", "opencood", "sumo", "artery"):
        assert versions[name] == {"commit": f"{name}-build-revision", "dirty": None}
    monkeypatch.setenv("OPENCOOD_WORKSPACE", str(root / "opencda"))
    monkeypatch.delenv("OPENCOOD_COMMIT")
    assert run_metadata.collect_code_versions()["opencood"] == {"commit": None, "dirty": None}


@pytest.mark.parametrize("git_available", [False, True])
def test_unavailable_component_commits_do_not_abort(tmp_path, monkeypatch, git_available):
    """Keep an explicit unknown revision when Git or component sources are absent.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory without component checkouts.
    monkeypatch : pytest.MonkeyPatch
        Fixture controlling component paths and Git availability.
    git_available : bool
        Whether the Git executable can be located.
    """
    monkeypatch.setattr(run_metadata, "REPO_ROOT", tmp_path / "opencda")
    for name in ("CAVISE", "OPENCDA", "OPENCOOD", "SUMO", "ARTERY"):
        monkeypatch.delenv(f"{name}_COMMIT", raising=False)
        monkeypatch.delenv(f"{name}_WORKSPACE", raising=False)
    if not git_available:
        monkeypatch.setenv("PATH", "")
    assert all(value == {"commit": None, "dirty": None} for value in run_metadata.collect_code_versions().values())


def test_replay_compares_only_commits_and_dirty_flags(monkeypatch):
    """Warn about changed or unknown revisions and dirty original code.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture supplying the current component revisions.
    """
    current = {"opencda": {"commit": "new", "dirty": False}, "sumo": {"commit": None, "dirty": None}}
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: current)
    original = {"code": {"opencda": {"commit": "old", "dirty": True}, "sumo": {"commit": "sumo-old", "dirty": False}}}
    warnings = run_metadata.compare_with_original(original)
    assert len(warnings) == 3
    assert any("commit differs" in warning for warning in warnings)
    assert any("uncommitted" in warning for warning in warnings)
    assert any("unavailable" in warning for warning in warnings)
    assert run_metadata.compare_with_original({"code": {"opencda": current["opencda"]}}) == []
    legacy = {"code": {"opencda": {"git": current["opencda"], "source_digest": "ignored"}, "packages": {"torch": "ignored"}}}
    assert run_metadata.compare_with_original(legacy) == []
