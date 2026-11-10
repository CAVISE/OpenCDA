"""Tests for run reproducibility artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from opencda.scenario_testing.utils import run_metadata


def _make_opt(**overrides) -> argparse.Namespace:
    values = {
        "test_scenario": "demo",
        "with_coperception": False,
        "with_advcp": False,
        "model_dir": None,
        "model_id": None,
        "save_predictions": True,
        "replay_from": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("kind", ["xodr", "sumocfg", "network", "routes", "additional"])
@pytest.mark.parametrize("change", ["modify", "delete"])
def test_selected_simulation_inputs_are_compared_without_git(tmp_path, monkeypatch, merged_config, kind, change):
    """Detect changed selected map and SUMO inputs in Git-free runs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary scenario workspace.
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing code provenance and the working directory.
    merged_config : omegaconf.DictConfig
        Resolved scenario configuration fixture.
    kind, change : str
        Selected runtime input and the mutation applied before replay.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {"opencda": {"git": None}})
    assets = tmp_path / "opencda/sumo-assets/demo"
    assets.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    network = external / "network.net.xml"
    network.write_text("<net/>\n")
    routes = assets / "routes.rou.xml"
    routes.write_text("<routes/>\n")
    additional = assets / "extra.add.xml"
    additional.write_text("<additional/>\n")
    config = assets / "demo.sumocfg"
    config.write_text(
        f'<configuration><input><net-file value="{network}"/><route-files value="routes.rou.xml"/>'
        '<additional-files value="extra.add.xml"/></input></configuration>'
    )
    xodr = assets / "demo.xodr"
    xodr.write_text("<OpenDRIVE/>\n")
    opt = _make_opt(xodr=True, cosim=kind != "xodr")
    original = run_metadata.write_run_artifacts(tmp_path / "run", opt=opt, argv=[], merged_config=merged_config, current_time="t")
    assert not any("Simulation inputs" in warning for warning in run_metadata.compare_with_original(original, opt))
    selected = {"xodr": xodr, "sumocfg": config, "network": network, "routes": routes, "additional": additional}[kind]
    assert run_metadata.sha256_file(selected) in original["simulation_inputs"].values()
    if change == "delete":
        selected.unlink()
    else:
        selected.write_text(selected.read_text() + "\n<!-- changed -->\n")
    assert any("Simulation inputs" in warning for warning in run_metadata.compare_with_original(original, opt))


def test_simulation_input_selection_and_legacy_manifests(tmp_path, monkeypatch):
    """Keep town-only scenarios and legacy manifests compatible.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty scenario workspace.
    monkeypatch : pytest.MonkeyPatch
        Fixture changing the working directory.
    """
    monkeypatch.chdir(tmp_path)
    assert run_metadata.collect_simulation_input_hashes(_make_opt()) == {}
    missing = _make_opt(xodr=True)
    hashes = run_metadata.collect_simulation_input_hashes(missing)
    assert hashes == {"opencda/sumo-assets/demo/demo.xodr": None}
    assert any("cannot be verified" in warning for warning in run_metadata.compare_with_original({"simulation_inputs": hashes}, missing))
    assert any("did not record simulation input hashes" in warning for warning in run_metadata.compare_with_original({}, missing))
    assert not any("simulation input hashes" in warning for warning in run_metadata.compare_with_original({}, _make_opt()))
    assets = tmp_path / "opencda/sumo-assets/demo"
    assets.mkdir(parents=True)
    config = assets / "demo.sumocfg"
    config.write_text("<invalid")
    opt = _make_opt(cosim=True)
    hashes = run_metadata.collect_simulation_input_hashes(opt)
    assert hashes["opencda/sumo-assets/demo/demo.sumocfg"] == run_metadata.sha256_file(config)
    assert hashes["opencda/sumo-assets/demo/demo.sumocfg:input-references"] is None
    assert any("cannot be verified" in warning for warning in run_metadata.compare_with_original({"simulation_inputs": hashes}, opt))
    config.write_text('<configuration><input><route-files value="first.xml, second.xml"/></input></configuration>')
    (assets / "first.xml").write_text("<routes/>")
    (assets / "second.xml").write_text("<routes/>")
    hashes = run_metadata.collect_simulation_input_hashes(opt)
    assert len(hashes) == 3 and all(hashes.values())
    assert any("Simulation inputs" in warning for warning in run_metadata.compare_with_original({"simulation_inputs": hashes}, _make_opt()))


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("opencda_entrypoint", Path(__file__).resolve().parents[1] / "opencda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def merged_config():
    default = OmegaConf.create({"world": {"seed": 11, "client_port": 2000}, "scenario": {"ticks": 10}})
    scenario = OmegaConf.create({"scenario": {"ticks": 50}, "derived": "${world.client_port}"})
    return OmegaConf.merge(default, scenario)


def test_write_run_artifacts_layout(tmp_path: Path, merged_config):
    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text("scenario:\n  ticks: 50\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    manifest = run_metadata.write_run_artifacts(
        run_dir,
        opt=_make_opt(),
        argv=["--test-scenario", "demo", "--ticks", "5"],
        raw_argv=["-t", "demo", "--ticks=5"],
        merged_config=merged_config,
        current_time="2026_11_05_12_00_00",
        input_files={"scenario_config": scenario_yaml, "missing": None},
    )

    repro = run_dir / run_metadata.REPRODUCIBILITY_DIRNAME
    saved_config = OmegaConf.load(repro / run_metadata.CONFIG_FILENAME)
    assert saved_config.scenario.ticks == 50
    assert saved_config.derived == 2000  # interpolation is resolved

    on_disk = json.loads((repro / run_metadata.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert on_disk["seed"] == 11 == manifest["seed"]
    assert on_disk["status"] == "running"
    assert on_disk["launch"]["argv"] == ["--test-scenario", "demo", "--ticks", "5"]
    assert on_disk["launch"]["raw_argv"] == ["-t", "demo", "--ticks=5"]
    assert on_disk["launch"]["command"] == "python3 opencda.py --test-scenario demo --ticks 5"
    assert on_disk["launch"]["options"]["test_scenario"] == "demo"
    assert on_disk["code"]["opencda"]["version"]
    assert "hostname" in on_disk["device"]
    assert on_disk["checkpoint"] is None
    assert (repro / "inputs" / "scenario_config.yaml").is_file()
    assert set(on_disk["inputs"]) == {"scenario_config"}

    assert not (repro / "reproduce.sh").exists()
    assert "reproduce_script" not in on_disk["launch"]


def test_missing_seed_is_recorded_as_null(tmp_path: Path):
    manifest = run_metadata.write_run_artifacts(
        tmp_path, opt=_make_opt(), argv=["--test-scenario", "demo"], merged_config=OmegaConf.create({"world": {}}), current_time="t"
    )
    assert manifest["seed"] is None


def _make_bundle(root: Path, bundle_id: str = "pointpillar_late") -> Path:
    bundle = root / "coperception" / bundle_id
    bundle.mkdir(parents=True)
    weights = bundle / "net_epoch10.pth"
    weights.write_bytes(b"weights")
    digest = run_metadata.sha256_file(weights)
    (bundle / "meta.yaml").write_text(
        f"schema_version: 1\nid: {bundle_id}\nkind: coperception-checkpoint\n"
        f"artifacts:\n  - path: net_epoch10.pth\n    size: 7\n    sha256: {digest}\n",
        encoding="utf-8",
    )
    (bundle / "config.yaml").write_text("postprocess:\n  nms_thresh: 0.15\n", encoding="utf-8")
    return bundle


def _artifact(checkpoint, path):
    return next(item for item in checkpoint["artifacts"] if item["path"] == path)


def test_checkpoint_without_git_is_identified_by_hash(tmp_path: Path, merged_config):
    bundle = _make_bundle(tmp_path / "models")
    opt = _make_opt(with_coperception=True, model_dir=str(bundle), model_id="pointpillar_late")

    manifest = run_metadata.write_run_artifacts(tmp_path / "run", opt=opt, argv=[], merged_config=merged_config, current_time="t")

    checkpoint = manifest["checkpoint"]
    assert checkpoint["id"] == "pointpillar_late"
    digest = run_metadata.sha256_file(bundle / "net_epoch10.pth")
    artifacts = {item["path"]: item for item in checkpoint["artifacts"]}
    assert artifacts["net_epoch10.pth"] == {"path": "net_epoch10.pth", "size": 7, "sha256": digest, "expected_sha256": digest, "matches_meta": True}
    # Hash the model config even if it is absent from meta.yaml.
    assert artifacts["config.yaml"]["sha256"] == run_metadata.sha256_file(bundle / "config.yaml")
    assert checkpoint["url"] is None
    assert (tmp_path / "run" / "reproducibility" / "inputs" / "model_config.yaml").is_file()
    assert manifest["outputs"]["predictions"] == "../predictions"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_checkpoint_links_to_models_repository_commit(tmp_path: Path, merged_config):
    models = tmp_path / "models"
    bundle = _make_bundle(models)
    _git(models, "init", "-q")
    _git(models, "remote", "add", "origin", "git@github.com:CAVISE/models.git")
    _git(models, "add", ".")
    _git(models, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "bundle")
    commit = _git(models, "rev-parse", "HEAD")
    opt = _make_opt(with_coperception=True, model_dir=str(bundle), model_id="pointpillar_late")

    checkpoint = run_metadata.collect_checkpoint_info(opt)

    assert checkpoint["commit"] == commit
    assert checkpoint["repository"] == "https://github.com/CAVISE/models"
    assert checkpoint["path_in_repository"] == "coperception/pointpillar_late"
    assert checkpoint["url"] == f"https://github.com/CAVISE/models/tree/{commit}/coperception/pointpillar_late"

    # Local model changes must not be attributed to the original commit.
    (bundle / "config.yaml").write_text("changed: true\n", encoding="utf-8")
    assert run_metadata.collect_checkpoint_info(opt)["url"] is None


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        ("https://github.com/CAVISE/models.git", "https://github.com/CAVISE/models"),
        ("git@github.com:CAVISE/models.git", "https://github.com/CAVISE/models"),
        ("ssh://git@github.com/CAVISE/models.git", "https://github.com/CAVISE/models"),
        ("/local/path", None),
        (None, None),
    ],
)
def test_repository_web_url(remote, expected):
    assert run_metadata.repository_web_url(remote) == expected


def test_finalize_updates_status(tmp_path: Path, merged_config):
    run_metadata.write_run_artifacts(tmp_path, opt=_make_opt(), argv=[], merged_config=merged_config, current_time="t")
    run_metadata.finalize_run_manifest(tmp_path, status="completed", duration_seconds=1.5)

    manifest = run_metadata.load_run_manifest(tmp_path)
    assert manifest["status"] == "completed"
    assert manifest["duration_seconds"] == 1.5
    assert manifest["finished_at"]


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_finalize_redacts_failure_urls(tmp_path: Path, merged_config, scheme):
    """Preserve failure diagnostics without persisting URL credentials."""
    run_metadata.write_run_artifacts(tmp_path, opt=_make_opt(), argv=[], merged_config=merged_config, current_time="t")
    original = run_metadata.load_run_manifest(tmp_path)
    url = f"{scheme}://review-user:TEST_FAILURE_PASSWORD@adapter.example/v1/status?token=TEST_FAILURE_TOKEN"
    error = repr(RuntimeError(f"Adapter {url} is not idle"))

    run_metadata.finalize_run_manifest(tmp_path, status="failed", duration_seconds=1.5, error=error)

    manifest = run_metadata.load_run_manifest(tmp_path)
    serialized = (tmp_path / "reproducibility/run_manifest.json").read_text()
    assert "TEST_FAILURE_PASSWORD" not in serialized
    assert "TEST_FAILURE_TOKEN" not in serialized
    assert "review-user" not in serialized
    assert "adapter.example/v1/status" in manifest["error"]
    assert "is not idle" in manifest["error"]
    assert manifest["status"] == "failed" and manifest["duration_seconds"] == 1.5
    assert manifest["started_at"] == original["started_at"]
    assert manifest["finished_at"]
    assert manifest["seed"] == original["seed"]
    assert manifest["config_sha256"] == original["config_sha256"]


def test_finalize_without_manifest_is_noop(tmp_path: Path):
    run_metadata.finalize_run_manifest(tmp_path, status="failed")
    assert not (tmp_path / run_metadata.REPRODUCIBILITY_DIRNAME / run_metadata.MANIFEST_FILENAME).exists()


def test_load_run_manifest_requires_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_metadata.load_run_manifest(tmp_path)


def test_compare_with_original_detects_checkpoint_change(tmp_path: Path, merged_config):
    bundle = _make_bundle(tmp_path / "models")
    weights = bundle / "net_epoch10.pth"
    # Calculate checksums from the files when meta.yaml is missing.
    (bundle / "meta.yaml").unlink()
    opt = _make_opt(with_coperception=True, model_dir=str(bundle))
    original = run_metadata.write_run_artifacts(tmp_path / "run", opt=opt, argv=[], merged_config=merged_config, current_time="t")

    assert not any("Model files" in warning for warning in run_metadata.compare_with_original(original, opt))

    weights.write_bytes(b"other")
    assert any("Model files" in warning for warning in run_metadata.compare_with_original(original, opt))


def test_weights_changed_without_updating_meta_are_detected(tmp_path: Path, merged_config, caplog):
    """Check hashes against actual model files, not just meta.yaml."""
    bundle = _make_bundle(tmp_path / "models")
    opt = _make_opt(with_coperception=True, model_dir=str(bundle))
    original = run_metadata.write_run_artifacts(tmp_path / "run", opt=opt, argv=[], merged_config=merged_config, current_time="t")

    (bundle / "net_epoch10.pth").write_bytes(b"tampered")  # meta.yaml still contains the original checksum

    with caplog.at_level("WARNING"):
        artifact = _artifact(run_metadata.collect_checkpoint_info(opt), "net_epoch10.pth")
    assert artifact["sha256"] == run_metadata.sha256_file(bundle / "net_epoch10.pth")
    assert artifact["sha256"] != artifact["expected_sha256"]
    assert artifact["matches_meta"] is False
    assert "does not match the checksum" in caplog.text
    assert any("Model files" in warning for warning in run_metadata.compare_with_original(original, opt))


def test_missing_weights_listed_in_meta_are_reported(tmp_path: Path):
    bundle = _make_bundle(tmp_path / "models")
    (bundle / "net_epoch10.pth").unlink()

    artifact = _artifact(run_metadata.collect_checkpoint_info(_make_opt(with_coperception=True, model_dir=str(bundle))), "net_epoch10.pth")

    assert artifact["sha256"] is None
    assert artifact["matches_meta"] is False


def test_load_prediction_records(tmp_path: Path):
    predictions = tmp_path / run_metadata.PREDICTIONS_DIRNAME
    predictions.mkdir()
    for tick in (2, 1):
        np.savez_compressed(
            predictions / f"tick_{tick:06d}.npz",
            tick=np.int64(tick),
            pred_boxes=np.zeros((1, 8, 3)),
            has_pred_boxes=np.bool_(True),
            pred_scores=np.array([0.9]),
            has_pred_scores=np.bool_(True),
            gt_boxes=np.empty((0,)),
            has_gt_boxes=np.bool_(False),
            score_threshold=np.float64(0.2),
        )

    records = run_metadata.load_prediction_records(tmp_path)
    assert [record["tick"] for record in records] == [1, 2]
    assert records[0]["pred_scores"].tolist() == [0.9]
    assert records[0]["gt_boxes"] is None
    assert float(records[0]["score_threshold"]) == pytest.approx(0.2)


def test_command_is_rebuilt_from_options():
    entrypoint = _load_entrypoint()
    opt = entrypoint.arg_parse(["--ticks=5", "-t", "demo", "-c", "--verbose", "1", "--no-save-predictions", "--carla-host", "carla"])

    argv = entrypoint.build_command_from_options(opt)

    # Check the canonical command without default options.
    assert argv == ["--test-scenario", "demo", "--no-save-predictions", "--cosim", "--verbose", "1", "--ticks", "5"]
    assert entrypoint.arg_parse(argv) == opt


def test_replay_restores_saved_arguments(tmp_path: Path, merged_config):
    entrypoint = _load_entrypoint()
    run_metadata.write_run_artifacts(
        tmp_path,
        opt=_make_opt(),
        argv=["--test-scenario", "demo", "--carla-host", "carla-1", "--ticks", "5"],
        merged_config=merged_config,
        current_time="t",
    )

    opt, argv, manifest = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), "--carla-host", "localhost"])

    assert manifest["seed"] == 11
    assert opt.test_scenario == "demo"
    assert opt.ticks == 5
    assert opt.carla_host == "localhost"  # CLI values override saved parameters.
    assert opt.replay_from == str(tmp_path.resolve())
    assert argv == ["--test-scenario", "demo", "--carla-host", "localhost", "--ticks", "5"]


def test_scenario_or_replay_is_required():
    entrypoint = _load_entrypoint()
    with pytest.raises(SystemExit):
        entrypoint.arg_parse([])
    assert entrypoint.arg_parse(["-t", "demo"]).save_predictions is True
    assert entrypoint.arg_parse(["-t", "demo", "--no-save-predictions"]).save_predictions is False


def test_code_diff_warnings(tmp_path: Path, caplog):
    with caplog.at_level("WARNING"):
        assert run_metadata._save_code_diff(tmp_path, {"dirty": False, "untracked_files": ["new_module.py"]}) is None
    assert "untracked files (1)" in caplog.text
    assert "only their names are saved" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        run_metadata._save_code_diff(tmp_path, {"dirty": False, "untracked_files": []})
    assert caplog.text == ""


def test_replay_detects_changed_model_config(tmp_path: Path, merged_config):
    """Changing the model config (for example, its NMS threshold) must warn on replay."""
    bundle = _make_bundle(tmp_path / "models")
    opt = _make_opt(with_coperception=True, model_dir=str(bundle))
    original = run_metadata.write_run_artifacts(tmp_path / "run", opt=opt, argv=[], merged_config=merged_config, current_time="t")

    (bundle / "config.yaml").write_text("postprocess:\n  nms_thresh: 0.5\n", encoding="utf-8")

    warnings = run_metadata.compare_with_original(original, opt)
    assert any("coperception/config.yaml" in warning for warning in warnings)


def test_replay_detects_changed_advcp_assets(tmp_path: Path, merged_config):
    bundle = _make_bundle(tmp_path / "models")
    assets = tmp_path / "models" / "advcp" / "base-car"
    assets.mkdir(parents=True)
    mesh = assets / "car.pt"
    mesh.write_bytes(b"mesh-v1")
    opt = _make_opt(with_coperception=True, model_dir=str(bundle), with_advcp=True, advcp_assets_dir=str(assets), advcp_assets_id="base-car")
    original = run_metadata.write_run_artifacts(tmp_path / "run", opt=opt, argv=[], merged_config=merged_config, current_time="t")
    assert not any("Model files" in warning for warning in run_metadata.compare_with_original(original, opt))

    mesh.write_bytes(b"mesh-v2")

    assert any("advcp/car.pt" in warning for warning in run_metadata.compare_with_original(original, opt))


def test_opencood_is_located_before_import(tmp_path: Path, monkeypatch):
    """Locate the OpenCOOD checkout before the package is imported."""
    package = tmp_path / "opencood_checkout" / "fakepkg_for_test"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.delitem(sys.modules, "fakepkg_for_test", raising=False)

    assert run_metadata._module_location("fakepkg_for_test") == package.resolve()
    assert "fakepkg_for_test" not in sys.modules


def test_replay_cli_model_dir_replaces_saved_model_id(tmp_path: Path, merged_config):
    entrypoint = _load_entrypoint()
    run_metadata.write_run_artifacts(
        tmp_path,
        opt=_make_opt(),
        argv=["--test-scenario", "demo", "--with-coperception", "--model-id", "pointpillar-late-opv2v-30"],
        merged_config=merged_config,
        current_time="t",
    )

    opt, argv, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), "--model-dir", "/models/local"])

    assert opt.model_dir == "/models/local"
    assert opt.model_id is None
    assert "--model-id" not in argv


def test_manifest_is_failed_when_scenario_import_fails(tmp_path: Path, monkeypatch):
    """Failures after manifest creation must update the run status."""
    import types

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
    import types

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["opencda.py", *argv])
    monkeypatch.setattr(entrypoint, "get_default_output_path", lambda scenario, current_time: run_dir)
    monkeypatch.setattr(entrypoint, "create_logger", lambda level, filename: entrypoint.logging.getLogger("test"))
    monkeypatch.setattr(entrypoint, "install_traceback_handler", lambda verbose: None)
    monkeypatch.setattr("opencda.core.plan.model_resolver.resolve_runtime_models", lambda opt: None)
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
        current_time="t",
        input_files={"advcp_config": repo / "opencda/scenario_testing/config_yaml/advcp-configs/example.yaml"},
    )

    manifest = _replay_main(entrypoint, monkeypatch, tmp_path / "replay", ["--replay-from", str(original_dir), *override])

    assert manifest["status"] == "failed"
    assert manifest["launch"]["options"]["advcp_config"].endswith(expected_suffix)


def test_replay_warns_when_opencood_commit_differs(monkeypatch):
    original = {"code": {"opencood": {"git": {"commit": "aaa", "dirty": False}}}}
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {"opencood": {"git": {"commit": "bbb", "dirty": False}}})

    warnings = run_metadata.compare_with_original(original, _make_opt())

    assert any("opencood commit differs" in warning for warning in warnings)


def test_replay_warns_when_gpu_differs(monkeypatch):
    original = {
        "device": {"torch": {"device": "cuda", "gpus": [{"name": "NVIDIA GeForce RTX 4060 Ti"}], "cuda_version": "13.0", "cudnn_version": 91000}}
    }
    current = {"torch": {"device": "cuda", "gpus": [{"name": "NVIDIA GeForce RTX 5070 Ti"}], "cuda_version": "13.0", "cudnn_version": 91000}}
    monkeypatch.setattr(run_metadata, "collect_device_info", lambda: current)

    warnings = run_metadata.compare_with_original(original, _make_opt())

    assert any(warning.startswith("GPU differs") for warning in warnings)
    assert not any(warning.startswith("CUDA version differs") for warning in warnings)


def test_replay_does_not_reuse_original_log_file(tmp_path: Path, merged_config):
    """Replay must not overwrite the original run log."""
    entrypoint = _load_entrypoint()
    run_metadata.write_run_artifacts(
        tmp_path,
        opt=_make_opt(),
        argv=["--test-scenario", "demo", "--log-file", str(tmp_path / "original.log.json")],
        merged_config=merged_config,
        current_time="t",
    )

    opt, argv, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path)])
    assert opt.log_file is None
    assert "--log-file" not in argv

    opt, _, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), "--log-file", "replay.log.json"])
    assert opt.log_file == "replay.log.json"


def test_replay_can_turn_prediction_saving_back_on(tmp_path: Path, merged_config):
    entrypoint = _load_entrypoint()
    run_metadata.write_run_artifacts(
        tmp_path,
        opt=_make_opt(),
        argv=["--test-scenario", "demo", "--no-save-predictions"],
        merged_config=merged_config,
        current_time="t",
    )

    opt, argv, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path)])
    assert opt.save_predictions is False
    assert "--no-save-predictions" in argv

    opt, argv, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), "--save-predictions"])
    assert opt.save_predictions is True
    assert "--no-save-predictions" not in argv and "--save-predictions" not in argv  # Default options are omitted


def test_saved_advcp_config_keeps_relative_asset_paths_working(tmp_path: Path, merged_config):
    """AdvCP resolves paths from its config directory, so saved relative paths become absolute."""
    import yaml

    config_dir = tmp_path / "advcp-configs"
    config_dir.mkdir()
    source = config_dir / "custom.yaml"
    source.write_text("mode: spoofing\ncar_mesh_path: meshes/car.ply\ncar_mesh_divide_path: /abs/divide.pkl\n", encoding="utf-8")

    run_metadata.write_run_artifacts(
        tmp_path / "run", opt=_make_opt(), argv=[], merged_config=merged_config, current_time="t", input_files={"advcp_config": source}
    )

    saved = yaml.safe_load((tmp_path / "run" / "reproducibility" / "inputs" / "advcp_config.yaml").read_text(encoding="utf-8"))
    assert saved["car_mesh_path"] == str((config_dir / "meshes" / "car.ply").resolve())
    assert saved["car_mesh_divide_path"] == "/abs/divide.pkl"
    assert saved["mode"] == "spoofing"


@pytest.mark.parametrize("path_key", run_metadata.ADVCP_PATH_KEYS)
def test_saved_advcp_symlink_uses_target_asset_directory(tmp_path: Path, merged_config, path_key):
    """Resolve linked YAML assets exactly as the runtime configuration loader."""
    import yaml

    actual = tmp_path / "actual"
    links = tmp_path / "links"
    for directory in (actual, links):
        (directory / "assets").mkdir(parents=True)
    asset = actual / "assets/mesh.bin"
    asset.write_bytes(b"runtime mesh")
    (links / "assets/mesh.bin").write_bytes(b"wrong mesh beside symlink")
    (actual / "attack.yaml").write_text(f"mode: spoofing\n{path_key}: assets/mesh.bin\n")
    source = links / "attack.yaml"
    source.symlink_to("../actual/attack.yaml")
    opt = _make_opt(with_coperception=True, with_advcp=True, advcp_config=str(source))
    run_dir = tmp_path / "run"

    manifest = run_metadata.write_run_artifacts(
        run_dir, opt=opt, argv=[], merged_config=merged_config, current_time="t", input_files={"advcp_config": source}
    )

    saved = run_dir / "reproducibility" / manifest["inputs"]["advcp_config"]["copy"]
    assert yaml.safe_load(saved.read_text())[path_key] == str(asset)
    assert manifest["inputs"]["advcp_config"]["source"] == str(source)
    assert manifest["inputs"]["advcp_config"]["sha256"] == run_metadata.sha256_file(saved)
    original_asset = manifest["checkpoint"]["advcp_overrides"]["artifacts"][0]
    assert original_asset["source"] == str(asset)
    assert original_asset["sha256"] == run_metadata.sha256_file(asset)
    source.unlink()
    replay_opt = _make_opt(with_coperception=True, with_advcp=True, advcp_config=str(saved))
    assert run_metadata._model_input_hashes(manifest["checkpoint"]) == run_metadata._model_input_hashes(
        run_metadata.collect_checkpoint_info(replay_opt)
    )


def test_advcp_config_without_relative_paths_is_copied_as_is(tmp_path: Path, merged_config):
    source = Path(__file__).resolve().parents[1] / "opencda/scenario_testing/config_yaml/advcp-configs/example.yaml"

    run_metadata.write_run_artifacts(
        tmp_path / "run", opt=_make_opt(), argv=[], merged_config=merged_config, current_time="t", input_files={"advcp_config": source}
    )

    assert (tmp_path / "run" / "reproducibility" / "inputs" / "advcp_config.yaml").read_bytes() == source.read_bytes()


def test_replay_warns_when_package_versions_differ(monkeypatch):
    original = {"code": {"packages": {"torch": "2.15.0", "numpy": "2.5.4"}}}
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {"packages": {"torch": "2.16.0", "numpy": "2.5.4"}})

    warnings = run_metadata.compare_with_original(original, _make_opt())

    assert any(warning.startswith("torch version differs") for warning in warnings)
    assert not any(warning.startswith("numpy") for warning in warnings)


def test_replay_warns_when_opencood_has_uncommitted_changes(monkeypatch):
    """Warn when the replayed OpenCOOD worktree contains uncommitted changes."""
    original = {"code": {"opencood": {"git": {"commit": "aaa", "dirty": True}}}}
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {"opencood": {"git": {"commit": "aaa", "dirty": True}}})

    warnings = run_metadata.compare_with_original(original, _make_opt())

    assert any("opencood replay worktree has uncommitted changes" in warning for warning in warnings)


def test_model_without_remote_does_not_crash_on_dirty_checkout(tmp_path: Path, caplog):
    """A model checkout can have no origin remote and still have local edits."""
    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    models = tmp_path / "models"
    bundle = _make_bundle(models)
    _git(models, "init", "-q")
    _git(models, "add", ".")
    _git(models, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "bundle")
    (bundle / "config.yaml").write_text("postprocess:\n  nms_thresh: 0.5\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = run_metadata.describe_bundle(str(bundle), bundle_id="pointpillar_late", opt=_make_opt())
    assert result["commit"] is None
    assert "Model bundle" in caplog.text


def test_authenticated_model_remote_is_redacted_from_logs(tmp_path: Path, caplog):
    """Dirty model bundles must never log Git URL credentials."""
    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    models = tmp_path / "models"
    bundle = _make_bundle(models)
    _git(models, "init", "-q")
    _git(models, "remote", "add", "origin", "https://user:TEST_SECRET_987@example.com/repo.git")
    _git(models, "add", ".")
    _git(models, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "bundle")
    (bundle / "config.yaml").write_text("changed: true\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = run_metadata.describe_bundle(str(bundle), bundle_id="pointpillar_late", opt=_make_opt())
    assert result["commit"] is None
    assert "TEST_SECRET_987" not in caplog.text
    assert "TEST_SECRET_987" not in str(result)


def test_opencood_source_is_fingerprinted_without_git(tmp_path: Path, monkeypatch):
    """Container installations without .git still record OpenCOOD provenance."""
    package = tmp_path / "opencood"
    package.mkdir()
    source = package / "inference.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")

    monkeypatch.setattr(run_metadata, "_module_location", lambda name: package if name == "opencood" else None)
    monkeypatch.setattr(run_metadata, "collect_git_info", lambda root: None)
    digest = run_metadata.collect_source_digest
    monkeypatch.setattr(run_metadata, "collect_source_digest", lambda directory=None: digest(directory) if directory else "fixed-opencda")

    first = run_metadata.collect_code_versions()
    assert first["opencood"]["git"] is None
    assert first["opencood"]["source_digest"]

    source.write_text("VERSION = 2\n", encoding="utf-8")
    second = run_metadata.collect_code_versions()
    assert first["opencood"]["source_digest"] != second["opencood"]["source_digest"]

    original = {"code": first}
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: second)
    monkeypatch.setattr(run_metadata, "collect_checkpoint_info", lambda _: None)
    warnings = run_metadata.compare_with_original(original, _make_opt())
    assert any("OpenCOOD source digest differs" in message for message in warnings)


def test_replay_warns_when_python_version_differs(monkeypatch):
    original = {"code": {"python": "3.11.0"}}
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {"python": "3.12.0"})
    warnings = run_metadata.compare_with_original(original, _make_opt())
    assert any("Python version differs" in warning for warning in warnings)


def test_config_integrity_and_legacy_manifest(tmp_path: Path, merged_config):
    """Reject modified configurations but allow older manifests without hashes."""
    entrypoint = _load_entrypoint()
    run_metadata.write_run_artifacts(tmp_path, opt=_make_opt(), argv=["--test-scenario", "demo"], merged_config=merged_config, current_time="t")
    repro = tmp_path / "reproducibility"
    config = repro / "config.yaml"
    manifest_path = repro / "run_manifest.json"
    original = config.read_text(encoding="utf-8")
    config.write_text(original + "\nscenario:\n  ticks: 999\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path)])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["config_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    opt, _, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path)])
    assert opt.test_scenario == "demo"


def test_saved_yaml_artifacts_redact_credentials(tmp_path: Path):
    """Config and copied input YAML files must not expose credentials."""
    entrypoint = _load_entrypoint()
    password = "TEST_CONFIG_SECRET_987"
    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text(f"auth:\n  password: {password}\n", encoding="utf-8")
    config = OmegaConf.create({"world": {"seed": 42}, "auth": {"password": password}, "service_auth": password})
    run_metadata.write_run_artifacts(
        tmp_path / "run",
        opt=_make_opt(),
        argv=["--test-scenario", "demo"],
        merged_config=config,
        current_time="t",
        input_files={"scenario_config": scenario_yaml},
    )
    repro = tmp_path / "run" / "reproducibility"
    manifest = run_metadata.load_run_manifest(tmp_path / "run")
    assert manifest["config_redacted"] is True
    assert manifest["inputs"]["scenario_config"]["redacted"] is True
    assert password not in (repro / "config.yaml").read_text(encoding="utf-8")
    assert password not in (repro / "inputs" / "scenario_config.yaml").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="redacted credentials"):
        entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path / "run")])


@pytest.mark.parametrize(
    "fake_diff",
    [
        "+repo=https://user:TEST_SECRET_123@example.com/models.git\n",
        "+password: TEST_SECRET_123\n",
        "+api_key=TEST_SECRET_123\n",
        '-API_KEY="TEST_SECRET_123"\n',
        " context_password: TEST_SECRET_123\n",
    ],
)
def test_sensitive_git_diff_is_not_saved(tmp_path: Path, monkeypatch, caplog, fake_diff):
    """Persisting a credential-bearing Git patch would expose repository access."""
    git_info = {"dirty": True, "untracked_files": []}
    monkeypatch.setattr(run_metadata, "_run_git", lambda *args, **kwargs: fake_diff)
    with caplog.at_level("WARNING"):
        result = run_metadata._save_code_diff(tmp_path, git_info)
    assert result is None
    assert not (tmp_path / run_metadata.CODE_DIFF_FILENAME).exists()
    assert "TEST_SECRET_123" not in caplog.text
    assert "refusing to persist" in caplog.text


def test_replay_warns_if_original_dirty_diff_is_missing(monkeypatch):
    original = {"code": {"opencda": {"git": {"dirty": True}}}, "code_diff": None}
    monkeypatch.setattr(run_metadata, "collect_code_versions", lambda: {"opencda": {"git": {"dirty": False}}})
    warnings = run_metadata.compare_with_original(original, _make_opt())
    assert any("code diff is not available" in warning for warning in warnings)


@pytest.mark.parametrize("key", ["model_path", "mesh_divide_path"])
def test_saved_advcp_legacy_path_aliases(tmp_path, key):
    import yaml

    source = tmp_path / "original" / "attack.yaml"
    source.parent.mkdir()
    source.write_text(f"{key}: assets/car.pt\n")
    target = tmp_path / "copy.yaml"
    run_metadata._copy_advcp_config(source, target)
    assert yaml.safe_load(target.read_text())[key] == str(source.parent / "assets/car.pt")
    source.write_text(f"{key}: /absolute/car.pt\n")
    run_metadata._copy_advcp_config(source, target)
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("field", ["machine", "platform"])
def test_replay_warns_when_cpu_environment_changes(monkeypatch, field):
    original = {"torch": {"device": "cpu"}, "machine": "x86_64", "platform": "Linux"}
    current = {**original, field: "different"}
    monkeypatch.setattr(run_metadata, "collect_device_info", lambda: current)
    warnings = run_metadata.compare_with_original({"device": original}, _make_opt())
    assert any("CPU" in warning and "different" in warning for warning in warnings)
    monkeypatch.setattr(run_metadata, "collect_device_info", lambda: original)
    assert not any("CPU" in warning for warning in run_metadata.compare_with_original({"device": original}, _make_opt()))


@pytest.mark.parametrize("override", [False, True])
def test_replay_checks_saved_advcp_hash(tmp_path, monkeypatch, merged_config, override):
    entrypoint = _load_entrypoint()
    source = tmp_path / "attack.yaml"
    source.write_text("mode: spoofing\n")
    run_dir = tmp_path / "original"
    run_metadata.write_run_artifacts(
        run_dir,
        opt=_make_opt(),
        argv=["-t", "single_town06_carla", "--with-coperception", "--with-advcp", "--advcp-config", str(source)],
        merged_config=merged_config,
        current_time="t",
        input_files={"advcp_config": source},
    )
    (run_dir / "reproducibility/inputs/advcp_config.yaml").write_text("mode: removal\n")
    argv = ["--replay-from", str(run_dir)]
    if override:
        manifest = _replay_main(entrypoint, monkeypatch, tmp_path / "new", [*argv, "--advcp-config", str(source)])
        assert manifest["launch"]["options"]["advcp_config"] == str(source)
    else:
        monkeypatch.setattr(sys, "argv", ["opencda.py", *argv])
        monkeypatch.setattr(entrypoint, "create_logger", lambda **kwargs: entrypoint.logging.getLogger("test"))
        monkeypatch.setattr(entrypoint, "install_traceback_handler", lambda **kwargs: None)
        with pytest.raises(ValueError, match="AdvCP configuration SHA-256 mismatch"):
            entrypoint.main()


def test_safe_git_patch_preserves_whitespace(tmp_path, monkeypatch):
    patch = "diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1,2 +1,2 @@\n-old = 1\n+new = 2\n \n"
    monkeypatch.setattr(run_metadata, "_run_git", lambda *args, **kwargs: patch)
    assert run_metadata._save_code_diff(tmp_path, {"dirty": True}) == run_metadata.CODE_DIFF_FILENAME
    assert (tmp_path / run_metadata.CODE_DIFF_FILENAME).read_text() == patch


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
        current_time="t",
        input_files={"advcp_config": source},
    )
    saved = original / "reproducibility/inputs/advcp_config.yml"
    assert saved.is_file()
    source.unlink()

    replay = _replay_main(entrypoint, monkeypatch, tmp_path / "replay", ["--replay-from", str(original)])
    assert replay["launch"]["options"]["advcp_config"] == str(saved.resolve())
    saved.write_text("mode: removal\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["opencda.py", "--replay-from", str(original)])
    with pytest.raises(ValueError, match="AdvCP configuration SHA-256 mismatch"):
        entrypoint.main()


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
        current_time="t",
        input_files={"advcp_config": source},
    )
    manifest_path = original / "reproducibility/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["advcp_config"]["copy"] = "../../attack.yaml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["opencda.py", "--replay-from", str(original)])
    with pytest.raises(ValueError, match="outside the run inputs directory"):
        entrypoint.main()


@pytest.mark.parametrize("path_key", ["car_mesh_path", "model_path", "mesh_divide_path", "remove_adv_shape_perturb_path"])
def test_replay_detects_external_advcp_override_changes(tmp_path: Path, merged_config, path_key: str):
    bundle = _make_bundle(tmp_path / "models")
    external = tmp_path / "external"
    external.mkdir()
    asset = external / "asset.pkl"
    asset.write_bytes(b"asset-v1")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "attack.yml"
    config.write_text(f"mode: spoofing\n{path_key}: ../external/asset.pkl\n", encoding="utf-8")
    opt = _make_opt(
        with_coperception=True,
        model_dir=str(bundle),
        with_advcp=True,
        advcp_config=str(config),
    )
    original = run_metadata.write_run_artifacts(
        tmp_path / "run", opt=opt, argv=[], merged_config=merged_config, current_time="t", input_files={"advcp_config": config}
    )
    hashes = run_metadata._model_input_hashes(original["checkpoint"])
    assert hashes[f"advcp_override/{path_key}"] == run_metadata.sha256_file(asset)
    assert not any("Model files differ" in warning for warning in run_metadata.compare_with_original(original, opt))
    asset.write_bytes(b"asset-v2")
    assert any(f"advcp_override/{path_key}" in warning for warning in run_metadata.compare_with_original(original, opt))
    asset.unlink()
    assert any(f"advcp_override/{path_key}" in warning for warning in run_metadata.compare_with_original(original, opt))


@pytest.mark.parametrize("failure_stage", [None, "start", "scenario", "stop"])
def test_managed_components_finalize_manifest(tmp_path, monkeypatch, failure_stage):
    from contextlib import contextmanager
    from types import ModuleType

    entrypoint = _load_entrypoint()
    run_dir = tmp_path / "run"
    events = []
    failure_url = "https://review-user:TEST_LIFECYCLE_PASSWORD@adapter.example/v1/status?api_key=TEST_LIFECYCLE_KEY"

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
        assert "TEST_LIFECYCLE_PASSWORD" not in manifest["error"]
        assert "TEST_LIFECYCLE_KEY" not in manifest["error"]
        assert "review-user" not in manifest["error"]


@pytest.mark.parametrize("credentials", [False, True])
def test_model_config_with_numpy_tags_is_copied_without_deserialization(tmp_path, merged_config, credentials):
    import yaml

    bundle = _make_bundle(tmp_path / "models")
    config = """model:
  grid_size: !!python/object/apply:numpy.core.multiarray._reconstruct
    args:
    - !!python/name:numpy.ndarray ''
    - !!python/tuple [0]
    - !!binary Yg==
    state: !!python/tuple [1, [3], null, false, !!binary YAEAAAAAAADIAAAAAAAAAAEAAAAAAAAA]
postprocess: &postprocess
  nms_thresh: 0.15
copy: *postprocess
"""
    if credentials:
        config += "password: TEST_NUMPY_PASSWORD\nendpoint: https://user:TEST_NUMPY_PASSWORD@example.com\n"
    (bundle / "config.yaml").write_text(config)
    run_dir = tmp_path / "run"
    manifest = run_metadata.write_run_artifacts(
        run_dir,
        opt=_make_opt(with_coperception=True, model_dir=str(bundle)),
        argv=[],
        merged_config=merged_config,
        current_time="t",
    )
    copied = run_dir / "reproducibility/inputs/model_config.yaml"
    saved = copied.read_text()
    assert manifest["inputs"]["model_config"]["redacted"] is credentials
    assert manifest["inputs"]["model_config"]["sha256"] == run_metadata.sha256_file(copied)
    assert "TEST_NUMPY_PASSWORD" not in saved
    if not credentials:
        assert saved == config
    original_nodes = dict((key.value, value) for key, value in yaml.compose(config).value)
    saved_nodes = dict((key.value, value) for key, value in yaml.compose(saved).value)
    assert yaml.serialize(saved_nodes["model"]) == yaml.serialize(original_nodes["model"])
    assert saved_nodes["postprocess"] is saved_nodes["copy"]


def test_yaml_redaction_never_executes_python_tags(tmp_path):
    import yaml

    marker = tmp_path / "should-not-exist"
    source = tmp_path / "model.yaml"
    source.write_text(f'payload: !!python/object/apply:os.system ["touch {marker}"]\napi_key: TEST_TAG_SECRET\n')
    assert run_metadata._redact_yaml_file(source)
    assert not marker.exists()
    assert "TEST_TAG_SECRET" not in source.read_text()
    nodes = dict((key.value, value) for key, value in yaml.compose(source.read_text()).value)
    assert nodes["payload"].tag == "tag:yaml.org,2002:python/object/apply:os.system"


def test_yaml_redaction_handles_recursive_aliases_and_null_credentials(tmp_path):
    source = tmp_path / "recursive.yaml"
    source.write_text("loop: &loop {self: *loop, api_key: TEST_ALIAS_SECRET}\npassword: null\n")
    assert run_metadata._redact_yaml_file(source)
    assert "TEST_ALIAS_SECRET" not in source.read_text()
    source.write_text("password: null\n")
    assert not run_metadata._redact_yaml_file(source)


@pytest.mark.parametrize(
    "config",
    [
        "password: &p TEST_ALIAS_SECRET\nconnection_value: *p\n",
        "connection_value: &p TEST_ALIAS_SECRET\npassword: *p\n",
        "password: &p [TEST_ALIAS_SECRET]\nconnection_value: *p\n",
        "password: &p {value: &v TEST_ALIAS_SECRET}\nconnection_value: *p\nchild: *v\n",
        "connection_value: &p {value: &v TEST_ALIAS_SECRET}\nchild: *v\npassword: *p\n",
        "password: &p {value: TEST_ALIAS_SECRET, self: *p}\nconnection_value: *p\n",
        "password: TEST_ALIAS_SECRET\nconnection_value: prefix/TEST_ALIAS_SECRET/suffix\n",
        "password: &p TEST_ALIAS_SECRET\n? *p\n: aliased mapping key\n",
        'password: &p 123456789\nconnection_value: *p\ncopied_number: "123456789"\n',
        'password: &p ""\nconnection_value: *p\n',
    ],
)
def test_copied_yaml_redacts_credentials_through_all_references(tmp_path, config):
    """Remove aliased and copied secrets from persisted source inputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    config : str
        YAML fixture containing credential aliases or duplicate values.
    """
    source = tmp_path / "source.yaml"
    original = config + "unrelated: preserved\n"
    source.write_text(original, encoding="utf-8")
    output = tmp_path / "run"

    inputs = run_metadata._copy_inputs(output, {"scenario_config": source}, _make_opt())

    saved_path = output / inputs["scenario_config"]["copy"]
    saved = saved_path.read_text(encoding="utf-8")
    assert "TEST_ALIAS_SECRET" not in saved
    assert "123456789" not in saved
    assert inputs["scenario_config"]["redacted"] is True
    assert inputs["scenario_config"]["sha256"] == run_metadata.sha256_file(saved_path)
    values = run_metadata.yaml.safe_load(saved)
    assert values["password"] == run_metadata.REDACTED
    if "connection_value" in values:
        assert values["connection_value"] == run_metadata.REDACTED
    assert values["unrelated"] == "preserved"
    assert source.read_text(encoding="utf-8") == original


def test_merged_config_redacts_copies_of_credential_values(tmp_path, merged_config):
    """Redact resolved credential interpolations and reject unsafe replay.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    merged_config : omegaconf.DictConfig
        Scenario configuration to augment with a credential interpolation.
    """
    config = OmegaConf.merge(merged_config, OmegaConf.create({"password": "TEST_MERGED_SECRET", "connection_value": "${password}"}))

    manifest = run_metadata.write_run_artifacts(tmp_path, opt=_make_opt(), argv=["-t", "demo"], merged_config=config, current_time="t")

    saved_path = tmp_path / "reproducibility/config.yaml"
    assert "TEST_MERGED_SECRET" not in saved_path.read_text(encoding="utf-8")
    saved = OmegaConf.load(saved_path)
    assert saved.password == saved.connection_value == run_metadata.REDACTED
    assert saved.world.seed == merged_config.world.seed
    assert manifest["config_redacted"] is True
    assert manifest["config_sha256"] == run_metadata.sha256_file(saved_path)
    with pytest.raises(ValueError, match="redacted credentials"):
        _load_entrypoint().resolve_launch_arguments(["--replay-from", str(tmp_path)])


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
@pytest.mark.parametrize("invalid_utf8", [False, True])
def test_saved_crlf_patch_is_byte_exact_and_reverse_applicable(tmp_path, monkeypatch, invalid_utf8):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "false")
    source = repo / "crlf.yaml"
    suffix = b"\xff" if invalid_utf8 else b""
    source.write_bytes(b"first: old" + suffix + b"\r\nsecond: 2\r\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "Add CRLF fixture")
    source.write_bytes(b"first: new" + suffix + b"\r\nsecond: 2\r\n")
    monkeypatch.setattr(run_metadata, "REPO_ROOT", repo)
    raw = subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--binary"], check=True, capture_output=True).stdout
    assert run_metadata._save_code_diff(tmp_path, {"dirty": True}) == run_metadata.CODE_DIFF_FILENAME
    saved = tmp_path / run_metadata.CODE_DIFF_FILENAME
    assert saved.read_bytes() == raw
    subprocess.run(["git", "-C", str(repo), "apply", "--check", "--reverse", str(saved)], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_bundle_commit_requires_ignored_weights_to_match_tree(tmp_path):
    models = tmp_path / "models"
    bundle = _make_bundle(models)
    (models / ".gitignore").write_text("*.pth\n")
    _git(models, "init", "-q")
    _git(models, "remote", "add", "origin", "https://github.com/CAVISE/models.git")
    _git(models, "add", ".")
    _git(models, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "Add model configuration")
    opt = _make_opt(with_coperception=True, model_dir=str(bundle))
    checkpoint = run_metadata.collect_checkpoint_info(opt)
    assert checkpoint["commit"] is None and checkpoint["url"] is None
    assert _artifact(checkpoint, "net_epoch10.pth")["sha256"] == run_metadata.sha256_file(bundle / "net_epoch10.pth")
    _git(models, "add", "-f", str(bundle / "net_epoch10.pth"))
    _git(models, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "Add model weights")
    assert run_metadata.collect_checkpoint_info(opt)["commit"] == _git(models, "rev-parse", "HEAD")
    _git(models, "update-index", "--assume-unchanged", str(bundle / "net_epoch10.pth"))
    (bundle / "net_epoch10.pth").write_bytes(b"different weights")
    assert not _git(models, "status", "--porcelain")
    checkpoint = run_metadata.collect_checkpoint_info(opt)
    assert checkpoint["commit"] is None and checkpoint["url"] is None


@pytest.mark.parametrize(
    "relative_path",
    ["AIM/models/mtp/GNN_mtl_gnn/weights/model.pth", "opencda/core/application/behavior/services/aim_server/assets/yaw_dict_10m.pkl"],
)
def test_git_free_provenance_detects_changed_aim_assets(tmp_path, monkeypatch, relative_path):
    asset = tmp_path / relative_path
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"first asset")
    monkeypatch.setattr(run_metadata, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_metadata, "_module_location", lambda name: None)
    original = {"code": run_metadata.collect_code_versions()}
    assert original["code"]["opencda"]["git"] is None
    assert not any("source digest" in warning for warning in run_metadata.compare_with_original(original, _make_opt()))
    asset.write_bytes(b"second asset")
    assert any("OpenCDA source digest differs" in warning for warning in run_metadata.compare_with_original(original, _make_opt()))
    asset.unlink()
    assert any("OpenCDA source digest differs" in warning for warning in run_metadata.compare_with_original(original, _make_opt()))


def test_replay_parses_cli_once_and_applies_explicit_defaults(tmp_path, monkeypatch):
    from unittest.mock import Mock

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
    ("saved", "override", "expected"),
    [("--no-auto-fetch-models", "--auto-fetch-models", False), ("--auto-fetch-models", "--no-auto-fetch-models", True)],
)
def test_replay_inverse_model_fetch_policy(tmp_path, saved, override, expected):
    """Preserve the existing negative flag and allow fetching to be re-enabled."""
    entrypoint = _load_entrypoint()
    run_metadata.write_json(tmp_path / "run_manifest.json", {"launch": {"argv": ["-t", "demo", saved]}})

    opt, command, _ = entrypoint.resolve_launch_arguments(["--replay-from", str(tmp_path), override])

    assert opt.no_auto_fetch_models is expected
    assert entrypoint.arg_parse(command).no_auto_fetch_models is expected
    assert "--no-no-auto-fetch-models" not in entrypoint.build_parser().format_help()


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
