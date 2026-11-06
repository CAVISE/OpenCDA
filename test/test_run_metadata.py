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
    assert saved_config.derived == 2000  # значение подставилось

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

    script = (repro / run_metadata.REPRODUCE_SCRIPT_FILENAME).read_text(encoding="utf-8")
    assert "# python3 opencda.py --test-scenario demo --ticks 5" in script
    assert 'python3 opencda.py --replay-from "$RUN_DIR"' in script


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
    # config.yaml модели хешируется, даже если его нет в meta.yaml
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

    # Локально изменённая модель не совпадает с коммитом.
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


def test_finalize_without_manifest_is_noop(tmp_path: Path):
    run_metadata.finalize_run_manifest(tmp_path, status="failed")
    assert not (tmp_path / run_metadata.REPRODUCIBILITY_DIRNAME / run_metadata.MANIFEST_FILENAME).exists()


def test_load_run_manifest_requires_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_metadata.load_run_manifest(tmp_path)


def test_compare_with_original_detects_checkpoint_change(tmp_path: Path, merged_config):
    bundle = _make_bundle(tmp_path / "models")
    weights = bundle / "net_epoch10.pth"
    # Без meta.yaml хеши считаются по файлам.
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

    (bundle / "net_epoch10.pth").write_bytes(b"tampered")  # в meta.yaml хеш не меняли

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

    # Проверяем формат команды без значений по умолчанию.
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
    assert opt.carla_host == "localhost"  # CLI перекрывает сохранённые параметры.
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
    """Изменённый config.yaml модели (например, порог NMS) должен давать предупреждение при повторе."""
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
    """OpenCOOD ещё не импортирован, когда пишется манифест, но его репозиторий всё равно находится."""
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
    """Если упало после записи манифеста (например, импорт сценария), статус не должен остаться running."""
    import types

    entrypoint = _load_entrypoint()
    run_dir = tmp_path / "run"
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["opencda.py", "-t", "single_town06_carla"])
    monkeypatch.setattr(entrypoint, "get_default_output_path", lambda scenario, current_time: run_dir)
    monkeypatch.setattr(entrypoint, "create_logger", lambda level, filename: entrypoint.logging.getLogger("test"))
    monkeypatch.setattr(entrypoint, "install_traceback_handler", lambda verbose: None)
    # модуль без run_scenario: import внутри main() упадёт с ImportError
    monkeypatch.setitem(sys.modules, "opencda.scenario_testing.scenario", types.ModuleType("opencda.scenario_testing.scenario"))

    with pytest.raises(ImportError):
        entrypoint.main()

    manifest = run_metadata.load_run_manifest(run_dir)
    assert manifest["status"] == "failed"
    assert "ImportError" in manifest["error"]
