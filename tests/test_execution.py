import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import frontierwright.service as service_module
from frontierwright.data import DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import (
    HardBudgets,
    PermissionLevel,
    compute_execution_request_digest,
)
from frontierwright.local_executor import atomic_write_json
from frontierwright.paths import TrainingPathId
from frontierwright.registry import Registry
from frontierwright.service import (
    add_local_dataset,
    calibrate_training_plan,
    create_training_plan,
    execute_training_plan,
    get_data_view,
    get_paths_view,
    get_run_view,
    import_local_model,
    preflight_training_calibration,
    prepare_dataset_snapshot,
    promote_candidate,
    reconcile_training_run,
    repair_run_receipt,
)


def make_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"champion-weights")
    return root


def make_data(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "sft.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    return root


def write_backend(root: Path) -> Path:
    root.mkdir(parents=True)
    script = root / "backend.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if request["operation"] == "calibrate":
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "feasible": True,
        "representative_steps": 3,
        "step_time_seconds": 0.05,
        "tokens_per_second": 2000.0,
        "peak_vram_bytes": 1024,
        "peak_ram_bytes": 2048,
        "projected_storage_bytes": 128,
        "projected_wall_seconds": 2.0,
        "gpu_count": 1
    }))
elif request["operation"] == "train":
    output = Path(request["output_root"]) / "model"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps({"model_type": "fixture-trained"}),
        encoding="utf-8",
    )
    (output / "model.safetensors").write_bytes(b"candidate-trained-weights")
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "output_model_path": str(output),
        "metrics": {"steps": 4, "loss": 0.25}
    }))
else:
    raise SystemExit(2)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec = root / "backend.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": "fixture-backend",
                "supported_paths": ["LORA_SFT"],
                "calibrate_argv": [sys.executable, str(script), "{request_json}"],
                "train_argv": [sys.executable, str(script), "{request_json}"],
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    return spec


def write_storage_watchdog_backend(root: Path) -> Path:
    root.mkdir(parents=True)
    script = root / "backend.py"
    script.write_text(
        """
import json
import sys
import time
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if request["operation"] == "calibrate":
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "feasible": True,
        "representative_steps": 1,
        "step_time_seconds": 0.05,
        "tokens_per_second": 1000.0,
        "peak_vram_bytes": 1024,
        "peak_ram_bytes": 2048,
        "projected_storage_bytes": 64,
        "projected_wall_seconds": 5.0,
        "gpu_count": 1
    }))
elif request["operation"] == "train":
    output = Path(request["output_root"]) / "model"
    output.mkdir(parents=True, exist_ok=True)
    (output / "oversized.bin").write_bytes(b"x" * 8192)
    time.sleep(5)
    (output / "completed.txt").write_text("should-not-exist", encoding="utf-8")
    (output / "config.json").write_text(
        json.dumps({"model_type": "watchdog-fixture"}),
        encoding="utf-8",
    )
    (output / "model.safetensors").write_bytes(b"candidate")
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "output_model_path": str(output),
        "metrics": {"steps": 1}
    }))
else:
    raise SystemExit(2)
""".strip()
        + chr(10),
        encoding="utf-8",
    )
    spec = root / "backend.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": "storage-watchdog-backend",
                "supported_paths": ["LORA_SFT"],
                "calibrate_argv": [sys.executable, str(script), "{request_json}"],
                "train_argv": [sys.executable, str(script), "{request_json}"],
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    return spec


def setup_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_model(tmp_path / "champion"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="NOVA",
    )
    add_local_dataset(
        project,
        make_data(tmp_path / "data"),
        name="SFT",
        role=DatasetRole.SFT,
        token_count=4000,
    )
    return project, write_backend(tmp_path / "backend")


def make_ready_plan(
    project: Path,
    backend_spec: Path,
    *,
    max_runs: int = 2,
) -> str:
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=max_runs, max_storage_bytes=4096),
        config={"epochs": 1},
    )
    calibrate_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=backend_spec,
    )
    return plan.plan_id or ""


def reserve_run(project: Path, plan_id: str) -> tuple[str, str, str]:
    registry = Registry(project)
    plan = registry.get_plan(plan_id)
    calibration = registry.latest_calibration_for_plan(plan_id)
    assert calibration is not None
    calibration_id = calibration["calibration_id"]
    assert isinstance(calibration_id, str)
    digest = compute_execution_request_digest(
        plan,
        calibration_id=calibration_id,
    )
    run_id, existing = registry.start_run(
        plan_id,
        rerun=False,
        calibration_id=calibration_id,
        request_digest=digest,
        owner_pid=os.getpid(),
    )
    assert existing is False
    return run_id, calibration_id, digest


def test_command_backend_full_plan_calibrate_run_candidate_flow(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)

    first = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=2, max_storage_bytes=4096),
        config={"epochs": 1},
    )
    second = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=2, max_storage_bytes=4096),
        config={"epochs": 1},
    )
    assert first.plan_id == second.plan_id
    assert first.ready is False
    assert "representative calibration required" in first.blockers

    calibrated = calibrate_training_plan(
        project,
        plan_id=first.plan_id or "",
        backend_spec_path=backend_spec,
    )
    assert calibrated.ready is True
    assert calibrated.calibration is not None
    assert calibrated.calibration["tokens_per_second"] == 2000.0

    paths = get_paths_view(project)
    lora = next(item for item in paths.paths if item["path_id"] == "LORA_SFT")
    assert lora["availability"] == "READY"
    assert lora["ready_plan_id"] == first.plan_id

    dry = execute_training_plan(
        project,
        plan_id=first.plan_id or "",
        backend_spec_path=backend_spec,
        dry_run=True,
        rerun=False,
    )
    assert dry.status == "DRY_RUN"
    assert dry.run_id is None

    champion_before = Registry(project).read().project["champion_id"]
    completed = execute_training_plan(
        project,
        plan_id=first.plan_id or "",
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )
    assert completed.status == "COMPLETED"
    assert completed.candidate_model_id is not None
    assert completed.metrics == {"steps": 4, "loss": 0.25}
    assert completed.usage["wall_seconds"] >= 0
    assert completed.usage["output_storage_bytes"] > 0
    assert completed.usage["gpu_count"] == 1
    assert completed.usage["gpu_count_provenance"] == "CALIBRATION_REPORTED"
    assert completed.usage["accounted_gpu_hours"] >= 0
    assert completed.usage["money_spent"] is None

    state = Registry(project).read()
    assert state.project["champion_id"] == champion_before
    assert len(state.candidates) == 1
    assert state.candidates[0].model.model_id == completed.candidate_model_id

    duplicate = execute_training_plan(
        project,
        plan_id=first.plan_id or "",
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )
    assert duplicate.run_id == completed.run_id
    assert len(Registry(project).read().candidates) == 1

    persisted = get_run_view(project, completed.run_id or "")
    assert persisted.status == "COMPLETED"


def test_plan_permission_blocks_calibration_and_execution(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.PLAN,
        budgets=HardBudgets(),
        config={},
    )

    with pytest.raises(FrontierwrightError, match="DRY_RUN permission"):
        calibrate_training_plan(
            project,
            plan_id=plan.plan_id or "",
            backend_spec_path=backend_spec,
        )


def test_backend_spec_drift_is_rejected(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.DRY_RUN,
        budgets=HardBudgets(),
        config={},
    )

    raw = json.loads(backend_spec.read_text(encoding="utf-8"))
    raw["environment"] = {"CHANGED": "1"}
    backend_spec.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(FrontierwrightError, match="spec content changed"):
        calibrate_training_plan(
            project,
            plan_id=plan.plan_id or "",
            backend_spec_path=backend_spec,
        )


def test_calibration_can_prove_budget_blocker_without_running(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_storage_bytes=64),
        config={},
    )
    calibrated = calibrate_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=backend_spec,
    )
    assert calibrated.ready is False
    assert "projected storage exceeds max_storage_bytes budget" in calibrated.blockers

    with pytest.raises(FrontierwrightError, match="projected storage"):
        execute_training_plan(
            project,
            plan_id=plan.plan_id or "",
            backend_spec_path=backend_spec,
            dry_run=False,
            rerun=False,
        )
    assert Registry(project).read().runs == ()


def test_plan_reports_actual_budget_enforcement_modes(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_BOUNDED,
        budgets=HardBudgets(
            max_wall_seconds=30,
            max_gpu_hours=1,
            max_runs=2,
            max_storage_bytes=4096,
        ),
        config={},
    )
    calibrated = calibrate_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=backend_spec,
    )

    assert calibrated.ready is True
    assert calibrated.budget_enforcement == {
        "max_gpu_hours": "HARD_ACCOUNTED_TIMEOUT:CALIBRATION_REPORTED",
        "max_runs": "HARD_REGISTRY_ADMISSION",
        "max_storage_bytes": "RUNTIME_WATCHDOG_AND_FINALIZATION_GATE",
        "max_wall_seconds": "HARD_LOCAL_TIMEOUT",
    }


def test_money_budget_stays_blocked_without_runtime_enforcement(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_BOUNDED,
        budgets=HardBudgets(max_money=10),
        config={},
    )
    calibrated = calibrate_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=backend_spec,
    )

    assert calibrated.ready is False
    assert calibrated.budget_enforcement["max_money"] == "NOT_RUNTIME_ENFORCEABLE"
    assert "money budget cannot be evaluated without projected_money" in calibrated.blockers


def test_max_runs_one_still_replays_completed_run(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec, max_runs=1)

    first = execute_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )
    second = execute_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )

    assert first.status == "COMPLETED"
    assert second.run_id == first.run_id
    assert len(Registry(project).read().runs) == 1


def test_completed_replay_survives_champion_change(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec, max_runs=1)
    completed = execute_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )
    assert completed.candidate_model_id is not None

    Registry(project).promote_candidate(completed.candidate_model_id)

    replay = execute_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )
    assert replay.run_id == completed.run_id
    assert replay.status == "COMPLETED"


def test_rerun_is_blocked_while_prior_attempt_is_unresolved(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec, max_runs=3)
    run_id, _, _ = reserve_run(project, plan_id)
    Registry(project).attach_run_worker(
        run_id,
        worker_pid=os.getpid(),
        worker_start_token=None,
        result_path=str(tmp_path / "missing-result.json"),
    )

    with pytest.raises(FrontierwrightError, match="prior attempt"):
        execute_training_plan(
            project,
            plan_id=plan_id,
            backend_spec_path=backend_spec,
            dry_run=False,
            rerun=True,
        )

    run = Registry(project).get_run(run_id)
    assert run["status"] == "RUNNING"
    assert run["liveness_state"] == "UNRESOLVED"


def test_dead_worker_without_result_reconciles_to_incomplete(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec)
    run_id, _, _ = reserve_run(project, plan_id)
    Registry(project).attach_run_worker(
        run_id,
        worker_pid=999_999_999,
        worker_start_token=None,
        result_path=str(tmp_path / "missing-result.json"),
    )

    reconciled = reconcile_training_run(project, run_id)

    assert reconciled.status == "INCOMPLETE"
    assert reconciled.error_code == "EXECUTOR_LOST"
    assert reconciled.liveness_state == "TERMINAL"


def test_durable_result_reconciliation_creates_exactly_one_candidate(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec)
    run_id, calibration_id, digest = reserve_run(project, plan_id)

    output = tmp_path / "recovered-output"
    output.mkdir()
    (output / "config.json").write_text(
        json.dumps({"model_type": "fixture-recovered"}),
        encoding="utf-8",
    )
    (output / "model.safetensors").write_bytes(b"recovered-candidate-weights")

    result_path = tmp_path / "executor-result.json"
    Registry(project).attach_run_worker(
        run_id,
        worker_pid=999_999_998,
        worker_start_token=None,
        result_path=str(result_path),
    )
    atomic_write_json(
        result_path,
        {
            "schema_version": 1,
            "ok": True,
            "run_id": run_id,
            "plan_id": plan_id,
            "calibration_id": calibration_id,
            "request_digest": digest,
            "output_model_path": str(output),
            "metrics": {"steps": 4, "loss": 0.2},
            "usage": {
                "wall_seconds": 3.5,
                "output_storage_bytes": 64,
                "gpu_count": 2,
                "gpu_count_provenance": "CALIBRATION_REPORTED",
                "accounted_gpu_hours": 3.5 * 2 / 3600,
                "money_spent": None,
                "measured_by": "fixture-executor",
            },
            "worker_pid": 999_999_998,
        },
    )

    first = reconcile_training_run(project, run_id)
    second = reconcile_training_run(project, run_id)

    state = Registry(project).read()
    assert first.status == "COMPLETED"
    assert second.run_id == first.run_id
    assert first.candidate_model_id is not None
    assert first.usage["wall_seconds"] == 3.5
    assert first.usage["gpu_count"] == 2
    assert first.usage["measured_by"] == "fixture-executor"
    assert len(state.candidates) == 1
    assert state.project["champion_id"] != first.candidate_model_id


def test_runtime_storage_watchdog_terminates_backend_before_completion(
    tmp_path: Path,
) -> None:
    project, _ = setup_project(tmp_path)
    backend_spec = write_storage_watchdog_backend(tmp_path / "watchdog-backend")
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_BOUNDED,
        budgets=HardBudgets(
            max_wall_seconds=20,
            max_storage_bytes=1024,
        ),
        config={},
    )
    calibrated = calibrate_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=backend_spec,
    )
    assert calibrated.ready is True
    assert calibrated.budget_enforcement["max_storage_bytes"] == (
        "RUNTIME_WATCHDOG_AND_FINALIZATION_GATE"
    )

    with pytest.raises(FrontierwrightError, match="runtime storage budget"):
        execute_training_plan(
            project,
            plan_id=plan.plan_id or "",
            backend_spec_path=backend_spec,
            dry_run=False,
            rerun=False,
        )

    state = Registry(project).read()
    assert len(state.runs) == 1
    run = state.runs[0]
    assert run["status"] == "INCOMPLETE"
    assert run["error_code"] == "STORAGE_BUDGET_REACHED"
    assert state.candidates == ()
    output_root = registry_output = project / ".frontierwright" / "candidates" / str(run["run_id"])
    assert (registry_output / "model" / "oversized.bin").is_file()
    assert not (output_root / "model" / "completed.txt").exists()


def test_measured_output_storage_budget_blocks_candidate_finalization(
    tmp_path: Path,
) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec)
    run_id, calibration_id, digest = reserve_run(project, plan_id)

    output = tmp_path / "budget-output"
    output.mkdir()
    (output / "config.json").write_text(
        json.dumps({"model_type": "fixture-budget"}),
        encoding="utf-8",
    )
    (output / "model.safetensors").write_bytes(b"small-candidate")

    result_path = tmp_path / "budget-result.json"
    Registry(project).attach_run_worker(
        run_id,
        worker_pid=999_999_997,
        worker_start_token=None,
        result_path=str(result_path),
    )
    atomic_write_json(
        result_path,
        {
            "schema_version": 1,
            "ok": True,
            "run_id": run_id,
            "plan_id": plan_id,
            "calibration_id": calibration_id,
            "request_digest": digest,
            "output_model_path": str(output),
            "metrics": {"steps": 1},
            "usage": {
                "wall_seconds": 1.0,
                "output_storage_bytes": 5000,
                "gpu_count": 1,
                "gpu_count_provenance": "CALIBRATION_REPORTED",
                "accounted_gpu_hours": 1 / 3600,
                "money_spent": None,
                "measured_by": "fixture-executor",
            },
            "worker_pid": 999_999_997,
        },
    )

    reconciled = reconcile_training_run(project, run_id)

    assert reconciled.status == "INCOMPLETE"
    assert reconciled.error_code == "STORAGE_BUDGET_REACHED"
    assert reconciled.candidate_model_id is None
    assert reconciled.usage["output_storage_bytes"] == 5000
    assert Registry(project).read().candidates == ()


def test_receipt_export_failure_does_not_downgrade_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec)

    def fail_receipt(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FrontierwrightError("STATE_FILE_ERROR", "simulated export failure", 4)

    monkeypatch.setattr(service_module, "_publish_run_receipt", fail_receipt)

    completed = execute_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )

    assert completed.status == "COMPLETED"
    persisted = Registry(project).get_run(completed.run_id or "")
    assert persisted["status"] == "COMPLETED"
    assert persisted["candidate_model_id"] == completed.candidate_model_id


def test_missing_run_receipt_is_repairable_from_registry(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec)
    completed = execute_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )
    assert completed.run_id is not None
    receipt = Registry(project).state_dir / "runs" / f"{completed.run_id}.json"
    assert receipt.is_file()
    receipt.unlink()

    repaired = repair_run_receipt(project, completed.run_id)

    assert repaired.status == "COMPLETED"
    assert receipt.is_file()


def test_concurrent_identical_admission_creates_one_active_run(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec, max_runs=2)
    registry = Registry(project)
    plan = registry.get_plan(plan_id)
    calibration = registry.latest_calibration_for_plan(plan_id)
    assert calibration is not None
    calibration_id = calibration["calibration_id"]
    assert isinstance(calibration_id, str)
    digest = compute_execution_request_digest(plan, calibration_id=calibration_id)

    def admit() -> tuple[str, bool]:
        return Registry(project).start_run(
            plan_id,
            rerun=False,
            calibration_id=calibration_id,
            request_digest=digest,
            owner_pid=os.getpid(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: admit(), range(2)))

    run_ids = {run_id for run_id, _ in results}
    assert len(run_ids) == 1
    state = Registry(project).read()
    assert len(state.runs) == 1
    assert state.runs[0]["status"] == "RUNNING"


def test_completed_run_registers_sealed_managed_artifact(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec)

    completed = execute_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )

    assert completed.candidate_model_id is not None
    registry = Registry(project)
    record = registry.get_sealed_artifact(completed.candidate_model_id)
    assert record is not None

    model = registry.get_model(completed.candidate_model_id)
    managed_root = (registry.state_dir / "artifacts").resolve()
    checkpoint = Path(model.checkpoint).resolve()
    assert checkpoint.is_relative_to(managed_root)
    assert checkpoint.exists()
    assert Path(str(record["manifest_path"])).is_file()
    assert str(record["model_fingerprint"]) == model.fingerprint
    assert str(record["run_id"]) == completed.run_id


def test_tampered_sealed_candidate_cannot_be_promoted_with_override(
    tmp_path: Path,
) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan_id = make_ready_plan(project, backend_spec)

    completed = execute_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )

    assert completed.candidate_model_id is not None
    registry = Registry(project)
    record = registry.get_sealed_artifact(completed.candidate_model_id)
    assert record is not None

    model_path = Path(str(record["model_path"]))
    weights = model_path / "model.safetensors"
    assert weights.is_file()
    weights.write_bytes(b"tampered-candidate-weights")

    with pytest.raises(FrontierwrightError, match="artifact"):
        promote_candidate(
            project,
            completed.candidate_model_id,
            allow_unmeasured=True,
        )




def test_distillation_run_records_explicit_distilled_from_lineage(
    tmp_path: Path,
) -> None:
    project = tmp_path / "distill-project"
    teacher = make_model(tmp_path / "distill-teacher")
    import_local_model(
        project,
        teacher,
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Teacher",
    )
    corpus = tmp_path / "distill-data"
    corpus.mkdir()
    (corpus / "train.txt").write_text(
        "knowledge distillation corpus\n" * 20,
        encoding="utf-8",
    )
    add_local_dataset(
        project,
        corpus,
        name="Distillation corpus",
        role=DatasetRole.PRETRAIN,
        token_count=1000,
    )

    backend_root = tmp_path / "distill-backend"
    backend_root.mkdir()
    script = backend_root / "backend.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if request["operation"] == "calibrate":
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "feasible": True,
        "representative_steps": 1,
        "step_time_seconds": 0.01,
        "tokens_per_second": 1000.0,
        "peak_vram_bytes": None,
        "peak_ram_bytes": 1024,
        "projected_storage_bytes": 256,
        "projected_wall_seconds": 0.1,
        "gpu_count": 0
    }))
elif request["operation"] == "train":
    output = Path(request["output_root"]) / "model"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps({"model_type": "fixture-distilled"}),
        encoding="utf-8",
    )
    (output / "model.safetensors").write_bytes(b"distilled-student-weights")
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "output_model_path": str(output),
        "metrics": {
            "method": "knowledge_distillation",
            "student_preset": "fixture-small"
        }
    }))
else:
    raise SystemExit(2)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec = backend_root / "backend.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": "fixture-distill-backend",
                "supported_paths": ["DISTILL"],
                "calibrate_argv": [sys.executable, str(script), "{request_json}"],
                "train_argv": [sys.executable, str(script), "{request_json}"],
                "environment": {},
            }
        ),
        encoding="utf-8",
    )

    registry = Registry(project)
    teacher_id = registry.read().project["champion_id"]
    assert isinstance(teacher_id, str)

    plan = create_training_plan(
        project,
        path_id=TrainingPathId.DISTILL,
        backend_spec_path=spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=1, max_storage_bytes=4096),
        config={
            "student_preset": "fixture-small",
            "distill_temperature": 2.0,
            "distill_alpha": 0.7,
        },
    )
    assert plan.intervention_id == "frontierwright.evolve.distill"
    calibrate_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=spec,
    )
    completed = execute_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=spec,
        dry_run=False,
        rerun=False,
    )
    assert completed.status == "COMPLETED"
    assert completed.candidate_model_id is not None

    lineage = registry.get_model_lineage(completed.candidate_model_id)
    distilled = [
        item for item in lineage if item["relation"] == "DISTILLED_FROM"
    ]
    assert len(distilled) == 1
    assert distilled[0]["parent_model_id"] == teacher_id
    assert distilled[0]["details"]["intervention_id"] == (
        "frontierwright.evolve.distill"
    )
    assert distilled[0]["details"]["run_id"] == completed.run_id


def test_sealed_candidate_manifest_binds_preparation_recipe(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    raw = get_data_view(project).datasets[0]
    prepared_view = prepare_dataset_snapshot(
        project,
        dataset_id=str(raw["dataset_id"]),
        name="Prepared SFT",
    )
    prepared = next(item for item in prepared_view.datasets if item["managed"] is True)

    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=2, max_storage_bytes=4096),
        config={"epochs": 1},
    )
    calibrate_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=backend_spec,
    )
    completed = execute_training_plan(
        project,
        plan_id=plan.plan_id or "",
        backend_spec_path=backend_spec,
        dry_run=False,
        rerun=False,
    )

    assert completed.candidate_model_id is not None
    record = Registry(project).get_sealed_artifact(completed.candidate_model_id)
    assert record is not None
    manifest = json.loads(
        Path(str(record["manifest_path"])).read_text(encoding="utf-8")
    )
    assert manifest["intervention_id"] == "frontierwright.specialize.lora-sft"
    assert manifest["intervention_version"] == "1"
    assert manifest["intervention_family"] == "SPECIALIZE"
    assert manifest["dataset_id"] == prepared["dataset_id"]
    assert manifest["dataset_fingerprint"] == prepared["fingerprint"]
    assert manifest["dataset_recipe_id"] == prepared["preparation_recipe_id"]
    assert manifest["dataset_recipe_hash"] == prepared["preparation_recipe_hash"]


def test_calibration_replays_by_default_and_rerun_is_explicit(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=1, max_storage_bytes=4096),
        config={"epochs": 1},
    )
    plan_id = plan.plan_id or ""

    first = calibrate_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
    )
    first_id = first.calibration["calibration_id"] if first.calibration else None
    assert isinstance(first_id, str)
    assert len(Registry(project).read().calibrations) == 1

    replay = calibrate_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
    )
    assert replay.calibration is not None
    assert replay.calibration["calibration_id"] == first_id
    assert len(Registry(project).read().calibrations) == 1

    rerun = calibrate_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
        rerun=True,
    )
    assert rerun.calibration is not None
    assert rerun.calibration["calibration_id"] != first_id
    assert len(Registry(project).read().calibrations) == 2


def test_calibration_preflight_has_no_backend_or_registry_side_effects(tmp_path: Path) -> None:
    project, backend_spec = setup_project(tmp_path)
    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend_spec,
        dataset_id=None,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=1, max_storage_bytes=4096),
        config={"epochs": 1},
    )
    plan_id = plan.plan_id or ""

    preview = preflight_training_calibration(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
    )
    assert preview.action == "calibrate"
    assert preview.ready is True
    assert preview.would_replay is False
    assert preview.details["plan_id"] == plan_id
    assert len(Registry(project).read().calibrations) == 0

    calibrated = calibrate_training_plan(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
    )
    assert calibrated.calibration is not None

    replay_preview = preflight_training_calibration(
        project,
        plan_id=plan_id,
        backend_spec_path=backend_spec,
    )
    assert replay_preview.ready is True
    assert replay_preview.would_replay is True
    assert replay_preview.details["existing_calibration_id"] == calibrated.calibration[
        "calibration_id"
    ]
    assert len(Registry(project).read().calibrations) == 1
