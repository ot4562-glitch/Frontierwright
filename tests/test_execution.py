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
    get_paths_view,
    get_run_view,
    import_local_model,
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
            "worker_pid": 999_999_998,
        },
    )

    first = reconcile_training_run(project, run_id)
    second = reconcile_training_run(project, run_id)

    state = Registry(project).read()
    assert first.status == "COMPLETED"
    assert second.run_id == first.run_id
    assert first.candidate_model_id is not None
    assert len(state.candidates) == 1
    assert state.project["champion_id"] != first.candidate_model_id


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
