import sqlite3
from pathlib import Path

import pytest

from frontierwright.domain import ModelOrigin
from frontierwright.execution import HardBudgets, PermissionLevel, TrainingPlan
from frontierwright.paths import TrainingPathId
from frontierwright.registry import SCHEMA_VERSION, Registry


@pytest.mark.parametrize("old_version", [1, 2, 3, 4, 5])
def test_previous_registry_versions_migrate_to_current(
    tmp_path: Path,
    old_version: int,
) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute("DROP TABLE sealed_artifacts")
        connection.execute("DROP TABLE runs")
        connection.execute("DROP TABLE calibrations")
        connection.execute("DROP TABLE plans")
        if old_version <= 4:
            connection.execute("DROP TABLE datasets")
        if old_version <= 3:
            connection.execute("DROP TABLE capability_profiles")
            connection.execute("DROP TABLE capability_scales")
            connection.execute("DROP TABLE evaluation_receipts")
        if old_version <= 2:
            connection.execute("DROP TABLE build_state")
        if old_version == 1:
            connection.execute("DROP TABLE resource_profiles")
            connection.execute("DROP TABLE model_artifacts")
        connection.execute(f"PRAGMA user_version = {old_version}")
        connection.commit()

    state = registry.read()
    if old_version <= 2:
        assert state.build_state is None
    if old_version <= 3:
        assert state.capability_profile is None
    if old_version <= 4:
        assert state.datasets == ()
    assert state.plans == ()
    assert state.calibrations == ()
    assert state.runs == ()

    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert version == SCHEMA_VERSION == 9
    assert "model_artifacts" in tables
    assert "resource_profiles" in tables
    assert "build_state" in tables
    assert "evaluation_receipts" in tables
    assert "capability_scales" in tables
    assert "capability_profiles" in tables
    assert "datasets" in tables
    assert "plans" in tables
    assert "calibrations" in tables
    assert "runs" in tables
    assert "run_attempts" in tables
    assert "sealed_artifacts" in tables


def test_v7_running_attempt_migrates_to_incomplete(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute("DROP TABLE sealed_artifacts")
        connection.execute("DROP INDEX IF EXISTS one_running_run_per_plan")
        connection.execute("DROP TABLE run_attempts")
        connection.execute(
            "INSERT INTO plans ("
            "plan_id, idempotency_key, path_id, backend_id, backend_spec_hash, "
            "model_id, model_fingerprint, model_source_path, dataset_id, "
            "dataset_fingerprint, dataset_source_path, resource_profile_id, "
            "permission, budgets_json, config_json, created_at"
            ") VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, NULL, ?, ?, ?, ?)",
            (
                "plan-legacy",
                "sha256:legacy-plan",
                "LORA_SFT",
                "legacy-backend",
                "sha256:backend",
                "dataset-legacy",
                "sha256:data",
                "/tmp/data",
                "EXECUTE_SINGLE",
                "{}",
                "{}",
                "2026-09-23T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO runs ("
            "run_id, plan_id, status, started_at, finished_at, candidate_model_id, "
            "metrics_json, error_code, error_message"
            ") VALUES (?, ?, 'RUNNING', ?, NULL, NULL, '{}', NULL, NULL)",
            ("run-legacy", "plan-legacy", "2026-09-23T00:00:01+00:00"),
        )
        connection.execute("PRAGMA user_version = 7")
        connection.commit()

    run = registry.get_run("run-legacy")
    assert run["status"] == "INCOMPLETE"
    assert run["error_code"] == "MIGRATED_RUN_UNRECOVERABLE"
    assert run["liveness_state"] is None

    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION == 9


def test_v7_migrated_plan_column_order_accepts_new_plan(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute("DROP TABLE sealed_artifacts")
        connection.execute("DROP INDEX IF EXISTS one_running_run_per_plan")
        connection.execute("DROP TABLE run_attempts")
        connection.execute("DROP TABLE runs")
        connection.execute("DROP TABLE calibrations")
        connection.execute("DROP TABLE plans")
        connection.execute(
            """
            CREATE TABLE plans (
                plan_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                path_id TEXT NOT NULL,
                backend_id TEXT NOT NULL,
                backend_spec_hash TEXT NOT NULL,
                model_id TEXT REFERENCES models(model_id),
                model_fingerprint TEXT,
                dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
                dataset_fingerprint TEXT NOT NULL,
                resource_profile_id TEXT,
                permission TEXT NOT NULL,
                budgets_json TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model_source_path TEXT,
                dataset_source_path TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE calibrations (
                calibration_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL REFERENCES plans(plan_id),
                backend_id TEXT NOT NULL,
                feasible INTEGER NOT NULL CHECK (feasible IN (0,1)),
                representative_steps INTEGER NOT NULL,
                step_time_seconds REAL NOT NULL,
                tokens_per_second REAL,
                peak_vram_bytes INTEGER,
                peak_ram_bytes INTEGER,
                projected_storage_bytes INTEGER,
                projected_wall_seconds REAL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL REFERENCES plans(plan_id),
                status TEXT NOT NULL CHECK (
                    status IN ('RUNNING','COMPLETED','FAILED','INCOMPLETE')),
                started_at TEXT NOT NULL,
                finished_at TEXT,
                candidate_model_id TEXT REFERENCES models(model_id),
                metrics_json TEXT NOT NULL,
                error_code TEXT,
                error_message TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO datasets ("
            "dataset_id, name, role, provenance, source_path, fingerprint, "
            "total_bytes, file_count, manifest_json, license, domain, language, "
            "token_count, created_at, active"
            ") VALUES (?, ?, 'SFT', 'LOCAL_USER', ?, ?, 4, 1, '[]', NULL, NULL, "
            "NULL, NULL, ?, 1)",
            (
                "dataset-v7",
                "fixture",
                "/tmp/v7-data",
                "sha256:v7-data",
                "2026-09-23T00:00:00+00:00",
            ),
        )
        connection.execute("PRAGMA user_version = 7")
        connection.commit()

    registry.read()

    plan = TrainingPlan(
        plan_id="plan-after-migration",
        path_id=TrainingPathId.LORA_SFT,
        backend_id="backend-v1",
        backend_spec_hash="sha256:backend-v1",
        model_id=None,
        model_fingerprint=None,
        model_source_path=None,
        dataset_id="dataset-v7",
        dataset_fingerprint="sha256:v7-data",
        dataset_source_path="/tmp/v7-data",
        resource_profile_id=None,
        permission=PermissionLevel.PLAN,
        budgets=HardBudgets(max_runs=1),
        config={"epochs": 1},
        idempotency_key="sha256:plan-after-migration",
    )
    stored = registry.store_plan(plan)

    assert stored.plan_id == plan.plan_id
    assert stored.model_source_path is None
    assert stored.dataset_id == "dataset-v7"
    assert stored.dataset_source_path == "/tmp/v7-data"
    assert stored.resource_profile_id is None
    assert stored.permission is PermissionLevel.PLAN
    assert stored.budgets.max_runs == 1
    assert stored.config == {"epochs": 1}
