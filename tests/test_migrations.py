import json
import sqlite3
from dataclasses import asdict
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
        connection.execute("DROP TABLE tokenizer_artifacts")
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

    assert version == SCHEMA_VERSION == 23
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
    assert "model_births" in tables
    assert "data_recipes" in tables
    assert "lab_adapters" in tables
    assert "tokenizer_artifacts" in tables
    assert "workload_profiles" in tables
    assert state.project["edition_profile"] == "ACADEMY"


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
    assert version == SCHEMA_VERSION == 23


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
        intervention_id="frontierwright.specialize.lora-sft",
        intervention_version="1",
        intervention_family="SPECIALIZE",
        backend_id="backend-v1",
        backend_spec_hash="sha256:backend-v1",
        model_id=None,
        model_fingerprint=None,
        model_source_path=None,
        dataset_id="dataset-v7",
        dataset_fingerprint="sha256:v7-data",
        dataset_source_path="/tmp/v7-data",
        dataset_recipe_id=None,
        dataset_recipe_hash=None,
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


def test_v9_project_migrates_to_origin_appropriate_edition_profile(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path)
    registry.initialize("Legacy Zero", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute("ALTER TABLE project RENAME TO project_v10")
        connection.execute(
            """
            CREATE TABLE project (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                project_id TEXT NOT NULL UNIQUE,
                identity_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                language TEXT NOT NULL CHECK (language IN ('en','ko')),
                origin TEXT NOT NULL CHECK (
                    origin IN ('ZERO','IMPORTED_LOCAL','INTERNAL_LAB')),
                history_confidence TEXT NOT NULL CHECK (
                    history_confidence IN ('COMPLETE','VERIFIED','PARTIAL','UNKNOWN')),
                created_at TEXT NOT NULL,
                champion_id TEXT REFERENCES models(model_id)
            )
            """
        )
        connection.execute(
            "INSERT INTO project (singleton, project_id, identity_id, name, language, "
            "origin, history_confidence, created_at, champion_id) "
            "SELECT singleton, project_id, identity_id, name, language, origin, "
            "history_confidence, created_at, champion_id FROM project_v10"
        )
        connection.execute("DROP TABLE project_v10")
        connection.execute("PRAGMA user_version = 9")
        connection.commit()

    state = registry.read()

    assert state.project["edition_profile"] == "ACADEMY"
    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION == 23


def test_v11_migrates_data_recipe_schema(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute("DROP TABLE data_recipes")
        connection.execute("PRAGMA user_version = 11")
        connection.commit()

    registry.read()

    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        dataset_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(datasets)")
        }
        plan_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(plans)")
        }

    assert version == SCHEMA_VERSION == 23
    assert "data_recipes" in tables
    assert {
        "source_dataset_id",
        "preparation_recipe_id",
        "preparation_recipe_hash",
    }.issubset(dataset_columns)
    assert {"dataset_recipe_id", "dataset_recipe_hash"}.issubset(plan_columns)


def test_v12_backfills_intervention_identity_for_existing_plan(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute(
            "INSERT INTO datasets ("
            "dataset_id, name, role, provenance, source_path, fingerprint, "
            "total_bytes, file_count, manifest_json, license, domain, language, "
            "token_count, created_at, active, source_dataset_id, "
            "preparation_recipe_id, preparation_recipe_hash"
            ") VALUES (?, ?, 'SFT', 'LOCAL_USER', ?, ?, 4, 1, '[]', NULL, NULL, "
            "NULL, NULL, ?, 1, NULL, NULL, NULL)",
            (
                "dataset-v12",
                "fixture",
                "/tmp/v12-data",
                "sha256:v12-data",
                "2026-09-23T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO plans ("
            "plan_id, idempotency_key, path_id, intervention_id, "
            "intervention_version, intervention_family, backend_id, "
            "backend_spec_hash, model_id, model_fingerprint, model_source_path, "
            "dataset_id, dataset_fingerprint, dataset_source_path, "
            "dataset_recipe_id, dataset_recipe_hash, resource_profile_id, "
            "permission, budgets_json, config_json, created_at"
            ") VALUES (?, ?, 'LORA_SFT', NULL, NULL, NULL, ?, ?, NULL, NULL, NULL, "
            "?, ?, ?, NULL, NULL, NULL, 'PLAN', '{}', '{}', ?)",
            (
                "plan-v12",
                "sha256:plan-v12",
                "backend-v12",
                "sha256:backend-v12",
                "dataset-v12",
                "sha256:v12-data",
                "/tmp/v12-data",
                "2026-09-23T00:00:00+00:00",
            ),
        )
        connection.execute("PRAGMA user_version = 12")
        connection.commit()

    plan = registry.get_plan("plan-v12")

    assert plan.intervention_id == "frontierwright.specialize.lora-sft"
    assert plan.intervention_version == "1"
    assert plan.intervention_family == "SPECIALIZE"
    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION == 23


def test_v13_binds_numeric_build_schema_without_inventing_scale(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute("DROP TABLE build_state")
        connection.execute(
            """
            CREATE TABLE build_state (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                mode TEXT NOT NULL CHECK (mode IN ('INTENT','TARGETS_FLOORS')),
                archetype TEXT,
                priorities_json TEXT,
                targets_json TEXT,
                floors_json TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO build_state VALUES "
            "(1, 'TARGETS_FLOORS', NULL, NULL, ?, ?, ?)",
            (
                '{"coding": 140}',
                '{"general": 100}',
                "2026-09-23T00:00:00+00:00",
            ),
        )
        connection.execute("PRAGMA user_version = 13")
        connection.commit()

    state = registry.read()

    assert state.build_state is not None
    assert state.build_state["targets"] == {"coding": 140}
    assert state.build_state["floors"] == {"general": 100}
    assert state.build_state["scale_hash"] is None
    assert state.build_state["scale_id"] is None
    assert state.build_state["scale_version"] is None

    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(build_state)")
        }

    assert version == SCHEMA_VERSION == 23
    assert {"scale_hash", "scale_id", "scale_version"}.issubset(columns)


def test_v14_backfills_dataset_classification_and_plan_boundary(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("Lab", ModelOrigin.INTERNAL_LAB)

    with sqlite3.connect(registry.path) as connection:
        connection.execute(
            "INSERT INTO datasets ("
            "dataset_id, name, role, provenance, classification, source_path, fingerprint, "
            "total_bytes, file_count, manifest_json, created_at, active"
            ") VALUES (?, ?, 'SFT', 'PUBLIC_DISCOVERED', 'PRIVATE', ?, ?, 1, 1, '[]', ?, 1)",
            (
                "dataset-public",
                "Public",
                "/tmp/public",
                "sha256:public",
                "2026-09-23T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO datasets ("
            "dataset_id, name, role, provenance, classification, source_path, fingerprint, "
            "total_bytes, file_count, manifest_json, created_at, active"
            ") VALUES (?, ?, 'SFT', 'INTERNAL_CONNECTED', 'PRIVATE', ?, ?, 1, 1, '[]', ?, 1)",
            (
                "dataset-internal",
                "Internal",
                "/tmp/internal",
                "sha256:internal",
                "2026-09-23T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO plans ("
            "plan_id, idempotency_key, path_id, intervention_id, intervention_version, "
            "intervention_family, backend_id, backend_spec_hash, dataset_id, "
            "dataset_fingerprint, dataset_source_path, permission, budgets_json, "
            "config_json, created_at"
            ") VALUES (?, ?, 'LORA_SFT', 'frontierwright.specialize.lora-sft', '1', "
            "'SPECIALIZE', ?, ?, ?, ?, ?, 'PLAN', '{}', '{}', ?)",
            (
                "plan-v14",
                "sha256:plan-v14",
                "legacy-local",
                "sha256:legacy-local",
                "dataset-public",
                "sha256:public",
                "/tmp/public",
                "2026-09-23T00:00:00+00:00",
            ),
        )
        connection.execute("UPDATE plans SET dataset_classification = 'PRIVATE'")
        connection.execute("PRAGMA user_version = 14")
        connection.commit()

    registry.read()
    plan = registry.get_plan("plan-v14")

    with sqlite3.connect(registry.path) as connection:
        rows = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT dataset_id, classification FROM datasets "
                "WHERE dataset_id IN ('dataset-public','dataset-internal')"
            )
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert version == SCHEMA_VERSION == 23
    assert rows == {
        "dataset-public": "PUBLIC",
        "dataset-internal": "INTERNAL",
    }
    assert plan.dataset_classification == "PUBLIC"
    assert plan.backend_data_boundary == "LOCAL_MACHINE"


def test_v16_adds_durable_run_usage_ledger(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute("ALTER TABLE runs DROP COLUMN usage_json")
        connection.execute("PRAGMA user_version = 16")
        connection.commit()

    registry.read()

    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(runs)")
        }

    assert version == SCHEMA_VERSION == 23
    assert "usage_json" in columns


def test_v17_migration_preserves_datasets_and_allows_preference_role(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute(
            "INSERT INTO datasets ("
            "dataset_id, name, role, provenance, classification, source_path, "
            "fingerprint, total_bytes, file_count, manifest_json, created_at, active"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "dataset-existing",
                "Existing SFT",
                "SFT",
                "LOCAL_USER",
                "PRIVATE",
                str(tmp_path / "existing"),
                "sha256:existing",
                12,
                1,
                "[]",
                "2026-09-23T00:00:00Z",
                1,
            ),
        )
        connection.execute("PRAGMA user_version = 17")
        connection.commit()

    state = registry.read()
    assert any(item["dataset_id"] == "dataset-existing" for item in state.datasets)

    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'datasets'"
        ).fetchone()[0]
        self_fk = connection.execute("PRAGMA foreign_key_list(datasets)").fetchall()
        connection.execute(
            "INSERT INTO datasets ("
            "dataset_id, name, role, provenance, classification, source_path, "
            "fingerprint, total_bytes, file_count, manifest_json, created_at, active"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "dataset-preference",
                "Preference pairs",
                "PREFERENCE",
                "LOCAL_USER",
                "PRIVATE",
                str(tmp_path / "preference"),
                "sha256:preference",
                24,
                1,
                "[]",
                "2026-09-23T00:00:01Z",
                1,
            ),
        )
        connection.commit()

    assert version == SCHEMA_VERSION == 23
    assert "PREFERENCE" in schema
    assert any(row[2] == "datasets" for row in self_fk)


def test_v18_migration_backfills_model_lineage_edges(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)
    state = registry.read()

    from frontierwright.domain import ModelState

    parent = ModelState(
        model_id="lineage-parent",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint="parent",
        fingerprint="sha256:parent",
    )
    child = ModelState(
        model_id="lineage-child",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint="child",
        fingerprint="sha256:child",
        parent_model_id=parent.model_id,
    )

    with sqlite3.connect(registry.path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO models(model_id,parent_model_id,snapshot) VALUES (?,?,?)",
            (parent.model_id, None, json.dumps(asdict(parent), sort_keys=True)),
        )
        connection.execute(
            "INSERT INTO models(model_id,parent_model_id,snapshot) VALUES (?,?,?)",
            (
                child.model_id,
                parent.model_id,
                json.dumps(asdict(child), sort_keys=True),
            ),
        )
        connection.execute("DROP TABLE model_lineage_edges")
        connection.execute("PRAGMA user_version = 18")
        connection.commit()

    lineage = registry.get_model_lineage(child.model_id)
    assert lineage == (
        {
            "child_model_id": child.model_id,
            "parent_model_id": parent.model_id,
            "relation": "DERIVED_FROM",
            "ordinal": 0,
            "created_at": lineage[0]["created_at"],
            "details": {"source": "legacy_parent_model_id"},
        },
    )

    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION == 23


def test_v19_adds_tokenizer_artifact_registry(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    with sqlite3.connect(registry.path) as connection:
        connection.execute("DROP TABLE tokenizer_artifacts")
        connection.execute("PRAGMA user_version = 19")
        connection.commit()

    registry.read()

    with sqlite3.connect(registry.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(tokenizer_artifacts)")
        }

    assert version == SCHEMA_VERSION == 23
    assert "tokenizer_artifacts" in tables
    assert {
        "artifact_id",
        "fingerprint",
        "source_dataset_id",
        "source_dataset_fingerprint",
        "requested_vocab_size",
        "vocab_size",
        "max_training_bytes",
        "training_bytes",
        "path",
        "created_at",
    }.issubset(columns)
