"""Versioned SQLite registry for Frontierwright project state."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from frontierwright.artifact_store import SealedArtifact
from frontierwright.data import (
    DatasetClassification,
    DatasetDescriptor,
    DatasetProvenance,
    DatasetRole,
    default_dataset_classification,
)
from frontierwright.domain import (
    Axis,
    BuildIntent,
    BuildMode,
    BuildTargets,
    Candidate,
    CandidateStatus,
    CapabilityStat,
    Champion,
    HistoryConfidence,
    ModelFormat,
    ModelOrigin,
    ModelState,
    build_mode,
    require_text,
)
from frontierwright.editions import EditionProfile, default_edition_for_origin
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import CapabilityScale, EvaluationReceipt
from frontierwright.execution import (
    CalibrationReceipt,
    HardBudgets,
    PermissionLevel,
    RunStatus,
    TrainingPlan,
)
from frontierwright.models import HistoryEvidenceResult, ImportedModelDescriptor
from frontierwright.paths import TrainingPathId
from frontierwright.recipes import DataPreparationRecipe
from frontierwright.resources import ResourceSnapshot

SCHEMA_VERSION = 15
STATE_DIR = ".frontierwright"
REGISTRY_NAME = "registry.sqlite"

SCHEMA = """
CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    parent_model_id TEXT REFERENCES models(model_id),
    snapshot TEXT NOT NULL
);
CREATE TABLE project (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    project_id TEXT NOT NULL UNIQUE,
    identity_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('en','ko')),
    origin TEXT NOT NULL CHECK (origin IN ('ZERO','IMPORTED_LOCAL','INTERNAL_LAB')),
    history_confidence TEXT NOT NULL CHECK (
        history_confidence IN ('COMPLETE','VERIFIED','PARTIAL','UNKNOWN')),
    edition_profile TEXT NOT NULL CHECK (
        edition_profile IN ('STUDIO','ACADEMY','LAB')),
    created_at TEXT NOT NULL,
    champion_id TEXT REFERENCES models(model_id)
);
CREATE TABLE candidates (
    model_id TEXT PRIMARY KEY REFERENCES models(model_id),
    status TEXT NOT NULL CHECK (status IN ('PENDING','ACCEPTED','REJECTED','INCOMPLETE'))
);
CREATE TABLE events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    details TEXT NOT NULL
);
CREATE TABLE model_artifacts (
    model_id TEXT PRIMARY KEY REFERENCES models(model_id),
    source_path TEXT NOT NULL,
    model_format TEXT NOT NULL,
    trainable INTEGER NOT NULL CHECK (trainable IN (0,1)),
    total_bytes INTEGER NOT NULL,
    files_json TEXT NOT NULL,
    evidence_confidence TEXT NOT NULL,
    evidence_files_json TEXT NOT NULL,
    evidence_reason TEXT NOT NULL
);
CREATE TABLE resource_profiles (
    profile_id TEXT PRIMARY KEY,
    profile_name TEXT NOT NULL,
    provenance TEXT NOT NULL,
    created_at TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0,1))
);
CREATE TABLE build_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    mode TEXT NOT NULL CHECK (mode IN ('INTENT','TARGETS_FLOORS')),
    archetype TEXT,
    priorities_json TEXT,
    targets_json TEXT,
    floors_json TEXT,
    scale_hash TEXT,
    scale_id TEXT,
    scale_version TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE evaluation_receipts (
    receipt_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    model_fingerprint TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL UNIQUE,
    conditions_json TEXT NOT NULL,
    measurements_json TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE capability_scales (
    scale_hash TEXT PRIMARY KEY,
    scale_id TEXT NOT NULL,
    scale_version TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE capability_profiles (
    profile_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    receipt_id TEXT NOT NULL REFERENCES evaluation_receipts(receipt_id),
    scale_hash TEXT NOT NULL REFERENCES capability_scales(scale_hash),
    stats_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0,1))
);
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('PRETRAIN','SFT')),
    provenance TEXT NOT NULL CHECK (
        provenance IN ('LOCAL_USER','INTERNAL_CONNECTED','PUBLIC_DISCOVERED')),
    classification TEXT NOT NULL DEFAULT 'PRIVATE' CHECK (
        classification IN ('PUBLIC','INTERNAL','CONFIDENTIAL','PRIVATE')),
    source_path TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    total_bytes INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    manifest_json TEXT NOT NULL,
    license TEXT,
    domain TEXT,
    language TEXT,
    token_count INTEGER,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0,1)),
    source_dataset_id TEXT REFERENCES datasets(dataset_id),
    preparation_recipe_id TEXT,
    preparation_recipe_hash TEXT
);
CREATE TABLE data_recipes (
    recipe_id TEXT PRIMARY KEY,
    recipe_hash TEXT NOT NULL UNIQUE,
    plugin_id TEXT NOT NULL,
    plugin_version TEXT NOT NULL,
    source_dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    source_fingerprint TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    path_id TEXT NOT NULL,
    intervention_id TEXT,
    intervention_version TEXT,
    intervention_family TEXT,
    backend_id TEXT NOT NULL,
    backend_spec_hash TEXT NOT NULL,
    model_id TEXT REFERENCES models(model_id),
    model_fingerprint TEXT,
    model_source_path TEXT,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    dataset_fingerprint TEXT NOT NULL,
    dataset_source_path TEXT NOT NULL,
    dataset_recipe_id TEXT,
    dataset_recipe_hash TEXT,
    dataset_classification TEXT NOT NULL DEFAULT 'PRIVATE',
    backend_data_boundary TEXT NOT NULL DEFAULT 'LOCAL_MACHINE',
    resource_profile_id TEXT,
    permission TEXT NOT NULL,
    budgets_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
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
);
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES plans(plan_id),
    status TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETED','FAILED','INCOMPLETE')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    candidate_model_id TEXT REFERENCES models(model_id),
    metrics_json TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS run_attempts (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    attempt_token TEXT NOT NULL UNIQUE,
    calibration_id TEXT NOT NULL REFERENCES calibrations(calibration_id),
    request_digest TEXT NOT NULL,
    executor_kind TEXT NOT NULL,
    owner_pid INTEGER,
    worker_pid INTEGER,
    worker_start_token TEXT,
    liveness_state TEXT NOT NULL CHECK (
        liveness_state IN ('RESERVED','LIVE','TERMINAL','UNRESOLVED')),
    result_path TEXT,
    result_sha256 TEXT,
    result_json TEXT,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_run_per_plan
ON runs(plan_id) WHERE status = 'RUNNING';

CREATE TABLE sealed_artifacts (
    model_id TEXT PRIMARY KEY REFERENCES models(model_id),
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
    artifact_root TEXT NOT NULL,
    model_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE model_births (
    model_id TEXT PRIMARY KEY REFERENCES models(model_id),
    preset TEXT NOT NULL,
    seed INTEGER NOT NULL CHECK (seed >= 0),
    backend_id TEXT NOT NULL,
    runtime_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

MIGRATION_1_TO_2 = """
CREATE TABLE model_artifacts (
    model_id TEXT PRIMARY KEY REFERENCES models(model_id),
    source_path TEXT NOT NULL,
    model_format TEXT NOT NULL,
    trainable INTEGER NOT NULL CHECK (trainable IN (0,1)),
    total_bytes INTEGER NOT NULL,
    files_json TEXT NOT NULL,
    evidence_confidence TEXT NOT NULL,
    evidence_files_json TEXT NOT NULL,
    evidence_reason TEXT NOT NULL
);
CREATE TABLE resource_profiles (
    profile_id TEXT PRIMARY KEY,
    profile_name TEXT NOT NULL,
    provenance TEXT NOT NULL,
    created_at TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0,1))
);
"""

MIGRATION_2_TO_3 = """
CREATE TABLE build_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    mode TEXT NOT NULL CHECK (mode IN ('INTENT','TARGETS_FLOORS')),
    archetype TEXT,
    priorities_json TEXT,
    targets_json TEXT,
    floors_json TEXT,
    updated_at TEXT NOT NULL
);
"""

MIGRATION_3_TO_4 = """
CREATE TABLE evaluation_receipts (
    receipt_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    model_fingerprint TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL UNIQUE,
    conditions_json TEXT NOT NULL,
    measurements_json TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE capability_scales (
    scale_hash TEXT PRIMARY KEY,
    scale_id TEXT NOT NULL,
    scale_version TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE capability_profiles (
    profile_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    receipt_id TEXT NOT NULL REFERENCES evaluation_receipts(receipt_id),
    scale_hash TEXT NOT NULL REFERENCES capability_scales(scale_hash),
    stats_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0,1))
);
"""

MIGRATION_4_TO_5 = """
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('PRETRAIN','SFT')),
    provenance TEXT NOT NULL CHECK (
        provenance IN ('LOCAL_USER','INTERNAL_CONNECTED','PUBLIC_DISCOVERED')),
    source_path TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    total_bytes INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    manifest_json TEXT NOT NULL,
    license TEXT,
    domain TEXT,
    language TEXT,
    token_count INTEGER,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0,1))
);
"""

MIGRATION_5_TO_6 = """
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
    created_at TEXT NOT NULL
);
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
);
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES plans(plan_id),
    status TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETED','FAILED','INCOMPLETE')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    candidate_model_id TEXT REFERENCES models(model_id),
    metrics_json TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT
);
"""

MIGRATION_6_TO_7 = """
ALTER TABLE plans ADD COLUMN model_source_path TEXT;
ALTER TABLE plans ADD COLUMN dataset_source_path TEXT;
UPDATE plans
SET dataset_source_path = (
    SELECT datasets.source_path
    FROM datasets
    WHERE datasets.dataset_id = plans.dataset_id
)
WHERE dataset_source_path IS NULL;
UPDATE plans
SET model_source_path = (
    SELECT model_artifacts.source_path
    FROM model_artifacts
    WHERE model_artifacts.model_id = plans.model_id
)
WHERE model_id IS NOT NULL AND model_source_path IS NULL;
"""

MIGRATION_7_TO_8 = """
UPDATE runs
SET status = 'INCOMPLETE',
    finished_at = COALESCE(finished_at, CURRENT_TIMESTAMP),
    error_code = COALESCE(error_code, 'MIGRATED_RUN_UNRECOVERABLE'),
    error_message = COALESCE(
        error_message,
        'Legacy RUNNING attempt lacked durable executor identity and requires explicit rerun.'
    )
WHERE status = 'RUNNING';

CREATE TABLE IF NOT EXISTS run_attempts (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    attempt_token TEXT NOT NULL UNIQUE,
    calibration_id TEXT NOT NULL REFERENCES calibrations(calibration_id),
    request_digest TEXT NOT NULL,
    executor_kind TEXT NOT NULL,
    owner_pid INTEGER,
    worker_pid INTEGER,
    worker_start_token TEXT,
    liveness_state TEXT NOT NULL CHECK (
        liveness_state IN ('RESERVED','LIVE','TERMINAL','UNRESOLVED')),
    result_path TEXT,
    result_sha256 TEXT,
    result_json TEXT,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_run_per_plan
ON runs(plan_id) WHERE status = 'RUNNING';
"""

MIGRATION_8_TO_9 = """
CREATE TABLE sealed_artifacts (
    model_id TEXT PRIMARY KEY REFERENCES models(model_id),
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
    artifact_root TEXT NOT NULL,
    model_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

MIGRATION_9_TO_10 = """
ALTER TABLE project ADD COLUMN edition_profile TEXT NOT NULL DEFAULT 'STUDIO'
CHECK (edition_profile IN ('STUDIO','ACADEMY','LAB'));
UPDATE project
SET edition_profile = CASE origin
    WHEN 'ZERO' THEN 'ACADEMY'
    WHEN 'INTERNAL_LAB' THEN 'LAB'
    ELSE 'STUDIO'
END;
"""

MIGRATION_10_TO_11 = """
CREATE TABLE IF NOT EXISTS model_births (
    model_id TEXT PRIMARY KEY REFERENCES models(model_id),
    preset TEXT NOT NULL,
    seed INTEGER NOT NULL CHECK (seed >= 0),
    backend_id TEXT NOT NULL,
    runtime_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def decode_model(snapshot: str) -> ModelState:
    raw = json.loads(snapshot)
    raw["origin"] = ModelOrigin(raw["origin"])
    raw["model_format"] = ModelFormat(raw.get("model_format", ModelFormat.CUSTOM.value))
    raw["trainable"] = bool(raw.get("trainable", True))
    raw["stats"] = tuple(
        CapabilityStat(**{**stat, "axis": Axis(stat["axis"])}) for stat in raw["stats"]
    )
    return ModelState(**raw)


def decode_plan(row: sqlite3.Row) -> TrainingPlan:
    budgets_raw = json.loads(row["budgets_json"])
    config_raw = json.loads(row["config_json"])
    return TrainingPlan(
        plan_id=row["plan_id"],
        path_id=TrainingPathId(row["path_id"]),
        intervention_id=row["intervention_id"],
        intervention_version=row["intervention_version"],
        intervention_family=row["intervention_family"],
        backend_id=row["backend_id"],
        backend_spec_hash=row["backend_spec_hash"],
        model_id=row["model_id"],
        model_fingerprint=row["model_fingerprint"],
        model_source_path=row["model_source_path"],
        dataset_id=row["dataset_id"],
        dataset_fingerprint=row["dataset_fingerprint"],
        dataset_source_path=row["dataset_source_path"],
        dataset_recipe_id=row["dataset_recipe_id"],
        dataset_recipe_hash=row["dataset_recipe_hash"],
        dataset_classification=row["dataset_classification"],
        backend_data_boundary=row["backend_data_boundary"],
        resource_profile_id=row["resource_profile_id"],
        permission=PermissionLevel[row["permission"]],
        budgets=HardBudgets(**budgets_raw),
        config=config_raw,
        idempotency_key=row["idempotency_key"],
    )


def default_history_confidence(origin: ModelOrigin) -> HistoryConfidence:
    if origin is ModelOrigin.ZERO:
        return HistoryConfidence.COMPLETE
    return HistoryConfidence.UNKNOWN


@dataclass(frozen=True)
class ProjectState:
    project: dict[str, Any]
    champion: Champion | None
    champion_artifact: dict[str, Any] | None
    candidates: tuple[Candidate, ...]
    history: tuple[dict[str, Any], ...]
    resource_profile: dict[str, Any] | None
    build_state: dict[str, Any] | None
    capability_profile: dict[str, Any] | None
    datasets: tuple[dict[str, Any], ...]
    plans: tuple[dict[str, Any], ...]
    calibrations: tuple[dict[str, Any], ...]
    runs: tuple[dict[str, Any], ...]


class Registry:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.state_dir = self.root / STATE_DIR
        self.path = self.state_dir / REGISTRY_NAME

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def _migrate_if_needed(self) -> None:
        if not self.path.is_file():
            return
        try:
            connection = sqlite3.connect(self.path, timeout=5)
            connection.execute("PRAGMA foreign_keys = ON")
            version = connection.execute("PRAGMA user_version").fetchone()[0]

            if version == 1:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_1_TO_2)
                connection.execute("PRAGMA user_version = 2")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 1, "to_version": 2},
                )
                connection.commit()
                version = 2

            if version == 2:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_2_TO_3)
                connection.execute("PRAGMA user_version = 3")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 2, "to_version": 3},
                )
                connection.commit()
                version = 3

            if version == 3:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_3_TO_4)
                connection.execute("PRAGMA user_version = 4")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 3, "to_version": 4},
                )
                connection.commit()
                version = 4

            if version == 4:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_4_TO_5)
                connection.execute("PRAGMA user_version = 5")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 4, "to_version": 5},
                )
                connection.commit()
                version = 5

            if version == 5:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_5_TO_6)
                connection.execute("PRAGMA user_version = 6")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 5, "to_version": 6},
                )
                connection.commit()
                version = 6

            if version == 6:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_6_TO_7)
                connection.execute("PRAGMA user_version = 7")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 6, "to_version": 7},
                )
                connection.commit()
                version = 7

            if version == 7:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_7_TO_8)
                connection.execute("PRAGMA user_version = 8")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 7, "to_version": 8},
                )
                connection.commit()
                version = 8

            if version == 8:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_8_TO_9)
                connection.execute("PRAGMA user_version = 9")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 8, "to_version": 9},
                )
                connection.commit()
                version = 9

            if version == 9:
                project_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(project)")
                }
                if "edition_profile" not in project_columns:
                    connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_9_TO_10)
                else:
                    connection.execute("BEGIN IMMEDIATE")
                connection.execute("PRAGMA user_version = 10")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 9, "to_version": 10},
                )
                connection.commit()
                version = 10

            if version == 10:
                connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_10_TO_11)
                connection.execute("PRAGMA user_version = 11")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 10, "to_version": 11},
                )
                connection.commit()
                version = 11

            if version == 11:
                dataset_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(datasets)")
                }
                plan_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(plans)")
                }
                connection.execute("BEGIN IMMEDIATE")
                if "source_dataset_id" not in dataset_columns:
                    connection.execute(
                        "ALTER TABLE datasets ADD COLUMN source_dataset_id "
                        "TEXT REFERENCES datasets(dataset_id)"
                    )
                if "preparation_recipe_id" not in dataset_columns:
                    connection.execute(
                        "ALTER TABLE datasets ADD COLUMN preparation_recipe_id TEXT"
                    )
                if "preparation_recipe_hash" not in dataset_columns:
                    connection.execute(
                        "ALTER TABLE datasets ADD COLUMN preparation_recipe_hash TEXT"
                    )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS data_recipes ("
                    "recipe_id TEXT PRIMARY KEY, "
                    "recipe_hash TEXT NOT NULL UNIQUE, "
                    "plugin_id TEXT NOT NULL, "
                    "plugin_version TEXT NOT NULL, "
                    "source_dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id), "
                    "source_fingerprint TEXT NOT NULL, "
                    "config_json TEXT NOT NULL, "
                    "created_at TEXT NOT NULL)"
                )
                if "dataset_recipe_id" not in plan_columns:
                    connection.execute(
                        "ALTER TABLE plans ADD COLUMN dataset_recipe_id TEXT"
                    )
                if "dataset_recipe_hash" not in plan_columns:
                    connection.execute(
                        "ALTER TABLE plans ADD COLUMN dataset_recipe_hash TEXT"
                    )
                connection.execute("PRAGMA user_version = 12")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 11, "to_version": 12},
                )
                connection.commit()
                version = 12

            if version == 12:
                plan_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(plans)")
                }
                connection.execute("BEGIN IMMEDIATE")
                if "intervention_id" not in plan_columns:
                    connection.execute(
                        "ALTER TABLE plans ADD COLUMN intervention_id TEXT"
                    )
                if "intervention_version" not in plan_columns:
                    connection.execute(
                        "ALTER TABLE plans ADD COLUMN intervention_version TEXT"
                    )
                if "intervention_family" not in plan_columns:
                    connection.execute(
                        "ALTER TABLE plans ADD COLUMN intervention_family TEXT"
                    )
                connection.execute(
                    "UPDATE plans SET intervention_id = CASE path_id "
                    "WHEN 'FROM_SCRATCH_PRETRAINING' THEN 'frontierwright.learn.pretrain' "
                    "WHEN 'CONTINUED_PRETRAINING' THEN 'frontierwright.learn.continued-pretrain' "
                    "WHEN 'FULL_SFT' THEN 'frontierwright.specialize.full-sft' "
                    "WHEN 'LORA_SFT' THEN 'frontierwright.specialize.lora-sft' "
                    "WHEN 'QLORA_SFT' THEN 'frontierwright.specialize.qlora-sft' "
                    "ELSE intervention_id END "
                    "WHERE intervention_id IS NULL"
                )
                connection.execute(
                    "UPDATE plans SET intervention_family = CASE path_id "
                    "WHEN 'FROM_SCRATCH_PRETRAINING' THEN 'LEARN' "
                    "WHEN 'CONTINUED_PRETRAINING' THEN 'LEARN' "
                    "WHEN 'FULL_SFT' THEN 'SPECIALIZE' "
                    "WHEN 'LORA_SFT' THEN 'SPECIALIZE' "
                    "WHEN 'QLORA_SFT' THEN 'SPECIALIZE' "
                    "ELSE intervention_family END "
                    "WHERE intervention_family IS NULL"
                )
                connection.execute(
                    "UPDATE plans SET intervention_version = '1' "
                    "WHERE intervention_version IS NULL"
                )
                connection.execute("PRAGMA user_version = 13")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 12, "to_version": 13},
                )
                connection.commit()
                version = 13

            if version == 13:
                build_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(build_state)")
                }
                connection.execute("BEGIN IMMEDIATE")
                if "scale_hash" not in build_columns:
                    connection.execute(
                        "ALTER TABLE build_state ADD COLUMN scale_hash TEXT"
                    )
                if "scale_id" not in build_columns:
                    connection.execute(
                        "ALTER TABLE build_state ADD COLUMN scale_id TEXT"
                    )
                if "scale_version" not in build_columns:
                    connection.execute(
                        "ALTER TABLE build_state ADD COLUMN scale_version TEXT"
                    )
                connection.execute("PRAGMA user_version = 14")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 13, "to_version": 14},
                )
                connection.commit()
                version = 14

            if version == 14:
                dataset_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(datasets)")
                }
                plan_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(plans)")
                }
                connection.execute("BEGIN IMMEDIATE")
                if "classification" not in dataset_columns:
                    connection.execute(
                        "ALTER TABLE datasets ADD COLUMN classification TEXT NOT NULL "
                        "DEFAULT 'PRIVATE' CHECK (classification IN "
                        "('PUBLIC','INTERNAL','CONFIDENTIAL','PRIVATE'))"
                    )
                connection.execute(
                    "UPDATE datasets SET classification = CASE provenance "
                    "WHEN 'PUBLIC_DISCOVERED' THEN 'PUBLIC' "
                    "WHEN 'INTERNAL_CONNECTED' THEN 'INTERNAL' "
                    "ELSE COALESCE(classification, 'PRIVATE') END"
                )
                if "dataset_classification" not in plan_columns:
                    connection.execute(
                        "ALTER TABLE plans ADD COLUMN dataset_classification TEXT NOT NULL "
                        "DEFAULT 'PRIVATE'"
                    )
                if "backend_data_boundary" not in plan_columns:
                    connection.execute(
                        "ALTER TABLE plans ADD COLUMN backend_data_boundary TEXT NOT NULL "
                        "DEFAULT 'LOCAL_MACHINE'"
                    )
                connection.execute(
                    "UPDATE plans SET dataset_classification = COALESCE(("
                    "SELECT datasets.classification FROM datasets "
                    "WHERE datasets.dataset_id = plans.dataset_id"
                    "), dataset_classification, 'PRIVATE')"
                )
                connection.execute("PRAGMA user_version = 15")
                self.event(
                    connection,
                    "SCHEMA_MIGRATED",
                    {"from_version": 14, "to_version": 15},
                )
                connection.commit()
                version = 15

            if version != SCHEMA_VERSION:
                raise FrontierwrightError(
                    "UNSUPPORTED_SCHEMA",
                    f"Unsupported registry schema version: {version}",
                    4,
                )
        except sqlite3.Error as exc:
            raise FrontierwrightError(
                "REGISTRY_ERROR",
                "Could not migrate the project registry.",
                4,
            ) from exc
        finally:
            if "connection" in locals():
                connection.close()

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if not self.path.is_file():
            raise FrontierwrightError(
                "NOT_INITIALIZED",
                "Run frontierwright project init first.",
                10,
            )
        self._migrate_if_needed()
        mode = "rw" if write else "ro"
        try:
            connection = sqlite3.connect(
                self.path.as_uri() + f"?mode={mode}",
                uri=True,
                timeout=5,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise FrontierwrightError(
                    "UNSUPPORTED_SCHEMA",
                    f"Unsupported registry schema version: {version}",
                    4,
                )
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise FrontierwrightError(
                "REGISTRY_ERROR",
                "Could not read or update the registry.",
                4,
            ) from exc

    def initialize(
        self,
        name: str,
        origin: ModelOrigin,
        *,
        language: str = "en",
        edition_profile: EditionProfile | None = None,
    ) -> None:
        require_text(name, "name")
        if language not in {"en", "ko"}:
            raise FrontierwrightError("INVALID_LANGUAGE", "language must be 'en' or 'ko'", 2)
        edition = edition_profile or default_edition_for_origin(origin)

        self.state_dir.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("xb"):
                pass
        except FileExistsError as exc:
            raise FrontierwrightError(
                "ALREADY_INITIALIZED",
                "Project registry already exists.",
                3,
            ) from exc

        project_id = str(uuid4())
        identity_id = str(uuid4())
        confidence = default_history_confidence(origin)

        try:
            connection = sqlite3.connect(self.path)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript("BEGIN IMMEDIATE;\n" + SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            created_at = timestamp()
            connection.execute(
                "INSERT INTO project ("
                "singleton, project_id, identity_id, name, language, origin, "
                "history_confidence, edition_profile, created_at, champion_id"
                ") VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    project_id,
                    identity_id,
                    name,
                    language,
                    origin.value,
                    confidence.value,
                    edition.value,
                    created_at,
                ),
            )
            self.event(
                connection,
                "PROJECT_INITIALIZED",
                {
                    "origin": origin.value,
                    "history_confidence": confidence.value,
                    "edition_profile": edition.value,
                    "language": language,
                },
            )
            connection.commit()
        except sqlite3.Error as exc:
            if self.path.exists():
                self.path.unlink(missing_ok=True)
            raise FrontierwrightError(
                "REGISTRY_ERROR",
                "Could not initialize the registry.",
                4,
            ) from exc
        finally:
            if "connection" in locals():
                connection.close()

        self._write_project_toml(
            {
                "project_id": project_id,
                "identity_id": identity_id,
                "name": name,
                "language": language,
                "origin": origin.value,
                "edition_profile": edition.value,
            }
        )

    def _write_project_toml(self, project: dict[str, Any]) -> None:
        (self.state_dir / "project.toml").write_text(
            "\n".join(
                [
                    f"schema_version = {SCHEMA_VERSION}",
                    f'project_id = "{project["project_id"]}"',
                    f'identity_id = "{project["identity_id"]}"',
                    f'name = {json.dumps(project["name"], ensure_ascii=False)}',
                    f'language = "{project["language"]}"',
                    f'origin = "{project["origin"]}"',
                    f'edition_profile = "{project["edition_profile"]}"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def set_edition_profile(self, edition: EditionProfile) -> None:
        with self.connect(write=True) as connection:
            row = connection.execute(
                "SELECT edition_profile FROM project WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise FrontierwrightError("REGISTRY_ERROR", "Missing project metadata.", 4)
            previous = EditionProfile(row["edition_profile"])
            if previous is edition:
                return
            connection.execute(
                "UPDATE project SET edition_profile = ? WHERE singleton = 1",
                (edition.value,),
            )
            self.event(
                connection,
                "EDITION_PROFILE_CHANGED",
                {"from": previous.value, "to": edition.value},
            )

        state = self.read()
        self._write_project_toml(state.project)

    @staticmethod
    def event(connection: sqlite3.Connection, kind: str, details: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO events(kind, recorded_at, details) VALUES (?, ?, ?)",
            (kind, timestamp(), json.dumps(details, sort_keys=True, allow_nan=False)),
        )

    @staticmethod
    def insert_model(connection: sqlite3.Connection, model: ModelState) -> None:
        connection.execute(
            "INSERT INTO models VALUES (?, ?, ?)",
            (
                model.model_id,
                model.parent_model_id,
                json.dumps(asdict(model), sort_keys=True, allow_nan=False),
            ),
        )

    @staticmethod
    def insert_model_artifact(
        connection: sqlite3.Connection,
        model_id: str,
        descriptor: ImportedModelDescriptor,
        evidence: HistoryEvidenceResult,
    ) -> None:
        connection.execute(
            "INSERT INTO model_artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                model_id,
                str(descriptor.source_path),
                descriptor.model_format.value,
                int(descriptor.trainable),
                descriptor.total_bytes,
                json.dumps(
                    [asdict(item) for item in descriptor.files],
                    sort_keys=True,
                    allow_nan=False,
                ),
                evidence.confidence.value,
                json.dumps(evidence.evidence_files, sort_keys=True, allow_nan=False),
                evidence.reason,
            ),
        )

    def set_history_confidence(self, confidence: HistoryConfidence) -> None:
        with self.connect(write=True) as connection:
            connection.execute(
                "UPDATE project SET history_confidence = ? WHERE singleton = 1",
                (confidence.value,),
            )
            self.event(
                connection,
                "HISTORY_CONFIDENCE_CHANGED",
                {"history_confidence": confidence.value},
            )

    def set_initial_imported_model(
        self,
        model: ModelState,
        descriptor: ImportedModelDescriptor,
        evidence: HistoryEvidenceResult,
    ) -> None:
        with self.connect(write=True) as connection:
            project = connection.execute(
                "SELECT identity_id, origin, champion_id FROM project WHERE singleton = 1"
            ).fetchone()
            if project is None:
                raise FrontierwrightError("REGISTRY_ERROR", "Missing project metadata.", 4)
            if project["champion_id"] is not None:
                raise FrontierwrightError(
                    "MODEL_ALREADY_PRESENT",
                    "Project already has a current champion model.",
                    13,
                )
            if model.identity_id != project["identity_id"]:
                raise FrontierwrightError(
                    "IDENTITY_MISMATCH",
                    "Imported model does not belong to this Frontierwright identity.",
                    13,
                )
            if model.origin.value != project["origin"]:
                raise FrontierwrightError(
                    "ORIGIN_MISMATCH",
                    "Imported model origin does not match the project origin.",
                    13,
                )

            self.insert_model(connection, model)
            self.insert_model_artifact(connection, model.model_id, descriptor, evidence)
            connection.execute(
                "UPDATE project SET champion_id = ?, history_confidence = ? WHERE singleton = 1",
                (model.model_id, evidence.confidence.value),
            )
            self.event(
                connection,
                "MODEL_IMPORTED",
                {
                    "model_id": model.model_id,
                    "fingerprint": model.fingerprint,
                    "model_format": model.model_format.value,
                    "trainable": model.trainable,
                    "history_confidence": evidence.confidence.value,
                    "evidence_files": list(evidence.evidence_files),
                },
            )

    def register_birth_model(
        self,
        model: ModelState,
        descriptor: ImportedModelDescriptor,
        *,
        preset: str,
        seed: int,
        backend_id: str,
        runtime: dict[str, object],
    ) -> None:
        with self.connect(write=True) as connection:
            project = connection.execute(
                "SELECT identity_id, origin, champion_id FROM project WHERE singleton = 1"
            ).fetchone()
            if project is None:
                raise FrontierwrightError("REGISTRY_ERROR", "Missing project metadata.", 4)
            if project["origin"] != ModelOrigin.ZERO.value:
                raise FrontierwrightError(
                    "BIRTH_ORIGIN_MISMATCH",
                    "Model birth is only valid for ZERO-origin projects.",
                    13,
                )
            if model.identity_id != project["identity_id"]:
                raise FrontierwrightError(
                    "IDENTITY_MISMATCH",
                    "Born model does not belong to this Frontierwright identity.",
                    13,
                )
            if model.origin is not ModelOrigin.ZERO:
                raise FrontierwrightError(
                    "ORIGIN_MISMATCH",
                    "Born root model must have ZERO origin.",
                    13,
                )

            champion_id = project["champion_id"]
            if champion_id is not None:
                if champion_id == model.model_id:
                    existing = connection.execute(
                        "SELECT model_id FROM model_births WHERE model_id = ?",
                        (model.model_id,),
                    ).fetchone()
                    if existing is not None:
                        return
                raise FrontierwrightError(
                    "MODEL_ALREADY_BORN",
                    "Project already has a materialized current model.",
                    13,
                )

            evidence = HistoryEvidenceResult(
                confidence=HistoryConfidence.COMPLETE,
                evidence_files=(
                    f"birth-backend:{backend_id}",
                    f"birth-preset:{preset}",
                    f"birth-seed:{seed}",
                ),
                reason=(
                    "Root model bytes were materialized by Frontierwright birth and "
                    "fingerprinted before becoming the current model."
                ),
            )
            self.insert_model(connection, model)
            self.insert_model_artifact(connection, model.model_id, descriptor, evidence)
            connection.execute(
                "INSERT INTO model_births "
                "(model_id, preset, seed, backend_id, runtime_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    model.model_id,
                    preset,
                    seed,
                    backend_id,
                    json.dumps(runtime, sort_keys=True, allow_nan=False),
                    timestamp(),
                ),
            )
            connection.execute(
                "UPDATE project SET champion_id = ?, history_confidence = ? "
                "WHERE singleton = 1",
                (model.model_id, HistoryConfidence.COMPLETE.value),
            )
            self.event(
                connection,
                "MODEL_BORN",
                {
                    "model_id": model.model_id,
                    "fingerprint": model.fingerprint,
                    "preset": preset,
                    "seed": seed,
                    "backend_id": backend_id,
                    "trained_steps": 0,
                },
            )

    def get_model_birth(self, model_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_births WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["runtime"] = json.loads(result.pop("runtime_json"))
            return result

    def register_candidate(self, model: ModelState) -> None:
        with self.connect(write=True) as connection:
            project = connection.execute(
                "SELECT identity_id FROM project WHERE singleton = 1"
            ).fetchone()
            if project is None:
                raise FrontierwrightError("REGISTRY_ERROR", "Missing project metadata.", 4)
            if model.identity_id != project["identity_id"]:
                raise FrontierwrightError(
                    "IDENTITY_MISMATCH",
                    "Candidate does not belong to this Frontierwright identity.",
                    13,
                )
            self.insert_model(connection, model)
            connection.execute(
                "INSERT INTO candidates(model_id, status) VALUES (?, ?)",
                (model.model_id, CandidateStatus.PENDING.value),
            )
            self.event(connection, "CANDIDATE_REGISTERED", {"model_id": model.model_id})

    def register_training_candidate(
        self,
        model: ModelState,
        sealed: SealedArtifact,
        *,
        run_id: str,
        metrics: dict[str, object],
    ) -> None:
        with self.connect(write=True) as connection:
            run = connection.execute(
                "SELECT status, candidate_model_id FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise FrontierwrightError("RUN_NOT_FOUND", "Run does not exist.", 3)
            if RunStatus(run["status"]) is RunStatus.COMPLETED:
                if run["candidate_model_id"] == model.model_id:
                    return
                raise FrontierwrightError(
                    "RUN_STATE_CONFLICT",
                    "Completed run is already bound to a different candidate.",
                    13,
                )
            if RunStatus(run["status"]) is not RunStatus.RUNNING:
                raise FrontierwrightError(
                    "RUN_STATE_CONFLICT",
                    "Only RUNNING attempts can finalize a candidate.",
                    13,
                )

            attempt = connection.execute(
                "SELECT result_sha256, result_json FROM run_attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if (
                attempt is None
                or attempt["result_sha256"] is None
                or attempt["result_json"] is None
            ):
                raise FrontierwrightError(
                    "RUN_RESULT_REQUIRED",
                    "Durable executor result evidence is required before candidate registration.",
                    13,
                )

            project = connection.execute(
                "SELECT identity_id, history_confidence FROM project WHERE singleton = 1"
            ).fetchone()
            if project is None:
                raise FrontierwrightError("REGISTRY_ERROR", "Missing project metadata.", 4)
            if model.identity_id != project["identity_id"]:
                raise FrontierwrightError(
                    "IDENTITY_MISMATCH",
                    "Candidate does not belong to this Frontierwright identity.",
                    13,
                )
            if sealed.model_id != model.model_id or sealed.run_id != run_id:
                raise FrontierwrightError(
                    "ARTIFACT_ID_CONFLICT",
                    "Sealed artifact identity does not match candidate/run.",
                    13,
                )
            if sealed.descriptor.fingerprint != model.fingerprint:
                raise FrontierwrightError(
                    "ARTIFACT_FINGERPRINT_MISMATCH",
                    "Sealed artifact fingerprint does not match candidate model.",
                    13,
                )

            evidence = HistoryEvidenceResult(
                confidence=HistoryConfidence(project["history_confidence"]),
                evidence_files=(
                    f"run:{run_id}",
                    f"result:{attempt['result_sha256']}",
                    f"artifact-manifest:{sealed.manifest_sha256}",
                ),
                reason=(
                    f"Candidate produced by Frontierwright run {run_id}; "
                    "durable executor evidence and sealed artifact recorded before registration."
                ),
            )
            self.insert_model(connection, model)
            self.insert_model_artifact(
                connection,
                model.model_id,
                sealed.descriptor,
                evidence,
            )
            connection.execute(
                "INSERT INTO sealed_artifacts "
                "(model_id, run_id, artifact_root, model_path, manifest_path, "
                "manifest_sha256, model_fingerprint, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model.model_id,
                    run_id,
                    str(sealed.artifact_root),
                    str(sealed.model_path),
                    str(sealed.manifest_path),
                    sealed.manifest_sha256,
                    model.fingerprint,
                    timestamp(),
                ),
            )
            connection.execute(
                "INSERT INTO candidates(model_id, status) VALUES (?, ?)",
                (model.model_id, CandidateStatus.PENDING.value),
            )
            finished = timestamp()
            changed = connection.execute(
                "UPDATE runs SET status = ?, finished_at = ?, candidate_model_id = ?, "
                "metrics_json = ? WHERE run_id = ? AND status = ?",
                (
                    RunStatus.COMPLETED.value,
                    finished,
                    model.model_id,
                    json.dumps(metrics, sort_keys=True, allow_nan=False),
                    run_id,
                    RunStatus.RUNNING.value,
                ),
            ).rowcount
            if changed != 1:
                raise FrontierwrightError(
                    "RUN_STATE_CONFLICT",
                    "Run is not in RUNNING state.",
                    13,
                )
            connection.execute(
                "UPDATE run_attempts SET liveness_state = 'TERMINAL', updated_at = ? "
                "WHERE run_id = ?",
                (finished, run_id),
            )
            self.event(
                connection,
                "RUN_COMPLETED",
                {"run_id": run_id, "candidate_model_id": model.model_id},
            )
            self.event(
                connection,
                "CANDIDATE_REGISTERED",
                {"model_id": model.model_id, "run_id": run_id},
            )

    def get_model(self, model_id: str) -> ModelState:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT snapshot FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                raise FrontierwrightError("MODEL_NOT_FOUND", "Model does not exist.", 3)
            return decode_model(row["snapshot"])

    def get_model_artifact(self, model_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_artifacts WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["trainable"] = bool(result["trainable"])
            result["files"] = json.loads(result.pop("files_json"))
            result["evidence_files"] = json.loads(result.pop("evidence_files_json"))
            return result

    def get_sealed_artifact(self, model_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM sealed_artifacts WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def get_active_capability_profile(self, model_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT cp.*, cs.scale_id, cs.scale_version, "
                "er.evaluator_id, er.evaluator_version, er.receipt_sha256, "
                "er.conditions_json, er.measurements_json "
                "FROM capability_profiles cp "
                "JOIN capability_scales cs ON cs.scale_hash = cp.scale_hash "
                "JOIN evaluation_receipts er ON er.receipt_id = cp.receipt_id "
                "WHERE cp.model_id = ? AND cp.active = 1 "
                "ORDER BY cp.created_at DESC LIMIT 1",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["active"] = bool(payload["active"])
            payload["stats"] = json.loads(payload.pop("stats_json"))
            payload["conditions"] = json.loads(payload.pop("conditions_json"))
            payload["measurements"] = json.loads(payload.pop("measurements_json"))
            return payload

    def get_candidate(self, model_id: str) -> Candidate:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT m.snapshot, c.status FROM candidates c "
                "JOIN models m USING(model_id) WHERE c.model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                raise FrontierwrightError(
                    "CANDIDATE_NOT_FOUND",
                    "Candidate does not exist.",
                    3,
                )
            return Candidate(
                decode_model(row["snapshot"]),
                CandidateStatus(row["status"]),
            )

    def promote_candidate(
        self,
        model_id: str,
        *,
        expected_state: dict[str, str | None] | None = None,
        decision_details: dict[str, Any] | None = None,
    ) -> None:
        with self.connect(write=True) as connection:
            if expected_state is not None:
                project = connection.execute(
                    "SELECT champion_id FROM project WHERE singleton = 1"
                ).fetchone()
                if project is None:
                    raise FrontierwrightError(
                        "REGISTRY_ERROR",
                        "Missing project metadata.",
                        4,
                    )
                build = connection.execute(
                    "SELECT updated_at FROM build_state WHERE singleton = 1"
                ).fetchone()
                champion_id = project["champion_id"]
                champion_profile = (
                    connection.execute(
                        "SELECT profile_id FROM capability_profiles "
                        "WHERE model_id = ? AND active = 1 "
                        "ORDER BY created_at DESC LIMIT 1",
                        (champion_id,),
                    ).fetchone()
                    if champion_id is not None
                    else None
                )
                candidate_profile = connection.execute(
                    "SELECT profile_id FROM capability_profiles "
                    "WHERE model_id = ? AND active = 1 "
                    "ORDER BY created_at DESC LIMIT 1",
                    (model_id,),
                ).fetchone()
                actual_state: dict[str, str | None] = {
                    "champion_id": (
                        str(champion_id) if champion_id is not None else None
                    ),
                    "build_updated_at": (
                        str(build["updated_at"]) if build is not None else None
                    ),
                    "champion_profile_id": (
                        str(champion_profile["profile_id"])
                        if champion_profile is not None
                        else None
                    ),
                    "candidate_profile_id": (
                        str(candidate_profile["profile_id"])
                        if candidate_profile is not None
                        else None
                    ),
                }
                for key, expected in expected_state.items():
                    if key not in actual_state or actual_state[key] != expected:
                        raise FrontierwrightError(
                            "PROMOTION_STATE_CHANGED",
                            (
                                "Champion/build/evaluation evidence changed after the "
                                "promotion gate was evaluated; compare again."
                            ),
                            13,
                        )

            row = connection.execute(
                "SELECT status FROM candidates WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                raise FrontierwrightError("CANDIDATE_NOT_FOUND", "Candidate does not exist.", 3)
            status = CandidateStatus(row["status"])
            if status is CandidateStatus.INCOMPLETE:
                raise FrontierwrightError(
                    "CANDIDATE_INCOMPLETE",
                    "Incomplete candidates cannot become champion.",
                    13,
                )
            if status is not CandidateStatus.PENDING:
                raise FrontierwrightError(
                    "CANDIDATE_STATE_CONFLICT",
                    f"Only PENDING candidates can be promoted; current status is {status.value}.",
                    13,
                )
            connection.execute(
                "UPDATE candidates SET status = ? WHERE model_id = ? AND status = ?",
                (
                    CandidateStatus.ACCEPTED.value,
                    model_id,
                    CandidateStatus.PENDING.value,
                ),
            )
            connection.execute(
                "UPDATE project SET champion_id = ? WHERE singleton = 1",
                (model_id,),
            )
            self.event(
                connection,
                "CANDIDATE_PROMOTED",
                {
                    "model_id": model_id,
                    **(decision_details or {}),
                },
            )

    def reject_candidate(self, model_id: str) -> None:
        with self.connect(write=True) as connection:
            row = connection.execute(
                "SELECT status FROM candidates WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                raise FrontierwrightError("CANDIDATE_NOT_FOUND", "Candidate does not exist.", 3)
            status = CandidateStatus(row["status"])
            if status is not CandidateStatus.PENDING:
                raise FrontierwrightError(
                    "CANDIDATE_STATE_CONFLICT",
                    f"Only PENDING candidates can be rejected; current status is {status.value}.",
                    13,
                )
            connection.execute(
                "UPDATE candidates SET status = ? WHERE model_id = ? AND status = ?",
                (
                    CandidateStatus.REJECTED.value,
                    model_id,
                    CandidateStatus.PENDING.value,
                ),
            )
            self.event(connection, "CANDIDATE_REJECTED", {"model_id": model_id})

    def store_plan(self, plan: TrainingPlan) -> TrainingPlan:
        with self.connect(write=True) as connection:
            existing = connection.execute(
                "SELECT * FROM plans WHERE idempotency_key = ?",
                (plan.idempotency_key,),
            ).fetchone()
            if existing is not None:
                return decode_plan(existing)

            connection.execute(
                "INSERT INTO plans ("
                "plan_id, idempotency_key, path_id, intervention_id, "
                "intervention_version, intervention_family, backend_id, "
                "backend_spec_hash, model_id, model_fingerprint, model_source_path, "
                "dataset_id, dataset_fingerprint, dataset_source_path, "
                "dataset_recipe_id, dataset_recipe_hash, dataset_classification, "
                "backend_data_boundary, resource_profile_id, permission, budgets_json, "
                "config_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    plan.plan_id,
                    plan.idempotency_key,
                    plan.path_id.value,
                    plan.intervention_id,
                    plan.intervention_version,
                    plan.intervention_family,
                    plan.backend_id,
                    plan.backend_spec_hash,
                    plan.model_id,
                    plan.model_fingerprint,
                    plan.model_source_path,
                    plan.dataset_id,
                    plan.dataset_fingerprint,
                    plan.dataset_source_path,
                    plan.dataset_recipe_id,
                    plan.dataset_recipe_hash,
                    plan.dataset_classification,
                    plan.backend_data_boundary,
                    plan.resource_profile_id,
                    plan.permission.name,
                    json.dumps(plan.budgets.to_dict(), sort_keys=True, allow_nan=False),
                    json.dumps(plan.config, sort_keys=True, allow_nan=False),
                    timestamp(),
                ),
            )
            self.event(
                connection,
                "PLAN_CREATED",
                {
                    "plan_id": plan.plan_id,
                    "path_id": plan.path_id.value,
                    "intervention_id": plan.intervention_id,
                    "intervention_version": plan.intervention_version,
                    "intervention_family": plan.intervention_family,
                    "backend_id": plan.backend_id,
                    "idempotency_key": plan.idempotency_key,
                },
            )
            row = connection.execute(
                "SELECT * FROM plans WHERE plan_id = ?",
                (plan.plan_id,),
            ).fetchone()
            if row is None:
                raise FrontierwrightError("REGISTRY_ERROR", "Plan insert failed.", 4)
            return decode_plan(row)

    def get_plan(self, plan_id: str) -> TrainingPlan:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
            if row is None:
                raise FrontierwrightError("PLAN_NOT_FOUND", "Training plan does not exist.", 3)
            return decode_plan(row)

    def store_calibration(self, receipt: CalibrationReceipt) -> None:
        with self.connect(write=True) as connection:
            plan = connection.execute(
                "SELECT backend_id FROM plans WHERE plan_id = ?",
                (receipt.plan_id,),
            ).fetchone()
            if plan is None:
                raise FrontierwrightError("PLAN_NOT_FOUND", "Training plan does not exist.", 3)
            if plan["backend_id"] != receipt.backend_id:
                raise FrontierwrightError(
                    "BACKEND_MISMATCH",
                    "Calibration backend does not match the training plan.",
                    13,
                )
            connection.execute(
                "INSERT INTO calibrations ("
                "calibration_id, plan_id, backend_id, feasible, representative_steps, "
                "step_time_seconds, tokens_per_second, peak_vram_bytes, peak_ram_bytes, "
                "projected_storage_bytes, projected_wall_seconds, result_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.calibration_id,
                    receipt.plan_id,
                    receipt.backend_id,
                    int(receipt.feasible),
                    receipt.representative_steps,
                    receipt.step_time_seconds,
                    receipt.tokens_per_second,
                    receipt.peak_vram_bytes,
                    receipt.peak_ram_bytes,
                    receipt.projected_storage_bytes,
                    receipt.projected_wall_seconds,
                    json.dumps(receipt.backend_result, sort_keys=True, allow_nan=False),
                    timestamp(),
                ),
            )
            self.event(
                connection,
                "CALIBRATION_RECORDED",
                {
                    "calibration_id": receipt.calibration_id,
                    "plan_id": receipt.plan_id,
                    "feasible": receipt.feasible,
                },
            )

    def latest_calibration_for_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM calibrations WHERE plan_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (plan_id,),
            ).fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["feasible"] = bool(payload["feasible"])
            payload["backend_result"] = json.loads(payload.pop("result_json"))
            return payload

    def latest_run_for_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT r.*, a.attempt_token, a.calibration_id, a.request_digest, "
                "a.executor_kind, a.owner_pid, a.worker_pid, a.worker_start_token, "
                "a.liveness_state, a.result_path, a.result_sha256, a.result_json, "
                "a.updated_at AS attempt_updated_at "
                "FROM runs r LEFT JOIN run_attempts a USING(run_id) "
                "WHERE r.plan_id = ? ORDER BY r.started_at DESC, r.run_id DESC LIMIT 1",
                (plan_id,),
            ).fetchone()
            return self._decode_run_row(row) if row is not None else None

    @staticmethod
    def _decode_run_row(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        payload["metrics"] = json.loads(payload.pop("metrics_json"))
        result_raw = payload.pop("result_json", None)
        payload["result"] = json.loads(result_raw) if result_raw else None
        return payload

    def start_run(
        self,
        plan_id: str,
        *,
        rerun: bool,
        calibration_id: str,
        request_digest: str,
        owner_pid: int,
    ) -> tuple[str, bool]:
        with self.connect(write=True) as connection:
            plan = connection.execute(
                "SELECT plan_id, budgets_json FROM plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
            if plan is None:
                raise FrontierwrightError("PLAN_NOT_FOUND", "Training plan does not exist.", 3)

            latest = connection.execute(
                "SELECT * FROM runs WHERE plan_id = ? "
                "ORDER BY started_at DESC, run_id DESC LIMIT 1",
                (plan_id,),
            ).fetchone()
            if latest is not None:
                status = RunStatus(latest["status"])
                if not rerun:
                    if status in (RunStatus.RUNNING, RunStatus.COMPLETED):
                        return str(latest["run_id"]), True
                    raise FrontierwrightError(
                        "RUN_RERUN_REQUIRED",
                        "Previous run is terminal without success; use --rerun to retry.",
                        13,
                    )
                if status is RunStatus.RUNNING:
                    raise FrontierwrightError(
                        "RUN_ACTIVE",
                        "Cannot rerun while a prior attempt is still active or unresolved.",
                        13,
                    )

            budgets = HardBudgets(**json.loads(plan["budgets_json"]))
            count = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()[0]
            if budgets.max_runs is not None and count >= budgets.max_runs:
                raise FrontierwrightError(
                    "RUN_BUDGET_REACHED",
                    "Plan max_runs budget has been reached.",
                    13,
                )

            calibration = connection.execute(
                "SELECT calibration_id FROM calibrations "
                "WHERE calibration_id = ? AND plan_id = ?",
                (calibration_id, plan_id),
            ).fetchone()
            if calibration is None:
                raise FrontierwrightError(
                    "CALIBRATION_NOT_FOUND",
                    "Pinned calibration does not belong to this training plan.",
                    3,
                )

            run_id = f"run-{uuid4().hex}"
            attempt_token = f"attempt-{uuid4().hex}"
            now = timestamp()
            try:
                connection.execute(
                    "INSERT INTO runs ("
                    "run_id, plan_id, status, started_at, finished_at, "
                    "candidate_model_id, metrics_json, error_code, error_message"
                    ") VALUES (?, ?, ?, ?, NULL, NULL, '{}', NULL, NULL)",
                    (run_id, plan_id, RunStatus.RUNNING.value, now),
                )
                connection.execute(
                    "INSERT INTO run_attempts ("
                    "run_id, attempt_token, calibration_id, request_digest, executor_kind, "
                    "owner_pid, worker_pid, worker_start_token, liveness_state, result_path, "
                    "result_sha256, result_json, updated_at"
                    ") VALUES (?, ?, ?, ?, 'local', ?, NULL, NULL, 'RESERVED', "
                    "NULL, NULL, NULL, ?)",
                    (
                        run_id,
                        attempt_token,
                        calibration_id,
                        request_digest,
                        owner_pid,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                active = connection.execute(
                    "SELECT run_id FROM runs WHERE plan_id = ? AND status = 'RUNNING' "
                    "ORDER BY started_at DESC, run_id DESC LIMIT 1",
                    (plan_id,),
                ).fetchone()
                if active is not None:
                    return str(active["run_id"]), True
                raise FrontierwrightError(
                    "RUN_ADMISSION_CONFLICT",
                    "Could not reserve a unique training attempt.",
                    13,
                ) from exc

            self.event(
                connection,
                "RUN_RESERVED",
                {
                    "run_id": run_id,
                    "plan_id": plan_id,
                    "calibration_id": calibration_id,
                    "attempt_token": attempt_token,
                },
            )
            return run_id, False

    def attach_run_worker(
        self,
        run_id: str,
        *,
        worker_pid: int,
        worker_start_token: str | None,
        result_path: str,
    ) -> None:
        with self.connect(write=True) as connection:
            changed = connection.execute(
                "UPDATE run_attempts SET worker_pid = ?, worker_start_token = ?, "
                "liveness_state = 'LIVE', result_path = ?, updated_at = ? "
                "WHERE run_id = ?",
                (
                    worker_pid,
                    worker_start_token,
                    result_path,
                    timestamp(),
                    run_id,
                ),
            ).rowcount
            if changed != 1:
                raise FrontierwrightError(
                    "RUN_ATTEMPT_NOT_FOUND",
                    "Run attempt metadata does not exist.",
                    3,
                )
            self.event(
                connection,
                "RUN_WORKER_ATTACHED",
                {
                    "run_id": run_id,
                    "worker_pid": worker_pid,
                    "worker_start_token": worker_start_token,
                },
            )

    def set_run_liveness(self, run_id: str, state: str) -> None:
        if state not in {"RESERVED", "LIVE", "TERMINAL", "UNRESOLVED"}:
            raise ValueError("invalid run liveness state")
        with self.connect(write=True) as connection:
            changed = connection.execute(
                "UPDATE run_attempts SET liveness_state = ?, updated_at = ? WHERE run_id = ?",
                (state, timestamp(), run_id),
            ).rowcount
            if changed != 1:
                raise FrontierwrightError(
                    "RUN_ATTEMPT_NOT_FOUND",
                    "Run attempt metadata does not exist.",
                    3,
                )

    def record_run_result_evidence(
        self,
        run_id: str,
        *,
        result_path: str,
        result_sha256: str,
        result: dict[str, object],
    ) -> None:
        serialized = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self.connect(write=True) as connection:
            row = connection.execute(
                "SELECT result_sha256, result_json FROM run_attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise FrontierwrightError(
                    "RUN_ATTEMPT_NOT_FOUND",
                    "Run attempt metadata does not exist.",
                    3,
                )
            if row["result_sha256"] is not None:
                if (
                    row["result_sha256"] == result_sha256
                    and row["result_json"] == serialized
                ):
                    return
                raise FrontierwrightError(
                    "RUN_RESULT_CONFLICT",
                    "Run already has different durable executor result evidence.",
                    13,
                )
            connection.execute(
                "UPDATE run_attempts SET result_path = ?, result_sha256 = ?, result_json = ?, "
                "updated_at = ? WHERE run_id = ?",
                (result_path, result_sha256, serialized, timestamp(), run_id),
            )
            self.event(
                connection,
                "RUN_RESULT_RECORDED",
                {
                    "run_id": run_id,
                    "result_sha256": result_sha256,
                },
            )

    def get_run_attempt(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            payload = dict(row)
            result_raw = payload.pop("result_json")
            payload["result"] = json.loads(result_raw) if result_raw else None
            return payload

    def finish_run_failure(
        self,
        run_id: str,
        *,
        status: RunStatus,
        error_code: str,
        error_message: str,
    ) -> None:
        if status not in (RunStatus.FAILED, RunStatus.INCOMPLETE):
            raise ValueError("Failure completion must use FAILED or INCOMPLETE")
        with self.connect(write=True) as connection:
            changed = connection.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error_code = ?, "
                "error_message = ? WHERE run_id = ? AND status = ?",
                (
                    status.value,
                    timestamp(),
                    error_code,
                    error_message[:2000],
                    run_id,
                    RunStatus.RUNNING.value,
                ),
            ).rowcount
            if changed != 1:
                raise FrontierwrightError(
                    "RUN_STATE_CONFLICT",
                    "Run is not in RUNNING state.",
                    13,
                )
            connection.execute(
                "UPDATE run_attempts SET liveness_state = 'TERMINAL', updated_at = ? "
                "WHERE run_id = ?",
                (timestamp(), run_id),
            )
            self.event(
                connection,
                "RUN_STOPPED",
                {
                    "run_id": run_id,
                    "status": status.value,
                    "error_code": error_code,
                },
            )

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT r.*, a.attempt_token, a.calibration_id, a.request_digest, "
                "a.executor_kind, a.owner_pid, a.worker_pid, a.worker_start_token, "
                "a.liveness_state, a.result_path, a.result_sha256, a.result_json, "
                "a.updated_at AS attempt_updated_at "
                "FROM runs r LEFT JOIN run_attempts a USING(run_id) "
                "WHERE r.run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise FrontierwrightError("RUN_NOT_FOUND", "Run does not exist.", 3)
            return self._decode_run_row(row)

    def register_dataset(
        self,
        descriptor: DatasetDescriptor,
        *,
        name: str,
        role: DatasetRole,
        provenance: DatasetProvenance = DatasetProvenance.LOCAL_USER,
        classification: DatasetClassification | None = None,
        license_name: str | None = None,
        domain: str | None = None,
        language: str | None = None,
        token_count: int | None = None,
    ) -> str:
        require_text(name, "dataset name")
        effective_classification = (
            classification or default_dataset_classification(provenance)
        )
        if token_count is not None and token_count < 0:
            raise FrontierwrightError(
                "INVALID_TOKEN_COUNT",
                "Dataset token count must be nonnegative.",
                2,
            )

        with self.connect(write=True) as connection:
            existing = connection.execute(
                "SELECT dataset_id FROM datasets "
                "WHERE fingerprint = ? AND role = ? AND provenance = ? "
                "AND classification = ? AND preparation_recipe_hash IS NULL "
                "AND active = 1 LIMIT 1",
                (
                    descriptor.fingerprint,
                    role.value,
                    provenance.value,
                    effective_classification.value,
                ),
            ).fetchone()
            if existing is not None:
                return str(existing["dataset_id"])

            dataset_id = f"dataset-{uuid4().hex}"
            connection.execute(
                "INSERT INTO datasets ("
                "dataset_id, name, role, provenance, classification, source_path, "
                "fingerprint, total_bytes, file_count, manifest_json, license, domain, "
                "language, token_count, created_at, active, source_dataset_id, "
                "preparation_recipe_id, preparation_recipe_hash"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, NULL, NULL)",
                (
                    dataset_id,
                    name,
                    role.value,
                    provenance.value,
                    effective_classification.value,
                    str(descriptor.source_path),
                    descriptor.fingerprint,
                    descriptor.total_bytes,
                    descriptor.file_count,
                    json.dumps(descriptor.manifest(), sort_keys=True, allow_nan=False),
                    license_name,
                    domain,
                    language,
                    token_count,
                    timestamp(),
                ),
            )
            self.event(
                connection,
                "DATASET_REGISTERED",
                {
                    "dataset_id": dataset_id,
                    "name": name,
                    "role": role.value,
                    "provenance": provenance.value,
                    "classification": effective_classification.value,
                    "fingerprint": descriptor.fingerprint,
                    "total_bytes": descriptor.total_bytes,
                    "file_count": descriptor.file_count,
                },
            )
            return dataset_id

    def register_prepared_dataset(
        self,
        recipe: DataPreparationRecipe,
        descriptor: DatasetDescriptor,
        *,
        name: str | None = None,
    ) -> str:
        with self.connect(write=True) as connection:
            source = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id = ? AND active = 1",
                (recipe.source_dataset_id,),
            ).fetchone()
            if source is None:
                raise FrontierwrightError(
                    "DATASET_NOT_FOUND",
                    "Preparation recipe source dataset is not active.",
                    3,
                )
            if source["fingerprint"] != recipe.source_fingerprint:
                raise FrontierwrightError(
                    "DATA_RECIPE_SOURCE_DRIFT",
                    "Preparation recipe source fingerprint does not match the registry.",
                    13,
                )
            if descriptor.fingerprint != recipe.source_fingerprint:
                raise FrontierwrightError(
                    "DATA_PREP_COPY_MISMATCH",
                    "Prepared dataset fingerprint differs from the byte-preserving recipe source.",
                    13,
                )

            existing_recipe = connection.execute(
                "SELECT * FROM data_recipes WHERE recipe_hash = ?",
                (recipe.recipe_hash,),
            ).fetchone()
            if existing_recipe is None:
                connection.execute(
                    "INSERT INTO data_recipes ("
                    "recipe_id, recipe_hash, plugin_id, plugin_version, source_dataset_id, "
                    "source_fingerprint, config_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        recipe.recipe_id,
                        recipe.recipe_hash,
                        recipe.plugin_id,
                        recipe.plugin_version,
                        recipe.source_dataset_id,
                        recipe.source_fingerprint,
                        json.dumps(recipe.config, sort_keys=True, allow_nan=False),
                        timestamp(),
                    ),
                )
            elif (
                existing_recipe["recipe_id"] != recipe.recipe_id
                or existing_recipe["plugin_id"] != recipe.plugin_id
                or existing_recipe["plugin_version"] != recipe.plugin_version
                or existing_recipe["source_dataset_id"] != recipe.source_dataset_id
                or existing_recipe["source_fingerprint"] != recipe.source_fingerprint
                or json.loads(existing_recipe["config_json"]) != recipe.config
            ):
                raise FrontierwrightError(
                    "DATA_RECIPE_CONFLICT",
                    "Existing data recipe hash is bound to different recipe content.",
                    13,
                )

            existing = connection.execute(
                "SELECT dataset_id FROM datasets "
                "WHERE preparation_recipe_hash = ? AND active = 1 LIMIT 1",
                (recipe.recipe_hash,),
            ).fetchone()
            if existing is not None:
                return str(existing["dataset_id"])

            dataset_id = f"dataset-{uuid4().hex}"
            prepared_name = name or f"{source['name']} · prepared"
            connection.execute(
                "INSERT INTO datasets ("
                "dataset_id, name, role, provenance, classification, source_path, "
                "fingerprint, total_bytes, file_count, manifest_json, license, domain, "
                "language, token_count, created_at, active, source_dataset_id, "
                "preparation_recipe_id, preparation_recipe_hash"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                (
                    dataset_id,
                    prepared_name,
                    source["role"],
                    source["provenance"],
                    source["classification"],
                    str(descriptor.source_path),
                    descriptor.fingerprint,
                    descriptor.total_bytes,
                    descriptor.file_count,
                    json.dumps(descriptor.manifest(), sort_keys=True, allow_nan=False),
                    source["license"],
                    source["domain"],
                    source["language"],
                    source["token_count"],
                    timestamp(),
                    recipe.source_dataset_id,
                    recipe.recipe_id,
                    recipe.recipe_hash,
                ),
            )
            self.event(
                connection,
                "DATASET_PREPARED",
                {
                    "dataset_id": dataset_id,
                    "source_dataset_id": recipe.source_dataset_id,
                    "recipe_id": recipe.recipe_id,
                    "recipe_hash": recipe.recipe_hash,
                    "plugin_id": recipe.plugin_id,
                    "plugin_version": recipe.plugin_version,
                    "classification": source["classification"],
                    "fingerprint": descriptor.fingerprint,
                },
            )
            return dataset_id

    def get_data_recipe(self, recipe_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM data_recipes WHERE recipe_id = ?",
                (recipe_id,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["config"] = json.loads(result.pop("config_json"))
            return result

    def store_evaluation_receipt(self, receipt: EvaluationReceipt) -> None:
        with self.connect(write=True) as connection:
            model_row = connection.execute(
                "SELECT snapshot FROM models WHERE model_id = ?",
                (receipt.model_id,),
            ).fetchone()
            if model_row is None:
                raise FrontierwrightError(
                    "EVALUATION_MODEL_NOT_FOUND",
                    "Evaluation receipt references a model not present in this project.",
                    12,
                )
            model = decode_model(model_row["snapshot"])
            if model.fingerprint != receipt.model_fingerprint:
                raise FrontierwrightError(
                    "EVALUATION_FINGERPRINT_MISMATCH",
                    "Evaluation receipt fingerprint does not match the referenced model.",
                    12,
                )

            existing = connection.execute(
                "SELECT receipt_sha256 FROM evaluation_receipts WHERE receipt_id = ?",
                (receipt.receipt_id,),
            ).fetchone()
            if existing is not None:
                if existing["receipt_sha256"] == receipt.sha256:
                    return
                raise FrontierwrightError(
                    "EVALUATION_RECEIPT_CONFLICT",
                    "Receipt ID already exists with different content.",
                    13,
                )

            connection.execute(
                "INSERT INTO evaluation_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.receipt_id,
                    receipt.model_id,
                    receipt.model_fingerprint,
                    receipt.evaluator_id,
                    receipt.evaluator_version,
                    receipt.sha256,
                    json.dumps(receipt.conditions, sort_keys=True, allow_nan=False),
                    json.dumps(
                        [asdict(item) for item in receipt.measurements],
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    timestamp(),
                ),
            )
            self.event(
                connection,
                "EVALUATION_RECEIPT_IMPORTED",
                {
                    "receipt_id": receipt.receipt_id,
                    "model_id": receipt.model_id,
                    "receipt_sha256": receipt.sha256,
                    "measurement_count": len(receipt.measurements),
                },
            )

    def store_capability_scale(self, scale: CapabilityScale) -> None:
        with self.connect(write=True) as connection:
            existing = connection.execute(
                "SELECT manifest_json FROM capability_scales WHERE scale_hash = ?",
                (scale.sha256,),
            ).fetchone()
            manifest = json.dumps(
                scale.canonical_payload(),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            if existing is not None:
                if existing["manifest_json"] == manifest:
                    return
                raise FrontierwrightError(
                    "CAPABILITY_SCALE_CONFLICT",
                    "Capability scale hash already exists with different content.",
                    13,
                )
            connection.execute(
                "INSERT INTO capability_scales VALUES (?, ?, ?, ?, ?)",
                (
                    scale.sha256,
                    scale.scale_id,
                    scale.scale_version,
                    manifest,
                    timestamp(),
                ),
            )
            self.event(
                connection,
                "CAPABILITY_SCALE_IMPORTED",
                {
                    "scale_hash": scale.sha256,
                    "scale_id": scale.scale_id,
                    "scale_version": scale.scale_version,
                },
            )

    def activate_capability_profile(
        self,
        receipt: EvaluationReceipt,
        scale: CapabilityScale,
        stats: tuple[CapabilityStat, ...],
    ) -> str:
        if not stats:
            raise FrontierwrightError(
                "NO_MEASURABLE_AXES",
                "Frozen scale could not produce any complete capability axis.",
                12,
            )
        self.store_evaluation_receipt(receipt)
        self.store_capability_scale(scale)

        profile_id = f"profile-{uuid4().hex}"
        with self.connect(write=True) as connection:
            connection.execute(
                "UPDATE capability_profiles SET active = 0 WHERE model_id = ?",
                (receipt.model_id,),
            )
            connection.execute(
                "INSERT INTO capability_profiles VALUES (?, ?, ?, ?, ?, ?, 1)",
                (
                    profile_id,
                    receipt.model_id,
                    receipt.receipt_id,
                    scale.sha256,
                    json.dumps([asdict(stat) for stat in stats], sort_keys=True, allow_nan=False),
                    timestamp(),
                ),
            )
            self.event(
                connection,
                "CAPABILITY_PROFILE_ACTIVATED",
                {
                    "profile_id": profile_id,
                    "model_id": receipt.model_id,
                    "receipt_id": receipt.receipt_id,
                    "scale_hash": scale.sha256,
                    "axes": [stat.axis.value for stat in stats],
                },
            )
        return profile_id

    @staticmethod
    def _active_profile_stats(
        connection: sqlite3.Connection,
        model_id: str,
    ) -> tuple[CapabilityStat, ...]:
        row = connection.execute(
            "SELECT stats_json FROM capability_profiles "
            "WHERE model_id = ? AND active = 1 ORDER BY created_at DESC LIMIT 1",
            (model_id,),
        ).fetchone()
        if row is None:
            return ()
        raw = json.loads(row["stats_json"])
        return tuple(
            CapabilityStat(**{**item, "axis": Axis(item["axis"])})
            for item in raw
        )

    def _current_build_mode(self, connection: sqlite3.Connection) -> BuildMode:
        project = connection.execute(
            "SELECT origin, champion_id FROM project WHERE singleton = 1"
        ).fetchone()
        if project is None:
            raise FrontierwrightError("REGISTRY_ERROR", "Missing project metadata.", 4)
        champion = None
        if project["champion_id"] is not None:
            row = connection.execute(
                "SELECT snapshot FROM models WHERE model_id = ?",
                (project["champion_id"],),
            ).fetchone()
            if row is not None:
                champion = Champion(decode_model(row["snapshot"]))
                if champion.model.measured:
                    return BuildMode.TARGETS_FLOORS
                if self._active_profile_stats(connection, champion.model.model_id):
                    return BuildMode.TARGETS_FLOORS
        return build_mode(ModelOrigin(project["origin"]), champion)

    def set_build_intent(self, intent: BuildIntent) -> None:
        with self.connect(write=True) as connection:
            mode = self._current_build_mode(connection)
            if mode.value != "INTENT":
                raise FrontierwrightError(
                    "BUILD_MODE_MISMATCH",
                    f"Build intent is unavailable while current build mode is {mode.value}.",
                    13,
                )
            payload = intent.to_dict()
            connection.execute("DELETE FROM build_state WHERE singleton = 1")
            connection.execute(
                "INSERT INTO build_state ("
                "singleton, mode, archetype, priorities_json, targets_json, floors_json, "
                "scale_hash, scale_id, scale_version, updated_at"
                ") VALUES (1, 'INTENT', ?, ?, NULL, NULL, NULL, NULL, NULL, ?)",
                (
                    intent.archetype,
                    json.dumps(payload["priorities"], sort_keys=True),
                    timestamp(),
                ),
            )
            self.event(connection, "BUILD_INTENT_SET", payload)

    def set_build_targets(
        self,
        targets: BuildTargets,
        *,
        scale_hash: str,
        scale_id: str,
        scale_version: str,
    ) -> None:
        for name, value in (
            ("scale_hash", scale_hash),
            ("scale_id", scale_id),
            ("scale_version", scale_version),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be nonempty")
        with self.connect(write=True) as connection:
            mode = self._current_build_mode(connection)
            if mode.value != "TARGETS_FLOORS":
                raise FrontierwrightError(
                    "BUILD_MODE_MISMATCH",
                    (
                        "Numeric targets/floors are unavailable while current build "
                        f"mode is {mode.value}."
                    ),
                    13,
                )
            payload = targets.to_dict()
            connection.execute("DELETE FROM build_state WHERE singleton = 1")
            connection.execute(
                "INSERT INTO build_state ("
                "singleton, mode, archetype, priorities_json, targets_json, floors_json, "
                "scale_hash, scale_id, scale_version, updated_at"
                ") VALUES (1, 'TARGETS_FLOORS', NULL, NULL, ?, ?, ?, ?, ?, ?)",
                (
                    json.dumps(payload["targets"], sort_keys=True),
                    json.dumps(payload["floors"], sort_keys=True),
                    scale_hash,
                    scale_id,
                    scale_version,
                    timestamp(),
                ),
            )
            self.event(
                connection,
                "BUILD_TARGETS_SET",
                {
                    **payload,
                    "scale_hash": scale_hash,
                    "scale_id": scale_id,
                    "scale_version": scale_version,
                },
            )

    def save_resource_snapshot(
        self,
        snapshot: ResourceSnapshot,
        *,
        profile_name: str = "local",
    ) -> str:
        profile_id = f"resource-{uuid4().hex}"
        with self.connect(write=True) as connection:
            connection.execute(
                "UPDATE resource_profiles SET active = 0 WHERE profile_name = ?",
                (profile_name,),
            )
            connection.execute(
                "INSERT INTO resource_profiles VALUES (?, ?, ?, ?, ?, 1)",
                (
                    profile_id,
                    profile_name,
                    snapshot.provenance.value,
                    timestamp(),
                    json.dumps(snapshot.to_dict(), sort_keys=True, allow_nan=False),
                ),
            )
            self.event(
                connection,
                "RESOURCE_PROFILE_DETECTED",
                {"profile_id": profile_id, "profile_name": profile_name},
            )
        return profile_id

    def read(self) -> ProjectState:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM project WHERE singleton = 1").fetchone()
            if row is None:
                raise FrontierwrightError("REGISTRY_ERROR", "Registry has no project metadata.", 4)
            project = dict(row)
            del project["singleton"]

            champion_row = connection.execute(
                "SELECT snapshot FROM models WHERE model_id = ?",
                (project["champion_id"],),
            ).fetchone()
            champion = Champion(decode_model(champion_row[0])) if champion_row else None

            artifact_row = connection.execute(
                "SELECT * FROM model_artifacts WHERE model_id = ?",
                (project["champion_id"],),
            ).fetchone()
            champion_artifact = dict(artifact_row) if artifact_row else None
            if champion_artifact is not None:
                champion_artifact["trainable"] = bool(champion_artifact["trainable"])
                champion_artifact["files"] = json.loads(champion_artifact.pop("files_json"))
                champion_artifact["evidence_files"] = json.loads(
                    champion_artifact.pop("evidence_files_json")
                )

            candidates = tuple(
                Candidate(
                    decode_model(candidate_row["snapshot"]),
                    CandidateStatus(candidate_row["status"]),
                )
                for candidate_row in connection.execute(
                    "SELECT snapshot, status FROM candidates JOIN models USING(model_id) "
                    "ORDER BY model_id"
                )
            )
            history = tuple(
                {**dict(event_row), "details": json.loads(event_row["details"])}
                for event_row in connection.execute("SELECT * FROM events ORDER BY sequence")
            )

            resource_row = connection.execute(
                "SELECT * FROM resource_profiles WHERE active = 1 "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            resource_profile = dict(resource_row) if resource_row else None
            if resource_profile is not None:
                resource_profile["snapshot"] = json.loads(resource_profile["snapshot"])
                resource_profile["active"] = bool(resource_profile["active"])

            build_row = connection.execute(
                "SELECT * FROM build_state WHERE singleton = 1"
            ).fetchone()
            build_state = dict(build_row) if build_row else None
            if build_state is not None:
                build_state.pop("singleton", None)
                for stored, public in (
                    ("priorities_json", "priorities"),
                    ("targets_json", "targets"),
                    ("floors_json", "floors"),
                ):
                    raw = build_state.pop(stored)
                    build_state[public] = json.loads(raw) if raw else {}

            capability_profile = None
            if project["champion_id"] is not None:
                profile_row = connection.execute(
                    "SELECT cp.*, cs.scale_id, cs.scale_version, "
                    "er.evaluator_id, er.evaluator_version, er.receipt_sha256, "
                    "er.conditions_json, er.measurements_json "
                    "FROM capability_profiles cp "
                    "JOIN capability_scales cs ON cs.scale_hash = cp.scale_hash "
                    "JOIN evaluation_receipts er ON er.receipt_id = cp.receipt_id "
                    "WHERE cp.model_id = ? AND cp.active = 1 "
                    "ORDER BY cp.created_at DESC LIMIT 1",
                    (project["champion_id"],),
                ).fetchone()
                capability_profile = dict(profile_row) if profile_row else None
                if capability_profile is not None:
                    capability_profile["active"] = bool(capability_profile["active"])
                    capability_profile["stats"] = json.loads(
                        capability_profile.pop("stats_json")
                    )
                    capability_profile["conditions"] = json.loads(
                        capability_profile.pop("conditions_json")
                    )
                    capability_profile["measurements"] = json.loads(
                        capability_profile.pop("measurements_json")
                    )

            datasets: list[dict[str, Any]] = []
            for dataset_row in connection.execute(
                "SELECT * FROM datasets WHERE active = 1 ORDER BY created_at, dataset_id"
            ):
                dataset = dict(dataset_row)
                dataset["active"] = bool(dataset["active"])
                dataset["manifest"] = json.loads(dataset.pop("manifest_json"))
                datasets.append(dataset)

            plans: list[dict[str, Any]] = []
            for plan_row in connection.execute(
                "SELECT * FROM plans ORDER BY created_at, plan_id"
            ):
                plan = dict(plan_row)
                plan["budgets"] = json.loads(plan.pop("budgets_json"))
                plan["config"] = json.loads(plan.pop("config_json"))
                plans.append(plan)

            calibrations: list[dict[str, Any]] = []
            for calibration_row in connection.execute(
                "SELECT * FROM calibrations ORDER BY created_at, calibration_id"
            ):
                calibration = dict(calibration_row)
                calibration["feasible"] = bool(calibration["feasible"])
                calibration["backend_result"] = json.loads(
                    calibration.pop("result_json")
                )
                calibrations.append(calibration)

            runs: list[dict[str, Any]] = []
            for run_row in connection.execute(
                "SELECT r.*, a.attempt_token, a.calibration_id, a.request_digest, "
                "a.executor_kind, a.owner_pid, a.worker_pid, a.worker_start_token, "
                "a.liveness_state, a.result_path, a.result_sha256, a.result_json, "
                "a.updated_at AS attempt_updated_at "
                "FROM runs r LEFT JOIN run_attempts a USING(run_id) "
                "ORDER BY r.started_at, r.run_id"
            ):
                runs.append(self._decode_run_row(run_row))

            return ProjectState(
                project,
                champion,
                champion_artifact,
                candidates,
                history,
                resource_profile,
                build_state,
                capability_profile,
                tuple(datasets),
                tuple(plans),
                tuple(calibrations),
                tuple(runs),
            )
