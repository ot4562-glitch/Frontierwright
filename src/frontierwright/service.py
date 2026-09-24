"""Shared application/service layer used by both CLI/JSON and TUI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast
from uuid import uuid4

from frontierwright.artifact_store import (
    seal_training_artifact,
    verify_manifest_digest,
    verify_sealed_artifact,
)
from frontierwright.capability_v1 import (
    CAPABILITY_V1_BUNDLE_ID,
    CAPABILITY_V1_BUNDLE_VERSION,
    CAPABILITY_V1_EVALUATOR_ID,
    CAPABILITY_V1_EVALUATOR_VERSION,
    CAPABILITY_V1_SCALE,
    CAPABILITY_V1_SCORING,
    CAPABILITY_V1_TASKS,
    capability_v1_bundle_hash,
    capability_v1_task_counts,
    capability_v1_uncertainty,
    exact_mcnemar_paired_binary,
)
from frontierwright.data import (
    DatasetClassification,
    DatasetDescriptor,
    DatasetProvenance,
    DatasetRole,
    inspect_local_dataset,
)
from frontierwright.domain import (
    Axis,
    BuildIntent,
    BuildMode,
    BuildTargets,
    CandidateStatus,
    HistoryConfidence,
    LineageRelation,
    ModelFormat,
    ModelOrigin,
    ModelState,
    build_mode,
)
from frontierwright.editions import EditionProfile, policy_for
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import (
    BUILTIN_EVALUATION_PACKS,
    REFERENCE_LM_PACK,
    EvaluationReceipt,
    RawMeasurement,
    apply_scale,
    load_capability_scale,
    load_evaluation_receipt,
)
from frontierwright.evaluator_adapters import (
    ExternalEvaluationImport,
    import_external_evaluation_manifest,
    import_lighteval_results,
    import_lm_eval_results,
)
from frontierwright.execution import (
    BackendDataBoundary,
    CommandBackendSpec,
    HardBudgets,
    PermissionLevel,
    RunStatus,
    RunUsage,
    TrainingPlan,
    backend_allows_dataset,
    compute_execution_request_digest,
    compute_plan_idempotency_key,
    load_command_backend_spec,
    run_calibration_backend,
    run_structured_command,
)
from frontierwright.exporting import (
    portable_export_id_for,
    publish_portable_export,
    verify_portable_export,
)
from frontierwright.fit_planner import FitOpportunityPlan, plan_fit_opportunities
from frontierwright.interventions import (
    assess_training_interventions,
    intervention_by_id,
    intervention_for_training_path,
    intervention_plugins,
)
from frontierwright.lab_adapters import LabAdapterKind, load_lab_adapter_manifest
from frontierwright.local_executor import (
    LocalAttemptSpec,
    atomic_write_json,
    file_sha256,
    launch_worker,
    process_liveness,
    process_start_token,
    terminate_worker_tree,
)
from frontierwright.models import discover_history_evidence, inspect_local_model
from frontierwright.paths import PathAvailability, PathContext, TrainingPathId
from frontierwright.recipes import (
    BUILTIN_DATA_PREPARATION_PLUGINS,
    PREFERENCE_JSONL_PLUGIN_ID,
    SNAPSHOT_COPY_PLUGIN_ID,
    TEXT_LINES_PLUGIN_ID,
    WEIGHTED_TEXT_MIXTURE_PLUGIN_ID,
    DataPreparationPlugin,
    DataPreparationRecipe,
    data_preparation_plugin,
)
from frontierwright.reference_backend import PRESETS, REFERENCE_BACKEND_ID
from frontierwright.reference_tokenizer import (
    DEFAULT_MAX_TRAINING_BYTES,
    TokenizerArtifact,
    load_reference_tokenizer,
    load_tokenizer_payload,
    read_training_bytes,
    tokenizer_fingerprint,
    tokenizer_vocab_size,
    write_tokenizer_artifact,
)
from frontierwright.registry import ProjectState, Registry
from frontierwright.resources import detect_local_resources
from frontierwright.rl import RLExperimentSpec, rl_spec_from_config
from frontierwright.serving_adapters import (
    ServingResourceImport,
    VLLMServingImport,
    import_serving_resource_manifest,
    import_vllm_bench_serve,
)
from frontierwright.workload_evaluations import (
    WORKLOAD_EVAL_BINDING_KIND,
    WorkloadEvaluationBinding,
    aggregate_workload_evaluation_coverage,
    bind_manifest_to_receipt,
    load_workload_evaluation_manifest,
)
from frontierwright.workloads import (
    ParetoDirection,
    ParetoMetricInput,
    WorkloadProfile,
    assess_workload_fit,
    compare_explicit_user_utility,
    compare_pareto_metrics,
    workload_profile_from_payload,
)


@dataclass(frozen=True)
class StatusView:
    schema_version: int = 1
    initialized: bool = False
    project_id: str | None = None
    project_name: str | None = None
    language: str = "en"
    nickname: str | None = None
    edition_profile: str | None = None
    edition_name: str | None = None
    edition_tagline: str | None = None
    edition_starting_point: str | None = None
    origin: str | None = None
    history_confidence: str | None = None
    history_evidence_reason: str | None = None
    measurement_state: str = "NOT_READY"
    build_mode: str | None = None
    strong_recommendation_allowed: bool = False
    champion_model_id: str | None = None
    model_format: str | None = None
    trainable: bool | None = None
    model_fingerprint: str | None = None
    model_source_path: str | None = None
    model_total_bytes: int | None = None
    candidate_count: int = 0
    stats: dict[str, float | None] = field(default_factory=dict)
    stat_uncertainty: dict[str, dict[str, object]] = field(default_factory=dict)
    resource_profile_available: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BirthView:
    schema_version: int = 1
    born: bool = False
    model_id: str | None = None
    model_fingerprint: str | None = None
    checkpoint: str | None = None
    preset: str | None = None
    seed: int | None = None
    backend_id: str | None = None
    parameter_count: int | None = None
    tokenizer_artifact_id: str | None = None
    tokenizer_fingerprint: str | None = None
    vocab_size: int | None = None
    runtime: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerView:
    schema_version: int = 1
    intervention_id: str = "frontierwright.birth.train-tokenizer"
    intervention_version: str = "1"
    artifact_id: str | None = None
    fingerprint: str | None = None
    path: str | None = None
    source_dataset_id: str | None = None
    source_dataset_fingerprint: str | None = None
    requested_vocab_size: int | None = None
    vocab_size: int | None = None
    max_training_bytes: int | None = None
    training_bytes: int | None = None
    merge_count: int | None = None
    replayed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizersView:
    schema_version: int = 1
    tokenizers: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceView:
    schema_version: int = 1
    available: bool = False
    profile_id: str | None = None
    profile_name: str | None = None
    provenance: str | None = None
    detected_at: str | None = None
    snapshot: dict[str, object] = field(default_factory=dict)
    headroom: dict[str, object] = field(default_factory=dict)
    model_fit: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadView:
    schema_version: int = 1
    configured: bool = False
    profile_id: str | None = None
    profile_hash: str | None = None
    profile_name: str | None = None
    source: str | None = None
    created_at: str | None = None
    profile: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadFitView:
    schema_version: int = 1
    configured: bool = False
    model_id: str | None = None
    model_fingerprint: str | None = None
    workload_profile_id: str | None = None
    workload_profile_hash: str | None = None
    overall_status: str = "NOT_CONFIGURED"
    counts: dict[str, int] = field(default_factory=dict)
    constraints: list[dict[str, object]] = field(default_factory=list)
    workload_evaluation_coverage: dict[str, object] = field(default_factory=dict)
    synthetic_utility_score: None = None
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BuildView:
    schema_version: int = 1
    mode: str | None = None
    configured: bool = False
    archetype: str | None = None
    priorities: dict[str, int] = field(default_factory=dict)
    targets: dict[str, int] = field(default_factory=dict)
    floors: dict[str, int] = field(default_factory=dict)
    scale_bound: bool = False
    scale_hash: str | None = None
    scale_id: str | None = None
    scale_version: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StatsView:
    schema_version: int = 1
    measured: bool = False
    model_id: str | None = None
    model_fingerprint: str | None = None
    profile_id: str | None = None
    scale_id: str | None = None
    scale_version: str | None = None
    scale_hash: str | None = None
    receipt_id: str | None = None
    receipt_sha256: str | None = None
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    stats: dict[str, float | None] = field(default_factory=dict)
    raw_measurements: list[dict[str, object]] = field(default_factory=list)
    conditions: dict[str, object] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRunView:
    schema_version: int = 1
    pack_id: str | None = None
    pack_version: str | None = None
    receipt_id: str | None = None
    receipt_sha256: str | None = None
    model_id: str | None = None
    model_fingerprint: str | None = None
    dataset_id: str | None = None
    dataset_fingerprint: str | None = None
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    replayed: bool = False
    measurements: list[dict[str, object]] = field(default_factory=list)
    conditions: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CapabilityV1RunView:
    schema_version: int = 1
    bundle_id: str = CAPABILITY_V1_BUNDLE_ID
    bundle_version: str = CAPABILITY_V1_BUNDLE_VERSION
    bundle_hash: str | None = None
    scale_id: str | None = None
    scale_version: str | None = None
    scale_hash: str | None = None
    model_id: str | None = None
    model_fingerprint: str | None = None
    receipt_id: str | None = None
    receipt_sha256: str | None = None
    replayed: bool = False
    task_counts: dict[str, int] = field(default_factory=dict)
    axis_results: list[dict[str, object]] = field(default_factory=list)
    stats: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationCompareView:
    schema_version: int = 1
    pack_id: str | None = None
    pack_version: str | None = None
    dataset_id: str | None = None
    dataset_fingerprint: str | None = None
    champion_model_id: str | None = None
    candidate_model_id: str | None = None
    candidate_status: str | None = None
    comparable: bool = False
    reason: str | None = None
    champion_receipt_id: str | None = None
    candidate_receipt_id: str | None = None
    measurements: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DataView:
    schema_version: int = 1
    datasets: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PathsView:
    schema_version: int = 1
    paths: list[dict[str, object]] = field(default_factory=list)
    recommended_path: str | None = None
    recommendation_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InterventionsView:
    schema_version: int = 1
    interventions: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LabAdaptersView:
    schema_version: int = 1
    adapters: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlanView:
    schema_version: int = 1
    plan_id: str | None = None
    path_id: str | None = None
    intervention_id: str | None = None
    intervention_version: str | None = None
    intervention_family: str | None = None
    backend_id: str | None = None
    backend_spec_hash: str | None = None
    model_id: str | None = None
    dataset_id: str | None = None
    dataset_recipe_id: str | None = None
    dataset_recipe_hash: str | None = None
    dataset_classification: str | None = None
    backend_data_boundary: str | None = None
    backend_adapter_ref: str | None = None
    backend_adapter_hash: str | None = None
    resource_profile_id: str | None = None
    permission: str | None = None
    budgets: dict[str, object] = field(default_factory=dict)
    budget_enforcement: dict[str, str] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)
    idempotency_key: str | None = None
    calibration: dict[str, object] | None = None
    ready: bool = False
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunView:
    schema_version: int = 1
    run_id: str | None = None
    plan_id: str | None = None
    status: str | None = None
    candidate_model_id: str | None = None
    metrics: dict[str, object] = field(default_factory=dict)
    usage: dict[str, object] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    dry_run: bool = False
    liveness_state: str | None = None
    calibration_id: str | None = None
    request_digest: str | None = None
    result_evidence_available: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionPreflightView:
    schema_version: int = 1
    action: str = ""
    ready: bool = False
    would_replay: bool = False
    blockers: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateView:
    schema_version: int = 1
    candidates: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MergeView:
    schema_version: int = 1
    intervention_id: str = "frontierwright.evolve.linear-merge"
    intervention_version: str = "1"
    transform_id: str | None = None
    replayed: bool = False
    primary_model_id: str | None = None
    other_model_id: str | None = None
    primary_weight: float | None = None
    other_weight: float | None = None
    candidate_model_id: str | None = None
    model_fingerprint: str | None = None
    checkpoint: str | None = None
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuantizeView:
    schema_version: int = 1
    intervention_id: str = "frontierwright.optimize.symmetric-int8"
    intervention_version: str = "1"
    transform_id: str | None = None
    replayed: bool = False
    source_model_id: str | None = None
    candidate_model_id: str | None = None
    model_fingerprint: str | None = None
    checkpoint: str | None = None
    model_format: str | None = None
    trainable: bool | None = None
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExportView:
    schema_version: int = 1
    intervention_id: str = "frontierwright.operate.portable-export"
    intervention_version: str = "1"
    export_id: str | None = None
    replayed: bool = False
    model_id: str | None = None
    model_fingerprint: str | None = None
    model_format: str | None = None
    trainable: bool | None = None
    destination: str | None = None
    model_path: str | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExportVerifyView:
    schema_version: int = 1
    valid: bool = True
    export_id: str | None = None
    model_fingerprint: str | None = None
    model_format: str | None = None
    trainable: bool | None = None
    destination: str | None = None
    model_path: str | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    authenticity: str = "NOT_SIGNED"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationView:
    schema_version: int = 1
    intervention_id: str = "frontierwright.operate.generate-reference"
    intervention_version: str = "1"
    model_id: str | None = None
    model_fingerprint: str | None = None
    model_format: str | None = None
    prompt: str = ""
    continuation_text: str = ""
    generated_text: str = ""
    generated_token_ids: list[int] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceProfileView:
    schema_version: int = 1
    intervention_id: str = "frontierwright.operate.profile-reference"
    intervention_version: str = "1"
    model_id: str | None = None
    model_fingerprint: str | None = None
    model_format: str | None = None
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompareView:
    schema_version: int = 1
    champion_model_id: str | None = None
    candidate_model_id: str | None = None
    candidate_status: str | None = None
    scale_comparable: bool = False
    scale_reason: str | None = None
    champion_stats: dict[str, float | None] = field(default_factory=dict)
    candidate_stats: dict[str, float | None] = field(default_factory=dict)
    deltas: dict[str, float | None] = field(default_factory=dict)
    build_constraints: list[dict[str, object]] = field(default_factory=list)
    promotion_eligible: bool = False
    promotion_blockers: list[dict[str, object]] = field(default_factory=list)
    build_scale_hash: str | None = None
    raw_evaluation_comparisons: list[dict[str, object]] = field(default_factory=list)
    paired_capability_evidence: dict[str, object] = field(default_factory=dict)
    workload_comparison: dict[str, object] = field(default_factory=dict)
    pareto: dict[str, object] = field(default_factory=dict)
    run: dict[str, object] | None = None
    calibration: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HistoryView:
    schema_version: int = 1
    events: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _empty_stats() -> dict[str, float | None]:
    return {
        "general": None,
        "reasoning": None,
        "math": None,
        "coding": None,
    }


def _capability_uncertainty_from_profile(
    profile: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    if profile is None:
        return {}
    conditions = profile.get("conditions")
    if not isinstance(conditions, dict):
        return {}
    raw_axis_results = conditions.get("axis_results")
    if not isinstance(raw_axis_results, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for raw in raw_axis_results:
        if not isinstance(raw, dict):
            continue
        axis = raw.get("axis")
        correct = raw.get("correct")
        total = raw.get("total")
        if not isinstance(axis, str):
            continue
        uncertainty = raw.get("uncertainty")
        if isinstance(uncertainty, dict):
            result[axis] = dict(uncertainty)
            continue
        if (
            isinstance(correct, int)
            and not isinstance(correct, bool)
            and isinstance(total, int)
            and not isinstance(total, bool)
            and total > 0
            and 0 <= correct <= total
        ):
            result[axis] = capability_v1_uncertainty(correct, total)
    return result


def get_status(root: Path) -> StatusView:
    registry = Registry(root)
    if not registry.exists:
        return StatusView(stats=_empty_stats())

    state = registry.read()
    project = state.project
    origin = ModelOrigin(project["origin"])
    confidence = HistoryConfidence(project["history_confidence"])
    edition = EditionProfile(project["edition_profile"])
    edition_policy = policy_for(edition)
    stats = _empty_stats()
    if state.capability_profile is not None:
        for stat in state.capability_profile.get("stats", []):
            axis = str(stat.get("axis", "")).lower()
            value = stat.get("value")
            if axis in stats and isinstance(value, (int, float)) and not isinstance(value, bool):
                stats[axis] = float(value)
    elif state.champion is not None:
        for stat in state.champion.model.stats:
            stats[stat.axis.value.lower()] = stat.value

    measured = any(value is not None for value in stats.values())
    if measured:
        mode = BuildMode.TARGETS_FLOORS
    else:
        mode = build_mode(origin, state.champion)
    artifact = state.champion_artifact or {}

    return StatusView(
        initialized=True,
        project_id=project["project_id"],
        project_name=project["name"],
        language=project["language"],
        nickname=project["name"],
        edition_profile=edition.value,
        edition_name=edition_policy.display_name,
        edition_tagline=edition_policy.tagline,
        edition_starting_point=edition_policy.starting_point,
        origin=origin.value,
        history_confidence=confidence.value,
        history_evidence_reason=artifact.get("evidence_reason"),
        measurement_state="MEASURED" if measured else "NOT_READY",
        build_mode=mode.value,
        strong_recommendation_allowed=confidence.allows_recommendation,
        champion_model_id=state.champion.model.model_id if state.champion else None,
        model_format=(
            state.champion.model.model_format.value if state.champion is not None else None
        ),
        trainable=state.champion.model.trainable if state.champion is not None else None,
        model_fingerprint=(
            state.champion.model.fingerprint if state.champion is not None else None
        ),
        model_source_path=artifact.get("source_path"),
        model_total_bytes=artifact.get("total_bytes"),
        candidate_count=len(state.candidates),
        stats=stats,
        stat_uncertainty=_capability_uncertainty_from_profile(state.capability_profile),
        resource_profile_available=state.resource_profile is not None,
    )


def initialize_project(
    root: Path,
    *,
    name: str,
    origin: ModelOrigin,
    language: str = "en",
    edition_profile: EditionProfile | None = None,
) -> StatusView:
    registry = Registry(root)
    registry.initialize(
        name,
        origin,
        language=language,
        edition_profile=edition_profile,
    )
    return get_status(root)


def set_project_edition(root: Path, edition: EditionProfile) -> StatusView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before changing its edition profile.",
            10,
        )
    registry.set_edition_profile(edition)
    return get_status(root)


def get_lab_adapters(root: Path) -> LabAdaptersView:
    registry = Registry(root)
    if not registry.exists:
        return LabAdaptersView()

    adapters = []
    for item in registry.list_lab_adapters():
        adapters.append(
            {
                "adapter_ref": item["adapter_ref"],
                "adapter_id": item["adapter_id"],
                "adapter_version": item["adapter_version"],
                "display_name": item["display_name"],
                "manifest_hash": item["manifest_hash"],
                "source_path": item["source_path"],
                "data_boundary": item["data_boundary"],
                "network_scope": item["network_scope"],
                "kinds": item["kinds"],
                "capabilities": item["capabilities"],
                "connected_at": item["connected_at"],
            }
        )
    return LabAdaptersView(adapters=adapters)


def connect_lab_adapter(root: Path, manifest_path: Path) -> LabAdaptersView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before connecting a Lab adapter.",
            10,
        )

    state = registry.read()
    if EditionProfile(state.project["edition_profile"]) is not EditionProfile.LAB:
        raise FrontierwrightError(
            "LAB_EDITION_REQUIRED",
            "Private infrastructure adapters can only be connected in Frontierwright Lab.",
            12,
        )

    manifest = load_lab_adapter_manifest(manifest_path.expanduser().resolve())
    registry.register_lab_adapter(
        manifest,
        source_path=manifest_path.expanduser().resolve(),
    )
    return get_lab_adapters(root)


def disconnect_lab_adapter(root: Path, adapter_ref: str) -> LabAdaptersView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before disconnecting a Lab adapter.",
            10,
        )
    registry.disconnect_lab_adapter(adapter_ref)
    return get_lab_adapters(root)


def get_birth_view(root: Path) -> BirthView:
    registry = Registry(root)
    if not registry.exists:
        return BirthView()
    state = registry.read()
    if state.champion is None:
        return BirthView()

    birth = registry.get_model_birth(state.champion.model.model_id)
    if birth is None:
        return BirthView()

    runtime = birth.get("runtime")
    runtime_map = runtime if isinstance(runtime, dict) else {}
    parameter_count = runtime_map.get("parameter_count")
    return BirthView(
        born=True,
        model_id=state.champion.model.model_id,
        model_fingerprint=state.champion.model.fingerprint,
        checkpoint=state.champion.model.checkpoint,
        preset=str(birth["preset"]),
        seed=int(birth["seed"]),
        backend_id=str(birth["backend_id"]),
        parameter_count=(
            parameter_count
            if isinstance(parameter_count, int) and not isinstance(parameter_count, bool)
            else None
        ),
        tokenizer_artifact_id=(
            str(birth["tokenizer_artifact_id"])
            if birth.get("tokenizer_artifact_id") is not None
            else None
        ),
        tokenizer_fingerprint=(
            str(birth["tokenizer_fingerprint"])
            if birth.get("tokenizer_fingerprint") is not None
            else None
        ),
        vocab_size=(
            int(runtime_map["vocab_size"])
            if isinstance(runtime_map.get("vocab_size"), int)
            and not isinstance(runtime_map.get("vocab_size"), bool)
            else None
        ),
        runtime=dict(runtime_map),
    )


def _tokenizer_view_from_record(
    record: dict[str, object],
    *,
    replayed: bool,
) -> TokenizerView:
    path = Path(str(record["path"])).expanduser().resolve()
    payload = load_tokenizer_payload(path)
    fingerprint = tokenizer_fingerprint(payload)
    if fingerprint != str(record["fingerprint"]):
        raise FrontierwrightError(
            "TOKENIZER_ARTIFACT_TAMPERED",
            "Tokenizer artifact bytes no longer match the registry fingerprint.",
            13,
        )
    merges = payload.get("merges")
    if not isinstance(merges, list):
        raise FrontierwrightError(
            "TOKENIZER_ARTIFACT_INVALID",
            "Tokenizer artifact merges must be a list.",
            13,
        )
    numeric_fields: dict[str, int] = {}
    for key in (
        "requested_vocab_size",
        "vocab_size",
        "max_training_bytes",
        "training_bytes",
    ):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FrontierwrightError(
                "TOKENIZER_REGISTRY_INVALID",
                f"Tokenizer registry field {key} must be a positive integer.",
                4,
            )
        numeric_fields[key] = value

    return TokenizerView(
        artifact_id=str(record["artifact_id"]),
        fingerprint=str(record["fingerprint"]),
        path=str(path),
        source_dataset_id=str(record["source_dataset_id"]),
        source_dataset_fingerprint=str(record["source_dataset_fingerprint"]),
        requested_vocab_size=numeric_fields["requested_vocab_size"],
        vocab_size=numeric_fields["vocab_size"],
        max_training_bytes=numeric_fields["max_training_bytes"],
        training_bytes=numeric_fields["training_bytes"],
        merge_count=len(merges),
        replayed=replayed,
    )


def get_tokenizers_view(root: Path) -> TokenizersView:
    registry = Registry(root)
    if not registry.exists:
        return TokenizersView()
    items = [
        _tokenizer_view_from_record(record, replayed=True).to_dict()
        for record in registry.list_tokenizer_artifacts()
    ]
    return TokenizersView(tokenizers=items)


def _tokenizer_birth_context(
    root: Path,
    *,
    dataset_id: str,
    vocab_size: int,
    max_training_bytes: int,
) -> tuple[Registry, dict[str, object], Path, str, bytes, dict[str, object] | None]:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a ZERO-origin Frontierwright project before tokenizer birth.",
            10,
        )
    state = registry.read()
    if ModelOrigin(state.project["origin"]) is not ModelOrigin.ZERO:
        raise FrontierwrightError(
            "TOKENIZER_BIRTH_ORIGIN_MISMATCH",
            "Tokenizer birth is only valid for ZERO-origin projects.",
            13,
        )
    if state.champion is not None:
        raise FrontierwrightError(
            "TOKENIZER_AFTER_MODEL_BIRTH",
            "Train/select the tokenizer before materializing the zero-model root.",
            13,
        )
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size < 256
        or vocab_size > 65_536
    ):
        raise FrontierwrightError(
            "TOKENIZER_CONFIG_INVALID",
            "vocab_size must be an integer between 256 and 65536.",
            2,
        )

    dataset = next(
        (item for item in state.datasets if item["dataset_id"] == dataset_id),
        None,
    )
    if dataset is None:
        raise FrontierwrightError(
            "TOKENIZER_SOURCE_DATASET_NOT_FOUND",
            "Tokenizer source dataset is not active in this project.",
            12,
        )
    if dataset["role"] != DatasetRole.PRETRAIN.value:
        raise FrontierwrightError(
            "TOKENIZER_SOURCE_ROLE_INVALID",
            "Tokenizer birth requires a PRETRAIN dataset.",
            12,
        )

    intervention = intervention_by_id("frontierwright.birth.train-tokenizer")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in tokenizer birth intervention is unavailable.",
            4,
        )

    source_path = Path(str(dataset["source_path"])).expanduser().resolve()
    descriptor = inspect_local_dataset(source_path)
    source_fingerprint = str(dataset["fingerprint"])
    if descriptor.fingerprint != source_fingerprint:
        raise FrontierwrightError(
            "TOKENIZER_SOURCE_DATASET_DRIFT",
            "Tokenizer source dataset bytes no longer match the registered fingerprint.",
            13,
        )

    corpus = read_training_bytes(
        source_path,
        max_training_bytes=max_training_bytes,
    )
    existing = next(
        (
            item
            for item in registry.list_tokenizer_artifacts()
            if item.get("source_dataset_id") == dataset_id
            and item.get("source_dataset_fingerprint") == source_fingerprint
            and item.get("requested_vocab_size") == vocab_size
            and item.get("max_training_bytes") == max_training_bytes
            and item.get("training_bytes") == len(corpus)
        ),
        None,
    )
    return registry, dataset, source_path, source_fingerprint, corpus, existing


def preflight_project_tokenizer(
    root: Path,
    *,
    dataset_id: str,
    vocab_size: int = 512,
    max_training_bytes: int = DEFAULT_MAX_TRAINING_BYTES,
) -> ActionPreflightView:
    _, _, source_path, source_fingerprint, corpus, existing = _tokenizer_birth_context(
        root,
        dataset_id=dataset_id,
        vocab_size=vocab_size,
        max_training_bytes=max_training_bytes,
    )
    return ActionPreflightView(
        action="birth-tokenizer",
        ready=True,
        would_replay=existing is not None,
        details={
            "dataset_id": dataset_id,
            "dataset_fingerprint": source_fingerprint,
            "source_path": str(source_path),
            "requested_vocab_size": vocab_size,
            "max_training_bytes": max_training_bytes,
            "training_bytes": len(corpus),
            "existing_artifact_id": (existing.get("artifact_id") if existing is not None else None),
        },
    )


def train_project_tokenizer(
    root: Path,
    *,
    dataset_id: str,
    vocab_size: int = 512,
    max_training_bytes: int = DEFAULT_MAX_TRAINING_BYTES,
) -> TokenizerView:
    registry, _, _, source_fingerprint, corpus, existing = _tokenizer_birth_context(
        root,
        dataset_id=dataset_id,
        vocab_size=vocab_size,
        max_training_bytes=max_training_bytes,
    )
    if existing is not None:
        return _tokenizer_view_from_record(existing, replayed=True)

    tokenizers_root = registry.state_dir / "tokenizers"
    staging_root = tokenizers_root / ".staging" / uuid4().hex
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        staged = write_tokenizer_artifact(
            staging_root,
            source_dataset_id=dataset_id,
            source_dataset_fingerprint=source_fingerprint,
            requested_vocab_size=vocab_size,
            max_training_bytes=max_training_bytes,
            corpus=corpus,
        )
        existing_by_id = registry.get_tokenizer_artifact(staged.artifact_id)
        if existing_by_id is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
            return _tokenizer_view_from_record(existing_by_id, replayed=True)

        final_root = tokenizers_root / staged.artifact_id
        final_path = final_root / "tokenizer.json"
        if final_root.exists():
            payload = load_tokenizer_payload(final_path)
            if tokenizer_fingerprint(payload) != staged.fingerprint:
                raise FrontierwrightError(
                    "TOKENIZER_ARTIFACT_CONFLICT",
                    "Managed tokenizer destination contains different bytes.",
                    13,
                )
            shutil.rmtree(staging_root, ignore_errors=True)
        else:
            final_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_root, final_root)

        final_artifact = TokenizerArtifact(
            artifact_id=staged.artifact_id,
            fingerprint=staged.fingerprint,
            path=final_path.resolve(),
            source_dataset_id=staged.source_dataset_id,
            source_dataset_fingerprint=staged.source_dataset_fingerprint,
            requested_vocab_size=staged.requested_vocab_size,
            vocab_size=staged.vocab_size,
            max_training_bytes=staged.max_training_bytes,
            training_bytes=staged.training_bytes,
            merges=staged.merges,
        )
        registry.register_tokenizer_artifact(final_artifact)
        record = registry.get_tokenizer_artifact(final_artifact.artifact_id)
        if record is None:
            raise FrontierwrightError(
                "TOKENIZER_REGISTRY_ERROR",
                "Tokenizer artifact was not persisted in the registry.",
                4,
            )
        return _tokenizer_view_from_record(record, replayed=False)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise


def preflight_zero_birth(
    root: Path,
    *,
    preset: str,
    seed: int,
    python_executable: str,
    tokenizer_artifact_id: str | None = None,
    timeout_seconds: float = 300.0,
) -> ActionPreflightView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a ZERO-origin Frontierwright project before model birth.",
            10,
        )
    if preset not in PRESETS:
        raise FrontierwrightError(
            "BIRTH_PRESET_UNKNOWN",
            f"Unknown zero-model preset: {preset}",
            2,
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FrontierwrightError(
            "BIRTH_SEED_INVALID",
            "Birth seed must be a nonnegative integer.",
            2,
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise FrontierwrightError(
            "INVALID_TIMEOUT",
            "Birth timeout must be a positive finite number.",
            2,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Training Python executable must be nonempty.",
            2,
        )

    state = registry.read()
    if ModelOrigin(state.project["origin"]) is not ModelOrigin.ZERO:
        raise FrontierwrightError(
            "BIRTH_ORIGIN_MISMATCH",
            "Model birth is only valid for ZERO-origin projects.",
            13,
        )

    tokenizer_fingerprint: str | None = None
    tokenizer_vocab_size: int | None = None
    if tokenizer_artifact_id is not None:
        if not tokenizer_artifact_id.strip():
            raise FrontierwrightError(
                "BIRTH_TOKENIZER_INVALID",
                "Tokenizer artifact ID must be nonempty when supplied.",
                2,
            )
        tokenizer = next(
            (
                item
                for item in get_tokenizers_view(root).tokenizers
                if item.get("artifact_id") == tokenizer_artifact_id
            ),
            None,
        )
        if tokenizer is None:
            raise FrontierwrightError(
                "BIRTH_TOKENIZER_NOT_FOUND",
                "Selected tokenizer artifact is not registered in this project.",
                12,
            )
        raw_fingerprint = tokenizer.get("fingerprint")
        raw_vocab_size = tokenizer.get("vocab_size")
        if not isinstance(raw_fingerprint, str) or not raw_fingerprint:
            raise FrontierwrightError(
                "BIRTH_TOKENIZER_INVALID",
                "Selected tokenizer artifact has no stable fingerprint.",
                4,
            )
        if (
            isinstance(raw_vocab_size, bool)
            or not isinstance(raw_vocab_size, int)
            or raw_vocab_size <= 0
        ):
            raise FrontierwrightError(
                "BIRTH_TOKENIZER_INVALID",
                "Selected tokenizer artifact has no valid vocabulary size.",
                4,
            )
        tokenizer_fingerprint = raw_fingerprint
        tokenizer_vocab_size = raw_vocab_size

    existing_model_id: str | None = None
    would_replay = False
    if state.champion is not None:
        existing_model_id = state.champion.model.model_id
        existing = registry.get_model_birth(existing_model_id)
        would_replay = bool(
            existing is not None
            and existing.get("preset") == preset
            and existing.get("seed") == seed
            and existing.get("tokenizer_artifact_id") == tokenizer_artifact_id
            and existing.get("tokenizer_fingerprint") == tokenizer_fingerprint
        )
        if not would_replay:
            raise FrontierwrightError(
                "MODEL_ALREADY_BORN",
                "Project already has a materialized current model.",
                13,
            )

    return ActionPreflightView(
        action="birth-zero",
        ready=True,
        would_replay=would_replay,
        details={
            "preset": preset,
            "seed": seed,
            "python_executable": python_executable,
            "timeout_seconds": float(timeout_seconds),
            "tokenizer_artifact_id": tokenizer_artifact_id,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "existing_model_id": existing_model_id,
        },
    )


def birth_zero_model(
    root: Path,
    *,
    preset: str,
    seed: int,
    python_executable: str,
    tokenizer_artifact_id: str | None = None,
    timeout_seconds: float = 300.0,
) -> BirthView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a ZERO-origin Frontierwright project before model birth.",
            10,
        )
    if preset not in PRESETS:
        raise FrontierwrightError(
            "BIRTH_PRESET_UNKNOWN",
            f"Unknown zero-model preset: {preset}",
            2,
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FrontierwrightError(
            "BIRTH_SEED_INVALID",
            "Birth seed must be a nonnegative integer.",
            2,
        )
    if timeout_seconds <= 0:
        raise FrontierwrightError(
            "INVALID_TIMEOUT",
            "Birth timeout must be positive.",
            2,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Training Python executable must be nonempty.",
            2,
        )

    state = registry.read()
    if ModelOrigin(state.project["origin"]) is not ModelOrigin.ZERO:
        raise FrontierwrightError(
            "BIRTH_ORIGIN_MISMATCH",
            "Model birth is only valid for ZERO-origin projects.",
            13,
        )

    selected_tokenizer: TokenizerView | None = None
    if tokenizer_artifact_id is not None:
        if not tokenizer_artifact_id.strip():
            raise FrontierwrightError(
                "BIRTH_TOKENIZER_INVALID",
                "Tokenizer artifact ID must be nonempty when supplied.",
                2,
            )
        tokenizer_record = registry.get_tokenizer_artifact(tokenizer_artifact_id)
        if tokenizer_record is None:
            raise FrontierwrightError(
                "BIRTH_TOKENIZER_NOT_FOUND",
                "Selected tokenizer artifact is not registered in this project.",
                12,
            )
        selected_tokenizer = _tokenizer_view_from_record(
            tokenizer_record,
            replayed=True,
        )

    selected_tokenizer_fingerprint = (
        selected_tokenizer.fingerprint if selected_tokenizer is not None else None
    )
    selected_tokenizer_path = selected_tokenizer.path if selected_tokenizer is not None else None
    selected_vocab_size = selected_tokenizer.vocab_size if selected_tokenizer is not None else None

    if state.champion is not None:
        existing = registry.get_model_birth(state.champion.model.model_id)
        if (
            existing is not None
            and existing.get("preset") == preset
            and existing.get("seed") == seed
            and existing.get("tokenizer_artifact_id") == tokenizer_artifact_id
            and existing.get("tokenizer_fingerprint") == selected_tokenizer_fingerprint
        ):
            return get_birth_view(root)
        raise FrontierwrightError(
            "MODEL_ALREADY_BORN",
            "Project already has a materialized current model.",
            13,
        )

    birth_token = uuid4().hex
    births_root = registry.state_dir / "births"
    staging_root = births_root / ".staging" / birth_token
    request_path = births_root / "requests" / f"{birth_token}.json"
    staging_root.mkdir(parents=True, exist_ok=False)

    request: dict[str, object] = {
        "schema_version": 1,
        "backend_id": REFERENCE_BACKEND_ID,
        "operation": "birth",
        "preset": preset,
        "seed": seed,
        "output_root": str(staging_root.resolve()),
    }
    if selected_tokenizer_path is not None:
        request["tokenizer_path"] = selected_tokenizer_path
        request["tokenizer_fingerprint"] = selected_tokenizer_fingerprint
    _write_state_json(request_path, request)

    try:
        result = run_structured_command(
            (
                python_executable,
                "-m",
                "frontierwright.reference_backend",
                "{request_json}",
            ),
            environment_overrides={"PYTHONUNBUFFERED": "1"},
            request_path=request_path,
            timeout_seconds=timeout_seconds,
        )
        if result.get("operation") != "birth":
            raise FrontierwrightError(
                "BIRTH_RESULT_INVALID",
                "Birth backend did not return a birth result.",
                14,
            )
        output_raw = result.get("output_model_path")
        metrics = result.get("metrics")
        if not isinstance(output_raw, str) or not output_raw:
            raise FrontierwrightError(
                "BIRTH_RESULT_INVALID",
                "Birth backend result lacks output_model_path.",
                14,
            )
        if not isinstance(metrics, dict):
            raise FrontierwrightError(
                "BIRTH_RESULT_INVALID",
                "Birth backend result metrics must be an object.",
                14,
            )
        if metrics.get("preset") != preset or metrics.get("seed") != seed:
            raise FrontierwrightError(
                "BIRTH_RESULT_INVALID",
                "Birth backend result does not match the requested preset/seed.",
                14,
            )
        if selected_tokenizer is not None:
            if metrics.get("tokenizer_fingerprint") != selected_tokenizer_fingerprint:
                raise FrontierwrightError(
                    "BIRTH_TOKENIZER_MISMATCH",
                    "Birth backend did not bind the selected tokenizer fingerprint.",
                    14,
                )
            if metrics.get("vocab_size") != selected_vocab_size:
                raise FrontierwrightError(
                    "BIRTH_TOKENIZER_MISMATCH",
                    "Birth backend vocab size does not match the selected tokenizer.",
                    14,
                )

        backend_model_path = Path(output_raw).expanduser().resolve()
        try:
            backend_model_path.relative_to(staging_root.resolve())
        except ValueError as exc:
            raise FrontierwrightError(
                "BIRTH_OUTPUT_ESCAPE",
                "Birth backend output escaped the managed staging directory.",
                14,
            ) from exc

        staged_descriptor = inspect_local_model(backend_model_path)
        digest = hashlib.sha256(
            (str(state.project["project_id"]) + "\0" + staged_descriptor.fingerprint).encode(
                "utf-8"
            )
        ).hexdigest()
        model_id = f"model-birth-{digest[:32]}"
        final_root = births_root / model_id

        if final_root.exists():
            final_model_path = final_root / "model"
            final_descriptor = inspect_local_model(final_model_path)
            if final_descriptor.fingerprint != staged_descriptor.fingerprint:
                raise FrontierwrightError(
                    "BIRTH_ARTIFACT_CONFLICT",
                    "Existing birth artifact differs from the newly materialized root.",
                    13,
                )
            shutil.rmtree(staging_root, ignore_errors=True)
        else:
            final_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_root, final_root)
            final_model_path = final_root / "model"
            final_descriptor = inspect_local_model(final_model_path)
            if final_descriptor.fingerprint != staged_descriptor.fingerprint:
                raise FrontierwrightError(
                    "BIRTH_ARTIFACT_MISMATCH",
                    "Materialized root fingerprint changed during publication.",
                    14,
                )

        published_tokenizer_path = final_descriptor.source_path / "tokenizer.json"
        published_tokenizer = load_reference_tokenizer(published_tokenizer_path)
        published_tokenizer_fingerprint = tokenizer_fingerprint(published_tokenizer)
        published_vocab_size = tokenizer_vocab_size(published_tokenizer)
        if selected_tokenizer is not None:
            if published_tokenizer_fingerprint != selected_tokenizer_fingerprint:
                raise FrontierwrightError(
                    "BIRTH_TOKENIZER_MISMATCH",
                    "Published root tokenizer differs from the selected artifact.",
                    14,
                )
            if published_vocab_size != selected_vocab_size:
                raise FrontierwrightError(
                    "BIRTH_TOKENIZER_MISMATCH",
                    "Published root tokenizer vocab size differs from the selected artifact.",
                    14,
                )

        model = ModelState(
            model_id=model_id,
            identity_id=str(state.project["identity_id"]),
            origin=ModelOrigin.ZERO,
            checkpoint=str(final_descriptor.source_path),
            fingerprint=final_descriptor.fingerprint,
            parent_model_id=None,
            stats=(),
            model_format=final_descriptor.model_format,
            trainable=final_descriptor.trainable,
        )
        registry.register_birth_model(
            model,
            final_descriptor,
            preset=preset,
            seed=seed,
            backend_id=REFERENCE_BACKEND_ID,
            runtime=dict(metrics),
            tokenizer_artifact_id=tokenizer_artifact_id,
            tokenizer_fingerprint=selected_tokenizer_fingerprint,
        )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return get_birth_view(root)


def _merge_view_from_candidate(
    registry: Registry,
    *,
    transform_id: str,
    candidate_model_id: str,
    primary_model_id: str,
    other_model_id: str,
    primary_weight: float,
    other_weight: float,
    replayed: bool,
) -> MergeView:
    candidate = registry.get_candidate(candidate_model_id)
    _verify_model_artifact_integrity(registry, candidate_model_id)
    metrics: dict[str, object] = {}
    metadata_path = Path(candidate.model.checkpoint) / "frontierwright-transform.json"
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raw = {}
    if isinstance(raw, dict):
        metrics = dict(raw)
    return MergeView(
        transform_id=transform_id,
        replayed=replayed,
        primary_model_id=primary_model_id,
        other_model_id=other_model_id,
        primary_weight=primary_weight,
        other_weight=other_weight,
        candidate_model_id=candidate.model.model_id,
        model_fingerprint=candidate.model.fingerprint,
        checkpoint=candidate.model.checkpoint,
        metrics=metrics,
    )


def preflight_merge_reference_models(
    root: Path,
    *,
    other_model_id: str,
    other_weight: float,
    python_executable: str,
    timeout_seconds: float = 300.0,
) -> ActionPreflightView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before merging models.",
            10,
        )
    if (
        isinstance(other_weight, bool)
        or not isinstance(other_weight, (int, float))
        or not math.isfinite(float(other_weight))
        or not 0.0 < float(other_weight) < 1.0
    ):
        raise FrontierwrightError(
            "MERGE_WEIGHT_INVALID",
            "Merge other-model weight must be strictly between 0 and 1.",
            2,
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise FrontierwrightError("INVALID_TIMEOUT", "Merge timeout must be positive.", 2)
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Training Python executable must be nonempty.",
            2,
        )

    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required before model merge.",
            12,
        )
    primary = state.champion.model
    other = registry.get_model(other_model_id)
    if other.model_id == primary.model_id:
        raise FrontierwrightError(
            "MERGE_PARENT_CONFLICT",
            "Merge requires two different model identities.",
            2,
        )
    if other.identity_id != primary.identity_id:
        raise FrontierwrightError(
            "IDENTITY_MISMATCH",
            "Merge parents must belong to the same Frontierwright identity.",
            13,
        )
    _verify_model_artifact_integrity(registry, primary.model_id)
    _verify_model_artifact_integrity(registry, other.model_id)

    intervention = intervention_by_id("frontierwright.evolve.linear-merge")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in linear merge intervention is unavailable.",
            4,
        )
    other_weight_value = float(other_weight)
    primary_weight = 1.0 - other_weight_value
    contract: dict[str, object] = {
        "intervention_id": intervention.intervention_id,
        "intervention_version": intervention.version,
        "primary": {
            "model_id": primary.model_id,
            "fingerprint": primary.fingerprint,
            "weight": primary_weight,
        },
        "other": {
            "model_id": other.model_id,
            "fingerprint": other.fingerprint,
            "weight": other_weight_value,
        },
    }
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    transform_id = f"transform-merge-{digest[:32]}"
    candidate_model_id = f"model-merge-{digest[:32]}"
    try:
        registry.get_candidate(candidate_model_id)
        would_replay = True
    except FrontierwrightError as exc:
        if exc.code != "CANDIDATE_NOT_FOUND":
            raise
        would_replay = (registry.state_dir / "transforms" / transform_id).exists()

    return ActionPreflightView(
        action="evolve-merge",
        ready=True,
        would_replay=would_replay,
        details={
            "transform_id": transform_id,
            "candidate_model_id": candidate_model_id,
            "primary_model_id": primary.model_id,
            "other_model_id": other.model_id,
            "primary_weight": primary_weight,
            "other_weight": other_weight_value,
            "python_executable": python_executable,
            "timeout_seconds": float(timeout_seconds),
        },
    )


def merge_reference_models(
    root: Path,
    *,
    other_model_id: str,
    other_weight: float,
    python_executable: str,
    timeout_seconds: float = 300.0,
) -> MergeView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before merging models.",
            10,
        )
    if (
        isinstance(other_weight, bool)
        or not isinstance(other_weight, (int, float))
        or not math.isfinite(float(other_weight))
        or not 0.0 < float(other_weight) < 1.0
    ):
        raise FrontierwrightError(
            "MERGE_WEIGHT_INVALID",
            "Merge other-model weight must be strictly between 0 and 1.",
            2,
        )
    if timeout_seconds <= 0:
        raise FrontierwrightError(
            "INVALID_TIMEOUT",
            "Merge timeout must be positive.",
            2,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Training Python executable must be nonempty.",
            2,
        )

    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required before model merge.",
            12,
        )
    primary = state.champion.model
    other = registry.get_model(other_model_id)
    if other.model_id == primary.model_id:
        raise FrontierwrightError(
            "MERGE_PARENT_CONFLICT",
            "Merge requires two different model identities.",
            2,
        )
    if other.identity_id != primary.identity_id:
        raise FrontierwrightError(
            "IDENTITY_MISMATCH",
            "Merge parents must belong to the same Frontierwright identity.",
            13,
        )

    _verify_model_artifact_integrity(registry, primary.model_id)
    _verify_model_artifact_integrity(registry, other.model_id)

    intervention = intervention_by_id("frontierwright.evolve.linear-merge")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in linear merge intervention is unavailable.",
            4,
        )

    other_weight_value = float(other_weight)
    primary_weight = 1.0 - other_weight_value
    contract: dict[str, object] = {
        "intervention_id": intervention.intervention_id,
        "intervention_version": intervention.version,
        "primary": {
            "model_id": primary.model_id,
            "fingerprint": primary.fingerprint,
            "weight": primary_weight,
        },
        "other": {
            "model_id": other.model_id,
            "fingerprint": other.fingerprint,
            "weight": other_weight_value,
        },
    }
    contract_json = json.dumps(
        contract,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    transform_id = f"transform-merge-{digest[:32]}"
    candidate_model_id = f"model-merge-{digest[:32]}"

    try:
        registry.get_candidate(candidate_model_id)
    except FrontierwrightError as exc:
        if exc.code != "CANDIDATE_NOT_FOUND":
            raise
    else:
        return _merge_view_from_candidate(
            registry,
            transform_id=transform_id,
            candidate_model_id=candidate_model_id,
            primary_model_id=primary.model_id,
            other_model_id=other.model_id,
            primary_weight=primary_weight,
            other_weight=other_weight_value,
            replayed=True,
        )

    transforms_root = registry.state_dir / "transforms"
    final_root = transforms_root / transform_id
    request_path = transforms_root / "requests" / f"{transform_id}.json"

    if final_root.exists():
        final_descriptor = inspect_local_model(final_root / "model")
        model = ModelState(
            model_id=candidate_model_id,
            identity_id=primary.identity_id,
            origin=primary.origin,
            checkpoint=str(final_descriptor.source_path),
            fingerprint=final_descriptor.fingerprint,
            parent_model_id=primary.model_id,
            stats=(),
            model_format=final_descriptor.model_format,
            trainable=final_descriptor.trainable,
        )
        registry.register_transform_candidate(
            model,
            final_descriptor,
            intervention_id=intervention.intervention_id,
            intervention_version=intervention.version,
            parents=(
                (primary.model_id, primary_weight),
                (other.model_id, other_weight_value),
            ),
            details={"transform_id": transform_id, "recovered_existing_artifact": True},
        )
        return _merge_view_from_candidate(
            registry,
            transform_id=transform_id,
            candidate_model_id=candidate_model_id,
            primary_model_id=primary.model_id,
            other_model_id=other.model_id,
            primary_weight=primary_weight,
            other_weight=other_weight_value,
            replayed=True,
        )

    staging_root = transforms_root / ".staging" / f"{transform_id}-{uuid4().hex}"
    staging_root.mkdir(parents=True, exist_ok=False)
    request: dict[str, object] = {
        "schema_version": 1,
        "backend_id": REFERENCE_BACKEND_ID,
        "operation": "merge",
        "model_source_path": primary.checkpoint,
        "other_model_source_path": other.checkpoint,
        "other_weight": other_weight_value,
        "parents": [
            {
                "model_id": primary.model_id,
                "fingerprint": primary.fingerprint,
                "weight": primary_weight,
            },
            {
                "model_id": other.model_id,
                "fingerprint": other.fingerprint,
                "weight": other_weight_value,
            },
        ],
        "output_root": str(staging_root.resolve()),
    }
    _write_state_json(request_path, request)

    try:
        result = run_structured_command(
            (
                python_executable,
                "-m",
                "frontierwright.reference_backend",
                "{request_json}",
            ),
            environment_overrides={"PYTHONUNBUFFERED": "1"},
            request_path=request_path,
            timeout_seconds=timeout_seconds,
        )
        if result.get("operation") != "merge":
            raise FrontierwrightError(
                "MERGE_RESULT_INVALID",
                "Merge backend did not return a merge result.",
                14,
            )
        output_raw = result.get("output_model_path")
        metrics = result.get("metrics")
        if not isinstance(output_raw, str) or not output_raw:
            raise FrontierwrightError(
                "MERGE_RESULT_INVALID",
                "Merge backend result lacks output_model_path.",
                14,
            )
        if not isinstance(metrics, dict):
            raise FrontierwrightError(
                "MERGE_RESULT_INVALID",
                "Merge backend result metrics must be an object.",
                14,
            )
        if metrics.get("intervention_id") != intervention.intervention_id:
            raise FrontierwrightError(
                "MERGE_RESULT_INVALID",
                "Merge backend intervention identity does not match the request.",
                14,
            )

        backend_model_path = Path(output_raw).expanduser().resolve()
        try:
            backend_model_path.relative_to(staging_root.resolve())
        except ValueError as exc:
            raise FrontierwrightError(
                "MERGE_OUTPUT_ESCAPE",
                "Merge backend output escaped the managed staging directory.",
                14,
            ) from exc

        staged_descriptor = inspect_local_model(backend_model_path)
        if final_root.exists():
            final_descriptor = inspect_local_model(final_root / "model")
            if final_descriptor.fingerprint != staged_descriptor.fingerprint:
                raise FrontierwrightError(
                    "MERGE_ARTIFACT_CONFLICT",
                    "Existing merge artifact differs from the new transform result.",
                    13,
                )
            shutil.rmtree(staging_root, ignore_errors=True)
        else:
            final_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_root, final_root)
            final_descriptor = inspect_local_model(final_root / "model")
            if final_descriptor.fingerprint != staged_descriptor.fingerprint:
                raise FrontierwrightError(
                    "MERGE_ARTIFACT_MISMATCH",
                    "Merge artifact fingerprint changed during publication.",
                    14,
                )

        model = ModelState(
            model_id=candidate_model_id,
            identity_id=primary.identity_id,
            origin=primary.origin,
            checkpoint=str(final_descriptor.source_path),
            fingerprint=final_descriptor.fingerprint,
            parent_model_id=primary.model_id,
            stats=(),
            model_format=final_descriptor.model_format,
            trainable=final_descriptor.trainable,
        )
        registry.register_transform_candidate(
            model,
            final_descriptor,
            intervention_id=intervention.intervention_id,
            intervention_version=intervention.version,
            parents=(
                (primary.model_id, primary_weight),
                (other.model_id, other_weight_value),
            ),
            details={
                "transform_id": transform_id,
                "metrics": dict(metrics),
            },
        )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return _merge_view_from_candidate(
        registry,
        transform_id=transform_id,
        candidate_model_id=candidate_model_id,
        primary_model_id=primary.model_id,
        other_model_id=other.model_id,
        primary_weight=primary_weight,
        other_weight=other_weight_value,
        replayed=False,
    )


def _quantize_view_from_candidate(
    registry: Registry,
    *,
    transform_id: str,
    candidate_model_id: str,
    source_model_id: str,
    replayed: bool,
) -> QuantizeView:
    candidate = registry.get_candidate(candidate_model_id)
    _verify_model_artifact_integrity(registry, candidate_model_id)
    metrics: dict[str, object] = {}
    metadata_path = Path(candidate.model.checkpoint) / "frontierwright-transform.json"
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raw = {}
    if isinstance(raw, dict):
        metrics = dict(raw)
    return QuantizeView(
        transform_id=transform_id,
        replayed=replayed,
        source_model_id=source_model_id,
        candidate_model_id=candidate.model.model_id,
        model_fingerprint=candidate.model.fingerprint,
        checkpoint=candidate.model.checkpoint,
        model_format=candidate.model.model_format.value,
        trainable=candidate.model.trainable,
        metrics=metrics,
    )


def preflight_quantize_reference_model(
    root: Path,
    *,
    python_executable: str,
    timeout_seconds: float = 300.0,
) -> ActionPreflightView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before quantizing a model.",
            10,
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise FrontierwrightError(
            "INVALID_TIMEOUT",
            "Quantization timeout must be positive.",
            2,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Training Python executable must be nonempty.",
            2,
        )
    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required before quantization.",
            12,
        )
    source = state.champion.model
    _verify_model_artifact_integrity(registry, source.model_id)
    if registry.get_model_birth(source.model_id) is not None:
        raise FrontierwrightError(
            "QUANTIZATION_SOURCE_NOT_TRAINED",
            "An untrained birth root must complete initial pretraining before optimization.",
            12,
        )
    if source.model_format is not ModelFormat.HUGGINGFACE or source.trainable is not True:
        raise FrontierwrightError(
            "QUANTIZATION_SOURCE_UNSUPPORTED",
            "Reference int8 quantization requires a full trainable model checkpoint.",
            12,
        )
    intervention = intervention_by_id("frontierwright.optimize.symmetric-int8")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in symmetric int8 intervention is unavailable.",
            4,
        )
    contract: dict[str, object] = {
        "intervention_id": intervention.intervention_id,
        "intervention_version": intervention.version,
        "source": {
            "model_id": source.model_id,
            "fingerprint": source.fingerprint,
        },
        "format": "frontierwright-symmetric-int8-v1",
    }
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    transform_id = f"transform-int8-{digest[:32]}"
    candidate_model_id = f"model-int8-{digest[:32]}"
    try:
        registry.get_candidate(candidate_model_id)
        would_replay = True
    except FrontierwrightError as exc:
        if exc.code != "CANDIDATE_NOT_FOUND":
            raise
        would_replay = (registry.state_dir / "transforms" / transform_id).exists()

    return ActionPreflightView(
        action="optimize-quantize",
        ready=True,
        would_replay=would_replay,
        details={
            "transform_id": transform_id,
            "candidate_model_id": candidate_model_id,
            "source_model_id": source.model_id,
            "source_fingerprint": source.fingerprint,
            "format": "frontierwright-symmetric-int8-v1",
            "python_executable": python_executable,
            "timeout_seconds": float(timeout_seconds),
        },
    )


def quantize_reference_model(
    root: Path,
    *,
    python_executable: str,
    timeout_seconds: float = 300.0,
) -> QuantizeView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before quantizing a model.",
            10,
        )
    if timeout_seconds <= 0:
        raise FrontierwrightError("INVALID_TIMEOUT", "Quantization timeout must be positive.", 2)
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Training Python executable must be nonempty.",
            2,
        )

    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required before quantization.",
            12,
        )
    source = state.champion.model
    _verify_model_artifact_integrity(registry, source.model_id)
    if registry.get_model_birth(source.model_id) is not None:
        raise FrontierwrightError(
            "QUANTIZATION_SOURCE_NOT_TRAINED",
            "An untrained birth root must complete initial pretraining before optimization.",
            12,
        )
    if source.model_format is not ModelFormat.HUGGINGFACE or source.trainable is not True:
        raise FrontierwrightError(
            "QUANTIZATION_SOURCE_UNSUPPORTED",
            "Reference int8 quantization requires a full trainable model checkpoint.",
            12,
        )

    intervention = intervention_by_id("frontierwright.optimize.symmetric-int8")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in symmetric int8 intervention is unavailable.",
            4,
        )

    contract: dict[str, object] = {
        "intervention_id": intervention.intervention_id,
        "intervention_version": intervention.version,
        "source": {
            "model_id": source.model_id,
            "fingerprint": source.fingerprint,
        },
        "format": "frontierwright-symmetric-int8-v1",
    }
    contract_json = json.dumps(
        contract,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    transform_id = f"transform-int8-{digest[:32]}"
    candidate_model_id = f"model-int8-{digest[:32]}"

    try:
        registry.get_candidate(candidate_model_id)
    except FrontierwrightError as exc:
        if exc.code != "CANDIDATE_NOT_FOUND":
            raise
    else:
        return _quantize_view_from_candidate(
            registry,
            transform_id=transform_id,
            candidate_model_id=candidate_model_id,
            source_model_id=source.model_id,
            replayed=True,
        )

    transforms_root = registry.state_dir / "transforms"
    final_root = transforms_root / transform_id
    request_path = transforms_root / "requests" / f"{transform_id}.json"

    if final_root.exists():
        descriptor = inspect_local_model(final_root / "model")
        if (
            descriptor.model_format is not ModelFormat.FRONTIERWRIGHT_QUANTIZED
            or descriptor.trainable
        ):
            raise FrontierwrightError(
                "QUANTIZATION_ARTIFACT_INVALID",
                "Existing quantization artifact is not a non-trainable quantized model.",
                13,
            )
        model = ModelState(
            model_id=candidate_model_id,
            identity_id=source.identity_id,
            origin=source.origin,
            checkpoint=str(descriptor.source_path),
            fingerprint=descriptor.fingerprint,
            parent_model_id=source.model_id,
            stats=(),
            model_format=descriptor.model_format,
            trainable=descriptor.trainable,
        )
        registry.register_transform_candidate(
            model,
            descriptor,
            intervention_id=intervention.intervention_id,
            intervention_version=intervention.version,
            parents=((source.model_id, 1.0),),
            relation=LineageRelation.TRANSFORMED_FROM,
            details={
                "transform_id": transform_id,
                "recovered_existing_artifact": True,
            },
        )
        return _quantize_view_from_candidate(
            registry,
            transform_id=transform_id,
            candidate_model_id=candidate_model_id,
            source_model_id=source.model_id,
            replayed=True,
        )

    staging_root = transforms_root / ".staging" / f"{transform_id}-{uuid4().hex}"
    staging_root.mkdir(parents=True, exist_ok=False)
    request: dict[str, object] = {
        "schema_version": 1,
        "backend_id": REFERENCE_BACKEND_ID,
        "operation": "quantize",
        "model_source_path": source.checkpoint,
        "parent": {
            "model_id": source.model_id,
            "fingerprint": source.fingerprint,
        },
        "output_root": str(staging_root.resolve()),
    }
    _write_state_json(request_path, request)

    try:
        result = run_structured_command(
            (
                python_executable,
                "-m",
                "frontierwright.reference_backend",
                "{request_json}",
            ),
            environment_overrides={"PYTHONUNBUFFERED": "1"},
            request_path=request_path,
            timeout_seconds=timeout_seconds,
        )
        if result.get("operation") != "quantize":
            raise FrontierwrightError(
                "QUANTIZATION_RESULT_INVALID",
                "Quantization backend did not return a quantize result.",
                14,
            )
        output_raw = result.get("output_model_path")
        metrics = result.get("metrics")
        if not isinstance(output_raw, str) or not output_raw:
            raise FrontierwrightError(
                "QUANTIZATION_RESULT_INVALID",
                "Quantization result lacks output_model_path.",
                14,
            )
        if not isinstance(metrics, dict):
            raise FrontierwrightError(
                "QUANTIZATION_RESULT_INVALID",
                "Quantization result metrics must be an object.",
                14,
            )
        if metrics.get("intervention_id") != intervention.intervention_id:
            raise FrontierwrightError(
                "QUANTIZATION_RESULT_INVALID",
                "Quantization intervention identity does not match the request.",
                14,
            )

        backend_model_path = Path(output_raw).expanduser().resolve()
        try:
            backend_model_path.relative_to(staging_root.resolve())
        except ValueError as exc:
            raise FrontierwrightError(
                "QUANTIZATION_OUTPUT_ESCAPE",
                "Quantization output escaped the managed staging directory.",
                14,
            ) from exc

        staged_descriptor = inspect_local_model(backend_model_path)
        if (
            staged_descriptor.model_format is not ModelFormat.FRONTIERWRIGHT_QUANTIZED
            or staged_descriptor.trainable
        ):
            raise FrontierwrightError(
                "QUANTIZATION_ARTIFACT_INVALID",
                "Quantization backend did not produce a non-trainable quantized artifact.",
                14,
            )

        if final_root.exists():
            final_descriptor = inspect_local_model(final_root / "model")
            if final_descriptor.fingerprint != staged_descriptor.fingerprint:
                raise FrontierwrightError(
                    "QUANTIZATION_ARTIFACT_CONFLICT",
                    "Existing quantized artifact differs from the new result.",
                    13,
                )
            shutil.rmtree(staging_root, ignore_errors=True)
        else:
            final_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_root, final_root)
            final_descriptor = inspect_local_model(final_root / "model")
            if final_descriptor.fingerprint != staged_descriptor.fingerprint:
                raise FrontierwrightError(
                    "QUANTIZATION_ARTIFACT_MISMATCH",
                    "Quantized artifact fingerprint changed during publication.",
                    14,
                )

        model = ModelState(
            model_id=candidate_model_id,
            identity_id=source.identity_id,
            origin=source.origin,
            checkpoint=str(final_descriptor.source_path),
            fingerprint=final_descriptor.fingerprint,
            parent_model_id=source.model_id,
            stats=(),
            model_format=final_descriptor.model_format,
            trainable=final_descriptor.trainable,
        )
        registry.register_transform_candidate(
            model,
            final_descriptor,
            intervention_id=intervention.intervention_id,
            intervention_version=intervention.version,
            parents=((source.model_id, 1.0),),
            relation=LineageRelation.TRANSFORMED_FROM,
            details={"transform_id": transform_id, "metrics": dict(metrics)},
        )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return _quantize_view_from_candidate(
        registry,
        transform_id=transform_id,
        candidate_model_id=candidate_model_id,
        source_model_id=source.model_id,
        replayed=False,
    )


def _portable_capability_evidence(
    profile: dict[str, object] | None,
) -> dict[str, object] | None:
    if profile is None:
        return None
    keys = (
        "profile_id",
        "scale_id",
        "scale_version",
        "scale_hash",
        "receipt_id",
        "receipt_sha256",
        "evaluator_id",
        "evaluator_version",
        "stats",
        "measurements",
    )
    return {key: profile.get(key) for key in keys}


def preflight_export_champion_bundle(
    root: Path,
    destination: Path,
) -> ActionPreflightView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before exporting a model.",
            10,
        )
    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required before export.",
            12,
        )
    model = state.champion.model
    _verify_model_artifact_integrity(registry, model.model_id)
    descriptor = inspect_local_model(Path(model.checkpoint))
    if descriptor.fingerprint != model.fingerprint:
        raise FrontierwrightError(
            "ARTIFACT_TAMPERED",
            "Current champion bytes no longer match the registered fingerprint.",
            13,
        )
    intervention = intervention_by_id("frontierwright.operate.portable-export")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in portable export intervention is unavailable.",
            4,
        )

    artifact = registry.get_model_artifact(model.model_id)
    history_evidence: dict[str, object] | None = None
    if artifact is not None:
        history_evidence = {
            "confidence": artifact.get("evidence_confidence"),
            "evidence_files": artifact.get("evidence_files", []),
            "reason": artifact.get("evidence_reason"),
        }
    lineage = [
        {
            "parent_model_id": edge.get("parent_model_id"),
            "relation": edge.get("relation"),
            "ordinal": edge.get("ordinal"),
            "details": edge.get("details", {}),
        }
        for edge in registry.get_model_lineage(model.model_id)
    ]
    profile = registry.get_active_capability_profile(model.model_id)
    provenance: dict[str, object] = {
        "project": {
            "project_id": state.project.get("project_id"),
            "name": state.project.get("name"),
            "edition_profile": state.project.get("edition_profile"),
            "language": state.project.get("language"),
        },
        "model": {
            "model_id": model.model_id,
            "identity_id": model.identity_id,
            "origin": model.origin.value,
            "model_format": model.model_format.value,
            "trainable": model.trainable,
            "fingerprint": model.fingerprint,
            "parent_model_id": model.parent_model_id,
        },
        "history_evidence": history_evidence,
        "lineage": lineage,
        "capability_evidence": _portable_capability_evidence(profile),
    }
    export_id = portable_export_id_for(
        model_fingerprint=descriptor.fingerprint,
        provenance=provenance,
    )
    resolved_destination = destination.expanduser().resolve()
    source = descriptor.source_path.resolve()
    if source.is_dir():
        try:
            resolved_destination.relative_to(source)
        except ValueError:
            pass
        else:
            raise FrontierwrightError(
                "EXPORT_DESTINATION_UNSAFE",
                "Export destination must not be inside the source model directory.",
                2,
            )

    would_replay = False
    if resolved_destination.exists():
        try:
            verified = verify_portable_export(resolved_destination)
        except FrontierwrightError as exc:
            raise FrontierwrightError(
                "EXPORT_DESTINATION_CONFLICT",
                "Export destination already exists but is not the expected portable export.",
                13,
            ) from exc
        if (
            verified.export_id != export_id
            or verified.descriptor.fingerprint != descriptor.fingerprint
        ):
            raise FrontierwrightError(
                "EXPORT_DESTINATION_CONFLICT",
                "Export destination is bound to a different export identity.",
                13,
            )
        would_replay = True

    return ActionPreflightView(
        action="operate-export",
        ready=True,
        would_replay=would_replay,
        details={
            "export_id": export_id,
            "model_id": model.model_id,
            "model_fingerprint": model.fingerprint,
            "destination": str(resolved_destination),
            "total_bytes": descriptor.total_bytes,
        },
    )


def export_champion_bundle(
    root: Path,
    destination: Path,
) -> ExportView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before exporting a model.",
            10,
        )

    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required before export.",
            12,
        )
    model = state.champion.model
    _verify_model_artifact_integrity(registry, model.model_id)

    descriptor = inspect_local_model(Path(model.checkpoint))
    if descriptor.fingerprint != model.fingerprint:
        raise FrontierwrightError(
            "ARTIFACT_TAMPERED",
            "Current champion bytes no longer match the registered fingerprint.",
            13,
        )

    intervention = intervention_by_id("frontierwright.operate.portable-export")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in portable export intervention is unavailable.",
            4,
        )

    artifact = registry.get_model_artifact(model.model_id)
    history_evidence: dict[str, object] | None = None
    if artifact is not None:
        history_evidence = {
            "confidence": artifact.get("evidence_confidence"),
            "evidence_files": artifact.get("evidence_files", []),
            "reason": artifact.get("evidence_reason"),
        }

    lineage: list[dict[str, object]] = []
    for edge in registry.get_model_lineage(model.model_id):
        lineage.append(
            {
                "parent_model_id": edge.get("parent_model_id"),
                "relation": edge.get("relation"),
                "ordinal": edge.get("ordinal"),
                "details": edge.get("details", {}),
            }
        )

    profile = registry.get_active_capability_profile(model.model_id)
    provenance: dict[str, object] = {
        "project": {
            "project_id": state.project.get("project_id"),
            "name": state.project.get("name"),
            "edition_profile": state.project.get("edition_profile"),
            "language": state.project.get("language"),
        },
        "model": {
            "model_id": model.model_id,
            "identity_id": model.identity_id,
            "origin": model.origin.value,
            "model_format": model.model_format.value,
            "trainable": model.trainable,
            "fingerprint": model.fingerprint,
            "parent_model_id": model.parent_model_id,
        },
        "history_evidence": history_evidence,
        "lineage": lineage,
        "capability_evidence": _portable_capability_evidence(profile),
    }

    bundle = publish_portable_export(
        destination,
        descriptor=descriptor,
        provenance=provenance,
    )
    return ExportView(
        export_id=bundle.export_id,
        replayed=bundle.replayed,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        model_format=bundle.descriptor.model_format.value,
        trainable=bundle.descriptor.trainable,
        destination=str(bundle.destination),
        model_path=str(bundle.model_path),
        manifest_path=str(bundle.manifest_path),
        manifest_sha256=bundle.manifest_sha256,
    )


def verify_export_bundle(destination: Path) -> ExportVerifyView:
    verified = verify_portable_export(destination)
    return ExportVerifyView(
        valid=True,
        export_id=verified.export_id,
        model_fingerprint=verified.descriptor.fingerprint,
        model_format=verified.descriptor.model_format.value,
        trainable=verified.descriptor.trainable,
        destination=str(verified.destination),
        model_path=str(verified.model_path),
        manifest_path=str(verified.manifest_path),
        manifest_sha256=verified.manifest_sha256,
        authenticity="NOT_SIGNED",
    )


def _reference_model_preset(model: ModelState) -> str:
    if model.model_format not in {
        ModelFormat.HUGGINGFACE,
        ModelFormat.FRONTIERWRIGHT_QUANTIZED,
    }:
        raise FrontierwrightError(
            "GENERATION_MODEL_UNSUPPORTED",
            "Reference generation requires a Frontierwright reference model artifact.",
            12,
        )
    config_path = Path(model.checkpoint).resolve() / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "GENERATION_MODEL_UNSUPPORTED",
            "Current model does not contain a readable reference config.json.",
            12,
        ) from exc
    if (
        not isinstance(raw, dict)
        or raw.get("frontierwright_reference_backend") != REFERENCE_BACKEND_ID
    ):
        raise FrontierwrightError(
            "GENERATION_MODEL_UNSUPPORTED",
            "Current model is not a Frontierwright reference-backend model.",
            12,
        )
    preset = raw.get("preset")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise FrontierwrightError(
            "GENERATION_MODEL_UNSUPPORTED",
            "Current reference model has an unsupported preset.",
            12,
        )
    return preset


def generate_reference_text(
    root: Path,
    *,
    prompt: str,
    python_executable: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 42,
    device: str = "auto",
    timeout_seconds: float = 60.0,
) -> GenerationView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before generation.",
            10,
        )
    if not isinstance(prompt, str) or not prompt:
        raise FrontierwrightError(
            "GENERATION_PROMPT_INVALID",
            "Prompt must be a nonempty string.",
            2,
        )
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise FrontierwrightError(
            "GENERATION_CONFIG_INVALID",
            "max_new_tokens must be a positive integer.",
            2,
        )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) < 0
    ):
        raise FrontierwrightError(
            "GENERATION_CONFIG_INVALID",
            "temperature must be finite and nonnegative.",
            2,
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FrontierwrightError(
            "GENERATION_CONFIG_INVALID",
            "seed must be a nonnegative integer.",
            2,
        )
    if device not in {"auto", "cpu", "cuda"}:
        raise FrontierwrightError(
            "GENERATION_CONFIG_INVALID",
            "device must be auto, cpu, or cuda.",
            2,
        )
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise FrontierwrightError(
            "INVALID_TIMEOUT",
            "Generation timeout must be finite and positive.",
            2,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Generation Python executable must be nonempty.",
            2,
        )

    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required before generation.",
            12,
        )
    model = state.champion.model
    _verify_model_artifact_integrity(registry, model.model_id)
    preset = _reference_model_preset(model)

    intervention = intervention_by_id("frontierwright.operate.generate-reference")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in reference generation intervention is unavailable.",
            4,
        )

    operations_root = registry.state_dir / "operations"
    request_path = operations_root / f"generate-{uuid4().hex}.json"
    request: dict[str, object] = {
        "schema_version": 1,
        "backend_id": REFERENCE_BACKEND_ID,
        "operation": "generate",
        "model_source_path": str(Path(model.checkpoint).resolve()),
        "prompt": prompt,
        "config": {
            "preset": preset,
            "max_new_tokens": max_new_tokens,
            "temperature": float(temperature),
            "seed": seed,
            "device": device,
        },
    }
    _write_state_json(request_path, request)

    cleanup_error: OSError | None = None
    try:
        result = run_structured_command(
            (
                python_executable,
                "-m",
                "frontierwright.reference_backend",
                "{request_json}",
            ),
            environment_overrides={"PYTHONUNBUFFERED": "1"},
            request_path=request_path,
            timeout_seconds=timeout_seconds,
        )
    finally:
        try:
            request_path.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_error = exc
        try:
            operations_root.rmdir()
        except OSError:
            pass

    if cleanup_error is not None:
        raise FrontierwrightError(
            "GENERATION_REQUEST_CLEANUP_FAILED",
            "Generation completed but the temporary prompt request file could not be removed.",
            4,
        ) from cleanup_error

    if result.get("operation") != "generate" or result.get("prompt") != prompt:
        raise FrontierwrightError(
            "GENERATION_RESULT_INVALID",
            "Generation backend returned mismatched operation or prompt evidence.",
            14,
        )
    continuation = result.get("continuation_text")
    generated_text = result.get("generated_text")
    token_ids = result.get("generated_token_ids")
    metrics = result.get("metrics")
    if (
        not isinstance(continuation, str)
        or not isinstance(generated_text, str)
        or generated_text != prompt + continuation
        or not isinstance(token_ids, list)
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and 0 <= item < 256
            for item in token_ids
        )
        or len(token_ids) != max_new_tokens
        or not isinstance(metrics, dict)
    ):
        raise FrontierwrightError(
            "GENERATION_RESULT_INVALID",
            "Generation backend returned invalid structured output.",
            14,
        )

    return GenerationView(
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        model_format=model.model_format.value,
        prompt=prompt,
        continuation_text=continuation,
        generated_text=generated_text,
        generated_token_ids=list(token_ids),
        metrics=dict(metrics),
    )


def profile_reference_inference(
    root: Path,
    *,
    python_executable: str,
    model_id: str | None = None,
    max_new_tokens: int = 16,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    device: str = "auto",
    timeout_seconds: float = 120.0,
) -> InferenceProfileView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before inference profiling.",
            10,
        )
    for label, value in (
        ("max_new_tokens", max_new_tokens),
        ("warmup_runs", warmup_runs),
        ("measured_runs", measured_runs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FrontierwrightError(
                "INFERENCE_PROFILE_CONFIG_INVALID",
                f"{label} must be a positive integer.",
                2,
            )
    if device not in {"auto", "cpu", "cuda"}:
        raise FrontierwrightError(
            "INFERENCE_PROFILE_CONFIG_INVALID",
            "device must be auto, cpu, or cuda.",
            2,
        )
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise FrontierwrightError(
            "INVALID_TIMEOUT",
            "Inference profile timeout must be finite and positive.",
            2,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Inference profile Python executable must be nonempty.",
            2,
        )

    state = registry.read()
    if model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "A current champion is required before inference profiling.",
                12,
            )
        model = state.champion.model
    else:
        model = registry.get_model(model_id)
    _verify_model_artifact_integrity(registry, model.model_id)
    preset = _reference_model_preset(model)

    intervention = intervention_by_id("frontierwright.operate.profile-reference")
    if intervention is None:
        raise FrontierwrightError(
            "INTERVENTION_NOT_FOUND",
            "Built-in reference inference profile intervention is unavailable.",
            4,
        )

    operations_root = registry.state_dir / "operations"
    request_path = operations_root / f"profile-{uuid4().hex}.json"
    request: dict[str, object] = {
        "schema_version": 1,
        "backend_id": REFERENCE_BACKEND_ID,
        "operation": "profile",
        "model_source_path": str(Path(model.checkpoint).resolve()),
        "config": {
            "preset": preset,
            "max_new_tokens": max_new_tokens,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "device": device,
        },
    }
    _write_state_json(request_path, request)
    try:
        result = run_structured_command(
            (
                python_executable,
                "-m",
                "frontierwright.reference_backend",
                "{request_json}",
            ),
            environment_overrides={"PYTHONUNBUFFERED": "1"},
            request_path=request_path,
            timeout_seconds=timeout_seconds,
        )
    finally:
        request_path.unlink(missing_ok=True)
        try:
            operations_root.rmdir()
        except OSError:
            pass

    metrics = result.get("metrics")
    if result.get("operation") != "profile" or not isinstance(metrics, dict):
        raise FrontierwrightError(
            "INFERENCE_PROFILE_RESULT_INVALID",
            "Inference profile backend returned invalid structured output.",
            14,
        )

    required_numeric = (
        "latency_seconds_mean",
        "latency_seconds_p50",
        "tokens_per_second_mean",
        "tokens_per_second_p50",
    )
    for key in required_numeric:
        metric_value = metrics.get(key)
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(float(metric_value))
            or float(metric_value) <= 0
        ):
            raise FrontierwrightError(
                "INFERENCE_PROFILE_RESULT_INVALID",
                f"Inference profile metric {key} must be finite and positive.",
                14,
            )
    for key in ("max_new_tokens", "warmup_runs", "measured_runs", "parameter_count"):
        integer_metric_value = metrics.get(key)
        if (
            isinstance(integer_metric_value, bool)
            or not isinstance(integer_metric_value, int)
            or integer_metric_value <= 0
        ):
            raise FrontierwrightError(
                "INFERENCE_PROFILE_RESULT_INVALID",
                f"Inference profile metric {key} must be a positive integer.",
                14,
            )
    if metrics.get("max_new_tokens") != max_new_tokens:
        raise FrontierwrightError(
            "INFERENCE_PROFILE_RESULT_INVALID",
            "Inference profile result does not match the requested token count.",
            14,
        )

    history_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"sample_generated_token_ids", "sample_continuation_text"}
    }
    registry.record_event(
        "INFERENCE_PROFILE_MEASURED",
        {
            "intervention_id": intervention.intervention_id,
            "intervention_version": intervention.version,
            "model_id": model.model_id,
            "model_fingerprint": model.fingerprint,
            "config": {
                "preset": preset,
                "max_new_tokens": max_new_tokens,
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "device": device,
            },
            "metrics": history_metrics,
        },
    )
    return InferenceProfileView(
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        model_format=model.model_format.value,
        metrics=dict(metrics),
    )


def get_stats_view(root: Path, model_id: str | None = None) -> StatsView:
    registry = Registry(root)
    if not registry.exists:
        return StatsView(stats=_empty_stats(), reason="Project is not initialized.")

    if model_id is None:
        state = registry.read()
        if state.champion is None:
            return StatsView(
                stats=_empty_stats(),
                reason="No current champion model is available for evaluation.",
            )
        model = state.champion.model
    else:
        model = registry.get_model(model_id)

    profile = registry.get_active_capability_profile(model.model_id)
    if profile is None:
        stats = _empty_stats()
        for stat in model.stats:
            stats[stat.axis.value.lower()] = stat.value
        if any(value is not None for value in stats.values()):
            return StatsView(
                measured=True,
                model_id=model.model_id,
                model_fingerprint=model.fingerprint,
                stats=stats,
                reason=(
                    "Legacy embedded capability evidence is present; "
                    "no raw receipt/profile metadata is attached."
                ),
            )
        return StatsView(
            model_id=model.model_id,
            model_fingerprint=model.fingerprint,
            stats=stats,
            reason="Model has not been measured under a frozen capability scale.",
        )

    stats = _empty_stats()
    for item in profile["stats"]:
        axis = str(item.get("axis", "")).lower()
        value = item.get("value")
        if axis in stats and isinstance(value, (int, float)) and not isinstance(value, bool):
            stats[axis] = float(value)

    return StatsView(
        measured=any(value is not None for value in stats.values()),
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        profile_id=profile["profile_id"],
        scale_id=profile["scale_id"],
        scale_version=profile["scale_version"],
        scale_hash=profile["scale_hash"],
        receipt_id=profile["receipt_id"],
        receipt_sha256=profile["receipt_sha256"],
        evaluator_id=profile["evaluator_id"],
        evaluator_version=profile["evaluator_version"],
        stats=stats,
        raw_measurements=profile["measurements"],
        conditions=profile["conditions"],
    )


def _verify_model_artifact_integrity(registry: Registry, model_id: str) -> None:
    model = registry.get_model(model_id)
    record = registry.get_sealed_artifact(model_id)
    if record is None:
        artifact = registry.get_model_artifact(model_id)
        if artifact is None:
            # Synthetic/test-only registry models may have no artifact row.
            return
        source = artifact.get("source_path")
        if not isinstance(source, str):
            raise FrontierwrightError(
                "ARTIFACT_REGISTRY_MISMATCH",
                "Registered model artifact lacks a source path.",
                13,
            )
        try:
            current = inspect_local_model(Path(source))
        except FrontierwrightError as exc:
            raise FrontierwrightError(
                "ARTIFACT_TAMPERED",
                "Registered model artifact is no longer readable.",
                13,
            ) from exc
        if current.fingerprint != model.fingerprint:
            raise FrontierwrightError(
                "ARTIFACT_TAMPERED",
                "Registered model artifact bytes no longer match its fingerprint.",
                13,
            )
        if Path(model.checkpoint).resolve() != current.source_path.resolve():
            raise FrontierwrightError(
                "ARTIFACT_REGISTRY_MISMATCH",
                "Registered model checkpoint does not match its artifact source.",
                13,
            )
        return

    manifest_path = Path(str(record["manifest_path"]))
    artifact_root = Path(str(record["artifact_root"]))
    verify_manifest_digest(manifest_path, str(record["manifest_sha256"]))
    sealed = verify_sealed_artifact(artifact_root)

    if sealed.model_id != model_id:
        raise FrontierwrightError(
            "ARTIFACT_ID_CONFLICT",
            "Managed artifact model identity does not match registry model.",
            13,
        )
    if sealed.descriptor.fingerprint != model.fingerprint:
        raise FrontierwrightError(
            "ARTIFACT_TAMPERED",
            "Managed artifact bytes no longer match the registered model fingerprint.",
            13,
        )
    if str(record["model_fingerprint"]) != model.fingerprint:
        raise FrontierwrightError(
            "ARTIFACT_REGISTRY_MISMATCH",
            "Managed artifact registry fingerprint does not match the model.",
            13,
        )
    if Path(model.checkpoint).resolve() != sealed.model_path.resolve():
        raise FrontierwrightError(
            "ARTIFACT_REGISTRY_MISMATCH",
            "Registered model checkpoint does not point to its sealed artifact.",
            13,
        )


def import_lm_eval_evidence(
    root: Path,
    *,
    result_path: Path,
    model_id: str | None,
    harness_version: str,
) -> ExternalEvaluationImport:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before importing evaluation evidence.",
            10,
        )
    state = registry.read()
    target_model_id = model_id
    if target_model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "Specify --model or establish a Champion before importing lm-eval evidence.",
                12,
            )
        target_model_id = state.champion.model.model_id
    model = registry.get_model(target_model_id)
    _verify_model_artifact_integrity(registry, model.model_id)
    imported = import_lm_eval_results(
        result_path,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        harness_version=harness_version,
    )
    registry.store_evaluation_receipt(imported.receipt, provenance="IMPORTED")
    return imported


def import_external_evaluation_evidence(
    root: Path,
    *,
    manifest_path: Path,
    model_id: str | None,
) -> ExternalEvaluationImport:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before importing evaluation evidence.",
            10,
        )
    state = registry.read()
    target_model_id = model_id
    if target_model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "Specify --model or establish a Champion before importing evaluation evidence.",
                12,
            )
        target_model_id = state.champion.model.model_id
    model = registry.get_model(target_model_id)
    _verify_model_artifact_integrity(registry, model.model_id)
    imported = import_external_evaluation_manifest(
        manifest_path,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
    )
    registry.store_evaluation_receipt(imported.receipt, provenance="IMPORTED")
    return imported


def import_lighteval_evidence(
    root: Path,
    *,
    result_path: Path,
    model_id: str | None,
    lighteval_version: str,
) -> ExternalEvaluationImport:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before importing evaluation evidence.",
            10,
        )
    state = registry.read()
    target_model_id = model_id
    if target_model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "Specify --model or establish a Champion before importing LightEval evidence.",
                12,
            )
        target_model_id = state.champion.model.model_id
    model = registry.get_model(target_model_id)
    _verify_model_artifact_integrity(registry, model.model_id)
    imported = import_lighteval_results(
        result_path,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        lighteval_version=lighteval_version,
    )
    registry.store_evaluation_receipt(imported.receipt, provenance="IMPORTED")
    return imported


def bind_workload_evaluation(
    root: Path,
    *,
    manifest_path: Path,
    model_id: str | None = None,
) -> WorkloadEvaluationBinding:
    """Bind exact stored evaluation measurements to explicit workload requirements."""

    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before binding workload evaluation evidence.",
            10,
        )
    stored_workload = registry.get_active_workload_profile()
    if stored_workload is None:
        raise FrontierwrightError(
            "WORKLOAD_NOT_CONFIGURED",
            "Define a Workload Profile before binding workload evaluation evidence.",
            12,
        )
    profile_hash = stored_workload.get("profile_hash")
    if not isinstance(profile_hash, str) or not profile_hash:
        raise FrontierwrightError(
            "REGISTRY_ERROR",
            "Active Workload Profile is missing its identity hash.",
            4,
        )

    state = registry.read()
    target_model_id = model_id
    if target_model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "Specify --model or establish a Champion before binding workload evidence.",
                12,
            )
        target_model_id = state.champion.model.model_id
    model = registry.get_model(target_model_id)
    _verify_model_artifact_integrity(registry, model.model_id)

    manifest = load_workload_evaluation_manifest(manifest_path)
    receipt = registry.get_evaluation_receipt(manifest.receipt_id)
    if receipt is None:
        raise FrontierwrightError(
            "EVALUATION_RECEIPT_NOT_FOUND",
            f"Evaluation receipt does not exist: {manifest.receipt_id}",
            3,
        )
    binding = bind_manifest_to_receipt(
        manifest,
        active_profile_hash=profile_hash,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        receipt=receipt,
    )
    payload = binding.to_payload()
    for event in state.history:
        if event.get("kind") != WORKLOAD_EVAL_BINDING_KIND:
            continue
        details = event.get("details")
        if isinstance(details, dict) and details.get("binding_id") == binding.binding_id:
            if details != payload:
                raise FrontierwrightError(
                    "WORKLOAD_EVALUATION_BINDING_CONFLICT",
                    "Binding identity already exists with different evidence.",
                    13,
                )
            return binding
    registry.record_event(WORKLOAD_EVAL_BINDING_KIND, payload)
    return binding


def _workload_eval_coverage_from_state(
    profile: WorkloadProfile,
    *,
    profile_hash: str,
    model_id: str,
    state: ProjectState,
) -> dict[str, object]:
    bindings: list[dict[str, object]] = []
    for event in state.history:
        if event.get("kind") != WORKLOAD_EVAL_BINDING_KIND:
            continue
        details = event.get("details")
        if isinstance(details, dict):
            bindings.append(dict(details))
    return aggregate_workload_evaluation_coverage(
        profile,
        profile_hash=profile_hash,
        model_id=model_id,
        binding_payloads=bindings,
    ).to_payload()


def ingest_stats(
    root: Path,
    *,
    receipt_path: Path,
    scale_path: Path,
) -> StatsView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before ingesting evaluation evidence.",
            10,
        )

    receipt = load_evaluation_receipt(receipt_path)
    scale = load_capability_scale(scale_path)
    model = registry.get_model(receipt.model_id)
    _verify_model_artifact_integrity(registry, model.model_id)

    if receipt.model_fingerprint != model.fingerprint:
        raise FrontierwrightError(
            "EVALUATION_FINGERPRINT_MISMATCH",
            "Evaluation receipt fingerprint does not match the referenced model.",
            12,
        )

    stats = apply_scale(receipt, scale)
    registry.activate_capability_profile(receipt, scale, stats)
    return get_stats_view(root, receipt.model_id)


def _capability_v1_run_view(
    root: Path,
    receipt: EvaluationReceipt,
    *,
    replayed: bool,
) -> CapabilityV1RunView:
    stats_view = get_stats_view(root, receipt.model_id)
    raw_axis_results = receipt.conditions.get("axis_results", [])
    axis_results = (
        [dict(item) for item in raw_axis_results if isinstance(item, dict)]
        if isinstance(raw_axis_results, list)
        else []
    )
    for item in axis_results:
        if isinstance(item.get("uncertainty"), dict):
            continue
        correct = item.get("correct")
        total = item.get("total")
        if (
            isinstance(correct, int)
            and not isinstance(correct, bool)
            and isinstance(total, int)
            and not isinstance(total, bool)
            and total > 0
            and 0 <= correct <= total
        ):
            item["uncertainty"] = capability_v1_uncertainty(correct, total)
    return CapabilityV1RunView(
        bundle_hash=capability_v1_bundle_hash(),
        scale_id=CAPABILITY_V1_SCALE.scale_id,
        scale_version=CAPABILITY_V1_SCALE.scale_version,
        scale_hash=CAPABILITY_V1_SCALE.sha256,
        model_id=receipt.model_id,
        model_fingerprint=receipt.model_fingerprint,
        receipt_id=receipt.receipt_id,
        receipt_sha256=receipt.sha256,
        replayed=replayed,
        task_counts=capability_v1_task_counts(),
        axis_results=axis_results,
        stats=dict(stats_view.stats),
    )


def preflight_capability_v1(
    root: Path,
    *,
    model_id: str | None = None,
    python_executable: str,
    device: str = "auto",
    timeout_seconds: float = 300.0,
) -> ActionPreflightView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before capability evaluation.",
            10,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Capability evaluator Python executable must be nonempty.",
            2,
        )
    if device not in {"auto", "cpu", "cuda"}:
        raise FrontierwrightError(
            "EVALUATION_CONFIG_INVALID",
            "device must be auto, cpu, or cuda.",
            2,
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise FrontierwrightError(
            "EVALUATION_CONFIG_INVALID",
            "timeout_seconds must be a positive finite number.",
            2,
        )

    state = registry.read()
    if model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "A current champion or explicit --model is required for capability evaluation.",
                12,
            )
        model = state.champion.model
    else:
        model = registry.get_model(model_id)
    _verify_model_artifact_integrity(registry, model.model_id)

    bundle_hash = capability_v1_bundle_hash()
    task_counts = capability_v1_task_counts()
    config = {"device": device}
    identity_payload: dict[str, object] = {
        "schema_version": 1,
        "bundle_id": CAPABILITY_V1_BUNDLE_ID,
        "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
        "bundle_hash": bundle_hash,
        "scale_hash": CAPABILITY_V1_SCALE.sha256,
        "scoring": CAPABILITY_V1_SCORING,
        "evaluator_id": CAPABILITY_V1_EVALUATOR_ID,
        "evaluator_version": CAPABILITY_V1_EVALUATOR_VERSION,
        "evidence_schema_version": 2,
        "model_id": model.model_id,
        "model_fingerprint": model.fingerprint,
        "config": config,
    }
    identity_digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt_id = f"receipt-cap-v1-{identity_digest[:32]}"

    existing = registry.get_evaluation_receipt(receipt_id)
    if existing is not None:
        receipt = _receipt_from_registry_row(existing)
        expected = {
            "bundle_id": CAPABILITY_V1_BUNDLE_ID,
            "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
            "bundle_hash": bundle_hash,
            "scale_id": CAPABILITY_V1_SCALE.scale_id,
            "scale_version": CAPABILITY_V1_SCALE.scale_version,
            "scale_hash": CAPABILITY_V1_SCALE.sha256,
            "scoring": CAPABILITY_V1_SCORING,
            "evidence_schema_version": 2,
            "task_counts": task_counts,
            "config": config,
        }
        if not (
            receipt.model_id == model.model_id
            and receipt.model_fingerprint == model.fingerprint
            and receipt.evaluator_id == CAPABILITY_V1_EVALUATOR_ID
            and receipt.evaluator_version == CAPABILITY_V1_EVALUATOR_VERSION
            and all(receipt.conditions.get(key) == value for key, value in expected.items())
        ):
            raise FrontierwrightError(
                "EVALUATION_RECEIPT_CONFLICT",
                "Capability v1 receipt identity is occupied by different evidence.",
                13,
            )

    return ActionPreflightView(
        action="eval-capability-v1",
        ready=True,
        would_replay=existing is not None,
        details={
            "model_id": model.model_id,
            "model_fingerprint": model.fingerprint,
            "bundle_id": CAPABILITY_V1_BUNDLE_ID,
            "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
            "bundle_hash": bundle_hash,
            "scale_hash": CAPABILITY_V1_SCALE.sha256,
            "receipt_id": receipt_id,
            "device": device,
            "python_executable": python_executable,
            "timeout_seconds": float(timeout_seconds),
        },
    )


def run_capability_v1(
    root: Path,
    *,
    model_id: str | None = None,
    python_executable: str,
    device: str = "auto",
    timeout_seconds: float = 300.0,
) -> CapabilityV1RunView:
    """Run the frozen Frontierwright Capability v1 bundle and activate its stats."""

    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before capability evaluation.",
            10,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Capability evaluator Python executable must be nonempty.",
            2,
        )
    if device not in {"auto", "cpu", "cuda"}:
        raise FrontierwrightError(
            "EVALUATION_CONFIG_INVALID",
            "device must be auto, cpu, or cuda.",
            2,
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise FrontierwrightError(
            "EVALUATION_CONFIG_INVALID",
            "timeout_seconds must be a positive finite number.",
            2,
        )

    state = registry.read()
    if model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "A current champion or explicit --model is required for capability evaluation.",
                12,
            )
        model = state.champion.model
    else:
        model = registry.get_model(model_id)
    _verify_model_artifact_integrity(registry, model.model_id)

    bundle_hash = capability_v1_bundle_hash()
    task_counts = capability_v1_task_counts()
    identity_payload: dict[str, object] = {
        "schema_version": 1,
        "bundle_id": CAPABILITY_V1_BUNDLE_ID,
        "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
        "bundle_hash": bundle_hash,
        "scale_hash": CAPABILITY_V1_SCALE.sha256,
        "scoring": CAPABILITY_V1_SCORING,
        "evaluator_id": CAPABILITY_V1_EVALUATOR_ID,
        "evaluator_version": CAPABILITY_V1_EVALUATOR_VERSION,
        "evidence_schema_version": 2,
        "model_id": model.model_id,
        "model_fingerprint": model.fingerprint,
        "config": {"device": device},
    }
    identity_digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt_id = f"receipt-cap-v1-{identity_digest[:32]}"

    existing = registry.get_evaluation_receipt(receipt_id)
    if existing is not None:
        receipt = _receipt_from_registry_row(existing)
        expected = {
            "bundle_id": CAPABILITY_V1_BUNDLE_ID,
            "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
            "bundle_hash": bundle_hash,
            "scale_id": CAPABILITY_V1_SCALE.scale_id,
            "scale_version": CAPABILITY_V1_SCALE.scale_version,
            "scale_hash": CAPABILITY_V1_SCALE.sha256,
            "scoring": CAPABILITY_V1_SCORING,
            "evidence_schema_version": 2,
            "task_counts": task_counts,
            "config": {"device": device},
        }
        if not (
            receipt.model_id == model.model_id
            and receipt.model_fingerprint == model.fingerprint
            and receipt.evaluator_id == CAPABILITY_V1_EVALUATOR_ID
            and receipt.evaluator_version == CAPABILITY_V1_EVALUATOR_VERSION
            and all(receipt.conditions.get(key) == value for key, value in expected.items())
        ):
            raise FrontierwrightError(
                "EVALUATION_RECEIPT_CONFLICT",
                "Capability v1 receipt identity is occupied by different evidence.",
                13,
            )
        stats = apply_scale(receipt, CAPABILITY_V1_SCALE)
        profile = registry.get_active_capability_profile(model.model_id)
        if not (
            profile is not None
            and profile.get("receipt_id") == receipt.receipt_id
            and profile.get("scale_hash") == CAPABILITY_V1_SCALE.sha256
        ):
            registry.activate_capability_profile(receipt, CAPABILITY_V1_SCALE, stats)
        return _capability_v1_run_view(root, receipt, replayed=True)

    request_path = registry.state_dir / "evaluations" / "requests" / f"{receipt_id}.json"
    request: dict[str, object] = {
        "schema_version": 1,
        "backend_id": REFERENCE_BACKEND_ID,
        "operation": "capability_v1",
        "model_source_path": model.checkpoint,
        "config": {"device": device},
    }
    _write_state_json(request_path, request)
    result = run_structured_command(
        (
            python_executable,
            "-m",
            "frontierwright.reference_backend",
            "{request_json}",
        ),
        environment_overrides={"PYTHONUNBUFFERED": "1"},
        request_path=request_path,
        timeout_seconds=float(timeout_seconds),
    )
    if result.get("operation") != "capability_v1":
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Capability evaluator did not return a capability_v1 result.",
            14,
        )
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Capability evaluator metrics must be an object.",
            14,
        )
    required_identity = {
        "bundle_id": CAPABILITY_V1_BUNDLE_ID,
        "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
        "bundle_hash": bundle_hash,
        "scoring": CAPABILITY_V1_SCORING,
        "task_counts": task_counts,
    }
    if any(metrics.get(key) != value for key, value in required_identity.items()):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Capability evaluator bundle/scoring identity does not match frozen v1.",
            14,
        )
    axes_raw = metrics.get("axes")
    if not isinstance(axes_raw, list):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Capability evaluator axes must be a list.",
            14,
        )

    axis_results: list[dict[str, object]] = []
    measurements: list[RawMeasurement] = []
    seen_axes: set[str] = set()
    for raw in axes_raw:
        if not isinstance(raw, dict):
            raise FrontierwrightError(
                "EVALUATION_RESULT_INVALID",
                "Capability axis result must be an object.",
                14,
            )
        axis = raw.get("axis")
        task_id = raw.get("task_id")
        task_version = raw.get("task_version")
        accuracy = raw.get("accuracy")
        correct = raw.get("correct")
        total = raw.get("total")
        margin = raw.get("mean_correct_margin_nats")
        if (
            not isinstance(axis, str)
            or axis not in task_counts
            or axis in seen_axes
            or task_id != f"{CAPABILITY_V1_BUNDLE_ID}.{axis}"
            or task_version != CAPABILITY_V1_BUNDLE_VERSION
            or isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(float(accuracy))
            or not 0.0 <= float(accuracy) <= 1.0
            or isinstance(correct, bool)
            or not isinstance(correct, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total != task_counts[axis]
            or not 0 <= correct <= total
            or abs(float(accuracy) - (correct / total)) > 1e-12
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
        ):
            raise FrontierwrightError(
                "EVALUATION_RESULT_INVALID",
                f"Capability axis result is invalid for {axis!r}.",
                14,
            )
        seen_axes.add(axis)
        normalized = {
            "axis": axis,
            "task_id": task_id,
            "task_version": task_version,
            "accuracy": float(accuracy),
            "correct": correct,
            "total": total,
            "mean_correct_margin_nats": float(margin),
            "uncertainty": capability_v1_uncertainty(correct, total),
        }
        axis_results.append(normalized)
        measurements.extend(
            (
                RawMeasurement(
                    task_id=str(task_id),
                    task_version=str(task_version),
                    metric="accuracy",
                    value=float(accuracy),
                    higher_is_better=True,
                ),
                RawMeasurement(
                    task_id=str(task_id),
                    task_version=str(task_version),
                    metric="mean_correct_margin_nats",
                    value=float(margin),
                    higher_is_better=True,
                ),
            )
        )
    if seen_axes != set(task_counts):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Capability evaluator did not return all four frozen axes.",
            14,
        )
    axis_results.sort(key=lambda item: str(item["axis"]))

    items_raw = metrics.get("item_results")
    if not isinstance(items_raw, list):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Capability evaluator must return per-item paired evidence.",
            14,
        )
    expected_items = {task.item_id: task.axis.value.lower() for task in CAPABILITY_V1_TASKS}
    seen_items: set[str] = set()
    normalized_items_by_id: dict[str, dict[str, object]] = {}
    for raw in items_raw:
        if not isinstance(raw, dict):
            raise FrontierwrightError(
                "EVALUATION_RESULT_INVALID",
                "Capability item result must be an object.",
                14,
            )
        item_id = raw.get("item_id")
        axis = raw.get("axis")
        correct = raw.get("correct")
        margin = raw.get("correct_margin_nats")
        if (
            not isinstance(item_id, str)
            or item_id not in expected_items
            or item_id in seen_items
            or axis != expected_items.get(item_id)
            or not isinstance(correct, bool)
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
        ):
            raise FrontierwrightError(
                "EVALUATION_RESULT_INVALID",
                f"Capability item result is invalid for {item_id!r}.",
                14,
            )
        seen_items.add(item_id)
        normalized_items_by_id[item_id] = {
            "item_id": item_id,
            "axis": axis,
            "correct": correct,
            "correct_margin_nats": float(margin),
        }
    if seen_items != set(expected_items):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Capability evaluator did not return all 64 frozen item results exactly once.",
            14,
        )

    normalized_items = [normalized_items_by_id[task.item_id] for task in CAPABILITY_V1_TASKS]
    axis_summary = {str(item["axis"]): item for item in axis_results}
    for axis, total in task_counts.items():
        axis_items = [item for item in normalized_items if item["axis"] == axis]
        correct_count = sum(1 for item in axis_items if item["correct"] is True)
        mean_margin = (
            sum(float(cast(int | float, item["correct_margin_nats"])) for item in axis_items)
            / total
        )
        summary = axis_summary[axis]
        summary_margin = float(cast(int | float, summary["mean_correct_margin_nats"]))
        if (
            len(axis_items) != total
            or summary.get("correct") != correct_count
            or abs(summary_margin - mean_margin) > 1e-9
        ):
            raise FrontierwrightError(
                "EVALUATION_RESULT_INVALID",
                f"Capability item evidence does not reproduce aggregate axis {axis!r}.",
                14,
            )

    python_version = metrics.get("python_version")
    torch_version = metrics.get("torch_version")
    if not isinstance(python_version, str) or not isinstance(torch_version, str):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Capability evaluator must report Python and PyTorch versions.",
            14,
        )
    conditions: dict[str, object] = {
        "bundle_id": CAPABILITY_V1_BUNDLE_ID,
        "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
        "bundle_hash": bundle_hash,
        "scale_id": CAPABILITY_V1_SCALE.scale_id,
        "scale_version": CAPABILITY_V1_SCALE.scale_version,
        "scale_hash": CAPABILITY_V1_SCALE.sha256,
        "scoring": CAPABILITY_V1_SCORING,
        "evidence_schema_version": 2,
        "task_counts": task_counts,
        "axis_results": axis_results,
        "item_results": normalized_items,
        "config": {"device": device},
        "backend_id": metrics.get("backend_id"),
        "preset": metrics.get("preset"),
        "device": metrics.get("device"),
        "parameter_count": metrics.get("parameter_count"),
        "tokenizer_fingerprint": metrics.get("tokenizer_fingerprint"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "python_version": python_version,
        "torch_version": torch_version,
    }
    receipt = EvaluationReceipt(
        receipt_id=receipt_id,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        evaluator_id=CAPABILITY_V1_EVALUATOR_ID,
        evaluator_version=CAPABILITY_V1_EVALUATOR_VERSION,
        conditions=conditions,
        measurements=tuple(measurements),
    )
    registry.store_evaluation_receipt(receipt, provenance="GENERATED")
    stats = apply_scale(receipt, CAPABILITY_V1_SCALE)
    registry.activate_capability_profile(receipt, CAPABILITY_V1_SCALE, stats)
    return _capability_v1_run_view(root, receipt, replayed=False)


def get_evaluation_packs() -> list[dict[str, object]]:
    return [descriptor.to_dict() for descriptor in BUILTIN_EVALUATION_PACKS]


def _evaluation_run_view(
    receipt: EvaluationReceipt,
    *,
    dataset_id: str,
    dataset_fingerprint: str,
    replayed: bool,
) -> EvaluationRunView:
    pack_id = receipt.conditions.get("pack_id")
    pack_version = receipt.conditions.get("pack_version")
    return EvaluationRunView(
        pack_id=str(pack_id) if isinstance(pack_id, str) else None,
        pack_version=str(pack_version) if isinstance(pack_version, str) else None,
        receipt_id=receipt.receipt_id,
        receipt_sha256=receipt.sha256,
        model_id=receipt.model_id,
        model_fingerprint=receipt.model_fingerprint,
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        evaluator_id=receipt.evaluator_id,
        evaluator_version=receipt.evaluator_version,
        replayed=replayed,
        measurements=[asdict(item) for item in receipt.measurements],
        conditions=dict(receipt.conditions),
    )


def _receipt_from_registry_row(row: dict[str, object]) -> EvaluationReceipt:
    measurements_raw = row.get("measurements")
    conditions = row.get("conditions")
    if not isinstance(measurements_raw, list) or not isinstance(conditions, dict):
        raise FrontierwrightError(
            "EVALUATION_RECEIPT_INVALID",
            "Stored evaluation receipt is malformed.",
            4,
        )
    try:
        measurements_list: list[RawMeasurement] = []
        for item in measurements_raw:
            if not isinstance(item, dict):
                raise ValueError("measurement must be an object")
            direction = item["higher_is_better"]
            if not isinstance(direction, bool):
                raise ValueError("measurement.higher_is_better must be a boolean")
            measurements_list.append(
                RawMeasurement(
                    task_id=str(item["task_id"]),
                    task_version=str(item["task_version"]),
                    metric=str(item["metric"]),
                    value=float(item["value"]),
                    higher_is_better=direction,
                )
            )
        measurements = tuple(measurements_list)
        return EvaluationReceipt(
            receipt_id=str(row["receipt_id"]),
            model_id=str(row["model_id"]),
            model_fingerprint=str(row["model_fingerprint"]),
            evaluator_id=str(row["evaluator_id"]),
            evaluator_version=str(row["evaluator_version"]),
            conditions=dict(conditions),
            measurements=measurements,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "EVALUATION_RECEIPT_INVALID",
            f"Stored evaluation receipt is malformed: {exc}",
            4,
        ) from exc


def _raw_measurement_map(
    items: list[dict[str, object]],
) -> dict[tuple[str, str, str], tuple[float, bool]]:
    result: dict[tuple[str, str, str], tuple[float, bool]] = {}
    for item in items:
        task_id = item.get("task_id")
        task_version = item.get("task_version")
        metric = item.get("metric")
        value = item.get("value")
        direction = item.get("higher_is_better")
        if (
            not isinstance(task_id, str)
            or not isinstance(task_version, str)
            or not isinstance(metric, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not isinstance(direction, bool)
        ):
            raise FrontierwrightError(
                "EVALUATION_RECEIPT_INVALID",
                "Evaluation comparison encountered malformed raw measurement evidence.",
                4,
            )
        key = (task_id, task_version, metric)
        if key in result:
            raise FrontierwrightError(
                "EVALUATION_RECEIPT_INVALID",
                "Evaluation comparison encountered duplicate raw measurement evidence.",
                4,
            )
        result[key] = (float(value), direction)
    return result


def preflight_reference_evaluation(
    root: Path,
    *,
    dataset_id: str,
    model_id: str | None = None,
    python_executable: str,
    device: str = "auto",
    batch_size: int = 4,
    max_batches: int = 16,
    max_dataset_bytes: int = 64 * 1024 * 1024,
    timeout_seconds: float = 300.0,
) -> ActionPreflightView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before evaluation.",
            10,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Evaluation Python executable must be nonempty.",
            2,
        )
    if device not in {"auto", "cpu", "cuda"}:
        raise FrontierwrightError(
            "EVALUATION_CONFIG_INVALID",
            "device must be auto, cpu, or cuda.",
            2,
        )
    for label, value in (
        ("batch_size", batch_size),
        ("max_batches", max_batches),
        ("max_dataset_bytes", max_dataset_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FrontierwrightError(
                "EVALUATION_CONFIG_INVALID",
                f"{label} must be a positive integer.",
                2,
            )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise FrontierwrightError(
            "EVALUATION_CONFIG_INVALID",
            "timeout_seconds must be a positive finite number.",
            2,
        )

    state = registry.read()
    if model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "A current champion or explicit --model is required for evaluation.",
                12,
            )
        model = state.champion.model
    else:
        model = registry.get_model(model_id)
    _verify_model_artifact_integrity(registry, model.model_id)

    dataset = next(
        (item for item in state.datasets if item.get("dataset_id") == dataset_id),
        None,
    )
    if dataset is None:
        raise FrontierwrightError(
            "DATASET_NOT_FOUND",
            f"Active evaluation dataset not found: {dataset_id}",
            3,
        )
    source_path = dataset.get("source_path")
    fingerprint = dataset.get("fingerprint")
    if not isinstance(source_path, str) or not isinstance(fingerprint, str):
        raise FrontierwrightError(
            "DATASET_INVALID",
            "Evaluation dataset registry metadata is incomplete.",
            4,
        )
    descriptor = inspect_local_dataset(Path(source_path))
    if descriptor.fingerprint != fingerprint:
        raise FrontierwrightError(
            "DATASET_CONTENT_DRIFT",
            "Evaluation dataset content changed after registration.",
            13,
        )

    config: dict[str, object] = {
        "device": device,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "max_dataset_bytes": max_dataset_bytes,
    }
    identity_payload: dict[str, object] = {
        "schema_version": 1,
        "pack_id": REFERENCE_LM_PACK.pack_id,
        "pack_version": REFERENCE_LM_PACK.pack_version,
        "model_id": model.model_id,
        "model_fingerprint": model.fingerprint,
        "dataset_id": dataset_id,
        "dataset_fingerprint": fingerprint,
        "config": config,
    }
    identity_digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt_id = f"receipt-eval-{identity_digest[:32]}"
    existing = registry.get_evaluation_receipt(receipt_id)
    if existing is not None:
        receipt = _receipt_from_registry_row(existing)
        expected_conditions = {
            "pack_id": REFERENCE_LM_PACK.pack_id,
            "pack_version": REFERENCE_LM_PACK.pack_version,
            "dataset_id": dataset_id,
            "dataset_fingerprint": fingerprint,
            "config": config,
        }
        if not (
            receipt.model_id == model.model_id
            and receipt.model_fingerprint == model.fingerprint
            and receipt.evaluator_id == REFERENCE_LM_PACK.evaluator_id
            and receipt.evaluator_version == REFERENCE_LM_PACK.evaluator_version
            and all(
                receipt.conditions.get(key) == value for key, value in expected_conditions.items()
            )
        ):
            raise FrontierwrightError(
                "EVALUATION_RECEIPT_CONFLICT",
                "Deterministic evaluation receipt ID is occupied by different evidence.",
                13,
            )

    return ActionPreflightView(
        action="eval-run",
        ready=True,
        would_replay=existing is not None,
        details={
            "pack_id": REFERENCE_LM_PACK.pack_id,
            "pack_version": REFERENCE_LM_PACK.pack_version,
            "model_id": model.model_id,
            "model_fingerprint": model.fingerprint,
            "dataset_id": dataset_id,
            "dataset_fingerprint": fingerprint,
            "receipt_id": receipt_id,
            "config": config,
            "python_executable": python_executable,
            "timeout_seconds": float(timeout_seconds),
        },
    )


def preflight_evaluation_pack(
    root: Path,
    *,
    pack_id: str,
    dataset_id: str,
    model_id: str | None = None,
    python_executable: str,
    device: str = "auto",
    batch_size: int = 4,
    max_batches: int = 16,
    max_dataset_bytes: int = 64 * 1024 * 1024,
    timeout_seconds: float = 300.0,
) -> ActionPreflightView:
    if pack_id != REFERENCE_LM_PACK.pack_id:
        raise FrontierwrightError(
            "EVALUATION_PACK_UNSUPPORTED",
            f"Evaluation pack is not executable in this build: {pack_id}",
            12,
        )
    return preflight_reference_evaluation(
        root,
        dataset_id=dataset_id,
        model_id=model_id,
        python_executable=python_executable,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
        max_dataset_bytes=max_dataset_bytes,
        timeout_seconds=timeout_seconds,
    )


def run_reference_evaluation(
    root: Path,
    *,
    dataset_id: str,
    model_id: str | None = None,
    python_executable: str,
    device: str = "auto",
    batch_size: int = 4,
    max_batches: int = 16,
    max_dataset_bytes: int = 64 * 1024 * 1024,
    timeout_seconds: float = 300.0,
) -> EvaluationRunView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before evaluation.",
            10,
        )
    if not python_executable.strip():
        raise FrontierwrightError(
            "BACKEND_PYTHON_UNAVAILABLE",
            "Evaluation Python executable must be nonempty.",
            2,
        )
    if device not in {"auto", "cpu", "cuda"}:
        raise FrontierwrightError(
            "EVALUATION_CONFIG_INVALID",
            "device must be auto, cpu, or cuda.",
            2,
        )
    for label, value in (
        ("batch_size", batch_size),
        ("max_batches", max_batches),
        ("max_dataset_bytes", max_dataset_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FrontierwrightError(
                "EVALUATION_CONFIG_INVALID",
                f"{label} must be a positive integer.",
                2,
            )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise FrontierwrightError(
            "EVALUATION_CONFIG_INVALID",
            "timeout_seconds must be a positive finite number.",
            2,
        )

    state = registry.read()
    if model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "A current champion or explicit --model is required for evaluation.",
                12,
            )
        model = state.champion.model
    else:
        model = registry.get_model(model_id)

    _verify_model_artifact_integrity(registry, model.model_id)

    dataset = next(
        (item for item in state.datasets if item.get("dataset_id") == dataset_id),
        None,
    )
    if dataset is None:
        raise FrontierwrightError(
            "DATASET_NOT_FOUND",
            f"Active evaluation dataset not found: {dataset_id}",
            3,
        )
    source_path = dataset.get("source_path")
    fingerprint = dataset.get("fingerprint")
    if not isinstance(source_path, str) or not isinstance(fingerprint, str):
        raise FrontierwrightError(
            "DATASET_INVALID",
            "Evaluation dataset registry metadata is incomplete.",
            4,
        )
    descriptor = inspect_local_dataset(Path(source_path))
    if descriptor.fingerprint != fingerprint:
        raise FrontierwrightError(
            "DATASET_CONTENT_DRIFT",
            "Evaluation dataset content changed after registration.",
            13,
        )

    config: dict[str, object] = {
        "device": device,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "max_dataset_bytes": max_dataset_bytes,
    }
    identity_payload: dict[str, object] = {
        "schema_version": 1,
        "pack_id": REFERENCE_LM_PACK.pack_id,
        "pack_version": REFERENCE_LM_PACK.pack_version,
        "model_id": model.model_id,
        "model_fingerprint": model.fingerprint,
        "dataset_id": dataset_id,
        "dataset_fingerprint": fingerprint,
        "config": config,
    }
    identity_digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt_id = f"receipt-eval-{identity_digest[:32]}"

    existing = registry.get_evaluation_receipt(receipt_id)
    if existing is not None:
        receipt = _receipt_from_registry_row(existing)
        expected_conditions = {
            "pack_id": REFERENCE_LM_PACK.pack_id,
            "pack_version": REFERENCE_LM_PACK.pack_version,
            "dataset_id": dataset_id,
            "dataset_fingerprint": fingerprint,
            "config": config,
        }
        replay_identity_matches = bool(
            receipt.model_id == model.model_id
            and receipt.model_fingerprint == model.fingerprint
            and receipt.evaluator_id == REFERENCE_LM_PACK.evaluator_id
            and receipt.evaluator_version == REFERENCE_LM_PACK.evaluator_version
            and all(
                receipt.conditions.get(key) == value for key, value in expected_conditions.items()
            )
        )
        if not replay_identity_matches:
            raise FrontierwrightError(
                "EVALUATION_RECEIPT_CONFLICT",
                (
                    "Deterministic evaluation receipt ID is already occupied by "
                    "different model/data/config evidence."
                ),
                13,
            )
        return _evaluation_run_view(
            receipt,
            dataset_id=dataset_id,
            dataset_fingerprint=fingerprint,
            replayed=True,
        )

    request_path = registry.state_dir / "evaluations" / "requests" / f"{receipt_id}.json"
    request: dict[str, object] = {
        "schema_version": 1,
        "backend_id": REFERENCE_BACKEND_ID,
        "operation": "evaluate",
        "model_source_path": model.checkpoint,
        "dataset_source_path": source_path,
        "config": config,
    }
    _write_state_json(request_path, request)
    result = run_structured_command(
        (
            python_executable,
            "-m",
            "frontierwright.reference_backend",
            "{request_json}",
        ),
        environment_overrides={"PYTHONUNBUFFERED": "1"},
        request_path=request_path,
        timeout_seconds=float(timeout_seconds),
    )
    if result.get("operation") != "evaluate":
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Reference evaluator did not return an evaluate result.",
            14,
        )
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Reference evaluator metrics must be an object.",
            14,
        )

    def finite_metric(name: str, *, positive: bool = False) -> float:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FrontierwrightError(
                "EVALUATION_RESULT_INVALID",
                f"Reference evaluator metric {name} must be numeric.",
                14,
            )
        result_value = float(value)
        if not math.isfinite(result_value) or (positive and result_value <= 0):
            raise FrontierwrightError(
                "EVALUATION_RESULT_INVALID",
                f"Reference evaluator metric {name} is invalid.",
                14,
            )
        return result_value

    cross_entropy = finite_metric("cross_entropy_nats_per_token")
    if cross_entropy < 0:
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Cross-entropy cannot be negative.",
            14,
        )
    perplexity = finite_metric("perplexity", positive=True)
    tokens_evaluated = metrics.get("tokens_evaluated")
    windows_evaluated = metrics.get("windows_evaluated")
    python_version = metrics.get("python_version")
    torch_version = metrics.get("torch_version")
    if (
        not isinstance(python_version, str)
        or not python_version
        or not isinstance(torch_version, str)
        or not torch_version
    ):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Reference evaluator must report Python and PyTorch versions.",
            14,
        )
    if (
        isinstance(tokens_evaluated, bool)
        or not isinstance(tokens_evaluated, int)
        or tokens_evaluated <= 0
        or isinstance(windows_evaluated, bool)
        or not isinstance(windows_evaluated, int)
        or windows_evaluated <= 0
    ):
        raise FrontierwrightError(
            "EVALUATION_RESULT_INVALID",
            "Reference evaluator must report positive token/window counts.",
            14,
        )

    conditions: dict[str, object] = {
        "pack_id": REFERENCE_LM_PACK.pack_id,
        "pack_version": REFERENCE_LM_PACK.pack_version,
        "backend_id": REFERENCE_BACKEND_ID,
        "dataset_id": dataset_id,
        "dataset_fingerprint": fingerprint,
        "dataset_role": dataset.get("role"),
        "dataset_classification": dataset.get("classification"),
        "dataset_recipe_id": dataset.get("preparation_recipe_id"),
        "dataset_recipe_hash": dataset.get("preparation_recipe_hash"),
        "config": config,
        "device": metrics.get("device"),
        "tokens_evaluated": tokens_evaluated,
        "windows_evaluated": windows_evaluated,
        "parameter_count": metrics.get("parameter_count"),
        "python_version": python_version,
        "torch_version": torch_version,
    }
    receipt = EvaluationReceipt(
        receipt_id=receipt_id,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        evaluator_id=REFERENCE_LM_PACK.evaluator_id,
        evaluator_version=REFERENCE_LM_PACK.evaluator_version,
        conditions=conditions,
        measurements=(
            RawMeasurement(
                task_id=REFERENCE_LM_PACK.task_id,
                task_version=REFERENCE_LM_PACK.task_version,
                metric="cross_entropy_nats_per_token",
                value=cross_entropy,
                higher_is_better=False,
            ),
            RawMeasurement(
                task_id=REFERENCE_LM_PACK.task_id,
                task_version=REFERENCE_LM_PACK.task_version,
                metric="perplexity",
                value=perplexity,
                higher_is_better=False,
            ),
        ),
    )
    registry.store_evaluation_receipt(receipt, provenance="GENERATED")

    receipt_path = registry.state_dir / "evaluations" / "receipts" / f"{receipt.receipt_id}.json"
    _write_state_json(
        receipt_path,
        {
            "schema_version": 1,
            **receipt.canonical_payload(),
            "receipt_sha256": receipt.sha256,
        },
    )
    return _evaluation_run_view(
        receipt,
        dataset_id=dataset_id,
        dataset_fingerprint=fingerprint,
        replayed=False,
    )


def run_evaluation_pack(
    root: Path,
    *,
    pack_id: str,
    dataset_id: str,
    model_id: str | None = None,
    python_executable: str,
    device: str = "auto",
    batch_size: int = 4,
    max_batches: int = 16,
    max_dataset_bytes: int = 64 * 1024 * 1024,
    timeout_seconds: float = 300.0,
) -> EvaluationRunView:
    if pack_id != REFERENCE_LM_PACK.pack_id:
        raise FrontierwrightError(
            "EVALUATION_PACK_UNSUPPORTED",
            f"Evaluation pack is not executable in this build: {pack_id}",
            12,
        )
    return run_reference_evaluation(
        root,
        dataset_id=dataset_id,
        model_id=model_id,
        python_executable=python_executable,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
        max_dataset_bytes=max_dataset_bytes,
        timeout_seconds=timeout_seconds,
    )


def compare_candidate_evaluation(
    root: Path,
    *,
    candidate_model_id: str,
    pack_id: str,
    dataset_id: str,
    python_executable: str,
    device: str = "auto",
    batch_size: int = 4,
    max_batches: int = 16,
    max_dataset_bytes: int = 64 * 1024 * 1024,
    timeout_seconds: float = 300.0,
) -> EvaluationCompareView:
    registry = Registry(root)
    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required for raw evaluation comparison.",
            12,
        )
    candidate = registry.get_candidate(candidate_model_id)

    champion_view = run_evaluation_pack(
        root,
        pack_id=pack_id,
        dataset_id=dataset_id,
        model_id=state.champion.model.model_id,
        python_executable=python_executable,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
        max_dataset_bytes=max_dataset_bytes,
        timeout_seconds=timeout_seconds,
    )
    candidate_view = run_evaluation_pack(
        root,
        pack_id=pack_id,
        dataset_id=dataset_id,
        model_id=candidate.model.model_id,
        python_executable=python_executable,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
        max_dataset_bytes=max_dataset_bytes,
        timeout_seconds=timeout_seconds,
    )

    if (
        champion_view.pack_id != candidate_view.pack_id
        or champion_view.pack_version != candidate_view.pack_version
        or champion_view.dataset_fingerprint != candidate_view.dataset_fingerprint
        or champion_view.evaluator_id != candidate_view.evaluator_id
        or champion_view.evaluator_version != candidate_view.evaluator_version
    ):
        return EvaluationCompareView(
            pack_id=pack_id,
            pack_version=champion_view.pack_version,
            dataset_id=dataset_id,
            dataset_fingerprint=champion_view.dataset_fingerprint,
            champion_model_id=state.champion.model.model_id,
            candidate_model_id=candidate.model.model_id,
            candidate_status=candidate.status.value,
            comparable=False,
            reason="Champion and candidate evaluation evidence provenance differs.",
            champion_receipt_id=champion_view.receipt_id,
            candidate_receipt_id=candidate_view.receipt_id,
        )

    champion_measurements = _raw_measurement_map(champion_view.measurements)
    candidate_measurements = _raw_measurement_map(candidate_view.measurements)
    if champion_measurements.keys() != candidate_measurements.keys():
        return EvaluationCompareView(
            pack_id=pack_id,
            pack_version=champion_view.pack_version,
            dataset_id=dataset_id,
            dataset_fingerprint=champion_view.dataset_fingerprint,
            champion_model_id=state.champion.model.model_id,
            candidate_model_id=candidate.model.model_id,
            candidate_status=candidate.status.value,
            comparable=False,
            reason="Champion and candidate evaluation receipts expose different metrics.",
            champion_receipt_id=champion_view.receipt_id,
            candidate_receipt_id=candidate_view.receipt_id,
        )

    comparisons: list[dict[str, object]] = []
    for key in sorted(champion_measurements):
        champion_value, champion_direction = champion_measurements[key]
        candidate_value, candidate_direction = candidate_measurements[key]
        if champion_direction != candidate_direction:
            return EvaluationCompareView(
                pack_id=pack_id,
                pack_version=champion_view.pack_version,
                dataset_id=dataset_id,
                dataset_fingerprint=champion_view.dataset_fingerprint,
                champion_model_id=state.champion.model.model_id,
                candidate_model_id=candidate.model.model_id,
                candidate_status=candidate.status.value,
                comparable=False,
                reason="Champion and candidate metric direction metadata differs.",
                champion_receipt_id=champion_view.receipt_id,
                candidate_receipt_id=candidate_view.receipt_id,
            )
        raw_delta = candidate_value - champion_value
        improvement_delta = raw_delta if champion_direction else -raw_delta
        comparisons.append(
            {
                "task_id": key[0],
                "task_version": key[1],
                "metric": key[2],
                "higher_is_better": champion_direction,
                "champion_value": champion_value,
                "candidate_value": candidate_value,
                "raw_delta": raw_delta,
                "improvement_delta": improvement_delta,
            }
        )

    return EvaluationCompareView(
        pack_id=pack_id,
        pack_version=champion_view.pack_version,
        dataset_id=dataset_id,
        dataset_fingerprint=champion_view.dataset_fingerprint,
        champion_model_id=state.champion.model.model_id,
        candidate_model_id=candidate.model.model_id,
        candidate_status=candidate.status.value,
        comparable=True,
        reason=(
            "Raw metrics were measured with the same evaluation pack, evaluator version, "
            "dataset fingerprint, and evaluation config. Positive improvement_delta means "
            "the candidate moved in the metric's declared better direction."
        ),
        champion_receipt_id=champion_view.receipt_id,
        candidate_receipt_id=candidate_view.receipt_id,
        measurements=comparisons,
    )


def _stored_raw_evaluation_comparisons(
    registry: Registry,
    *,
    champion_model_id: str,
    candidate_model_id: str,
) -> list[dict[str, object]]:
    def evidence_identity(
        receipt: EvaluationReceipt,
    ) -> tuple[str, dict[str, object]] | None:
        conditions = receipt.conditions
        pack_id = conditions.get("pack_id")
        pack_version = conditions.get("pack_version")
        dataset_id = conditions.get("dataset_id")
        dataset_fingerprint = conditions.get("dataset_fingerprint")
        config = conditions.get("config")
        if (
            isinstance(pack_id, str)
            and isinstance(pack_version, str)
            and isinstance(dataset_id, str)
            and isinstance(dataset_fingerprint, str)
            and isinstance(config, dict)
        ):
            identity_payload: dict[str, object] = {
                "kind": "FRONTIERWRIGHT_PACK",
                "pack_id": pack_id,
                "pack_version": pack_version,
                "evaluator_id": receipt.evaluator_id,
                "evaluator_version": receipt.evaluator_version,
                "dataset_id": dataset_id,
                "dataset_fingerprint": dataset_fingerprint,
                "config": config,
            }
            metadata = dict(identity_payload)
        elif conditions.get("adapter_id") == "frontierwright.evaluator.lm-eval-import":
            task_versions = conditions.get("task_versions")
            configs = conditions.get("configs")
            n_samples = conditions.get("n_samples")
            n_shot = conditions.get("n_shot")
            if not isinstance(task_versions, dict):
                return None
            if not isinstance(configs, dict):
                return None
            if not isinstance(n_samples, dict):
                return None
            if not isinstance(n_shot, dict):
                return None
            identity_payload = {
                "kind": "EXTERNAL_LM_EVAL",
                "adapter_id": conditions.get("adapter_id"),
                "adapter_version": conditions.get("adapter_version"),
                "source_format": conditions.get("source_format"),
                "evaluator_id": receipt.evaluator_id,
                "evaluator_version": receipt.evaluator_version,
                "task_versions": task_versions,
                "configs": configs,
                "n_samples": n_samples,
                "n_shot": n_shot,
            }
            metadata = {
                "kind": "EXTERNAL_LM_EVAL",
                "adapter_id": conditions.get("adapter_id"),
                "adapter_version": conditions.get("adapter_version"),
                "evaluator_id": receipt.evaluator_id,
                "evaluator_version": receipt.evaluator_version,
                "task_count": len(task_versions),
            }
        else:
            return None
        try:
            identity_json = json.dumps(
                identity_payload,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return None
        return identity_json, metadata

    def keyed_receipts(
        model_id: str,
    ) -> dict[str, tuple[EvaluationReceipt, dict[str, object]]]:
        result: dict[str, tuple[EvaluationReceipt, dict[str, object]]] = {}
        for row in registry.list_evaluation_receipts(model_id):
            receipt = _receipt_from_registry_row(row)
            identity = evidence_identity(receipt)
            if identity is None:
                continue
            identity_json, metadata = identity
            result.setdefault(identity_json, (receipt, metadata))
        return result

    champion_receipts = keyed_receipts(champion_model_id)
    candidate_receipts = keyed_receipts(candidate_model_id)
    comparisons: list[dict[str, object]] = []

    for evidence_key in sorted(champion_receipts.keys() & candidate_receipts.keys()):
        champion_receipt, metadata = champion_receipts[evidence_key]
        candidate_receipt, _ = candidate_receipts[evidence_key]
        champion_map = _raw_measurement_map(
            [asdict(item) for item in champion_receipt.measurements]
        )
        candidate_map = _raw_measurement_map(
            [asdict(item) for item in candidate_receipt.measurements]
        )
        if champion_map.keys() != candidate_map.keys():
            continue

        measurements: list[dict[str, object]] = []
        direction_mismatch = False
        for measurement_key in sorted(champion_map):
            champion_value, champion_direction = champion_map[measurement_key]
            candidate_value, candidate_direction = candidate_map[measurement_key]
            if champion_direction != candidate_direction:
                direction_mismatch = True
                break
            raw_delta = candidate_value - champion_value
            measurements.append(
                {
                    "task_id": measurement_key[0],
                    "task_version": measurement_key[1],
                    "metric": measurement_key[2],
                    "higher_is_better": champion_direction,
                    "champion_value": champion_value,
                    "candidate_value": candidate_value,
                    "raw_delta": raw_delta,
                    "improvement_delta": (raw_delta if champion_direction else -raw_delta),
                }
            )
        if direction_mismatch:
            continue

        comparison: dict[str, object] = {
            **metadata,
            "champion_receipt_id": champion_receipt.receipt_id,
            "candidate_receipt_id": candidate_receipt.receipt_id,
            "measurements": measurements,
        }
        comparisons.append(comparison)
    return comparisons


def _resource_headroom_from_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    """Summarize point-in-time system availability without claiming model fit.

    These values describe what the OS/driver reported when resources were detected.
    They are not a model-specific peak-memory receipt. Frontierwright must combine a
    future/current model runtime profile with this evidence before claiming how much
    headroom remains *after loading a model*.
    """

    def amount(value: object) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    def fraction(free: int | None, total: int | None) -> float | None:
        if free is None or total is None or total <= 0:
            return None
        return min(1.0, max(0.0, free / total))

    ram_total = amount(snapshot.get("ram_total_bytes"))
    ram_available = amount(snapshot.get("ram_available_bytes"))
    disk_total = amount(snapshot.get("disk_total_bytes"))
    disk_free = amount(snapshot.get("disk_free_bytes"))

    gpus: list[dict[str, object]] = []
    raw_gpus = snapshot.get("gpus")
    if isinstance(raw_gpus, list):
        for index, raw in enumerate(raw_gpus):
            if not isinstance(raw, dict):
                continue
            total = amount(raw.get("memory_total_bytes"))
            free = amount(raw.get("memory_free_bytes"))
            gpus.append(
                {
                    "index": index,
                    "vendor": raw.get("vendor"),
                    "name": raw.get("name"),
                    "memory_total_bytes": total,
                    "memory_available_bytes": free,
                    "available_fraction": fraction(free, total),
                }
            )

    return {
        "semantics": "SYSTEM_AVAILABLE_NOW",
        "model_specific": False,
        "note": (
            "Point-in-time OS/driver availability. Run a model/inference profile "
            "before treating this as post-load model headroom."
        ),
        "ram_available_bytes": ram_available,
        "ram_total_bytes": ram_total,
        "ram_available_fraction": fraction(ram_available, ram_total),
        "disk_available_bytes": disk_free,
        "disk_total_bytes": disk_total,
        "disk_available_fraction": fraction(disk_free, disk_total),
        "gpus": gpus,
    }


def _inference_event_metrics(
    event: dict[str, object], *, model_id: str
) -> tuple[dict[str, object], dict[str, object]] | None:
    if event.get("kind") != "INFERENCE_PROFILE_MEASURED":
        return None
    details = event.get("details")
    if not isinstance(details, dict) or details.get("model_id") != model_id:
        return None
    metrics = details.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return details, metrics


def _same_composable_profile(anchor: dict[str, object], other: dict[str, object]) -> bool:
    condition_hash = anchor.get("profile_condition_hash")
    if not isinstance(condition_hash, str) or not condition_hash:
        return False
    if other.get("profile_condition_hash") != condition_hash:
        return False
    for key in ("execution_boundary", "runtime_id", "runtime_version"):
        anchor_value = anchor.get(key)
        other_value = other.get(key)
        if anchor_value is not None or other_value is not None:
            if anchor_value != other_value:
                return False
    return True


def _latest_model_fit_from_state(
    state: ProjectState, model_id: str | None = None
) -> dict[str, object]:
    if model_id is None:
        if state.champion is None:
            return {}
        model_id = state.champion.model.model_id

    matched: list[tuple[dict[str, object], dict[str, object], dict[str, object]]] = []
    for event in reversed(state.history):
        parsed = _inference_event_metrics(event, model_id=model_id)
        if parsed is None:
            continue
        details, metrics = parsed
        matched.append((event, details, metrics))

    if not matched:
        return {}

    anchor_event, anchor_details, anchor_metrics = matched[0]
    merged = dict(anchor_metrics)
    evidence_sources: list[dict[str, object]] = [
        {
            "source_adapter": anchor_metrics.get("source_adapter"),
            "source_adapter_version": anchor_metrics.get("source_adapter_version"),
            "source_sha256": anchor_metrics.get("source_sha256"),
            "measurement_scope": anchor_metrics.get("measurement_scope"),
            "measured_at": anchor_event.get("created_at"),
        }
    ]

    # External serving evidence is intentionally composable only when the benchmark
    # workload condition, execution boundary, and runtime identity all match exactly.
    # Newer non-null measurements win; older compatible receipts may only fill gaps.
    if isinstance(anchor_metrics.get("profile_condition_hash"), str):
        for event, _details, metrics in matched[1:]:
            if not _same_composable_profile(anchor_metrics, metrics):
                continue
            filled = False
            for key, value in metrics.items():
                if value is None:
                    continue
                if key not in merged or merged[key] is None:
                    merged[key] = value
                    filled = True
            if filled:
                evidence_sources.append(
                    {
                        "source_adapter": metrics.get("source_adapter"),
                        "source_adapter_version": metrics.get("source_adapter_version"),
                        "source_sha256": metrics.get("source_sha256"),
                        "measurement_scope": metrics.get("measurement_scope"),
                        "measured_at": event.get("created_at"),
                    }
                )

    total = merged.get("cuda_memory_total_bytes")
    free_min = merged.get("cuda_memory_free_min_sampled_bytes")
    free_fraction: float | None = None
    if (
        isinstance(total, int)
        and not isinstance(total, bool)
        and total > 0
        and isinstance(free_min, int)
        and not isinstance(free_min, bool)
        and free_min >= 0
    ):
        free_fraction = min(1.0, max(0.0, free_min / total))

    scopes = [
        item.get("measurement_scope")
        for item in evidence_sources
        if isinstance(item.get("measurement_scope"), str)
    ]
    return {
        "semantics": "MEASURED_INFERENCE_PROFILE",
        "model_id": model_id,
        "model_fingerprint": anchor_details.get("model_fingerprint"),
        "measured_at": anchor_event.get("created_at"),
        "measurement_scope": (
            "+".join(dict.fromkeys(str(value) for value in scopes)) if scopes else None
        ),
        "execution_boundary": merged.get("execution_boundary", "LOCAL_MACHINE"),
        "device": merged.get("device"),
        "runtime_id": merged.get("runtime_id"),
        "runtime_version": merged.get("runtime_version"),
        "source_adapter": merged.get("source_adapter"),
        "source_adapter_version": merged.get("source_adapter_version"),
        "source_sha256": merged.get("source_sha256"),
        "evidence_sources": evidence_sources,
        "profile_condition_hash": merged.get("profile_condition_hash"),
        "latency_seconds_p50": merged.get("latency_seconds_p50"),
        "tokens_per_second_p50": merged.get("tokens_per_second_p50"),
        "ttft_seconds_p50": merged.get("ttft_seconds_p50"),
        "tpot_seconds_p50": merged.get("tpot_seconds_p50"),
        "itl_seconds_p50": merged.get("itl_seconds_p50"),
        "request_throughput_per_second": merged.get("request_throughput_per_second"),
        "output_tokens_per_second_aggregate": merged.get("output_tokens_per_second_aggregate"),
        "total_tokens_per_second_aggregate": merged.get("total_tokens_per_second_aggregate"),
        "max_sampled_process_rss_bytes": merged.get("max_sampled_process_rss_bytes"),
        "peak_vram_bytes": merged.get("peak_vram_bytes"),
        "cuda_memory_total_bytes": total,
        "cuda_memory_free_min_sampled_bytes": free_min,
        "cuda_memory_free_after_profile_bytes": merged.get("cuda_memory_free_after_profile_bytes"),
        "cuda_memory_free_fraction_min_sampled": free_fraction,
        "max_new_tokens": merged.get("max_new_tokens"),
        "measured_runs": merged.get("measured_runs"),
        "context_length": merged.get("context_length"),
        "note": (
            "Measured inference evidence for this exact model. Multiple receipts are "
            "combined only when serving-condition hash, execution boundary, and runtime "
            "identity match exactly; missing dimensions remain UNKNOWN."
        ),
    }


def import_vllm_serving_evidence(
    root: Path,
    *,
    result_path: Path,
    model_id: str | None,
    vllm_version: str,
    execution_boundary: BackendDataBoundary,
) -> VLLMServingImport:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before importing serving evidence.",
            10,
        )
    state = registry.read()
    target_model_id = model_id
    if target_model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "Specify --model or establish a Champion before importing serving evidence.",
                12,
            )
        target_model_id = state.champion.model.model_id
    model = registry.get_model(target_model_id)
    _verify_model_artifact_integrity(registry, model.model_id)
    imported = import_vllm_bench_serve(
        result_path,
        vllm_version=vllm_version,
        execution_boundary=execution_boundary,
    )
    for event in reversed(state.history):
        if event.get("kind") != "INFERENCE_PROFILE_MEASURED":
            continue
        details = event.get("details")
        if not isinstance(details, dict) or details.get("model_id") != model.model_id:
            continue
        metrics = details.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if (
            metrics.get("source_adapter") == imported.metrics.get("source_adapter")
            and metrics.get("source_sha256") == imported.source_sha256
            and metrics.get("vllm_version") == imported.vllm_version
            and metrics.get("execution_boundary") == execution_boundary.value
        ):
            return imported
    registry.record_event(
        "INFERENCE_PROFILE_MEASURED",
        {
            "model_id": model.model_id,
            "model_fingerprint": model.fingerprint,
            "metrics": imported.metrics,
            "provenance": "IMPORTED_VLLM_BENCH_SERVE",
        },
    )
    return imported


def import_serving_resource_evidence(
    root: Path,
    *,
    manifest_path: Path,
    model_id: str | None,
) -> ServingResourceImport:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before importing serving resource evidence.",
            10,
        )
    state = registry.read()
    target_model_id = model_id
    if target_model_id is None:
        if state.champion is None:
            raise FrontierwrightError(
                "NO_CHAMPION_MODEL",
                "Specify --model or establish a Champion before importing resource evidence.",
                12,
            )
        target_model_id = state.champion.model.model_id
    model = registry.get_model(target_model_id)
    _verify_model_artifact_integrity(registry, model.model_id)
    imported = import_serving_resource_manifest(manifest_path)
    if imported.model_fingerprint != model.fingerprint:
        raise FrontierwrightError(
            "SERVING_RESOURCE_MODEL_MISMATCH",
            "Serving resource manifest model fingerprint does not match the target model.",
            12,
        )

    for event in reversed(state.history):
        parsed = _inference_event_metrics(event, model_id=model.model_id)
        if parsed is None:
            continue
        _details, metrics = parsed
        if (
            metrics.get("source_adapter") == imported.metrics.get("source_adapter")
            and metrics.get("source_sha256") == imported.source_sha256
            and metrics.get("runtime_id") == imported.runtime_id
            and metrics.get("runtime_version") == imported.runtime_version
        ):
            return imported

    registry.record_event(
        "INFERENCE_PROFILE_MEASURED",
        {
            "model_id": model.model_id,
            "model_fingerprint": model.fingerprint,
            "metrics": imported.metrics,
            "provenance": "IMPORTED_SERVING_RESOURCE_MANIFEST",
        },
    )
    return imported


def get_resource_view(root: Path) -> ResourceView:
    registry = Registry(root)
    if not registry.exists:
        return ResourceView()
    state = registry.read()
    profile = state.resource_profile
    if profile is None:
        return ResourceView()
    return ResourceView(
        available=True,
        profile_id=profile["profile_id"],
        profile_name=profile["profile_name"],
        provenance=profile["provenance"],
        detected_at=profile["created_at"],
        snapshot=profile["snapshot"],
        headroom=_resource_headroom_from_snapshot(profile["snapshot"]),
        model_fit=_latest_model_fit_from_state(state),
    )


def detect_resources(root: Path) -> ResourceView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before detecting project resources.",
            10,
        )
    snapshot = detect_local_resources(root)
    registry.save_resource_snapshot(snapshot, profile_name="local")
    return get_resource_view(root)


def get_workload_view(root: Path) -> WorkloadView:
    registry = Registry(root)
    if not registry.exists:
        return WorkloadView()
    stored = registry.get_active_workload_profile()
    if stored is None:
        return WorkloadView()
    raw_profile = stored.get("profile")
    profile = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    return WorkloadView(
        configured=True,
        profile_id=(
            str(stored["profile_id"]) if isinstance(stored.get("profile_id"), str) else None
        ),
        profile_hash=(
            str(stored["profile_hash"]) if isinstance(stored.get("profile_hash"), str) else None
        ),
        profile_name=(
            str(stored["profile_name"]) if isinstance(stored.get("profile_name"), str) else None
        ),
        source=(str(stored["source"]) if isinstance(stored.get("source"), str) else None),
        created_at=(
            str(stored["created_at"]) if isinstance(stored.get("created_at"), str) else None
        ),
        profile=profile,
    )


def set_workload_profile(root: Path, profile: WorkloadProfile) -> WorkloadView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before setting its workload.",
            10,
        )
    registry.save_workload_profile(profile)
    return get_workload_view(root)


def get_workload_fit(root: Path, model_id: str | None = None) -> WorkloadFitView:
    registry = Registry(root)
    if not registry.exists:
        return WorkloadFitView(note="Project is not initialized.")

    stored = registry.get_active_workload_profile()
    if stored is None:
        return WorkloadFitView(
            note="Define a workload profile before judging user-specific model fit."
        )
    raw_profile = stored.get("profile")
    if not isinstance(raw_profile, dict):
        raise FrontierwrightError(
            "REGISTRY_ERROR",
            "Active workload profile payload is invalid.",
            4,
        )
    profile = workload_profile_from_payload(dict(raw_profile))

    state = registry.read()
    target_model_id = model_id
    if target_model_id is None and state.champion is not None:
        target_model_id = state.champion.model.model_id
    if target_model_id is None:
        return WorkloadFitView(
            configured=True,
            workload_profile_id=str(stored.get("profile_id")),
            workload_profile_hash=str(stored.get("profile_hash")),
            overall_status="UNKNOWN",
            note="No model exists yet, so workload fit cannot be measured.",
        )

    model = registry.get_model(target_model_id)
    stats = get_stats_view(root, target_model_id).stats
    model_fit = _latest_model_fit_from_state(state, target_model_id)
    profile_hash = str(stored.get("profile_hash"))
    workload_evaluation_coverage = _workload_eval_coverage_from_state(
        profile,
        profile_hash=profile_hash,
        model_id=target_model_id,
        state=state,
    )
    raw_context_length = model_fit.get("context_length")
    supported_context_tokens = (
        raw_context_length
        if isinstance(raw_context_length, int)
        and not isinstance(raw_context_length, bool)
        and raw_context_length > 0
        else None
    )
    assessment = assess_workload_fit(
        profile,
        capability_stats=stats,
        inference_metrics=model_fit,
        supported_context_tokens=supported_context_tokens,
        workload_eval_coverage=(
            True if workload_evaluation_coverage.get("complete") is True else None
        ),
        serving_boundary=(
            str(model_fit["execution_boundary"])
            if isinstance(model_fit.get("execution_boundary"), str)
            else None
        ),
    )
    payload = assessment.to_payload()
    counts = payload.get("counts")
    constraints = payload.get("constraints")
    return WorkloadFitView(
        configured=True,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        workload_profile_id=str(stored.get("profile_id")),
        workload_profile_hash=str(stored.get("profile_hash")),
        overall_status=assessment.overall_status.value,
        counts=(
            {str(key): int(value) for key, value in counts.items()}
            if isinstance(counts, dict)
            else {}
        ),
        constraints=(
            [dict(item) for item in constraints if isinstance(item, dict)]
            if isinstance(constraints, list)
            else []
        ),
        workload_evaluation_coverage=workload_evaluation_coverage,
        note=str(payload.get("note")) if payload.get("note") else None,
    )


def get_fit_opportunities(
    root: Path,
    model_id: str | None = None,
) -> FitOpportunityPlan:
    registry = Registry(root)
    if not registry.exists:
        return plan_fit_opportunities(
            edition=EditionProfile.STUDIO,
            model_id=None,
            workload_configured=False,
            constraints=[],
            model_fit=None,
            workload_evaluation_coverage=None,
        )

    state = registry.read()
    raw_edition = state.project.get("edition_profile")
    try:
        edition = EditionProfile(
            str(raw_edition) if raw_edition is not None else EditionProfile.STUDIO.value
        )
    except ValueError:
        edition = EditionProfile.STUDIO

    fit = get_workload_fit(root, model_id)
    target_model_id = model_id or fit.model_id
    if target_model_id is None and state.champion is not None:
        target_model_id = state.champion.model.model_id
    model_fit = (
        _latest_model_fit_from_state(state, target_model_id) if target_model_id is not None else {}
    )
    return plan_fit_opportunities(
        edition=edition,
        model_id=target_model_id,
        workload_configured=fit.configured,
        constraints=fit.constraints,
        model_fit=model_fit,
        workload_evaluation_coverage=fit.workload_evaluation_coverage,
    )


def import_local_model(
    root: Path,
    source: Path,
    *,
    origin: ModelOrigin = ModelOrigin.IMPORTED_LOCAL,
    project_name: str | None = None,
    language: str = "en",
    history_manifest: Path | None = None,
    edition_profile: EditionProfile | None = None,
) -> StatusView:
    if origin is ModelOrigin.ZERO:
        raise FrontierwrightError(
            "INVALID_IMPORT_ORIGIN",
            "Local model import origin must be IMPORTED_LOCAL or INTERNAL_LAB.",
            2,
        )

    descriptor = inspect_local_model(source)
    evidence = discover_history_evidence(
        descriptor,
        manifest_path=history_manifest,
    )

    registry = Registry(root)
    if not registry.exists:
        name = project_name or descriptor.source_path.stem or "Imported Model"
        registry.initialize(
            name,
            origin,
            language=language,
            edition_profile=edition_profile,
        )

    state = registry.read()
    project_origin = ModelOrigin(state.project["origin"])
    if project_origin is not origin:
        raise FrontierwrightError(
            "ORIGIN_MISMATCH",
            f"Project origin is {project_origin.value}, not {origin.value}.",
            13,
        )

    model = ModelState(
        model_id=f"model-{uuid4().hex}",
        identity_id=state.project["identity_id"],
        origin=origin,
        checkpoint=str(descriptor.source_path),
        fingerprint=descriptor.fingerprint,
        stats=(),
        model_format=descriptor.model_format,
        trainable=descriptor.trainable,
    )
    registry.set_initial_imported_model(model, descriptor, evidence)
    return get_status(root)


def _axis_pairs(values: dict[str, int]) -> tuple[tuple[Axis, int], ...]:
    pairs: list[tuple[Axis, int]] = []
    for raw_axis, value in values.items():
        try:
            axis = Axis(raw_axis.upper())
        except ValueError as exc:
            raise FrontierwrightError(
                "INVALID_AXIS",
                f"Unknown capability axis: {raw_axis}",
                2,
            ) from exc
        if isinstance(value, bool) or not isinstance(value, int):
            raise FrontierwrightError(
                "INVALID_BUILD_VALUE",
                f"Build value for {raw_axis} must be an integer.",
                2,
            )
        pairs.append((axis, value))
    return tuple(sorted(pairs, key=lambda item: item[0].value))


def get_build_view(root: Path) -> BuildView:
    registry = Registry(root)
    if not registry.exists:
        return BuildView(reason="Project is not initialized.")

    state = registry.read()
    profile_measured = bool(
        state.capability_profile is not None and state.capability_profile.get("stats")
    )
    if profile_measured:
        mode = BuildMode.TARGETS_FLOORS
    else:
        mode = build_mode(ModelOrigin(state.project["origin"]), state.champion)
    saved = state.build_state

    if mode is BuildMode.NOT_READY:
        return BuildView(
            mode=mode.value,
            reason="Evaluate the inserted model before setting numeric targets/floors.",
        )

    if mode is BuildMode.INTENT:
        if saved is not None and saved.get("mode") == BuildMode.INTENT.value:
            return BuildView(
                mode=mode.value,
                configured=True,
                archetype=saved.get("archetype"),
                priorities=saved.get("priorities", {}),
            )
        return BuildView(
            mode=mode.value,
            configured=False,
            archetype="Balanced",
            reason="Choose a build direction before meaningful stats exist.",
        )

    if saved is not None and saved.get("mode") == BuildMode.TARGETS_FLOORS.value:
        scale_hash = saved.get("scale_hash")
        scale_id = saved.get("scale_id")
        scale_version = saved.get("scale_version")
        bound = all(
            isinstance(value, str) and bool(value)
            for value in (scale_hash, scale_id, scale_version)
        )
        return BuildView(
            mode=mode.value,
            configured=True,
            targets=saved.get("targets", {}),
            floors=saved.get("floors", {}),
            scale_bound=bound,
            scale_hash=scale_hash if isinstance(scale_hash, str) else None,
            scale_id=scale_id if isinstance(scale_id, str) else None,
            scale_version=(scale_version if isinstance(scale_version, str) else None),
            reason=(
                None
                if bound
                else "Numeric build predates frozen-scale binding; re-save targets/floors."
            ),
        )
    return BuildView(
        mode=mode.value,
        configured=False,
        reason="Set numeric targets/floors relative to current measured stats.",
    )


def set_build_intent(
    root: Path,
    *,
    archetype: str,
    priorities: dict[str, int],
) -> BuildView:
    registry = Registry(root)
    try:
        intent = BuildIntent(archetype=archetype, priorities=_axis_pairs(priorities))
    except ValueError as exc:
        raise FrontierwrightError("INVALID_BUILD", str(exc), 2) from exc
    registry.set_build_intent(intent)
    return get_build_view(root)


def set_build_targets(
    root: Path,
    *,
    targets: dict[str, int],
    floors: dict[str, int],
) -> BuildView:
    registry = Registry(root)
    current_build = get_build_view(root)
    if current_build.mode != BuildMode.TARGETS_FLOORS.value:
        raise FrontierwrightError(
            "BUILD_MODE_MISMATCH",
            (
                "Numeric targets/floors are unavailable while current build mode is "
                f"{current_build.mode or BuildMode.NOT_READY.value}."
            ),
            13,
        )
    try:
        build = BuildTargets(
            targets=_axis_pairs(targets),
            floors=_axis_pairs(floors),
        )
    except ValueError as exc:
        raise FrontierwrightError("INVALID_BUILD", str(exc), 2) from exc

    stats = get_stats_view(root)
    if (
        not stats.measured
        or stats.scale_hash is None
        or stats.scale_id is None
        or stats.scale_version is None
    ):
        raise FrontierwrightError(
            "BUILD_SCALE_REQUIRED",
            (
                "Numeric targets/floors require the current champion to have an active "
                "frozen capability profile with exact scale identity."
            ),
            13,
        )
    registry.set_build_targets(
        build,
        scale_hash=stats.scale_hash,
        scale_id=stats.scale_id,
        scale_version=stats.scale_version,
    )
    return get_build_view(root)


def get_data_view(root: Path) -> DataView:
    registry = Registry(root)
    if not registry.exists:
        return DataView()

    state = registry.read()
    datasets: list[dict[str, object]] = []
    for item in state.datasets:
        datasets.append(
            {
                "dataset_id": item["dataset_id"],
                "name": item["name"],
                "role": item["role"],
                "provenance": item["provenance"],
                "classification": item["classification"],
                "source_path": item["source_path"],
                "fingerprint": item["fingerprint"],
                "total_bytes": item["total_bytes"],
                "file_count": item["file_count"],
                "license": item["license"],
                "domain": item["domain"],
                "language": item["language"],
                "token_count": item["token_count"],
                "source_dataset_id": item.get("source_dataset_id"),
                "preparation_recipe_id": item.get("preparation_recipe_id"),
                "preparation_recipe_hash": item.get("preparation_recipe_hash"),
                "managed": isinstance(item.get("preparation_recipe_hash"), str),
            }
        )
    return DataView(datasets=datasets)


def add_local_dataset(
    root: Path,
    source: Path,
    *,
    name: str,
    role: DatasetRole,
    classification: DatasetClassification | None = None,
    license_name: str | None = None,
    domain: str | None = None,
    language: str | None = None,
    token_count: int | None = None,
) -> DataView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before registering data.",
            10,
        )
    descriptor = inspect_local_dataset(source)
    registry.register_dataset(
        descriptor,
        name=name,
        role=role,
        provenance=DatasetProvenance.LOCAL_USER,
        classification=classification,
        license_name=license_name,
        domain=domain,
        language=language,
        token_count=token_count,
    )
    return get_data_view(root)


def get_data_preparation_plugins() -> list[dict[str, object]]:
    return [
        {
            "plugin_id": plugin.plugin_id,
            "plugin_version": plugin.plugin_version,
            "title": plugin.title,
        }
        for plugin in BUILTIN_DATA_PREPARATION_PLUGINS
    ]


def _resolve_preparation_sources(
    state: ProjectState,
    plugin: DataPreparationPlugin,
    recipe: DataPreparationRecipe,
) -> dict[str, DatasetDescriptor]:
    rows = {
        str(item["dataset_id"]): item
        for item in state.datasets
        if isinstance(item.get("dataset_id"), str)
    }
    resolved: dict[str, DatasetDescriptor] = {}
    for binding in plugin.source_bindings(recipe):
        row = rows.get(binding.dataset_id)
        if row is None:
            raise FrontierwrightError(
                "DATASET_NOT_FOUND",
                f"Preparation source dataset is not active: {binding.dataset_id}",
                3,
            )
        source_path = row.get("source_path")
        fingerprint = row.get("fingerprint")
        if not isinstance(source_path, str) or not isinstance(fingerprint, str):
            raise FrontierwrightError(
                "DATASET_INVALID",
                f"Preparation source metadata is incomplete: {binding.dataset_id}",
                4,
            )
        if fingerprint != binding.fingerprint:
            raise FrontierwrightError(
                "DATA_RECIPE_SOURCE_DRIFT",
                f"Preparation source registry fingerprint changed: {binding.dataset_id}",
                13,
            )
        descriptor = inspect_local_dataset(Path(source_path))
        if descriptor.fingerprint != binding.fingerprint:
            raise FrontierwrightError(
                "DATASET_CONTENT_DRIFT",
                (f"Preparation source content changed after registration: {binding.dataset_id}"),
                13,
            )
        resolved[binding.dataset_id] = descriptor
    return resolved


def preflight_prepare_dataset(
    root: Path,
    *,
    dataset_id: str,
    plugin_id: str,
    config: dict[str, object] | None = None,
) -> ActionPreflightView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before preparing data.",
            10,
        )
    state = registry.read()
    source = next(
        (item for item in state.datasets if item.get("dataset_id") == dataset_id),
        None,
    )
    if source is None:
        raise FrontierwrightError(
            "DATASET_NOT_FOUND",
            "Source dataset is not active in this project.",
            3,
        )
    plugin = data_preparation_plugin(plugin_id)
    source_role = source.get("role")
    if plugin_id == PREFERENCE_JSONL_PLUGIN_ID and source_role != DatasetRole.PREFERENCE.value:
        raise FrontierwrightError(
            "DATA_RECIPE_ROLE_INCOMPATIBLE",
            "preference-jsonl requires a PREFERENCE dataset.",
            12,
        )
    if source_role == DatasetRole.PREFERENCE.value and plugin_id not in {
        SNAPSHOT_COPY_PLUGIN_ID,
        PREFERENCE_JSONL_PLUGIN_ID,
    }:
        raise FrontierwrightError(
            "DATA_RECIPE_ROLE_INCOMPATIBLE",
            "PREFERENCE datasets may only use snapshot-copy or preference-jsonl preparation.",
            12,
        )
    source_is_prepared = isinstance(source.get("preparation_recipe_hash"), str)
    if source_is_prepared and not plugin.accepts_prepared_source:
        raise FrontierwrightError(
            "DATASET_ALREADY_PREPARED",
            "This preparation recipe does not accept an already prepared source.",
            13,
        )
    if plugin.requires_prepared_source and not source_is_prepared:
        raise FrontierwrightError(
            "DATA_RECIPE_PREPARED_SOURCE_REQUIRED",
            "This preparation recipe requires a managed prepared dataset as its source.",
            13,
        )
    source_path = source.get("source_path")
    source_fingerprint = source.get("fingerprint")
    if not isinstance(source_path, str) or not isinstance(source_fingerprint, str):
        raise FrontierwrightError(
            "DATASET_INVALID",
            "Source dataset registry metadata is incomplete.",
            4,
        )
    descriptor = inspect_local_dataset(Path(source_path))
    if descriptor.fingerprint != source_fingerprint:
        raise FrontierwrightError(
            "DATASET_CONTENT_DRIFT",
            "Source dataset content changed after registration; register the new content first.",
            13,
        )
    recipe = plugin.build_recipe(
        source_dataset_id=dataset_id,
        source_fingerprint=source_fingerprint,
        config=config,
    )
    resolved = _resolve_preparation_sources(state, plugin, recipe)
    existing_dataset_id = next(
        (
            str(item["dataset_id"])
            for item in state.datasets
            if item.get("preparation_recipe_hash") == recipe.recipe_hash
            and isinstance(item.get("dataset_id"), str)
        ),
        None,
    )
    artifact_root = registry.state_dir / "data" / "prepared" / recipe.recipe_id
    return ActionPreflightView(
        action="data-prepare",
        ready=True,
        would_replay=existing_dataset_id is not None,
        details={
            "dataset_id": dataset_id,
            "source_fingerprint": source_fingerprint,
            "plugin_id": recipe.plugin_id,
            "plugin_version": recipe.plugin_version,
            "recipe_id": recipe.recipe_id,
            "recipe_hash": recipe.recipe_hash,
            "config": recipe.config,
            "source_count": len(resolved),
            "existing_dataset_id": existing_dataset_id,
            "managed_artifact_exists": artifact_root.exists(),
        },
    )


def prepare_dataset(
    root: Path,
    *,
    dataset_id: str,
    plugin_id: str,
    config: dict[str, object] | None = None,
    name: str | None = None,
) -> DataView:
    """Materialize one reproducible managed dataset preparation recipe."""

    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before preparing data.",
            10,
        )

    state = registry.read()
    source = next(
        (item for item in state.datasets if item.get("dataset_id") == dataset_id),
        None,
    )
    if source is None:
        raise FrontierwrightError(
            "DATASET_NOT_FOUND",
            "Source dataset is not active in this project.",
            3,
        )
    plugin = data_preparation_plugin(plugin_id)
    source_role = source.get("role")
    if plugin_id == PREFERENCE_JSONL_PLUGIN_ID and source_role != DatasetRole.PREFERENCE.value:
        raise FrontierwrightError(
            "DATA_RECIPE_ROLE_INCOMPATIBLE",
            "preference-jsonl requires a PREFERENCE dataset.",
            12,
        )
    if source_role == DatasetRole.PREFERENCE.value and plugin_id not in {
        SNAPSHOT_COPY_PLUGIN_ID,
        PREFERENCE_JSONL_PLUGIN_ID,
    }:
        raise FrontierwrightError(
            "DATA_RECIPE_ROLE_INCOMPATIBLE",
            ("PREFERENCE datasets may only use snapshot-copy or preference-jsonl preparation."),
            12,
        )

    source_is_prepared = isinstance(source.get("preparation_recipe_hash"), str)
    if source_is_prepared and not plugin.accepts_prepared_source:
        raise FrontierwrightError(
            "DATASET_ALREADY_PREPARED",
            (
                "This preparation recipe does not accept an already prepared source. "
                "Choose a recipe that explicitly supports chained preparation."
            ),
            13,
        )
    if plugin.requires_prepared_source and not source_is_prepared:
        raise FrontierwrightError(
            "DATA_RECIPE_PREPARED_SOURCE_REQUIRED",
            "This preparation recipe requires a managed prepared dataset as its source.",
            13,
        )

    source_path = source.get("source_path")
    source_fingerprint = source.get("fingerprint")
    if not isinstance(source_path, str) or not isinstance(source_fingerprint, str):
        raise FrontierwrightError(
            "DATASET_INVALID",
            "Source dataset registry metadata is incomplete.",
            4,
        )

    descriptor = inspect_local_dataset(Path(source_path))
    if descriptor.fingerprint != source_fingerprint:
        raise FrontierwrightError(
            "DATASET_CONTENT_DRIFT",
            "Source dataset content changed after registration; register the new content first.",
            13,
        )

    recipe = plugin.build_recipe(
        source_dataset_id=dataset_id,
        source_fingerprint=source_fingerprint,
        config=config,
    )
    sources = _resolve_preparation_sources(state, plugin, recipe)
    prepared = plugin.materialize(
        registry.state_dir,
        sources=sources,
        recipe=recipe,
    )
    registry.register_prepared_dataset(recipe, prepared, name=name)
    return get_data_view(root)


def prepare_dataset_snapshot(
    root: Path,
    *,
    dataset_id: str,
    name: str | None = None,
) -> DataView:
    return prepare_dataset(
        root,
        dataset_id=dataset_id,
        plugin_id=SNAPSHOT_COPY_PLUGIN_ID,
        config=None,
        name=name,
    )


def preflight_prepare_dataset_mixture(
    root: Path,
    *,
    inputs: list[tuple[str, int]],
    max_output_bytes: int = 4 * 1024**3,
) -> ActionPreflightView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before mixing data.",
            10,
        )
    if len(inputs) < 2:
        raise FrontierwrightError(
            "DATA_MIXTURE_INPUTS_INVALID",
            "A weighted mixture requires at least two prepared datasets.",
            2,
        )
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
    ):
        raise FrontierwrightError(
            "DATA_MIXTURE_INPUTS_INVALID",
            "max_output_bytes must be a positive integer.",
            2,
        )

    state = registry.read()
    rows = {
        str(item["dataset_id"]): item
        for item in state.datasets
        if isinstance(item.get("dataset_id"), str)
    }
    seen_ids: set[str] = set()
    selected: list[dict[str, object]] = []
    source_specs: list[dict[str, object]] = []
    for dataset_id, parts in inputs:
        if dataset_id in seen_ids:
            raise FrontierwrightError(
                "DATA_MIXTURE_INPUTS_INVALID",
                f"Mixture dataset is duplicated: {dataset_id}",
                2,
            )
        seen_ids.add(dataset_id)
        if isinstance(parts, bool) or not isinstance(parts, int) or parts <= 0:
            raise FrontierwrightError(
                "DATA_MIXTURE_INPUTS_INVALID",
                f"Mixture parts must be a positive integer: {dataset_id}",
                2,
            )
        row = rows.get(dataset_id)
        if row is None:
            raise FrontierwrightError(
                "DATASET_NOT_FOUND",
                f"Mixture source dataset is not active: {dataset_id}",
                3,
            )
        recipe_id = row.get("preparation_recipe_id")
        fingerprint = row.get("fingerprint")
        if not isinstance(recipe_id, str) or not isinstance(fingerprint, str):
            raise FrontierwrightError(
                "DATA_MIXTURE_SOURCE_INVALID",
                f"Weighted mixture source is not managed prepared text: {dataset_id}",
                13,
            )
        source_recipe = registry.get_data_recipe(recipe_id)
        if source_recipe is None or source_recipe.get("plugin_id") not in {
            TEXT_LINES_PLUGIN_ID,
            WEIGHTED_TEXT_MIXTURE_PLUGIN_ID,
        }:
            raise FrontierwrightError(
                "DATA_MIXTURE_SOURCE_INVALID",
                f"Weighted mixture source has incompatible recipe: {dataset_id}",
                13,
            )
        selected.append(row)
        source_specs.append({"dataset_id": dataset_id, "fingerprint": fingerprint, "parts": parts})

    roles = {str(item["role"]) for item in selected}
    provenances = {str(item["provenance"]) for item in selected}
    if len(roles) != 1:
        raise FrontierwrightError(
            "DATA_MIXTURE_METADATA_CONFLICT",
            "Mixture sources must have the same dataset role.",
            13,
        )
    if len(provenances) != 1:
        raise FrontierwrightError(
            "DATA_MIXTURE_METADATA_CONFLICT",
            "Mixture sources must have the same provenance.",
            13,
        )

    primary = selected[0]
    plugin = data_preparation_plugin(WEIGHTED_TEXT_MIXTURE_PLUGIN_ID)
    recipe = plugin.build_recipe(
        source_dataset_id=str(primary["dataset_id"]),
        source_fingerprint=str(primary["fingerprint"]),
        config={"sources": source_specs, "max_output_bytes": max_output_bytes},
    )
    resolved = _resolve_preparation_sources(state, plugin, recipe)
    existing_dataset_id = next(
        (
            str(item["dataset_id"])
            for item in state.datasets
            if item.get("preparation_recipe_hash") == recipe.recipe_hash
            and isinstance(item.get("dataset_id"), str)
        ),
        None,
    )
    artifact_root = registry.state_dir / "data" / "prepared" / recipe.recipe_id
    return ActionPreflightView(
        action="data-mix",
        ready=True,
        would_replay=existing_dataset_id is not None,
        details={
            "recipe_id": recipe.recipe_id,
            "recipe_hash": recipe.recipe_hash,
            "plugin_id": recipe.plugin_id,
            "plugin_version": recipe.plugin_version,
            "sources": source_specs,
            "source_count": len(resolved),
            "max_output_bytes": max_output_bytes,
            "existing_dataset_id": existing_dataset_id,
            "managed_artifact_exists": artifact_root.exists(),
        },
    )


def prepare_dataset_mixture(
    root: Path,
    *,
    inputs: list[tuple[str, int]],
    name: str | None = None,
    max_output_bytes: int = 4 * 1024**3,
) -> DataView:
    """Create a deterministic weighted mixture from managed text datasets."""

    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before mixing data.",
            10,
        )
    if len(inputs) < 2:
        raise FrontierwrightError(
            "DATA_MIXTURE_INPUTS_INVALID",
            "A weighted mixture requires at least two prepared datasets.",
            2,
        )

    seen_ids: set[str] = set()
    state = registry.read()
    rows = {
        str(item["dataset_id"]): item
        for item in state.datasets
        if isinstance(item.get("dataset_id"), str)
    }
    selected: list[dict[str, object]] = []
    source_specs: list[dict[str, object]] = []

    for dataset_id, parts in inputs:
        if dataset_id in seen_ids:
            raise FrontierwrightError(
                "DATA_MIXTURE_INPUTS_INVALID",
                f"Mixture dataset is duplicated: {dataset_id}",
                2,
            )
        seen_ids.add(dataset_id)
        if isinstance(parts, bool) or not isinstance(parts, int) or parts <= 0:
            raise FrontierwrightError(
                "DATA_MIXTURE_INPUTS_INVALID",
                f"Mixture parts must be a positive integer: {dataset_id}",
                2,
            )
        row = rows.get(dataset_id)
        if row is None:
            raise FrontierwrightError(
                "DATASET_NOT_FOUND",
                f"Mixture source dataset is not active: {dataset_id}",
                3,
            )

        recipe_id = row.get("preparation_recipe_id")
        fingerprint = row.get("fingerprint")
        if not isinstance(recipe_id, str) or not isinstance(fingerprint, str):
            raise FrontierwrightError(
                "DATA_MIXTURE_SOURCE_INVALID",
                (
                    "Weighted mixture inputs must be managed prepared text datasets; "
                    f"{dataset_id} is not."
                ),
                13,
            )
        source_recipe = registry.get_data_recipe(recipe_id)
        if source_recipe is None or source_recipe.get("plugin_id") not in {
            TEXT_LINES_PLUGIN_ID,
            WEIGHTED_TEXT_MIXTURE_PLUGIN_ID,
        }:
            raise FrontierwrightError(
                "DATA_MIXTURE_SOURCE_INVALID",
                (
                    "Weighted mixture inputs must come from text-lines or prior "
                    f"weighted-text-mixture recipes: {dataset_id}"
                ),
                13,
            )

        selected.append(row)
        source_specs.append(
            {
                "dataset_id": dataset_id,
                "fingerprint": fingerprint,
                "parts": parts,
            }
        )

    roles = {str(item["role"]) for item in selected}
    provenances = {str(item["provenance"]) for item in selected}
    if len(roles) != 1:
        raise FrontierwrightError(
            "DATA_MIXTURE_METADATA_CONFLICT",
            "Mixture sources must have the same dataset role.",
            13,
        )
    if len(provenances) != 1:
        raise FrontierwrightError(
            "DATA_MIXTURE_METADATA_CONFLICT",
            "Mixture sources must have the same provenance.",
            13,
        )

    classification_rank = {
        DatasetClassification.PUBLIC: 0,
        DatasetClassification.INTERNAL: 1,
        DatasetClassification.CONFIDENTIAL: 2,
        DatasetClassification.PRIVATE: 3,
    }
    classifications = [DatasetClassification(str(item["classification"])) for item in selected]
    effective_classification = max(
        classifications,
        key=lambda item: classification_rank[item],
    )

    def common_optional_text(key: str) -> str | None:
        values = {item.get(key) for item in selected}
        if len(values) != 1:
            return None
        value = next(iter(values))
        return str(value) if isinstance(value, str) else None

    primary = selected[0]
    primary_id = str(primary["dataset_id"])
    primary_fingerprint = str(primary["fingerprint"])
    plugin = data_preparation_plugin(WEIGHTED_TEXT_MIXTURE_PLUGIN_ID)
    recipe = plugin.build_recipe(
        source_dataset_id=primary_id,
        source_fingerprint=primary_fingerprint,
        config={
            "sources": source_specs,
            "max_output_bytes": max_output_bytes,
        },
    )
    sources = _resolve_preparation_sources(state, plugin, recipe)
    prepared = plugin.materialize(
        registry.state_dir,
        sources=sources,
        recipe=recipe,
    )
    registry.register_prepared_dataset(
        recipe,
        prepared,
        name=name,
        metadata_overrides={
            "classification": effective_classification,
            "license": common_optional_text("license"),
            "domain": common_optional_text("domain"),
            "language": common_optional_text("language"),
        },
    )
    return get_data_view(root)


def _required_dataset_role(path_id: TrainingPathId) -> DatasetRole:
    if path_id in (
        TrainingPathId.FROM_SCRATCH_PRETRAINING,
        TrainingPathId.CONTINUED_PRETRAINING,
        TrainingPathId.DISTILL,
    ):
        return DatasetRole.PRETRAIN
    if path_id is TrainingPathId.DPO:
        return DatasetRole.PREFERENCE
    if path_id is TrainingPathId.RL_POLICY_OPTIMIZATION:
        return DatasetRole.ROLLOUT
    return DatasetRole.SFT


def _write_state_json(path: Path, payload: dict[str, object]) -> None:
    try:
        atomic_write_json(path, payload)
    except OSError as exc:
        raise FrontierwrightError(
            "STATE_FILE_ERROR",
            f"Could not atomically publish state file: {path}",
            4,
        ) from exc


def _find_dataset(
    datasets: tuple[dict[str, object], ...],
    *,
    dataset_id: str | None,
    role: DatasetRole,
) -> dict[str, object]:
    matching = [item for item in datasets if item.get("role") == role.value]
    if dataset_id is not None:
        for item in matching:
            if item.get("dataset_id") == dataset_id:
                return item
        raise FrontierwrightError(
            "DATASET_NOT_FOUND",
            f"Active {role.value} dataset not found: {dataset_id}",
            3,
        )
    if not matching:
        raise FrontierwrightError(
            "DATASET_REQUIRED",
            f"{role.value} dataset is required for this path.",
            12,
        )
    if len(matching) > 1:
        prepared = [
            item for item in matching if isinstance(item.get("preparation_recipe_hash"), str)
        ]
        if len(prepared) == 1:
            return prepared[0]
        raise FrontierwrightError(
            "DATASET_AMBIGUOUS",
            f"Multiple {role.value} datasets are registered; choose --dataset explicitly.",
            2,
        )
    return matching[0]


def _plan_input_blockers(
    registry: Registry,
    state: ProjectState,
    plan: TrainingPlan,
) -> list[str]:
    blockers: list[str] = []
    project_state = state

    champion = project_state.champion
    if plan.model_id is None:
        if champion is not None:
            blockers.append("current champion changed since plan creation")
    else:
        if champion is None:
            blockers.append("planned champion no longer exists")
        elif (
            champion.model.model_id != plan.model_id
            or champion.model.fingerprint != plan.model_fingerprint
        ):
            blockers.append("current champion differs from the model pinned by the plan")
        else:
            artifact = project_state.champion_artifact
            if artifact is not None:
                source = artifact.get("source_path")
                if isinstance(source, str):
                    try:
                        current = inspect_local_model(Path(source))
                    except FrontierwrightError:
                        blockers.append("pinned model artifacts are no longer readable")
                    else:
                        if current.fingerprint != plan.model_fingerprint:
                            blockers.append("pinned model content fingerprint drifted")

    dataset = next(
        (item for item in project_state.datasets if item.get("dataset_id") == plan.dataset_id),
        None,
    )
    if dataset is None:
        blockers.append("planned dataset is no longer active")
    else:
        if dataset.get("fingerprint") != plan.dataset_fingerprint:
            blockers.append("planned dataset registry fingerprint changed")
        if dataset.get("preparation_recipe_id") != plan.dataset_recipe_id:
            blockers.append("planned dataset preparation recipe identity changed")
        if dataset.get("preparation_recipe_hash") != plan.dataset_recipe_hash:
            blockers.append("planned dataset preparation recipe hash changed")
        if dataset.get("classification") != plan.dataset_classification:
            blockers.append("planned dataset classification changed")
        source = dataset.get("source_path")
        if isinstance(source, str):
            try:
                current_data = inspect_local_dataset(Path(source))
            except FrontierwrightError:
                blockers.append("planned dataset files are no longer readable")
            else:
                if current_data.fingerprint != plan.dataset_fingerprint:
                    blockers.append("planned dataset content fingerprint drifted")

    try:
        pinned_classification = DatasetClassification(plan.dataset_classification)
        pinned_boundary = BackendDataBoundary(plan.backend_data_boundary)
    except ValueError:
        blockers.append("plan contains invalid data-policy metadata")
    else:
        if not backend_allows_dataset(pinned_boundary, pinned_classification):
            blockers.append(
                "data policy blocks this dataset classification from the pinned backend"
            )

    blockers.extend(_lab_adapter_blockers(registry, plan))

    if plan.resource_profile_id is not None:
        profile = project_state.resource_profile
        if profile is None or profile.get("profile_id") != plan.resource_profile_id:
            blockers.append("active resource profile differs from the plan")
    return blockers


def _accounted_gpu_count(
    calibration: dict[str, object],
    resource_profile: dict[str, object] | None,
) -> tuple[int | None, str | None]:
    backend_result = calibration.get("backend_result")
    backend_map = backend_result if isinstance(backend_result, dict) else {}
    reported = backend_map.get("gpu_count")
    if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
        return reported, "CALIBRATION_REPORTED"

    if resource_profile is not None:
        snapshot = resource_profile.get("snapshot")
        if isinstance(snapshot, dict):
            gpus = snapshot.get("gpus")
            if isinstance(gpus, list) and gpus:
                return len(gpus), "RESOURCE_PROFILE_DETECTED"
    return None, None


def _calibration_budget_blockers(
    plan: TrainingPlan,
    calibration: dict[str, object] | None,
    resource_profile: dict[str, object] | None,
) -> list[str]:
    if calibration is None:
        return ["representative calibration required"]
    if calibration.get("feasible") is not True:
        return ["latest calibration reported infeasible"]

    blockers: list[str] = []
    if calibration.get("tokens_per_second") is None:
        blockers.append("calibration did not measure tokens_per_second")
    if calibration.get("projected_wall_seconds") is None:
        blockers.append("calibration did not project total wall time")
    if calibration.get("projected_storage_bytes") is None:
        blockers.append("calibration did not project new storage")
    if calibration.get("peak_vram_bytes") is None and calibration.get("peak_ram_bytes") is None:
        blockers.append("calibration did not measure peak memory")

    budgets = plan.budgets
    projected_wall = calibration.get("projected_wall_seconds")
    projected_storage = calibration.get("projected_storage_bytes")

    if (
        budgets.max_wall_seconds is not None
        and isinstance(projected_wall, (int, float))
        and not isinstance(projected_wall, bool)
        and float(projected_wall) > budgets.max_wall_seconds
    ):
        blockers.append("projected wall time exceeds max_wall_seconds budget")

    if (
        budgets.max_storage_bytes is not None
        and isinstance(projected_storage, int)
        and not isinstance(projected_storage, bool)
        and projected_storage > budgets.max_storage_bytes
    ):
        blockers.append("projected storage exceeds max_storage_bytes budget")

    backend_result = calibration.get("backend_result")
    backend_map = backend_result if isinstance(backend_result, dict) else {}

    if budgets.max_gpu_hours is not None:
        gpu_count, _ = _accounted_gpu_count(calibration, resource_profile)
        if gpu_count is None:
            blockers.append("GPU-hour budget cannot be enforced without GPU count")
        elif not isinstance(projected_wall, (int, float)) or isinstance(projected_wall, bool):
            blockers.append("GPU-hour budget cannot be enforced without projected wall time")
        else:
            projected_gpu_hours = float(projected_wall) * gpu_count / 3600.0
            if projected_gpu_hours > budgets.max_gpu_hours:
                blockers.append("projected GPU-hours exceed max_gpu_hours budget")

    if budgets.max_money is not None:
        projected_money = backend_map.get("projected_money")
        if not isinstance(projected_money, (int, float)) or isinstance(projected_money, bool):
            blockers.append("money budget cannot be evaluated without projected_money")
        elif float(projected_money) > budgets.max_money:
            blockers.append("projected money exceeds max_money budget")
        else:
            blockers.append("max_money cannot be hard-enforced by the local executor yet")

    return blockers


def _budget_enforcement_modes(
    plan: TrainingPlan,
    calibration: dict[str, object] | None,
    resource_profile: dict[str, object] | None,
) -> dict[str, str]:
    modes: dict[str, str] = {}
    if plan.budgets.max_runs is not None:
        modes["max_runs"] = "HARD_REGISTRY_ADMISSION"
    if plan.budgets.max_wall_seconds is not None:
        modes["max_wall_seconds"] = "HARD_LOCAL_TIMEOUT"
    if plan.budgets.max_gpu_hours is not None:
        if calibration is None:
            modes["max_gpu_hours"] = "UNAVAILABLE_WITHOUT_CALIBRATION"
        else:
            gpu_count, provenance = _accounted_gpu_count(
                calibration,
                resource_profile,
            )
            if gpu_count is None or provenance is None:
                modes["max_gpu_hours"] = "UNAVAILABLE_WITHOUT_GPU_COUNT"
            else:
                modes["max_gpu_hours"] = f"HARD_ACCOUNTED_TIMEOUT:{provenance}"
    if plan.budgets.max_storage_bytes is not None:
        modes["max_storage_bytes"] = "RUNTIME_WATCHDOG_AND_FINALIZATION_GATE"
    if plan.budgets.max_money is not None:
        modes["max_money"] = "NOT_RUNTIME_ENFORCEABLE"
    return modes


def _resolve_backend_adapter(
    registry: Registry,
    state: ProjectState,
    backend: CommandBackendSpec,
) -> tuple[str | None, str | None]:
    adapter_ref = backend.provider_adapter_ref
    if adapter_ref is None:
        if backend.data_boundary is BackendDataBoundary.CONTROLLED_PRIVATE:
            raise FrontierwrightError(
                "LAB_ADAPTER_REQUIRED",
                "CONTROLLED_PRIVATE backends must be bound to a connected Lab adapter.",
                12,
            )
        return None, None

    if EditionProfile(state.project["edition_profile"]) is not EditionProfile.LAB:
        raise FrontierwrightError(
            "LAB_EDITION_REQUIRED",
            "Backends bound to Lab adapters require the Frontierwright Lab profile.",
            12,
        )

    adapter = registry.get_lab_adapter(adapter_ref)
    if adapter is None:
        raise FrontierwrightError(
            "LAB_ADAPTER_NOT_CONNECTED",
            f"Backend references an unconnected Lab adapter: {adapter_ref}",
            12,
        )
    kinds = set(adapter.get("kinds", []))
    if not {
        LabAdapterKind.TRAINER.value,
        LabAdapterKind.EXECUTOR.value,
        LabAdapterKind.CLUSTER_EXECUTOR.value,
    }.intersection(kinds):
        raise FrontierwrightError(
            "LAB_ADAPTER_KIND_UNSUPPORTED",
            "Backend provider adapter must declare TRAINER or EXECUTOR capability.",
            12,
        )
    if adapter.get("data_boundary") != backend.data_boundary.value:
        raise FrontierwrightError(
            "LAB_ADAPTER_BOUNDARY_MISMATCH",
            "Backend data boundary differs from its connected Lab adapter manifest.",
            12,
        )
    manifest_hash = adapter.get("manifest_hash")
    if not isinstance(manifest_hash, str):
        raise FrontierwrightError(
            "LAB_ADAPTER_INVALID",
            "Connected Lab adapter lacks a manifest hash.",
            4,
        )
    return adapter_ref, manifest_hash


def _validate_rl_backend_adapter(
    registry: Registry,
    adapter_ref: str,
    spec: RLExperimentSpec,
) -> None:
    adapter = registry.get_lab_adapter(adapter_ref)
    if adapter is None:
        raise FrontierwrightError(
            "LAB_ADAPTER_NOT_CONNECTED",
            f"RL backend references an unconnected Lab adapter: {adapter_ref}",
            12,
        )
    kinds = set(adapter.get("kinds", []))
    required_kinds = {
        LabAdapterKind.TRAINER.value,
        LabAdapterKind.ROLLOUT_ENGINE.value,
        LabAdapterKind.REWARD_PROVIDER.value,
    }
    missing_kinds = sorted(required_kinds - kinds)
    if missing_kinds:
        raise FrontierwrightError(
            "LAB_RL_ADAPTER_INCOMPLETE",
            "Lab RL adapter is missing required roles: " + ", ".join(missing_kinds),
            12,
        )
    capabilities = set(adapter.get("capabilities", []))
    required_capabilities = {
        f"rl.algorithm.{spec.algorithm_id}",
        f"environment.kind.{spec.environment.kind}",
        f"reward.kind.{spec.reward.kind}",
    }
    missing_capabilities = sorted(required_capabilities - capabilities)
    if missing_capabilities:
        raise FrontierwrightError(
            "LAB_RL_CAPABILITY_UNSUPPORTED",
            "Lab RL adapter does not declare: " + ", ".join(missing_capabilities),
            12,
        )


def _lab_adapter_blockers(registry: Registry, plan: TrainingPlan) -> list[str]:
    if plan.backend_adapter_ref is None:
        if plan.backend_data_boundary == BackendDataBoundary.CONTROLLED_PRIVATE.value:
            return ["controlled-private backend has no pinned Lab adapter"]
        return []
    if plan.backend_adapter_hash is None:
        return ["plan lacks pinned Lab adapter manifest hash"]

    adapter = registry.get_lab_adapter(plan.backend_adapter_ref)
    if adapter is None:
        return ["pinned Lab adapter is not connected"]
    if adapter.get("manifest_hash") != plan.backend_adapter_hash:
        return ["pinned Lab adapter manifest hash changed"]
    if adapter.get("data_boundary") != plan.backend_data_boundary:
        return ["pinned Lab adapter data boundary changed"]
    kinds = set(adapter.get("kinds", []))
    if not {
        LabAdapterKind.TRAINER.value,
        LabAdapterKind.EXECUTOR.value,
        LabAdapterKind.CLUSTER_EXECUTOR.value,
    }.intersection(kinds):
        return ["pinned Lab adapter no longer provides trainer/executor capability"]
    return []


def get_plan_view(root: Path, plan_id: str) -> PlanView:
    registry = Registry(root)
    plan = registry.get_plan(plan_id)
    state = registry.read()
    calibration = registry.latest_calibration_for_plan(plan.plan_id)
    blockers = _plan_input_blockers(registry, state, plan)
    blockers.extend(_calibration_budget_blockers(plan, calibration, state.resource_profile))
    return PlanView(
        plan_id=plan.plan_id,
        path_id=plan.path_id.value,
        intervention_id=plan.intervention_id,
        intervention_version=plan.intervention_version,
        intervention_family=plan.intervention_family,
        backend_id=plan.backend_id,
        backend_spec_hash=plan.backend_spec_hash,
        model_id=plan.model_id,
        dataset_id=plan.dataset_id,
        dataset_recipe_id=plan.dataset_recipe_id,
        dataset_recipe_hash=plan.dataset_recipe_hash,
        dataset_classification=plan.dataset_classification,
        backend_data_boundary=plan.backend_data_boundary,
        backend_adapter_ref=plan.backend_adapter_ref,
        backend_adapter_hash=plan.backend_adapter_hash,
        resource_profile_id=plan.resource_profile_id,
        permission=plan.permission.name,
        budgets=plan.budgets.to_dict(),
        budget_enforcement=_budget_enforcement_modes(
            plan,
            calibration,
            state.resource_profile,
        ),
        config=plan.config,
        idempotency_key=plan.idempotency_key,
        calibration=calibration,
        ready=not blockers,
        blockers=list(dict.fromkeys(blockers)),
    )


def create_training_plan(
    root: Path,
    *,
    path_id: TrainingPathId,
    backend_spec_path: Path,
    dataset_id: str | None,
    permission: PermissionLevel,
    budgets: HardBudgets,
    config: dict[str, object],
) -> PlanView:
    registry = Registry(root)
    if not registry.exists:
        raise FrontierwrightError(
            "NOT_INITIALIZED",
            "Initialize a Frontierwright project before creating a training plan.",
            10,
        )
    backend = load_command_backend_spec(backend_spec_path)
    if path_id not in backend.supported_paths:
        raise FrontierwrightError(
            "BACKEND_PATH_UNSUPPORTED",
            f"Backend {backend.backend_id} does not support {path_id.value}.",
            12,
        )

    paths = get_paths_view(root)
    assessment = next(
        (item for item in paths.paths if item.get("path_id") == path_id.value),
        None,
    )
    if assessment is None:
        raise FrontierwrightError("PATH_NOT_FOUND", "Training path does not exist.", 3)
    if assessment.get("availability") == PathAvailability.LOCKED.value:
        blockers = assessment.get("blockers")
        raise FrontierwrightError(
            "PATH_LOCKED",
            f"Training path is locked: {blockers}",
            12,
        )

    state = registry.read()
    dataset = _find_dataset(
        state.datasets,
        dataset_id=dataset_id,
        role=_required_dataset_role(path_id),
    )
    intervention = intervention_for_training_path(path_id)
    rl_spec: RLExperimentSpec | None = None
    if path_id is TrainingPathId.RL_POLICY_OPTIMIZATION:
        # Parsing here makes environment/reward/algorithm identity mandatory before
        # plan hashing. The exact nested RL spec therefore participates in idempotency.
        rl_spec = rl_spec_from_config(config)
    try:
        dataset_classification = DatasetClassification(str(dataset["classification"]))
    except (KeyError, ValueError) as exc:
        raise FrontierwrightError(
            "DATASET_CLASSIFICATION_INVALID",
            "Dataset classification is missing or invalid.",
            4,
        ) from exc
    if not backend_allows_dataset(backend.data_boundary, dataset_classification):
        raise FrontierwrightError(
            "DATA_POLICY_LOCKED",
            (
                f"{dataset_classification.value} data cannot be used with "
                f"{backend.data_boundary.value} backend boundary."
            ),
            12,
        )

    backend_adapter_ref, backend_adapter_hash = _resolve_backend_adapter(
        registry,
        state,
        backend,
    )
    if rl_spec is not None and backend_adapter_ref is not None:
        _validate_rl_backend_adapter(registry, backend_adapter_ref, rl_spec)

    try:
        key = compute_plan_idempotency_key(
            path_id=path_id,
            intervention_id=intervention.intervention_id,
            intervention_version=intervention.version,
            backend_id=backend.backend_id,
            backend_spec_hash=backend.sha256,
            model_fingerprint=(
                state.champion.model.fingerprint if state.champion is not None else None
            ),
            dataset_fingerprint=str(dataset["fingerprint"]),
            dataset_recipe_hash=(
                str(dataset["preparation_recipe_hash"])
                if isinstance(dataset.get("preparation_recipe_hash"), str)
                else None
            ),
            resource_profile_id=(
                str(state.resource_profile["profile_id"])
                if state.resource_profile is not None
                else None
            ),
            permission=permission,
            budgets=budgets,
            config=config,
            dataset_classification=dataset_classification.value,
            backend_data_boundary=backend.data_boundary.value,
            backend_adapter_ref=backend_adapter_ref,
            backend_adapter_hash=backend_adapter_hash,
        )
    except (TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "INVALID_PLAN_CONFIG",
            f"Plan config must be finite JSON-compatible data: {exc}",
            2,
        ) from exc

    plan = TrainingPlan(
        plan_id=f"plan-{uuid4().hex}",
        path_id=path_id,
        intervention_id=intervention.intervention_id,
        intervention_version=intervention.version,
        intervention_family=intervention.family.value,
        backend_id=backend.backend_id,
        backend_spec_hash=backend.sha256,
        model_id=state.champion.model.model_id if state.champion is not None else None,
        model_fingerprint=(
            state.champion.model.fingerprint if state.champion is not None else None
        ),
        model_source_path=(
            str(state.champion_artifact["source_path"])
            if state.champion is not None
            and state.champion_artifact is not None
            and isinstance(state.champion_artifact.get("source_path"), str)
            else (state.champion.model.checkpoint if state.champion is not None else None)
        ),
        dataset_id=str(dataset["dataset_id"]),
        dataset_fingerprint=str(dataset["fingerprint"]),
        dataset_source_path=str(dataset["source_path"]),
        dataset_recipe_id=(
            str(dataset["preparation_recipe_id"])
            if isinstance(dataset.get("preparation_recipe_id"), str)
            else None
        ),
        dataset_recipe_hash=(
            str(dataset["preparation_recipe_hash"])
            if isinstance(dataset.get("preparation_recipe_hash"), str)
            else None
        ),
        dataset_classification=dataset_classification.value,
        backend_data_boundary=backend.data_boundary.value,
        backend_adapter_ref=backend_adapter_ref,
        backend_adapter_hash=backend_adapter_hash,
        resource_profile_id=(
            str(state.resource_profile["profile_id"])
            if state.resource_profile is not None
            else None
        ),
        permission=permission,
        budgets=budgets,
        config=config,
        idempotency_key=key,
    )
    stored = registry.store_plan(plan)
    _write_state_json(
        registry.state_dir / "plans" / f"{stored.plan_id}.json",
        stored.request_payload(),
    )
    return get_plan_view(root, stored.plan_id)


def _validate_backend_for_plan(
    plan: TrainingPlan,
    backend_spec_path: Path,
) -> CommandBackendSpec:
    backend = load_command_backend_spec(backend_spec_path)
    if backend.backend_id != plan.backend_id:
        raise FrontierwrightError(
            "BACKEND_MISMATCH",
            "Backend ID differs from the backend pinned by the plan.",
            13,
        )
    if backend.sha256 != plan.backend_spec_hash:
        raise FrontierwrightError(
            "BACKEND_SPEC_DRIFT",
            "Backend spec content changed after plan creation.",
            13,
        )
    if backend.provider_adapter_ref != plan.backend_adapter_ref:
        raise FrontierwrightError(
            "BACKEND_ADAPTER_DRIFT",
            "Backend provider adapter differs from the adapter pinned by the plan.",
            13,
        )
    if plan.path_id not in backend.supported_paths:
        raise FrontierwrightError(
            "BACKEND_PATH_UNSUPPORTED",
            f"Backend no longer supports {plan.path_id.value}.",
            13,
        )
    return backend


def _calibration_preflight_context(
    root: Path,
    *,
    plan_id: str,
    backend_spec_path: Path,
    timeout_seconds: float,
) -> tuple[Registry, TrainingPlan, CommandBackendSpec, float, dict[str, object] | None]:
    registry = Registry(root)
    plan = registry.get_plan(plan_id)
    if plan.permission < PermissionLevel.DRY_RUN:
        raise FrontierwrightError(
            "PERMISSION_DENIED",
            "Calibration requires DRY_RUN permission or higher.",
            13,
        )
    state = registry.read()
    blockers = _plan_input_blockers(registry, state, plan)
    if blockers:
        raise FrontierwrightError(
            "PLAN_INPUT_STALE",
            "; ".join(blockers),
            13,
        )
    backend = _validate_backend_for_plan(plan, backend_spec_path)

    effective_timeout = timeout_seconds
    if plan.budgets.max_wall_seconds is not None:
        effective_timeout = min(effective_timeout, plan.budgets.max_wall_seconds)
    if effective_timeout <= 0:
        raise FrontierwrightError(
            "INVALID_TIMEOUT",
            "Calibration timeout must be positive.",
            2,
        )
    existing = registry.latest_calibration_for_plan(plan_id)
    return registry, plan, backend, effective_timeout, existing


def preflight_training_calibration(
    root: Path,
    *,
    plan_id: str,
    backend_spec_path: Path,
    timeout_seconds: float = 300.0,
) -> ActionPreflightView:
    _, plan, backend, effective_timeout, existing = _calibration_preflight_context(
        root,
        plan_id=plan_id,
        backend_spec_path=backend_spec_path,
        timeout_seconds=timeout_seconds,
    )
    return ActionPreflightView(
        action="calibrate",
        ready=True,
        would_replay=existing is not None,
        details={
            "plan_id": plan.plan_id,
            "backend_id": backend.backend_id,
            "backend_spec_hash": backend.sha256,
            "effective_timeout_seconds": effective_timeout,
            "existing_calibration_id": (
                existing.get("calibration_id") if existing is not None else None
            ),
        },
    )


def calibrate_training_plan(
    root: Path,
    *,
    plan_id: str,
    backend_spec_path: Path,
    timeout_seconds: float = 300.0,
    rerun: bool = False,
) -> PlanView:
    registry, plan, backend, effective_timeout, existing = _calibration_preflight_context(
        root,
        plan_id=plan_id,
        backend_spec_path=backend_spec_path,
        timeout_seconds=timeout_seconds,
    )
    if existing is not None and not rerun:
        return get_plan_view(root, plan_id)

    calibration_id = f"calibration-{uuid4().hex}"
    request_path = (
        registry.state_dir / "profiles" / "calibrations" / f"{calibration_id}-request.json"
    )
    receipt = run_calibration_backend(
        backend,
        plan,
        request_path=request_path,
        calibration_id=calibration_id,
        timeout_seconds=effective_timeout,
    )
    registry.store_calibration(receipt)
    _write_state_json(
        registry.state_dir / "profiles" / "calibrations" / f"{calibration_id}.json",
        asdict(receipt),
    )
    return get_plan_view(root, plan_id)


def _run_timeout_seconds(
    plan: TrainingPlan,
    calibration: dict[str, object],
    resource_profile: dict[str, object] | None,
    requested: float,
) -> float:
    timeout = requested
    if plan.budgets.max_wall_seconds is not None:
        timeout = min(timeout, plan.budgets.max_wall_seconds)

    if plan.budgets.max_gpu_hours is not None:
        gpu_count, _ = _accounted_gpu_count(calibration, resource_profile)
        if gpu_count is not None:
            timeout = min(
                timeout,
                plan.budgets.max_gpu_hours * 3600.0 / gpu_count,
            )
    return timeout


def get_run_view(root: Path, run_id: str) -> RunView:
    run = Registry(root).get_run(run_id)
    return RunView(
        run_id=run["run_id"],
        plan_id=run["plan_id"],
        status=run["status"],
        candidate_model_id=run["candidate_model_id"],
        metrics=run["metrics"],
        usage=run["usage"],
        error_code=run["error_code"],
        error_message=run["error_message"],
        liveness_state=run.get("liveness_state"),
        calibration_id=run.get("calibration_id"),
        request_digest=run.get("request_digest"),
        result_evidence_available=run.get("result") is not None,
    )


def _deterministic_candidate_id(run_id: str, fingerprint: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{fingerprint}".encode()).hexdigest()
    return f"model-{digest[:32]}"


def _run_usage_from_result(result: dict[str, object]) -> RunUsage | None:
    raw = result.get("usage")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise FrontierwrightError(
            "RUN_USAGE_INVALID",
            "Executor usage evidence must be an object.",
            14,
        )
    try:
        return RunUsage(
            wall_seconds=float(raw["wall_seconds"]),
            output_storage_bytes=(
                int(raw["output_storage_bytes"])
                if raw.get("output_storage_bytes") is not None
                else None
            ),
            gpu_count=(int(raw["gpu_count"]) if raw.get("gpu_count") is not None else None),
            gpu_count_provenance=(
                str(raw["gpu_count_provenance"])
                if raw.get("gpu_count_provenance") is not None
                else None
            ),
            accounted_gpu_hours=(
                float(raw["accounted_gpu_hours"])
                if raw.get("accounted_gpu_hours") is not None
                else None
            ),
            money_spent=(float(raw["money_spent"]) if raw.get("money_spent") is not None else None),
            measured_by=str(raw.get("measured_by") or "frontierwright-local-executor-v1"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "RUN_USAGE_INVALID",
            f"Executor usage evidence is invalid: {exc}",
            14,
        ) from exc


def _validate_executor_result(
    run: dict[str, object],
    result: dict[str, object],
) -> None:
    if result.get("schema_version") != 1:
        raise FrontierwrightError(
            "RUN_RESULT_INVALID",
            "Executor result schema_version must be 1.",
            14,
        )
    for key in ("run_id", "plan_id", "calibration_id", "request_digest"):
        if result.get(key) != run.get(key):
            raise FrontierwrightError(
                "RUN_RESULT_BINDING_MISMATCH",
                f"Executor result {key} does not match the durable attempt.",
                14,
            )


def _publish_run_receipt(root: Path, run_id: str) -> None:
    registry = Registry(root)
    view = get_run_view(root, run_id)
    _write_state_json(
        registry.state_dir / "runs" / f"{run_id}.json",
        view.to_dict(),
    )


def repair_run_receipt(root: Path, run_id: str) -> RunView:
    view = get_run_view(root, run_id)
    _publish_run_receipt(root, run_id)
    return view


def reconcile_training_run(root: Path, run_id: str) -> RunView:
    registry = Registry(root)
    run = registry.get_run(run_id)
    status = RunStatus(run["status"])
    if status is not RunStatus.RUNNING:
        return get_run_view(root, run_id)

    attempt = registry.get_run_attempt(run_id)
    if attempt is None:
        registry.finish_run_failure(
            run_id,
            status=RunStatus.INCOMPLETE,
            error_code="RUN_ATTEMPT_METADATA_MISSING",
            error_message="RUNNING attempt has no durable executor metadata.",
        )
        return get_run_view(root, run_id)

    result = attempt.get("result")
    result_path_raw = attempt.get("result_path")
    result_path = (
        Path(result_path_raw) if isinstance(result_path_raw, str) and result_path_raw else None
    )

    if result is None and result_path is not None and result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FrontierwrightError(
                "RUN_RESULT_INVALID",
                "Executor result evidence is not valid UTF-8 JSON.",
                14,
            ) from exc
        if not isinstance(loaded, dict):
            raise FrontierwrightError(
                "RUN_RESULT_INVALID",
                "Executor result evidence must be an object.",
                14,
            )
        result = loaded
        _validate_executor_result(run, result)
        registry.record_run_result_evidence(
            run_id,
            result_path=str(result_path),
            result_sha256=file_sha256(result_path),
            result=result,
        )
        run = registry.get_run(run_id)
        attempt = registry.get_run_attempt(run_id)
        if attempt is None:
            raise FrontierwrightError(
                "RUN_ATTEMPT_NOT_FOUND",
                "Run attempt metadata disappeared after evidence persistence.",
                4,
            )
    elif isinstance(result, dict):
        _validate_executor_result(run, result)

    if isinstance(result, dict):
        usage = _run_usage_from_result(result)
        if usage is not None:
            registry.record_run_usage(run_id, usage)
        if result.get("ok") is not True:
            error = result.get("error")
            error_map = error if isinstance(error, dict) else {}
            code = str(error_map.get("code") or "BACKEND_FAILED")
            message = str(error_map.get("message") or "Executor reported failure.")
            terminal_status = (
                RunStatus.INCOMPLETE
                if code in {"BACKEND_TIMEOUT", "EXECUTOR_LOST", "STORAGE_BUDGET_REACHED"}
                else RunStatus.FAILED
            )
            registry.finish_run_failure(
                run_id,
                status=terminal_status,
                error_code=code,
                error_message=message,
            )
            return get_run_view(root, run_id)

        output = result.get("output_model_path")
        metrics = result.get("metrics", {})
        if not isinstance(output, str) or not output:
            raise FrontierwrightError(
                "RUN_RESULT_INVALID",
                "Successful executor result lacks output_model_path.",
                14,
            )
        if not isinstance(metrics, dict):
            raise FrontierwrightError(
                "RUN_RESULT_INVALID",
                "Successful executor result metrics must be an object.",
                14,
            )

        plan = registry.get_plan(str(run["plan_id"]))
        if (
            usage is not None
            and plan.budgets.max_storage_bytes is not None
            and usage.output_storage_bytes is not None
            and usage.output_storage_bytes > plan.budgets.max_storage_bytes
        ):
            registry.finish_run_failure(
                run_id,
                status=RunStatus.INCOMPLETE,
                error_code="STORAGE_BUDGET_REACHED",
                error_message=("Measured executor output exceeds max_storage_bytes budget."),
            )
            return get_run_view(root, run_id)

        backend_descriptor = inspect_local_model(Path(output))
        if (
            plan.budgets.max_storage_bytes is not None
            and backend_descriptor.total_bytes > plan.budgets.max_storage_bytes
        ):
            registry.finish_run_failure(
                run_id,
                status=RunStatus.INCOMPLETE,
                error_code="STORAGE_BUDGET_REACHED",
                error_message="Candidate output exceeds max_storage_bytes budget.",
            )
            return get_run_view(root, run_id)

        candidate_model_id = _deterministic_candidate_id(
            run_id,
            backend_descriptor.fingerprint,
        )
        sealed = seal_training_artifact(
            registry.state_dir,
            source_path=Path(output),
            model_id=candidate_model_id,
            run_id=run_id,
            plan=plan,
        )

        state = registry.read()
        candidate = ModelState(
            model_id=candidate_model_id,
            identity_id=state.project["identity_id"],
            origin=ModelOrigin(state.project["origin"]),
            checkpoint=str(sealed.model_path),
            fingerprint=sealed.descriptor.fingerprint,
            parent_model_id=plan.model_id,
            stats=(),
            model_format=sealed.descriptor.model_format,
            trainable=sealed.descriptor.trainable,
        )
        registry.register_training_candidate(
            candidate,
            sealed,
            run_id=run_id,
            metrics=dict(metrics),
        )
        return get_run_view(root, run_id)

    worker_pid = attempt.get("worker_pid")
    worker_start_token = attempt.get("worker_start_token")
    pid = worker_pid if isinstance(worker_pid, int) else None
    token = worker_start_token if isinstance(worker_start_token, str) else None

    if attempt.get("liveness_state") == "RESERVED" and pid is None:
        registry.finish_run_failure(
            run_id,
            status=RunStatus.INCOMPLETE,
            error_code="EXECUTOR_NOT_LAUNCHED",
            error_message="Attempt was reserved but no local worker was attached.",
        )
        return get_run_view(root, run_id)

    liveness = process_liveness(pid, token)
    if liveness == "LIVE":
        registry.set_run_liveness(run_id, "LIVE")
        return get_run_view(root, run_id)
    if liveness == "DEAD":
        registry.finish_run_failure(
            run_id,
            status=RunStatus.INCOMPLETE,
            error_code="EXECUTOR_LOST",
            error_message="Local worker exited without durable terminal result evidence.",
        )
        return get_run_view(root, run_id)

    registry.set_run_liveness(run_id, "UNRESOLVED")
    return get_run_view(root, run_id)


def execute_training_plan(
    root: Path,
    *,
    plan_id: str,
    backend_spec_path: Path,
    dry_run: bool,
    rerun: bool,
    timeout_seconds: float = 86400.0,
) -> RunView:
    registry = Registry(root)
    plan = registry.get_plan(plan_id)

    if not dry_run:
        latest = registry.latest_run_for_plan(plan.plan_id)
        if latest is not None:
            latest_status = RunStatus(latest["status"])
            if not rerun:
                if latest_status is RunStatus.COMPLETED:
                    view = get_run_view(root, str(latest["run_id"]))
                    try:
                        _publish_run_receipt(root, str(latest["run_id"]))
                    except FrontierwrightError:
                        pass
                    return view
                if latest_status is RunStatus.RUNNING:
                    reconciled = reconcile_training_run(root, str(latest["run_id"]))
                    if reconciled.status in {
                        RunStatus.RUNNING.value,
                        RunStatus.COMPLETED.value,
                    }:
                        try:
                            _publish_run_receipt(root, str(latest["run_id"]))
                        except FrontierwrightError:
                            pass
                        return reconciled
                    raise FrontierwrightError(
                        "RUN_RERUN_REQUIRED",
                        "Previous attempt became terminal; use --rerun to create a new attempt.",
                        13,
                    )
                raise FrontierwrightError(
                    "RUN_RERUN_REQUIRED",
                    "Previous attempt is terminal without success; use --rerun to retry.",
                    13,
                )

            if latest_status is RunStatus.RUNNING:
                reconciled = reconcile_training_run(root, str(latest["run_id"]))
                if reconciled.status == RunStatus.RUNNING.value:
                    raise FrontierwrightError(
                        "RUN_ACTIVE",
                        "Cannot rerun while the prior attempt is live or unresolved.",
                        13,
                    )

    required_permission = PermissionLevel.DRY_RUN if dry_run else PermissionLevel.EXECUTE_SINGLE
    if plan.permission < required_permission:
        raise FrontierwrightError(
            "PERMISSION_DENIED",
            f"Operation requires {required_permission.name} permission or higher.",
            13,
        )

    state = registry.read()
    input_blockers = _plan_input_blockers(registry, state, plan)
    if input_blockers:
        raise FrontierwrightError(
            "PLAN_INPUT_STALE",
            "; ".join(input_blockers),
            13,
        )
    _validate_backend_for_plan(plan, backend_spec_path)
    calibration = registry.latest_calibration_for_plan(plan.plan_id)
    budget_blockers = _calibration_budget_blockers(
        plan,
        calibration,
        state.resource_profile,
    )
    if budget_blockers:
        raise FrontierwrightError(
            "PLAN_NOT_READY",
            "; ".join(budget_blockers),
            13,
        )
    assert calibration is not None

    if dry_run:
        return RunView(
            plan_id=plan.plan_id,
            status="DRY_RUN",
            dry_run=True,
        )

    if timeout_seconds <= 0:
        raise FrontierwrightError("INVALID_TIMEOUT", "Run timeout must be positive.", 2)

    calibration_id = calibration.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise FrontierwrightError(
            "CALIBRATION_INVALID",
            "Latest calibration lacks a durable calibration_id.",
            14,
        )
    request_digest = compute_execution_request_digest(
        plan,
        calibration_id=calibration_id,
    )

    run_id, existing = registry.start_run(
        plan.plan_id,
        rerun=rerun,
        calibration_id=calibration_id,
        request_digest=request_digest,
        owner_pid=os.getpid(),
    )
    if existing:
        existing_view = get_run_view(root, run_id)
        if existing_view.status == RunStatus.RUNNING.value:
            existing_view = reconcile_training_run(root, run_id)
        return existing_view

    effective_timeout = _run_timeout_seconds(
        plan,
        calibration,
        state.resource_profile,
        timeout_seconds,
    )
    gpu_count, gpu_count_provenance = _accounted_gpu_count(
        calibration,
        state.resource_profile,
    )
    attempt_path = registry.state_dir / "runs" / f"{run_id}-attempt.json"
    request_path = registry.state_dir / "runs" / f"{run_id}-backend-request.json"
    result_path = registry.state_dir / "runs" / f"{run_id}-executor-result.json"
    output_root = registry.state_dir / "candidates" / run_id
    attempt = LocalAttemptSpec(
        project_root=registry.root,
        run_id=run_id,
        plan_id=plan.plan_id,
        calibration_id=calibration_id,
        backend_spec_path=backend_spec_path.resolve(),
        request_path=request_path,
        output_root=output_root,
        result_path=result_path,
        request_digest=request_digest,
        timeout_seconds=effective_timeout,
        gpu_count=gpu_count,
        gpu_count_provenance=gpu_count_provenance,
    )

    try:
        atomic_write_json(attempt_path, attempt.to_dict())
        worker = launch_worker(attempt_path)
        start_token = process_start_token(worker.pid)
        registry.attach_run_worker(
            run_id,
            worker_pid=worker.pid,
            worker_start_token=start_token,
            result_path=str(result_path),
        )
    except (OSError, FrontierwrightError) as exc:
        code = exc.code if isinstance(exc, FrontierwrightError) else "EXECUTOR_LAUNCH_FAILED"
        registry.finish_run_failure(
            run_id,
            status=RunStatus.FAILED,
            error_code=code,
            error_message=str(exc),
        )
        raise

    try:
        worker.wait(timeout=effective_timeout + 15.0)
    except subprocess.TimeoutExpired:
        try:
            terminate_worker_tree(worker)
        except FrontierwrightError:
            registry.set_run_liveness(run_id, "UNRESOLVED")
            return get_run_view(root, run_id)

    reconciled = reconcile_training_run(root, run_id)
    try:
        _publish_run_receipt(root, run_id)
    except FrontierwrightError:
        # SQLite and durable executor evidence are authoritative. Receipt export
        # is repairable and must never downgrade a committed outcome.
        pass

    if reconciled.status in {RunStatus.FAILED.value, RunStatus.INCOMPLETE.value}:
        raise FrontierwrightError(
            reconciled.error_code or "RUN_FAILED",
            reconciled.error_message or "Training attempt did not complete.",
            14,
        )
    return reconciled


def get_candidates_view(root: Path) -> CandidateView:
    registry = Registry(root)
    if not registry.exists:
        return CandidateView()

    state = registry.read()
    items: list[dict[str, object]] = []
    for candidate in state.candidates:
        stats = get_stats_view(root, candidate.model.model_id)
        run = next(
            (
                item
                for item in reversed(state.runs)
                if item.get("candidate_model_id") == candidate.model.model_id
            ),
            None,
        )
        plan = None
        if run is not None:
            plan_id = run.get("plan_id")
            plan = next(
                (item for item in state.plans if item.get("plan_id") == plan_id),
                None,
            )
        items.append(
            {
                "model_id": candidate.model.model_id,
                "parent_model_id": candidate.model.parent_model_id,
                "status": candidate.status.value,
                "fingerprint": candidate.model.fingerprint,
                "model_format": candidate.model.model_format.value,
                "trainable": candidate.model.trainable,
                "measured": stats.measured,
                "stats": stats.stats,
                "scale_id": stats.scale_id,
                "scale_version": stats.scale_version,
                "scale_hash": stats.scale_hash,
                "run_id": run.get("run_id") if run is not None else None,
                "path_id": plan.get("path_id") if plan is not None else None,
            }
        )
    return CandidateView(candidates=items)


def _candidate_build_constraints(
    build: dict[str, object] | None,
    candidate_stats: StatsView,
) -> list[dict[str, object]]:
    constraints: list[dict[str, object]] = []
    if build is None or build.get("mode") != BuildMode.TARGETS_FLOORS.value:
        return constraints

    for kind, values in (
        ("FLOOR", build.get("floors", {})),
        ("TARGET", build.get("targets", {})),
    ):
        if not isinstance(values, dict):
            continue
        for axis, threshold in sorted(values.items()):
            value = candidate_stats.stats.get(str(axis))
            if value is None:
                result = "UNKNOWN"
            elif kind == "FLOOR":
                result = "PASS" if value >= float(threshold) else "FAIL"
            else:
                result = "REACHED" if value >= float(threshold) else "NOT_REACHED"
            constraints.append(
                {
                    "axis": axis,
                    "kind": kind,
                    "threshold": threshold,
                    "candidate_value": value,
                    "status": result,
                }
            )
    return constraints


def _promotion_blockers(
    *,
    candidate_status: CandidateStatus,
    champion_stats: StatsView,
    candidate_stats: StatsView,
    build: dict[str, object] | None,
    constraints: list[dict[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []

    if candidate_status is not CandidateStatus.PENDING:
        blockers.append(
            {
                "code": "CANDIDATE_STATE_CONFLICT",
                "message": (
                    "Only PENDING candidates can be promoted; current status is "
                    f"{candidate_status.value}."
                ),
                "override": None,
            }
        )

    if not candidate_stats.measured:
        blockers.append(
            {
                "code": "CANDIDATE_NOT_MEASURED",
                "message": (
                    "Candidate must be evaluated before promotion; use "
                    "--allow-unmeasured explicitly to override."
                ),
                "override": "--allow-unmeasured",
            }
        )

    if (
        candidate_stats.measured
        and champion_stats.measured
        and (
            candidate_stats.scale_hash is None
            or champion_stats.scale_hash is None
            or candidate_stats.scale_hash != champion_stats.scale_hash
        )
    ):
        blockers.append(
            {
                "code": "CAPABILITY_SCALE_MISMATCH",
                "message": ("Champion and candidate do not use the same frozen capability scale."),
                "override": "--allow-unmeasured",
            }
        )

    floors = (
        build.get("floors", {})
        if build is not None and build.get("mode") == BuildMode.TARGETS_FLOORS.value
        else {}
    )
    if isinstance(floors, dict) and floors:
        build_scale_hash = build.get("scale_hash") if build is not None else None
        if not isinstance(build_scale_hash, str) or not build_scale_hash:
            blockers.append(
                {
                    "code": "BUILD_SCALE_UNBOUND",
                    "message": (
                        "Build floors are not bound to an exact frozen capability scale; "
                        "re-save the numeric build before promotion."
                    ),
                    "override": "--allow-build-violations",
                }
            )
        elif candidate_stats.scale_hash != build_scale_hash:
            blockers.append(
                {
                    "code": "BUILD_SCALE_MISMATCH",
                    "message": (
                        "Candidate capability scale does not match the scale pinned by "
                        "the current build floors."
                    ),
                    "override": "--allow-build-violations",
                }
            )
        else:
            for item in constraints:
                if item.get("kind") != "FLOOR":
                    continue
                status = item.get("status")
                axis = str(item.get("axis"))
                if status == "UNKNOWN":
                    blockers.append(
                        {
                            "code": "BUILD_FLOOR_UNMEASURED",
                            "message": f"Build floor axis {axis} is not measured.",
                            "axis": axis,
                            "override": "--allow-build-violations",
                        }
                    )
                elif status == "FAIL":
                    blockers.append(
                        {
                            "code": "BUILD_FLOOR_VIOLATION",
                            "message": (
                                f"Candidate is below the configured build floor for {axis}."
                            ),
                            "axis": axis,
                            "override": "--allow-build-violations",
                        }
                    )

    return blockers


def _paired_capability_evidence(
    registry: Registry,
    *,
    champion_model_id: str,
    candidate_model_id: str,
) -> dict[str, object]:
    def item_evidence(model_id: str) -> tuple[str, dict[str, dict[str, object]]] | None:
        profile = registry.get_active_capability_profile(model_id)
        if profile is None or not isinstance(profile.get("receipt_id"), str):
            return None
        row = registry.get_evaluation_receipt(str(profile["receipt_id"]))
        if row is None:
            return None
        receipt = _receipt_from_registry_row(row)
        if (
            receipt.evaluator_id != CAPABILITY_V1_EVALUATOR_ID
            or receipt.conditions.get("bundle_hash") != capability_v1_bundle_hash()
        ):
            return None
        raw_items = receipt.conditions.get("item_results")
        if not isinstance(raw_items, list):
            return None
        items: dict[str, dict[str, object]] = {}
        for raw in raw_items:
            if not isinstance(raw, dict):
                return None
            item_id = raw.get("item_id")
            axis = raw.get("axis")
            correct = raw.get("correct")
            if (
                not isinstance(item_id, str)
                or not isinstance(axis, str)
                or not isinstance(correct, bool)
                or item_id in items
            ):
                return None
            items[item_id] = dict(raw)
        if len(items) != len(CAPABILITY_V1_TASKS):
            return None
        return receipt.receipt_id, items

    champion = item_evidence(champion_model_id)
    candidate = item_evidence(candidate_model_id)
    if champion is None or candidate is None:
        return {
            "available": False,
            "method": "mcnemar-exact-binomial-two-sided-v1",
            "reason": (
                "Per-item Capability v1 evidence is unavailable for at least one model. "
                "Older aggregate-only receipts remain valid but cannot support paired flips."
            ),
        }
    champion_receipt_id, champion_items = champion
    candidate_receipt_id, candidate_items = candidate
    expected_ids = [task.item_id for task in CAPABILITY_V1_TASKS]
    if set(champion_items) != set(expected_ids) or set(candidate_items) != set(expected_ids):
        return {
            "available": False,
            "method": "mcnemar-exact-binomial-two-sided-v1",
            "reason": "Champion and candidate do not expose the same frozen item set.",
        }

    axes: dict[str, object] = {}
    overall_before: list[bool] = []
    overall_after: list[bool] = []
    for axis in ("general", "reasoning", "math", "coding"):
        axis_ids = [task.item_id for task in CAPABILITY_V1_TASKS if task.axis.value.lower() == axis]
        before = tuple(bool(champion_items[item_id]["correct"]) for item_id in axis_ids)
        after = tuple(bool(candidate_items[item_id]["correct"]) for item_id in axis_ids)
        evidence = exact_mcnemar_paired_binary(before, after)
        improvement_ids = [
            item_id
            for item_id, old, new in zip(axis_ids, before, after, strict=True)
            if not old and new
        ]
        regression_ids = [
            item_id
            for item_id, old, new in zip(axis_ids, before, after, strict=True)
            if old and not new
        ]
        axes[axis] = {
            **evidence,
            "improvement_item_ids": improvement_ids,
            "regression_item_ids": regression_ids,
        }
        overall_before.extend(before)
        overall_after.extend(after)

    overall = exact_mcnemar_paired_binary(tuple(overall_before), tuple(overall_after))
    return {
        "available": True,
        "method": "mcnemar-exact-binomial-two-sided-v1",
        "champion_receipt_id": champion_receipt_id,
        "candidate_receipt_id": candidate_receipt_id,
        "overall": overall,
        "axes": axes,
        "note": (
            "Same-item correct/incorrect flips are paired. Statistical detectability is "
            "evidence about change, not an automatic promotion rule or proof of equivalence."
        ),
    }


def _candidate_pareto_evidence(
    registry: Registry,
    *,
    champion_model_id: str,
    candidate_model_id: str,
    champion_stats: StatsView,
    candidate_stats: StatsView,
    scale_comparable: bool,
) -> dict[str, object]:
    state = registry.read()
    champion_profile = _latest_model_fit_from_state(state, champion_model_id)
    candidate_profile = _latest_model_fit_from_state(state, candidate_model_id)

    inference_comparable = bool(champion_profile and candidate_profile)
    mismatch: list[str] = []
    if inference_comparable:
        champion_contract = champion_profile.get("profile_condition_hash")
        candidate_contract = candidate_profile.get("profile_condition_hash")
        if isinstance(champion_contract, str) or isinstance(candidate_contract, str):
            if not (
                isinstance(champion_contract, str)
                and isinstance(candidate_contract, str)
                and champion_contract == candidate_contract
            ):
                mismatch.append("profile_condition_hash")
            for key in ("measurement_scope", "execution_boundary"):
                if champion_profile.get(key) != candidate_profile.get(key):
                    mismatch.append(key)
        else:
            for key in (
                "measurement_scope",
                "execution_boundary",
                "device",
                "max_new_tokens",
            ):
                if champion_profile.get(key) != candidate_profile.get(key):
                    mismatch.append(key)
        inference_comparable = not mismatch

    metrics: list[ParetoMetricInput] = []
    capability_source = (
        f"CAPABILITY_SCALE:{champion_stats.scale_hash}"
        if scale_comparable and champion_stats.scale_hash is not None
        else None
    )
    for axis in ("general", "reasoning", "math", "coding"):
        metrics.append(
            ParetoMetricInput(
                key=f"capability.{axis}",
                category="CAPABILITY",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=(champion_stats.stats.get(axis) if scale_comparable else None),
                candidate_value=(candidate_stats.stats.get(axis) if scale_comparable else None),
                unit="frontierwright-capability-v1",
                evidence_source=capability_source,
            )
        )

    def measured(profile: dict[str, object], key: str) -> float | int | None:
        if not inference_comparable:
            return None
        value = profile.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    inference_source = "INFERENCE_PROFILE_COMPARABLE" if inference_comparable else None
    metrics.extend(
        [
            ParetoMetricInput(
                key="serving.latency_p50",
                category="SERVING",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=measured(champion_profile, "latency_seconds_p50"),
                candidate_value=measured(candidate_profile, "latency_seconds_p50"),
                unit="seconds",
                evidence_source=inference_source,
            ),
            ParetoMetricInput(
                key="serving.throughput_p50",
                category="SERVING",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=measured(champion_profile, "tokens_per_second_p50"),
                candidate_value=measured(candidate_profile, "tokens_per_second_p50"),
                unit="tokens/second",
                evidence_source=inference_source,
            ),
            ParetoMetricInput(
                key="serving.ttft_p50",
                category="SERVING",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=measured(champion_profile, "ttft_seconds_p50"),
                candidate_value=measured(candidate_profile, "ttft_seconds_p50"),
                unit="seconds",
                evidence_source=inference_source,
            ),
            ParetoMetricInput(
                key="serving.tpot_p50",
                category="SERVING",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=measured(champion_profile, "tpot_seconds_p50"),
                candidate_value=measured(candidate_profile, "tpot_seconds_p50"),
                unit="seconds/token",
                evidence_source=inference_source,
            ),
            ParetoMetricInput(
                key="serving.itl_p50",
                category="SERVING",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=measured(champion_profile, "itl_seconds_p50"),
                candidate_value=measured(candidate_profile, "itl_seconds_p50"),
                unit="seconds",
                evidence_source=inference_source,
            ),
            ParetoMetricInput(
                key="serving.output_throughput_aggregate",
                category="SERVING",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=measured(champion_profile, "output_tokens_per_second_aggregate"),
                candidate_value=measured(candidate_profile, "output_tokens_per_second_aggregate"),
                unit="tokens/second",
                evidence_source=inference_source,
            ),
            ParetoMetricInput(
                key="resource.peak_vram",
                category="RESOURCE",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=measured(champion_profile, "peak_vram_bytes"),
                candidate_value=measured(candidate_profile, "peak_vram_bytes"),
                unit="bytes",
                evidence_source=inference_source,
            ),
            ParetoMetricInput(
                key="resource.process_rss",
                category="RESOURCE",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=measured(champion_profile, "max_sampled_process_rss_bytes"),
                candidate_value=measured(candidate_profile, "max_sampled_process_rss_bytes"),
                unit="bytes",
                evidence_source=inference_source,
            ),
        ]
    )

    champion_artifact = registry.get_model_artifact(champion_model_id) or {}
    candidate_artifact = registry.get_model_artifact(candidate_model_id) or {}
    metrics.append(
        ParetoMetricInput(
            key="storage.model_artifact",
            category="RESOURCE",
            direction=ParetoDirection.LOWER_BETTER,
            champion_value=(
                champion_artifact.get("total_bytes")
                if isinstance(champion_artifact.get("total_bytes"), int)
                else None
            ),
            candidate_value=(
                candidate_artifact.get("total_bytes")
                if isinstance(candidate_artifact.get("total_bytes"), int)
                else None
            ),
            unit="bytes",
            evidence_source="MODEL_ARTIFACT_MANIFEST",
        )
    )

    comparison = compare_pareto_metrics(metrics)
    payload = comparison.to_payload()
    workload_row = state.workload_profile
    if workload_row is not None and isinstance(workload_row.get("profile"), dict):
        profile = workload_profile_from_payload(dict(workload_row["profile"]))
        payload["explicit_user_utility"] = compare_explicit_user_utility(
            profile, comparison
        ).to_payload()
    else:
        payload["explicit_user_utility"] = {
            "status": "NOT_CONFIGURED",
            "relation": "UNKNOWN",
            "utility_delta": None,
            "contributions": [],
            "missing_metrics": [],
            "note": "No active workload profile defines explicit user utility settings.",
        }
    payload["inference_profile_comparable"] = inference_comparable
    if not champion_profile or not candidate_profile:
        payload["inference_profile_reason"] = (
            "Profile both Champion and Candidate before comparing runtime/resource dimensions."
        )
    elif mismatch:
        payload["inference_profile_reason"] = (
            "Inference profiles use different conditions: " + ", ".join(mismatch)
        )
    else:
        payload["inference_profile_reason"] = (
            "Champion and Candidate inference profiles use comparable measured conditions."
        )
    return payload


def compare_candidate(root: Path, candidate_model_id: str) -> CompareView:
    registry = Registry(root)
    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required for candidate comparison.",
            12,
        )
    candidate = registry.get_candidate(candidate_model_id)
    _verify_model_artifact_integrity(registry, state.champion.model.model_id)
    _verify_model_artifact_integrity(registry, candidate.model.model_id)
    champion_stats = get_stats_view(root, state.champion.model.model_id)
    candidate_stats = get_stats_view(root, candidate.model.model_id)

    comparable = bool(
        champion_stats.measured
        and candidate_stats.measured
        and champion_stats.scale_hash is not None
        and champion_stats.scale_hash == candidate_stats.scale_hash
    )
    if comparable:
        scale_reason = "Champion and candidate use the same frozen capability scale."
    elif not champion_stats.measured:
        scale_reason = "Champion is not measured under a frozen capability scale."
    elif not candidate_stats.measured:
        scale_reason = "Candidate is not measured under a frozen capability scale."
    elif champion_stats.scale_hash is None or candidate_stats.scale_hash is None:
        scale_reason = "At least one model lacks comparable frozen-scale provenance."
    else:
        scale_reason = "Champion and candidate use different frozen capability scales."

    deltas = _empty_stats()
    if comparable:
        for axis in deltas:
            champion_value = champion_stats.stats.get(axis)
            candidate_value = candidate_stats.stats.get(axis)
            if champion_value is not None and candidate_value is not None:
                deltas[axis] = candidate_value - champion_value

    build = state.build_state
    constraints = _candidate_build_constraints(build, candidate_stats)
    promotion_blockers = _promotion_blockers(
        candidate_status=candidate.status,
        champion_stats=champion_stats,
        candidate_stats=candidate_stats,
        build=build,
        constraints=constraints,
    )

    champion_workload = get_workload_fit(root, state.champion.model.model_id)
    candidate_workload = get_workload_fit(root, candidate.model.model_id)
    champion_constraints = {
        str(item.get("key")): item
        for item in champion_workload.constraints
        if isinstance(item.get("key"), str)
    }
    candidate_constraints = {
        str(item.get("key")): item
        for item in candidate_workload.constraints
        if isinstance(item.get("key"), str)
    }
    workload_regressions: list[dict[str, object]] = []
    for key, champion_constraint in champion_constraints.items():
        candidate_constraint = candidate_constraints.get(key)
        if candidate_constraint is None:
            continue
        champion_status = champion_constraint.get("status")
        candidate_status = candidate_constraint.get("status")
        if champion_status == "PASS" and candidate_status != "PASS":
            workload_regressions.append(
                {
                    "key": key,
                    "champion_status": champion_status,
                    "candidate_status": candidate_status,
                    "champion_observed": champion_constraint.get("observed"),
                    "candidate_observed": candidate_constraint.get("observed"),
                    "requirement": candidate_constraint.get("requirement"),
                }
            )
    workload_comparison: dict[str, object] = {
        "configured": champion_workload.configured or candidate_workload.configured,
        "profile_id": candidate_workload.workload_profile_id,
        "profile_hash": candidate_workload.workload_profile_hash,
        "champion": champion_workload.to_dict(),
        "candidate": candidate_workload.to_dict(),
        "regressions": workload_regressions,
        "note": (
            "Workload fit remains constraint-by-constraint evidence; no synthetic utility "
            "score is used for promotion decisions."
        ),
    }
    pareto = _candidate_pareto_evidence(
        registry,
        champion_model_id=state.champion.model.model_id,
        candidate_model_id=candidate.model.model_id,
        champion_stats=champion_stats,
        candidate_stats=candidate_stats,
        scale_comparable=comparable,
    )

    run = next(
        (
            item
            for item in reversed(state.runs)
            if item.get("candidate_model_id") == candidate.model.model_id
        ),
        None,
    )
    calibration = None
    if run is not None:
        run_plan_id = run.get("plan_id")
        calibration = next(
            (item for item in reversed(state.calibrations) if item.get("plan_id") == run_plan_id),
            None,
        )

    return CompareView(
        champion_model_id=state.champion.model.model_id,
        candidate_model_id=candidate.model.model_id,
        candidate_status=candidate.status.value,
        scale_comparable=comparable,
        scale_reason=scale_reason,
        champion_stats=champion_stats.stats,
        candidate_stats=candidate_stats.stats,
        deltas=deltas,
        build_constraints=constraints,
        promotion_eligible=not promotion_blockers,
        promotion_blockers=promotion_blockers,
        build_scale_hash=(
            str(build["scale_hash"])
            if build is not None and isinstance(build.get("scale_hash"), str)
            else None
        ),
        raw_evaluation_comparisons=_stored_raw_evaluation_comparisons(
            registry,
            champion_model_id=state.champion.model.model_id,
            candidate_model_id=candidate.model.model_id,
        ),
        paired_capability_evidence=_paired_capability_evidence(
            registry,
            champion_model_id=state.champion.model.model_id,
            candidate_model_id=candidate.model.model_id,
        ),
        workload_comparison=workload_comparison,
        pareto=pareto,
        run=dict(run) if run is not None else None,
        calibration=dict(calibration) if calibration is not None else None,
    )


def promote_candidate(
    root: Path,
    candidate_model_id: str,
    *,
    allow_unmeasured: bool = False,
    allow_build_violations: bool = False,
) -> StatusView:
    registry = Registry(root)
    state = registry.read()
    if state.champion is None:
        raise FrontierwrightError(
            "NO_CHAMPION_MODEL",
            "A current champion is required before promotion.",
            12,
        )

    candidate = registry.get_candidate(candidate_model_id)
    _verify_model_artifact_integrity(registry, state.champion.model.model_id)
    _verify_model_artifact_integrity(registry, candidate.model.model_id)

    champion_stats = get_stats_view(root, state.champion.model.model_id)
    candidate_stats = get_stats_view(root, candidate_model_id)
    constraints = _candidate_build_constraints(state.build_state, candidate_stats)
    blockers = _promotion_blockers(
        candidate_status=candidate.status,
        champion_stats=champion_stats,
        candidate_stats=candidate_stats,
        build=state.build_state,
        constraints=constraints,
    )

    overridden: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for blocker in blockers:
        override = blocker.get("override")
        if override == "--allow-unmeasured" and allow_unmeasured:
            overridden.append(blocker)
        elif override == "--allow-build-violations" and allow_build_violations:
            overridden.append(blocker)
        else:
            unresolved.append(blocker)

    if unresolved:
        first = unresolved[0]
        raise FrontierwrightError(
            str(first.get("code") or "PROMOTION_BLOCKED"),
            str(first.get("message") or "Candidate promotion is blocked."),
            13,
        )

    build_updated_at = (
        str(state.build_state["updated_at"])
        if state.build_state is not None and isinstance(state.build_state.get("updated_at"), str)
        else None
    )
    expected_state = {
        "champion_id": state.champion.model.model_id,
        "build_updated_at": build_updated_at,
        "champion_profile_id": champion_stats.profile_id,
        "candidate_profile_id": candidate_stats.profile_id,
    }
    decision_details: dict[str, object] = {
        "gate_default_eligible": not blockers,
        "allow_unmeasured": allow_unmeasured,
        "allow_build_violations": allow_build_violations,
        "overridden_blockers": [
            {
                "code": item.get("code"),
                "axis": item.get("axis"),
            }
            for item in overridden
        ],
        "champion_profile_id": champion_stats.profile_id,
        "candidate_profile_id": candidate_stats.profile_id,
        "champion_scale_hash": champion_stats.scale_hash,
        "candidate_scale_hash": candidate_stats.scale_hash,
        "build_scale_hash": (
            state.build_state.get("scale_hash") if state.build_state is not None else None
        ),
        "build_updated_at": build_updated_at,
    }
    registry.promote_candidate(
        candidate_model_id,
        expected_state=expected_state,
        decision_details=decision_details,
    )
    return get_status(root)


def reject_candidate(root: Path, candidate_model_id: str) -> CandidateView:
    registry = Registry(root)
    registry.reject_candidate(candidate_model_id)
    return get_candidates_view(root)


def get_history_view(root: Path) -> HistoryView:
    registry = Registry(root)
    if not registry.exists:
        return HistoryView()
    state = registry.read()
    events = [
        {
            "sequence": event["sequence"],
            "kind": event["kind"],
            "recorded_at": event["recorded_at"],
            "details": event["details"],
        }
        for event in state.history
    ]
    return HistoryView(events=events)


def get_interventions_view() -> InterventionsView:
    return InterventionsView(
        interventions=[plugin.descriptor.machine_payload() for plugin in intervention_plugins()]
    )


def get_paths_view(root: Path) -> PathsView:
    registry = Registry(root)
    if not registry.exists:
        return PathsView(recommendation_reason="Project is not initialized.")

    state = registry.read()
    origin = ModelOrigin(state.project["origin"])
    confidence = HistoryConfidence(state.project["history_confidence"])
    roles = frozenset(DatasetRole(item["role"]) for item in state.datasets)
    champion_birth = (
        registry.get_model_birth(state.champion.model.model_id)
        if state.champion is not None
        else None
    )

    context = PathContext(
        origin=origin,
        champion_present=state.champion is not None,
        champion_trainable=(state.champion.model.trainable if state.champion is not None else None),
        history_confidence=confidence,
        resource_profile_available=state.resource_profile is not None,
        dataset_roles=roles,
        champion_is_birth_root=champion_birth is not None,
    )
    assessments = assess_training_interventions(context)

    paths: list[dict[str, object]] = []
    for intervention, item in assessments:
        payload: dict[str, object] = {
            "path_id": item.path_id.value,
            "intervention_id": intervention.intervention_id,
            "intervention_family": intervention.family.value,
            "intervention_version": intervention.version,
            "title": item.title,
            "availability": item.availability.value,
            "blockers": list(item.blockers),
            "next_checks": list(item.next_checks),
            "recommendation_eligible": item.recommendation_eligible,
        }

        if item.availability is PathAvailability.PLANNABLE:
            for stored_plan in reversed(state.plans):
                if stored_plan.get("path_id") != item.path_id.value:
                    continue
                plan_id = stored_plan.get("plan_id")
                if not isinstance(plan_id, str):
                    continue
                try:
                    plan_view = get_plan_view(root, plan_id)
                except FrontierwrightError:
                    continue
                if not plan_view.ready:
                    continue
                payload["availability"] = PathAvailability.READY.value
                payload["blockers"] = []
                payload["next_checks"] = []
                payload["ready_plan_id"] = plan_view.plan_id
                payload["backend_id"] = plan_view.backend_id
                payload["calibration"] = plan_view.calibration
                break

        paths.append(payload)

    if not confidence.allows_recommendation:
        recommendation_reason = (
            "History-aware recommendation is withheld until history is COMPLETE or VERIFIED."
        )
    else:
        recommendation_reason = (
            "No effect/recommendation model is implemented yet; showing factual "
            "path availability only."
        )

    return PathsView(
        paths=paths,
        recommended_path=None,
        recommendation_reason=recommendation_reason,
    )


def build_mode_for(root: Path) -> BuildMode | None:
    registry = Registry(root)
    if not registry.exists:
        return None
    state = registry.read()
    return build_mode(ModelOrigin(state.project["origin"]), state.champion)
