"""Shared application/service layer used by both CLI/JSON and TUI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from frontierwright.artifact_store import (
    seal_training_artifact,
    verify_manifest_digest,
    verify_sealed_artifact,
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
    ModelOrigin,
    ModelState,
    build_mode,
)
from frontierwright.editions import EditionProfile, policy_for
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import apply_scale, load_capability_scale, load_evaluation_receipt
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
from frontierwright.interventions import (
    assess_training_interventions,
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
    SNAPSHOT_COPY_PLUGIN_ID,
    TEXT_LINES_PLUGIN_ID,
    WEIGHTED_TEXT_MIXTURE_PLUGIN_ID,
    DataPreparationPlugin,
    DataPreparationRecipe,
    data_preparation_plugin,
)
from frontierwright.reference_backend import PRESETS, REFERENCE_BACKEND_ID
from frontierwright.registry import ProjectState, Registry
from frontierwright.resources import detect_local_resources


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
    runtime: dict[str, object] = field(default_factory=dict)

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
class CandidateView:
    schema_version: int = 1
    candidates: list[dict[str, object]] = field(default_factory=list)

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
        resource_profile_available=state.resource_profile is not None,
    )


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
        runtime=dict(runtime_map),
    )


def birth_zero_model(
    root: Path,
    *,
    preset: str,
    seed: int,
    python_executable: str,
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
    if state.champion is not None:
        existing = registry.get_model_birth(state.champion.model.model_id)
        if (
            existing is not None
            and existing.get("preset") == preset
            and existing.get("seed") == seed
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
            (
                str(state.project["project_id"])
                + "\0"
                + staged_descriptor.fingerprint
            ).encode("utf-8")
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
        )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return get_birth_view(root)


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
        state.capability_profile is not None
        and state.capability_profile.get("stats")
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
            scale_version=(
                scale_version if isinstance(scale_version, str) else None
            ),
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
                (
                    "Preparation source content changed after registration: "
                    f"{binding.dataset_id}"
                ),
                13,
            )
        resolved[binding.dataset_id] = descriptor
    return resolved


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
        (
            item
            for item in state.datasets
            if item.get("dataset_id") == dataset_id
        ),
        None,
    )
    if source is None:
        raise FrontierwrightError(
            "DATASET_NOT_FOUND",
            "Source dataset is not active in this project.",
            3,
        )
    plugin = data_preparation_plugin(plugin_id)
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
    classifications = [
        DatasetClassification(str(item["classification"])) for item in selected
    ]
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
    ):
        return DatasetRole.PRETRAIN
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
            item
            for item in matching
            if isinstance(item.get("preparation_recipe_hash"), str)
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
        (
            item
            for item in project_state.datasets
            if item.get("dataset_id") == plan.dataset_id
        ),
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
    if (
        isinstance(reported, int)
        and not isinstance(reported, bool)
        and reported > 0
    ):
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
    if (
        calibration.get("peak_vram_bytes") is None
        and calibration.get("peak_ram_bytes") is None
    ):
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
        if not isinstance(projected_money, (int, float)) or isinstance(
            projected_money, bool
        ):
            blockers.append("money budget cannot be evaluated without projected_money")
        elif float(projected_money) > budgets.max_money:
            blockers.append("projected money exceeds max_money budget")
        else:
            blockers.append(
                "max_money cannot be hard-enforced by the local executor yet"
            )

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
                modes["max_gpu_hours"] = (
                    f"HARD_ACCOUNTED_TIMEOUT:{provenance}"
                )
    if plan.budgets.max_storage_bytes is not None:
        modes["max_storage_bytes"] = "ADMISSION_AND_FINALIZATION_GATE"
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
    }.intersection(kinds):
        return ["pinned Lab adapter no longer provides trainer/executor capability"]
    return []


def get_plan_view(root: Path, plan_id: str) -> PlanView:
    registry = Registry(root)
    plan = registry.get_plan(plan_id)
    state = registry.read()
    calibration = registry.latest_calibration_for_plan(plan.plan_id)
    blockers = _plan_input_blockers(registry, state, plan)
    blockers.extend(
        _calibration_budget_blockers(plan, calibration, state.resource_profile)
    )
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
            else (
                state.champion.model.checkpoint
                if state.champion is not None
                else None
            )
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


def calibrate_training_plan(
    root: Path,
    *,
    plan_id: str,
    backend_spec_path: Path,
    timeout_seconds: float = 300.0,
) -> PlanView:
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

    calibration_id = f"calibration-{uuid4().hex}"
    request_path = (
        registry.state_dir
        / "profiles"
        / "calibrations"
        / f"{calibration_id}-request.json"
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
            gpu_count=(
                int(raw["gpu_count"])
                if raw.get("gpu_count") is not None
                else None
            ),
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
            money_spent=(
                float(raw["money_spent"])
                if raw.get("money_spent") is not None
                else None
            ),
            measured_by=str(
                raw.get("measured_by") or "frontierwright-local-executor-v1"
            ),
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
        Path(result_path_raw)
        if isinstance(result_path_raw, str) and result_path_raw
        else None
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
                error_message=(
                    "Measured executor output exceeds max_storage_bytes budget."
                ),
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

    required_permission = (
        PermissionLevel.DRY_RUN if dry_run else PermissionLevel.EXECUTE_SINGLE
    )
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
                "message": (
                    "Champion and candidate do not use the same frozen capability scale."
                ),
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
            (
                item
                for item in reversed(state.calibrations)
                if item.get("plan_id") == run_plan_id
            ),
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
        if state.build_state is not None
        and isinstance(state.build_state.get("updated_at"), str)
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
            state.build_state.get("scale_hash")
            if state.build_state is not None
            else None
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
        interventions=[
            plugin.descriptor.machine_payload()
            for plugin in intervention_plugins()
        ]
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
        champion_trainable=(
            state.champion.model.trainable if state.champion is not None else None
        ),
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
