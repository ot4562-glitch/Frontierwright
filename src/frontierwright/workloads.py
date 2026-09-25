"""Versioned user-workload contracts for Frontierwright user-fit optimization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from frontierwright.data import DatasetClassification
from frontierwright.errors import FrontierwrightError

WORKLOAD_PROFILE_SCHEMA_VERSION = 3
WORKLOAD_PROFILE_SOURCE_EXPLICIT = "EXPLICIT_USER"

SUPPORTED_UTILITY_METRICS: frozenset[str] = frozenset(
    {
        "capability.general",
        "capability.reasoning",
        "capability.math",
        "capability.coding",
        "serving.latency_p50",
        "serving.throughput_p50",
        "serving.context_p95",
        "serving.ttft_p50",
        "serving.tpot_p50",
        "serving.itl_p50",
        "serving.output_throughput_aggregate",
        "resource.peak_vram",
        "resource.process_rss",
        "storage.model_artifact",
    }
)


def _clean_labels(values: list[str] | tuple[str, ...], field_name: str) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                f"{field_name} entries must be strings.",
                2,
            )
        value = raw.strip()
        if not value:
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                f"{field_name} entries must be nonempty.",
                2,
            )
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(value)
    return tuple(cleaned)


def _positive_optional(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrontierwrightError(
            "INVALID_WORKLOAD_PROFILE",
            f"{field_name} must be numeric.",
            2,
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise FrontierwrightError(
            "INVALID_WORKLOAD_PROFILE",
            f"{field_name} must be finite and positive.",
            2,
        )
    return result


@dataclass(frozen=True)
class WorkloadProfile:
    """Explicit evidence describing what this user actually needs from a model."""

    name: str
    languages: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    task_weights: dict[str, float] = field(default_factory=dict)
    context_tokens_p50: int | None = None
    context_tokens_p95: int | None = None
    max_latency_seconds: float | None = None
    min_tokens_per_second: float | None = None
    privacy: DatasetClassification = DatasetClassification.PRIVATE
    critical_floors: dict[str, float] = field(default_factory=dict)
    utility_weights: dict[str, float] = field(default_factory=dict)
    utility_scales: dict[str, float] = field(default_factory=dict)
    improvement_margins: dict[str, float] = field(default_factory=dict)
    source: str = WORKLOAD_PROFILE_SOURCE_EXPLICIT
    schema_version: int = WORKLOAD_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                "Workload profile name must be nonempty.",
                2,
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "languages", _clean_labels(self.languages, "languages"))
        object.__setattr__(self, "domains", _clean_labels(self.domains, "domains"))

        if self.source != WORKLOAD_PROFILE_SOURCE_EXPLICIT:
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                "Only EXPLICIT_USER workload profiles are supported in this release.",
                2,
            )

        for field_name in ("context_tokens_p50", "context_tokens_p95"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"{field_name} must be a positive integer.",
                    2,
                )
        if (
            self.context_tokens_p50 is not None
            and self.context_tokens_p95 is not None
            and self.context_tokens_p95 < self.context_tokens_p50
        ):
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                "context_tokens_p95 cannot be smaller than context_tokens_p50.",
                2,
            )

        object.__setattr__(
            self,
            "max_latency_seconds",
            _positive_optional(self.max_latency_seconds, "max_latency_seconds"),
        )
        object.__setattr__(
            self,
            "min_tokens_per_second",
            _positive_optional(self.min_tokens_per_second, "min_tokens_per_second"),
        )

        task_weights: dict[str, float] = {}
        for raw_key, raw_value in self.task_weights.items():
            key = str(raw_key).strip()
            if not key:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    "task_weights keys must be nonempty.",
                    2,
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"task weight for {key!r} must be numeric.",
                    2,
                )
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"task weight for {key!r} must be finite and positive.",
                    2,
                )
            task_weights[key] = value
        object.__setattr__(self, "task_weights", dict(sorted(task_weights.items())))

        floors: dict[str, float] = {}
        allowed_axes = {"general", "reasoning", "math", "coding"}
        for raw_axis, raw_value in self.critical_floors.items():
            axis = str(raw_axis).strip().lower()
            if axis not in allowed_axes:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"unsupported critical floor axis: {axis!r}",
                    2,
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"critical floor for {axis} must be numeric.",
                    2,
                )
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"critical floor for {axis} must be finite and nonnegative.",
                    2,
                )
            floors[axis] = value
        object.__setattr__(self, "critical_floors", dict(sorted(floors.items())))

        utility_weights: dict[str, float] = {}
        utility_scales: dict[str, float] = {}
        for raw_key, raw_value in self.utility_weights.items():
            key = str(raw_key).strip()
            if key not in SUPPORTED_UTILITY_METRICS:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"unsupported utility metric: {key!r}",
                    2,
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"utility weight for {key!r} must be numeric.",
                    2,
                )
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"utility weight for {key!r} must be finite and positive.",
                    2,
                )
            utility_weights[key] = value
        for raw_key, raw_value in self.utility_scales.items():
            key = str(raw_key).strip()
            if key not in SUPPORTED_UTILITY_METRICS:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"unsupported utility metric: {key!r}",
                    2,
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"utility scale for {key!r} must be numeric.",
                    2,
                )
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"utility scale for {key!r} must be finite and positive.",
                    2,
                )
            utility_scales[key] = value
        if set(utility_weights) != set(utility_scales):
            missing_scales = sorted(set(utility_weights) - set(utility_scales))
            missing_weights = sorted(set(utility_scales) - set(utility_weights))
            details: list[str] = []
            if missing_scales:
                details.append("missing scales for " + ", ".join(missing_scales))
            if missing_weights:
                details.append("missing weights for " + ", ".join(missing_weights))
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                "utility_weights and utility_scales must name the same metrics: "
                + "; ".join(details),
                2,
            )
        object.__setattr__(self, "utility_weights", dict(sorted(utility_weights.items())))
        object.__setattr__(self, "utility_scales", dict(sorted(utility_scales.items())))

        margins: dict[str, float] = {}
        for raw_key, raw_value in self.improvement_margins.items():
            key = str(raw_key).strip()
            if key not in SUPPORTED_UTILITY_METRICS:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"unsupported improvement-margin metric: {key!r}",
                    2,
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"improvement margin for {key!r} must be numeric.",
                    2,
                )
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0:
                raise FrontierwrightError(
                    "INVALID_WORKLOAD_PROFILE",
                    f"improvement margin for {key!r} must be finite and positive.",
                    2,
                )
            margins[key] = value
        object.__setattr__(self, "improvement_margins", dict(sorted(margins.items())))

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["languages"] = list(self.languages)
        payload["domains"] = list(self.domains)
        payload["privacy"] = self.privacy.value
        return payload

    @property
    def profile_hash(self) -> str:
        canonical = json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(canonical).hexdigest()


class WorkloadConstraintStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNKNOWN = "UNKNOWN"


class MeasurementResolutionState(StrEnum):
    """User-facing resolution state for a conservative decision status."""

    RESOLVED = "RESOLVED"
    MEASUREMENT_NEEDED = "MEASUREMENT_NEEDED"
    MORE_EVIDENCE_NEEDED = "MORE_EVIDENCE_NEEDED"


def measurement_resolution_state(
    status: WorkloadConstraintStatus,
) -> MeasurementResolutionState:
    if status is WorkloadConstraintStatus.UNKNOWN:
        return MeasurementResolutionState.MEASUREMENT_NEEDED
    if status is WorkloadConstraintStatus.INCONCLUSIVE:
        return MeasurementResolutionState.MORE_EVIDENCE_NEEDED
    return MeasurementResolutionState.RESOLVED


def human_fit_status(status: WorkloadConstraintStatus | str) -> str:
    """Translate conservative decision states into actionable human-facing copy."""

    try:
        normalized = (
            status
            if isinstance(status, WorkloadConstraintStatus)
            else WorkloadConstraintStatus(str(status))
        )
    except ValueError:
        return str(status)
    if normalized is WorkloadConstraintStatus.UNKNOWN:
        return "MEASUREMENT NEEDED"
    if normalized is WorkloadConstraintStatus.INCONCLUSIVE:
        return "MORE EVIDENCE NEEDED"
    return normalized.value


@dataclass(frozen=True)
class WorkloadFitConstraint:
    key: str
    category: str
    requirement: object
    observed: object | None
    unit: str | None
    status: WorkloadConstraintStatus
    evidence_source: str | None
    reason: str

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["resolution_state"] = measurement_resolution_state(self.status).value
        return payload


@dataclass(frozen=True)
class WorkloadFitAssessment:
    overall_status: WorkloadConstraintStatus
    constraints: tuple[WorkloadFitConstraint, ...]

    def to_payload(self) -> dict[str, object]:
        counts = {status.value: 0 for status in WorkloadConstraintStatus}
        for item in self.constraints:
            counts[item.status.value] += 1
        resolution_counts = {state.value: 0 for state in MeasurementResolutionState}
        for item in self.constraints:
            resolution_counts[measurement_resolution_state(item.status).value] += 1
        return {
            "overall_status": self.overall_status.value,
            "overall_resolution_state": measurement_resolution_state(
                self.overall_status
            ).value,
            "counts": counts,
            "resolution_counts": resolution_counts,
            "constraints": [item.to_payload() for item in self.constraints],
            "synthetic_utility_score": None,
            "note": (
                "Frontierwright does not collapse user fit into one synthetic score. "
                "Each declared requirement remains independently auditable."
            ),
        }


def workload_profile_from_payload(payload: dict[str, object]) -> WorkloadProfile:
    raw_privacy = payload.get("privacy", DatasetClassification.PRIVATE.value)
    try:
        privacy = DatasetClassification(str(raw_privacy))
    except ValueError as exc:
        raise FrontierwrightError(
            "INVALID_WORKLOAD_PROFILE",
            f"unsupported workload privacy classification: {raw_privacy!r}",
            2,
        ) from exc

    def labels(key: str) -> tuple[str, ...]:
        raw = payload.get(key, [])
        if not isinstance(raw, list):
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                f"{key} must be a list.",
                2,
            )
        return tuple(str(item) for item in raw)

    def mapping(key: str) -> dict[str, float]:
        raw = payload.get(key, {})
        if not isinstance(raw, dict):
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                f"{key} must be an object.",
                2,
            )
        return {str(name): float(value) for name, value in raw.items()}

    def optional_int(key: str) -> int | None:
        raw = payload.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                f"{key} must be an integer.",
                2,
            )
        return int(raw)

    def optional_float(key: str) -> float | None:
        raw = payload.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise FrontierwrightError(
                "INVALID_WORKLOAD_PROFILE",
                f"{key} must be numeric.",
                2,
            )
        return float(raw)

    raw_schema_version = payload.get("schema_version", WORKLOAD_PROFILE_SCHEMA_VERSION)
    if isinstance(raw_schema_version, bool) or not isinstance(
        raw_schema_version, (int, float, str)
    ):
        raise FrontierwrightError(
            "INVALID_WORKLOAD_PROFILE",
            "schema_version must be an integer.",
            2,
        )

    return WorkloadProfile(
        name=str(payload.get("name", "")),
        languages=labels("languages"),
        domains=labels("domains"),
        task_weights=mapping("task_weights"),
        context_tokens_p50=optional_int("context_tokens_p50"),
        context_tokens_p95=optional_int("context_tokens_p95"),
        max_latency_seconds=optional_float("max_latency_seconds"),
        min_tokens_per_second=optional_float("min_tokens_per_second"),
        privacy=privacy,
        critical_floors=mapping("critical_floors"),
        utility_weights=mapping("utility_weights"),
        utility_scales=mapping("utility_scales"),
        improvement_margins=mapping("improvement_margins"),
        source=str(payload.get("source", WORKLOAD_PROFILE_SOURCE_EXPLICIT)),
        schema_version=int(raw_schema_version),
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def assess_workload_fit(
    profile: WorkloadProfile,
    *,
    capability_stats: dict[str, float | None],
    inference_metrics: dict[str, object] | None = None,
    supported_context_tokens: int | None = None,
    workload_eval_coverage: bool | None = None,
    workload_acceptance_status: str | None = None,
    serving_boundary: str | None = None,
) -> WorkloadFitAssessment:
    """Compare declared requirements with measured evidence without inventing utility.

    A missing measurement is UNKNOWN, never an inferred pass. This deliberately keeps
    public benchmark capability separate from user-specific task utility.
    """

    constraints: list[WorkloadFitConstraint] = []
    metrics = inference_metrics or {}

    for axis, floor in profile.critical_floors.items():
        observed = _finite_number(capability_stats.get(axis))
        if observed is None:
            status = WorkloadConstraintStatus.UNKNOWN
            reason = "No comparable measured capability value is available for this axis."
        elif observed >= floor:
            status = WorkloadConstraintStatus.PASS
            reason = "Measured capability satisfies the declared hard floor."
        else:
            status = WorkloadConstraintStatus.FAIL
            reason = "Measured capability is below the declared hard floor."
        constraints.append(
            WorkloadFitConstraint(
                key=f"capability.{axis}",
                category="CAPABILITY",
                requirement=floor,
                observed=observed,
                unit="frontierwright-capability-v1",
                status=status,
                evidence_source="CAPABILITY_PROFILE" if observed is not None else None,
                reason=reason,
            )
        )

    if profile.max_latency_seconds is not None:
        observed = _finite_number(metrics.get("latency_seconds_p50"))
        if observed is None:
            status = WorkloadConstraintStatus.UNKNOWN
            reason = "No measured inference latency receipt exists for this model."
        elif observed <= profile.max_latency_seconds:
            status = WorkloadConstraintStatus.PASS
            reason = "Measured p50 latency is within the declared ceiling."
        else:
            status = WorkloadConstraintStatus.FAIL
            reason = "Measured p50 latency exceeds the declared ceiling."
        constraints.append(
            WorkloadFitConstraint(
                key="serving.latency_p50",
                category="SERVING",
                requirement=profile.max_latency_seconds,
                observed=observed,
                unit="seconds",
                status=status,
                evidence_source="INFERENCE_PROFILE" if observed is not None else None,
                reason=reason,
            )
        )

    if profile.min_tokens_per_second is not None:
        observed = _finite_number(metrics.get("tokens_per_second_p50"))
        if observed is None:
            status = WorkloadConstraintStatus.UNKNOWN
            reason = "No measured inference throughput receipt exists for this model."
        elif observed >= profile.min_tokens_per_second:
            status = WorkloadConstraintStatus.PASS
            reason = "Measured p50 throughput satisfies the declared floor."
        else:
            status = WorkloadConstraintStatus.FAIL
            reason = "Measured p50 throughput is below the declared floor."
        constraints.append(
            WorkloadFitConstraint(
                key="serving.throughput_p50",
                category="SERVING",
                requirement=profile.min_tokens_per_second,
                observed=observed,
                unit="tokens/second",
                status=status,
                evidence_source="INFERENCE_PROFILE" if observed is not None else None,
                reason=reason,
            )
        )

    if profile.context_tokens_p95 is not None:
        observed_context = (
            supported_context_tokens
            if isinstance(supported_context_tokens, int)
            and not isinstance(supported_context_tokens, bool)
            and supported_context_tokens > 0
            else None
        )
        if observed_context is None:
            status = WorkloadConstraintStatus.UNKNOWN
            reason = "Model context capacity has not been verified by a compatible receipt."
        elif observed_context >= profile.context_tokens_p95:
            status = WorkloadConstraintStatus.PASS
            reason = "Verified model context capacity covers workload p95 context demand."
        else:
            status = WorkloadConstraintStatus.FAIL
            reason = "Verified model context capacity is below workload p95 demand."
        constraints.append(
            WorkloadFitConstraint(
                key="serving.context_p95",
                category="SERVING",
                requirement=profile.context_tokens_p95,
                observed=observed_context,
                unit="tokens",
                status=status,
                evidence_source="MODEL_CONTEXT_RECEIPT" if observed_context else None,
                reason=reason,
            )
        )

    needs_workload_eval = bool(profile.languages or profile.domains or profile.task_weights)
    if needs_workload_eval:
        coverage_status = (
            WorkloadConstraintStatus.PASS
            if workload_eval_coverage is True
            else WorkloadConstraintStatus.UNKNOWN
        )
        coverage_reason = (
            "A workload-specific evaluation receipt covers the declared task surface."
            if workload_eval_coverage is True
            else (
                "Public/core capability evidence is not enough to prove performance on the "
                "declared languages, domains, and task mixture."
            )
        )
        constraints.append(
            WorkloadFitConstraint(
                key="evaluation.workload_coverage",
                category="EVALUATION",
                requirement={
                    "languages": list(profile.languages),
                    "domains": list(profile.domains),
                    "task_weights": dict(profile.task_weights),
                },
                observed=(True if workload_eval_coverage is True else None),
                unit=None,
                status=coverage_status,
                evidence_source=(
                    "WORKLOAD_EVALUATION_RECEIPT" if workload_eval_coverage is True else None
                ),
                reason=coverage_reason,
            )
        )

        normalized_acceptance = (workload_acceptance_status or "").upper()
        if normalized_acceptance == "PASS":
            acceptance_status = WorkloadConstraintStatus.PASS
            acceptance_reason = (
                "Every configured workload acceptance criterion passed under its declared "
                "evidence rule."
            )
            acceptance_observed: object | None = "PASS"
        elif normalized_acceptance == "FAIL":
            acceptance_status = WorkloadConstraintStatus.FAIL
            acceptance_reason = "At least one workload acceptance criterion failed."
            acceptance_observed = "FAIL"
        elif normalized_acceptance == "INCONCLUSIVE":
            acceptance_status = WorkloadConstraintStatus.INCONCLUSIVE
            acceptance_reason = (
                "Workload acceptance evidence overlaps at least one configured threshold."
            )
            acceptance_observed = "INCONCLUSIVE"
        else:
            acceptance_status = WorkloadConstraintStatus.UNKNOWN
            acceptance_reason = (
                "Success criteria are not configured or have not been assessed with exact "
                "compatible evidence."
            )
            acceptance_observed = None
        constraints.append(
            WorkloadFitConstraint(
                key="evaluation.workload_acceptance",
                category="EVALUATION",
                requirement="versioned explicit workload success criteria",
                observed=acceptance_observed,
                unit=None,
                status=acceptance_status,
                evidence_source=(
                    "WORKLOAD_ACCEPTANCE_ASSESSMENT" if acceptance_observed is not None else None
                ),
                reason=acceptance_reason,
            )
        )

    if profile.privacy is not DatasetClassification.PUBLIC:
        normalized_boundary = serving_boundary.upper() if serving_boundary else None
        allowed = {"LOCAL_MACHINE", "CONTROLLED_PRIVATE"}
        if normalized_boundary is None:
            status = WorkloadConstraintStatus.UNKNOWN
            reason = "No serving-boundary receipt proves where inference executes."
        elif normalized_boundary in allowed:
            status = WorkloadConstraintStatus.PASS
            reason = "Measured/declared serving boundary satisfies the non-public workload."
        else:
            status = WorkloadConstraintStatus.FAIL
            reason = "Serving boundary conflicts with the declared non-public workload."
        constraints.append(
            WorkloadFitConstraint(
                key="privacy.serving_boundary",
                category="PRIVACY",
                requirement=profile.privacy.value,
                observed=normalized_boundary,
                unit=None,
                status=status,
                evidence_source="SERVING_BOUNDARY_RECEIPT" if normalized_boundary else None,
                reason=reason,
            )
        )

    # Coverage answers "did we measure the declared workload surface?" and acceptance
    # answers "did the exact evidence satisfy the user's explicit success criteria?".
    # A user-specific PASS requires both whenever the profile declares workload labels.
    decision_constraints = constraints
    statuses = {item.status for item in decision_constraints}
    if WorkloadConstraintStatus.FAIL in statuses:
        overall = WorkloadConstraintStatus.FAIL
    elif WorkloadConstraintStatus.UNKNOWN in statuses:
        overall = WorkloadConstraintStatus.UNKNOWN
    elif WorkloadConstraintStatus.INCONCLUSIVE in statuses:
        overall = WorkloadConstraintStatus.INCONCLUSIVE
    elif decision_constraints and statuses == {WorkloadConstraintStatus.PASS}:
        overall = WorkloadConstraintStatus.PASS
    else:
        overall = WorkloadConstraintStatus.UNKNOWN
    return WorkloadFitAssessment(overall, tuple(constraints))


class ParetoDirection(StrEnum):
    HIGHER_BETTER = "HIGHER_BETTER"
    LOWER_BETTER = "LOWER_BETTER"


class ParetoMetricRelation(StrEnum):
    BETTER = "BETTER"
    WORSE = "WORSE"
    SAME = "SAME"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


class ParetoRelation(StrEnum):
    CANDIDATE_DOMINATES = "CANDIDATE_DOMINATES"
    CHAMPION_DOMINATES = "CHAMPION_DOMINATES"
    TRADEOFF = "TRADEOFF"
    EQUIVALENT = "EQUIVALENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ParetoMetricInput:
    """One comparable measured dimension for Champion/Candidate evidence."""

    key: str
    category: str
    direction: ParetoDirection
    champion_value: float | int | None
    candidate_value: float | int | None
    unit: str | None = None
    evidence_source: str | None = None
    improvement_interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class ParetoMetricEvidence:
    key: str
    category: str
    direction: ParetoDirection
    champion_value: float | None
    candidate_value: float | None
    raw_delta: float | None
    improvement_delta: float | None
    relation: ParetoMetricRelation
    unit: str | None
    evidence_source: str | None
    improvement_interval: tuple[float, float] | None

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["direction"] = self.direction.value
        payload["relation"] = self.relation.value
        payload["improvement_interval"] = (
            list(self.improvement_interval) if self.improvement_interval is not None else None
        )
        return payload


@dataclass(frozen=True)
class ParetoComparison:
    relation: ParetoRelation
    metrics: tuple[ParetoMetricEvidence, ...]

    def to_payload(self) -> dict[str, object]:
        counts = {status.value: 0 for status in ParetoMetricRelation}
        for metric in self.metrics:
            counts[metric.relation.value] += 1
        unknown_count = counts[ParetoMetricRelation.UNKNOWN.value]
        uncertain_count = counts[ParetoMetricRelation.UNCERTAIN.value]
        complete = unknown_count == 0 and uncertain_count == 0 and bool(self.metrics)
        return {
            "relation": self.relation.value,
            "relation_scope": ("ALL_LISTED_DIMENSIONS" if complete else "MEASURED_DIMENSIONS_ONLY"),
            "complete": complete,
            "known_metric_count": len(self.metrics) - unknown_count,
            "unknown_metric_count": unknown_count,
            "uncertain_metric_count": uncertain_count,
            "unqualified_dominance_claim_eligible": bool(
                complete and self.relation is ParetoRelation.CANDIDATE_DOMINATES
            ),
            "counts": counts,
            "metrics": [metric.to_payload() for metric in self.metrics],
            "synthetic_utility_score": None,
            "note": (
                "Pareto relation uses comparable measured evidence and explicit uncertainty "
                "intervals when available. UNKNOWN or UNCERTAIN dimensions block an "
                "unqualified dominance claim."
            ),
        }


def compare_pareto_metrics(
    inputs: tuple[ParetoMetricInput, ...] | list[ParetoMetricInput],
) -> ParetoComparison:
    """Compare measured dimensions without inventing cross-unit utility weights."""

    metrics: list[ParetoMetricEvidence] = []
    for item in inputs:
        champion = _finite_number(item.champion_value)
        candidate = _finite_number(item.candidate_value)
        if champion is None or candidate is None:
            metrics.append(
                ParetoMetricEvidence(
                    key=item.key,
                    category=item.category,
                    direction=item.direction,
                    champion_value=champion,
                    candidate_value=candidate,
                    raw_delta=None,
                    improvement_delta=None,
                    relation=ParetoMetricRelation.UNKNOWN,
                    unit=item.unit,
                    evidence_source=item.evidence_source,
                    improvement_interval=None,
                )
            )
            continue

        raw_delta = candidate - champion
        improvement_delta = (
            raw_delta if item.direction is ParetoDirection.HIGHER_BETTER else -raw_delta
        )
        interval = item.improvement_interval
        if interval is not None:
            lower = _finite_number(interval[0])
            upper = _finite_number(interval[1])
            if lower is None or upper is None or lower > upper:
                raise ValueError("improvement_interval must be finite and ordered")
            interval = (lower, upper)
            if math.isclose(lower, 0.0, abs_tol=1e-12) and math.isclose(
                upper, 0.0, abs_tol=1e-12
            ):
                relation = ParetoMetricRelation.SAME
            elif lower <= 0 <= upper:
                relation = ParetoMetricRelation.UNCERTAIN
            elif lower > 0:
                relation = ParetoMetricRelation.BETTER
            else:
                relation = ParetoMetricRelation.WORSE
        elif math.isclose(candidate, champion, rel_tol=1e-9, abs_tol=1e-12):
            relation = ParetoMetricRelation.SAME
        else:
            relation = (
                ParetoMetricRelation.BETTER
                if improvement_delta > 0
                else ParetoMetricRelation.WORSE
            )
        metrics.append(
            ParetoMetricEvidence(
                key=item.key,
                category=item.category,
                direction=item.direction,
                champion_value=champion,
                candidate_value=candidate,
                raw_delta=raw_delta,
                improvement_delta=improvement_delta,
                relation=relation,
                unit=item.unit,
                evidence_source=item.evidence_source,
                improvement_interval=interval,
            )
        )

    decisive = [
        metric
        for metric in metrics
        if metric.relation
        not in {ParetoMetricRelation.UNKNOWN, ParetoMetricRelation.UNCERTAIN}
    ]
    uncertain = any(
        metric.relation is ParetoMetricRelation.UNCERTAIN for metric in metrics
    )
    better = any(metric.relation is ParetoMetricRelation.BETTER for metric in decisive)
    worse = any(metric.relation is ParetoMetricRelation.WORSE for metric in decisive)
    if better and worse:
        final_relation = ParetoRelation.TRADEOFF
    elif uncertain:
        final_relation = ParetoRelation.INSUFFICIENT_EVIDENCE
    elif not decisive:
        final_relation = ParetoRelation.INSUFFICIENT_EVIDENCE
    elif better:
        final_relation = ParetoRelation.CANDIDATE_DOMINATES
    elif worse:
        final_relation = ParetoRelation.CHAMPION_DOMINATES
    else:
        final_relation = ParetoRelation.EQUIVALENT
    return ParetoComparison(relation=final_relation, metrics=tuple(metrics))


class ExplicitUtilityStatus(StrEnum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class ExplicitUtilityRelation(StrEnum):
    CANDIDATE_PREFERRED = "CANDIDATE_PREFERRED"
    CHAMPION_PREFERRED = "CHAMPION_PREFERRED"
    TIE = "TIE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class UtilityContribution:
    metric_key: str
    weight: float
    scale: float
    improvement_delta: float
    normalized_improvement: float
    contribution: float
    evidence_source: str | None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExplicitUtilityComparison:
    status: ExplicitUtilityStatus
    relation: ExplicitUtilityRelation
    utility_delta: float | None
    contributions: tuple[UtilityContribution, ...]
    missing_metrics: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "relation": self.relation.value,
            "utility_delta": self.utility_delta,
            "contributions": [item.to_payload() for item in self.contributions],
            "missing_metrics": list(self.missing_metrics),
            "note": (
                "This is an explicit user-defined relative utility delta, not a model "
                "capability score. Every weighted metric uses a user-supplied normalization "
                "scale and remains auditable beside the raw Pareto evidence."
            ),
        }


def compare_explicit_user_utility(
    profile: WorkloadProfile,
    pareto: ParetoComparison,
) -> ExplicitUtilityComparison:
    """Apply only explicit user weights/scales to directly comparable evidence.

    Frontierwright never invents cross-unit normalization. If the user has weighted a
    metric but comparable Champion/Candidate evidence is missing, utility remains
    INCOMPLETE rather than silently renormalizing the remaining dimensions.
    """

    if not profile.utility_weights:
        return ExplicitUtilityComparison(
            status=ExplicitUtilityStatus.NOT_CONFIGURED,
            relation=ExplicitUtilityRelation.UNKNOWN,
            utility_delta=None,
            contributions=(),
        )

    by_key = {item.key: item for item in pareto.metrics}
    missing: list[str] = []
    contributions: list[UtilityContribution] = []
    for key, weight in profile.utility_weights.items():
        item = by_key.get(key)
        if item is None or item.improvement_delta is None:
            missing.append(key)
            continue
        scale = profile.utility_scales[key]
        normalized = item.improvement_delta / scale
        contribution = weight * normalized
        contributions.append(
            UtilityContribution(
                metric_key=key,
                weight=weight,
                scale=scale,
                improvement_delta=item.improvement_delta,
                normalized_improvement=normalized,
                contribution=contribution,
                evidence_source=item.evidence_source,
            )
        )

    if missing:
        return ExplicitUtilityComparison(
            status=ExplicitUtilityStatus.INCOMPLETE,
            relation=ExplicitUtilityRelation.UNKNOWN,
            utility_delta=None,
            contributions=tuple(contributions),
            missing_metrics=tuple(sorted(missing)),
        )

    utility_delta = sum(item.contribution for item in contributions)
    if math.isclose(utility_delta, 0.0, rel_tol=1e-9, abs_tol=1e-12):
        relation = ExplicitUtilityRelation.TIE
        utility_delta = 0.0
    elif utility_delta > 0:
        relation = ExplicitUtilityRelation.CANDIDATE_PREFERRED
    else:
        relation = ExplicitUtilityRelation.CHAMPION_PREFERRED
    return ExplicitUtilityComparison(
        status=ExplicitUtilityStatus.COMPLETE,
        relation=relation,
        utility_delta=utility_delta,
        contributions=tuple(contributions),
    )


class WorkloadDecisionClaimStatus(StrEnum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    INELIGIBLE = "INELIGIBLE"
    INCOMPLETE = "INCOMPLETE"
    ELIGIBLE = "ELIGIBLE"


class WorkloadDecisionClaimKind(StrEnum):
    NONE = "NONE"
    PARETO_IMPROVEMENT = "PARETO_IMPROVEMENT"
    EXPLICIT_UTILITY_PREFERENCE = "EXPLICIT_UTILITY_PREFERENCE"


@dataclass(frozen=True)
class WorkloadDecisionClaim:
    status: WorkloadDecisionClaimStatus
    kind: WorkloadDecisionClaimKind
    decision_eligible: bool
    material_metrics: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()
    regressions: tuple[str, ...] = ()
    within_margin: tuple[str, ...] = ()
    missing_metrics: tuple[str, ...] = ()
    missing_margins: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "kind": self.kind.value,
            "decision_eligible": self.decision_eligible,
            "material_metrics": list(self.material_metrics),
            "improvements": list(self.improvements),
            "regressions": list(self.regressions),
            "within_margin": list(self.within_margin),
            "missing_metrics": list(self.missing_metrics),
            "missing_margins": list(self.missing_margins),
            "reasons": list(self.reasons),
            "better_for_workload_statement_allowed": False,
            "claim_scope": "LOCAL_DECISION_EVIDENCE",
            "note": (
                "Decision eligibility is not a confirmatory scientific claim. Frontierwright "
                "does not currently certify an unqualified 'better for this workload' statement "
                "after adaptive candidate selection without independent confirmation evidence."
            ),
        }


def workload_material_metric_keys(profile: WorkloadProfile) -> tuple[str, ...]:
    """Return numeric dimensions the user explicitly made material to a decision."""

    keys = set(profile.utility_weights)
    keys.update(f"capability.{axis}" for axis in profile.critical_floors)
    if profile.max_latency_seconds is not None:
        keys.add("serving.latency_p50")
    if profile.min_tokens_per_second is not None:
        keys.add("serving.throughput_p50")
    if profile.context_tokens_p95 is not None:
        keys.add("serving.context_p95")
    return tuple(sorted(keys))


def assess_workload_decision_claim(
    profile: WorkloadProfile,
    pareto: ParetoComparison,
    utility: ExplicitUtilityComparison,
    *,
    mandatory_fit_eligible: bool,
) -> WorkloadDecisionClaim:
    """Qualify a local user-fit decision without turning partial evidence into superiority."""

    material = workload_material_metric_keys(profile)
    reasons: list[str] = []
    if not mandatory_fit_eligible:
        reasons.append("MANDATORY_WORKLOAD_GATE_NOT_PASS")
    if profile.languages or profile.domains or profile.task_weights:
        reasons.append("WORKLOAD_RESULT_THRESHOLD_UNDECLARED")
    if not material:
        reasons.append("NO_MATERIAL_COMPARISON_METRICS")

    missing_margins = tuple(key for key in material if key not in profile.improvement_margins)
    if missing_margins:
        reasons.append("PRACTICAL_IMPROVEMENT_MARGIN_MISSING")

    by_key = {item.key: item for item in pareto.metrics}
    missing_metrics = tuple(
        key for key in material if key not in by_key or by_key[key].improvement_delta is None
    )
    if missing_metrics:
        return WorkloadDecisionClaim(
            status=WorkloadDecisionClaimStatus.INCOMPLETE,
            kind=WorkloadDecisionClaimKind.NONE,
            decision_eligible=False,
            material_metrics=material,
            missing_metrics=missing_metrics,
            missing_margins=missing_margins,
            reasons=tuple(dict.fromkeys(reasons + ["MATERIAL_EVIDENCE_MISSING"])),
        )

    if reasons:
        return WorkloadDecisionClaim(
            status=WorkloadDecisionClaimStatus.INELIGIBLE,
            kind=WorkloadDecisionClaimKind.NONE,
            decision_eligible=False,
            material_metrics=material,
            missing_margins=missing_margins,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    improvements: list[str] = []
    regressions: list[str] = []
    within_margin: list[str] = []
    for key in material:
        delta = by_key[key].improvement_delta
        assert delta is not None
        margin = profile.improvement_margins[key]
        if delta >= margin:
            improvements.append(key)
        elif delta <= -margin:
            regressions.append(key)
        else:
            within_margin.append(key)

    if not improvements:
        return WorkloadDecisionClaim(
            status=WorkloadDecisionClaimStatus.INELIGIBLE,
            kind=WorkloadDecisionClaimKind.NONE,
            decision_eligible=False,
            material_metrics=material,
            regressions=tuple(regressions),
            within_margin=tuple(within_margin),
            reasons=("NO_PRACTICALLY_MEANINGFUL_IMPROVEMENT",),
        )

    if regressions:
        utility_covers_material = set(material).issubset(profile.utility_weights)
        if not (
            utility.status is ExplicitUtilityStatus.COMPLETE
            and utility.relation is ExplicitUtilityRelation.CANDIDATE_PREFERRED
            and utility_covers_material
        ):
            return WorkloadDecisionClaim(
                status=WorkloadDecisionClaimStatus.INELIGIBLE,
                kind=WorkloadDecisionClaimKind.NONE,
                decision_eligible=False,
                material_metrics=material,
                improvements=tuple(improvements),
                regressions=tuple(regressions),
                within_margin=tuple(within_margin),
                reasons=("REGRESSION_WITHOUT_COMPLETE_EXPLICIT_TRADEOFF_RULE",),
            )
        kind = WorkloadDecisionClaimKind.EXPLICIT_UTILITY_PREFERENCE
    else:
        kind = WorkloadDecisionClaimKind.PARETO_IMPROVEMENT

    return WorkloadDecisionClaim(
        status=WorkloadDecisionClaimStatus.ELIGIBLE,
        kind=kind,
        decision_eligible=True,
        material_metrics=material,
        improvements=tuple(improvements),
        regressions=tuple(regressions),
        within_margin=tuple(within_margin),
    )
