"""Privacy-minimal evidence from real model use.

Usage observations are not reinforcement-learning rewards. They record what happened when
an exact model was used so Frontierwright can find workload gaps and propose falsifiable
next experiments without storing prompts/responses by default.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import cast

from frontierwright.data import DatasetClassification
from frontierwright.errors import FrontierwrightError

OBSERVATION_EVENT_KIND = "USAGE_OBSERVATION_RECORDED"
OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_SOURCE_EXPLICIT = "EXPLICIT_USER"


class ObservationOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CORRECTED = "CORRECTED"
    ABSTAINED = "ABSTAINED"


def _label(value: str | None, field_name: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise FrontierwrightError(
                "INVALID_USAGE_OBSERVATION",
                f"{field_name} must be nonempty.",
                2,
            )
        return None
    if not isinstance(value, str):
        raise FrontierwrightError(
            "INVALID_USAGE_OBSERVATION",
            f"{field_name} must be a string.",
            2,
        )
    cleaned = value.strip()
    if not cleaned:
        if required:
            raise FrontierwrightError(
                "INVALID_USAGE_OBSERVATION",
                f"{field_name} must be nonempty.",
                2,
            )
        return None
    if "\x00" in cleaned or len(cleaned) > 200:
        raise FrontierwrightError(
            "INVALID_USAGE_OBSERVATION",
            f"{field_name} must be at most 200 characters and NUL-free.",
            2,
        )
    return cleaned


def _latency(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrontierwrightError(
            "INVALID_USAGE_OBSERVATION",
            "latency_seconds must be numeric.",
            2,
        )
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise FrontierwrightError(
            "INVALID_USAGE_OBSERVATION",
            "latency_seconds must be finite and nonnegative.",
            2,
        )
    return result


@dataclass(frozen=True)
class UsageObservation:
    observation_id: str
    model_id: str
    model_fingerprint: str
    task: str
    outcome: ObservationOutcome
    workload_profile_hash: str | None = None
    domain: str | None = None
    language: str | None = None
    failure_category: str | None = None
    latency_seconds: float | None = None
    classification: DatasetClassification = DatasetClassification.PRIVATE
    source: str = OBSERVATION_SOURCE_EXPLICIT
    idempotency_key: str | None = None
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise FrontierwrightError(
                "INVALID_USAGE_OBSERVATION", "observation_id must be nonempty.", 2
            )
        if not self.model_id.strip() or not self.model_fingerprint.strip():
            raise FrontierwrightError(
                "INVALID_USAGE_OBSERVATION",
                "model identity/fingerprint must be nonempty.",
                2,
            )
        object.__setattr__(self, "task", _label(self.task, "task", required=True))
        object.__setattr__(self, "domain", _label(self.domain, "domain"))
        object.__setattr__(self, "language", _label(self.language, "language"))
        object.__setattr__(
            self,
            "failure_category",
            _label(self.failure_category, "failure_category"),
        )
        object.__setattr__(self, "latency_seconds", _latency(self.latency_seconds))
        object.__setattr__(
            self,
            "idempotency_key",
            _label(self.idempotency_key, "idempotency_key"),
        )
        if self.outcome is ObservationOutcome.SUCCESS and self.failure_category is not None:
            raise FrontierwrightError(
                "INVALID_USAGE_OBSERVATION",
                "A SUCCESS observation cannot declare a failure_category.",
                2,
            )
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise FrontierwrightError(
                "INVALID_USAGE_OBSERVATION",
                f"Unsupported observation schema_version {self.schema_version}.",
                2,
            )

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        payload["classification"] = self.classification.value
        payload["content_stored"] = False
        return payload


@dataclass(frozen=True)
class TaskObservationSummary:
    task: str
    total: int
    direct_successes: int
    failures: int
    corrected: int
    abstained: int
    direct_success_rate: float | None
    direct_success_rate_ci95: tuple[float, float] | None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationSummary:
    model_id: str | None
    total: int = 0
    direct_successes: int = 0
    failures: int = 0
    corrected: int = 0
    abstained: int = 0
    evaluable: int = 0
    direct_success_rate: float | None = None
    direct_success_rate_ci95: tuple[float, float] | None = None
    by_task: tuple[TaskObservationSummary, ...] = ()
    failure_categories: dict[str, int] = field(default_factory=dict)
    workload_profile_hashes: tuple[str, ...] = ()
    note: str = (
        "Usage observations are operational evidence, not RL rewards. Raw prompt/response "
        "content is not stored by this observation contract."
    )

    def to_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "by_task": [item.to_payload() for item in self.by_task],
        }


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def observation_from_event(details: dict[str, object]) -> UsageObservation | None:
    try:
        if details.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
            return None
        outcome = ObservationOutcome(str(details["outcome"]))
        classification = DatasetClassification(str(details["classification"]))
        return UsageObservation(
            observation_id=str(details["observation_id"]),
            model_id=str(details["model_id"]),
            model_fingerprint=str(details["model_fingerprint"]),
            task=str(details["task"]),
            outcome=outcome,
            workload_profile_hash=(
                str(details["workload_profile_hash"])
                if isinstance(details.get("workload_profile_hash"), str)
                else None
            ),
            domain=(str(details["domain"]) if isinstance(details.get("domain"), str) else None),
            language=(
                str(details["language"]) if isinstance(details.get("language"), str) else None
            ),
            failure_category=(
                str(details["failure_category"])
                if isinstance(details.get("failure_category"), str)
                else None
            ),
            latency_seconds=(
                float(cast(int | float, details["latency_seconds"]))
                if isinstance(details.get("latency_seconds"), (int, float))
                and not isinstance(details.get("latency_seconds"), bool)
                else None
            ),
            classification=classification,
            source=str(details.get("source") or OBSERVATION_SOURCE_EXPLICIT),
            idempotency_key=(
                str(details["idempotency_key"])
                if isinstance(details.get("idempotency_key"), str)
                else None
            ),
        )
    except (KeyError, ValueError, FrontierwrightError):
        return None


def summarize_observations(
    observations: list[UsageObservation],
    *,
    model_id: str | None,
) -> ObservationSummary:
    selected = [item for item in observations if model_id is None or item.model_id == model_id]
    counts = {outcome: 0 for outcome in ObservationOutcome}
    by_task_raw: dict[str, dict[ObservationOutcome, int]] = {}
    failure_categories: dict[str, int] = {}
    workload_hashes: set[str] = set()

    for item in selected:
        counts[item.outcome] += 1
        task_counts = by_task_raw.setdefault(
            item.task, {outcome: 0 for outcome in ObservationOutcome}
        )
        task_counts[item.outcome] += 1
        if item.failure_category is not None:
            failure_categories[item.failure_category] = (
                failure_categories.get(item.failure_category, 0) + 1
            )
        if item.workload_profile_hash is not None:
            workload_hashes.add(item.workload_profile_hash)

    direct_successes = counts[ObservationOutcome.SUCCESS]
    failures = counts[ObservationOutcome.FAILURE]
    corrected = counts[ObservationOutcome.CORRECTED]
    abstained = counts[ObservationOutcome.ABSTAINED]
    evaluable = direct_successes + failures + corrected
    rate = direct_successes / evaluable if evaluable else None
    interval = wilson_interval(direct_successes, evaluable) if evaluable else None

    by_task: list[TaskObservationSummary] = []
    for task, task_counts in by_task_raw.items():
        task_success = task_counts[ObservationOutcome.SUCCESS]
        task_failure = task_counts[ObservationOutcome.FAILURE]
        task_corrected = task_counts[ObservationOutcome.CORRECTED]
        task_abstained = task_counts[ObservationOutcome.ABSTAINED]
        task_evaluable = task_success + task_failure + task_corrected
        by_task.append(
            TaskObservationSummary(
                task=task,
                total=sum(task_counts.values()),
                direct_successes=task_success,
                failures=task_failure,
                corrected=task_corrected,
                abstained=task_abstained,
                direct_success_rate=(task_success / task_evaluable if task_evaluable else None),
                direct_success_rate_ci95=(
                    wilson_interval(task_success, task_evaluable) if task_evaluable else None
                ),
            )
        )
    by_task.sort(
        key=lambda item: (
            -(item.failures + item.corrected),
            -item.total,
            item.task.casefold(),
        )
    )
    ordered_categories = dict(
        sorted(failure_categories.items(), key=lambda item: (-item[1], item[0].casefold()))
    )
    return ObservationSummary(
        model_id=model_id,
        total=len(selected),
        direct_successes=direct_successes,
        failures=failures,
        corrected=corrected,
        abstained=abstained,
        evaluable=evaluable,
        direct_success_rate=rate,
        direct_success_rate_ci95=interval,
        by_task=tuple(by_task),
        failure_categories=ordered_categories,
        workload_profile_hashes=tuple(sorted(workload_hashes)),
    )
