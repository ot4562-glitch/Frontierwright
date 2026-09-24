"""Exact bindings from user-workload requirements to measured evaluation evidence.

Frontierwright never infers workload coverage from evaluator task names. A binding must
name an exact stored evaluation receipt and exact task/version/metric measurements that
support each claimed language, domain, or task requirement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from frontierwright.errors import FrontierwrightError
from frontierwright.workloads import WorkloadProfile

WORKLOAD_EVAL_BINDING_SCHEMA_VERSION = 1
WORKLOAD_EVAL_BINDING_KIND = "WORKLOAD_EVALUATION_BOUND"
COVERAGE_KINDS = ("languages", "domains", "tasks")


@dataclass(frozen=True, order=True)
class WorkloadEvidenceSelector:
    task_id: str
    task_version: str
    metric: str

    def __post_init__(self) -> None:
        for name in ("task_id", "task_version", "metric"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise FrontierwrightError(
                    "WORKLOAD_EVALUATION_BINDING_INVALID",
                    f"{name} must be nonempty.",
                    2,
                )
            object.__setattr__(self, name, value)

    def to_payload(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadEvaluationManifest:
    workload_profile_hash: str
    receipt_id: str
    coverage: dict[str, dict[str, tuple[WorkloadEvidenceSelector, ...]]]
    schema_version: int = WORKLOAD_EVAL_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_EVAL_BINDING_SCHEMA_VERSION:
            raise FrontierwrightError(
                "WORKLOAD_EVALUATION_BINDING_INVALID",
                f"Unsupported workload evaluation binding schema: {self.schema_version}",
                2,
            )
        profile_hash = self.workload_profile_hash.strip()
        receipt_id = self.receipt_id.strip()
        if not profile_hash.startswith("sha256:") or len(profile_hash) != 71:
            raise FrontierwrightError(
                "WORKLOAD_EVALUATION_BINDING_INVALID",
                "workload_profile_hash must be a sha256: digest.",
                2,
            )
        if not receipt_id:
            raise FrontierwrightError(
                "WORKLOAD_EVALUATION_BINDING_INVALID",
                "receipt_id must be nonempty.",
                2,
            )
        object.__setattr__(self, "workload_profile_hash", profile_hash)
        object.__setattr__(self, "receipt_id", receipt_id)

        normalized: dict[str, dict[str, tuple[WorkloadEvidenceSelector, ...]]] = {}
        for kind in COVERAGE_KINDS:
            raw_claims = self.coverage.get(kind, {})
            claims: dict[str, tuple[WorkloadEvidenceSelector, ...]] = {}
            for raw_label, selectors in raw_claims.items():
                label = str(raw_label).strip()
                if not label:
                    raise FrontierwrightError(
                        "WORKLOAD_EVALUATION_BINDING_INVALID",
                        f"{kind} coverage labels must be nonempty.",
                        2,
                    )
                if not selectors:
                    raise FrontierwrightError(
                        "WORKLOAD_EVALUATION_BINDING_INVALID",
                        f"{kind}.{label} must reference at least one exact measurement.",
                        2,
                    )
                claims[label] = tuple(sorted(set(selectors)))
            normalized[kind] = dict(sorted(claims.items()))
        unknown = sorted(set(self.coverage) - set(COVERAGE_KINDS))
        if unknown:
            raise FrontierwrightError(
                "WORKLOAD_EVALUATION_BINDING_INVALID",
                "Unsupported coverage kinds: " + ", ".join(unknown),
                2,
            )
        if not any(normalized[kind] for kind in COVERAGE_KINDS):
            raise FrontierwrightError(
                "WORKLOAD_EVALUATION_BINDING_INVALID",
                "At least one workload requirement must be bound to evaluation evidence.",
                2,
            )
        object.__setattr__(self, "coverage", normalized)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "workload_profile_hash": self.workload_profile_hash,
            "receipt_id": self.receipt_id,
            "coverage": {
                kind: {
                    label: [selector.to_payload() for selector in selectors]
                    for label, selectors in claims.items()
                }
                for kind, claims in self.coverage.items()
            },
        }

    @property
    def manifest_hash(self) -> str:
        encoded = json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WorkloadEvaluationBinding:
    binding_id: str
    manifest_hash: str
    workload_profile_hash: str
    model_id: str
    model_fingerprint: str
    receipt_id: str
    receipt_sha256: str
    evaluator_id: str
    evaluator_version: str
    coverage: dict[str, dict[str, tuple[WorkloadEvidenceSelector, ...]]]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "binding_id": self.binding_id,
            "manifest_hash": self.manifest_hash,
            "workload_profile_hash": self.workload_profile_hash,
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "receipt_id": self.receipt_id,
            "receipt_sha256": self.receipt_sha256,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "coverage": {
                kind: {
                    label: [selector.to_payload() for selector in selectors]
                    for label, selectors in claims.items()
                }
                for kind, claims in self.coverage.items()
            },
        }


@dataclass(frozen=True)
class WorkloadEvaluationCoverage:
    complete: bool
    profile_hash: str
    model_id: str
    binding_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    covered: dict[str, tuple[str, ...]]
    missing: dict[str, tuple[str, ...]]

    def to_payload(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "profile_hash": self.profile_hash,
            "model_id": self.model_id,
            "binding_ids": list(self.binding_ids),
            "receipt_ids": list(self.receipt_ids),
            "covered": {key: list(values) for key, values in self.covered.items()},
            "missing": {key: list(values) for key, values in self.missing.items()},
        }


def _selector_from_payload(value: object, label: str) -> WorkloadEvidenceSelector:
    if not isinstance(value, dict):
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_BINDING_INVALID",
            f"{label} evidence selectors must be objects.",
            2,
        )
    return WorkloadEvidenceSelector(
        task_id=str(value.get("task_id", "")),
        task_version=str(value.get("task_version", "")),
        metric=str(value.get("metric", "")),
    )


def load_workload_evaluation_manifest(path: Path) -> WorkloadEvaluationManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_BINDING_INVALID",
            "Workload evaluation binding manifest must be readable UTF-8 JSON.",
            2,
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_BINDING_INVALID",
            "Workload evaluation binding manifest root must be an object.",
            2,
        )
    coverage_raw = payload.get("coverage", {})
    if not isinstance(coverage_raw, dict):
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_BINDING_INVALID",
            "coverage must be an object.",
            2,
        )
    coverage: dict[str, dict[str, tuple[WorkloadEvidenceSelector, ...]]] = {}
    for raw_kind, raw_claims in coverage_raw.items():
        kind = str(raw_kind)
        if not isinstance(raw_claims, dict):
            raise FrontierwrightError(
                "WORKLOAD_EVALUATION_BINDING_INVALID",
                f"coverage.{kind} must be an object.",
                2,
            )
        claims: dict[str, tuple[WorkloadEvidenceSelector, ...]] = {}
        for raw_label, raw_selectors in raw_claims.items():
            label = str(raw_label)
            if not isinstance(raw_selectors, list):
                raise FrontierwrightError(
                    "WORKLOAD_EVALUATION_BINDING_INVALID",
                    f"coverage.{kind}.{label} must be a list.",
                    2,
                )
            claims[label] = tuple(
                _selector_from_payload(
                    selector,
                    f"coverage.{kind}.{label}",
                )
                for selector in raw_selectors
            )
        coverage[kind] = claims

    raw_schema = payload.get("schema_version", WORKLOAD_EVAL_BINDING_SCHEMA_VERSION)
    if isinstance(raw_schema, bool) or not isinstance(raw_schema, int):
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_BINDING_INVALID",
            "schema_version must be an integer.",
            2,
        )
    return WorkloadEvaluationManifest(
        schema_version=raw_schema,
        workload_profile_hash=str(payload.get("workload_profile_hash", "")),
        receipt_id=str(payload.get("receipt_id", "")),
        coverage=coverage,
    )


def _measurement_identity(item: object) -> tuple[str, str, str] | None:
    if not isinstance(item, dict):
        return None
    task_id = item.get("task_id")
    task_version = item.get("task_version")
    metric = item.get("metric")
    if not isinstance(task_id, str) or not task_id:
        return None
    if not isinstance(task_version, str) or not task_version:
        return None
    if not isinstance(metric, str) or not metric:
        return None
    return (task_id, task_version, metric)


def bind_manifest_to_receipt(
    manifest: WorkloadEvaluationManifest,
    *,
    active_profile_hash: str,
    model_id: str,
    model_fingerprint: str,
    receipt: dict[str, Any],
) -> WorkloadEvaluationBinding:
    """Validate every claimed workload dimension against exact stored measurements."""

    if manifest.workload_profile_hash != active_profile_hash:
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_PROFILE_MISMATCH",
            "Binding manifest targets a different Workload Profile hash.",
            12,
        )
    if receipt.get("receipt_id") != manifest.receipt_id:
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_RECEIPT_MISMATCH",
            "Binding manifest receipt_id does not match the selected receipt.",
            12,
        )
    if receipt.get("model_id") != model_id or receipt.get("model_fingerprint") != model_fingerprint:
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_MODEL_MISMATCH",
            "Evaluation receipt does not belong to the exact model being bound.",
            12,
        )

    available = {
        identity
        for item in receipt.get("measurements", [])
        if (identity := _measurement_identity(item)) is not None
    }
    for kind, claims in manifest.coverage.items():
        for label, selectors in claims.items():
            for selector in selectors:
                identity = (selector.task_id, selector.task_version, selector.metric)
                if identity not in available:
                    raise FrontierwrightError(
                        "WORKLOAD_EVALUATION_MEASUREMENT_MISSING",
                        (
                            f"{kind}.{label} references absent measurement "
                            f"{selector.task_id}/{selector.task_version}/{selector.metric}."
                        ),
                        12,
                    )

    receipt_sha256_raw = receipt.get("receipt_sha256")
    evaluator_id_raw = receipt.get("evaluator_id")
    evaluator_version_raw = receipt.get("evaluator_version")
    if (
        not isinstance(receipt_sha256_raw, str)
        or not receipt_sha256_raw
        or not isinstance(evaluator_id_raw, str)
        or not evaluator_id_raw
        or not isinstance(evaluator_version_raw, str)
        or not evaluator_version_raw
    ):
        raise FrontierwrightError(
            "WORKLOAD_EVALUATION_RECEIPT_INVALID",
            "Stored evaluation receipt identity is incomplete.",
            4,
        )

    receipt_sha256 = receipt_sha256_raw
    evaluator_id = evaluator_id_raw
    evaluator_version = evaluator_version_raw
    binding_identity = json.dumps(
        {
            "manifest_hash": manifest.manifest_hash,
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "receipt_sha256": receipt_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    binding_id = "workload-eval-" + hashlib.sha256(binding_identity).hexdigest()[:32]
    return WorkloadEvaluationBinding(
        binding_id=binding_id,
        manifest_hash=manifest.manifest_hash,
        workload_profile_hash=manifest.workload_profile_hash,
        model_id=model_id,
        model_fingerprint=model_fingerprint,
        receipt_id=manifest.receipt_id,
        receipt_sha256=receipt_sha256,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        coverage=manifest.coverage,
    )


def aggregate_workload_evaluation_coverage(
    profile: WorkloadProfile,
    *,
    profile_hash: str,
    model_id: str,
    binding_payloads: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> WorkloadEvaluationCoverage:
    """Aggregate explicit bindings for one exact model/profile without name inference."""

    required = {
        "languages": tuple(profile.languages),
        "domains": tuple(profile.domains),
        "tasks": tuple(sorted(profile.task_weights)),
    }
    covered_casefold: dict[str, set[str]] = {kind: set() for kind in COVERAGE_KINDS}
    covered_labels: dict[str, set[str]] = {kind: set() for kind in COVERAGE_KINDS}
    binding_ids: set[str] = set()
    receipt_ids: set[str] = set()

    for payload in binding_payloads:
        if payload.get("workload_profile_hash") != profile_hash:
            continue
        if payload.get("model_id") != model_id:
            continue
        coverage = payload.get("coverage")
        if not isinstance(coverage, dict):
            continue
        binding_id = payload.get("binding_id")
        receipt_id = payload.get("receipt_id")
        if isinstance(binding_id, str):
            binding_ids.add(binding_id)
        if isinstance(receipt_id, str):
            receipt_ids.add(receipt_id)
        for kind in COVERAGE_KINDS:
            claims = coverage.get(kind)
            if not isinstance(claims, dict):
                continue
            for raw_label, selectors in claims.items():
                if (
                    not isinstance(raw_label, str)
                    or not isinstance(selectors, list)
                    or not selectors
                ):
                    continue
                covered_labels[kind].add(raw_label)
                covered_casefold[kind].add(raw_label.casefold())

    missing: dict[str, tuple[str, ...]] = {}
    covered: dict[str, tuple[str, ...]] = {}
    for kind, labels in required.items():
        if kind in {"languages", "domains"}:
            missing_labels = tuple(
                label for label in labels if label.casefold() not in covered_casefold[kind]
            )
            covered_required = tuple(
                label for label in labels if label.casefold() in covered_casefold[kind]
            )
        else:
            missing_labels = tuple(label for label in labels if label not in covered_labels[kind])
            covered_required = tuple(label for label in labels if label in covered_labels[kind])
        missing[kind] = missing_labels
        covered[kind] = covered_required

    complete = not any(missing.values())
    return WorkloadEvaluationCoverage(
        complete=complete,
        profile_hash=profile_hash,
        model_id=model_id,
        binding_ids=tuple(sorted(binding_ids)),
        receipt_ids=tuple(sorted(receipt_ids)),
        covered=covered,
        missing=missing,
    )
