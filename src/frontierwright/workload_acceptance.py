"""Versioned workload acceptance contracts and deterministic assessment.

Acceptance answers a narrow question: does exact stored evidence satisfy the user's
explicit workload success criteria? It does not create a universal model score, infer
missing units, choose favorable evidence, or promote a candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import RawMeasurement
from frontierwright.workload_evaluations import WorkloadEvidenceSelector

WORKLOAD_ACCEPTANCE_CONTRACT_SCHEMA_VERSION = 1
WORKLOAD_ACCEPTANCE_ASSESSMENT_SCHEMA_VERSION = 1


def workload_acceptance_contract_schema() -> dict[str, object]:
    """Return the public machine-readable schema for acceptance contract v1."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://frontierwright.local/schemas/workload-acceptance-v1.json",
        "title": "Frontierwright Workload Acceptance Contract v1",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "name", "workload_profile_hash", "criteria"],
        "properties": {
            "schema_version": {"const": WORKLOAD_ACCEPTANCE_CONTRACT_SCHEMA_VERSION},
            "name": {"type": "string", "minLength": 1},
            "workload_profile_hash": {
                "type": "string",
                "pattern": "^sha256:[0-9a-fA-F]{64}$",
            },
            "criteria": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "criterion_id",
                        "selector",
                        "unit",
                        "operator",
                        "threshold",
                        "evidence_rule",
                    ],
                    "properties": {
                        "criterion_id": {"type": "string", "minLength": 1},
                        "selector": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["task_id", "task_version", "metric"],
                            "properties": {
                                "task_id": {"type": "string", "minLength": 1},
                                "task_version": {"type": "string", "minLength": 1},
                                "metric": {"type": "string", "minLength": 1},
                            },
                        },
                        "unit": {"type": "string", "minLength": 1},
                        "operator": {"enum": ["AT_LEAST", "AT_MOST"]},
                        "threshold": {"type": "number"},
                        "evidence_rule": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["kind"],
                                    "properties": {
                                        "kind": {"const": "DETERMINISTIC_POINT"},
                                        "confidence_level": {"type": "null"},
                                    },
                                },
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["kind", "confidence_level"],
                                    "properties": {
                                        "kind": {"const": "EXPLICIT_INTERVAL"},
                                        "confidence_level": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                            "exclusiveMaximum": 1,
                                        },
                                    },
                                },
                            ]
                        },
                        "evaluator_id": {"type": ["string", "null"]},
                        "evaluator_version": {"type": ["string", "null"]},
                        "required_conditions": {"type": "object"},
                    },
                },
            },
        },
    }


def workload_acceptance_contract_example(workload_profile_hash: str) -> dict[str, object]:
    """Return one complete contract example bound to an exact workload profile."""

    _sha256_identity(workload_profile_hash, "workload_profile_hash")
    return {
        "schema_version": WORKLOAD_ACCEPTANCE_CONTRACT_SCHEMA_VERSION,
        "name": "My workload success criteria",
        "workload_profile_hash": workload_profile_hash,
        "criteria": [
            {
                "criterion_id": "task-success",
                "selector": {
                    "task_id": "my-heldout-task",
                    "task_version": "1",
                    "metric": "accuracy",
                },
                "unit": "fraction",
                "operator": "AT_LEAST",
                "threshold": 0.8,
                "evidence_rule": {
                    "kind": "EXPLICIT_INTERVAL",
                    "confidence_level": 0.95,
                },
                "evaluator_id": None,
                "evaluator_version": None,
                "required_conditions": {"split": "heldout"},
            }
        ],
    }


class AcceptanceOperator(StrEnum):
    AT_LEAST = "AT_LEAST"
    AT_MOST = "AT_MOST"


class AcceptanceEvidenceRuleKind(StrEnum):
    DETERMINISTIC_POINT = "DETERMINISTIC_POINT"
    EXPLICIT_INTERVAL = "EXPLICIT_INTERVAL"


class AcceptanceStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNKNOWN = "UNKNOWN"


class AcceptanceObservedRelation(StrEnum):
    MEETS = "MEETS"
    MISSES = "MISSES"
    UNAVAILABLE = "UNAVAILABLE"


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"{field_name} must be a nonempty string.",
            2,
        )
    cleaned = value.strip()
    if "\x00" in cleaned:
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"{field_name} must be NUL-free.",
            2,
        )
    return cleaned


def _sha256_identity(value: object, field_name: str) -> str:
    text = _nonempty(value, field_name)
    if not text.startswith("sha256:") or len(text) != 71:
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"{field_name} must be sha256:<64-hex>.",
            2,
        )
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"{field_name} must be sha256:<64-hex>.",
            2,
        ) from exc
    return text


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"{field_name} must be numeric.",
            2,
        )
    result = float(value)
    if not math.isfinite(result):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"{field_name} must be finite.",
            2,
        )
    return result


def _canonical_json_object(value: object, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"{field_name} must be a JSON object.",
            2,
        )
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"{field_name} must contain canonical JSON-compatible values.",
            2,
        ) from exc
    assert isinstance(decoded, dict)
    return decoded


def selector_key(selector: WorkloadEvidenceSelector) -> str:
    return json.dumps(
        selector.to_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class AcceptanceEvidenceRule:
    kind: AcceptanceEvidenceRuleKind
    confidence_level: float | None = None

    def __post_init__(self) -> None:
        if self.kind is AcceptanceEvidenceRuleKind.DETERMINISTIC_POINT:
            if self.confidence_level is not None:
                raise FrontierwrightError(
                    "WORKLOAD_ACCEPTANCE_INVALID",
                    "DETERMINISTIC_POINT must not declare confidence_level.",
                    2,
                )
            return
        if self.confidence_level is None:
            raise FrontierwrightError(
                "WORKLOAD_ACCEPTANCE_INVALID",
                "EXPLICIT_INTERVAL requires confidence_level.",
                2,
            )
        level = _finite(self.confidence_level, "confidence_level")
        if not 0 < level < 1:
            raise FrontierwrightError(
                "WORKLOAD_ACCEPTANCE_INVALID",
                "confidence_level must be strictly between 0 and 1.",
                2,
            )
        object.__setattr__(self, "confidence_level", level)

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class WorkloadAcceptanceCriterion:
    criterion_id: str
    selector: WorkloadEvidenceSelector
    unit: str
    operator: AcceptanceOperator
    threshold: float
    evidence_rule: AcceptanceEvidenceRule
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    required_conditions: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "criterion_id", _nonempty(self.criterion_id, "criterion_id"))
        object.__setattr__(self, "unit", _nonempty(self.unit, "unit"))
        object.__setattr__(self, "threshold", _finite(self.threshold, "threshold"))
        if self.evaluator_id is not None:
            object.__setattr__(
                self,
                "evaluator_id",
                _nonempty(self.evaluator_id, "evaluator_id"),
            )
        if self.evaluator_version is not None:
            object.__setattr__(
                self,
                "evaluator_version",
                _nonempty(self.evaluator_version, "evaluator_version"),
            )
        if (self.evaluator_id is None) != (self.evaluator_version is None):
            raise FrontierwrightError(
                "WORKLOAD_ACCEPTANCE_INVALID",
                "evaluator_id and evaluator_version must be specified together.",
                2,
            )
        object.__setattr__(
            self,
            "required_conditions",
            _canonical_json_object(self.required_conditions, "required_conditions"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "criterion_id": self.criterion_id,
            "selector": self.selector.to_payload(),
            "unit": self.unit,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "evidence_rule": self.evidence_rule.to_payload(),
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "required_conditions": self.required_conditions,
        }


@dataclass(frozen=True)
class WorkloadAcceptanceContractV1:
    name: str
    workload_profile_hash: str
    criteria: tuple[WorkloadAcceptanceCriterion, ...]
    schema_version: int = WORKLOAD_ACCEPTANCE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_ACCEPTANCE_CONTRACT_SCHEMA_VERSION:
            raise FrontierwrightError(
                "WORKLOAD_ACCEPTANCE_INVALID",
                f"Unsupported acceptance contract schema_version {self.schema_version}.",
                2,
            )
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        object.__setattr__(
            self,
            "workload_profile_hash",
            _sha256_identity(self.workload_profile_hash, "workload_profile_hash"),
        )
        if not self.criteria:
            raise FrontierwrightError(
                "WORKLOAD_ACCEPTANCE_INVALID",
                "Acceptance contract must contain at least one criterion.",
                2,
            )
        ordered = tuple(sorted(self.criteria, key=lambda item: item.criterion_id))
        ids = [item.criterion_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise FrontierwrightError(
                "WORKLOAD_ACCEPTANCE_INVALID",
                "Acceptance criterion IDs must be unique.",
                2,
            )
        object.__setattr__(self, "criteria", ordered)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "workload_profile_hash": self.workload_profile_hash,
            "criteria": [criterion.to_payload() for criterion in self.criteria],
        }

    @property
    def contract_hash(self) -> str:
        encoded = json.dumps(
            self.to_payload(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @property
    def contract_id(self) -> str:
        return "acceptance-" + self.contract_hash.removeprefix("sha256:")[:24]


@dataclass(frozen=True)
class AcceptanceEvidenceReceipt:
    receipt_id: str
    receipt_sha256: str
    model_id: str
    model_fingerprint: str
    evaluator_id: str
    evaluator_version: str
    conditions: dict[str, object]
    measurements: tuple[RawMeasurement, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "receipt_id",
            "receipt_sha256",
            "model_id",
            "model_fingerprint",
            "evaluator_id",
            "evaluator_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "conditions",
            _canonical_json_object(self.conditions, "conditions"),
        )

    def reference_payload(self) -> dict[str, str]:
        return {
            "receipt_id": self.receipt_id,
            "receipt_sha256": self.receipt_sha256,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
        }


@dataclass(frozen=True)
class WorkloadAcceptanceCriterionResult:
    criterion_id: str
    status: AcceptanceStatus
    observed_relation: AcceptanceObservedRelation
    reason: str
    selector: dict[str, str]
    unit: str
    operator: str
    threshold: float
    evidence_rule: dict[str, object]
    observed_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    evidence_receipt_id: str | None = None
    evidence_receipt_sha256: str | None = None
    evaluator_id: str | None = None
    evaluator_version: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["observed_relation"] = self.observed_relation.value
        payload["confidence_interval"] = (
            list(self.confidence_interval) if self.confidence_interval is not None else None
        )
        return payload


@dataclass(frozen=True)
class WorkloadAcceptanceAssessmentV1:
    contract_id: str
    contract_hash: str
    workload_profile_hash: str
    model_id: str
    model_fingerprint: str
    evidence_refs: tuple[dict[str, str], ...]
    criteria: tuple[WorkloadAcceptanceCriterionResult, ...]
    overall_status: AcceptanceStatus
    schema_version: int = WORKLOAD_ACCEPTANCE_ASSESSMENT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "workload_profile_hash": self.workload_profile_hash,
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "criteria": [item.to_payload() for item in self.criteria],
            "overall_status": self.overall_status.value,
            "note": (
                "PASS means every configured criterion passed under its declared evidence rule. "
                "It is not a joint confidence guarantee and does not auto-promote a model."
            ),
        }

    @property
    def assessment_hash(self) -> str:
        encoded = json.dumps(
            self.to_payload(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @property
    def assessment_id(self) -> str:
        return "acceptance-assessment-" + self.assessment_hash.removeprefix("sha256:")[:24]


def _criterion_from_payload(payload: object, index: int) -> WorkloadAcceptanceCriterion:
    if not isinstance(payload, dict):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"criteria[{index}] must be an object.",
            2,
        )
    selector_payload = payload.get("selector")
    if not isinstance(selector_payload, dict):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"criteria[{index}].selector must be an object.",
            2,
        )
    operator_raw = str(payload.get("operator", "")).upper()
    try:
        operator = AcceptanceOperator(operator_raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in AcceptanceOperator)
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"criteria[{index}].operator must be one of: {allowed}.",
            2,
        ) from exc

    rule_payload = payload.get("evidence_rule")
    if not isinstance(rule_payload, dict):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"criteria[{index}].evidence_rule must be an object.",
            2,
        )
    rule_kind_raw = str(rule_payload.get("kind", "")).upper()
    try:
        rule_kind = AcceptanceEvidenceRuleKind(rule_kind_raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in AcceptanceEvidenceRuleKind)
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            f"criteria[{index}].evidence_rule.kind must be one of: {allowed}.",
            2,
        ) from exc

    confidence_raw = rule_payload.get("confidence_level")
    if confidence_raw is None:
        confidence_level = None
    else:
        try:
            confidence_level = float(confidence_raw)
        except (TypeError, ValueError) as exc:
            raise FrontierwrightError(
                "WORKLOAD_ACCEPTANCE_INVALID",
                f"criteria[{index}].evidence_rule.confidence_level must be numeric.",
                2,
            ) from exc
    evidence_rule = AcceptanceEvidenceRule(
        kind=rule_kind,
        confidence_level=confidence_level,
    )

    return WorkloadAcceptanceCriterion(
        criterion_id=str(payload.get("criterion_id", "")),
        selector=WorkloadEvidenceSelector(
            task_id=str(selector_payload.get("task_id", "")),
            task_version=str(selector_payload.get("task_version", "")),
            metric=str(selector_payload.get("metric", "")),
        ),
        unit=str(payload.get("unit", "")),
        operator=operator,
        threshold=_finite(payload.get("threshold"), f"criteria[{index}].threshold"),
        evidence_rule=evidence_rule,
        evaluator_id=(
            str(payload["evaluator_id"]) if payload.get("evaluator_id") is not None else None
        ),
        evaluator_version=(
            str(payload["evaluator_version"])
            if payload.get("evaluator_version") is not None
            else None
        ),
        required_conditions=_canonical_json_object(
            payload.get("required_conditions"),
            f"criteria[{index}].required_conditions",
        ),
    )


def acceptance_contract_from_payload(payload: dict[str, object]) -> WorkloadAcceptanceContractV1:
    raw_criteria = payload.get("criteria")
    if not isinstance(raw_criteria, list):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            "criteria must be a list.",
            2,
        )
    raw_schema = payload.get(
        "schema_version",
        WORKLOAD_ACCEPTANCE_CONTRACT_SCHEMA_VERSION,
    )
    if isinstance(raw_schema, bool) or not isinstance(raw_schema, (int, float, str)):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            "schema_version must be an integer.",
            2,
        )
    return WorkloadAcceptanceContractV1(
        name=str(payload.get("name", "")),
        workload_profile_hash=str(payload.get("workload_profile_hash", "")),
        criteria=tuple(
            _criterion_from_payload(item, index)
            for index, item in enumerate(raw_criteria)
        ),
        schema_version=int(raw_schema),
    )


def load_workload_acceptance_contract(path: Path) -> WorkloadAcceptanceContractV1:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            "Acceptance contract must be readable UTF-8 JSON.",
            2,
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierwrightError(
            "WORKLOAD_ACCEPTANCE_INVALID",
            "Acceptance contract root must be an object.",
            2,
        )
    return acceptance_contract_from_payload(payload)


def _measurement_match(
    criterion: WorkloadAcceptanceCriterion,
    receipt: AcceptanceEvidenceReceipt,
) -> RawMeasurement | None:
    matches = [
        item
        for item in receipt.measurements
        if (
            item.task_id == criterion.selector.task_id
            and item.task_version == criterion.selector.task_version
            and item.metric == criterion.selector.metric
        )
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _required_conditions_match(
    required: dict[str, object],
    actual: dict[str, object],
) -> bool:
    return all(key in actual and actual[key] == value for key, value in required.items())


def _unknown_result(
    criterion: WorkloadAcceptanceCriterion,
    reason: str,
) -> WorkloadAcceptanceCriterionResult:
    return WorkloadAcceptanceCriterionResult(
        criterion_id=criterion.criterion_id,
        status=AcceptanceStatus.UNKNOWN,
        observed_relation=AcceptanceObservedRelation.UNAVAILABLE,
        reason=reason,
        selector=criterion.selector.to_payload(),
        unit=criterion.unit,
        operator=criterion.operator.value,
        threshold=criterion.threshold,
        evidence_rule=criterion.evidence_rule.to_payload(),
    )


def _overall_status(
    results: tuple[WorkloadAcceptanceCriterionResult, ...],
) -> AcceptanceStatus:
    statuses = {item.status for item in results}
    if AcceptanceStatus.FAIL in statuses:
        return AcceptanceStatus.FAIL
    if AcceptanceStatus.UNKNOWN in statuses:
        return AcceptanceStatus.UNKNOWN
    if AcceptanceStatus.INCONCLUSIVE in statuses:
        return AcceptanceStatus.INCONCLUSIVE
    return AcceptanceStatus.PASS


def assess_workload_acceptance(
    contract: WorkloadAcceptanceContractV1,
    *,
    model_id: str,
    model_fingerprint: str,
    receipts: tuple[AcceptanceEvidenceReceipt, ...],
) -> WorkloadAcceptanceAssessmentV1:
    """Assess one model against one immutable contract and explicit evidence set."""

    model_id = _nonempty(model_id, "model_id")
    model_fingerprint = _nonempty(model_fingerprint, "model_fingerprint")
    unique_receipts: dict[str, AcceptanceEvidenceReceipt] = {}
    for receipt in receipts:
        if receipt.receipt_id in unique_receipts:
            if unique_receipts[receipt.receipt_id] != receipt:
                raise FrontierwrightError(
                    "WORKLOAD_ACCEPTANCE_EVIDENCE_CONFLICT",
                    "Duplicate receipt ID refers to different evidence.",
                    13,
                )
            continue
        if receipt.model_id != model_id or receipt.model_fingerprint != model_fingerprint:
            raise FrontierwrightError(
                "WORKLOAD_ACCEPTANCE_EVIDENCE_MODEL_MISMATCH",
                "Every selected receipt must match the assessed model and fingerprint.",
                12,
            )
        unique_receipts[receipt.receipt_id] = receipt
    ordered_receipts = tuple(
        unique_receipts[key] for key in sorted(unique_receipts)
    )

    results: list[WorkloadAcceptanceCriterionResult] = []
    for criterion in contract.criteria:
        candidates: list[tuple[AcceptanceEvidenceReceipt, RawMeasurement]] = []
        for receipt in ordered_receipts:
            measurement = _measurement_match(criterion, receipt)
            if measurement is None:
                continue
            if (
                criterion.evaluator_id is not None
                and (
                    receipt.evaluator_id != criterion.evaluator_id
                    or receipt.evaluator_version != criterion.evaluator_version
                )
            ):
                continue
            if not _required_conditions_match(
                criterion.required_conditions,
                receipt.conditions,
            ):
                continue
            candidates.append((receipt, measurement))

        if not candidates:
            results.append(
                _unknown_result(
                    criterion,
                    "No explicitly selected compatible receipt contains this exact criterion.",
                )
            )
            continue
        if len(candidates) > 1:
            results.append(
                _unknown_result(
                    criterion,
                    (
                        "Multiple selected receipts satisfy this criterion; "
                        "evidence selection is ambiguous."
                    ),
                )
            )
            continue

        receipt, measurement = candidates[0]
        expected_direction = criterion.operator is AcceptanceOperator.AT_LEAST
        if measurement.higher_is_better is not expected_direction:
            results.append(
                _unknown_result(
                    criterion,
                    "Metric direction conflicts with the acceptance operator.",
                )
            )
            continue

        key = selector_key(criterion.selector)
        raw_units = receipt.conditions.get("measurement_units")
        unit = raw_units.get(key) if isinstance(raw_units, dict) else None
        if unit != criterion.unit:
            results.append(
                _unknown_result(
                    criterion,
                    "Evidence unit is missing or does not match the acceptance contract.",
                )
            )
            continue

        observed = float(measurement.value)
        relation: AcceptanceObservedRelation
        status: AcceptanceStatus
        interval: tuple[float, float] | None = None
        reason: str

        if criterion.evidence_rule.kind is AcceptanceEvidenceRuleKind.DETERMINISTIC_POINT:
            meets = (
                observed >= criterion.threshold
                if criterion.operator is AcceptanceOperator.AT_LEAST
                else observed <= criterion.threshold
            )
            relation = (
                AcceptanceObservedRelation.MEETS
                if meets
                else AcceptanceObservedRelation.MISSES
            )
            status = AcceptanceStatus.PASS if meets else AcceptanceStatus.FAIL
            reason = "Deterministic point evidence satisfies the threshold." if meets else (
                "Deterministic point evidence misses the threshold."
            )
        else:
            raw_uncertainty = receipt.conditions.get("measurement_uncertainty")
            stats = raw_uncertainty.get(key) if isinstance(raw_uncertainty, dict) else None
            raw_interval = stats.get("confidence_interval") if isinstance(stats, dict) else None
            raw_level = stats.get("confidence_level") if isinstance(stats, dict) else None
            if (
                not isinstance(raw_interval, list)
                or len(raw_interval) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in raw_interval
                )
                or isinstance(raw_level, bool)
                or not isinstance(raw_level, (int, float))
                or not math.isfinite(float(raw_level))
                or criterion.evidence_rule.confidence_level is None
                or not math.isclose(
                    float(raw_level),
                    criterion.evidence_rule.confidence_level,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                results.append(
                    _unknown_result(
                        criterion,
                        "Required explicit confidence interval/level is missing or incompatible.",
                    )
                )
                continue
            lower = float(raw_interval[0])
            upper = float(raw_interval[1])
            if lower > upper:
                results.append(
                    _unknown_result(
                        criterion,
                        "Stored confidence interval is invalid.",
                    )
                )
                continue
            interval = (lower, upper)
            if criterion.operator is AcceptanceOperator.AT_LEAST:
                if lower >= criterion.threshold:
                    status = AcceptanceStatus.PASS
                    relation = AcceptanceObservedRelation.MEETS
                    reason = (
                        "The entire configured confidence interval meets the minimum threshold."
                    )
                elif upper < criterion.threshold:
                    status = AcceptanceStatus.FAIL
                    relation = AcceptanceObservedRelation.MISSES
                    reason = (
                        "The entire configured confidence interval is below the minimum threshold."
                    )
                else:
                    status = AcceptanceStatus.INCONCLUSIVE
                    relation = AcceptanceObservedRelation.UNAVAILABLE
                    reason = "The configured confidence interval overlaps the minimum threshold."
            else:
                if upper <= criterion.threshold:
                    status = AcceptanceStatus.PASS
                    relation = AcceptanceObservedRelation.MEETS
                    reason = (
                        "The entire configured confidence interval meets the maximum threshold."
                    )
                elif lower > criterion.threshold:
                    status = AcceptanceStatus.FAIL
                    relation = AcceptanceObservedRelation.MISSES
                    reason = (
                        "The entire configured confidence interval is above the maximum threshold."
                    )
                else:
                    status = AcceptanceStatus.INCONCLUSIVE
                    relation = AcceptanceObservedRelation.UNAVAILABLE
                    reason = "The configured confidence interval overlaps the maximum threshold."

        results.append(
            WorkloadAcceptanceCriterionResult(
                criterion_id=criterion.criterion_id,
                status=status,
                observed_relation=relation,
                reason=reason,
                selector=criterion.selector.to_payload(),
                unit=criterion.unit,
                operator=criterion.operator.value,
                threshold=criterion.threshold,
                evidence_rule=criterion.evidence_rule.to_payload(),
                observed_value=observed,
                confidence_interval=interval,
                evidence_receipt_id=receipt.receipt_id,
                evidence_receipt_sha256=receipt.receipt_sha256,
                evaluator_id=receipt.evaluator_id,
                evaluator_version=receipt.evaluator_version,
            )
        )

    result_tuple = tuple(results)
    evidence_refs = tuple(
        receipt.reference_payload() for receipt in ordered_receipts
    )
    return WorkloadAcceptanceAssessmentV1(
        contract_id=contract.contract_id,
        contract_hash=contract.contract_hash,
        workload_profile_hash=contract.workload_profile_hash,
        model_id=model_id,
        model_fingerprint=model_fingerprint,
        evidence_refs=evidence_refs,
        criteria=result_tuple,
        overall_status=_overall_status(result_tuple),
    )
