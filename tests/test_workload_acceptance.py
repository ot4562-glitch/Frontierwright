from __future__ import annotations

from frontierwright.evaluations import RawMeasurement
from frontierwright.workload_acceptance import (
    AcceptanceEvidenceReceipt,
    AcceptanceEvidenceRule,
    AcceptanceEvidenceRuleKind,
    AcceptanceObservedRelation,
    AcceptanceOperator,
    AcceptanceStatus,
    WorkloadAcceptanceContractV1,
    WorkloadAcceptanceCriterion,
    assess_workload_acceptance,
    selector_key,
)
from frontierwright.workload_evaluations import WorkloadEvidenceSelector

WORKLOAD_HASH = "sha256:" + "1" * 64
MODEL_FP = "sha256:" + "2" * 64
RECEIPT_SHA = "a" * 64


def _criterion(
    criterion_id: str = "translation-quality",
    *,
    operator: AcceptanceOperator = AcceptanceOperator.AT_LEAST,
    threshold: float = 0.8,
    rule: AcceptanceEvidenceRuleKind = AcceptanceEvidenceRuleKind.DETERMINISTIC_POINT,
    confidence_level: float | None = None,
    unit: str = "accuracy_fraction",
    task_id: str = "translation-ko-en",
) -> WorkloadAcceptanceCriterion:
    return WorkloadAcceptanceCriterion(
        criterion_id=criterion_id,
        selector=WorkloadEvidenceSelector(
            task_id=task_id,
            task_version="v1",
            metric="accuracy",
        ),
        unit=unit,
        operator=operator,
        threshold=threshold,
        evidence_rule=AcceptanceEvidenceRule(
            kind=rule,
            confidence_level=confidence_level,
        ),
        evaluator_id="private-eval",
        evaluator_version="2026.09",
        required_conditions={"split": "heldout"},
    )


def _contract(*criteria: WorkloadAcceptanceCriterion) -> WorkloadAcceptanceContractV1:
    return WorkloadAcceptanceContractV1(
        name="My workload acceptance",
        workload_profile_hash=WORKLOAD_HASH,
        criteria=criteria or (_criterion(),),
    )


def _receipt(
    criterion: WorkloadAcceptanceCriterion,
    *,
    value: float,
    receipt_id: str = "receipt-1",
    unit: str | None = None,
    confidence_interval: tuple[float, float] | None = None,
    confidence_level: float | None = None,
    evaluator_id: str = "private-eval",
    evaluator_version: str = "2026.09",
    conditions: dict[str, object] | None = None,
) -> AcceptanceEvidenceReceipt:
    key = selector_key(criterion.selector)
    merged_conditions: dict[str, object] = {
        "split": "heldout",
        "measurement_units": {key: unit if unit is not None else criterion.unit},
    }
    if confidence_interval is not None or confidence_level is not None:
        stats: dict[str, object] = {}
        if confidence_interval is not None:
            stats["confidence_interval"] = list(confidence_interval)
        if confidence_level is not None:
            stats["confidence_level"] = confidence_level
        merged_conditions["measurement_uncertainty"] = {key: stats}
    if conditions is not None:
        merged_conditions.update(conditions)

    return AcceptanceEvidenceReceipt(
        receipt_id=receipt_id,
        receipt_sha256=RECEIPT_SHA,
        model_id="model-1",
        model_fingerprint=MODEL_FP,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        conditions=merged_conditions,
        measurements=(
            RawMeasurement(
                task_id=criterion.selector.task_id,
                task_version=criterion.selector.task_version,
                metric=criterion.selector.metric,
                value=value,
                higher_is_better=(criterion.operator is AcceptanceOperator.AT_LEAST),
            ),
        ),
    )


def _assess(
    contract: WorkloadAcceptanceContractV1,
    *receipts: AcceptanceEvidenceReceipt,
):
    return assess_workload_acceptance(
        contract,
        model_id="model-1",
        model_fingerprint=MODEL_FP,
        receipts=tuple(receipts),
    )


def test_deterministic_point_supports_both_threshold_directions_and_equality() -> None:
    minimum = _criterion(threshold=0.8)
    minimum_result = _assess(_contract(minimum), _receipt(minimum, value=0.8))
    assert minimum_result.overall_status is AcceptanceStatus.PASS
    assert minimum_result.criteria[0].observed_relation is AcceptanceObservedRelation.MEETS

    maximum = WorkloadAcceptanceCriterion(
        criterion_id="latency",
        selector=WorkloadEvidenceSelector("latency-task", "v1", "seconds"),
        unit="seconds",
        operator=AcceptanceOperator.AT_MOST,
        threshold=2.0,
        evidence_rule=AcceptanceEvidenceRule(AcceptanceEvidenceRuleKind.DETERMINISTIC_POINT),
        evaluator_id="private-eval",
        evaluator_version="2026.09",
        required_conditions={"split": "heldout"},
    )
    maximum_result = _assess(_contract(maximum), _receipt(maximum, value=2.0))
    assert maximum_result.overall_status is AcceptanceStatus.PASS

    miss = _assess(_contract(minimum), _receipt(minimum, value=0.79))
    assert miss.overall_status is AcceptanceStatus.FAIL
    assert miss.criteria[0].observed_relation is AcceptanceObservedRelation.MISSES


def test_explicit_interval_pass_fail_and_overlap_are_distinct() -> None:
    criterion = _criterion(
        threshold=0.8,
        rule=AcceptanceEvidenceRuleKind.EXPLICIT_INTERVAL,
        confidence_level=0.95,
    )
    contract = _contract(criterion)

    passed = _assess(
        contract,
        _receipt(
            criterion,
            value=0.87,
            confidence_interval=(0.82, 0.92),
            confidence_level=0.95,
        ),
    )
    assert passed.overall_status is AcceptanceStatus.PASS

    failed = _assess(
        contract,
        _receipt(
            criterion,
            value=0.70,
            confidence_interval=(0.65, 0.78),
            confidence_level=0.95,
        ),
    )
    assert failed.overall_status is AcceptanceStatus.FAIL

    overlap = _assess(
        contract,
        _receipt(
            criterion,
            value=0.81,
            confidence_interval=(0.74, 0.87),
            confidence_level=0.95,
        ),
    )
    assert overlap.overall_status is AcceptanceStatus.INCONCLUSIVE
    assert "overlaps" in overlap.criteria[0].reason


def test_explicit_interval_at_most_uses_the_correct_bounds() -> None:
    criterion = WorkloadAcceptanceCriterion(
        criterion_id="latency",
        selector=WorkloadEvidenceSelector("latency-task", "v1", "seconds"),
        unit="seconds",
        operator=AcceptanceOperator.AT_MOST,
        threshold=2.0,
        evidence_rule=AcceptanceEvidenceRule(
            AcceptanceEvidenceRuleKind.EXPLICIT_INTERVAL,
            confidence_level=0.95,
        ),
        evaluator_id="private-eval",
        evaluator_version="2026.09",
        required_conditions={"split": "heldout"},
    )
    contract = _contract(criterion)

    passed = _assess(
        contract,
        _receipt(
            criterion,
            value=1.6,
            confidence_interval=(1.4, 1.9),
            confidence_level=0.95,
        ),
    )
    assert passed.overall_status is AcceptanceStatus.PASS

    failed = _assess(
        contract,
        _receipt(
            criterion,
            value=2.5,
            confidence_interval=(2.2, 2.8),
            confidence_level=0.95,
        ),
    )
    assert failed.overall_status is AcceptanceStatus.FAIL

    overlap = _assess(
        contract,
        _receipt(
            criterion,
            value=2.0,
            confidence_interval=(1.8, 2.2),
            confidence_level=0.95,
        ),
    )
    assert overlap.overall_status is AcceptanceStatus.INCONCLUSIVE


def test_missing_or_incompatible_evidence_stays_unknown() -> None:
    interval = _criterion(
        rule=AcceptanceEvidenceRuleKind.EXPLICIT_INTERVAL,
        confidence_level=0.95,
    )
    no_interval = _assess(_contract(interval), _receipt(interval, value=0.9))
    assert no_interval.overall_status is AcceptanceStatus.UNKNOWN

    wrong_unit = _assess(
        _contract(interval),
        _receipt(
            interval,
            value=0.9,
            unit="percent",
            confidence_interval=(0.85, 0.95),
            confidence_level=0.95,
        ),
    )
    assert wrong_unit.overall_status is AcceptanceStatus.UNKNOWN
    assert "unit" in wrong_unit.criteria[0].reason.lower()

    wrong_level = _assess(
        _contract(interval),
        _receipt(
            interval,
            value=0.9,
            confidence_interval=(0.85, 0.95),
            confidence_level=0.90,
        ),
    )
    assert wrong_level.overall_status is AcceptanceStatus.UNKNOWN

    wrong_conditions = _receipt(
        interval,
        value=0.9,
        confidence_interval=(0.85, 0.95),
        confidence_level=0.95,
        conditions={"split": "train"},
    )
    conditions_result = _assess(_contract(interval), wrong_conditions)
    assert conditions_result.overall_status is AcceptanceStatus.UNKNOWN


def test_ambiguous_selected_receipts_do_not_choose_the_favorable_result() -> None:
    criterion = _criterion()
    contract = _contract(criterion)
    good = _receipt(criterion, value=0.9, receipt_id="receipt-good")
    bad = _receipt(criterion, value=0.2, receipt_id="receipt-bad")

    result = _assess(contract, good, bad)

    assert result.overall_status is AcceptanceStatus.UNKNOWN
    assert result.criteria[0].observed_value is None
    assert "ambiguous" in result.criteria[0].reason.lower()


def test_overall_status_precedence_is_fail_then_unknown_then_inconclusive_then_pass() -> None:
    failing = _criterion("fail", threshold=0.9, task_id="fail-task")
    unknown = _criterion("unknown", threshold=0.5, task_id="unknown-task")
    inconclusive = _criterion(
        "inconclusive",
        threshold=0.8,
        task_id="inconclusive-task",
        rule=AcceptanceEvidenceRuleKind.EXPLICIT_INTERVAL,
        confidence_level=0.95,
    )
    passing = _criterion("pass", threshold=0.5, task_id="pass-task")
    contract = _contract(failing, unknown, inconclusive, passing)

    result = _assess(
        contract,
        _receipt(failing, value=0.4, receipt_id="fail-receipt"),
        _receipt(
            inconclusive,
            value=0.8,
            receipt_id="inc-receipt",
            confidence_interval=(0.7, 0.9),
            confidence_level=0.95,
        ),
        _receipt(passing, value=0.9, receipt_id="pass-receipt"),
    )
    assert result.overall_status is AcceptanceStatus.FAIL

    no_fail_contract = _contract(unknown, inconclusive, passing)
    no_fail = _assess(
        no_fail_contract,
        _receipt(
            inconclusive,
            value=0.8,
            receipt_id="inc-receipt",
            confidence_interval=(0.7, 0.9),
            confidence_level=0.95,
        ),
        _receipt(passing, value=0.9, receipt_id="pass-receipt"),
    )
    assert no_fail.overall_status is AcceptanceStatus.UNKNOWN

    no_unknown_contract = _contract(inconclusive, passing)
    no_unknown = _assess(
        no_unknown_contract,
        _receipt(
            inconclusive,
            value=0.8,
            receipt_id="inc-receipt",
            confidence_interval=(0.7, 0.9),
            confidence_level=0.95,
        ),
        _receipt(passing, value=0.9, receipt_id="pass-receipt"),
    )
    assert no_unknown.overall_status is AcceptanceStatus.INCONCLUSIVE


def test_contract_identity_is_immutable_and_deterministic() -> None:
    first = _contract(_criterion(threshold=0.8))
    replay = _contract(_criterion(threshold=0.8))
    changed = _contract(_criterion(threshold=0.81))

    assert first.contract_hash == replay.contract_hash
    assert first.contract_id == replay.contract_id
    assert first.contract_hash != changed.contract_hash
    assert first.contract_id != changed.contract_id
