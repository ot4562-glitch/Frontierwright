from __future__ import annotations

import pytest

from frontierwright.service import _observed_range_improvement_interval
from frontierwright.workloads import (
    ParetoDirection,
    ParetoMetricInput,
    ParetoMetricRelation,
    ParetoRelation,
    compare_pareto_metrics,
)


def test_uncertain_interval_blocks_false_pareto_winner() -> None:
    comparison = compare_pareto_metrics(
        [
            ParetoMetricInput(
                key="serving.latency_p50",
                category="SERVING",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=0.0585,
                candidate_value=0.0588,
                improvement_interval=(-0.002, 0.001),
            )
        ]
    )

    assert comparison.relation is ParetoRelation.INSUFFICIENT_EVIDENCE
    assert comparison.metrics[0].relation is ParetoMetricRelation.UNCERTAIN
    payload = comparison.to_payload()
    assert payload["complete"] is False
    assert payload["uncertain_metric_count"] == 1
    assert payload["unqualified_dominance_claim_eligible"] is False


def test_nonoverlapping_improvement_interval_can_support_direction() -> None:
    comparison = compare_pareto_metrics(
        [
            ParetoMetricInput(
                key="serving.throughput_p50",
                category="SERVING",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=100.0,
                candidate_value=115.0,
                improvement_interval=(8.0, 21.0),
            )
        ]
    )

    assert comparison.relation is ParetoRelation.CANDIDATE_DOMINATES
    assert comparison.metrics[0].relation is ParetoMetricRelation.BETTER


def test_runtime_observed_ranges_are_direction_normalized() -> None:
    champion = {
        "latency_seconds_runs": [0.058, 0.060, 0.059],
        "tokens_per_second_runs": [130.0, 134.0, 132.0],
    }
    candidate = {
        "latency_seconds_runs": [0.057, 0.061, 0.0595],
        "tokens_per_second_runs": [131.0, 135.0, 133.0],
    }

    latency = _observed_range_improvement_interval(
        champion,
        candidate,
        samples_key="latency_seconds_runs",
        direction=ParetoDirection.LOWER_BETTER,
    )
    throughput = _observed_range_improvement_interval(
        champion,
        candidate,
        samples_key="tokens_per_second_runs",
        direction=ParetoDirection.HIGHER_BETTER,
    )

    assert latency is not None
    assert latency == pytest.approx((-0.003, 0.003))
    assert throughput is not None
    assert throughput == pytest.approx((-3.0, 5.0))
