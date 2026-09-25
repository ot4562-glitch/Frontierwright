from __future__ import annotations

import pytest

from frontierwright.benchmark_sources import BENCHMARK_SOURCES, StatAxisV2
from frontierwright.stat_v2 import (
    BenchmarkEstimate,
    StatRelation,
    compare_origin_estimates,
)


def test_origin_is_permanent_zero_reference() -> None:
    origin = BenchmarkEstimate(
        benchmark_id="livecodebench",
        benchmark_version="2026-06",
        metric="pass_rate",
        value=0.30,
        confidence_interval=(0.28, 0.32),
        sample_count=400,
    )
    current = BenchmarkEstimate(
        benchmark_id="livecodebench",
        benchmark_version="2026-06",
        metric="pass_rate",
        value=0.41,
        confidence_interval=(0.39, 0.43),
        sample_count=400,
    )

    stat = compare_origin_estimates(
        axis=StatAxisV2.CODING,
        origin=origin,
        current=current,
    )

    assert stat.display_delta == pytest.approx(11.0)
    assert stat.relation is StatRelation.BETTER
    assert stat.origin_value == pytest.approx(0.30)
    assert stat.current_value == pytest.approx(0.41)


def test_overlapping_uncertainty_is_not_called_better_or_worse() -> None:
    origin = BenchmarkEstimate(
        benchmark_id="reasoning-suite",
        benchmark_version="1",
        metric="accuracy",
        value=0.60,
        confidence_interval=(0.56, 0.64),
        sample_count=128,
    )
    current = BenchmarkEstimate(
        benchmark_id="reasoning-suite",
        benchmark_version="1",
        metric="accuracy",
        value=0.62,
        confidence_interval=(0.58, 0.66),
        sample_count=128,
    )

    stat = compare_origin_estimates(
        axis=StatAxisV2.REASONING,
        origin=origin,
        current=current,
    )

    assert stat.display_delta == pytest.approx(2.0)
    assert stat.relation is StatRelation.UNCERTAIN
    assert stat.delta_interval is not None
    assert stat.delta_interval[0] < 0 < stat.delta_interval[1]


def test_benchmark_catalog_separates_model_system_and_framework_sources() -> None:
    by_id = {source.source_id: source for source in BENCHMARK_SOURCES}

    assert StatAxisV2.CONTEXT in by_id["longbench-v2"].axes
    assert by_id["terminal-bench"].kind.value == "SYSTEM"
    assert by_id["lm-eval"].kind.value == "FRAMEWORK"
    assert by_id["inspect-ai"].kind.value == "FRAMEWORK"
