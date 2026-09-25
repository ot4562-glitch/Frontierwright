"""Origin-relative Frontierwright Stat v2 primitives.

Stat v2 deliberately separates raw benchmark evidence from the player-facing
origin-relative stat. The project origin is the permanent zero point; promoting
a new Champion never resets accumulated progress.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum

from frontierwright.benchmark_sources import StatAxisV2


class StatRelation(StrEnum):
    BETTER = "BETTER"
    WORSE = "WORSE"
    SAME = "SAME"
    UNCERTAIN = "UNCERTAIN"
    UNMEASURED = "UNMEASURED"


@dataclass(frozen=True)
class BenchmarkEstimate:
    benchmark_id: str
    benchmark_version: str
    metric: str
    value: float
    higher_is_better: bool = True
    confidence_interval: tuple[float, float] | None = None
    sample_count: int | None = None

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "benchmark_version", "metric"):
            value = getattr(self, name)
            if not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if not math.isfinite(self.value):
            raise ValueError("benchmark value must be finite")
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive when provided")
        if self.confidence_interval is not None:
            lower, upper = self.confidence_interval
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError("confidence_interval must be finite and ordered")
            if not lower <= self.value <= upper:
                raise ValueError("confidence_interval must contain value")


@dataclass(frozen=True)
class OriginRelativeStat:
    axis: StatAxisV2
    origin_value: float
    current_value: float
    display_delta: float
    relation: StatRelation
    delta_interval: tuple[float, float] | None
    source_count: int

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["axis"] = self.axis.value
        payload["relation"] = self.relation.value
        payload["delta_interval"] = (
            list(self.delta_interval) if self.delta_interval is not None else None
        )
        return payload


def compare_origin_estimates(
    *,
    axis: StatAxisV2,
    origin: BenchmarkEstimate,
    current: BenchmarkEstimate,
    display_scale: float = 100.0,
) -> OriginRelativeStat:
    """Compare one pinned benchmark against the permanent project origin."""

    identity = (origin.benchmark_id, origin.benchmark_version, origin.metric)
    if identity != (current.benchmark_id, current.benchmark_version, current.metric):
        raise ValueError("origin and current estimates must have identical benchmark identity")
    if origin.higher_is_better != current.higher_is_better:
        raise ValueError("origin and current estimates disagree on metric direction")
    if not math.isfinite(display_scale) or display_scale <= 0:
        raise ValueError("display_scale must be positive and finite")

    raw_delta = current.value - origin.value
    improvement_delta = raw_delta if current.higher_is_better else -raw_delta
    delta_interval: tuple[float, float] | None = None
    relation = (
        StatRelation.SAME
        if math.isclose(raw_delta, 0.0, abs_tol=1e-12)
        else (StatRelation.BETTER if improvement_delta > 0 else StatRelation.WORSE)
    )

    if origin.confidence_interval is not None and current.confidence_interval is not None:
        origin_lower, origin_upper = origin.confidence_interval
        current_lower, current_upper = current.confidence_interval
        raw_lower = current_lower - origin_upper
        raw_upper = current_upper - origin_lower
        if current.higher_is_better:
            improvement_lower, improvement_upper = raw_lower, raw_upper
        else:
            improvement_lower, improvement_upper = -raw_upper, -raw_lower
        delta_interval = (
            improvement_lower * display_scale,
            improvement_upper * display_scale,
        )
        if improvement_lower <= 0 <= improvement_upper:
            relation = StatRelation.UNCERTAIN
        elif improvement_lower > 0:
            relation = StatRelation.BETTER
        else:
            relation = StatRelation.WORSE

    source_count = sum(
        count for count in (origin.sample_count, current.sample_count) if count is not None
    )
    return OriginRelativeStat(
        axis=axis,
        origin_value=origin.value,
        current_value=current.value,
        display_delta=improvement_delta * display_scale,
        relation=relation,
        delta_interval=delta_interval,
        source_count=source_count,
    )
