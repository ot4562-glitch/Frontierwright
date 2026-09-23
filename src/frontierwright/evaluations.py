"""Evaluation receipts and frozen capability-scale application.

Frontierwright never turns an arbitrary benchmark number into a player-facing
capability stat unless a frozen scale manifest explicitly defines the mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from frontierwright.domain import Axis, CapabilityStat, require_text
from frontierwright.errors import FrontierwrightError


@dataclass(frozen=True)
class RawMeasurement:
    task_id: str
    task_version: str
    metric: str
    value: float
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        for field in ("task_id", "task_version", "metric"):
            require_text(getattr(self, field), field)
        if isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("raw measurement value must be finite")


@dataclass(frozen=True)
class EvaluationReceipt:
    receipt_id: str
    model_id: str
    model_fingerprint: str
    evaluator_id: str
    evaluator_version: str
    conditions: dict[str, object]
    measurements: tuple[RawMeasurement, ...]

    def __post_init__(self) -> None:
        for field in (
            "receipt_id",
            "model_id",
            "model_fingerprint",
            "evaluator_id",
            "evaluator_version",
        ):
            require_text(getattr(self, field), field)
        if not self.measurements:
            raise ValueError("evaluation receipt must contain at least one measurement")
        keys = [
            (item.task_id, item.task_version, item.metric)
            for item in self.measurements
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("evaluation receipt contains duplicate task/version/metric keys")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "conditions": self.conditions,
            "measurements": [asdict(item) for item in self.measurements],
        }

    @property
    def sha256(self) -> str:
        data = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class EvaluationPackDescriptor:
    pack_id: str
    pack_version: str
    title: str
    evaluator_id: str
    evaluator_version: str
    task_id: str
    task_version: str
    metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "pack_id",
            "pack_version",
            "title",
            "evaluator_id",
            "evaluator_version",
            "task_id",
            "task_version",
        ):
            require_text(getattr(self, field), field)
        if not self.metrics:
            raise ValueError("evaluation pack must expose at least one metric")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("evaluation pack metrics must be unique")
        for metric in self.metrics:
            require_text(metric, "metric")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metrics"] = list(self.metrics)
        return payload


REFERENCE_LM_PACK = EvaluationPackDescriptor(
    pack_id="frontierwright.eval.reference-heldout-lm",
    pack_version="1",
    title="Reference held-out causal language-model evaluation",
    evaluator_id="frontierwright.reference-lm-evaluator",
    evaluator_version="1",
    task_id="frontierwright.reference.heldout-causal-lm",
    task_version="1",
    metrics=("cross_entropy_nats_per_token", "perplexity"),
)

BUILTIN_EVALUATION_PACKS: tuple[EvaluationPackDescriptor, ...] = (
    REFERENCE_LM_PACK,
)


def evaluation_pack(pack_id: str) -> EvaluationPackDescriptor:
    for descriptor in BUILTIN_EVALUATION_PACKS:
        if descriptor.pack_id == pack_id:
            return descriptor
    raise FrontierwrightError(
        "EVALUATION_PACK_NOT_FOUND",
        f"Unknown evaluation pack: {pack_id}",
        3,
    )


@dataclass(frozen=True)
class ScaleTask:
    task_id: str
    task_version: str
    metric: str
    weight: float
    raw_anchor: float
    raw_unit: float
    display_per_unit: float
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        for field in ("task_id", "task_version", "metric"):
            require_text(getattr(self, field), field)
        for name in ("weight", "raw_anchor", "raw_unit", "display_per_unit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.weight <= 0:
            raise ValueError("scale task weight must be positive")
        if self.raw_unit <= 0:
            raise ValueError("scale task raw_unit must be positive")


@dataclass(frozen=True)
class AxisScale:
    axis: Axis
    display_anchor: float
    tasks: tuple[ScaleTask, ...]

    def __post_init__(self) -> None:
        if isinstance(self.display_anchor, bool) or not math.isfinite(self.display_anchor):
            raise ValueError("display_anchor must be finite")
        if not self.tasks:
            raise ValueError("axis scale must define at least one task")


@dataclass(frozen=True)
class CapabilityScale:
    scale_id: str
    scale_version: str
    axes: tuple[AxisScale, ...]
    frozen: bool

    def __post_init__(self) -> None:
        require_text(self.scale_id, "scale_id")
        require_text(self.scale_version, "scale_version")
        if not self.frozen:
            raise ValueError("capability scale must be explicitly frozen before use")
        if not self.axes:
            raise ValueError("capability scale must define at least one axis")
        if len({item.axis for item in self.axes}) != len(self.axes):
            raise ValueError("capability scale cannot repeat an axis")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "scale_id": self.scale_id,
            "scale_version": self.scale_version,
            "frozen": self.frozen,
            "axes": [
                {
                    "axis": axis.axis.value,
                    "display_anchor": axis.display_anchor,
                    "tasks": [asdict(task) for task in axis.tasks],
                }
                for axis in self.axes
            ],
        }

    @property
    def sha256(self) -> str:
        data = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


def _require_mapping(raw: object, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise FrontierwrightError("INVALID_EVALUATION_FILE", f"{label} must be an object.", 2)
    return raw


def _strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def load_evaluation_receipt(path: Path) -> EvaluationReceipt:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "INVALID_EVALUATION_FILE",
            "Evaluation receipt must be valid UTF-8 JSON.",
            2,
        ) from exc

    obj = _require_mapping(raw, "evaluation receipt")
    if obj.get("schema_version") != 1:
        raise FrontierwrightError(
            "INVALID_EVALUATION_FILE",
            "Evaluation receipt schema_version must be 1.",
            2,
        )

    measurements_raw = obj.get("measurements")
    if not isinstance(measurements_raw, list):
        raise FrontierwrightError(
            "INVALID_EVALUATION_FILE",
            "measurements must be a list.",
            2,
        )

    try:
        measurements = tuple(
            RawMeasurement(
                task_id=str(_require_mapping(item, "measurement")["task_id"]),
                task_version=str(_require_mapping(item, "measurement")["task_version"]),
                metric=str(_require_mapping(item, "measurement")["metric"]),
                value=float(_require_mapping(item, "measurement")["value"]),
                higher_is_better=_strict_bool(
                    _require_mapping(item, "measurement").get("higher_is_better", True),
                    "measurement.higher_is_better",
                ),
            )
            for item in measurements_raw
        )
        conditions = obj.get("conditions", {})
        if not isinstance(conditions, dict):
            raise ValueError("conditions must be an object")
        return EvaluationReceipt(
            receipt_id=str(obj["receipt_id"]),
            model_id=str(obj["model_id"]),
            model_fingerprint=str(obj["model_fingerprint"]),
            evaluator_id=str(obj["evaluator_id"]),
            evaluator_version=str(obj["evaluator_version"]),
            conditions=conditions,
            measurements=measurements,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "INVALID_EVALUATION_FILE",
            f"Invalid evaluation receipt: {exc}",
            2,
        ) from exc


def load_capability_scale(path: Path) -> CapabilityScale:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "INVALID_SCALE_FILE",
            "Capability scale must be valid UTF-8 JSON.",
            2,
        ) from exc

    obj = _require_mapping(raw, "capability scale")
    if obj.get("schema_version") != 1:
        raise FrontierwrightError(
            "INVALID_SCALE_FILE",
            "Capability scale schema_version must be 1.",
            2,
        )
    axes_raw = obj.get("axes")
    if not isinstance(axes_raw, list):
        raise FrontierwrightError("INVALID_SCALE_FILE", "axes must be a list.", 2)

    try:
        axes: list[AxisScale] = []
        for axis_raw in axes_raw:
            axis_obj = _require_mapping(axis_raw, "axis scale")
            tasks_raw = axis_obj["tasks"]
            if not isinstance(tasks_raw, list):
                raise ValueError("axis tasks must be a list")
            tasks = tuple(
                ScaleTask(
                    task_id=str(_require_mapping(item, "scale task")["task_id"]),
                    task_version=str(_require_mapping(item, "scale task")["task_version"]),
                    metric=str(_require_mapping(item, "scale task")["metric"]),
                    weight=float(_require_mapping(item, "scale task")["weight"]),
                    raw_anchor=float(_require_mapping(item, "scale task")["raw_anchor"]),
                    raw_unit=float(_require_mapping(item, "scale task")["raw_unit"]),
                    display_per_unit=float(
                        _require_mapping(item, "scale task")["display_per_unit"]
                    ),
                    higher_is_better=_strict_bool(
                        _require_mapping(item, "scale task").get("higher_is_better", True),
                        "scale task.higher_is_better",
                    ),
                )
                for item in tasks_raw
            )
            axes.append(
                AxisScale(
                    axis=Axis(str(axis_obj["axis"]).upper()),
                    display_anchor=float(axis_obj["display_anchor"]),
                    tasks=tasks,
                )
            )
        return CapabilityScale(
            scale_id=str(obj["scale_id"]),
            scale_version=str(obj["scale_version"]),
            axes=tuple(axes),
            frozen=_strict_bool(obj["frozen"], "frozen"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "INVALID_SCALE_FILE",
            f"Invalid capability scale: {exc}",
            2,
        ) from exc


def apply_scale(
    receipt: EvaluationReceipt,
    scale: CapabilityScale,
) -> tuple[CapabilityStat, ...]:
    measurements = {
        (item.task_id, item.task_version, item.metric): item
        for item in receipt.measurements
    }
    result: list[CapabilityStat] = []

    for axis_scale in scale.axes:
        weighted_delta = 0.0
        total_weight = 0.0
        missing: list[str] = []

        for task in axis_scale.tasks:
            key = (task.task_id, task.task_version, task.metric)
            measurement = measurements.get(key)
            if measurement is None:
                missing.append("/".join(key))
                continue
            if measurement.higher_is_better != task.higher_is_better:
                raise FrontierwrightError(
                    "EVALUATION_DIRECTION_MISMATCH",
                    f"Metric direction mismatch for {'/'.join(key)}.",
                    2,
                )
            direction = 1.0 if task.higher_is_better else -1.0
            normalized = direction * (measurement.value - task.raw_anchor) / task.raw_unit
            weighted_delta += task.weight * normalized * task.display_per_unit
            total_weight += task.weight

        if missing:
            # An axis is unknown unless every frozen contributing task is present.
            continue
        if total_weight <= 0:
            continue

        value = axis_scale.display_anchor + weighted_delta / total_weight
        result.append(
            CapabilityStat(
                axis=axis_scale.axis,
                value=value,
                scale_version=f"{scale.scale_id}:{scale.scale_version}:{scale.sha256}",
                evaluation_receipt=receipt.receipt_id,
            )
        )

    return tuple(result)
