"""Strict adapters for importing external evaluator evidence.

External benchmark outputs remain raw evaluation evidence. Frontierwright never turns
an arbitrary external score into a capability stat unless a separate frozen scale
explicitly maps the exact task/version/metric identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import EvaluationReceipt, RawMeasurement

LM_EVAL_ADAPTER_ID = "frontierwright.evaluator.lm-eval-import"
LM_EVAL_ADAPTER_VERSION = "1"
LM_EVAL_EVALUATOR_ID = "eleutherai.lm-evaluation-harness"


@dataclass(frozen=True)
class ExternalEvaluationImport:
    adapter_id: str
    adapter_version: str
    source_sha256: str
    evaluator_id: str
    evaluator_version: str
    receipt: EvaluationReceipt
    task_count: int
    measurement_count: int
    stderr_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "source_sha256": self.source_sha256,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "receipt_id": self.receipt.receipt_id,
            "receipt_sha256": self.receipt.sha256,
            "model_id": self.receipt.model_id,
            "model_fingerprint": self.receipt.model_fingerprint,
            "task_count": self.task_count,
            "measurement_count": self.measurement_count,
            "stderr_count": self.stderr_count,
            "capability_stats_activated": False,
            "note": (
                "Imported external metrics are raw evidence only. A frozen capability "
                "scale must explicitly map exact task/version/metric identities before "
                "they can become Frontierwright capability stats."
            ),
        }


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_INVALID",
            f"lm-eval {label} must be an object.",
            2,
        )
    return value


def _source_payload(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw_bytes = path.read_bytes()
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_INVALID",
            "lm-eval results must be readable UTF-8 JSON.",
            2,
        ) from exc
    if not isinstance(parsed, dict):
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_INVALID",
            "lm-eval results root must be an object.",
            2,
        )
    return raw_bytes, parsed


def _task_version(versions: dict[str, Any], task_name: str) -> str:
    raw = versions.get(task_name)
    if raw is None:
        return "UNVERSIONED"
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_INVALID",
            f"lm-eval task version for {task_name!r} must be scalar.",
            2,
        )
    return str(raw)


def _higher_is_better(mapping: dict[str, Any], task_name: str, metric_name: str) -> bool | None:
    task = mapping.get(task_name)
    if not isinstance(task, dict):
        return None
    raw = task.get(metric_name)
    if isinstance(raw, bool):
        return raw
    # Some lm-eval outputs key this mapping by the metric before its aggregation suffix.
    base_metric = metric_name.split(",", 1)[0]
    raw = task.get(base_metric)
    return raw if isinstance(raw, bool) else None


def _stderr_metric_name(metric_name: str) -> str | None:
    if metric_name.endswith(",stderr"):
        return metric_name[: -len(",stderr")]
    # Current result tables may carry the aggregation suffix after _stderr.
    if "_stderr," in metric_name:
        base, suffix = metric_name.split("_stderr,", 1)
        return f"{base},{suffix}"
    if metric_name.endswith("_stderr"):
        return metric_name[: -len("_stderr")]
    return None


def import_lm_eval_results(
    path: Path,
    *,
    model_id: str,
    model_fingerprint: str,
    harness_version: str,
) -> ExternalEvaluationImport:
    """Convert one lm-evaluation-harness result file into a raw Frontierwright receipt.

    The adapter intentionally ignores *,stderr entries as primary measurements and
    preserves them in receipt conditions instead. Non-numeric aggregate metadata is not
    silently coerced into a score.
    """

    if not harness_version.strip():
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_VERSION_REQUIRED",
            "An exact lm-evaluation-harness version/revision is required.",
            2,
        )

    raw_bytes, payload = _source_payload(path)
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    results = _mapping(payload.get("results"), "results")
    versions = _mapping(payload.get("versions", {}), "versions")
    higher_map = _mapping(payload.get("higher_is_better", {}), "higher_is_better")
    configs = _mapping(payload.get("configs", {}), "configs")
    n_samples = _mapping(payload.get("n-samples", {}), "n-samples")
    n_shot = _mapping(payload.get("n-shot", {}), "n-shot")

    measurements: list[RawMeasurement] = []
    stderr: dict[str, dict[str, float]] = {}
    unknown_direction: list[str] = []
    task_versions: dict[str, str] = {}

    for task_name, raw_task_result in sorted(results.items()):
        if not isinstance(task_name, str) or not task_name:
            raise FrontierwrightError(
                "EXTERNAL_EVALUATION_INVALID",
                "lm-eval task names must be nonempty strings.",
                2,
            )
        task_result = _mapping(raw_task_result, f"results[{task_name!r}]")
        version = _task_version(versions, task_name)
        task_versions[task_name] = version
        task_stderr: dict[str, float] = {}
        for metric_name, raw_value in sorted(task_result.items()):
            if not isinstance(metric_name, str) or not metric_name:
                continue
            stderr_metric = _stderr_metric_name(metric_name)
            if stderr_metric is not None:
                if (
                    not isinstance(raw_value, bool)
                    and isinstance(raw_value, (int, float))
                    and math.isfinite(float(raw_value))
                ):
                    task_stderr[stderr_metric] = float(raw_value)
                continue
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                raise FrontierwrightError(
                    "EXTERNAL_EVALUATION_INVALID",
                    f"lm-eval metric {task_name}/{metric_name} is not finite.",
                    2,
                )
            direction = _higher_is_better(higher_map, task_name, metric_name)
            if direction is None:
                unknown_direction.append(f"{task_name}/{metric_name}")
                continue
            measurements.append(
                RawMeasurement(
                    task_id=f"lm-eval:{task_name}",
                    task_version=version,
                    metric=metric_name,
                    value=value,
                    higher_is_better=direction,
                )
            )
        if task_stderr:
            stderr[task_name] = task_stderr

    if unknown_direction:
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_DIRECTION_UNKNOWN",
            "lm-eval result is missing higher_is_better for: " + ", ".join(unknown_direction),
            2,
        )
    if not measurements:
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_EMPTY",
            "lm-eval results contain no finite numeric primary metrics.",
            2,
        )

    identity = json.dumps(
        {
            "adapter_id": LM_EVAL_ADAPTER_ID,
            "adapter_version": LM_EVAL_ADAPTER_VERSION,
            "source_sha256": source_sha256,
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "evaluator_version": harness_version.strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt_id = "receipt-lm-eval-" + hashlib.sha256(identity).hexdigest()[:32]
    conditions: dict[str, object] = {
        "adapter_id": LM_EVAL_ADAPTER_ID,
        "adapter_version": LM_EVAL_ADAPTER_VERSION,
        "source_format": "lm-evaluation-harness.simple_evaluate",
        "source_sha256": source_sha256,
        "source_filename": path.name,
        "task_versions": task_versions,
        "configs": configs,
        "n_samples": n_samples,
        "n_shot": n_shot,
        "metric_stderr": stderr,
        "unknown_higher_is_better": unknown_direction,
        "samples_present": isinstance(payload.get("samples"), dict),
        "privacy_note": (
            "Sample-level prompts/responses are not copied into the receipt. Keep any "
            "lm-eval sample logs under the project's applicable data boundary."
        ),
    }
    receipt = EvaluationReceipt(
        receipt_id=receipt_id,
        model_id=model_id,
        model_fingerprint=model_fingerprint,
        evaluator_id=LM_EVAL_EVALUATOR_ID,
        evaluator_version=harness_version.strip(),
        conditions=conditions,
        measurements=tuple(measurements),
    )
    return ExternalEvaluationImport(
        adapter_id=LM_EVAL_ADAPTER_ID,
        adapter_version=LM_EVAL_ADAPTER_VERSION,
        source_sha256=source_sha256,
        evaluator_id=LM_EVAL_EVALUATOR_ID,
        evaluator_version=harness_version.strip(),
        receipt=receipt,
        task_count=len(task_versions),
        measurement_count=len(measurements),
        stderr_count=sum(len(item) for item in stderr.values()),
    )
