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
LIGHTEVAL_ADAPTER_ID = "frontierwright.evaluator.lighteval-import"
LIGHTEVAL_ADAPTER_VERSION = "1"
LIGHTEVAL_EVALUATOR_ID = "huggingface.lighteval"


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


def _lighteval_task_config(
    config_tasks: dict[str, Any],
    result_task_id: str,
) -> dict[str, Any]:
    base_name = result_task_id
    if "|" in result_task_id:
        parts = result_task_id.split("|")
        if parts[-1].isdigit():
            base_name = "|".join(parts[:-1])
    matches: list[dict[str, Any]] = []
    for raw_config in config_tasks.values():
        if not isinstance(raw_config, dict):
            continue
        name = raw_config.get("name")
        if isinstance(name, str) and name in {result_task_id, base_name}:
            matches.append(raw_config)
    if len(matches) != 1:
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_TASK_CONFIG_AMBIGUOUS",
            (
                "LightEval task result must match exactly one config_tasks entry by its "
                f"declared task name: {result_task_id!r}. Found {len(matches)}."
            ),
            2,
        )
    return matches[0]


def _lighteval_metric_directions(task_config: dict[str, Any], task_id: str) -> dict[str, bool]:
    raw_metrics = task_config.get("metric")
    if not isinstance(raw_metrics, list):
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_INVALID",
            f"LightEval config for {task_id!r} must include metric definitions.",
            2,
        )
    directions: dict[str, bool] = {}
    for item in raw_metrics:
        if not isinstance(item, dict):
            continue
        name = item.get("metric_name")
        direction = item.get("higher_is_better")
        if isinstance(name, str) and name and isinstance(direction, bool):
            directions[name] = direction
    if not directions:
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_DIRECTION_UNKNOWN",
            f"LightEval task {task_id!r} has no metric direction metadata.",
            2,
        )
    return directions


def import_lighteval_results(
    path: Path,
    *,
    model_id: str,
    model_fingerprint: str,
    lighteval_version: str,
) -> ExternalEvaluationImport:
    """Convert a LightEval result JSON into raw Frontierwright evaluation evidence.

    The importer follows LightEval's saved result structure and requires exact metric
    direction metadata from config_tasks. Aggregate  rows are skipped because they
    mix task identities and therefore are unsuitable as atomic workload evidence.
    """

    version = lighteval_version.strip()
    if not version:
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_VERSION_REQUIRED",
            "An exact LightEval version or immutable revision is required.",
            2,
        )
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_INVALID",
            "LightEval results must be readable UTF-8 JSON.",
            2,
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_INVALID",
            "LightEval results root must be an object.",
            2,
        )

    results = _mapping(payload.get("results"), "results")
    versions = _mapping(payload.get("versions", {}), "versions")
    config_tasks = _mapping(payload.get("config_tasks", {}), "config_tasks")
    config_general = _mapping(payload.get("config_general", {}), "config_general")
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    measurements: list[RawMeasurement] = []
    stderr: dict[str, dict[str, float]] = {}
    task_versions: dict[str, str] = {}
    task_config_keys: dict[str, str] = {}

    for result_task_id, raw_task_result in sorted(results.items()):
        if result_task_id == "all":
            continue
        if not isinstance(result_task_id, str) or not result_task_id:
            raise FrontierwrightError(
                "EXTERNAL_EVALUATION_INVALID",
                "LightEval task result IDs must be nonempty strings.",
                2,
            )
        task_result = _mapping(raw_task_result, f"results[{result_task_id!r}]")
        raw_version = versions.get(result_task_id, "UNVERSIONED")
        if isinstance(raw_version, bool) or not isinstance(raw_version, (str, int, float)):
            raise FrontierwrightError(
                "EXTERNAL_EVALUATION_INVALID",
                f"LightEval task version for {result_task_id!r} must be scalar.",
                2,
            )
        task_version = str(raw_version)
        task_versions[result_task_id] = task_version
        task_config = _lighteval_task_config(config_tasks, result_task_id)
        directions = _lighteval_metric_directions(task_config, result_task_id)
        config_name = task_config.get("name")
        if isinstance(config_name, str):
            task_config_keys[result_task_id] = config_name

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
                    f"LightEval metric {result_task_id}/{metric_name} is not finite.",
                    2,
                )
            direction = directions.get(metric_name)
            if direction is None:
                raise FrontierwrightError(
                    "EXTERNAL_EVALUATION_DIRECTION_UNKNOWN",
                    f"LightEval metric direction is missing for {result_task_id}/{metric_name}.",
                    2,
                )
            measurements.append(
                RawMeasurement(
                    task_id=f"lighteval:{result_task_id}",
                    task_version=task_version,
                    metric=metric_name,
                    value=value,
                    higher_is_better=direction,
                )
            )
        if task_stderr:
            stderr[result_task_id] = task_stderr

    if not measurements:
        raise FrontierwrightError(
            "EXTERNAL_EVALUATION_EMPTY",
            "LightEval results contain no finite numeric task metrics.",
            2,
        )

    reported_sha = config_general.get("lighteval_sha")
    reported_model_sha = config_general.get("model_sha")
    identity = json.dumps(
        {
            "adapter_id": LIGHTEVAL_ADAPTER_ID,
            "adapter_version": LIGHTEVAL_ADAPTER_VERSION,
            "source_sha256": source_sha256,
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "evaluator_version": version,
            "reported_lighteval_sha": reported_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt_id = "receipt-lighteval-" + hashlib.sha256(identity).hexdigest()[:32]
    receipt = EvaluationReceipt(
        receipt_id=receipt_id,
        model_id=model_id,
        model_fingerprint=model_fingerprint,
        evaluator_id=LIGHTEVAL_EVALUATOR_ID,
        evaluator_version=version,
        conditions={
            "adapter_id": LIGHTEVAL_ADAPTER_ID,
            "adapter_version": LIGHTEVAL_ADAPTER_VERSION,
            "source_format": "lighteval.saved-results",
            "source_sha256": source_sha256,
            "source_filename": path.name,
            "reported_lighteval_sha": (reported_sha if isinstance(reported_sha, str) else None),
            "reported_model_sha": (
                reported_model_sha if isinstance(reported_model_sha, str) else None
            ),
            "model_name": (
                config_general.get("model_name")
                if isinstance(config_general.get("model_name"), str)
                else None
            ),
            "task_versions": task_versions,
            "task_config_names": task_config_keys,
            "metric_stderr": stderr,
            "aggregate_all_row_imported": False,
            "privacy_note": (
                "Only aggregate result JSON is imported. LightEval detail Parquet files may "
                "contain prompts/model responses and remain under the project's data boundary."
            ),
        },
        measurements=tuple(measurements),
    )
    return ExternalEvaluationImport(
        adapter_id=LIGHTEVAL_ADAPTER_ID,
        adapter_version=LIGHTEVAL_ADAPTER_VERSION,
        source_sha256=source_sha256,
        evaluator_id=LIGHTEVAL_EVALUATOR_ID,
        evaluator_version=version,
        receipt=receipt,
        task_count=len(task_versions),
        measurement_count=len(measurements),
        stderr_count=sum(len(item) for item in stderr.values()),
    )
