"""Adapters for importing measured external serving benchmark evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontierwright.errors import FrontierwrightError
from frontierwright.execution import BackendDataBoundary

VLLM_SERVE_ADAPTER_ID = "frontierwright.serving.vllm-bench-serve-import"
VLLM_SERVE_ADAPTER_VERSION = "1"
SERVING_RESOURCE_ADAPTER_ID = "frontierwright.serving.resource-manifest-import"
SERVING_RESOURCE_ADAPTER_VERSION = "1"

# Result fields emitted by vLLM's online serve benchmark. They are intentionally
# excluded from the condition hash; the hash describes the benchmark workload, not its
# outcome.
_VLLM_RESULT_FIELDS = {
    "completed",
    "failed",
    "duration",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "percentiles_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "percentiles_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "percentiles_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "std_e2el_ms",
    "percentiles_e2el_ms",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
    "outputs",
    "date",
}

# Model identity is pinned separately by Frontierwright and therefore must not make two
# otherwise identical Champion/Candidate benchmark workloads incomparable.
_VLLM_MODEL_IDENTITY_FIELDS = {
    "model_id",
    "model",
    "served_model_name",
    "tokenizer_id",
    "tokenizer",
}


@dataclass(frozen=True)
class ServingResourceImport:
    source_sha256: str
    runtime_id: str
    runtime_version: str
    model_fingerprint: str
    profile_condition_hash: str
    execution_boundary: BackendDataBoundary
    metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": SERVING_RESOURCE_ADAPTER_ID,
            "adapter_version": SERVING_RESOURCE_ADAPTER_VERSION,
            "source_sha256": self.source_sha256,
            "runtime_id": self.runtime_id,
            "runtime_version": self.runtime_version,
            "model_fingerprint": self.model_fingerprint,
            "profile_condition_hash": self.profile_condition_hash,
            "execution_boundary": self.execution_boundary.value,
            "metrics": dict(self.metrics),
        }


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            f"{field} must be a nonempty string.",
            2,
        )
    return value.strip()


def _sha256_identity(value: object, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    if not text.startswith("sha256:") or len(text) != 71:
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            f"{field} must be a sha256:<64-hex> identity.",
            2,
        )
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            f"{field} must be a sha256:<64-hex> identity.",
            2,
        ) from exc
    return text


def _nonnegative_int_metric(metrics: dict[str, Any], key: str) -> int | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            f"resource metric {key!r} must be a nonnegative integer number of bytes.",
            2,
        )
    return int(raw)


def import_serving_resource_manifest(path: Path) -> ServingResourceImport:
    """Import server-side memory evidence pinned to an exact serving condition.

    The manifest is deliberately framework-neutral.  It does not infer memory from model
    size and it cannot contribute latency/throughput metrics.  Those belong to a client
    or runtime benchmark receipt with the same condition hash.
    """

    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            "Serving resource manifest must be readable UTF-8 JSON.",
            2,
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            "Serving resource manifest must be a schema_version=1 JSON object.",
            2,
        )

    runtime_id = _nonempty_string(raw.get("runtime_id"), field="runtime_id")
    runtime_version = _nonempty_string(raw.get("runtime_version"), field="runtime_version")
    model_fingerprint = _sha256_identity(raw.get("model_fingerprint"), field="model_fingerprint")
    profile_condition_hash = _sha256_identity(
        raw.get("profile_condition_hash"), field="profile_condition_hash"
    )
    measurement_scope = _nonempty_string(raw.get("measurement_scope"), field="measurement_scope")
    try:
        execution_boundary = BackendDataBoundary(
            _nonempty_string(raw.get("execution_boundary"), field="execution_boundary").upper()
        )
    except ValueError as exc:
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            "execution_boundary must be LOCAL_MACHINE, CONTROLLED_PRIVATE, or EXTERNAL.",
            2,
        ) from exc
    if execution_boundary is BackendDataBoundary.UNKNOWN:
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            "UNKNOWN execution_boundary is not valid measured serving evidence.",
            2,
        )

    raw_metrics = raw.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            "metrics must be a JSON object.",
            2,
        )
    allowed = (
        "peak_vram_bytes",
        "max_sampled_process_rss_bytes",
        "cuda_memory_total_bytes",
        "cuda_memory_free_min_sampled_bytes",
        "cuda_memory_free_after_profile_bytes",
    )
    measured = {key: _nonnegative_int_metric(raw_metrics, key) for key in allowed}
    if all(value is None for value in measured.values()):
        raise FrontierwrightError(
            "SERVING_RESOURCE_MANIFEST_INVALID",
            "At least one server memory metric must be measured.",
            2,
        )
    total = measured["cuda_memory_total_bytes"]
    for free_key in (
        "cuda_memory_free_min_sampled_bytes",
        "cuda_memory_free_after_profile_bytes",
    ):
        free = measured[free_key]
        if total is not None and free is not None and free > total:
            raise FrontierwrightError(
                "SERVING_RESOURCE_MANIFEST_INVALID",
                f"{free_key} cannot exceed cuda_memory_total_bytes.",
                2,
            )

    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    metrics: dict[str, object] = {
        "source_adapter": SERVING_RESOURCE_ADAPTER_ID,
        "source_adapter_version": SERVING_RESOURCE_ADAPTER_VERSION,
        "source_sha256": source_sha256,
        "runtime_id": runtime_id,
        "runtime_version": runtime_version,
        "measurement_scope": measurement_scope,
        "execution_boundary": execution_boundary.value,
        "profile_condition_hash": profile_condition_hash,
        **measured,
    }
    return ServingResourceImport(
        source_sha256=source_sha256,
        runtime_id=runtime_id,
        runtime_version=runtime_version,
        model_fingerprint=model_fingerprint,
        profile_condition_hash=profile_condition_hash,
        execution_boundary=execution_boundary,
        metrics=metrics,
    )


@dataclass(frozen=True)
class VLLMServingImport:
    source_sha256: str
    vllm_version: str
    profile_condition_hash: str
    source_model_id: str | None
    metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": VLLM_SERVE_ADAPTER_ID,
            "adapter_version": VLLM_SERVE_ADAPTER_VERSION,
            "source_sha256": self.source_sha256,
            "vllm_version": self.vllm_version,
            "profile_condition_hash": self.profile_condition_hash,
            "source_model_id": self.source_model_id,
            "metrics": dict(self.metrics),
        }


def _finite_number(payload: dict[str, Any], key: str) -> float | None:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise FrontierwrightError(
            "SERVING_BENCHMARK_INVALID",
            f"vLLM benchmark metric {key!r} must be finite.",
            2,
        )
    return value


def _milliseconds(payload: dict[str, Any], key: str) -> float | None:
    value = _finite_number(payload, key)
    return value / 1000.0 if value is not None else None


def _canonical_conditions(payload: dict[str, Any]) -> dict[str, object]:
    conditions: dict[str, object] = {}
    for key, value in sorted(payload.items()):
        if key in _VLLM_RESULT_FIELDS or key in _VLLM_MODEL_IDENTITY_FIELDS:
            continue
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            continue
        conditions[key] = value
    return conditions


def import_vllm_bench_serve(
    path: Path,
    *,
    vllm_version: str,
    execution_boundary: BackendDataBoundary,
) -> VLLMServingImport:
    if not vllm_version.strip():
        raise FrontierwrightError(
            "SERVING_BENCHMARK_VERSION_REQUIRED",
            "An exact vLLM version or immutable revision is required.",
            2,
        )
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "SERVING_BENCHMARK_INVALID",
            "vLLM benchmark result must be readable UTF-8 JSON.",
            2,
        ) from exc
    if not isinstance(raw, dict):
        raise FrontierwrightError(
            "SERVING_BENCHMARK_INVALID",
            "vLLM benchmark result root must be an object.",
            2,
        )

    conditions = _canonical_conditions(raw)
    condition_bytes = json.dumps(
        conditions,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    profile_condition_hash = "sha256:" + hashlib.sha256(condition_bytes).hexdigest()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    source_model = raw.get("model_id") or raw.get("model")
    source_model_id = source_model if isinstance(source_model, str) else None

    metrics: dict[str, object] = {
        "source_adapter": VLLM_SERVE_ADAPTER_ID,
        "source_adapter_version": VLLM_SERVE_ADAPTER_VERSION,
        "source_sha256": source_sha256,
        "vllm_version": vllm_version.strip(),
        "runtime_id": "vllm",
        "runtime_version": vllm_version.strip(),
        "measurement_scope": "vllm_bench_serve_client",
        "execution_boundary": execution_boundary.value,
        "profile_condition_hash": profile_condition_hash,
        "benchmark_conditions": conditions,
        "source_model_id": source_model_id,
        "latency_seconds_p50": _milliseconds(raw, "median_e2el_ms"),
        "ttft_seconds_p50": _milliseconds(raw, "median_ttft_ms"),
        "tpot_seconds_p50": _milliseconds(raw, "median_tpot_ms"),
        "itl_seconds_p50": _milliseconds(raw, "median_itl_ms"),
        "request_throughput_per_second": _finite_number(raw, "request_throughput"),
        "output_tokens_per_second_aggregate": _finite_number(raw, "output_throughput"),
        "total_tokens_per_second_aggregate": _finite_number(raw, "total_token_throughput"),
        "completed_requests": _finite_number(raw, "completed"),
        "failed_requests": _finite_number(raw, "failed"),
        # Deliberately not populated: client-side vLLM serve results do not establish
        # process RSS or GPU peak allocation for the serving process.
        "max_sampled_process_rss_bytes": None,
        "peak_vram_bytes": None,
        # Deliberately not mapped to tokens_per_second_p50: aggregate output throughput
        # under concurrency is not a per-request median generation rate.
        "tokens_per_second_p50": None,
    }
    if metrics["latency_seconds_p50"] is None:
        raise FrontierwrightError(
            "SERVING_BENCHMARK_METRIC_MISSING",
            "vLLM bench serve result must include median_e2el_ms.",
            2,
        )
    return VLLMServingImport(
        source_sha256=source_sha256,
        vllm_version=vllm_version.strip(),
        profile_condition_hash=profile_condition_hash,
        source_model_id=source_model_id,
        metrics=metrics,
    )
