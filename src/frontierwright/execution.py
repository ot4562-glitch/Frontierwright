"""Structured training backend, calibration, plan, and run contracts.

External trainers are invoked with argv, never shell interpolation. The backend
receives one JSON request file and must emit one JSON result object on stdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, BinaryIO

from frontierwright.errors import FrontierwrightError
from frontierwright.paths import TrainingPathId


class PermissionLevel(IntEnum):
    INSPECT_ONLY = 0
    PLAN = 1
    DRY_RUN = 2
    EXECUTE_SINGLE = 3
    EXECUTE_BOUNDED = 4
    RECURSIVE_EXECUTE = 5


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class HardBudgets:
    max_wall_seconds: float | None = None
    max_gpu_hours: float | None = None
    max_runs: int | None = None
    max_storage_bytes: int | None = None
    max_money: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_wall_seconds", "max_gpu_hours", "max_money"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("max_runs", "max_storage_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CommandBackendSpec:
    backend_id: str
    supported_paths: tuple[TrainingPathId, ...]
    calibrate_argv: tuple[str, ...]
    train_argv: tuple[str, ...]
    environment: dict[str, str]

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id must be nonempty")
        if not self.supported_paths:
            raise ValueError("supported_paths must be nonempty")
        if not self.calibrate_argv:
            raise ValueError("calibrate_argv must be nonempty")
        if not self.train_argv:
            raise ValueError("train_argv must be nonempty")
        for argv in (self.calibrate_argv, self.train_argv):
            if any(not item or "\x00" in item for item in argv):
                raise ValueError("backend argv entries must be nonempty and NUL-free")
        for key, value in self.environment.items():
            if not key or "\x00" in key or "\x00" in value:
                raise ValueError("backend environment entries must be NUL-free")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "supported_paths": [item.value for item in self.supported_paths],
            "calibrate_argv": list(self.calibrate_argv),
            "train_argv": list(self.train_argv),
            "environment": self.environment,
        }

    @property
    def sha256(self) -> str:
        data = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(data).hexdigest()}"


@dataclass(frozen=True)
class TrainingPlan:
    plan_id: str
    path_id: TrainingPathId
    backend_id: str
    backend_spec_hash: str
    model_id: str | None
    model_fingerprint: str | None
    model_source_path: str | None
    dataset_id: str
    dataset_fingerprint: str
    dataset_source_path: str
    resource_profile_id: str | None
    permission: PermissionLevel
    budgets: HardBudgets
    config: dict[str, object]
    idempotency_key: str

    def request_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "path_id": self.path_id.value,
            "backend_id": self.backend_id,
            "backend_spec_hash": self.backend_spec_hash,
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "model_source_path": self.model_source_path,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_source_path": self.dataset_source_path,
            "resource_profile_id": self.resource_profile_id,
            "permission": self.permission.name,
            "budgets": self.budgets.to_dict(),
            "config": self.config,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class CalibrationReceipt:
    calibration_id: str
    plan_id: str
    backend_id: str
    feasible: bool
    representative_steps: int
    step_time_seconds: float
    tokens_per_second: float | None
    peak_vram_bytes: int | None
    peak_ram_bytes: int | None
    projected_storage_bytes: int | None
    projected_wall_seconds: float | None
    backend_result: dict[str, object]

    def __post_init__(self) -> None:
        if self.representative_steps <= 0:
            raise ValueError("representative_steps must be positive")
        if (
            isinstance(self.step_time_seconds, bool)
            or not math.isfinite(self.step_time_seconds)
            or self.step_time_seconds <= 0
        ):
            raise ValueError("step_time_seconds must be positive and finite")
        if self.tokens_per_second is not None and (
            isinstance(self.tokens_per_second, bool)
            or not math.isfinite(self.tokens_per_second)
            or self.tokens_per_second <= 0
        ):
            raise ValueError("tokens_per_second must be positive and finite")
        for name in ("peak_vram_bytes", "peak_ram_bytes", "projected_storage_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.projected_wall_seconds is not None and (
            isinstance(self.projected_wall_seconds, bool)
            or not math.isfinite(self.projected_wall_seconds)
            or self.projected_wall_seconds <= 0
        ):
            raise ValueError("projected_wall_seconds must be positive and finite")


@dataclass(frozen=True)
class TrainingBackendResult:
    output_model_path: Path
    metrics: dict[str, object]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "INVALID_BACKEND_SPEC",
            f"{label} must be valid UTF-8 JSON.",
            2,
        ) from exc
    if not isinstance(raw, dict):
        raise FrontierwrightError("INVALID_BACKEND_SPEC", f"{label} must be an object.", 2)
    return raw


def load_command_backend_spec(path: Path) -> CommandBackendSpec:
    raw = _load_json(path, "Backend spec")
    if raw.get("schema_version") != 1:
        raise FrontierwrightError(
            "INVALID_BACKEND_SPEC",
            "Backend spec schema_version must be 1.",
            2,
        )
    try:
        supported_raw = raw["supported_paths"]
        calibrate_raw = raw["calibrate_argv"]
        train_raw = raw["train_argv"]
        environment_raw = raw.get("environment", {})
        if not isinstance(supported_raw, list):
            raise ValueError("supported_paths must be a list")
        if not isinstance(calibrate_raw, list) or not all(
            isinstance(item, str) for item in calibrate_raw
        ):
            raise ValueError("calibrate_argv must be a string list")
        if not isinstance(train_raw, list) or not all(
            isinstance(item, str) for item in train_raw
        ):
            raise ValueError("train_argv must be a string list")
        if not isinstance(environment_raw, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment_raw.items()
        ):
            raise ValueError("environment must map strings to strings")
        return CommandBackendSpec(
            backend_id=str(raw["backend_id"]),
            supported_paths=tuple(TrainingPathId(str(item)) for item in supported_raw),
            calibrate_argv=tuple(calibrate_raw),
            train_argv=tuple(train_raw),
            environment=dict(environment_raw),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "INVALID_BACKEND_SPEC",
            f"Invalid backend spec: {exc}",
            2,
        ) from exc


def compute_execution_request_digest(
    plan: TrainingPlan,
    *,
    calibration_id: str,
) -> str:
    if not calibration_id:
        raise ValueError("calibration_id must be nonempty")
    payload = {
        "plan": plan.request_payload(),
        "calibration_id": calibration_id,
    }
    data = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def compute_plan_idempotency_key(
    *,
    path_id: TrainingPathId,
    backend_id: str,
    backend_spec_hash: str,
    model_fingerprint: str | None,
    dataset_fingerprint: str,
    resource_profile_id: str | None,
    permission: PermissionLevel,
    budgets: HardBudgets,
    config: dict[str, object],
) -> str:
    payload = {
        "path_id": path_id.value,
        "backend_id": backend_id,
        "backend_spec_hash": backend_spec_hash,
        "model_fingerprint": model_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "resource_profile_id": resource_profile_id,
        "permission": permission.name,
        "budgets": budgets.to_dict(),
        "config": config,
    }
    data = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _materialize_request(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _substitute_argv(argv: tuple[str, ...], *, request_json: Path) -> list[str]:
    replacements = {"{request_json}": str(request_json)}
    result: list[str] = []
    for item in argv:
        rendered = item
        for token, value in replacements.items():
            rendered = rendered.replace(token, value)
        if "{" in rendered or "}" in rendered:
            raise FrontierwrightError(
                "BACKEND_PLACEHOLDER_UNKNOWN",
                f"Unsupported backend argv placeholder in: {item}",
                2,
            )
        result.append(rendered)
    return result


MAX_BACKEND_OUTPUT_BYTES = 1024 * 1024


def _read_bounded_output(handle: BinaryIO, label: str) -> str:
    handle.seek(0)
    raw = handle.read(MAX_BACKEND_OUTPUT_BYTES + 1)
    if len(raw) > MAX_BACKEND_OUTPUT_BYTES:
        raise FrontierwrightError(
            "BACKEND_OUTPUT_TOO_LARGE",
            f"Backend {label} exceeded {MAX_BACKEND_OUTPUT_BYTES} bytes.",
            14,
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrontierwrightError(
            "BACKEND_INVALID_OUTPUT",
            f"Backend {label} must be UTF-8 text.",
            14,
        ) from exc


def _run_backend(
    spec: CommandBackendSpec,
    argv_template: tuple[str, ...],
    *,
    request_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    argv = _substitute_argv(argv_template, request_json=request_path)
    environment = os.environ.copy()
    environment.update(spec.environment)

    try:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                argv,
                shell=False,
                stdout=stdout_file,
                stderr=stderr_file,
                env=environment,
            )
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise FrontierwrightError(
                    "BACKEND_TIMEOUT",
                    f"Backend exceeded {timeout_seconds:g} seconds.",
                    14,
                ) from exc

            stdout = _read_bounded_output(stdout_file, "stdout")
            stderr = _read_bounded_output(stderr_file, "stderr")
    except OSError as exc:
        raise FrontierwrightError(
            "BACKEND_LAUNCH_FAILED",
            f"Could not launch backend executable: {argv[0]}",
            14,
        ) from exc

    if return_code != 0:
        message = stderr.strip() or stdout.strip() or "backend failed"
        raise FrontierwrightError(
            "BACKEND_FAILED",
            f"Backend exited with code {return_code}: {message[:500]}",
            14,
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FrontierwrightError(
            "BACKEND_INVALID_RESULT",
            "Backend stdout must contain exactly one JSON result object.",
            14,
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise FrontierwrightError(
            "BACKEND_INVALID_RESULT",
            "Backend result schema_version must be 1.",
            14,
        )
    if payload.get("ok") is not True:
        raise FrontierwrightError(
            "BACKEND_FAILED",
            f"Backend reported failure: {payload.get('error', 'unknown error')}",
            14,
        )
    return payload


def run_calibration_backend(
    spec: CommandBackendSpec,
    plan: TrainingPlan,
    *,
    request_path: Path,
    calibration_id: str,
    timeout_seconds: float,
) -> CalibrationReceipt:
    if plan.path_id not in spec.supported_paths:
        raise FrontierwrightError(
            "BACKEND_PATH_UNSUPPORTED",
            f"Backend {spec.backend_id} does not support {plan.path_id.value}.",
            12,
        )
    payload = {
        **plan.request_payload(),
        "operation": "calibrate",
        "calibration_id": calibration_id,
    }
    _materialize_request(request_path, payload)
    result = _run_backend(
        spec,
        spec.calibrate_argv,
        request_path=request_path,
        timeout_seconds=timeout_seconds,
    )
    try:
        return CalibrationReceipt(
            calibration_id=calibration_id,
            plan_id=plan.plan_id,
            backend_id=spec.backend_id,
            feasible=result["feasible"] is True,
            representative_steps=int(result["representative_steps"]),
            step_time_seconds=float(result["step_time_seconds"]),
            tokens_per_second=(
                float(result["tokens_per_second"])
                if result.get("tokens_per_second") is not None
                else None
            ),
            peak_vram_bytes=(
                int(result["peak_vram_bytes"])
                if result.get("peak_vram_bytes") is not None
                else None
            ),
            peak_ram_bytes=(
                int(result["peak_ram_bytes"])
                if result.get("peak_ram_bytes") is not None
                else None
            ),
            projected_storage_bytes=(
                int(result["projected_storage_bytes"])
                if result.get("projected_storage_bytes") is not None
                else None
            ),
            projected_wall_seconds=(
                float(result["projected_wall_seconds"])
                if result.get("projected_wall_seconds") is not None
                else None
            ),
            backend_result=dict(result),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "BACKEND_INVALID_RESULT",
            f"Invalid calibration result: {exc}",
            14,
        ) from exc


def run_training_backend(
    spec: CommandBackendSpec,
    plan: TrainingPlan,
    *,
    request_path: Path,
    output_root: Path,
    run_id: str,
    timeout_seconds: float,
) -> TrainingBackendResult:
    if plan.path_id not in spec.supported_paths:
        raise FrontierwrightError(
            "BACKEND_PATH_UNSUPPORTED",
            f"Backend {spec.backend_id} does not support {plan.path_id.value}.",
            12,
        )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **plan.request_payload(),
        "operation": "train",
        "run_id": run_id,
        "output_root": str(output_root),
    }
    _materialize_request(request_path, payload)
    result = _run_backend(
        spec,
        spec.train_argv,
        request_path=request_path,
        timeout_seconds=timeout_seconds,
    )
    output = result.get("output_model_path")
    metrics = result.get("metrics", {})
    if not isinstance(output, str) or not output.strip():
        raise FrontierwrightError(
            "BACKEND_INVALID_RESULT",
            "Training result must include output_model_path.",
            14,
        )
    if not isinstance(metrics, dict):
        raise FrontierwrightError(
            "BACKEND_INVALID_RESULT",
            "Training result metrics must be an object.",
            14,
        )
    output_path = Path(output).expanduser().resolve()
    try:
        output_path.relative_to(output_root)
    except ValueError as exc:
        raise FrontierwrightError(
            "BACKEND_OUTPUT_ESCAPE",
            "Backend output_model_path must stay inside the assigned output_root.",
            14,
        ) from exc
    if not output_path.exists():
        raise FrontierwrightError(
            "BACKEND_OUTPUT_MISSING",
            "Backend output_model_path does not exist.",
            14,
        )
    return TrainingBackendResult(
        output_model_path=output_path,
        metrics=dict(metrics),
    )
