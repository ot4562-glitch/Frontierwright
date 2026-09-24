"""Structured training backend, calibration, plan, and run contracts.

External trainers are invoked with argv, never shell interpolation. The backend
receives one JSON request file and must emit one JSON result object on stdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, BinaryIO

from frontierwright.data import DatasetClassification
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


class BackendDataBoundary(StrEnum):
    LOCAL_MACHINE = "LOCAL_MACHINE"
    CONTROLLED_PRIVATE = "CONTROLLED_PRIVATE"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"


def backend_allows_dataset(
    boundary: BackendDataBoundary,
    classification: DatasetClassification,
) -> bool:
    if boundary in {
        BackendDataBoundary.LOCAL_MACHINE,
        BackendDataBoundary.CONTROLLED_PRIVATE,
    }:
        return True
    return classification is DatasetClassification.PUBLIC


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
    data_boundary: BackendDataBoundary = BackendDataBoundary.LOCAL_MACHINE
    data_boundary_explicit: bool = True
    provider_adapter_ref: str | None = None

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
        if self.provider_adapter_ref is not None and (
            not self.provider_adapter_ref.strip()
            or "\x00" in self.provider_adapter_ref
        ):
            raise ValueError("provider_adapter_ref must be nonempty and NUL-free")

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend_id": self.backend_id,
            "supported_paths": [item.value for item in self.supported_paths],
            "calibrate_argv": list(self.calibrate_argv),
            "train_argv": list(self.train_argv),
            "environment": self.environment,
        }
        if self.data_boundary_explicit:
            payload["data_boundary"] = self.data_boundary.value
        if self.provider_adapter_ref is not None:
            payload["provider_adapter_ref"] = self.provider_adapter_ref
        return payload

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
    intervention_id: str
    intervention_version: str
    intervention_family: str
    backend_id: str
    backend_spec_hash: str
    model_id: str | None
    model_fingerprint: str | None
    model_source_path: str | None
    dataset_id: str
    dataset_fingerprint: str
    dataset_source_path: str
    dataset_recipe_id: str | None
    dataset_recipe_hash: str | None
    resource_profile_id: str | None
    permission: PermissionLevel
    budgets: HardBudgets
    config: dict[str, object]
    idempotency_key: str
    dataset_classification: str = DatasetClassification.PRIVATE.value
    backend_data_boundary: str = BackendDataBoundary.LOCAL_MACHINE.value
    backend_adapter_ref: str | None = None
    backend_adapter_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "intervention_id",
            "intervention_version",
            "intervention_family",
        ):
            value = getattr(self, name)
            if not value.strip():
                raise ValueError(f"{name} must be nonempty")
        DatasetClassification(self.dataset_classification)
        BackendDataBoundary(self.backend_data_boundary)
        if self.backend_adapter_ref is not None and (
            not self.backend_adapter_ref.strip() or "\x00" in self.backend_adapter_ref
        ):
            raise ValueError("backend_adapter_ref must be nonempty and NUL-free")
        if self.backend_adapter_hash is not None and (
            not self.backend_adapter_hash.startswith("sha256:")
            or len(self.backend_adapter_hash) != 71
        ):
            raise ValueError("backend_adapter_hash must be a sha256: digest")
        if (self.backend_adapter_ref is None) != (self.backend_adapter_hash is None):
            raise ValueError(
                "backend_adapter_ref and backend_adapter_hash must be supplied together"
            )

    def request_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "path_id": self.path_id.value,
            "intervention_id": self.intervention_id,
            "intervention_version": self.intervention_version,
            "intervention_family": self.intervention_family,
            "backend_id": self.backend_id,
            "backend_spec_hash": self.backend_spec_hash,
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "model_source_path": self.model_source_path,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_source_path": self.dataset_source_path,
            "dataset_recipe_id": self.dataset_recipe_id,
            "dataset_recipe_hash": self.dataset_recipe_hash,
            "dataset_classification": self.dataset_classification,
            "backend_data_boundary": self.backend_data_boundary,
            "backend_adapter_ref": self.backend_adapter_ref,
            "backend_adapter_hash": self.backend_adapter_hash,
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


@dataclass(frozen=True)
class RunUsage:
    """Measured execution usage. Unknown dimensions stay explicitly unknown."""

    wall_seconds: float
    output_storage_bytes: int | None = None
    gpu_count: int | None = None
    gpu_count_provenance: str | None = None
    accounted_gpu_hours: float | None = None
    money_spent: float | None = None
    measured_by: str = "frontierwright-local-executor-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.wall_seconds, bool)
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds < 0
        ):
            raise ValueError("wall_seconds must be finite and nonnegative")
        if self.output_storage_bytes is not None and (
            isinstance(self.output_storage_bytes, bool)
            or not isinstance(self.output_storage_bytes, int)
            or self.output_storage_bytes < 0
        ):
            raise ValueError("output_storage_bytes must be a nonnegative integer")
        if self.gpu_count is not None and (
            isinstance(self.gpu_count, bool)
            or not isinstance(self.gpu_count, int)
            or self.gpu_count <= 0
        ):
            raise ValueError("gpu_count must be a positive integer")
        if (self.gpu_count is None) != (self.gpu_count_provenance is None):
            raise ValueError(
                "gpu_count and gpu_count_provenance must be supplied together"
            )
        if self.gpu_count_provenance is not None and not self.gpu_count_provenance.strip():
            raise ValueError("gpu_count_provenance must be nonempty")
        if self.accounted_gpu_hours is not None and (
            isinstance(self.accounted_gpu_hours, bool)
            or not math.isfinite(self.accounted_gpu_hours)
            or self.accounted_gpu_hours < 0
        ):
            raise ValueError("accounted_gpu_hours must be finite and nonnegative")
        if self.accounted_gpu_hours is not None and self.gpu_count is None:
            raise ValueError("accounted_gpu_hours requires gpu_count")
        if self.money_spent is not None and (
            isinstance(self.money_spent, bool)
            or not math.isfinite(self.money_spent)
            or self.money_spent < 0
        ):
            raise ValueError("money_spent must be finite and nonnegative")
        if not self.measured_by.strip():
            raise ValueError("measured_by must be nonempty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
        data_boundary_raw = raw.get(
            "data_boundary",
            BackendDataBoundary.LOCAL_MACHINE.value,
        )
        provider_adapter_ref_raw = raw.get("provider_adapter_ref")
        if provider_adapter_ref_raw is not None and not isinstance(
            provider_adapter_ref_raw,
            str,
        ):
            raise ValueError("provider_adapter_ref must be a string when supplied")
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
            data_boundary=BackendDataBoundary(str(data_boundary_raw)),
            data_boundary_explicit="data_boundary" in raw,
            provider_adapter_ref=provider_adapter_ref_raw,
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
    intervention_id: str,
    intervention_version: str,
    backend_id: str,
    backend_spec_hash: str,
    model_fingerprint: str | None,
    dataset_fingerprint: str,
    dataset_recipe_hash: str | None,
    resource_profile_id: str | None,
    permission: PermissionLevel,
    budgets: HardBudgets,
    config: dict[str, object],
    dataset_classification: str = DatasetClassification.PRIVATE.value,
    backend_data_boundary: str = BackendDataBoundary.LOCAL_MACHINE.value,
    backend_adapter_ref: str | None = None,
    backend_adapter_hash: str | None = None,
) -> str:
    DatasetClassification(dataset_classification)
    BackendDataBoundary(backend_data_boundary)
    if (backend_adapter_ref is None) != (backend_adapter_hash is None):
        raise ValueError(
            "backend_adapter_ref and backend_adapter_hash must be supplied together"
        )
    if backend_adapter_hash is not None and (
        not backend_adapter_hash.startswith("sha256:")
        or len(backend_adapter_hash) != 71
    ):
        raise ValueError("backend_adapter_hash must be a sha256: digest")
    payload = {
        "path_id": path_id.value,
        "intervention_id": intervention_id,
        "intervention_version": intervention_version,
        "backend_id": backend_id,
        "backend_spec_hash": backend_spec_hash,
        "model_fingerprint": model_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_recipe_hash": dataset_recipe_hash,
        "dataset_classification": dataset_classification,
        "backend_data_boundary": backend_data_boundary,
        "backend_adapter_ref": backend_adapter_ref,
        "backend_adapter_hash": backend_adapter_hash,
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
BACKEND_WATCH_INTERVAL_SECONDS = 0.1


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


def _output_tree_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        dirnames[:] = [
            name for name in dirnames if not (base / name).is_symlink()
        ]
        for name in filenames:
            path = base / name
            if path.is_symlink():
                continue
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _terminate_backend_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "nt":
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    process.wait(timeout=5)


def run_structured_command(
    argv_template: tuple[str, ...],
    *,
    environment_overrides: dict[str, str],
    request_path: Path,
    timeout_seconds: float,
    output_watch_root: Path | None = None,
    max_output_bytes: int | None = None,
) -> dict[str, Any]:
    """Run one structured JSON request/result command without shell interpolation."""

    if (output_watch_root is None) != (max_output_bytes is None):
        raise ValueError(
            "output_watch_root and max_output_bytes must be supplied together"
        )
    if max_output_bytes is not None and (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes < 0
    ):
        raise ValueError("max_output_bytes must be a nonnegative integer")

    argv = _substitute_argv(argv_template, request_json=request_path)
    environment = os.environ.copy()
    environment.update(environment_overrides)
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    else:
        popen_kwargs["start_new_session"] = True

    try:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                argv,
                shell=False,
                stdout=stdout_file,
                stderr=stderr_file,
                env=environment,
                **popen_kwargs,
            )
            started = time.monotonic()
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break

                if output_watch_root is not None and max_output_bytes is not None:
                    observed_bytes = _output_tree_size_bytes(output_watch_root)
                    if observed_bytes > max_output_bytes:
                        _terminate_backend_tree(process)
                        raise FrontierwrightError(
                            "STORAGE_BUDGET_REACHED",
                            (
                                "Backend output exceeded the runtime storage budget: "
                                f"{observed_bytes} > {max_output_bytes} bytes."
                            ),
                            14,
                        )

                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    _terminate_backend_tree(process)
                    raise FrontierwrightError(
                        "BACKEND_TIMEOUT",
                        f"Backend exceeded {timeout_seconds:g} seconds.",
                        14,
                    )
                time.sleep(
                    min(
                        BACKEND_WATCH_INTERVAL_SECONDS,
                        max(0.01, timeout_seconds - elapsed),
                    )
                )

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

def _run_backend(
    spec: CommandBackendSpec,
    argv_template: tuple[str, ...],
    *,
    request_path: Path,
    timeout_seconds: float,
    output_watch_root: Path | None = None,
    max_output_bytes: int | None = None,
) -> dict[str, Any]:
    return run_structured_command(
        argv_template,
        environment_overrides=spec.environment,
        request_path=request_path,
        timeout_seconds=timeout_seconds,
        output_watch_root=output_watch_root,
        max_output_bytes=max_output_bytes,
    )


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
        output_watch_root=(
            output_root if plan.budgets.max_storage_bytes is not None else None
        ),
        max_output_bytes=plan.budgets.max_storage_bytes,
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
