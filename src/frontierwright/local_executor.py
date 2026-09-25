"""Recoverable local execution supervisor for Frontierwright training attempts."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontierwright.errors import FrontierwrightError
from frontierwright.execution import RunUsage, load_command_backend_spec, run_training_backend
from frontierwright.registry import Registry

if sys.platform == "win32":

    def _kill_process_group(pid: int, *, force: bool = False) -> None:
        del pid, force
        raise RuntimeError("POSIX process groups are unavailable on Windows")

else:

    def _kill_process_group(pid: int, *, force: bool = False) -> None:
        os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)


WORKER_RESULT_SCHEMA = 1


@dataclass(frozen=True)
class LocalAttemptSpec:
    project_root: Path
    run_id: str
    plan_id: str
    calibration_id: str
    backend_spec_path: Path
    request_path: Path
    output_root: Path
    result_path: Path
    request_digest: str
    timeout_seconds: float
    gpu_count: int | None = None
    gpu_count_provenance: str | None = None

    def __post_init__(self) -> None:
        if (self.gpu_count is None) != (self.gpu_count_provenance is None):
            raise ValueError(
                "gpu_count and gpu_count_provenance must be supplied together"
            )
        if self.gpu_count is not None and (
            isinstance(self.gpu_count, bool)
            or not isinstance(self.gpu_count, int)
            or self.gpu_count <= 0
        ):
            raise ValueError("gpu_count must be a positive integer")
        if self.gpu_count_provenance is not None and not self.gpu_count_provenance.strip():
            raise ValueError("gpu_count_provenance must be nonempty")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "project_root": str(self.project_root),
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "calibration_id": self.calibration_id,
            "backend_spec_path": str(self.backend_spec_path),
            "request_path": str(self.request_path),
            "output_root": str(self.output_root),
            "result_path": str(self.result_path),
            "request_digest": self.request_digest,
            "timeout_seconds": self.timeout_seconds,
            "gpu_count": self.gpu_count,
            "gpu_count_provenance": self.gpu_count_provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LocalAttemptSpec:
        if payload.get("schema_version") != 1:
            raise ValueError("local attempt schema_version must be 1")
        return cls(
            project_root=Path(_require_str(payload, "project_root")),
            run_id=_require_str(payload, "run_id"),
            plan_id=_require_str(payload, "plan_id"),
            calibration_id=_require_str(payload, "calibration_id"),
            backend_spec_path=Path(_require_str(payload, "backend_spec_path")),
            request_path=Path(_require_str(payload, "request_path")),
            output_root=Path(_require_str(payload, "output_root")),
            result_path=Path(_require_str(payload, "result_path")),
            request_digest=_require_str(payload, "request_digest"),
            timeout_seconds=_require_positive_number(payload, "timeout_seconds"),
            gpu_count=_optional_positive_int(payload, "gpu_count"),
            gpu_count_provenance=_optional_str(payload, "gpu_count_provenance"),
        )


def _require_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _require_positive_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)


def _optional_positive_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer when supplied")
    return value


def _optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string when supplied")
    return value


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def process_start_token(pid: int) -> str | None:
    """Return an OS start token when it can be verified without optional packages."""

    if pid <= 0:
        return None
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            fields = proc_stat.read_text(encoding="utf-8").split()
        except OSError:
            return None
        if len(fields) > 21:
            return fields[21]
    return None


def process_liveness(pid: int | None, start_token: str | None) -> str:
    """Return LIVE, DEAD, or UNRESOLVED without guessing across PID reuse."""

    if pid is None or pid <= 0:
        return "UNRESOLVED"

    current_token = process_start_token(pid)
    if current_token is not None:
        if start_token is None:
            return "UNRESOLVED"
        return "LIVE" if current_token == start_token else "DEAD"

    if Path("/proc").is_dir():
        return "DEAD"

    # Portable existence checks cannot reliably disambiguate PID reuse.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "DEAD"
    except (PermissionError, OSError):
        return "UNRESOLVED"
    return "UNRESOLVED"


def launch_worker(attempt_file: Path) -> subprocess.Popen[bytes]:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "shell": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(
        [sys.executable, "-m", "frontierwright.local_executor", "--worker", str(attempt_file)],
        **kwargs,
    )


def terminate_worker_tree(process: subprocess.Popen[bytes]) -> None:
    """Conclude worker termination before the caller records a terminal timeout."""

    if process.poll() is not None:
        return

    if os.name != "nt":
        try:
            _kill_process_group(process.pid)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            try:
                _kill_process_group(process.pid, force=True)
            except ProcessLookupError:
                return
            process.wait(timeout=5)
            return

    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            shell=False,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FrontierwrightError(
            "EXECUTOR_TERMINATION_UNRESOLVED",
            "Could not confirm local worker process-tree termination.",
            14,
        ) from exc
    if result.returncode != 0 and process.poll() is None:
        raise FrontierwrightError(
            "EXECUTOR_TERMINATION_UNRESOLVED",
            "Could not confirm local worker process-tree termination.",
            14,
        )
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise FrontierwrightError(
            "EXECUTOR_TERMINATION_UNRESOLVED",
            "Local worker remained live after termination request.",
            14,
        ) from exc


def load_attempt(path: Path) -> LocalAttemptSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("local attempt file must contain an object")
    return LocalAttemptSpec.from_dict(raw)


def _tree_size_bytes(root: Path) -> int:
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


def _usage_for_attempt(
    attempt: LocalAttemptSpec,
    *,
    started_monotonic: float,
) -> RunUsage:
    wall_seconds = max(0.0, time.monotonic() - started_monotonic)
    accounted_gpu_hours = (
        wall_seconds * attempt.gpu_count / 3600.0
        if attempt.gpu_count is not None
        else None
    )
    return RunUsage(
        wall_seconds=wall_seconds,
        output_storage_bytes=_tree_size_bytes(attempt.output_root),
        gpu_count=attempt.gpu_count,
        gpu_count_provenance=attempt.gpu_count_provenance,
        accounted_gpu_hours=accounted_gpu_hours,
    )


def _success_payload(
    attempt: LocalAttemptSpec,
    *,
    output_model_path: Path,
    metrics: dict[str, object],
    usage: RunUsage,
) -> dict[str, object]:
    return {
        "schema_version": WORKER_RESULT_SCHEMA,
        "ok": True,
        "run_id": attempt.run_id,
        "plan_id": attempt.plan_id,
        "calibration_id": attempt.calibration_id,
        "request_digest": attempt.request_digest,
        "output_model_path": str(output_model_path),
        "metrics": metrics,
        "usage": usage.to_dict(),
        "worker_pid": os.getpid(),
    }


def _failure_payload(
    attempt: LocalAttemptSpec,
    *,
    code: str,
    message: str,
    usage: RunUsage,
) -> dict[str, object]:
    return {
        "schema_version": WORKER_RESULT_SCHEMA,
        "ok": False,
        "run_id": attempt.run_id,
        "plan_id": attempt.plan_id,
        "calibration_id": attempt.calibration_id,
        "request_digest": attempt.request_digest,
        "error": {"code": code, "message": message[:2000]},
        "usage": usage.to_dict(),
        "worker_pid": os.getpid(),
    }


def run_worker(attempt_file: Path) -> int:
    attempt = load_attempt(attempt_file)
    started_monotonic = time.monotonic()
    try:
        registry = Registry(attempt.project_root)
        plan = registry.get_plan(attempt.plan_id)
        backend = load_command_backend_spec(attempt.backend_spec_path)
        if backend.backend_id != plan.backend_id or backend.sha256 != plan.backend_spec_hash:
            raise FrontierwrightError(
                "BACKEND_SPEC_DRIFT",
                "Backend spec no longer matches the pinned training plan.",
                13,
            )
        result = run_training_backend(
            backend,
            plan,
            request_path=attempt.request_path,
            output_root=attempt.output_root,
            run_id=attempt.run_id,
            timeout_seconds=attempt.timeout_seconds,
        )
        atomic_write_json(
            attempt.result_path,
            _success_payload(
                attempt,
                output_model_path=result.output_model_path,
                metrics=result.metrics,
                usage=_usage_for_attempt(
                    attempt,
                    started_monotonic=started_monotonic,
                ),
            ),
        )
        return 0
    except FrontierwrightError as exc:
        atomic_write_json(
            attempt.result_path,
            _failure_payload(
                attempt,
                code=exc.code,
                message=str(exc),
                usage=_usage_for_attempt(
                    attempt,
                    started_monotonic=started_monotonic,
                ),
            ),
        )
        return exc.exit_code if 0 < exc.exit_code < 256 else 14
    except Exception as exc:
        atomic_write_json(
            attempt.result_path,
            _failure_payload(
                attempt,
                code="EXECUTOR_WORKER_ERROR",
                message=f"{type(exc).__name__}: {exc}",
                usage=_usage_for_attempt(
                    attempt,
                    started_monotonic=started_monotonic,
                ),
            ),
        )
        return 14


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "--worker":
        return 2
    return run_worker(Path(args[1]))


if __name__ == "__main__":
    raise SystemExit(main())
