"""Concrete Slurm bridge for controlled-private Lab training backends.

The bridge is intentionally narrow:
- Frontierwright still owns plan/data/model/budget/lineage identity.
- A Slurm profile pins the cluster submission contract and the worker argv.
- The worker remains a normal structured Frontierwright backend that receives one JSON
  request and emits one JSON result object.
- The submit host and compute nodes must share the paths in the request.
- Secrets are never stored in the profile; authentication belongs to the cluster/session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

from frontierwright.errors import FrontierwrightError
from frontierwright.execution import BackendDataBoundary
from frontierwright.lab_adapters import (
    LabAdapterKind,
    NetworkScope,
    load_lab_adapter_manifest,
)
from frontierwright.paths import TrainingPathId

SLURM_PROFILE_SCHEMA_VERSION = 1
SLURM_BRIDGE_ID = "frontierwright.lab.slurm"
SLURM_BRIDGE_VERSION = "1"

_SECRET_KEYS = {
    "token",
    "password",
    "secret",
    "api_key",
    "apikey",
    "credential",
    "credentials",
}


def _forbid_secret_keys(value: object, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            if key.casefold() in _SECRET_KEYS:
                raise FrontierwrightError(
                    "SLURM_PROFILE_SECRET_FORBIDDEN",
                    "Slurm profiles must not contain credentials or secrets.",
                    2,
                )
            _forbid_secret_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _forbid_secret_keys(item, path=f"{path}[{index}]")


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item and "\x00" not in item for item in value)
    ):
        raise ValueError(f"{label} must be a nonempty NUL-free string list")
    return tuple(value)


def _optional_label(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty NUL-free string when supplied")
    return value.strip()


def _positive_int(value: object, label: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_worker_argv(argv: tuple[str, ...], label: str) -> None:
    if not any("{request_json}" in item for item in argv):
        raise ValueError(f"{label} must reference {{request_json}}")
    for item in argv:
        rendered = item.replace("{request_json}", "")
        if "{" in rendered or "}" in rendered:
            raise ValueError(f"{label} contains an unsupported placeholder: {item}")


@dataclass(frozen=True)
class SlurmExecutorProfile:
    adapter_id: str
    adapter_version: str
    display_name: str
    supported_paths: tuple[TrainingPathId, ...]
    calibrate_worker_argv: tuple[str, ...]
    train_worker_argv: tuple[str, ...]
    sbatch_argv: tuple[str, ...] = ("sbatch",)
    scancel_argv: tuple[str, ...] = ("scancel",)
    sacct_argv: tuple[str, ...] = ("sacct",)
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodes: int = 1
    gpus_per_node: int | None = None
    cpus_per_task: int = 1
    memory_mb: int | None = None
    time_limit_minutes: int = 60
    max_queue_wait_seconds: int = 3600
    shared_filesystem: bool = True
    network_scope: NetworkScope = NetworkScope.PRIVATE_ONLY
    capabilities: tuple[str, ...] = ()
    schema_version: int = SLURM_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SLURM_PROFILE_SCHEMA_VERSION:
            raise ValueError("Slurm profile schema_version must be 1")
        if not self.adapter_id.strip() or not self.adapter_version.strip():
            raise ValueError("Slurm adapter identity must be nonempty")
        if not self.supported_paths:
            raise ValueError("supported_paths must be nonempty")
        _validate_worker_argv(self.calibrate_worker_argv, "calibrate_worker_argv")
        _validate_worker_argv(self.train_worker_argv, "train_worker_argv")
        if not self.sbatch_argv or not self.scancel_argv:
            raise ValueError("sbatch/scancel argv must be nonempty")
        if self.nodes <= 0 or self.cpus_per_task <= 0 or self.time_limit_minutes <= 0:
            raise ValueError("Slurm resource counts/time limit must be positive")
        if self.gpus_per_node is not None and self.gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive when supplied")
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError("memory_mb must be positive when supplied")
        if self.max_queue_wait_seconds <= 0:
            raise ValueError("max_queue_wait_seconds must be positive")
        if not self.shared_filesystem:
            raise ValueError(
                "Slurm bridge v1 requires a shared filesystem for model/data/request/output paths"
            )
        if self.network_scope is NetworkScope.EXTERNAL_ALLOWED:
            raise ValueError(
                "Slurm Lab executor must be OFFLINE or PRIVATE_ONLY; external data egress "
                "requires a different explicitly external backend boundary"
            )

    @property
    def adapter_ref(self) -> str:
        return f"{self.adapter_id}@{self.adapter_version}"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "display_name": self.display_name,
            "supported_paths": [item.value for item in self.supported_paths],
            "calibrate_worker_argv": list(self.calibrate_worker_argv),
            "train_worker_argv": list(self.train_worker_argv),
            "sbatch_argv": list(self.sbatch_argv),
            "scancel_argv": list(self.scancel_argv),
            "sacct_argv": list(self.sacct_argv),
            "partition": self.partition,
            "account": self.account,
            "qos": self.qos,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "cpus_per_task": self.cpus_per_task,
            "memory_mb": self.memory_mb,
            "time_limit_minutes": self.time_limit_minutes,
            "max_queue_wait_seconds": self.max_queue_wait_seconds,
            "shared_filesystem": self.shared_filesystem,
            "network_scope": self.network_scope.value,
            "capabilities": sorted(self.capabilities),
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def slurm_profile_schema() -> dict[str, object]:
    from frontierwright.lab_adapters import lab_adapter_manifest_schema

    base = lab_adapter_manifest_schema()
    raw_properties = base.get("properties")
    properties: dict[str, object] = (
        dict(raw_properties) if isinstance(raw_properties, dict) else {}
    )
    properties["slurm"] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "supported_paths",
            "calibrate_worker_argv",
            "train_worker_argv",
            "shared_filesystem",
        ],
        "properties": {
            "schema_version": {"const": 1},
            "supported_paths": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"enum": [item.value for item in TrainingPathId]},
            },
            "calibrate_worker_argv": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
            "train_worker_argv": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
            "sbatch_argv": {"type": "array", "items": {"type": "string"}},
            "scancel_argv": {"type": "array", "items": {"type": "string"}},
            "sacct_argv": {"type": "array", "items": {"type": "string"}},
            "partition": {"type": ["string", "null"]},
            "account": {"type": ["string", "null"]},
            "qos": {"type": ["string", "null"]},
            "nodes": {"type": "integer", "minimum": 1},
            "gpus_per_node": {"type": ["integer", "null"], "minimum": 1},
            "cpus_per_task": {"type": "integer", "minimum": 1},
            "memory_mb": {"type": ["integer", "null"], "minimum": 1},
            "time_limit_minutes": {"type": "integer", "minimum": 1},
            "max_queue_wait_seconds": {"type": "integer", "minimum": 1},
            "shared_filesystem": {"const": True},
        },
    }
    raw_required = base.get("required")
    required = list(raw_required) if isinstance(raw_required, list) else []
    return {
        **base,
        "title": "Frontierwright Controlled-Private Slurm Profile",
        "required": [*required, "slurm"],
        "properties": properties,
    }


def slurm_profile_example() -> dict[str, object]:
    return {
        "schema_version": 1,
        "adapter_id": "lab.slurm.private",
        "adapter_version": "1",
        "display_name": "Private Slurm Cluster",
        "kinds": ["TRAINER", "CLUSTER_EXECUTOR"],
        "data_boundary": "CONTROLLED_PRIVATE",
        "network_scope": "PRIVATE_ONLY",
        "capabilities": ["slurm", "lora-sft", "rl-policy-optimization"],
        "slurm": {
            "schema_version": 1,
            "supported_paths": ["LORA_SFT", "RL_POLICY_OPTIMIZATION"],
            "calibrate_worker_argv": [
                "python",
                "worker.py",
                "--request",
                "{request_json}",
            ],
            "train_worker_argv": [
                "python",
                "worker.py",
                "--request",
                "{request_json}",
            ],
            "sbatch_argv": ["sbatch"],
            "scancel_argv": ["scancel"],
            "sacct_argv": ["sacct"],
            "partition": "private",
            "nodes": 1,
            "gpus_per_node": 1,
            "cpus_per_task": 8,
            "memory_mb": 32768,
            "time_limit_minutes": 60,
            "max_queue_wait_seconds": 3600,
            "shared_filesystem": True,
        },
    }


def load_slurm_executor_profile(path: Path) -> SlurmExecutorProfile:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "SLURM_PROFILE_INVALID",
            f"Slurm executor profile is unreadable: {path}",
            2,
        ) from exc
    if not isinstance(raw, dict):
        raise FrontierwrightError(
            "SLURM_PROFILE_INVALID", "Slurm executor profile must be a JSON object.", 2
        )
    _forbid_secret_keys(raw)

    try:
        base = load_lab_adapter_manifest(path)
        if LabAdapterKind.CLUSTER_EXECUTOR not in base.kinds:
            raise ValueError("Lab adapter kinds must include CLUSTER_EXECUTOR")
        if base.data_boundary is not BackendDataBoundary.CONTROLLED_PRIVATE:
            raise ValueError("Slurm executor data_boundary must be CONTROLLED_PRIVATE")
        if base.network_scope is NetworkScope.EXTERNAL_ALLOWED:
            raise ValueError("Slurm executor network_scope must be OFFLINE or PRIVATE_ONLY")
        slurm = raw.get("slurm")
        if not isinstance(slurm, dict):
            raise ValueError("slurm must be an object")
        if slurm.get("schema_version") != 1:
            raise ValueError("slurm.schema_version must be 1")
        supported = _string_list(slurm.get("supported_paths"), "slurm.supported_paths")
        profile = SlurmExecutorProfile(
            adapter_id=base.adapter_id,
            adapter_version=base.adapter_version,
            display_name=base.display_name,
            supported_paths=tuple(TrainingPathId(item) for item in supported),
            calibrate_worker_argv=_string_list(
                slurm.get("calibrate_worker_argv"),
                "slurm.calibrate_worker_argv",
            ),
            train_worker_argv=_string_list(
                slurm.get("train_worker_argv"),
                "slurm.train_worker_argv",
            ),
            sbatch_argv=_string_list(slurm.get("sbatch_argv", ["sbatch"]), "slurm.sbatch_argv"),
            scancel_argv=_string_list(slurm.get("scancel_argv", ["scancel"]), "slurm.scancel_argv"),
            sacct_argv=_string_list(slurm.get("sacct_argv", ["sacct"]), "slurm.sacct_argv"),
            partition=_optional_label(slurm.get("partition"), "slurm.partition"),
            account=_optional_label(slurm.get("account"), "slurm.account"),
            qos=_optional_label(slurm.get("qos"), "slurm.qos"),
            nodes=_positive_int(slurm.get("nodes"), "slurm.nodes", default=1),
            gpus_per_node=(
                _positive_int(slurm.get("gpus_per_node"), "slurm.gpus_per_node")
                if slurm.get("gpus_per_node") is not None
                else None
            ),
            cpus_per_task=_positive_int(
                slurm.get("cpus_per_task"), "slurm.cpus_per_task", default=1
            ),
            memory_mb=(
                _positive_int(slurm.get("memory_mb"), "slurm.memory_mb")
                if slurm.get("memory_mb") is not None
                else None
            ),
            time_limit_minutes=_positive_int(
                slurm.get("time_limit_minutes"),
                "slurm.time_limit_minutes",
                default=60,
            ),
            max_queue_wait_seconds=_positive_int(
                slurm.get("max_queue_wait_seconds"),
                "slurm.max_queue_wait_seconds",
                default=3600,
            ),
            shared_filesystem=slurm.get("shared_filesystem") is True,
            network_scope=base.network_scope,
            capabilities=base.capabilities,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "SLURM_PROFILE_INVALID", f"Invalid Slurm executor profile: {exc}", 2
        ) from exc
    return profile


def slurm_backend_spec_payload(
    profile_path: Path,
    profile: SlurmExecutorProfile,
    *,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    resolved = profile_path.resolve()
    common = [
        python_executable,
        "-m",
        "frontierwright.slurm_executor",
        "--profile",
        str(resolved),
        "--expected-profile-hash",
        profile.sha256,
        "--request",
        "{request_json}",
    ]
    return {
        "schema_version": 1,
        "backend_id": f"{SLURM_BRIDGE_ID}.{profile.adapter_id}",
        "supported_paths": [item.value for item in profile.supported_paths],
        "calibrate_argv": common,
        "train_argv": common,
        "environment": {},
        "data_boundary": BackendDataBoundary.CONTROLLED_PRIVATE.value,
        "provider_adapter_ref": profile.adapter_ref,
        "slurm_profile_hash": profile.sha256,
    }


def inspect_slurm_availability(profile: SlurmExecutorProfile) -> dict[str, object]:
    def command_status(argv: tuple[str, ...]) -> dict[str, object]:
        executable = argv[0]
        if os.path.isabs(executable):
            resolved = executable if os.path.isfile(executable) else None
        else:
            resolved = shutil.which(executable)
        return {
            "configured": list(argv),
            "resolved_executable": resolved,
            "available": resolved is not None,
        }

    submit = command_status(profile.sbatch_argv)
    cancel = command_status(profile.scancel_argv)
    accounting = command_status(profile.sacct_argv)
    return {
        "schema_version": 1,
        "profile_hash": profile.sha256,
        "adapter_ref": profile.adapter_ref,
        "platform": sys.platform,
        "supported_platform": os.name != "nt",
        "shared_filesystem": profile.shared_filesystem,
        "submit": submit,
        "cancel": cancel,
        "accounting": accounting,
        "ready": (
            os.name != "nt"
            and submit["available"] is True
            and cancel["available"] is True
            and profile.shared_filesystem
        ),
        "accounting_optional": True,
    }


def _load_request(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "SLURM_REQUEST_INVALID", "Slurm request must be readable UTF-8 JSON.", 14
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise FrontierwrightError(
            "SLURM_REQUEST_INVALID", "Slurm request schema_version must be 1.", 14
        )
    return raw


def _render_worker_argv(template: tuple[str, ...], *, request_path: Path) -> list[str]:
    result: list[str] = []
    for item in template:
        rendered = item.replace("{request_json}", str(request_path))
        if "{" in rendered or "}" in rendered:
            raise FrontierwrightError(
                "SLURM_PROFILE_INVALID",
                f"Unsupported worker placeholder in: {item}",
                2,
            )
        result.append(rendered)
    return result


def _job_time_minutes(profile: SlurmExecutorProfile, request: dict[str, Any]) -> int:
    budgets = request.get("budgets")
    wall_seconds: float | None = None
    if isinstance(budgets, dict):
        raw = budgets.get("max_wall_seconds")
        if (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
            and float(raw) > 0
        ):
            wall_seconds = float(raw)
    if wall_seconds is None:
        return profile.time_limit_minutes
    return max(1, min(profile.time_limit_minutes, math.ceil(wall_seconds / 60.0)))


def _build_submit_argv(
    profile: SlurmExecutorProfile,
    *,
    request: dict[str, Any],
    script_path: Path,
    stdout_pattern: Path,
    stderr_pattern: Path,
) -> list[str]:
    job_name = (
        f"fw-{str(request.get('operation', 'job'))[:8]}-{str(request.get('plan_id', 'plan'))[-8:]}"
    )
    argv = [
        *profile.sbatch_argv,
        "--parsable",
        "--wait",
        "--no-requeue",
        "--job-name",
        job_name,
        "--nodes",
        str(profile.nodes),
        "--cpus-per-task",
        str(profile.cpus_per_task),
        "--time",
        str(_job_time_minutes(profile, request)),
        "--output",
        str(stdout_pattern),
        "--error",
        str(stderr_pattern),
    ]
    if profile.gpus_per_node is not None:
        argv.extend(["--gpus-per-node", str(profile.gpus_per_node)])
    if profile.memory_mb is not None:
        argv.extend(["--mem", str(profile.memory_mb)])
    if profile.partition is not None:
        argv.extend(["--partition", profile.partition])
    if profile.account is not None:
        argv.extend(["--account", profile.account])
    if profile.qos is not None:
        argv.extend(["--qos", profile.qos])
    argv.append(str(script_path))
    return argv


def _readline_with_timeout(stream: IO[str], timeout_seconds: float) -> str:
    output: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

    def reader() -> None:
        try:
            output.put(stream.readline())
        except BaseException as exc:  # pragma: no cover - defensive thread boundary
            output.put(exc)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        value = output.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise FrontierwrightError(
            "SLURM_SUBMIT_TIMEOUT",
            f"sbatch did not return a job ID within {timeout_seconds:g} seconds.",
            14,
        ) from exc
    if isinstance(value, BaseException):
        raise FrontierwrightError(
            "SLURM_SUBMIT_FAILED", f"Could not read sbatch job ID: {value}", 14
        )
    return value


def _parse_job_id(line: str) -> str:
    job_id = line.strip().split(";", 1)[0]
    if not job_id or not job_id.isdigit():
        raise FrontierwrightError(
            "SLURM_SUBMIT_INVALID",
            f"sbatch --parsable did not return a numeric job ID: {line.strip()!r}",
            14,
        )
    return job_id


def _cancel_job(profile: SlurmExecutorProfile, job_id: str) -> dict[str, object]:
    try:
        result = subprocess.run(
            [*profile.scancel_argv, job_id],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return {
            "attempted": True,
            "return_code": result.returncode,
            "stderr": result.stderr.strip()[:500],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"attempted": True, "return_code": None, "error": str(exc)[:500]}


def _account_job(profile: SlurmExecutorProfile, job_id: str) -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                *profile.sacct_argv,
                "-j",
                job_id,
                "-X",
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,MaxRSS,MaxVMSize,ConsumedEnergyRaw",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)[:500]}
    if result.returncode != 0:
        return {
            "available": False,
            "return_code": result.returncode,
            "stderr": result.stderr.strip()[:500],
        }
    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 8 or not parts[0]:
            continue
        rows.append(
            {
                "job_id_raw": parts[0],
                "state": parts[1],
                "exit_code": parts[2],
                "elapsed_raw_seconds": (int(parts[3]) if parts[3].isdigit() else None),
                "alloc_tres": parts[4],
                "max_rss": parts[5],
                "max_vmsize": parts[6],
                "consumed_energy_raw": (int(parts[7]) if parts[7].isdigit() else None),
            }
        )
    top = next((item for item in rows if item["job_id_raw"] == job_id), None)
    return {"available": bool(rows), "job": top, "rows": rows}


def _read_worker_result(path: Path) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "SLURM_WORKER_RESULT_INVALID",
            "Slurm worker stdout must contain exactly one UTF-8 JSON result object.",
            14,
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("ok") is not True
    ):
        raise FrontierwrightError(
            "SLURM_WORKER_RESULT_INVALID",
            "Slurm worker result must be schema_version=1 with ok=true.",
            14,
        )
    return payload


def run_slurm_request(
    profile_path: Path,
    *,
    expected_profile_hash: str,
    request_path: Path,
    submit_id_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    if os.name == "nt":
        raise FrontierwrightError(
            "SLURM_EXECUTOR_UNSUPPORTED_OS",
            "The built-in Slurm bridge requires a POSIX submit host.",
            14,
        )
    profile = load_slurm_executor_profile(profile_path)
    if profile.sha256 != expected_profile_hash:
        raise FrontierwrightError(
            "SLURM_PROFILE_DRIFT",
            "Slurm executor profile hash changed after the backend spec was pinned.",
            13,
        )
    availability = inspect_slurm_availability(profile)
    if availability["ready"] is not True:
        raise FrontierwrightError(
            "SLURM_EXECUTOR_NOT_READY",
            "sbatch/scancel are not available on this submit host.",
            12,
        )

    request_path = request_path.resolve()
    request = _load_request(request_path)
    operation = request.get("operation")
    if operation not in {"calibrate", "train"}:
        raise FrontierwrightError(
            "SLURM_REQUEST_INVALID",
            "Slurm bridge supports only calibrate/train backend operations.",
            14,
        )
    try:
        path_id = TrainingPathId(str(request["path_id"]))
    except (KeyError, ValueError) as exc:
        raise FrontierwrightError(
            "SLURM_REQUEST_INVALID", "Request contains an invalid path_id.", 14
        ) from exc
    if path_id not in profile.supported_paths:
        raise FrontierwrightError(
            "SLURM_PATH_UNSUPPORTED",
            f"Slurm profile does not support {path_id.value}.",
            12,
        )
    if request.get("backend_data_boundary") != BackendDataBoundary.CONTROLLED_PRIVATE.value:
        raise FrontierwrightError(
            "SLURM_BOUNDARY_MISMATCH",
            "Slurm bridge requires CONTROLLED_PRIVATE backend data boundary.",
            13,
        )
    if request.get("backend_adapter_ref") != profile.adapter_ref:
        raise FrontierwrightError(
            "SLURM_ADAPTER_MISMATCH",
            "Training plan adapter reference does not match the pinned Slurm profile.",
            13,
        )

    template = (
        profile.calibrate_worker_argv if operation == "calibrate" else profile.train_worker_argv
    )
    worker_argv = _render_worker_argv(template, request_path=request_path)
    job_root = request_path.parent / f"slurm-{operation}-{uuid4().hex[:12]}"
    job_root.mkdir(parents=True, exist_ok=False)
    script_path = job_root / "job.sh"
    stdout_pattern = job_root / "worker-%j.json"
    stderr_pattern = job_root / "worker-%j.stderr"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(worker_argv) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    script_path.chmod(0o700)

    submit_argv = _build_submit_argv(
        profile,
        request=request,
        script_path=script_path,
        stdout_pattern=stdout_pattern,
        stderr_pattern=stderr_pattern,
    )
    try:
        process = subprocess.Popen(
            submit_argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise FrontierwrightError(
            "SLURM_SUBMIT_FAILED",
            f"Could not launch sbatch command: {submit_argv[0]}",
            14,
        ) from exc

    job_id: str | None = None
    cancel_requested = False

    def request_cancel(_signum: int, _frame: object) -> None:
        nonlocal cancel_requested
        cancel_requested = True

    previous_term = signal.signal(signal.SIGTERM, request_cancel)
    previous_int = signal.signal(signal.SIGINT, request_cancel)
    try:
        assert process.stdout is not None
        first_line = _readline_with_timeout(process.stdout, submit_id_timeout_seconds)
        job_id = _parse_job_id(first_line)

        wall_minutes = _job_time_minutes(profile, request)
        deadline = time.monotonic() + profile.max_queue_wait_seconds + wall_minutes * 60 + 60
        while process.poll() is None:
            if cancel_requested:
                cancellation = _cancel_job(profile, job_id)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise FrontierwrightError(
                    "SLURM_JOB_CANCELLED",
                    f"Slurm job {job_id} was cancelled after local interruption: {cancellation}.",
                    14,
                )
            if time.monotonic() >= deadline:
                cancellation = _cancel_job(profile, job_id)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise FrontierwrightError(
                    "SLURM_JOB_TIMEOUT",
                    f"Slurm job {job_id} exceeded queue+execution wait bound: {cancellation}.",
                    14,
                )
            time.sleep(0.2)

        assert process.stderr is not None
        stderr = process.stderr.read()
        return_code = process.returncode
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)

    assert job_id is not None
    stdout_path = Path(str(stdout_pattern).replace("%j", job_id))
    stderr_path = Path(str(stderr_pattern).replace("%j", job_id))
    worker_stderr = ""
    if stderr_path.exists():
        worker_stderr = stderr_path.read_text(encoding="utf-8", errors="replace")[:2000]
    if return_code != 0:
        raise FrontierwrightError(
            "SLURM_JOB_FAILED",
            (
                f"Slurm job {job_id} exited with code {return_code}. "
                f"sbatch stderr={stderr.strip()[:500]!r}; "
                f"worker stderr={worker_stderr[:1000]!r}"
            ),
            14,
        )

    payload = _read_worker_result(stdout_path)
    accounting = _account_job(profile, job_id)
    scheduler = {
        "kind": "SLURM",
        "bridge_id": SLURM_BRIDGE_ID,
        "bridge_version": SLURM_BRIDGE_VERSION,
        "job_id": job_id,
        "profile_hash": profile.sha256,
        "adapter_ref": profile.adapter_ref,
        "submit_argv": submit_argv[:-1] + ["<job-script>"],
        "accounting": accounting,
        "worker_stderr_empty": not bool(worker_stderr.strip()),
    }
    payload["scheduler"] = scheduler
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        metrics["scheduler"] = scheduler
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-hash", required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = run_slurm_request(
            args.profile,
            expected_profile_hash=args.expected_profile_hash,
            request_path=args.request,
        )
    except FrontierwrightError as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": {"code": exc.code, "message": str(exc)},
                },
                sort_keys=True,
            )
        )
        return exc.exit_code or 1
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
