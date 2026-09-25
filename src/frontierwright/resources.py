"""Local resource detection with explicit provenance and actionable diagnostics."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from frontierwright.domain import ResourceProvenance


@dataclass(frozen=True)
class GPUResource:
    vendor: str
    name: str
    memory_total_bytes: int | None
    memory_free_bytes: int | None
    driver_version: str | None
    provenance: ResourceProvenance = ResourceProvenance.DETECTED


@dataclass(frozen=True)
class ResourceDiagnostic:
    """Explain what a probe established and how to resolve missing evidence."""

    probe_id: str
    status: str
    reason_code: str
    detail: str
    next_action: str | None = None
    physical_absence_proven: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "next_action": self.next_action,
            "physical_absence_proven": self.physical_absence_proven,
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    provenance: ResourceProvenance
    platform: str
    cpu_model: str
    cpu_logical_count: int | None
    ram_total_bytes: int | None
    ram_available_bytes: int | None
    disk_total_bytes: int
    disk_free_bytes: int
    gpus: tuple[GPUResource, ...]
    torch_version: str | None
    cuda_toolkit_version: str | None
    rocm_version: str | None
    bf16_supported: bool | None
    fp16_supported: bool | None
    diagnostics: tuple[ResourceDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "provenance": self.provenance.value,
            "platform": self.platform,
            "cpu_model": self.cpu_model,
            "cpu_logical_count": self.cpu_logical_count,
            "ram_total_bytes": self.ram_total_bytes,
            "ram_available_bytes": self.ram_available_bytes,
            "disk_total_bytes": self.disk_total_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "gpus": [
                {
                    "vendor": gpu.vendor,
                    "name": gpu.name,
                    "memory_total_bytes": gpu.memory_total_bytes,
                    "memory_free_bytes": gpu.memory_free_bytes,
                    "driver_version": gpu.driver_version,
                    "provenance": gpu.provenance.value,
                }
                for gpu in self.gpus
            ],
            "torch_version": self.torch_version,
            "cuda_toolkit_version": self.cuda_toolkit_version,
            "rocm_version": self.rocm_version,
            "bf16_supported": self.bf16_supported,
            "fp16_supported": self.fp16_supported,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _cpu_model() -> str:
    candidate = platform.processor().strip()
    if candidate:
        return candidate

    # platform.processor() is frequently empty on Windows Python builds.
    windows_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if windows_identifier:
        return windows_identifier

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()

    return platform.machine() or "UNKNOWN"


def _linux_memory() -> tuple[int | None, int | None]:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None, None
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        match = re.search(r"(\d+)", raw)
        if match:
            values[key] = int(match.group(1)) * 1024
    return values.get("MemTotal"), values.get("MemAvailable")


def _windows_memory() -> tuple[int | None, int | None]:
    if os.name != "nt":
        return None, None
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        windll = getattr(ctypes, "windll", None)
        kernel32 = getattr(windll, "kernel32", None) if windll is not None else None
        if kernel32 is not None and kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (AttributeError, OSError):
        pass
    return None, None


def _memory() -> tuple[int | None, int | None]:
    total, available = _linux_memory()
    if total is not None:
        return total, available
    return _windows_memory()


def _run(command: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _detail_from_process(result: subprocess.CompletedProcess[str]) -> str:
    raw = (result.stderr or result.stdout or "").strip().replace("\r", " ").replace("\n", " ")
    return raw[:400] if raw else f"process exited with code {result.returncode}"


def _probe_nvidia() -> tuple[tuple[GPUResource, ...], ResourceDiagnostic]:
    executable = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if executable is None:
        return (), ResourceDiagnostic(
            probe_id="gpu.nvidia",
            status="MEASUREMENT_NEEDED",
            reason_code="NVIDIA_SMI_NOT_FOUND",
            detail="NVIDIA probe executable was not found in this runtime.",
            next_action=(
                "Install or repair the NVIDIA driver/runtime, or verify the device from the "
                "host OS, then refresh resources."
            ),
        )

    result = _run(
        [
            executable,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None:
        return (), ResourceDiagnostic(
            probe_id="gpu.nvidia",
            status="MEASUREMENT_NEEDED",
            reason_code="NVIDIA_SMI_UNAVAILABLE",
            detail="The NVIDIA probe could not be executed or did not complete before timeout.",
            next_action="Run nvidia-smi directly, repair the driver/runtime if needed, then retry.",
        )
    if result.returncode != 0:
        return (), ResourceDiagnostic(
            probe_id="gpu.nvidia",
            status="MEASUREMENT_NEEDED",
            reason_code="NVIDIA_SMI_FAILED",
            detail=_detail_from_process(result),
            next_action="Resolve the reported NVIDIA driver/runtime error, then refresh resources.",
        )

    gpus: list[GPUResource] = []
    malformed = 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            malformed += 1
            continue
        name, total_mib, free_mib, driver = parts
        try:
            total = int(float(total_mib) * 1024 * 1024)
            free = int(float(free_mib) * 1024 * 1024)
        except ValueError:
            total = None
            free = None
        gpus.append(
            GPUResource(
                vendor="NVIDIA",
                name=name,
                memory_total_bytes=total,
                memory_free_bytes=free,
                driver_version=driver or None,
            )
        )

    if gpus:
        return tuple(gpus), ResourceDiagnostic(
            probe_id="gpu.nvidia",
            status="MEASURED",
            reason_code="NVIDIA_GPU_VISIBLE",
            detail=f"nvidia-smi reported {len(gpus)} visible NVIDIA GPU(s).",
        )

    reason = "NVIDIA_SMI_UNPARSEABLE" if malformed else "NVIDIA_SMI_NO_VISIBLE_GPU"
    detail = (
        f"nvidia-smi completed but {malformed} non-empty row(s) could not be parsed."
        if malformed
        else "nvidia-smi completed but returned no visible GPU rows."
    )
    return (), ResourceDiagnostic(
        probe_id="gpu.nvidia",
        status="MEASUREMENT_NEEDED",
        reason_code=reason,
        detail=detail,
        next_action=(
            "Check host device visibility, driver configuration, containers/WSL passthrough, "
            "then refresh resources."
        ),
    )


def _detect_nvidia() -> tuple[GPUResource, ...]:
    """Compatibility helper for callers that only need measured GPU records."""

    return _probe_nvidia()[0]


def _probe_rocm() -> tuple[tuple[GPUResource, ...], ResourceDiagnostic]:
    executable = shutil.which("rocm-smi")
    if executable is None:
        return (), ResourceDiagnostic(
            probe_id="gpu.rocm",
            status="MEASUREMENT_NEEDED",
            reason_code="ROCM_SMI_NOT_FOUND",
            detail="ROCm probe executable was not found in this runtime.",
            next_action=(
                "If this machine is expected to use an AMD GPU, install/repair ROCm or verify "
                "the device from the host OS, then refresh resources."
            ),
        )

    result = _run([executable, "--showproductname", "--showmeminfo", "vram", "--json"])
    if result is None:
        return (), ResourceDiagnostic(
            probe_id="gpu.rocm",
            status="MEASUREMENT_NEEDED",
            reason_code="ROCM_SMI_UNAVAILABLE",
            detail="The ROCm probe could not be executed or did not complete before timeout.",
            next_action="Run rocm-smi directly, repair ROCm if needed, then refresh resources.",
        )
    if result.returncode != 0:
        return (), ResourceDiagnostic(
            probe_id="gpu.rocm",
            status="MEASUREMENT_NEEDED",
            reason_code="ROCM_SMI_FAILED",
            detail=_detail_from_process(result),
            next_action="Resolve the reported ROCm/runtime error, then refresh resources.",
        )

    # ROCm JSON field names vary substantially across versions. Preserve the raw
    # probe success as diagnostic evidence instead of inventing VRAM values.
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and parsed:
        return (), ResourceDiagnostic(
            probe_id="gpu.rocm",
            status="UNSUPPORTED_HERE",
            reason_code="ROCM_GPU_PARSER_NOT_IMPLEMENTED",
            detail=(
                "rocm-smi returned structured device evidence but this build "
                "cannot safely map it."
            ),
            next_action=(
                "Use a serving/runtime resource receipt for the exact AMD model workload, or add "
                "a versioned ROCm parser before treating VRAM as measured."
            ),
        )

    return (), ResourceDiagnostic(
        probe_id="gpu.rocm",
        status="MEASUREMENT_NEEDED",
        reason_code="ROCM_SMI_NO_STRUCTURED_GPU",
        detail="rocm-smi completed but did not return structured GPU evidence.",
        next_action="Check ROCm device visibility and retry resource detection.",
    )


def _detect_rocm_gpus() -> tuple[GPUResource, ...]:
    """Compatibility helper for callers that only need measured GPU records."""

    return _probe_rocm()[0]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_toolkit_version() -> str | None:
    executable = shutil.which("nvcc")
    if executable is None:
        return None
    result = _run([executable, "--version"])
    if result is None:
        return None
    match = re.search(r"release\s+([0-9.]+)", result.stdout + result.stderr)
    return match.group(1) if match else None


def _rocm_version() -> str | None:
    executable = shutil.which("rocminfo")
    if executable is None:
        return None
    result = _run([executable])
    if result is None or result.returncode != 0:
        return None
    match = re.search(r"Runtime Version:\s*([0-9.]+)", result.stdout)
    return match.group(1) if match else "DETECTED"


def detect_local_resources(root: Path) -> ResourceSnapshot:
    root = root.resolve()
    disk = shutil.disk_usage(root)
    ram_total, ram_available = _memory()

    nvidia_gpus, nvidia_diagnostic = _probe_nvidia()
    diagnostics: list[ResourceDiagnostic] = [nvidia_diagnostic]
    gpus = nvidia_gpus

    if not gpus:
        rocm_gpus, rocm_diagnostic = _probe_rocm()
        diagnostics.append(rocm_diagnostic)
        gpus = rocm_gpus

    diagnostics.extend(
        [
            ResourceDiagnostic(
                probe_id="precision.bf16",
                status="MEASUREMENT_NEEDED",
                reason_code="BACKEND_CALIBRATION_REQUIRED",
                detail=(
                    "bf16 support depends on the exact backend/runtime and has not "
                    "been calibrated."
                ),
                next_action="Calibrate the intended backend/runtime before claiming bf16 support.",
            ),
            ResourceDiagnostic(
                probe_id="precision.fp16",
                status="MEASUREMENT_NEEDED",
                reason_code="BACKEND_CALIBRATION_REQUIRED",
                detail=(
                    "fp16 support depends on the exact backend/runtime and has not "
                    "been calibrated."
                ),
                next_action="Calibrate the intended backend/runtime before claiming fp16 support.",
            ),
        ]
    )

    return ResourceSnapshot(
        provenance=ResourceProvenance.DETECTED,
        platform=platform.platform(),
        cpu_model=_cpu_model(),
        cpu_logical_count=os.cpu_count(),
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,
        gpus=gpus,
        torch_version=_package_version("torch"),
        cuda_toolkit_version=_cuda_toolkit_version(),
        rocm_version=_rocm_version(),
        bf16_supported=None,
        fp16_supported=None,
        diagnostics=tuple(diagnostics),
    )
