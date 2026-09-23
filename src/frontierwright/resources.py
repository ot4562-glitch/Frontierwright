"""Local resource detection with explicit provenance and no guessed capability flags."""

from __future__ import annotations

import importlib.metadata
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
        }


def _cpu_model() -> str:
    candidate = platform.processor().strip()
    if candidate:
        return candidate
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


def _detect_nvidia() -> tuple[GPUResource, ...]:
    executable = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if executable is None:
        return ()
    result = _run(
        [
            executable,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        return ()

    gpus: list[GPUResource] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
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
    return tuple(gpus)


def _detect_rocm_gpus() -> tuple[GPUResource, ...]:
    executable = shutil.which("rocm-smi")
    if executable is None:
        return ()
    result = _run([executable, "--showproductname", "--showmeminfo", "vram", "--json"])
    if result is None or result.returncode != 0:
        return ()
    # rocm-smi JSON varies by version. Until a stable parser is implemented,
    # record ROCm runtime separately and avoid inventing GPU memory values.
    return ()


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
    gpus = _detect_nvidia()
    if not gpus:
        gpus = _detect_rocm_gpus()

    # Precision support remains unknown until a backend-specific capability
    # probe is run. Hardware-name heuristics would be fake certainty.
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
    )
