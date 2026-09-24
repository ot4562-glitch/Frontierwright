from pathlib import Path

from frontierwright.domain import ModelOrigin, ResourceProvenance
from frontierwright.registry import Registry
from frontierwright.resources import GPUResource, ResourceSnapshot, detect_local_resources
from frontierwright.service import get_resource_view


def test_local_resource_detection_reports_detected_facts(tmp_path: Path) -> None:
    snapshot = detect_local_resources(tmp_path)

    assert snapshot.provenance is ResourceProvenance.DETECTED
    assert snapshot.cpu_model
    assert snapshot.cpu_logical_count is None or snapshot.cpu_logical_count > 0
    assert snapshot.disk_total_bytes > 0
    assert snapshot.disk_free_bytes >= 0
    assert snapshot.bf16_supported is None
    assert snapshot.fp16_supported is None


def test_resource_snapshot_persists_as_active_profile(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    snapshot = ResourceSnapshot(
        provenance=ResourceProvenance.DETECTED,
        platform="test-platform",
        cpu_model="test-cpu",
        cpu_logical_count=8,
        ram_total_bytes=64 * 1024**3,
        ram_available_bytes=32 * 1024**3,
        disk_total_bytes=1024 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        gpus=(),
        torch_version=None,
        cuda_toolkit_version=None,
        rocm_version=None,
        bf16_supported=None,
        fp16_supported=None,
    )
    profile_id = registry.save_resource_snapshot(snapshot)

    view = get_resource_view(tmp_path)
    assert view.available is True
    assert view.profile_id == profile_id
    assert view.provenance == "DETECTED"
    assert view.snapshot["cpu_model"] == "test-cpu"
    assert view.snapshot["ram_total_bytes"] == 64 * 1024**3


def test_resource_view_exposes_point_in_time_headroom_without_model_fit_claim(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path)
    registry.initialize("HEADROOM", ModelOrigin.IMPORTED_LOCAL)
    gib = 1024**3
    snapshot = ResourceSnapshot(
        provenance=ResourceProvenance.DETECTED,
        platform="test-platform",
        cpu_model="test-cpu",
        cpu_logical_count=16,
        ram_total_bytes=32 * gib,
        ram_available_bytes=18 * gib,
        disk_total_bytes=1000 * gib,
        disk_free_bytes=400 * gib,
        gpus=(
            GPUResource(
                vendor="NVIDIA",
                name="Example GPU",
                memory_total_bytes=12 * gib,
                memory_free_bytes=3 * gib,
                driver_version="test-driver",
            ),
        ),
        torch_version=None,
        cuda_toolkit_version=None,
        rocm_version=None,
        bf16_supported=None,
        fp16_supported=None,
    )
    registry.save_resource_snapshot(snapshot)

    view = get_resource_view(tmp_path)

    assert view.headroom["semantics"] == "SYSTEM_AVAILABLE_NOW"
    assert view.headroom["model_specific"] is False
    assert view.headroom["ram_available_bytes"] == 18 * gib
    assert view.headroom["ram_available_fraction"] == 18 / 32
    gpus = view.headroom["gpus"]
    assert isinstance(gpus, list)
    assert gpus[0]["memory_available_bytes"] == 3 * gib
    assert gpus[0]["memory_total_bytes"] == 12 * gib
    assert gpus[0]["available_fraction"] == 0.25
    assert "model/inference profile" in str(view.headroom["note"])
