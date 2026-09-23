from pathlib import Path

from frontierwright.domain import ModelOrigin, ResourceProvenance
from frontierwright.registry import Registry
from frontierwright.resources import ResourceSnapshot, detect_local_resources
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
