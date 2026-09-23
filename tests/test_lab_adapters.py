import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.data import DatasetClassification, DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import HardBudgets, PermissionLevel
from frontierwright.paths import TrainingPathId
from frontierwright.registry import Registry
from frontierwright.service import (
    add_local_dataset,
    connect_lab_adapter,
    create_training_plan,
    disconnect_lab_adapter,
    get_plan_view,
    import_local_model,
)

runner = CliRunner()


def make_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"lab-model")
    return root


def make_data(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "sft.jsonl").write_text('{"text":"internal"}\n', encoding="utf-8")
    return root


def write_manifest(
    path: Path,
    *,
    adapter_id: str = "corp.trainer",
    adapter_version: str = "1",
    boundary: str = "CONTROLLED_PRIVATE",
    kinds: list[str] | None = None,
    capabilities: list[str] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "display_name": "Corporate Trainer",
        "kinds": kinds or ["TRAINER", "EXECUTOR"],
        "data_boundary": boundary,
        "network_scope": "PRIVATE_ONLY",
        "capabilities": capabilities or ["sft", "lora"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_backend(
    path: Path,
    *,
    boundary: str = "CONTROLLED_PRIVATE",
    adapter_ref: str | None = "corp.trainer@1",
) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "backend_id": "corp-backend",
        "supported_paths": ["LORA_SFT"],
        "calibrate_argv": [sys.executable, "-c", "print('{}')", "{request_json}"],
        "train_argv": [sys.executable, "-c", "print('{}')", "{request_json}"],
        "environment": {},
        "data_boundary": boundary,
    }
    if adapter_ref is not None:
        payload["provider_adapter_ref"] = adapter_ref
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def setup_lab_project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_model(tmp_path / "model"),
        origin=ModelOrigin.INTERNAL_LAB,
        project_name="Private Lab",
    )
    data = add_local_dataset(
        project,
        make_data(tmp_path / "data"),
        name="Internal SFT",
        role=DatasetRole.SFT,
        classification=DatasetClassification.PRIVATE,
    )
    return project, str(data.datasets[0]["dataset_id"])


def test_lab_adapter_connect_is_idempotent_and_disconnectable(tmp_path: Path) -> None:
    project, _ = setup_lab_project(tmp_path)
    manifest = write_manifest(tmp_path / "adapter.json")

    first = connect_lab_adapter(project, manifest)
    second = connect_lab_adapter(project, manifest)

    assert first == second
    assert len(first.adapters) == 1
    assert first.adapters[0]["adapter_ref"] == "corp.trainer@1"
    assert str(first.adapters[0]["manifest_hash"]).startswith("sha256:")

    disconnected = disconnect_lab_adapter(project, "corp.trainer@1")
    assert disconnected.adapters == []

    reconnected = connect_lab_adapter(project, manifest)
    assert len(reconnected.adapters) == 1


def test_same_adapter_ref_cannot_change_manifest_content(tmp_path: Path) -> None:
    project, _ = setup_lab_project(tmp_path)
    first = write_manifest(tmp_path / "first.json")
    connect_lab_adapter(project, first)

    second = write_manifest(
        tmp_path / "second.json",
        capabilities=["sft", "lora", "different-capability"],
    )
    with pytest.raises(FrontierwrightError) as captured:
        connect_lab_adapter(project, second)

    assert captured.value.code == "LAB_ADAPTER_REF_CONFLICT"


def test_lab_adapter_manifest_rejects_embedded_secret(tmp_path: Path) -> None:
    project, _ = setup_lab_project(tmp_path)
    manifest = tmp_path / "adapter.json"
    payload = {
        "schema_version": 1,
        "adapter_id": "corp.trainer",
        "adapter_version": "1",
        "display_name": "Corporate Trainer",
        "kinds": ["TRAINER"],
        "data_boundary": "CONTROLLED_PRIVATE",
        "network_scope": "PRIVATE_ONLY",
        "capabilities": [],
        "api_key": "[REDACTED_SECRET]",
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FrontierwrightError) as captured:
        connect_lab_adapter(project, manifest)

    assert captured.value.code == "LAB_ADAPTER_SECRET_FORBIDDEN"


def test_adapter_connection_requires_lab_profile(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("Studio", ModelOrigin.IMPORTED_LOCAL)
    manifest = write_manifest(tmp_path / "adapter.json")

    with pytest.raises(FrontierwrightError) as captured:
        connect_lab_adapter(project, manifest)

    assert captured.value.code == "LAB_EDITION_REQUIRED"


def test_controlled_private_backend_requires_connected_adapter(tmp_path: Path) -> None:
    project, dataset_id = setup_lab_project(tmp_path)
    backend = write_backend(tmp_path / "backend.json", adapter_ref=None)

    with pytest.raises(FrontierwrightError) as captured:
        create_training_plan(
            project,
            path_id=TrainingPathId.LORA_SFT,
            backend_spec_path=backend,
            dataset_id=dataset_id,
            permission=PermissionLevel.PLAN,
            budgets=HardBudgets(max_runs=1),
            config={"epochs": 1},
        )

    assert captured.value.code == "LAB_ADAPTER_REQUIRED"


def test_plan_pins_adapter_manifest_and_disconnect_stales_plan(tmp_path: Path) -> None:
    project, dataset_id = setup_lab_project(tmp_path)
    manifest = write_manifest(tmp_path / "adapter.json")
    connected = connect_lab_adapter(project, manifest)
    adapter = connected.adapters[0]
    backend = write_backend(tmp_path / "backend.json")

    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend,
        dataset_id=dataset_id,
        permission=PermissionLevel.PLAN,
        budgets=HardBudgets(max_runs=1),
        config={"epochs": 1},
    )

    assert plan.backend_adapter_ref == "corp.trainer@1"
    assert plan.backend_adapter_hash == adapter["manifest_hash"]

    stored = Registry(project).get_plan(str(plan.plan_id))
    assert stored.backend_adapter_ref == "corp.trainer@1"
    assert stored.backend_adapter_hash == adapter["manifest_hash"]

    disconnect_lab_adapter(project, "corp.trainer@1")
    stale = get_plan_view(project, str(plan.plan_id))
    assert "pinned Lab adapter is not connected" in stale.blockers


def test_backend_and_adapter_boundary_must_match(tmp_path: Path) -> None:
    project, dataset_id = setup_lab_project(tmp_path)
    connect_lab_adapter(
        project,
        write_manifest(tmp_path / "adapter.json", boundary="LOCAL_MACHINE"),
    )
    backend = write_backend(tmp_path / "backend.json", boundary="CONTROLLED_PRIVATE")

    with pytest.raises(FrontierwrightError) as captured:
        create_training_plan(
            project,
            path_id=TrainingPathId.LORA_SFT,
            backend_spec_path=backend,
            dataset_id=dataset_id,
            permission=PermissionLevel.PLAN,
            budgets=HardBudgets(max_runs=1),
            config={"epochs": 1},
        )

    assert captured.value.code == "LAB_ADAPTER_BOUNDARY_MISMATCH"


def test_backend_provider_requires_trainer_or_executor_kind(tmp_path: Path) -> None:
    project, dataset_id = setup_lab_project(tmp_path)
    connect_lab_adapter(
        project,
        write_manifest(tmp_path / "adapter.json", kinds=["DATASET_SOURCE"]),
    )
    backend = write_backend(tmp_path / "backend.json")

    with pytest.raises(FrontierwrightError) as captured:
        create_training_plan(
            project,
            path_id=TrainingPathId.LORA_SFT,
            backend_spec_path=backend,
            dataset_id=dataset_id,
            permission=PermissionLevel.PLAN,
            budgets=HardBudgets(max_runs=1),
            config={"epochs": 1},
        )

    assert captured.value.code == "LAB_ADAPTER_KIND_UNSUPPORTED"


def test_lab_adapter_cli_json_surface(tmp_path: Path) -> None:
    project, _ = setup_lab_project(tmp_path)
    manifest = write_manifest(tmp_path / "adapter.json")

    connected = runner.invoke(
        app,
        [
            "lab",
            "adapters",
            "connect",
            str(manifest),
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert connected.exit_code == 0, connected.output
    payload = json.loads(connected.stdout)
    assert payload["ok"] is True
    assert payload["adapters"][0]["adapter_ref"] == "corp.trainer@1"

    listed = runner.invoke(
        app,
        [
            "lab",
            "adapters",
            "list",
            "--path",
            str(project),
            "--json",
            "--non-interactive",
        ],
    )
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout) == payload

    disconnected = runner.invoke(
        app,
        [
            "lab",
            "adapters",
            "disconnect",
            "corp.trainer@1",
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert disconnected.exit_code == 0, disconnected.output
    assert json.loads(disconnected.stdout)["adapters"] == []
