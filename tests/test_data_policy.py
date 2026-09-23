import hashlib
import json
import sys
from pathlib import Path

import pytest

from frontierwright.data import DatasetClassification, DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import (
    BackendDataBoundary,
    HardBudgets,
    PermissionLevel,
    backend_allows_dataset,
    load_command_backend_spec,
)
from frontierwright.paths import TrainingPathId
from frontierwright.service import (
    add_local_dataset,
    create_training_plan,
    get_data_view,
    import_local_model,
    prepare_dataset_snapshot,
)


def make_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def make_data(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "sft.jsonl").write_text('{"text":"private"}\n', encoding="utf-8")
    return root


def write_backend(
    root: Path,
    *,
    boundary: str | None,
) -> Path:
    root.mkdir(parents=True)
    spec: dict[str, object] = {
        "schema_version": 1,
        "backend_id": "policy-fixture",
        "supported_paths": ["LORA_SFT"],
        "calibrate_argv": [sys.executable, "-c", "print('{}')", "{request_json}"],
        "train_argv": [sys.executable, "-c", "print('{}')", "{request_json}"],
        "environment": {},
    }
    if boundary is not None:
        spec["data_boundary"] = boundary
    path = root / "backend.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def setup_project(
    tmp_path: Path,
    *,
    classification: DatasetClassification | None = None,
) -> tuple[Path, str]:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Policy",
    )
    view = add_local_dataset(
        project,
        make_data(tmp_path / "data"),
        name="SFT",
        role=DatasetRole.SFT,
        classification=classification,
    )
    return project, str(view.datasets[0]["dataset_id"])


def test_local_user_data_defaults_private_and_preparation_preserves_it(
    tmp_path: Path,
) -> None:
    project, dataset_id = setup_project(tmp_path)

    before = get_data_view(project)
    source = next(item for item in before.datasets if item["dataset_id"] == dataset_id)
    assert source["classification"] == "PRIVATE"

    after = prepare_dataset_snapshot(project, dataset_id=dataset_id)
    prepared = next(item for item in after.datasets if item["managed"])
    assert prepared["classification"] == "PRIVATE"
    assert prepared["source_dataset_id"] == dataset_id


@pytest.mark.parametrize(
    ("boundary", "classification", "allowed"),
    [
        (BackendDataBoundary.LOCAL_MACHINE, DatasetClassification.PRIVATE, True),
        (BackendDataBoundary.CONTROLLED_PRIVATE, DatasetClassification.CONFIDENTIAL, True),
        (BackendDataBoundary.CONTROLLED_PRIVATE, DatasetClassification.INTERNAL, True),
        (BackendDataBoundary.EXTERNAL, DatasetClassification.PUBLIC, True),
        (BackendDataBoundary.EXTERNAL, DatasetClassification.INTERNAL, False),
        (BackendDataBoundary.EXTERNAL, DatasetClassification.CONFIDENTIAL, False),
        (BackendDataBoundary.EXTERNAL, DatasetClassification.PRIVATE, False),
        (BackendDataBoundary.UNKNOWN, DatasetClassification.PUBLIC, True),
        (BackendDataBoundary.UNKNOWN, DatasetClassification.PRIVATE, False),
    ],
)
def test_backend_data_boundary_matrix(
    boundary: BackendDataBoundary,
    classification: DatasetClassification,
    allowed: bool,
) -> None:
    assert backend_allows_dataset(boundary, classification) is allowed


def test_external_backend_cannot_plan_with_private_dataset(tmp_path: Path) -> None:
    project, dataset_id = setup_project(tmp_path)
    backend = write_backend(tmp_path / "backend", boundary="EXTERNAL")

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

    assert captured.value.code == "DATA_POLICY_LOCKED"
    assert "PRIVATE data cannot be used with EXTERNAL" in str(captured.value)


def test_external_backend_can_plan_with_explicit_public_dataset(tmp_path: Path) -> None:
    project, dataset_id = setup_project(
        tmp_path,
        classification=DatasetClassification.PUBLIC,
    )
    backend = write_backend(tmp_path / "backend", boundary="EXTERNAL")

    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=backend,
        dataset_id=dataset_id,
        permission=PermissionLevel.PLAN,
        budgets=HardBudgets(max_runs=1),
        config={"epochs": 1},
    )

    assert plan.dataset_classification == "PUBLIC"
    assert plan.backend_data_boundary == "EXTERNAL"
    assert plan.plan_id is not None


def test_legacy_backend_spec_preserves_pre_boundary_hash(tmp_path: Path) -> None:
    path = write_backend(tmp_path / "legacy", boundary=None)
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_bytes = json.dumps(
        {
            "backend_id": raw["backend_id"],
            "supported_paths": raw["supported_paths"],
            "calibrate_argv": raw["calibrate_argv"],
            "train_argv": raw["train_argv"],
            "environment": raw["environment"],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    expected = f"sha256:{hashlib.sha256(expected_bytes).hexdigest()}"

    backend = load_command_backend_spec(path)

    assert backend.data_boundary is BackendDataBoundary.LOCAL_MACHINE
    assert backend.data_boundary_explicit is False
    assert backend.sha256 == expected


def test_explicit_backend_boundary_is_pinned_into_hash(tmp_path: Path) -> None:
    local = load_command_backend_spec(
        write_backend(tmp_path / "local", boundary="LOCAL_MACHINE")
    )
    external = load_command_backend_spec(
        write_backend(tmp_path / "external", boundary="EXTERNAL")
    )

    assert local.data_boundary_explicit is True
    assert external.data_boundary_explicit is True
    assert local.sha256 != external.sha256
