import json
import sys
from pathlib import Path

import pytest

from frontierwright.data import DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import HardBudgets, PermissionLevel
from frontierwright.paths import TrainingPathId
from frontierwright.registry import Registry
from frontierwright.service import (
    add_local_dataset,
    create_training_plan,
    get_data_view,
    import_local_model,
    prepare_dataset_snapshot,
)


def make_dataset(root: Path, text: str = "alpha\nbeta\n") -> Path:
    root.mkdir(parents=True)
    (root / "train.txt").write_text(text, encoding="utf-8")
    return root


def make_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def write_backend_spec(root: Path) -> Path:
    root.mkdir(parents=True)
    script = root / "noop.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    spec = root / "backend.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": "recipe-test-backend",
                "supported_paths": ["LORA_SFT"],
                "calibrate_argv": [sys.executable, str(script), "{request_json}"],
                "train_argv": [sys.executable, str(script), "{request_json}"],
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    return spec


def test_snapshot_recipe_is_content_preserving_and_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    add_local_dataset(
        project,
        make_dataset(tmp_path / "data"),
        name="Raw pretrain",
        role=DatasetRole.PRETRAIN,
        license_name="Apache-2.0",
        language="en",
    )
    raw = get_data_view(project).datasets[0]

    first = prepare_dataset_snapshot(
        project,
        dataset_id=str(raw["dataset_id"]),
        name="Frozen pretrain",
    )
    second = prepare_dataset_snapshot(
        project,
        dataset_id=str(raw["dataset_id"]),
        name="Ignored duplicate name",
    )

    assert len(first.datasets) == 2
    assert len(second.datasets) == 2
    prepared = next(item for item in second.datasets if item["managed"] is True)
    assert prepared["fingerprint"] == raw["fingerprint"]
    assert prepared["source_dataset_id"] == raw["dataset_id"]
    assert prepared["provenance"] == "LOCAL_USER"
    assert prepared["license"] == "Apache-2.0"
    assert prepared["language"] == "en"
    assert isinstance(prepared["preparation_recipe_id"], str)
    assert str(prepared["preparation_recipe_hash"]).startswith("sha256:")
    assert Path(str(prepared["source_path"])).is_dir()
    assert (
        Path(str(prepared["source_path"])).read_bytes()
        if Path(str(prepared["source_path"])).is_file()
        else None
    ) is None

    recipe = Registry(project).get_data_recipe(str(prepared["preparation_recipe_id"]))
    assert recipe is not None
    assert recipe["plugin_id"] == "frontierwright.data.snapshot-copy"
    assert recipe["plugin_version"] == "1"
    assert recipe["source_dataset_id"] == raw["dataset_id"]
    assert recipe["source_fingerprint"] == raw["fingerprint"]
    assert recipe["config"] == {"mode": "byte_preserving_snapshot"}


def test_source_drift_blocks_snapshot_preparation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    source = make_dataset(tmp_path / "data", "original\n")
    add_local_dataset(
        project,
        source,
        name="Raw",
        role=DatasetRole.PRETRAIN,
    )
    raw = get_data_view(project).datasets[0]

    (source / "train.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(FrontierwrightError, match="content changed"):
        prepare_dataset_snapshot(project, dataset_id=str(raw["dataset_id"]))


def test_plan_prefers_single_prepared_dataset_and_pins_recipe(tmp_path: Path) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="NOVA",
    )
    add_local_dataset(
        project,
        make_dataset(tmp_path / "sft"),
        name="Raw SFT",
        role=DatasetRole.SFT,
    )
    raw = get_data_view(project).datasets[0]
    prepared_view = prepare_dataset_snapshot(
        project,
        dataset_id=str(raw["dataset_id"]),
        name="Prepared SFT",
    )
    prepared = next(item for item in prepared_view.datasets if item["managed"] is True)

    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=write_backend_spec(tmp_path / "backend"),
        dataset_id=None,
        permission=PermissionLevel.PLAN,
        budgets=HardBudgets(max_runs=1),
        config={"epochs": 1},
    )

    assert plan.dataset_id == prepared["dataset_id"]
    assert plan.dataset_recipe_id == prepared["preparation_recipe_id"]
    assert plan.dataset_recipe_hash == prepared["preparation_recipe_hash"]

    stored = Registry(project).get_plan(str(plan.plan_id))
    assert stored.dataset_id == prepared["dataset_id"]
    assert stored.dataset_recipe_id == prepared["preparation_recipe_id"]
    assert stored.dataset_recipe_hash == prepared["preparation_recipe_hash"]
    assert stored.request_payload()["dataset_recipe_hash"] == prepared["preparation_recipe_hash"]
