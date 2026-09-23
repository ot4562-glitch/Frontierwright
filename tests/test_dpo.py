import json
from pathlib import Path

from frontierwright.data import DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.execution import HardBudgets, PermissionLevel
from frontierwright.paths import TrainingPathId
from frontierwright.service import (
    add_local_dataset,
    create_training_plan,
    import_local_model,
)


def fake_hf_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"type":"fixture"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"fixture-weights")
    return root


def backend_spec(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": "fixture-dpo-backend",
                "data_boundary": "LOCAL_MACHINE",
                "supported_paths": ["DPO"],
                "calibrate_argv": ["fixture", "calibrate", "{request_json}"],
                "train_argv": ["fixture", "train", "{request_json}"],
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dpo_plan_pins_preference_dataset_and_align_intervention(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        fake_hf_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="DPO",
    )
    data = tmp_path / "preference"
    data.mkdir()
    (data / "pairs.jsonl").write_text(
        '{"prompt":"Q:","chosen":" good","rejected":" bad"}\n',
        encoding="utf-8",
    )
    data_view = add_local_dataset(
        project,
        data,
        name="Preference pairs",
        role=DatasetRole.PREFERENCE,
    )
    dataset_id = str(data_view.datasets[0]["dataset_id"])

    view = create_training_plan(
        project,
        path_id=TrainingPathId.DPO,
        backend_spec_path=backend_spec(tmp_path / "backend.json"),
        dataset_id=dataset_id,
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=1),
        config={"dpo_beta": 0.1},
    )

    assert view.path_id == "DPO"
    assert view.intervention_id == "frontierwright.align.dpo"
    assert view.intervention_family == "ALIGN"
    assert view.dataset_id == dataset_id
    assert view.config == {"dpo_beta": 0.1}
    assert view.ready is False
    assert "representative calibration required" in view.blockers
