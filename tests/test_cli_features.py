import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.registry import Registry

runner = CliRunner()


def test_data_and_paths_machine_surface(tmp_path: Path) -> None:
    project = tmp_path / "project"
    data = tmp_path / "pretrain"
    data.mkdir()
    (data / "data.txt").write_text("hello model\n", encoding="utf-8")

    init = runner.invoke(
        app,
        [
            "project",
            "init",
            str(project),
            "--name",
            "ZERO",
            "--origin",
            "ZERO",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert init.exit_code == 0, init.output

    interventions = runner.invoke(
        app,
        ["interventions", "--json", "--non-interactive"],
    )
    assert interventions.exit_code == 0, interventions.output
    intervention_payload = json.loads(interventions.stdout)
    assert intervention_payload["interventions"][0]["intervention_id"] == (
        "frontierwright.birth.zero"
    )
    assert {
        item["intervention_id"] for item in intervention_payload["interventions"]
    } >= {
        "frontierwright.learn.pretrain",
        "frontierwright.specialize.lora-sft",
    }

    recipes = runner.invoke(
        app,
        ["data", "recipes", "--json", "--non-interactive"],
    )
    assert recipes.exit_code == 0, recipes.output
    recipe_payload = json.loads(recipes.stdout)
    assert recipe_payload["plugins"] == [
        {
            "plugin_id": "frontierwright.data.snapshot-copy",
            "plugin_version": "1",
            "title": "Byte-preserving managed snapshot",
        }
    ]

    added = runner.invoke(
        app,
        [
            "data",
            "add",
            str(data),
            "--role",
            "PRETRAIN",
            "--path",
            str(project),
            "--name",
            "Local pretrain",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert added.exit_code == 0, added.output
    data_payload = json.loads(added.stdout)
    assert data_payload["ok"] is True
    assert data_payload["datasets"][0]["role"] == "PRETRAIN"
    assert data_payload["datasets"][0]["provenance"] == "LOCAL_USER"
    raw_dataset_id = data_payload["datasets"][0]["dataset_id"]

    prepared = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--dataset",
            raw_dataset_id,
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    prepared_payload = json.loads(prepared.stdout)
    managed = next(
        item for item in prepared_payload["datasets"] if item["managed"] is True
    )
    assert managed["source_dataset_id"] == raw_dataset_id
    assert managed["fingerprint"] == data_payload["datasets"][0]["fingerprint"]
    assert managed["preparation_recipe_id"].startswith("data-recipe-")
    assert managed["preparation_recipe_hash"].startswith("sha256:")

    paths = runner.invoke(
        app,
        ["paths", "--path", str(project), "--json", "--non-interactive"],
    )
    assert paths.exit_code == 0, paths.output
    paths_payload = json.loads(paths.stdout)
    zero = next(
        item
        for item in paths_payload["paths"]
        if item["path_id"] == "FROM_SCRATCH_PRETRAINING"
    )
    assert zero["availability"] == "LOCKED"
    assert zero["intervention_family"] == "LEARN"
    assert zero["intervention_id"] == "frontierwright.learn.pretrain"
    assert "zero-model birth required before from-scratch pretraining" in zero["blockers"]


def test_stats_machine_surface_ingests_frozen_evidence(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry = Registry(project)
    registry.initialize("NOVA", ModelOrigin.ZERO)
    state = registry.read()
    model = ModelState(
        model_id="model-current",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint="checkpoint",
        fingerprint="sha256:current",
    )
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_id": "receipt-cli",
                "model_id": model.model_id,
                "model_fingerprint": model.fingerprint,
                "evaluator_id": "cli-fixture",
                "evaluator_version": "1",
                "conditions": {"seed": 1},
                "measurements": [
                    {
                        "task_id": "coding-task",
                        "task_version": "1",
                        "metric": "pass_rate",
                        "value": 0.8,
                        "higher_is_better": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scale = tmp_path / "scale.json"
    scale.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scale_id": "cli-scale",
                "scale_version": "1",
                "frozen": True,
                "axes": [
                    {
                        "axis": "CODING",
                        "display_anchor": 100,
                        "tasks": [
                            {
                                "task_id": "coding-task",
                                "task_version": "1",
                                "metric": "pass_rate",
                                "weight": 1,
                                "raw_anchor": 0.5,
                                "raw_unit": 0.1,
                                "display_per_unit": 10,
                                "higher_is_better": True,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ingest = runner.invoke(
        app,
        [
            "stats",
            "ingest",
            "--receipt",
            str(receipt),
            "--scale",
            str(scale),
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert ingest.exit_code == 0, ingest.output
    payload = json.loads(ingest.stdout)
    assert payload["measured"] is True
    assert payload["stats"]["coding"] == 130.0
    assert payload["scale_id"] == "cli-scale"
    assert payload["receipt_id"] == "receipt-cli"

    show = runner.invoke(
        app,
        ["stats", "show", "--path", str(project), "--json", "--non-interactive"],
    )
    assert show.exit_code == 0, show.output
    assert json.loads(show.stdout) == payload
