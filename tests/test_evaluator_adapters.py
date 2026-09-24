from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.registry import Registry
from frontierwright.service import get_stats_view

runner = CliRunner()


def setup_project(root: Path) -> tuple[Registry, ModelState]:
    registry = Registry(root)
    registry.initialize("NOVA", ModelOrigin.IMPORTED_LOCAL)
    state = registry.read()
    model = ModelState(
        model_id="model-lm-eval",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="synthetic-checkpoint",
        fingerprint="sha256:lm-eval-model",
    )
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    return registry, model


def write_lm_eval_result(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "results": {
                    "hellaswag": {
                        "acc,none": 0.61,
                        "acc_stderr,none": 0.012,
                        "acc_norm,none": 0.72,
                        "acc_norm_stderr,none": 0.011,
                        "alias": "hellaswag",
                    },
                    "mmlu_abstract_algebra": {
                        "acc,none": 0.44,
                        "acc_stderr,none": 0.03,
                    },
                },
                "configs": {
                    "hellaswag": {"num_fewshot": 0, "metadata": {"version": 1}},
                    "mmlu_abstract_algebra": {
                        "num_fewshot": 5,
                        "metadata": {"version": "1.0"},
                    },
                },
                "versions": {
                    "hellaswag": 1,
                    "mmlu_abstract_algebra": "1.0",
                },
                "n-shot": {"hellaswag": 0, "mmlu_abstract_algebra": 5},
                "higher_is_better": {
                    "hellaswag": {"acc": True, "acc_norm": True},
                    "mmlu_abstract_algebra": {"acc": True},
                },
                "n-samples": {
                    "hellaswag": {"original": 10042, "effective": 10042},
                    "mmlu_abstract_algebra": {"original": 100, "effective": 100},
                },
                "samples": {
                    "hellaswag": [{"doc_id": 0, "target": 1, "resps": [["private"]]}]
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_cli_import_lm_eval_preserves_raw_evidence_without_activating_stats(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, model = setup_project(project)
    result_path = write_lm_eval_result(tmp_path / "lm-eval.json")

    first = runner.invoke(
        app,
        [
            "eval",
            "import-lm-eval",
            str(result_path),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--harness-version",
            "0.4.9@deadbeef",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert first.exit_code == 0, first.output
    payload = json.loads(first.stdout)
    assert payload["ok"] is True
    assert payload["task_count"] == 2
    assert payload["measurement_count"] == 3
    assert payload["stderr_count"] == 3
    assert payload["capability_stats_activated"] is False

    receipt = registry.get_evaluation_receipt(payload["receipt_id"])
    assert receipt is not None
    assert receipt["evaluator_id"] == "eleutherai.lm-evaluation-harness"
    assert receipt["evaluator_version"] == "0.4.9@deadbeef"
    assert receipt["conditions"]["samples_present"] is True
    assert "samples" not in receipt["conditions"]
    assert receipt["conditions"]["n_samples"]["hellaswag"]["effective"] == 10042
    assert receipt["conditions"]["metric_stderr"]["hellaswag"]["acc,none"] == 0.012
    assert {item["metric"] for item in receipt["measurements"]} == {
        "acc,none",
        "acc_norm,none",
    }
    assert get_stats_view(project, model.model_id).measured is False

    second = runner.invoke(
        app,
        [
            "eval",
            "import-lm-eval",
            str(result_path),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--harness-version",
            "0.4.9@deadbeef",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert second.exit_code == 0, second.output
    assert json.loads(second.stdout)["receipt_id"] == payload["receipt_id"]
    imported_events = [
        item
        for item in registry.read().history
        if item["kind"] == "EVALUATION_RECEIPT_IMPORTED"
    ]
    assert len(imported_events) == 1


def test_lm_eval_import_rejects_missing_metric_direction(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, model = setup_project(project)
    result_path = tmp_path / "bad.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {"task": {"accuracy,none": 0.5}},
                "versions": {"task": 1},
                "configs": {},
                "n-shot": {},
                "n-samples": {},
                "higher_is_better": {},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "import-lm-eval",
            str(result_path),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--harness-version",
            "0.4.9",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "EXTERNAL_EVALUATION_DIRECTION_UNKNOWN"
