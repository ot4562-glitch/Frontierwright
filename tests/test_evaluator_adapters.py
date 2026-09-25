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
                "samples": {"hellaswag": [{"doc_id": 0, "target": 1, "resps": [["private"]]}]},
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
        item for item in registry.read().history if item["kind"] == "EVALUATION_RECEIPT_IMPORTED"
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


def write_lighteval_result(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "config_general": {
                    "lighteval_sha": "203045a8431bc9b77245c9998e05fc54509ea07f",
                    "model_name": "gpt2",
                    "model_sha": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                },
                "results": {
                    "gsm8k|0": {
                        "em": 0.42,
                        "em_stderr": 0.02,
                        "maj@8": 0.51,
                        "maj@8_stderr": 0.03,
                    },
                    "all": {"em": 0.42, "maj@8": 0.51},
                },
                "versions": {"gsm8k|0": 0},
                "config_tasks": {
                    "lighteval|gsm8k": {
                        "name": "gsm8k",
                        "metric": [
                            {"metric_name": "em", "higher_is_better": True},
                            {"metric_name": "maj@8", "higher_is_better": True},
                        ],
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_cli_import_lighteval_preserves_exact_task_metric_evidence(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry, model = setup_project(project)
    result_path = write_lighteval_result(tmp_path / "lighteval.json")

    result = runner.invoke(
        app,
        [
            "eval",
            "import-lighteval",
            str(result_path),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--lighteval-version",
            "0.13.0",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["evaluator_id"] == "huggingface.lighteval"
    assert payload["evaluator_version"] == "0.13.0"
    assert payload["task_count"] == 1
    assert payload["measurement_count"] == 2
    assert payload["stderr_count"] == 2
    assert payload["capability_stats_activated"] is False

    receipt = registry.get_evaluation_receipt(payload["receipt_id"])
    assert receipt is not None
    assert receipt["conditions"]["reported_lighteval_sha"] == (
        "203045a8431bc9b77245c9998e05fc54509ea07f"
    )
    assert receipt["conditions"]["aggregate_all_row_imported"] is False
    assert receipt["conditions"]["metric_stderr"]["gsm8k|0"] == {
        "em": 0.02,
        "maj@8": 0.03,
    }
    assert {
        (item["task_id"], item["task_version"], item["metric"], item["higher_is_better"])
        for item in receipt["measurements"]
    } == {
        ("lighteval:gsm8k|0", "0", "em", True),
        ("lighteval:gsm8k|0", "0", "maj@8", True),
    }
    assert get_stats_view(project, model.model_id).measured is False


def test_lighteval_import_requires_direction_metadata(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, model = setup_project(project)
    result_path = tmp_path / "bad-lighteval.json"
    result_path.write_text(
        json.dumps(
            {
                "config_general": {"lighteval_sha": "abc"},
                "results": {"gsm8k|0": {"em": 0.5}},
                "versions": {"gsm8k|0": 0},
                "config_tasks": {
                    "lighteval|gsm8k": {
                        "name": "gsm8k",
                        "metric": [{"metric_name": "em"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "import-lighteval",
            str(result_path),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--lighteval-version",
            "0.13.0",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "EXTERNAL_EVALUATION_DIRECTION_UNKNOWN"


def write_generic_eval_manifest(
    path: Path,
    *,
    model_id: str,
    model_fingerprint: str,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluator": {
                    "id": "example.execution-code-eval",
                    "version": "2026-09-25@abc123",
                },
                "model_id": model_id,
                "model_fingerprint": model_fingerprint,
                "source": {
                    "suite": "private-code-eval",
                    "revision": "abc123",
                    "execution_boundary": "CONTROLLED_PRIVATE",
                },
                "conditions": {
                    "sandbox": "container-v3",
                    "timeout_seconds": 5,
                },
                "measurements": [
                    {
                        "task_id": "private-code-eval:python",
                        "task_version": "2026-09-25",
                        "metric": "pass_rate",
                        "value": 0.81,
                        "higher_is_better": True,
                        "unit": "fraction",
                        "stderr": 0.025,
                        "sample_count": 200,
                        "confidence_interval": [0.76, 0.86],
                        "confidence_level": 0.95,
                    },
                    {
                        "task_id": "private-code-eval:python",
                        "task_version": "2026-09-25",
                        "metric": "mean_runtime_seconds",
                        "value": 0.14,
                        "higher_is_better": False,
                        "unit": "seconds",
                        "sample_count": 200,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_cli_import_generic_manifest_preserves_explicit_uncertainty_and_direction(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, model = setup_project(project)
    manifest = write_generic_eval_manifest(
        tmp_path / "external-eval.json",
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
    )

    first = runner.invoke(
        app,
        [
            "eval",
            "import-manifest",
            str(manifest),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert first.exit_code == 0, first.output
    payload = json.loads(first.stdout)
    assert payload["evaluator_id"] == "example.execution-code-eval"
    assert payload["evaluator_version"] == "2026-09-25@abc123"
    assert payload["task_count"] == 1
    assert payload["measurement_count"] == 2
    assert payload["stderr_count"] == 1
    assert payload["capability_stats_activated"] is False

    receipt = registry.get_evaluation_receipt(payload["receipt_id"])
    assert receipt is not None
    assert receipt["conditions"]["source"]["execution_boundary"] == "CONTROLLED_PRIVATE"
    assert receipt["conditions"]["sandbox"] == "container-v3"
    measurements = {item["metric"]: item for item in receipt["measurements"]}
    assert measurements["pass_rate"]["higher_is_better"] is True
    assert measurements["mean_runtime_seconds"]["higher_is_better"] is False

    uncertainty = receipt["conditions"]["measurement_uncertainty"]
    pass_selector = json.dumps(
        {
            "task_id": "private-code-eval:python",
            "task_version": "2026-09-25",
            "metric": "pass_rate",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert uncertainty[pass_selector] == {
        "stderr": 0.025,
        "sample_count": 200,
        "confidence_interval": [0.76, 0.86],
        "confidence_level": 0.95,
    }
    assert receipt["conditions"]["measurement_units"][pass_selector] == "fraction"

    second = runner.invoke(
        app,
        [
            "eval",
            "import-manifest",
            str(manifest),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert second.exit_code == 0, second.output
    assert json.loads(second.stdout)["receipt_id"] == payload["receipt_id"]
    assert get_stats_view(project, model.model_id).measured is False


def test_generic_manifest_rejects_wrong_model_fingerprint(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, model = setup_project(project)
    manifest = write_generic_eval_manifest(
        tmp_path / "wrong-model.json",
        model_id=model.model_id,
        model_fingerprint="sha256:not-the-target-model",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "import-manifest",
            str(manifest),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "EXTERNAL_EVALUATION_MODEL_MISMATCH"


def test_generic_manifest_requires_explicit_metric_direction(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, model = setup_project(project)
    manifest = tmp_path / "missing-direction.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluator": {"id": "custom.eval", "version": "1"},
                "model_id": model.model_id,
                "model_fingerprint": model.fingerprint,
                "measurements": [
                    {
                        "task_id": "task",
                        "task_version": "1",
                        "metric": "score",
                        "value": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "import-manifest",
            str(manifest),
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "EXTERNAL_EVALUATION_DIRECTION_UNKNOWN"
