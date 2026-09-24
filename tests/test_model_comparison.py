import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.data import DatasetClassification
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.registry import Registry
from frontierwright.service import compare_models, ingest_stats, set_workload_profile
from frontierwright.workloads import WorkloadProfile

runner = CliRunner()


def _write_scale(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scale_id": "stock-descendant-scale",
                "scale_version": "1",
                "frozen": True,
                "axes": [
                    {
                        "axis": "GENERAL",
                        "display_anchor": 100,
                        "tasks": [
                            {
                                "task_id": "general-task",
                                "task_version": "1",
                                "metric": "accuracy",
                                "weight": 1,
                                "raw_anchor": 0.5,
                                "raw_unit": 0.1,
                                "display_per_unit": 10,
                                "higher_is_better": True,
                            }
                        ],
                    },
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
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_receipt(
    path: Path,
    model: ModelState,
    *,
    receipt_id: str,
    general: float,
    coding: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_id": receipt_id,
                "model_id": model.model_id,
                "model_fingerprint": model.fingerprint,
                "evaluator_id": "stock-descendant-fixture",
                "evaluator_version": "1",
                "conditions": {"same_workload": True},
                "measurements": [
                    {
                        "task_id": "general-task",
                        "task_version": "1",
                        "metric": "accuracy",
                        "value": general,
                        "higher_is_better": True,
                    },
                    {
                        "task_id": "coding-task",
                        "task_version": "1",
                        "metric": "pass_rate",
                        "value": coding,
                        "higher_is_better": True,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _setup_historical_stock_and_current_descendant(
    tmp_path: Path,
) -> tuple[Path, ModelState, ModelState]:
    project = tmp_path / "project"
    registry = Registry(project)
    registry.initialize("PersonalModel", ModelOrigin.IMPORTED_LOCAL)
    state = registry.read()
    identity_id = str(state.project["identity_id"])

    stock = ModelState(
        model_id="stock-open-model",
        identity_id=identity_id,
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="stock-checkpoint",
        fingerprint="sha256:stock-model",
    )
    registry.register_candidate(stock)
    registry.promote_candidate(stock.model_id)

    descendant = ModelState(
        model_id="my-custom-descendant",
        identity_id=identity_id,
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="descendant-checkpoint",
        fingerprint="sha256:custom-descendant",
        parent_model_id=stock.model_id,
    )
    registry.register_candidate(descendant)

    scale = _write_scale(tmp_path / "scale.json")
    ingest_stats(
        project,
        receipt_path=_write_receipt(
            tmp_path / "stock.json",
            stock,
            receipt_id="receipt-stock",
            general=0.8,
            coding=0.8,
        ),
        scale_path=scale,
    )
    ingest_stats(
        project,
        receipt_path=_write_receipt(
            tmp_path / "descendant.json",
            descendant,
            receipt_id="receipt-descendant",
            general=0.7,
            coding=0.9,
        ),
        scale_path=scale,
    )
    registry.promote_candidate(descendant.model_id)

    set_workload_profile(
        project,
        WorkloadProfile(
            name="My actual workload",
            privacy=DatasetClassification.PRIVATE,
            critical_floors={"coding": 135.0},
            utility_weights={
                "capability.general": 1.0,
                "capability.coding": 2.0,
            },
            utility_scales={
                "capability.general": 10.0,
                "capability.coding": 10.0,
            },
        ),
    )
    return project, stock, descendant


def test_compare_models_keeps_stock_baseline_after_descendant_promotion(tmp_path: Path) -> None:
    project, stock, descendant = _setup_historical_stock_and_current_descendant(tmp_path)

    view = compare_models(project, stock.model_id, descendant.model_id)

    assert view.baseline_model_id == stock.model_id
    assert view.contender_model_id == descendant.model_id
    assert view.direct_descendant is True
    assert view.scale_comparable is True
    assert view.baseline_stats["general"] == pytest.approx(130.0)
    assert view.contender_stats["general"] == pytest.approx(120.0)
    assert view.deltas["general"] == pytest.approx(-10.0)
    assert view.deltas["coding"] == pytest.approx(10.0)
    assert view.paired_capability_evidence["available"] is False

    workload = view.workload_comparison
    assert workload["configured"] is True
    improvements = workload["improvements"]
    assert isinstance(improvements, list)
    assert any(item["key"] == "capability.coding" for item in improvements)

    assert view.pareto["relation"] == "TRADEOFF"
    utility = view.pareto["explicit_user_utility"]
    assert isinstance(utility, dict)
    assert utility["status"] == "COMPLETE"
    assert utility["relation"] == "CANDIDATE_PREFERRED"
    assert utility["utility_delta"] == pytest.approx(1.0)


def test_compare_models_cli_json_is_candidate_status_independent(tmp_path: Path) -> None:
    project, stock, descendant = _setup_historical_stock_and_current_descendant(tmp_path)

    result = runner.invoke(
        app,
        [
            "workload",
            "compare-models",
            stock.model_id,
            descendant.model_id,
            "--path",
            str(project),
            "--json",
            "--non-interactive",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["baseline_model_id"] == stock.model_id
    assert payload["contender_model_id"] == descendant.model_id
    assert payload["direct_descendant"] is True
    assert payload["pareto"]["relation"] == "TRADEOFF"
    assert payload["pareto"]["explicit_user_utility"]["relation"] == "CANDIDATE_PREFERRED"
