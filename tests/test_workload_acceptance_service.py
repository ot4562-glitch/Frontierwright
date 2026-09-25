from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.data import DatasetClassification
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.registry import Registry
from frontierwright.service import (
    assess_workload_acceptance_evidence,
    bind_workload_evaluation,
    get_workload_acceptance_view,
    get_workload_fit,
    import_external_evaluation_evidence,
    set_workload_acceptance_contract,
    set_workload_profile,
)
from frontierwright.workload_acceptance import (
    AcceptanceEvidenceRule,
    AcceptanceEvidenceRuleKind,
    AcceptanceOperator,
    WorkloadAcceptanceContractV1,
    WorkloadAcceptanceCriterion,
)
from frontierwright.workload_evaluations import WorkloadEvidenceSelector
from frontierwright.workloads import WorkloadProfile

runner = CliRunner()


def _project(root: Path) -> tuple[Registry, ModelState]:
    registry = Registry(root)
    registry.initialize("FITMODEL", ModelOrigin.IMPORTED_LOCAL)
    state = registry.read()
    model = ModelState(
        model_id="model-workload-acceptance",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="synthetic-checkpoint",
        fingerprint="sha256:workload-acceptance-model",
    )
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    set_workload_profile(
        root,
        WorkloadProfile(
            name="Private research workload",
            task_weights={"research": 1.0},
            privacy=DatasetClassification.PUBLIC,
        ),
    )
    return registry, model


def _evaluation_manifest(path: Path, *, model: ModelState) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluator": {"id": "private-research-eval", "version": "v1"},
                "model_id": model.model_id,
                "model_fingerprint": model.fingerprint,
                "conditions": {"split": "heldout", "prompt_policy": "fixed-v1"},
                "measurements": [
                    {
                        "task_id": "private-research",
                        "task_version": "v1",
                        "metric": "success_rate",
                        "value": 0.85,
                        "higher_is_better": True,
                        "unit": "fraction",
                        "sample_count": 400,
                        "confidence_interval": [0.81, 0.88],
                        "confidence_level": 0.95,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _coverage_manifest(
    path: Path,
    *,
    profile_hash: str,
    receipt_id: str,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workload_profile_hash": profile_hash,
                "receipt_id": receipt_id,
                "coverage": {
                    "tasks": {
                        "research": [
                            {
                                "task_id": "private-research",
                                "task_version": "v1",
                                "metric": "success_rate",
                            }
                        ]
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _contract(profile_hash: str, *, threshold: float = 0.80) -> WorkloadAcceptanceContractV1:
    return WorkloadAcceptanceContractV1(
        name="Research success v1",
        workload_profile_hash=profile_hash,
        criteria=(
            WorkloadAcceptanceCriterion(
                criterion_id="research-success",
                selector=WorkloadEvidenceSelector(
                    task_id="private-research",
                    task_version="v1",
                    metric="success_rate",
                ),
                unit="fraction",
                operator=AcceptanceOperator.AT_LEAST,
                threshold=threshold,
                evidence_rule=AcceptanceEvidenceRule(
                    AcceptanceEvidenceRuleKind.EXPLICIT_INTERVAL,
                    confidence_level=0.95,
                ),
                evaluator_id="private-research-eval",
                evaluator_version="v1",
                required_conditions={
                    "split": "heldout",
                    "prompt_policy": "fixed-v1",
                },
            ),
        ),
    )


def test_exact_acceptance_contract_closes_workload_fit_without_reinterpreting_history(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, model = _project(project)
    stored_workload = registry.get_active_workload_profile()
    assert stored_workload is not None
    profile_hash = str(stored_workload["profile_hash"])

    imported = import_external_evaluation_evidence(
        project,
        manifest_path=_evaluation_manifest(tmp_path / "eval.json", model=model),
        model_id=model.model_id,
    )
    bind_workload_evaluation(
        project,
        manifest_path=_coverage_manifest(
            tmp_path / "coverage.json",
            profile_hash=profile_hash,
            receipt_id=imported.receipt.receipt_id,
        ),
        model_id=model.model_id,
    )

    coverage_only = get_workload_fit(project, model.model_id)
    assert coverage_only.overall_status == "UNKNOWN"
    assert coverage_only.workload_acceptance["configured"] is False

    first_contract = _contract(profile_hash, threshold=0.80)
    configured = set_workload_acceptance_contract(project, first_contract)
    assert configured.contract_id == first_contract.contract_id
    assert configured.overall_status == "NOT_ASSESSED"
    assert get_workload_fit(project, model.model_id).overall_status == "UNKNOWN"

    assessed = assess_workload_acceptance_evidence(
        project,
        contract_ref=first_contract.contract_id,
        model_id=model.model_id,
        receipt_ids=(imported.receipt.receipt_id,),
    )
    assert assessed.overall_status == "PASS"
    assert assessed.assessment_criteria[0]["confidence_interval"] == [0.81, 0.88]

    fit = get_workload_fit(project, model.model_id)
    assert fit.overall_status == "PASS"
    acceptance_constraint = next(
        item for item in fit.constraints if item["key"] == "evaluation.workload_acceptance"
    )
    assert acceptance_constraint["status"] == "PASS"

    stricter = _contract(profile_hash, threshold=0.90)
    set_workload_acceptance_contract(project, stricter)
    active = get_workload_acceptance_view(project, model_id=model.model_id)
    assert active.contract_id == stricter.contract_id
    assert active.overall_status == "NOT_ASSESSED"
    assert get_workload_fit(project, model.model_id).overall_status == "UNKNOWN"

    failed = assess_workload_acceptance_evidence(
        project,
        contract_ref=stricter.contract_id,
        model_id=model.model_id,
        receipt_ids=(imported.receipt.receipt_id,),
    )
    assert failed.overall_status == "FAIL"
    assert get_workload_fit(project, model.model_id).overall_status == "FAIL"

    historical = get_workload_acceptance_view(
        project,
        first_contract.contract_id,
        model.model_id,
    )
    assert historical.overall_status == "PASS"
    assert historical.assessment_hash != failed.assessment_hash


def test_legacy_workload_never_gets_implicit_acceptance_thresholds(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, model = _project(project)

    view = get_workload_acceptance_view(project, model_id=model.model_id)
    fit = get_workload_fit(project, model.model_id)

    assert view.configured is False
    assert "not configured" in (view.note or "").lower()
    assert fit.overall_status == "UNKNOWN"
    acceptance_constraint = next(
        item for item in fit.constraints if item["key"] == "evaluation.workload_acceptance"
    )
    assert acceptance_constraint["status"] == "UNKNOWN"
    assert acceptance_constraint["observed"] is None


def test_workload_acceptance_cli_create_show_assess_json_contract(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry, model = _project(project)
    stored_workload = registry.get_active_workload_profile()
    assert stored_workload is not None
    profile_hash = str(stored_workload["profile_hash"])

    imported = import_external_evaluation_evidence(
        project,
        manifest_path=_evaluation_manifest(tmp_path / "cli-eval.json", model=model),
        model_id=model.model_id,
    )
    bind_workload_evaluation(
        project,
        manifest_path=_coverage_manifest(
            tmp_path / "cli-coverage.json",
            profile_hash=profile_hash,
            receipt_id=imported.receipt.receipt_id,
        ),
        model_id=model.model_id,
    )
    contract = _contract(profile_hash)
    contract_path = tmp_path / "acceptance.json"
    contract_path.write_text(
        json.dumps(contract.to_payload(), sort_keys=True),
        encoding="utf-8",
    )

    created = runner.invoke(
        app,
        [
            "workload",
            "acceptance",
            "create",
            "--file",
            str(contract_path),
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert created.exit_code == 0, created.output
    created_payload = json.loads(created.stdout)
    assert created_payload["configured"] is True
    assert created_payload["contract_id"] == contract.contract_id
    assert created_payload["overall_status"] == "NOT_ASSESSED"

    shown = runner.invoke(
        app,
        [
            "workload",
            "acceptance",
            "show",
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--json",
            "--non-interactive",
        ],
    )
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.stdout)["contract_hash"] == contract.contract_hash

    assessed = runner.invoke(
        app,
        [
            "workload",
            "acceptance",
            "assess",
            contract.contract_id,
            "--evidence",
            imported.receipt.receipt_id,
            "--model",
            model.model_id,
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert assessed.exit_code == 0, assessed.output
    assessed_payload = json.loads(assessed.stdout)
    assert assessed_payload["overall_status"] == "PASS"
    assert assessed_payload["assessment_criteria"][0]["status"] == "PASS"
    assert get_workload_fit(project, model.model_id).overall_status == "PASS"
