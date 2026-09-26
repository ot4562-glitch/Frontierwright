from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.domain import ModelOrigin
from frontierwright.editions import EditionProfile
from frontierwright.fit_planner import OpportunityClass, plan_fit_opportunities
from frontierwright.registry import Registry
from frontierwright.service import get_fit_opportunities, initialize_project

runner = CliRunner()


def _constraint(key: str, status: str) -> dict[str, object]:
    return {
        "key": key,
        "category": "TEST",
        "requirement": 1,
        "observed": None,
        "status": status,
        "reason": "test evidence",
    }


def test_unknown_evidence_is_measured_before_intervention() -> None:
    plan = plan_fit_opportunities(
        edition=EditionProfile.STUDIO,
        model_id="model-1",
        workload_configured=True,
        constraints=[
            _constraint("capability.coding", "UNKNOWN"),
            _constraint("serving.latency_p50", "UNKNOWN"),
            _constraint("evaluation.workload_coverage", "UNKNOWN"),
        ],
        model_fit={},
        workload_evaluation_coverage={
            "complete": False,
            "missing": {
                "languages": ["ko"],
                "domains": ["economics"],
                "tasks": ["research"],
            },
        },
    )

    ids = [item.opportunity_id for item in plan.opportunities]
    assert ids == [
        "measure-capability",
        "profile-model-fit",
        "measure-workload-coverage",
    ]
    assert all(
        item.opportunity_class is OpportunityClass.BLOCKING_EVIDENCE for item in plan.opportunities
    )
    assert all(item.improvement_prediction is None for item in plan.opportunities)
    assert "languages=ko" in plan.opportunities[-1].reason
    assert "domains=economics" in plan.opportunities[-1].reason
    assert "tasks=research" in plan.opportunities[-1].reason




def test_missing_acceptance_maps_to_public_runnable_next_action() -> None:
    plan = plan_fit_opportunities(
        edition=EditionProfile.STUDIO,
        model_id="model-1",
        workload_configured=True,
        constraints=[_constraint("evaluation.workload_acceptance", "UNKNOWN")],
        model_fit={},
        workload_evaluation_coverage={"complete": True},
    )

    assert [item.opportunity_id for item in plan.opportunities] == [
        "configure-workload-acceptance"
    ]
    opportunity = plan.opportunities[0]
    assert opportunity.action == "frontierwright workload acceptance example --json"
    assert opportunity.evidence_keys == ("evaluation.workload_acceptance",)
    assert opportunity.argv_steps == (
        ("frontierwright", "workload", "acceptance", "example", "--json"),
        ("frontierwright", "workload", "acceptance", "create", "--help"),
    )


def test_existing_acceptance_contract_moves_next_action_to_assess() -> None:
    plan = plan_fit_opportunities(
        edition=EditionProfile.LAB,
        model_id="model-1",
        workload_configured=True,
        constraints=[_constraint("evaluation.workload_acceptance", "UNKNOWN")],
        model_fit={},
        workload_evaluation_coverage={"complete": True},
        workload_acceptance_status="NOT_ASSESSED",
    )

    assert [item.opportunity_id for item in plan.opportunities] == [
        "assess-workload-acceptance"
    ]
    opportunity = plan.opportunities[0]
    assert opportunity.action == "frontierwright workload acceptance assess --help"
    assert opportunity.argv_steps == (
        ("frontierwright", "workload", "acceptance", "assess", "--help"),
    )
    assert "already exists" in opportunity.reason


def test_inconclusive_fit_schedules_more_measurement_instead_of_stalling() -> None:
    plan = plan_fit_opportunities(
        edition=EditionProfile.STUDIO,
        model_id="model-1",
        workload_configured=True,
        constraints=[_constraint("evaluation.workload_acceptance", "INCONCLUSIVE")],
        model_fit={},
        workload_evaluation_coverage={"complete": True},
    )

    assert [item.opportunity_id for item in plan.opportunities] == [
        "reduce-decision-uncertainty"
    ]
    opportunity = plan.opportunities[0]
    assert opportunity.opportunity_class is OpportunityClass.BLOCKING_EVIDENCE
    assert "additional compatible" in opportunity.action
    assert "budget" in opportunity.success_criterion.lower()
    assert opportunity.improvement_prediction is None

def test_failed_fit_creates_falsifiable_training_and_efficiency_experiments() -> None:
    plan = plan_fit_opportunities(
        edition=EditionProfile.LAB,
        model_id="model-1",
        workload_configured=True,
        constraints=[
            _constraint("capability.math", "FAIL"),
            _constraint("serving.latency_p50", "FAIL"),
            _constraint("serving.throughput_p50", "FAIL"),
        ],
        model_fit={"latency_seconds_p50": 3.0, "tokens_per_second_p50": 5.0},
        workload_evaluation_coverage={"complete": True},
    )

    items = {item.opportunity_id: item for item in plan.opportunities}
    assert set(items) == {"specialize-capability-gap", "optimize-serving-gap"}
    assert items["specialize-capability-gap"].opportunity_class is OpportunityClass.FIT_GAP
    assert items["optimize-serving-gap"].opportunity_class is OpportunityClass.FIT_GAP
    assert "same" in items["specialize-capability-gap"].success_criterion.lower()
    assert "comparable" in items["optimize-serving-gap"].success_criterion.lower()
    assert all(item.improvement_prediction is None for item in items.values())


def test_passing_fit_plus_measured_vram_headroom_proposes_calibration_not_bigger_model_claim() -> (
    None
):
    plan = plan_fit_opportunities(
        edition=EditionProfile.STUDIO,
        model_id="model-1",
        workload_configured=True,
        constraints=[
            _constraint("capability.general", "PASS"),
            _constraint("serving.latency_p50", "PASS"),
        ],
        model_fit={"cuda_memory_free_min_sampled_bytes": 2_147_483_648},
        workload_evaluation_coverage={"complete": True},
    )

    assert len(plan.opportunities) == 1
    opportunity = plan.opportunities[0]
    assert opportunity.opportunity_id == "calibrate-headroom-experiment"
    assert opportunity.opportunity_class is OpportunityClass.HEADROOM_EXPERIMENT
    assert "not proof" in opportunity.reason
    assert "calibrat" in opportunity.action.lower()
    assert opportunity.improvement_prediction is None


def test_edition_changes_experience_copy_not_evidence_logic() -> None:
    common = dict(
        model_id="model-1",
        workload_configured=True,
        constraints=[_constraint("serving.latency_p50", "UNKNOWN")],
        model_fit={},
        workload_evaluation_coverage={"complete": True},
    )
    academy = plan_fit_opportunities(edition=EditionProfile.ACADEMY, **common)
    studio = plan_fit_opportunities(edition=EditionProfile.STUDIO, **common)
    lab = plan_fit_opportunities(edition=EditionProfile.LAB, **common)

    assert (
        [item.opportunity_id for item in academy.opportunities]
        == [item.opportunity_id for item in studio.opportunities]
        == [item.opportunity_id for item in lab.opportunities]
    )
    assert academy.opportunities[0].title != studio.opportunities[0].title
    assert studio.opportunities[0].title != lab.opportunities[0].title
    assert academy.opportunities[0].evidence_keys == studio.opportunities[0].evidence_keys
    assert studio.opportunities[0].evidence_keys == lab.opportunities[0].evidence_keys


def test_service_and_cli_expose_edition_specific_next_experiment_without_gain_prediction(
    tmp_path: Path,
) -> None:
    project = tmp_path / "academy"
    initialize_project(
        project,
        name="LEARNER",
        origin=ModelOrigin.ZERO,
        edition_profile=EditionProfile.ACADEMY,
    )
    service_plan = get_fit_opportunities(project)
    assert service_plan.edition is EditionProfile.ACADEMY
    assert service_plan.opportunities[0].opportunity_id == "define-workload"
    assert service_plan.opportunities[0].improvement_prediction is None

    result = runner.invoke(
        app,
        [
            "workload",
            "next",
            "--path",
            str(project),
            "--json",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["edition"] == "ACADEMY"
    assert payload["opportunities"][0]["opportunity_id"] == "define-workload"
    assert payload["opportunities"][0]["improvement_prediction"] is None


def test_uninitialized_cli_next_is_still_safe_and_nonpredictive(tmp_path: Path) -> None:
    project = tmp_path / "missing"
    result = runner.invoke(
        app,
        [
            "workload",
            "next",
            "--path",
            str(project),
            "--json",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["edition"] == "STUDIO"
    assert payload["opportunities"][0]["opportunity_id"] == "define-workload"
    assert payload["opportunities"][0]["improvement_prediction"] is None


def test_project_registry_edition_is_preserved_by_planner(tmp_path: Path) -> None:
    for edition, origin in (
        (EditionProfile.STUDIO, ModelOrigin.IMPORTED_LOCAL),
        (EditionProfile.ACADEMY, ModelOrigin.ZERO),
        (EditionProfile.LAB, ModelOrigin.INTERNAL_LAB),
    ):
        project = tmp_path / edition.value.lower()
        Registry(project).initialize("MODEL", origin, edition_profile=edition)
        plan = get_fit_opportunities(project)
        assert plan.edition is edition
