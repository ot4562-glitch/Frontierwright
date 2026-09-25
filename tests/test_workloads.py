from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.data import DatasetClassification
from frontierwright.domain import Axis, CapabilityStat, ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.registry import Registry
from frontierwright.service import get_workload_fit, get_workload_view, set_workload_profile
from frontierwright.workloads import (
    ExplicitUtilityRelation,
    ExplicitUtilityStatus,
    MeasurementResolutionState,
    ParetoDirection,
    ParetoMetricInput,
    ParetoRelation,
    WorkloadConstraintStatus,
    WorkloadDecisionClaimKind,
    WorkloadDecisionClaimStatus,
    WorkloadProfile,
    assess_workload_decision_claim,
    assess_workload_fit,
    compare_explicit_user_utility,
    compare_pareto_metrics,
    human_fit_status,
    workload_profile_from_payload,
)

runner = CliRunner()


def test_human_fit_status_turns_uncertainty_into_actionable_measurement_states() -> None:
    assert human_fit_status(WorkloadConstraintStatus.UNKNOWN) == "MEASUREMENT NEEDED"
    assert human_fit_status(WorkloadConstraintStatus.INCONCLUSIVE) == "MORE EVIDENCE NEEDED"
    assert human_fit_status(WorkloadConstraintStatus.PASS) == "PASS"
    assert human_fit_status(WorkloadConstraintStatus.FAIL) == "FAIL"

    profile = WorkloadProfile(
        name="Measured fit",
        privacy=DatasetClassification.PUBLIC,
        max_latency_seconds=1.0,
    )
    payload = assess_workload_fit(profile, capability_stats={}).to_payload()
    assert payload["overall_resolution_state"] == MeasurementResolutionState.MEASUREMENT_NEEDED
    assert payload["resolution_counts"][MeasurementResolutionState.MEASUREMENT_NEEDED.value] >= 1
    constraint = payload["constraints"][0]
    assert constraint["resolution_state"] == MeasurementResolutionState.MEASUREMENT_NEEDED



def test_workload_profile_v1_payload_remains_readable_without_invented_utility() -> None:
    profile = workload_profile_from_payload(
        {
            "schema_version": 1,
            "name": "Legacy workload",
            "languages": ["en"],
            "domains": [],
            "task_weights": {},
            "privacy": "PRIVATE",
            "critical_floors": {"general": 50},
        }
    )

    assert profile.schema_version == 1
    assert profile.utility_weights == {}
    assert profile.utility_scales == {}
    assert profile.to_payload()["schema_version"] == 1


def test_workload_profile_is_canonical_and_hash_stable() -> None:
    left = WorkloadProfile(
        name="Daily work",
        languages=("ko", "en", "KO"),
        domains=("Economics", "coding"),
        task_weights={"research": 2, "coding": 1.5},
        context_tokens_p50=2048,
        context_tokens_p95=8192,
        max_latency_seconds=2,
        min_tokens_per_second=15,
        privacy=DatasetClassification.PRIVATE,
        critical_floors={"general": 60, "coding": 55.5},
    )
    right = WorkloadProfile(
        name="  Daily work  ",
        languages=("ko", "en"),
        domains=("Economics", "coding"),
        task_weights={"coding": 1.5, "research": 2.0},
        context_tokens_p50=2048,
        context_tokens_p95=8192,
        max_latency_seconds=2.0,
        min_tokens_per_second=15.0,
        privacy=DatasetClassification.PRIVATE,
        critical_floors={"coding": 55.5, "general": 60.0},
    )

    assert left == right
    assert left.profile_hash == right.profile_hash
    assert left.profile_hash.startswith("sha256:")
    assert left.to_payload()["languages"] == ["ko", "en"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "name must be nonempty"),
        (
            {
                "name": "bad context",
                "context_tokens_p50": 4096,
                "context_tokens_p95": 1024,
            },
            "p95 cannot be smaller",
        ),
        (
            {"name": "bad speed", "min_tokens_per_second": 0},
            "must be finite and positive",
        ),
        (
            {"name": "bad floor", "critical_floors": {"vision": 10}},
            "unsupported critical floor axis",
        ),
    ],
)
def test_workload_profile_rejects_invalid_evidence(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(FrontierwrightError, match=message):
        WorkloadProfile(**kwargs)  # type: ignore[arg-type]


def test_workload_profile_persists_and_replays_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "project"
    Registry(root).initialize("NOVA", ModelOrigin.IMPORTED_LOCAL)
    profile = WorkloadProfile(
        name="My machine workload",
        languages=("en",),
        domains=("coding",),
        task_weights={"coding": 3.0},
        max_latency_seconds=1.5,
        min_tokens_per_second=20.0,
        critical_floors={"coding": 70.0},
    )

    first = set_workload_profile(root, profile)
    second = set_workload_profile(root, profile)

    assert first.configured is True
    assert first.profile_id == second.profile_id
    assert first.profile_hash == profile.profile_hash
    assert get_workload_view(root).profile == profile.to_payload()

    state = Registry(root).read()
    events = [item["kind"] for item in state.history if item["kind"].startswith("WORKLOAD_PROFILE")]
    assert events == ["WORKLOAD_PROFILE_SET"]


def test_cli_workload_set_and_show_json_contract(tmp_path: Path) -> None:
    project = tmp_path / "project"
    init = runner.invoke(
        app,
        [
            "project",
            "init",
            str(project),
            "--name",
            "PERSONAL",
            "--origin",
            "IMPORTED_LOCAL",
            "--edition",
            "STUDIO",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert init.exit_code == 0, init.output

    set_result = runner.invoke(
        app,
        [
            "workload",
            "set",
            "--path",
            str(project),
            "--name",
            "Korean economics and coding",
            "--language",
            "ko",
            "--language",
            "en",
            "--domain",
            "economics",
            "--domain",
            "coding",
            "--task",
            "research=2",
            "--task",
            "coding=1.5",
            "--context-p50",
            "2048",
            "--context-p95",
            "8192",
            "--max-latency",
            "2",
            "--min-tokens-per-second",
            "15",
            "--privacy",
            "PRIVATE",
            "--floor",
            "general=60",
            "--floor",
            "coding=55.5",
            "--utility-weight",
            "capability.coding=2",
            "--utility-scale",
            "capability.coding=10",
            "--utility-weight",
            "serving.latency_p50=1",
            "--utility-scale",
            "serving.latency_p50=0.25",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert set_result.exit_code == 0, set_result.output
    payload = json.loads(set_result.stdout)
    assert payload["ok"] is True
    assert payload["configured"] is True
    assert payload["profile"]["languages"] == ["ko", "en"]
    assert payload["profile"]["domains"] == ["economics", "coding"]
    assert payload["profile"]["task_weights"] == {"coding": 1.5, "research": 2.0}
    assert payload["profile"]["context_tokens_p50"] == 2048
    assert payload["profile"]["context_tokens_p95"] == 8192
    assert payload["profile"]["max_latency_seconds"] == 2.0
    assert payload["profile"]["min_tokens_per_second"] == 15.0
    assert payload["profile"]["privacy"] == "PRIVATE"
    assert payload["profile"]["critical_floors"] == {
        "coding": 55.5,
        "general": 60.0,
    }
    assert payload["profile"]["utility_weights"] == {
        "capability.coding": 2.0,
        "serving.latency_p50": 1.0,
    }
    assert payload["profile"]["utility_scales"] == {
        "capability.coding": 10.0,
        "serving.latency_p50": 0.25,
    }
    profile_hash = payload["profile_hash"]

    show_result = runner.invoke(
        app,
        [
            "workload",
            "show",
            "--path",
            str(project),
            "--json",
            "--non-interactive",
        ],
    )
    assert show_result.exit_code == 0, show_result.output
    shown = json.loads(show_result.stdout)
    assert shown["profile_hash"] == profile_hash
    assert shown["profile_name"] == "Korean economics and coding"


def test_cli_workload_rejects_invalid_privacy(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.IMPORTED_LOCAL)

    result = runner.invoke(
        app,
        [
            "workload",
            "set",
            "--path",
            str(project),
            "--name",
            "Bad",
            "--privacy",
            "SECRET",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_WORKLOAD_ARGUMENT"


def test_workload_fit_keeps_tradeoffs_auditable_without_single_score() -> None:
    profile = WorkloadProfile(
        name="Personal coding",
        languages=("ko",),
        domains=("coding",),
        task_weights={"coding": 2.0},
        context_tokens_p95=8192,
        max_latency_seconds=2.0,
        min_tokens_per_second=15.0,
        privacy=DatasetClassification.PRIVATE,
        critical_floors={"general": 60.0, "coding": 55.0},
    )
    assessment = assess_workload_fit(
        profile,
        capability_stats={
            "general": 75.0,
            "reasoning": 60.0,
            "math": 50.0,
            "coding": 50.0,
        },
        inference_metrics={
            "latency_seconds_p50": 1.2,
            "tokens_per_second_p50": 22.0,
        },
        supported_context_tokens=None,
        workload_eval_coverage=None,
        serving_boundary="LOCAL_MACHINE",
    )

    payload = assessment.to_payload()
    by_key = {item["key"]: item for item in payload["constraints"]}
    assert assessment.overall_status is WorkloadConstraintStatus.FAIL
    assert by_key["capability.general"]["status"] == "PASS"
    assert by_key["capability.coding"]["status"] == "FAIL"
    assert by_key["serving.latency_p50"]["status"] == "PASS"
    assert by_key["serving.throughput_p50"]["status"] == "PASS"
    assert by_key["serving.context_p95"]["status"] == "UNKNOWN"
    assert by_key["evaluation.workload_coverage"]["status"] == "UNKNOWN"
    assert by_key["privacy.serving_boundary"]["status"] == "PASS"
    assert payload["synthetic_utility_score"] is None


def test_workload_fit_unknown_when_only_generic_capability_is_insufficient() -> None:
    profile = WorkloadProfile(
        name="Economics research",
        languages=("ko",),
        domains=("economics",),
        task_weights={"research": 1.0},
        privacy=DatasetClassification.PUBLIC,
        critical_floors={"general": 50.0},
    )
    assessment = assess_workload_fit(
        profile,
        capability_stats={
            "general": 80.0,
            "reasoning": 80.0,
            "math": 80.0,
            "coding": 80.0,
        },
    )
    assert assessment.overall_status is WorkloadConstraintStatus.UNKNOWN
    assert [item.status for item in assessment.constraints] == [
        WorkloadConstraintStatus.PASS,
        WorkloadConstraintStatus.UNKNOWN,
        WorkloadConstraintStatus.UNKNOWN,
    ]
    assert assessment.constraints[-1].key == "evaluation.workload_acceptance"


def test_service_and_cli_workload_fit_use_current_champion(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry = Registry(project)
    registry.initialize("NOVA", ModelOrigin.IMPORTED_LOCAL)
    state = registry.read()
    champion = ModelState(
        model_id="champion",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="checkpoint",
        fingerprint="sha256:champion",
        stats=(
            CapabilityStat(
                Axis.GENERAL,
                65.0,
                "legacy-test-scale",
                "receipt-test",
            ),
        ),
    )
    registry.register_candidate(champion)
    registry.promote_candidate(champion.model_id)
    set_workload_profile(
        project,
        WorkloadProfile(
            name="Simple floor",
            privacy=DatasetClassification.PUBLIC,
            critical_floors={"general": 60.0},
        ),
    )

    view = get_workload_fit(project)
    assert view.model_id == "champion"
    assert view.overall_status == "PASS"
    assert view.synthetic_utility_score is None
    assert view.constraints[0]["status"] == "PASS"

    result = runner.invoke(
        app,
        [
            "workload",
            "fit",
            "--path",
            str(project),
            "--json",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["overall_status"] == "PASS"
    assert payload["model_id"] == "champion"
    assert payload["synthetic_utility_score"] is None


def test_pareto_comparison_preserves_tradeoffs_without_magic_score() -> None:
    comparison = compare_pareto_metrics(
        [
            ParetoMetricInput(
                key="capability.general",
                category="CAPABILITY",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=50.0,
                candidate_value=75.0,
                unit="stat",
                evidence_source="scale",
            ),
            ParetoMetricInput(
                key="capability.math",
                category="CAPABILITY",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=50.0,
                candidate_value=37.5,
                unit="stat",
                evidence_source="scale",
            ),
            ParetoMetricInput(
                key="serving.latency_p50",
                category="SERVING",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=0.4,
                candidate_value=0.3,
                unit="seconds",
                evidence_source="profile",
            ),
        ]
    )

    payload = comparison.to_payload()
    assert comparison.relation is ParetoRelation.TRADEOFF
    assert payload["synthetic_utility_score"] is None
    by_key = {item["key"]: item for item in payload["metrics"]}
    assert by_key["capability.general"]["relation"] == "BETTER"
    assert by_key["capability.math"]["relation"] == "WORSE"
    assert by_key["serving.latency_p50"]["relation"] == "BETTER"
    assert by_key["serving.latency_p50"]["improvement_delta"] == pytest.approx(0.1)


def test_pareto_comparison_refuses_to_infer_missing_evidence() -> None:
    comparison = compare_pareto_metrics(
        [
            ParetoMetricInput(
                key="resource.peak_vram",
                category="RESOURCE",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=None,
                candidate_value=5_000,
                unit="bytes",
            )
        ]
    )
    assert comparison.relation is ParetoRelation.INSUFFICIENT_EVIDENCE
    assert comparison.metrics[0].relation.value == "UNKNOWN"


def test_explicit_user_utility_requires_user_normalization_and_keeps_breakdown() -> None:
    profile = WorkloadProfile(
        name="Latency-sensitive reasoning",
        utility_weights={
            "capability.reasoning": 2.0,
            "serving.latency_p50": 1.0,
        },
        utility_scales={
            "capability.reasoning": 10.0,
            "serving.latency_p50": 0.1,
        },
    )
    pareto = compare_pareto_metrics(
        [
            ParetoMetricInput(
                key="capability.reasoning",
                category="CAPABILITY",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=50.0,
                candidate_value=60.0,
                unit="frontierwright-capability-v1",
                evidence_source="scale",
            ),
            ParetoMetricInput(
                key="serving.latency_p50",
                category="SERVING",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=0.4,
                candidate_value=0.3,
                unit="seconds",
                evidence_source="profile",
            ),
        ]
    )

    utility = compare_explicit_user_utility(profile, pareto)
    payload = utility.to_payload()

    assert utility.status is ExplicitUtilityStatus.COMPLETE
    assert utility.relation is ExplicitUtilityRelation.CANDIDATE_PREFERRED
    assert utility.utility_delta == pytest.approx(3.0)
    by_key = {item["metric_key"]: item for item in payload["contributions"]}
    assert by_key["capability.reasoning"]["normalized_improvement"] == pytest.approx(1.0)
    assert by_key["capability.reasoning"]["contribution"] == pytest.approx(2.0)
    assert by_key["serving.latency_p50"]["normalized_improvement"] == pytest.approx(1.0)
    assert by_key["serving.latency_p50"]["contribution"] == pytest.approx(1.0)


def test_explicit_user_utility_is_incomplete_if_any_weighted_metric_is_unmeasured() -> None:
    profile = WorkloadProfile(
        name="Memory-aware",
        utility_weights={"resource.peak_vram": 1.0},
        utility_scales={"resource.peak_vram": float(1024**3)},
    )
    pareto = compare_pareto_metrics(
        [
            ParetoMetricInput(
                key="resource.peak_vram",
                category="RESOURCE",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=None,
                candidate_value=5 * 1024**3,
                unit="bytes",
            )
        ]
    )

    utility = compare_explicit_user_utility(profile, pareto)

    assert utility.status is ExplicitUtilityStatus.INCOMPLETE
    assert utility.relation is ExplicitUtilityRelation.UNKNOWN
    assert utility.utility_delta is None
    assert utility.missing_metrics == ("resource.peak_vram",)


def test_workload_profile_rejects_partial_or_unsupported_utility_contract() -> None:
    with pytest.raises(FrontierwrightError, match="same metrics"):
        WorkloadProfile(
            name="Bad utility",
            utility_weights={"capability.general": 1.0},
        )
    with pytest.raises(FrontierwrightError, match="unsupported utility metric"):
        WorkloadProfile(
            name="Bad metric",
            utility_weights={"capability.vision": 1.0},
            utility_scales={"capability.vision": 10.0},
        )


def test_workload_context_requirement_uses_profiled_model_capacity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry = Registry(project)
    registry.initialize("NOVA", ModelOrigin.IMPORTED_LOCAL)
    state = registry.read()
    model = ModelState(
        model_id="context-model",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="checkpoint",
        fingerprint="sha256:context-model",
    )
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    registry.record_event(
        "INFERENCE_PROFILE_MEASURED",
        {
            "model_id": model.model_id,
            "model_fingerprint": model.fingerprint,
            "metrics": {
                "measurement_scope": "steady_state_generation_excludes_model_load",
                "execution_boundary": "LOCAL_MACHINE",
                "device": "cpu",
                "max_new_tokens": 16,
                "context_length": 128,
                "latency_seconds_p50": 0.2,
                "tokens_per_second_p50": 50.0,
            },
        },
    )

    set_workload_profile(
        project,
        WorkloadProfile(
            name="Fits context",
            privacy=DatasetClassification.PUBLIC,
            context_tokens_p95=64,
        ),
    )
    passing = get_workload_fit(project)
    by_key = {item["key"]: item for item in passing.constraints}
    assert by_key["serving.context_p95"]["status"] == "PASS"
    assert by_key["serving.context_p95"]["observed"] == 128

    set_workload_profile(
        project,
        WorkloadProfile(
            name="Too long",
            privacy=DatasetClassification.PUBLIC,
            context_tokens_p95=256,
        ),
    )
    failing = get_workload_fit(project)
    by_key = {item["key"]: item for item in failing.constraints}
    assert by_key["serving.context_p95"]["status"] == "FAIL"
    assert by_key["serving.context_p95"]["observed"] == 128


def test_decision_claim_blocks_missing_material_measurement_and_never_certifies_better() -> None:
    profile = WorkloadProfile(
        name="Measured decision",
        privacy=DatasetClassification.PUBLIC,
        utility_weights={
            "capability.general": 1.0,
            "serving.latency_p50": 1.0,
        },
        utility_scales={
            "capability.general": 10.0,
            "serving.latency_p50": 0.25,
        },
        improvement_margins={
            "capability.general": 2.0,
            "serving.latency_p50": 0.05,
        },
    )
    pareto = compare_pareto_metrics(
        [
            ParetoMetricInput(
                key="capability.general",
                category="CAPABILITY",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=50.0,
                candidate_value=60.0,
                unit="score",
                evidence_source="fixture",
            ),
            ParetoMetricInput(
                key="serving.latency_p50",
                category="SERVING",
                direction=ParetoDirection.LOWER_BETTER,
                champion_value=1.0,
                candidate_value=None,
                unit="seconds",
                evidence_source=None,
            ),
        ]
    )
    utility = compare_explicit_user_utility(profile, pareto)
    claim = assess_workload_decision_claim(
        profile,
        pareto,
        utility,
        mandatory_fit_eligible=True,
    )

    assert pareto.relation is ParetoRelation.CANDIDATE_DOMINATES
    pareto_payload = pareto.to_payload()
    assert pareto_payload["complete"] is False
    assert pareto_payload["unqualified_dominance_claim_eligible"] is False
    assert claim.status is WorkloadDecisionClaimStatus.INCOMPLETE
    assert claim.kind is WorkloadDecisionClaimKind.NONE
    assert claim.decision_eligible is False
    assert claim.missing_metrics == ("serving.latency_p50",)
    assert claim.to_payload()["better_for_workload_statement_allowed"] is False


def test_decision_claim_exposes_regression_even_when_explicit_tradeoff_prefers_candidate() -> None:
    profile = WorkloadProfile(
        name="Explicit tradeoff",
        privacy=DatasetClassification.PUBLIC,
        utility_weights={
            "capability.general": 3.0,
            "capability.coding": 1.0,
        },
        utility_scales={
            "capability.general": 10.0,
            "capability.coding": 10.0,
        },
        improvement_margins={
            "capability.general": 2.0,
            "capability.coding": 2.0,
        },
    )
    pareto = compare_pareto_metrics(
        [
            ParetoMetricInput(
                key="capability.general",
                category="CAPABILITY",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=50.0,
                candidate_value=60.0,
            ),
            ParetoMetricInput(
                key="capability.coding",
                category="CAPABILITY",
                direction=ParetoDirection.HIGHER_BETTER,
                champion_value=50.0,
                candidate_value=45.0,
            ),
        ]
    )
    utility = compare_explicit_user_utility(profile, pareto)
    claim = assess_workload_decision_claim(
        profile,
        pareto,
        utility,
        mandatory_fit_eligible=True,
    )

    assert utility.relation is ExplicitUtilityRelation.CANDIDATE_PREFERRED
    assert claim.status is WorkloadDecisionClaimStatus.ELIGIBLE
    assert claim.kind is WorkloadDecisionClaimKind.EXPLICIT_UTILITY_PREFERENCE
    assert claim.decision_eligible is True
    assert claim.regressions == ("capability.coding",)
    payload = claim.to_payload()
    assert payload["better_for_workload_statement_allowed"] is False
    assert payload["claim_scope"] == "LOCAL_DECISION_EVIDENCE"
