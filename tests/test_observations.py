import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.data import DatasetClassification
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.observations import ObservationOutcome, summarize_observations
from frontierwright.registry import Registry
from frontierwright.service import (
    get_fit_opportunities,
    get_usage_observation_summary,
    record_usage_observation,
    set_workload_profile,
)
from frontierwright.workloads import WorkloadProfile

runner = CliRunner()


def _project_with_champion(tmp_path: Path) -> tuple[Path, ModelState]:
    project = tmp_path / "project"
    registry = Registry(project)
    registry.initialize("OBS", ModelOrigin.IMPORTED_LOCAL)
    state = registry.read()
    model = ModelState(
        model_id="model-observed",
        identity_id=str(state.project["identity_id"]),
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="synthetic-observation-model",
        fingerprint="sha256:observed-model",
    )
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    return project, model


def test_usage_observation_inherits_workload_privacy_and_never_stores_content(
    tmp_path: Path,
) -> None:
    project, model = _project_with_champion(tmp_path)
    workload = set_workload_profile(
        project,
        WorkloadProfile(
            name="Private coding work",
            task_weights={"coding": 2.0},
            privacy=DatasetClassification.CONFIDENTIAL,
        ),
    )

    view = record_usage_observation(
        project,
        task="code-review",
        outcome=ObservationOutcome.CORRECTED,
        domain="python",
        language="en",
        failure_category="missed-edge-case",
        latency_seconds=1.25,
    )

    assert view.replayed is False
    payload = view.observation
    assert payload["model_id"] == model.model_id
    assert payload["model_fingerprint"] == model.fingerprint
    assert payload["workload_profile_hash"] == workload.profile_hash
    assert payload["classification"] == "CONFIDENTIAL"
    assert payload["content_stored"] is False
    assert "prompt" not in payload
    assert "response" not in payload


def test_usage_observation_idempotency_is_explicit_and_repeated_real_use_is_not_deduped(
    tmp_path: Path,
) -> None:
    project, _ = _project_with_champion(tmp_path)

    first = record_usage_observation(
        project,
        task="summarize",
        outcome=ObservationOutcome.FAILURE,
        failure_category="hallucinated-fact",
        idempotency_key="external-event-42",
    )
    replay = record_usage_observation(
        project,
        task="summarize",
        outcome=ObservationOutcome.FAILURE,
        failure_category="hallucinated-fact",
        idempotency_key="external-event-42",
    )
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.observation["observation_id"] == first.observation["observation_id"]

    with pytest.raises(FrontierwrightError) as exc_info:
        record_usage_observation(
            project,
            task="summarize",
            outcome=ObservationOutcome.SUCCESS,
            idempotency_key="external-event-42",
        )
    assert exc_info.value.code == "USAGE_OBSERVATION_IDEMPOTENCY_CONFLICT"

    distinct_a = record_usage_observation(
        project,
        task="summarize",
        outcome=ObservationOutcome.FAILURE,
        failure_category="hallucinated-fact",
    )
    distinct_b = record_usage_observation(
        project,
        task="summarize",
        outcome=ObservationOutcome.FAILURE,
        failure_category="hallucinated-fact",
    )
    assert distinct_a.observation["observation_id"] != distinct_b.observation["observation_id"]

    summary = get_usage_observation_summary(project)
    assert summary.total == 3
    assert summary.failures == 3


def test_usage_summary_separates_direct_success_correction_failure_and_abstention(
    tmp_path: Path,
) -> None:
    project, model = _project_with_champion(tmp_path)
    for outcome, task, category in (
        (ObservationOutcome.SUCCESS, "coding", None),
        (ObservationOutcome.FAILURE, "coding", "wrong-api"),
        (ObservationOutcome.CORRECTED, "coding", "wrong-api"),
        (ObservationOutcome.ABSTAINED, "research", "insufficient-evidence"),
    ):
        record_usage_observation(
            project,
            task=task,
            outcome=outcome,
            failure_category=category,
        )

    summary = get_usage_observation_summary(project, model.model_id)
    assert summary.total == 4
    assert summary.direct_successes == 1
    assert summary.failures == 1
    assert summary.corrected == 1
    assert summary.abstained == 1
    assert summary.evaluable == 3
    assert summary.direct_success_rate == pytest.approx(1 / 3)
    assert summary.direct_success_rate_ci95 is not None
    lo, hi = summary.direct_success_rate_ci95
    assert 0.0 <= lo < 1 / 3 < hi <= 1.0
    assert summary.by_task[0].task == "coding"
    assert summary.failure_categories == {"wrong-api": 2, "insufficient-evidence": 1}


def test_observed_failures_become_next_experiment_evidence_not_fake_rl_reward(
    tmp_path: Path,
) -> None:
    project, model = _project_with_champion(tmp_path)
    set_workload_profile(
        project,
        WorkloadProfile(
            name="My use",
            task_weights={"coding": 1.0},
            privacy=DatasetClassification.PRIVATE,
        ),
    )
    record_usage_observation(
        project,
        task="coding",
        outcome=ObservationOutcome.FAILURE,
        failure_category="tool-selection",
    )
    record_usage_observation(
        project,
        task="coding",
        outcome=ObservationOutcome.CORRECTED,
        failure_category="tool-selection",
    )

    plan = get_fit_opportunities(project, model.model_id)
    observed = [
        item for item in plan.opportunities if item.opportunity_id == "learn-from-observed-failures"
    ]
    assert len(observed) == 1
    item = observed[0]
    assert item.opportunity_class.value == "CONTINUAL_IMPROVEMENT"
    assert "1 failed and 1 user-corrected" in item.reason
    assert "not RL rewards" in item.reason
    assert item.improvement_prediction is None


def test_observe_cli_json_contract(tmp_path: Path) -> None:
    project, model = _project_with_champion(tmp_path)
    result = runner.invoke(
        app,
        [
            "observe",
            "record",
            "translation",
            "--outcome",
            "FAILURE",
            "--failure-category",
            "terminology",
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--idempotency-key",
            "use-001",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["observation"]["outcome"] == "FAILURE"
    assert payload["observation"]["content_stored"] is False

    replay = runner.invoke(
        app,
        [
            "observe",
            "record",
            "translation",
            "--outcome",
            "FAILURE",
            "--failure-category",
            "terminology",
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--idempotency-key",
            "use-001",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.stdout)["replayed"] is True

    summary = runner.invoke(
        app,
        [
            "observe",
            "summary",
            "--path",
            str(project),
            "--model",
            model.model_id,
            "--json",
            "--non-interactive",
        ],
    )
    assert summary.exit_code == 0, summary.output
    summary_payload = json.loads(summary.stdout)
    assert summary_payload["total"] == 1
    assert summary_payload["failures"] == 1
    assert "RL rewards" in summary_payload["note"]


def test_pure_summary_handles_empty_observations() -> None:
    summary = summarize_observations([], model_id="model-x")
    assert summary.total == 0
    assert summary.direct_success_rate is None
    assert summary.direct_success_rate_ci95 is None
