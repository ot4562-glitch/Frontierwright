import json
from pathlib import Path

import pytest

from frontierwright.capability_v1 import (
    CAPABILITY_V1_BUNDLE_ID,
    CAPABILITY_V1_BUNDLE_VERSION,
    CAPABILITY_V1_EVALUATOR_ID,
    CAPABILITY_V1_EVALUATOR_VERSION,
    CAPABILITY_V1_SCALE,
    CAPABILITY_V1_TASKS,
    capability_v1_bundle_hash,
)
from frontierwright.data import DatasetClassification
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import EvaluationReceipt, RawMeasurement, apply_scale
from frontierwright.registry import Registry
from frontierwright.service import (
    compare_candidate,
    get_candidates_view,
    get_history_view,
    get_status,
    ingest_stats,
    promote_candidate,
    reject_candidate,
    set_build_targets,
    set_workload_profile,
)
from frontierwright.workloads import WorkloadProfile


def create_project(root: Path) -> tuple[Registry, ModelState, ModelState]:
    registry = Registry(root)
    registry.initialize("NOVA", ModelOrigin.ZERO)
    state = registry.read()

    champion = ModelState(
        model_id="champion",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint="champion-checkpoint",
        fingerprint="sha256:champion",
    )
    registry.register_candidate(champion)
    registry.promote_candidate(champion.model_id)

    candidate = ModelState(
        model_id="candidate",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint="candidate-checkpoint",
        fingerprint="sha256:candidate",
        parent_model_id=champion.model_id,
    )
    registry.register_candidate(candidate)
    return registry, champion, candidate


def write_scale(path: Path, *, version: str = "1", coding_step: float = 10.0) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scale_id": "comparison-scale",
                "scale_version": version,
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
                                "display_per_unit": coding_step,
                                "higher_is_better": True,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_receipt(
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
                "evaluator_id": "candidate-fixture",
                "evaluator_version": "1",
                "conditions": {"seed": 1},
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
            }
        ),
        encoding="utf-8",
    )
    return path


def measure_pair(
    root: Path,
    champion: ModelState,
    candidate: ModelState,
    *,
    candidate_scale: Path | None = None,
) -> None:
    scale = write_scale(root / "scale.json")
    ingest_stats(
        root / "project",
        receipt_path=write_receipt(
            root / "champion-receipt.json",
            champion,
            receipt_id="champion-receipt",
            general=0.8,
            coding=0.8,
        ),
        scale_path=scale,
    )
    ingest_stats(
        root / "project",
        receipt_path=write_receipt(
            root / "candidate-receipt.json",
            candidate,
            receipt_id="candidate-receipt",
            general=0.7,
            coding=0.9,
        ),
        scale_path=candidate_scale or scale,
    )


def test_compare_uses_same_scale_and_checks_build_constraints(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, champion, candidate = create_project(project)
    measure_pair(tmp_path, champion, candidate)

    set_build_targets(
        project,
        targets={"coding": 145},
        floors={"general": 125},
    )
    view = compare_candidate(project, candidate.model_id)

    assert view.scale_comparable is True
    assert view.champion_stats["general"] == pytest.approx(130)
    assert view.candidate_stats["general"] == pytest.approx(120)
    assert view.deltas["general"] == pytest.approx(-10)
    assert view.deltas["coding"] == pytest.approx(10)

    floor = next(
        item
        for item in view.build_constraints
        if item["kind"] == "FLOOR" and item["axis"] == "general"
    )
    target = next(
        item
        for item in view.build_constraints
        if item["kind"] == "TARGET" and item["axis"] == "coding"
    )
    assert floor["status"] == "FAIL"
    assert target["status"] == "NOT_REACHED"
    assert view.promotion_eligible is False
    assert any(
        item["code"] == "BUILD_FLOOR_VIOLATION"
        and item["axis"] == "general"
        and item["override"] == "--allow-build-violations"
        for item in view.promotion_blockers
    )
    assert view.build_scale_hash is not None


def test_build_floor_blocks_promotion_until_explicit_override(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry, champion, candidate = create_project(project)
    measure_pair(tmp_path, champion, candidate)
    set_build_targets(
        project,
        targets={"coding": 145},
        floors={"general": 125},
    )

    with pytest.raises(FrontierwrightError, match="below the configured build floor"):
        promote_candidate(project, candidate.model_id)

    promoted = promote_candidate(
        project,
        candidate.model_id,
        allow_build_violations=True,
    )
    assert promoted.champion_model_id == candidate.model_id

    event = next(
        item
        for item in reversed(get_history_view(project).events)
        if item["kind"] == "CANDIDATE_PROMOTED"
    )
    assert event["details"]["gate_default_eligible"] is False
    assert event["details"]["allow_build_violations"] is True
    assert event["details"]["overridden_blockers"] == [
        {"axis": "general", "code": "BUILD_FLOOR_VIOLATION"}
    ]
    assert registry.read().project["champion_id"] == candidate.model_id


def test_promotion_transaction_rejects_build_state_change_after_gate(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, champion, candidate = create_project(project)
    measure_pair(tmp_path, champion, candidate)

    champion_profile = registry.get_active_capability_profile(champion.model_id)
    candidate_profile = registry.get_active_capability_profile(candidate.model_id)
    assert champion_profile is not None
    assert candidate_profile is not None

    expected_state = {
        "champion_id": champion.model_id,
        "build_updated_at": None,
        "champion_profile_id": str(champion_profile["profile_id"]),
        "candidate_profile_id": str(candidate_profile["profile_id"]),
    }

    set_build_targets(
        project,
        targets={"coding": 145},
        floors={"general": 100},
    )

    with pytest.raises(FrontierwrightError, match="changed after the promotion gate"):
        registry.promote_candidate(
            candidate.model_id,
            expected_state=expected_state,
        )


def test_different_scale_never_produces_fake_delta_or_default_promotion(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _, champion, candidate = create_project(project)
    different = write_scale(
        tmp_path / "candidate-scale.json",
        version="2",
        coding_step=12,
    )
    measure_pair(tmp_path, champion, candidate, candidate_scale=different)

    view = compare_candidate(project, candidate.model_id)
    assert view.scale_comparable is False
    assert all(value is None for value in view.deltas.values())

    with pytest.raises(FrontierwrightError, match="same frozen capability scale"):
        promote_candidate(project, candidate.model_id)


def test_unmeasured_candidate_requires_explicit_override(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry, _, candidate = create_project(project)

    with pytest.raises(FrontierwrightError, match="evaluated before promotion"):
        promote_candidate(project, candidate.model_id)

    status = promote_candidate(
        project,
        candidate.model_id,
        allow_unmeasured=True,
    )
    assert status.champion_model_id == candidate.model_id
    assert registry.read().project["champion_id"] == candidate.model_id


def test_promoted_candidate_becomes_current_state_even_when_one_axis_is_lower(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, champion, candidate = create_project(project)
    measure_pair(tmp_path, champion, candidate)

    promoted = promote_candidate(project, candidate.model_id)
    assert promoted.champion_model_id == candidate.model_id
    assert promoted.stats["general"] == pytest.approx(120)
    assert promoted.stats["coding"] == pytest.approx(140)

    history = get_history_view(project)
    assert any(
        event["kind"] == "CANDIDATE_PROMOTED" and event["details"]["model_id"] == candidate.model_id
        for event in history.events
    )
    assert registry.read().project["champion_id"] == candidate.model_id


def test_rejected_candidate_stays_in_history_and_cannot_be_promoted(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _, _, candidate = create_project(project)

    champion_before = get_status(project).champion_model_id
    view = reject_candidate(project, candidate.model_id)
    item = next(item for item in view.candidates if item["model_id"] == candidate.model_id)
    assert item["status"] == "REJECTED"
    assert get_status(project).champion_model_id == champion_before

    with pytest.raises(FrontierwrightError, match="Only PENDING"):
        promote_candidate(
            project,
            candidate.model_id,
            allow_unmeasured=True,
        )

    candidates = get_candidates_view(project)
    assert any(
        item["model_id"] == candidate.model_id and item["status"] == "REJECTED"
        for item in candidates.candidates
    )
    history = get_history_view(project)
    assert any(event["kind"] == "CANDIDATE_REJECTED" for event in history.events)


def test_champion_status_surface_reads_promoted_candidate_profile(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, champion, candidate = create_project(project)
    measure_pair(tmp_path, champion, candidate)
    promote_candidate(project, candidate.model_id)

    status = get_status(project)
    assert status.champion_model_id == candidate.model_id
    assert status.stats["general"] == pytest.approx(120)
    assert status.stats["coding"] == pytest.approx(140)


def test_compare_includes_workload_regressions_and_measured_pareto_tradeoffs(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, champion, candidate = create_project(project)
    measure_pair(tmp_path, champion, candidate)
    set_workload_profile(
        project,
        WorkloadProfile(
            name="Latency-sensitive coding",
            privacy=DatasetClassification.PUBLIC,
            max_latency_seconds=0.35,
            min_tokens_per_second=22.0,
            critical_floors={"general": 125.0},
            utility_weights={
                "capability.general": 1.0,
                "capability.coding": 2.0,
                "serving.latency_p50": 1.0,
                "resource.peak_vram": 0.5,
            },
            utility_scales={
                "capability.general": 10.0,
                "capability.coding": 10.0,
                "serving.latency_p50": 0.1,
                "resource.peak_vram": 1000.0,
            },
        ),
    )

    common = {
        "measurement_scope": "steady_state_generation_excludes_model_load",
        "execution_boundary": "LOCAL_MACHINE",
        "device": "cuda",
        "max_new_tokens": 16,
        "measured_runs": 3,
    }
    registry.record_event(
        "INFERENCE_PROFILE_MEASURED",
        {
            "model_id": champion.model_id,
            "model_fingerprint": champion.fingerprint,
            "metrics": {
                **common,
                "latency_seconds_p50": 0.4,
                "tokens_per_second_p50": 20.0,
                "peak_vram_bytes": 6_000,
                "max_sampled_process_rss_bytes": 3_000,
            },
        },
    )
    registry.record_event(
        "INFERENCE_PROFILE_MEASURED",
        {
            "model_id": candidate.model_id,
            "model_fingerprint": candidate.fingerprint,
            "metrics": {
                **common,
                "latency_seconds_p50": 0.3,
                "tokens_per_second_p50": 24.0,
                "peak_vram_bytes": 7_000,
                "max_sampled_process_rss_bytes": 2_800,
            },
        },
    )

    view = compare_candidate(project, candidate.model_id)

    assert view.workload_comparison["configured"] is True
    regressions = view.workload_comparison["regressions"]
    assert isinstance(regressions, list)
    assert any(
        item["key"] == "capability.general"
        and item["champion_status"] == "PASS"
        and item["candidate_status"] == "FAIL"
        for item in regressions
    )
    assert view.pareto["relation"] == "TRADEOFF"
    assert view.pareto["synthetic_utility_score"] is None
    assert view.pareto["inference_profile_comparable"] is True
    metrics = {item["key"]: item for item in view.pareto["metrics"]}
    assert metrics["capability.general"]["relation"] == "WORSE"
    assert metrics["capability.coding"]["relation"] == "BETTER"
    assert metrics["serving.latency_p50"]["relation"] == "BETTER"
    assert metrics["serving.throughput_p50"]["relation"] == "BETTER"
    assert metrics["resource.peak_vram"]["relation"] == "WORSE"
    utility = view.pareto["explicit_user_utility"]
    assert utility["status"] == "COMPLETE"
    assert utility["relation"] == "CANDIDATE_PREFERRED"
    assert utility["utility_delta"] == pytest.approx(1.5)
    contributions = {item["metric_key"]: item for item in utility["contributions"]}
    assert contributions["capability.general"]["contribution"] == pytest.approx(-1.0)
    assert contributions["capability.coding"]["contribution"] == pytest.approx(2.0)
    assert contributions["serving.latency_p50"]["contribution"] == pytest.approx(1.0)
    assert contributions["resource.peak_vram"]["contribution"] == pytest.approx(-0.5)


def activate_capability_fixture(
    registry: Registry,
    model: ModelState,
    *,
    correct_by_axis: dict[str, int],
    receipt_id: str,
) -> None:
    item_results: list[dict[str, object]] = []
    measurements: list[RawMeasurement] = []
    for axis in ("general", "reasoning", "math", "coding"):
        tasks = [task for task in CAPABILITY_V1_TASKS if task.axis.value.lower() == axis]
        correct = correct_by_axis[axis]
        for index, task in enumerate(tasks):
            item_results.append(
                {
                    "item_id": task.item_id,
                    "axis": axis,
                    "correct": index < correct,
                    "correct_margin_nats": 0.1 if index < correct else -0.1,
                }
            )
        measurements.append(
            RawMeasurement(
                task_id=f"{CAPABILITY_V1_BUNDLE_ID}.{axis}",
                task_version=CAPABILITY_V1_BUNDLE_VERSION,
                metric="accuracy",
                value=correct / len(tasks),
                higher_is_better=True,
            )
        )
    receipt = EvaluationReceipt(
        receipt_id=receipt_id,
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        evaluator_id=CAPABILITY_V1_EVALUATOR_ID,
        evaluator_version=CAPABILITY_V1_EVALUATOR_VERSION,
        conditions={
            "bundle_hash": capability_v1_bundle_hash(),
            "evidence_schema_version": 2,
            "item_results": item_results,
        },
        measurements=tuple(measurements),
    )
    registry.activate_capability_profile(
        receipt,
        CAPABILITY_V1_SCALE,
        apply_scale(receipt, CAPABILITY_V1_SCALE),
    )


def test_compare_exposes_same_item_paired_capability_flips(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry, champion, candidate = create_project(project)
    activate_capability_fixture(
        registry,
        champion,
        correct_by_axis={"general": 8, "reasoning": 8, "math": 8, "coding": 8},
        receipt_id="cap-champion-v2",
    )
    activate_capability_fixture(
        registry,
        candidate,
        correct_by_axis={"general": 12, "reasoning": 10, "math": 6, "coding": 8},
        receipt_id="cap-candidate-v2",
    )

    view = compare_candidate(project, candidate.model_id)

    assert view.scale_comparable is True
    paired = view.paired_capability_evidence
    assert paired["available"] is True
    axes = paired["axes"]
    assert isinstance(axes, dict)
    general = axes["general"]
    math = axes["math"]
    assert general["improvements"] == 4
    assert general["regressions"] == 0
    assert len(general["improvement_item_ids"]) == 4
    assert math["improvements"] == 0
    assert math["regressions"] == 2
    assert len(math["regression_item_ids"]) == 2
    assert view.deltas["general"] == pytest.approx(50.0)
    assert view.deltas["math"] == pytest.approx(-25.0)
    overall = paired["overall"]
    assert overall["improvements"] == 6
    assert overall["regressions"] == 2
    assert paired["method"] == "mcnemar-exact-binomial-two-sided-v1"
