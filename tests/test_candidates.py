import json
from pathlib import Path

import pytest

from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
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
)


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
        event["kind"] == "CANDIDATE_PROMOTED"
        and event["details"]["model_id"] == candidate.model_id
        for event in history.events
    )
    assert registry.read().project["champion_id"] == candidate.model_id


def test_rejected_candidate_stays_in_history_and_cannot_be_promoted(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _, _, candidate = create_project(project)

    view = reject_candidate(project, candidate.model_id)
    item = next(item for item in view.candidates if item["model_id"] == candidate.model_id)
    assert item["status"] == "REJECTED"

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
