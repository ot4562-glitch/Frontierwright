import json
from pathlib import Path

import pytest

from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import (
    apply_scale,
    load_capability_scale,
    load_evaluation_receipt,
)
from frontierwright.registry import Registry
from frontierwright.service import get_build_view, get_stats_view, get_status, ingest_stats


def make_champion(root: Path) -> ModelState:
    registry = Registry(root)
    registry.initialize("NOVA", ModelOrigin.ZERO)
    state = registry.read()
    model = ModelState(
        model_id="model-current",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint="checkpoint-current",
        fingerprint="sha256:model-current",
        stats=(),
    )
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    return model


def write_receipt(
    path: Path,
    model: ModelState,
    *,
    receipt_id: str = "receipt-1",
    general: float = 0.8,
    coding: float = 0.9,
    include_coding: bool = True,
    fingerprint: str | None = None,
) -> Path:
    measurements = [
        {
            "task_id": "general-task",
            "task_version": "1",
            "metric": "accuracy",
            "value": general,
            "higher_is_better": True,
        }
    ]
    if include_coding:
        measurements.append(
            {
                "task_id": "coding-task",
                "task_version": "1",
                "metric": "pass_rate",
                "value": coding,
                "higher_is_better": True,
            }
        )
    payload = {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "model_id": model.model_id,
        "model_fingerprint": fingerprint or model.fingerprint,
        "evaluator_id": "fixture-evaluator",
        "evaluator_version": "1.0",
        "conditions": {"temperature": 0, "seed": 7},
        "measurements": measurements,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_scale(path: Path, *, frozen: bool = True) -> Path:
    payload = {
        "schema_version": 1,
        "scale_id": "fixture-capability",
        "scale_version": "1",
        "frozen": frozen,
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
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_scale_produces_only_fully_evidenced_axes(tmp_path: Path) -> None:
    model = make_champion(tmp_path / "project")
    receipt_path = write_receipt(
        tmp_path / "receipt.json",
        model,
        include_coding=False,
    )
    scale_path = write_scale(tmp_path / "scale.json")

    receipt = load_evaluation_receipt(receipt_path)
    scale = load_capability_scale(scale_path)
    stats = apply_scale(receipt, scale)

    assert [(stat.axis.value, stat.value) for stat in stats] == [("GENERAL", 130.0)]


def test_unfrozen_scale_is_rejected(tmp_path: Path) -> None:
    scale_path = write_scale(tmp_path / "scale.json", frozen=False)
    with pytest.raises(FrontierwrightError, match="explicitly frozen"):
        load_capability_scale(scale_path)


def test_profile_ingest_updates_current_stats_and_unlocks_numeric_build(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    model = make_champion(project)
    receipt_path = write_receipt(tmp_path / "receipt.json", model)
    scale_path = write_scale(tmp_path / "scale.json")

    view = ingest_stats(
        project,
        receipt_path=receipt_path,
        scale_path=scale_path,
    )

    assert view.measured is True
    assert view.stats["general"] == pytest.approx(130.0)
    assert view.stats["coding"] == pytest.approx(140.0)
    assert view.receipt_id == "receipt-1"
    assert view.scale_id == "fixture-capability"
    assert view.raw_measurements[0]["task_id"] == "general-task"

    status = get_status(project)
    assert status.measurement_state == "MEASURED"
    assert status.stats["coding"] == pytest.approx(140.0)
    assert status.build_mode == "TARGETS_FLOORS"

    build = get_build_view(project)
    assert build.mode == "TARGETS_FLOORS"


def test_receipt_for_wrong_fingerprint_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    model = make_champion(project)
    receipt_path = write_receipt(
        tmp_path / "receipt.json",
        model,
        fingerprint="sha256:wrong",
    )
    scale_path = write_scale(tmp_path / "scale.json")

    with pytest.raises(FrontierwrightError, match="fingerprint"):
        ingest_stats(
            project,
            receipt_path=receipt_path,
            scale_path=scale_path,
        )


def test_receipt_for_unknown_model_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    model = make_champion(project)
    payload = json.loads(
        write_receipt(tmp_path / "receipt.json", model).read_text(encoding="utf-8")
    )
    payload["model_id"] = "model-someone-else"
    receipt_path = tmp_path / "receipt-other.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    scale_path = write_scale(tmp_path / "scale.json")

    with pytest.raises(FrontierwrightError, match="Model does not exist"):
        ingest_stats(
            project,
            receipt_path=receipt_path,
            scale_path=scale_path,
        )


def test_candidate_can_be_measured_before_promotion(tmp_path: Path) -> None:
    project = tmp_path / "project"
    champion = make_champion(project)
    registry = Registry(project)
    state = registry.read()
    candidate = ModelState(
        model_id="model-candidate",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint="candidate-checkpoint",
        fingerprint="sha256:candidate",
        parent_model_id=champion.model_id,
        stats=(),
    )
    registry.register_candidate(candidate)

    receipt_path = write_receipt(
        tmp_path / "candidate-receipt.json",
        candidate,
        receipt_id="receipt-candidate",
        general=0.75,
        coding=0.85,
    )
    scale_path = write_scale(tmp_path / "scale.json")

    view = ingest_stats(
        project,
        receipt_path=receipt_path,
        scale_path=scale_path,
    )

    assert view.model_id == candidate.model_id
    assert view.stats["general"] == pytest.approx(125.0)
    assert view.stats["coding"] == pytest.approx(135.0)
    assert registry.read().project["champion_id"] == champion.model_id


def test_new_active_profile_replaces_old_display_not_historical_peak(tmp_path: Path) -> None:
    project = tmp_path / "project"
    model = make_champion(project)
    scale_path = write_scale(tmp_path / "scale.json")

    first = write_receipt(
        tmp_path / "receipt-high.json",
        model,
        receipt_id="receipt-high",
        general=0.9,
        coding=0.9,
    )
    ingest_stats(project, receipt_path=first, scale_path=scale_path)
    assert get_stats_view(project).stats["coding"] == pytest.approx(140.0)

    second = write_receipt(
        tmp_path / "receipt-current.json",
        model,
        receipt_id="receipt-current",
        general=0.7,
        coding=0.7,
    )
    ingest_stats(project, receipt_path=second, scale_path=scale_path)

    current = get_stats_view(project)
    assert current.receipt_id == "receipt-current"
    assert current.stats["general"] == pytest.approx(120.0)
    assert current.stats["coding"] == pytest.approx(120.0)
