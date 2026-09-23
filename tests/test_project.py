from pathlib import Path

from frontierwright.domain import (
    Axis,
    CapabilityStat,
    HistoryConfidence,
    ModelOrigin,
    ModelState,
)
from frontierwright.registry import Registry
from frontierwright.service import get_status


def stat(axis: Axis, value: float, receipt: str) -> CapabilityStat:
    return CapabilityStat(
        axis=axis,
        value=value,
        scale_version="fw-capability-v1",
        evaluation_receipt=receipt,
    )


def candidate(
    registry: Registry,
    *,
    model_id: str,
    parent_model_id: str | None,
    coding: float,
    general: float,
) -> ModelState:
    state = registry.read()
    return ModelState(
        model_id=model_id,
        identity_id=state.project["identity_id"],
        origin=ModelOrigin(state.project["origin"]),
        checkpoint=f"checkpoint-{model_id}",
        fingerprint=f"sha256:{model_id}",
        parent_model_id=parent_model_id,
        stats=(
            stat(Axis.GENERAL, general, f"eval-{model_id}"),
            stat(Axis.CODING, coding, f"eval-{model_id}"),
        ),
    )


def test_zero_project_is_complete_from_birth_and_uses_sqlite(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO, language="ko")

    assert registry.path.is_file()
    assert (tmp_path / ".frontierwright" / "project.toml").is_file()

    state = registry.read()
    assert state.project["history_confidence"] == HistoryConfidence.COMPLETE.value
    assert state.project["language"] == "ko"

    view = get_status(tmp_path)
    assert view.measurement_state == "NOT_READY"
    assert view.build_mode == "INTENT"
    assert view.strong_recommendation_allowed is True


def test_imported_project_defaults_to_unknown_history(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("Imported", ModelOrigin.IMPORTED_LOCAL)

    view = get_status(tmp_path)
    assert view.history_confidence == "UNKNOWN"
    assert view.strong_recommendation_allowed is False
    assert view.build_mode == "NOT_READY"


def test_current_champion_replaces_historical_peak_in_current_display(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    strong = candidate(
        registry,
        model_id="strong",
        parent_model_id=None,
        coding=140,
        general=130,
    )
    registry.register_candidate(strong)
    registry.promote_candidate("strong")

    first = get_status(tmp_path)
    assert first.stats["coding"] == 140

    weaker = candidate(
        registry,
        model_id="weaker-current",
        parent_model_id="strong",
        coding=120,
        general=110,
    )
    registry.register_candidate(weaker)
    registry.promote_candidate("weaker-current")

    current = get_status(tmp_path)
    assert current.champion_model_id == "weaker-current"
    assert current.stats["coding"] == 120
    assert current.stats["general"] == 110

    history = registry.read().history
    assert any(event["details"].get("model_id") == "strong" for event in history)
    assert any(event["details"].get("model_id") == "weaker-current" for event in history)
