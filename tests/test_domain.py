import pytest

from frontierwright.domain import (
    Axis,
    BuildMode,
    CapabilityStat,
    Champion,
    HistoryConfidence,
    ModelOrigin,
    ModelState,
    build_mode,
)


def model(*, origin: ModelOrigin, stats: tuple[CapabilityStat, ...] = ()) -> ModelState:
    return ModelState(
        model_id="model-1",
        identity_id="identity-1",
        origin=origin,
        checkpoint="checkpoint",
        fingerprint="sha256:test",
        stats=stats,
    )


@pytest.mark.parametrize(
    ("confidence", "allowed"),
    [
        (HistoryConfidence.COMPLETE, True),
        (HistoryConfidence.VERIFIED, True),
        (HistoryConfidence.PARTIAL, False),
        (HistoryConfidence.UNKNOWN, False),
    ],
)
def test_history_confidence_gates_recommendation(
    confidence: HistoryConfidence, allowed: bool
) -> None:
    assert confidence.allows_recommendation is allowed


def test_unmeasured_zero_uses_intent_build_mode() -> None:
    assert build_mode(ModelOrigin.ZERO, None) is BuildMode.INTENT


@pytest.mark.parametrize(
    "origin",
    [ModelOrigin.IMPORTED_LOCAL, ModelOrigin.INTERNAL_LAB],
)
def test_unmeasured_inserted_model_is_not_ready_for_numeric_build(
    origin: ModelOrigin,
) -> None:
    assert build_mode(origin, None) is BuildMode.NOT_READY


def test_measured_model_unlocks_target_floor_mode() -> None:
    measured = CapabilityStat(
        axis=Axis.CODING,
        value=12,
        scale_version="fw-capability-v1",
        evaluation_receipt="eval-1",
    )
    champion = Champion(model(origin=ModelOrigin.ZERO, stats=(measured,)))
    assert build_mode(ModelOrigin.ZERO, champion) is BuildMode.TARGETS_FLOORS
