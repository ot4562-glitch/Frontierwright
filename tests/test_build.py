from pathlib import Path

import pytest

from frontierwright.domain import (
    Axis,
    CapabilityStat,
    ModelOrigin,
    ModelState,
)
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import (
    AxisScale,
    CapabilityScale,
    EvaluationReceipt,
    RawMeasurement,
    ScaleTask,
    apply_scale,
)
from frontierwright.registry import Registry
from frontierwright.service import (
    get_build_view,
    set_build_intent,
    set_build_targets,
    set_growth_goals,
)


def measured_candidate(registry: Registry, model_id: str = "measured") -> ModelState:
    state = registry.read()
    return ModelState(
        model_id=model_id,
        identity_id=state.project["identity_id"],
        origin=ModelOrigin(state.project["origin"]),
        checkpoint=f"checkpoint-{model_id}",
        fingerprint=f"sha256:{model_id}",
        stats=(
            CapabilityStat(
                axis=Axis.GENERAL,
                value=100,
                scale_version="fw-capability-v1",
                evaluation_receipt="eval-general",
            ),
            CapabilityStat(
                axis=Axis.CODING,
                value=110,
                scale_version="fw-capability-v1",
                evaluation_receipt="eval-coding",
            ),
        ),
    )





def activate_frozen_profile(registry: Registry, model: ModelState) -> None:
    receipt = EvaluationReceipt(
        receipt_id=f"receipt-{model.model_id}",
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        evaluator_id="build-fixture",
        evaluator_version="1",
        conditions={"seed": 1},
        measurements=(
            RawMeasurement("general", "1", "accuracy", 0.5),
            RawMeasurement("coding", "1", "pass_rate", 0.6),
        ),
    )
    scale = CapabilityScale(
        scale_id="build-test-scale",
        scale_version="1",
        frozen=True,
        axes=(
            AxisScale(
                axis=Axis.GENERAL,
                display_anchor=100,
                tasks=(
                    ScaleTask(
                        "general",
                        "1",
                        "accuracy",
                        1,
                        0.5,
                        0.1,
                        10,
                    ),
                ),
            ),
            AxisScale(
                axis=Axis.CODING,
                display_anchor=100,
                tasks=(
                    ScaleTask(
                        "coding",
                        "1",
                        "pass_rate",
                        1,
                        0.5,
                        0.1,
                        10,
                    ),
                ),
            ),
        ),
    )
    registry.activate_capability_profile(receipt, scale, apply_scale(receipt, scale))



def test_zero_model_starts_with_unconfigured_intent(tmp_path: Path) -> None:
    Registry(tmp_path).initialize("ZERO", ModelOrigin.ZERO)

    view = get_build_view(tmp_path)
    assert view.mode == "INTENT"
    assert view.configured is False
    assert view.archetype == "Balanced"


def test_zero_model_intent_persists_relative_priorities(tmp_path: Path) -> None:
    Registry(tmp_path).initialize("ZERO", ModelOrigin.ZERO)

    view = set_build_intent(
        tmp_path,
        archetype="Code-focused",
        priorities={"coding": 60, "math": 20, "general": 10},
    )
    assert view.configured is True
    assert view.archetype == "Code-focused"
    assert view.priorities == {"coding": 60, "general": 10, "math": 20}


def test_inserted_unmeasured_model_cannot_fake_numeric_build(tmp_path: Path) -> None:
    Registry(tmp_path).initialize("Imported", ModelOrigin.IMPORTED_LOCAL)

    view = get_build_view(tmp_path)
    assert view.mode == "NOT_READY"

    with pytest.raises(FrontierwrightError, match="current build mode is NOT_READY"):
        set_build_targets(
            tmp_path,
            targets={"coding": 140},
            floors={"general": 120},
        )


def test_internal_lab_unmeasured_model_cannot_fake_numeric_build(
    tmp_path: Path,
) -> None:
    Registry(tmp_path).initialize("Lab", ModelOrigin.INTERNAL_LAB)

    view = get_build_view(tmp_path)
    assert view.mode == "NOT_READY"

    with pytest.raises(FrontierwrightError, match="current build mode is NOT_READY"):
        set_build_targets(
            tmp_path,
            targets={"coding": 140},
            floors={"general": 120},
        )


def test_measured_model_unlocks_targets_and_floors(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("ZERO", ModelOrigin.ZERO)
    model = measured_candidate(registry)
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    activate_frozen_profile(registry, model)

    view = set_build_targets(
        tmp_path,
        targets={"coding": 140},
        floors={"general": 95},
    )
    assert view.mode == "TARGETS_FLOORS"
    assert view.configured is True
    assert view.targets == {"coding": 140}
    assert view.floors == {"general": 95}
    assert view.scale_bound is True
    assert view.scale_id == "build-test-scale"
    assert view.scale_version == "1"
    assert isinstance(view.scale_hash, str)


def test_old_intent_does_not_masquerade_as_numeric_build_after_measurement(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path)
    registry.initialize("ZERO", ModelOrigin.ZERO)
    set_build_intent(
        tmp_path,
        archetype="Code-focused",
        priorities={"coding": 70},
    )

    model = measured_candidate(registry)
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    activate_frozen_profile(registry, model)

    view = get_build_view(tmp_path)
    assert view.mode == "TARGETS_FLOORS"
    assert view.configured is False
    assert view.priorities == {}
    assert view.targets == {}





def test_legacy_embedded_stats_cannot_define_unbound_numeric_build(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path)
    registry.initialize("ZERO", ModelOrigin.ZERO)
    model = measured_candidate(registry, model_id="legacy-measured")
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)

    with pytest.raises(FrontierwrightError, match="active frozen capability profile"):
        set_build_targets(
            tmp_path,
            targets={"coding": 120},
            floors={"general": 90},
        )



def test_target_cannot_be_below_same_axis_floor(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("ZERO", ModelOrigin.ZERO)
    model = measured_candidate(registry)
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    activate_frozen_profile(registry, model)

    with pytest.raises(FrontierwrightError, match="cannot be below"):
        set_build_targets(
            tmp_path,
            targets={"coding": 100},
            floors={"coding": 110},
        )



def test_measured_model_persists_relative_growth_rules(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("Studio", ModelOrigin.IMPORTED_LOCAL)
    model = measured_candidate(registry, model_id="studio-origin")
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    activate_frozen_profile(registry, model)

    view = set_growth_goals(
        tmp_path,
        improve=("capability.coding",),
        protect=("capability.reasoning",),
        tolerance={"serving.latency_p50": (5.0, "PERCENT")},
        absolute_requirements={"serving.throughput_p50": 20.0},
        hard_constraints={"privacy": "PRIVATE"},
    )

    assert view.mode == "TARGETS_FLOORS"
    assert view.configured is True
    assert view.scale_bound is True
    assert view.growth_goals == [
        {
            "kind": "IMPROVE",
            "metric": "capability.coding",
            "unit": None,
            "value": None,
        },
        {
            "kind": "PROTECT",
            "metric": "capability.reasoning",
            "unit": None,
            "value": None,
        },
        {
            "kind": "TOLERANCE",
            "metric": "serving.latency_p50",
            "unit": "PERCENT",
            "value": 5.0,
        },
        {
            "kind": "ABSOLUTE_REQUIREMENT",
            "metric": "serving.throughput_p50",
            "unit": None,
            "value": 20.0,
        },
        {
            "kind": "HARD_CONSTRAINT",
            "metric": "privacy",
            "unit": None,
            "value": "PRIVATE",
        },
    ]
