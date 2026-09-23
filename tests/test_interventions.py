from frontierwright.data import DatasetRole
from frontierwright.domain import HistoryConfidence, ModelOrigin
from frontierwright.interventions import (
    BUILTIN_INTERVENTIONS,
    InterventionFamily,
    InterventionSurface,
    assess_training_interventions,
    intervention_by_id,
    intervention_for_training_path,
    intervention_plugin_by_id,
    intervention_plugins,
    training_intervention_for_path,
)
from frontierwright.paths import PathContext, TrainingPathId


def test_all_v1_training_paths_have_intervention_plugins() -> None:
    for path_id in TrainingPathId:
        item = intervention_for_training_path(path_id)
        plugin = training_intervention_for_path(path_id)

        assert item.training_path_id is path_id
        assert item.provider == "frontierwright"
        assert item.version == "1"
        assert item.surface is InterventionSurface.TRAINING_PATH
        assert plugin.descriptor == item
        assert plugin.path_plugin.path_id is path_id


def test_intervention_taxonomy_is_lifecycle_extensible() -> None:
    assert set(InterventionFamily) == {
        InterventionFamily.BIRTH,
        InterventionFamily.LEARN,
        InterventionFamily.SPECIALIZE,
        InterventionFamily.ALIGN,
        InterventionFamily.EVOLVE,
        InterventionFamily.OPTIMIZE,
        InterventionFamily.EVALUATE,
        InterventionFamily.OPERATE,
    }

    birth = intervention_by_id("frontierwright.birth.zero")
    assert birth is not None
    assert birth.family is InterventionFamily.BIRTH
    assert birth.surface is InterventionSurface.BIRTH
    assert birth.training_path_id is None

    families = {item.family for item in BUILTIN_INTERVENTIONS}
    assert InterventionFamily.LEARN in families
    assert InterventionFamily.SPECIALIZE in families


def test_plugin_registry_has_stable_machine_identity() -> None:
    plugins = intervention_plugins()
    ids = [plugin.descriptor.intervention_id for plugin in plugins]

    assert ids[0] == "frontierwright.birth.zero"
    assert len(ids) == len(set(ids))
    assert intervention_plugin_by_id("frontierwright.specialize.lora-sft") is not None

    payload = plugins[0].descriptor.machine_payload()
    assert payload == {
        "intervention_id": "frontierwright.birth.zero",
        "family": "BIRTH",
        "title": "Zero-model birth",
        "version": "1",
        "provider": "frontierwright",
        "surface": "BIRTH",
        "training_path_id": None,
    }


def test_training_assessment_executes_through_intervention_plugins() -> None:
    context = PathContext(
        origin=ModelOrigin.IMPORTED_LOCAL,
        champion_present=True,
        champion_trainable=True,
        history_confidence=HistoryConfidence.VERIFIED,
        resource_profile_available=True,
        dataset_roles=frozenset({DatasetRole.SFT}),
    )

    assessed = assess_training_interventions(context)
    by_path = {
        descriptor.training_path_id: assessment
        for descriptor, assessment in assessed
    }

    assert set(by_path) == set(TrainingPathId)
    lora = by_path[TrainingPathId.LORA_SFT]
    assert lora.availability.value == "PLANNABLE"
    assert lora.blockers == ()
    assert lora.recommendation_eligible is True
