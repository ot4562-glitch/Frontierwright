from frontierwright.interventions import (
    BUILTIN_INTERVENTIONS,
    InterventionFamily,
    intervention_by_id,
    intervention_for_training_path,
)
from frontierwright.paths import TrainingPathId


def test_all_v1_training_paths_have_intervention_descriptors() -> None:
    for path_id in TrainingPathId:
        item = intervention_for_training_path(path_id)
        assert item.training_path_id is path_id
        assert item.provider == "frontierwright"
        assert item.version == "1"


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
    assert birth.training_path_id is None

    families = {item.family for item in BUILTIN_INTERVENTIONS}
    assert InterventionFamily.LEARN in families
    assert InterventionFamily.SPECIALIZE in families
