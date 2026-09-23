"""Extensible intervention taxonomy shared by every Frontierwright edition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from frontierwright.paths import TrainingPathId


class InterventionFamily(StrEnum):
    BIRTH = "BIRTH"
    LEARN = "LEARN"
    SPECIALIZE = "SPECIALIZE"
    ALIGN = "ALIGN"
    EVOLVE = "EVOLVE"
    OPTIMIZE = "OPTIMIZE"
    EVALUATE = "EVALUATE"
    OPERATE = "OPERATE"


@dataclass(frozen=True)
class InterventionDescriptor:
    intervention_id: str
    family: InterventionFamily
    title: str
    version: str
    provider: str
    training_path_id: TrainingPathId | None = None


BUILTIN_INTERVENTIONS: tuple[InterventionDescriptor, ...] = (
    InterventionDescriptor(
        intervention_id="frontierwright.birth.zero",
        family=InterventionFamily.BIRTH,
        title="Zero-model birth",
        version="1",
        provider="frontierwright",
    ),
    InterventionDescriptor(
        intervention_id="frontierwright.learn.pretrain",
        family=InterventionFamily.LEARN,
        title="From-scratch pretraining",
        version="1",
        provider="frontierwright",
        training_path_id=TrainingPathId.FROM_SCRATCH_PRETRAINING,
    ),
    InterventionDescriptor(
        intervention_id="frontierwright.learn.continued-pretrain",
        family=InterventionFamily.LEARN,
        title="Continued pretraining",
        version="1",
        provider="frontierwright",
        training_path_id=TrainingPathId.CONTINUED_PRETRAINING,
    ),
    InterventionDescriptor(
        intervention_id="frontierwright.specialize.full-sft",
        family=InterventionFamily.SPECIALIZE,
        title="Full SFT",
        version="1",
        provider="frontierwright",
        training_path_id=TrainingPathId.FULL_SFT,
    ),
    InterventionDescriptor(
        intervention_id="frontierwright.specialize.lora-sft",
        family=InterventionFamily.SPECIALIZE,
        title="LoRA SFT",
        version="1",
        provider="frontierwright",
        training_path_id=TrainingPathId.LORA_SFT,
    ),
    InterventionDescriptor(
        intervention_id="frontierwright.specialize.qlora-sft",
        family=InterventionFamily.SPECIALIZE,
        title="QLoRA SFT",
        version="1",
        provider="frontierwright",
        training_path_id=TrainingPathId.QLORA_SFT,
    ),
)


_BY_ID = {item.intervention_id: item for item in BUILTIN_INTERVENTIONS}
_BY_PATH = {
    item.training_path_id: item
    for item in BUILTIN_INTERVENTIONS
    if item.training_path_id is not None
}


def intervention_by_id(intervention_id: str) -> InterventionDescriptor | None:
    return _BY_ID.get(intervention_id)


def intervention_for_training_path(
    path_id: TrainingPathId,
) -> InterventionDescriptor:
    return _BY_PATH[path_id]
