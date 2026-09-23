"""Extensible intervention plugins shared by every Frontierwright edition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from frontierwright.paths import (
    DEFAULT_PATHS,
    PathAssessment,
    PathContext,
    TrainingPathId,
    TrainingPathPlugin,
)


class InterventionFamily(StrEnum):
    BIRTH = "BIRTH"
    LEARN = "LEARN"
    SPECIALIZE = "SPECIALIZE"
    ALIGN = "ALIGN"
    EVOLVE = "EVOLVE"
    OPTIMIZE = "OPTIMIZE"
    EVALUATE = "EVALUATE"
    OPERATE = "OPERATE"


class InterventionSurface(StrEnum):
    BIRTH = "BIRTH"
    TRAINING_PATH = "TRAINING_PATH"
    ARTIFACT_TRANSFORM = "ARTIFACT_TRANSFORM"
    EVALUATION = "EVALUATION"
    OPERATION = "OPERATION"


@dataclass(frozen=True)
class InterventionDescriptor:
    intervention_id: str
    family: InterventionFamily
    title: str
    version: str
    provider: str
    surface: InterventionSurface
    training_path_id: TrainingPathId | None = None

    def __post_init__(self) -> None:
        for name in ("intervention_id", "title", "version", "provider"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if (
            self.surface is InterventionSurface.TRAINING_PATH
            and self.training_path_id is None
        ):
            raise ValueError("TRAINING_PATH interventions require training_path_id")
        if (
            self.surface is not InterventionSurface.TRAINING_PATH
            and self.training_path_id is not None
        ):
            raise ValueError(
                "Only TRAINING_PATH interventions may declare training_path_id"
            )

    def machine_payload(self) -> dict[str, object]:
        return {
            "intervention_id": self.intervention_id,
            "family": self.family.value,
            "title": self.title,
            "version": self.version,
            "provider": self.provider,
            "surface": self.surface.value,
            "training_path_id": (
                self.training_path_id.value
                if self.training_path_id is not None
                else None
            ),
        }


class InterventionPlugin(Protocol):
    @property
    def descriptor(self) -> InterventionDescriptor: ...


class TrainingInterventionPlugin(InterventionPlugin, Protocol):
    @property
    def path_plugin(self) -> TrainingPathPlugin: ...

    def assess(self, context: PathContext) -> PathAssessment: ...


@dataclass(frozen=True)
class BirthInterventionPlugin:
    descriptor: InterventionDescriptor


@dataclass(frozen=True)
class BuiltinTrainingInterventionPlugin:
    descriptor: InterventionDescriptor
    path_plugin: TrainingPathPlugin

    def __post_init__(self) -> None:
        if self.descriptor.surface is not InterventionSurface.TRAINING_PATH:
            raise ValueError("Training intervention must use TRAINING_PATH surface")
        if self.descriptor.training_path_id is not self.path_plugin.path_id:
            raise ValueError("Intervention descriptor/path plugin identity mismatch")

    def assess(self, context: PathContext) -> PathAssessment:
        return self.path_plugin.assess(context)


_PATH_PLUGINS = {plugin.path_id: plugin for plugin in DEFAULT_PATHS}


def _training_plugin(
    *,
    intervention_id: str,
    family: InterventionFamily,
    title: str,
    path_id: TrainingPathId,
) -> BuiltinTrainingInterventionPlugin:
    return BuiltinTrainingInterventionPlugin(
        descriptor=InterventionDescriptor(
            intervention_id=intervention_id,
            family=family,
            title=title,
            version="1",
            provider="frontierwright",
            surface=InterventionSurface.TRAINING_PATH,
            training_path_id=path_id,
        ),
        path_plugin=_PATH_PLUGINS[path_id],
    )


BUILTIN_INTERVENTION_PLUGINS: tuple[InterventionPlugin, ...] = (
    BirthInterventionPlugin(
        descriptor=InterventionDescriptor(
            intervention_id="frontierwright.birth.zero",
            family=InterventionFamily.BIRTH,
            title="Zero-model birth",
            version="1",
            provider="frontierwright",
            surface=InterventionSurface.BIRTH,
        )
    ),
    _training_plugin(
        intervention_id="frontierwright.learn.pretrain",
        family=InterventionFamily.LEARN,
        title="From-scratch pretraining",
        path_id=TrainingPathId.FROM_SCRATCH_PRETRAINING,
    ),
    _training_plugin(
        intervention_id="frontierwright.learn.continued-pretrain",
        family=InterventionFamily.LEARN,
        title="Continued pretraining",
        path_id=TrainingPathId.CONTINUED_PRETRAINING,
    ),
    _training_plugin(
        intervention_id="frontierwright.specialize.full-sft",
        family=InterventionFamily.SPECIALIZE,
        title="Full SFT",
        path_id=TrainingPathId.FULL_SFT,
    ),
    _training_plugin(
        intervention_id="frontierwright.specialize.lora-sft",
        family=InterventionFamily.SPECIALIZE,
        title="LoRA SFT",
        path_id=TrainingPathId.LORA_SFT,
    ),
    _training_plugin(
        intervention_id="frontierwright.specialize.qlora-sft",
        family=InterventionFamily.SPECIALIZE,
        title="QLoRA SFT",
        path_id=TrainingPathId.QLORA_SFT,
    ),
)

BUILTIN_INTERVENTIONS: tuple[InterventionDescriptor, ...] = tuple(
    plugin.descriptor for plugin in BUILTIN_INTERVENTION_PLUGINS
)

_BY_ID = {
    plugin.descriptor.intervention_id: plugin
    for plugin in BUILTIN_INTERVENTION_PLUGINS
}
_BY_PATH = {
    plugin.descriptor.training_path_id: plugin
    for plugin in BUILTIN_INTERVENTION_PLUGINS
    if plugin.descriptor.training_path_id is not None
}


def intervention_plugins() -> tuple[InterventionPlugin, ...]:
    return BUILTIN_INTERVENTION_PLUGINS


def intervention_plugin_by_id(intervention_id: str) -> InterventionPlugin | None:
    return _BY_ID.get(intervention_id)


def intervention_by_id(intervention_id: str) -> InterventionDescriptor | None:
    plugin = intervention_plugin_by_id(intervention_id)
    return plugin.descriptor if plugin is not None else None


def intervention_for_training_path(
    path_id: TrainingPathId,
) -> InterventionDescriptor:
    return _BY_PATH[path_id].descriptor


def training_intervention_for_path(
    path_id: TrainingPathId,
) -> TrainingInterventionPlugin:
    plugin = _BY_PATH[path_id]
    if not isinstance(plugin, BuiltinTrainingInterventionPlugin):
        raise TypeError(f"Intervention for {path_id.value} is not a training plugin")
    return plugin


def assess_training_interventions(
    context: PathContext,
) -> tuple[tuple[InterventionDescriptor, PathAssessment], ...]:
    results: list[tuple[InterventionDescriptor, PathAssessment]] = []
    for plugin in BUILTIN_INTERVENTION_PLUGINS:
        if not isinstance(plugin, BuiltinTrainingInterventionPlugin):
            continue
        results.append((plugin.descriptor, plugin.assess(context)))
    return tuple(results)
