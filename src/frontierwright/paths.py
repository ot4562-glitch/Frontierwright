"""Training-path plugin protocol and honest prerequisite feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from frontierwright.data import DatasetRole
from frontierwright.domain import HistoryConfidence, ModelOrigin


class PathAvailability(StrEnum):
    LOCKED = "LOCKED"
    PLANNABLE = "PLANNABLE"
    READY = "READY"


class TrainingPathId(StrEnum):
    FROM_SCRATCH_PRETRAINING = "FROM_SCRATCH_PRETRAINING"
    CONTINUED_PRETRAINING = "CONTINUED_PRETRAINING"
    FULL_SFT = "FULL_SFT"
    LORA_SFT = "LORA_SFT"
    QLORA_SFT = "QLORA_SFT"


@dataclass(frozen=True)
class PathContext:
    origin: ModelOrigin
    champion_present: bool
    champion_trainable: bool | None
    history_confidence: HistoryConfidence
    resource_profile_available: bool
    dataset_roles: frozenset[DatasetRole]
    champion_is_birth_root: bool = False


@dataclass(frozen=True)
class PathAssessment:
    path_id: TrainingPathId
    title: str
    availability: PathAvailability
    blockers: tuple[str, ...]
    next_checks: tuple[str, ...]
    recommendation_eligible: bool


class TrainingPathPlugin(Protocol):
    @property
    def path_id(self) -> TrainingPathId: ...

    @property
    def title(self) -> str: ...

    def assess(self, context: PathContext) -> PathAssessment: ...


def _finish(
    *,
    path_id: TrainingPathId,
    title: str,
    context: PathContext,
    blockers: list[str],
    next_checks: list[str],
) -> PathAssessment:
    if blockers:
        availability = PathAvailability.LOCKED
    else:
        # No backend/calibration engine exists yet. PLANNABLE means hard
        # prerequisites are satisfied, not that execution is ready.
        availability = PathAvailability.PLANNABLE
        next_checks.extend(
            [
                "training backend implementation",
                "representative calibration",
                "peak-memory/time/storage feasibility",
            ]
        )
    return PathAssessment(
        path_id=path_id,
        title=title,
        availability=availability,
        blockers=tuple(blockers),
        next_checks=tuple(dict.fromkeys(next_checks)),
        recommendation_eligible=context.history_confidence.allows_recommendation,
    )


@dataclass(frozen=True)
class FromScratchPretrainingPath:
    path_id: TrainingPathId = TrainingPathId.FROM_SCRATCH_PRETRAINING
    title: str = "From-scratch pretraining"

    def assess(self, context: PathContext) -> PathAssessment:
        blockers: list[str] = []
        next_checks: list[str] = []
        if context.origin is not ModelOrigin.ZERO:
            blockers.append("project is not a Frontierwright zero-model project")
        if not context.champion_present:
            blockers.append("zero-model birth required before from-scratch pretraining")
        elif not context.champion_is_birth_root:
            blockers.append(
                "current model is not an untrained zero-model birth root; "
                "use continued pretraining instead"
            )
        if DatasetRole.PRETRAIN not in context.dataset_roles:
            blockers.append("pretraining dataset required")
        if not context.resource_profile_available:
            next_checks.append("run frontierwright resources detect")
        return _finish(
            path_id=self.path_id,
            title=self.title,
            context=context,
            blockers=blockers,
            next_checks=next_checks,
        )


@dataclass(frozen=True)
class ContinuedPretrainingPath:
    path_id: TrainingPathId = TrainingPathId.CONTINUED_PRETRAINING
    title: str = "Continued pretraining"

    def assess(self, context: PathContext) -> PathAssessment:
        blockers: list[str] = []
        next_checks: list[str] = []
        if not context.champion_present:
            blockers.append("current model required")
        elif context.champion_is_birth_root:
            blockers.append(
                "born root has not completed initial pretraining; use from-scratch pretraining"
            )
        elif context.champion_trainable is not True:
            blockers.append("trainable model representation required")
        if DatasetRole.PRETRAIN not in context.dataset_roles:
            blockers.append("pretraining dataset required")
        if not context.resource_profile_available:
            next_checks.append("run frontierwright resources detect")
        return _finish(
            path_id=self.path_id,
            title=self.title,
            context=context,
            blockers=blockers,
            next_checks=next_checks,
        )


@dataclass(frozen=True)
class _SFTPath:
    path_id: TrainingPathId
    title: str

    def assess(self, context: PathContext) -> PathAssessment:
        blockers: list[str] = []
        next_checks: list[str] = []
        if not context.champion_present:
            blockers.append("current model required")
        elif context.champion_is_birth_root:
            blockers.append(
                "born root has not completed initial pretraining; "
                "fine-tuning requires a trained descendant"
            )
        elif context.champion_trainable is not True:
            blockers.append("trainable model representation required")
        if DatasetRole.SFT not in context.dataset_roles:
            blockers.append("SFT dataset required")
        if not context.resource_profile_available:
            next_checks.append("run frontierwright resources detect")
        return _finish(
            path_id=self.path_id,
            title=self.title,
            context=context,
            blockers=blockers,
            next_checks=next_checks,
        )


DEFAULT_PATHS: tuple[TrainingPathPlugin, ...] = (
    FromScratchPretrainingPath(),
    ContinuedPretrainingPath(),
    _SFTPath(TrainingPathId.FULL_SFT, "Full SFT"),
    _SFTPath(TrainingPathId.LORA_SFT, "LoRA SFT"),
    _SFTPath(TrainingPathId.QLORA_SFT, "QLoRA SFT"),
)


def assess_paths(
    context: PathContext,
    *,
    plugins: tuple[TrainingPathPlugin, ...] = DEFAULT_PATHS,
) -> tuple[PathAssessment, ...]:
    return tuple(plugin.assess(context) for plugin in plugins)
