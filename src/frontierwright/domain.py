"""Immutable domain values. No terminal, database, or training dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class ModelOrigin(StrEnum):
    ZERO = "ZERO"
    IMPORTED_LOCAL = "IMPORTED_LOCAL"
    INTERNAL_LAB = "INTERNAL_LAB"


class HistoryConfidence(StrEnum):
    COMPLETE = "COMPLETE"
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"

    @property
    def allows_recommendation(self) -> bool:
        return self in (self.COMPLETE, self.VERIFIED)


class Axis(StrEnum):
    GENERAL = "GENERAL"
    REASONING = "REASONING"
    MATH = "MATH"
    CODING = "CODING"


class BuildMode(StrEnum):
    INTENT = "INTENT"
    TARGETS_FLOORS = "TARGETS_FLOORS"
    NOT_READY = "NOT_READY"


class CandidateStatus(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    INCOMPLETE = "INCOMPLETE"


class LineageRelation(StrEnum):
    DERIVED_FROM = "DERIVED_FROM"
    MERGED_FROM = "MERGED_FROM"
    DISTILLED_FROM = "DISTILLED_FROM"
    TRANSFORMED_FROM = "TRANSFORMED_FROM"


class ModelFormat(StrEnum):
    HUGGINGFACE = "HUGGINGFACE"
    GGUF = "GGUF"
    FRONTIERWRIGHT_QUANTIZED = "FRONTIERWRIGHT_QUANTIZED"
    CUSTOM = "CUSTOM"


class ResourceProvenance(StrEnum):
    DETECTED = "DETECTED"
    MEASURED = "MEASURED"
    DECLARED = "DECLARED"
    CONNECTED = "CONNECTED"
    ESTIMATED = "ESTIMATED"
    HYPOTHETICAL = "HYPOTHETICAL"


def require_text(value: str, field: str) -> None:
    if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field} must be nonempty text without control characters")


@dataclass(frozen=True)
class CapabilityStat:
    """A supplied measurement tied to a frozen scale and evaluation receipt.

    This milestone consumes evidence; it does not implement an evaluator or invent
    an aggregation scale. Values above 100 are valid. Unknown axes are absent.
    """

    axis: Axis
    value: float
    scale_version: str
    evaluation_receipt: str

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not math.isfinite(self.value) or self.value < 0:
            raise ValueError("Capability values must be finite, nonnegative numbers")
        require_text(self.scale_version, "scale_version")
        require_text(self.evaluation_receipt, "evaluation_receipt")


@dataclass(frozen=True)
class ModelState:
    """One immutable, owned checkpoint; identity is distinct from its fingerprint."""

    model_id: str
    identity_id: str
    origin: ModelOrigin
    checkpoint: str
    fingerprint: str
    parent_model_id: str | None = None
    stats: tuple[CapabilityStat, ...] = ()
    model_format: ModelFormat = ModelFormat.CUSTOM
    trainable: bool = True

    def __post_init__(self) -> None:
        for field in ("model_id", "identity_id", "checkpoint", "fingerprint"):
            require_text(getattr(self, field), field)
        if self.parent_model_id is not None:
            require_text(self.parent_model_id, "parent_model_id")
        if self.parent_model_id == self.model_id:
            raise ValueError("A model cannot be its own parent")
        if len({stat.axis for stat in self.stats}) != len(self.stats):
            raise ValueError("A snapshot cannot contain duplicate capability axes")

    @property
    def measured(self) -> bool:
        return bool(self.stats)


@dataclass(frozen=True)
class BuildIntent:
    archetype: str = "Balanced"
    priorities: tuple[tuple[Axis, int], ...] = ()

    def __post_init__(self) -> None:
        require_text(self.archetype, "archetype")
        if len({axis for axis, _ in self.priorities}) != len(self.priorities):
            raise ValueError("Build intent cannot repeat an axis")
        if any(value < 0 for _, value in self.priorities):
            raise ValueError("Build priorities must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "archetype": self.archetype,
            "priorities": {axis.value.lower(): value for axis, value in self.priorities},
        }


@dataclass(frozen=True)
class BuildTargets:
    targets: tuple[tuple[Axis, int], ...]
    floors: tuple[tuple[Axis, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("At least one target capability is required")
        for label, values in (("targets", self.targets), ("floors", self.floors)):
            if len({axis for axis, _ in values}) != len(values):
                raise ValueError(f"Build {label} cannot repeat an axis")
            if any(value < 0 for _, value in values):
                raise ValueError(f"Build {label} must be nonnegative")

        floors = dict(self.floors)
        for axis, target in self.targets:
            floor = floors.get(axis)
            if floor is not None and target < floor:
                raise ValueError(
                    f"Target for {axis.value} cannot be below its declared floor"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "targets": {axis.value.lower(): value for axis, value in self.targets},
            "floors": {axis.value.lower(): value for axis, value in self.floors},
        }


@dataclass(frozen=True)
class Champion:
    model: ModelState


@dataclass(frozen=True)
class Candidate:
    model: ModelState
    status: CandidateStatus = CandidateStatus.PENDING


def build_mode(origin: ModelOrigin, champion: Champion | None) -> BuildMode:
    if champion is not None and champion.model.measured:
        return BuildMode.TARGETS_FLOORS
    if origin == ModelOrigin.ZERO:
        return BuildMode.INTENT
    return BuildMode.NOT_READY
