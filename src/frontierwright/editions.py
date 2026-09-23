"""Edition profiles for the shared Frontierwright product core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from frontierwright.domain import ModelOrigin


class EditionProfile(StrEnum):
    STUDIO = "STUDIO"
    ACADEMY = "ACADEMY"
    LAB = "LAB"


@dataclass(frozen=True)
class EditionPolicy:
    profile: EditionProfile
    display_name: str
    tagline: str
    starting_point: str
    birth_first: bool
    education_explanations: bool
    private_infrastructure_first: bool
    public_discovery_default: bool


EDITION_POLICIES: dict[EditionProfile, EditionPolicy] = {
    EditionProfile.STUDIO: EditionPolicy(
        profile=EditionProfile.STUDIO,
        display_name="Frontierwright Studio",
        tagline="Develop models you control.",
        starting_point="IMPORT_OR_CONTINUE",
        birth_first=False,
        education_explanations=False,
        private_infrastructure_first=False,
        public_discovery_default=True,
    ),
    EditionProfile.ACADEMY: EditionPolicy(
        profile=EditionProfile.ACADEMY,
        display_name="Frontierwright Academy",
        tagline="Build an AI model from birth.",
        starting_point="BIRTH",
        birth_first=True,
        education_explanations=True,
        private_infrastructure_first=False,
        public_discovery_default=True,
    ),
    EditionProfile.LAB: EditionPolicy(
        profile=EditionProfile.LAB,
        display_name="Frontierwright Lab",
        tagline="Develop private models inside controlled infrastructure.",
        starting_point="CONNECT_INFRASTRUCTURE",
        birth_first=False,
        education_explanations=False,
        private_infrastructure_first=True,
        public_discovery_default=False,
    ),
}


def policy_for(profile: EditionProfile) -> EditionPolicy:
    return EDITION_POLICIES[profile]


def default_edition_for_origin(origin: ModelOrigin) -> EditionProfile:
    """Choose an initial UX profile without making edition part of model identity."""

    if origin is ModelOrigin.ZERO:
        return EditionProfile.ACADEMY
    if origin is ModelOrigin.INTERNAL_LAB:
        return EditionProfile.LAB
    return EditionProfile.STUDIO
