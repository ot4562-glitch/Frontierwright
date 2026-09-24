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
    experience_goal: str
    action_center_title: str
    action_center_intro: str
    default_detail_level: str


EDITION_POLICIES: dict[EditionProfile, EditionPolicy] = {
    EditionProfile.STUDIO: EditionPolicy(
        profile=EditionProfile.STUDIO,
        display_name="Frontierwright Studio",
        tagline="Fit a model to your machine and workload.",
        starting_point="IMPORT_OR_CONTINUE",
        birth_first=False,
        education_explanations=False,
        private_infrastructure_first=False,
        public_discovery_default=True,
        experience_goal="Own and continuously shape a model for your real tasks.",
        action_center_title="STUDIO WORKBENCH",
        action_center_intro=(
            "Optimize the model you control for your workload, resource envelope, "
            "and explicit capability floors."
        ),
        default_detail_level="BALANCED",
    ),
    EditionProfile.ACADEMY: EditionPolicy(
        profile=EditionProfile.ACADEMY,
        display_name="Frontierwright Academy",
        tagline="Understand AI by building a real model from birth.",
        starting_point="BIRTH",
        birth_first=True,
        education_explanations=True,
        private_infrastructure_first=False,
        public_discovery_default=True,
        experience_goal="Learn real model development by changing real model state.",
        action_center_title="ACADEMY GUIDE",
        action_center_intro=(
            "Follow the real lifecycle one step at a time. Every action changes or "
            "measures actual model state; use the explanations to understand why."
        ),
        default_detail_level="GUIDED",
    ),
    EditionProfile.LAB: EditionPolicy(
        profile=EditionProfile.LAB,
        display_name="Frontierwright Lab",
        tagline="Push controlled private models toward the frontier.",
        starting_point="CONNECT_INFRASTRUCTURE",
        birth_first=False,
        education_explanations=False,
        private_infrastructure_first=True,
        public_discovery_default=False,
        experience_goal=(
            "Maximize capability, efficiency, and reliability with reproducible "
            "large-scale experiments."
        ),
        action_center_title="LAB CONTROL PLANE",
        action_center_intro=(
            "Operate on exact evidence: private infrastructure, evaluator/reward "
            "identity, resource topology, budgets, lineage, and regression gates."
        ),
        default_detail_level="EXPERT",
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
