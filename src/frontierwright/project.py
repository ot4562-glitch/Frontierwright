"""Compatibility helpers over the canonical SQLite Registry.

New code should prefer :class:`frontierwright.registry.Registry` directly.
"""

from __future__ import annotations

from pathlib import Path

from frontierwright.domain import ModelOrigin
from frontierwright.editions import EditionProfile
from frontierwright.registry import ProjectState, Registry


def project_exists(root: Path) -> bool:
    return Registry(root).exists


def initialize_project(
    root: Path,
    *,
    project_name: str,
    origin: ModelOrigin,
    language: str = "en",
    edition_profile: EditionProfile | None = None,
) -> ProjectState:
    registry = Registry(root)
    registry.initialize(
        project_name,
        origin,
        language=language,
        edition_profile=edition_profile,
    )
    return registry.read()


def load_project(root: Path) -> ProjectState:
    return Registry(root).read()


__all__ = ["ProjectState", "Registry", "initialize_project", "load_project", "project_exists"]
