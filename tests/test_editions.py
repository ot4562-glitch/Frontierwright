from pathlib import Path

from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.editions import EditionProfile, default_edition_for_origin, policy_for
from frontierwright.registry import Registry
from frontierwright.service import get_status, set_project_edition


def test_origin_selects_initial_edition_without_becoming_identity() -> None:
    assert default_edition_for_origin(ModelOrigin.ZERO) is EditionProfile.ACADEMY
    assert default_edition_for_origin(ModelOrigin.IMPORTED_LOCAL) is EditionProfile.STUDIO
    assert default_edition_for_origin(ModelOrigin.INTERNAL_LAB) is EditionProfile.LAB

    assert policy_for(EditionProfile.STUDIO).starting_point == "IMPORT_OR_CONTINUE"
    assert policy_for(EditionProfile.ACADEMY).starting_point == "BIRTH"
    assert policy_for(EditionProfile.LAB).starting_point == "CONNECT_INFRASTRUCTURE"


def test_edition_profile_can_change_without_rewriting_model_identity(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    initial = registry.read()
    assert initial.project["edition_profile"] == "ACADEMY"
    identity_id = initial.project["identity_id"]

    champion = ModelState(
        model_id="root-model",
        identity_id=identity_id,
        origin=ModelOrigin.ZERO,
        checkpoint="root-model",
        fingerprint="sha256:root-model",
    )
    registry.register_candidate(champion)
    registry.promote_candidate(champion.model_id)

    studio = set_project_edition(tmp_path, EditionProfile.STUDIO)
    state = registry.read()

    assert studio.edition_profile == "STUDIO"
    assert studio.edition_name == "Frontierwright Studio"
    assert studio.edition_starting_point == "IMPORT_OR_CONTINUE"
    assert state.project["identity_id"] == identity_id
    assert state.project["champion_id"] == champion.model_id
    assert state.champion is not None
    assert state.champion.model.model_id == champion.model_id

    toml = (registry.state_dir / "project.toml").read_text(encoding="utf-8")
    assert 'edition_profile = "STUDIO"' in toml
    assert any(
        event["kind"] == "EDITION_PROFILE_CHANGED"
        and event["details"] == {"from": "ACADEMY", "to": "STUDIO"}
        for event in state.history
    )


def test_explicit_edition_override_is_independent_of_origin(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize(
        "Research Zero",
        ModelOrigin.ZERO,
        edition_profile=EditionProfile.LAB,
    )

    view = get_status(tmp_path)

    assert view.origin == "ZERO"
    assert view.edition_profile == "LAB"
    assert view.edition_name == "Frontierwright Lab"
    assert view.edition_starting_point == "CONNECT_INFRASTRUCTURE"
