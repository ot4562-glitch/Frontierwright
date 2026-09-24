from pathlib import Path

from textual.widgets import TabbedContent

from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.registry import Registry
from frontierwright.service import (
    CompareView,
    StatusView,
    get_build_view,
    get_candidates_view,
    get_data_view,
    get_history_view,
    get_paths_view,
    get_resource_view,
    get_status,
)
from frontierwright.tui import FrontierwrightApp
from frontierwright.tui.app import (
    ActionCenterScreen,
    CandidateScreen,
    HelpScreen,
    WorkflowFormScreen,
    _compare_text,
)


def test_candidate_compare_text_includes_stored_raw_evaluation() -> None:
    view = CompareView(
        champion_model_id="champion",
        candidate_model_id="candidate",
        candidate_status="PENDING",
        scale_comparable=False,
        scale_reason="No frozen capability scale.",
        champion_stats={
            "general": None,
            "reasoning": None,
            "math": None,
            "coding": None,
        },
        candidate_stats={
            "general": None,
            "reasoning": None,
            "math": None,
            "coding": None,
        },
        deltas={
            "general": None,
            "reasoning": None,
            "math": None,
            "coding": None,
        },
        raw_evaluation_comparisons=[
            {
                "pack_id": "frontierwright.eval.reference-heldout-lm",
                "pack_version": "1",
                "dataset_id": "dataset-eval",
                "measurements": [
                    {
                        "metric": "perplexity",
                        "champion_value": 50.0,
                        "candidate_value": 40.0,
                        "improvement_delta": 10.0,
                    }
                ],
            }
        ],
    )

    rendered = _compare_text(view)
    assert "RAW EVALUATION" in rendered
    assert "frontierwright.eval.reference-heldout-lm@1" in rendered
    assert "perplexity: 50.0 -> 40.0 (improvement +10)" in rendered


def make_view() -> StatusView:
    return StatusView(
        initialized=True,
        project_id="fw-test",
        project_name="NOVA",
        language="en",
        nickname="NOVA",
        edition_profile="ACADEMY",
        edition_name="Frontierwright Academy",
        edition_tagline="Build an AI model from birth.",
        edition_starting_point="BIRTH",
        origin="ZERO",
        history_confidence="COMPLETE",
        measurement_state="NOT_READY",
        build_mode="INTENT",
        strong_recommendation_allowed=True,
        stats={"general": None, "reasoning": None, "math": None, "coding": None},
    )


async def test_tui_starts_with_required_sections_and_keyboard_navigation() -> None:
    app = FrontierwrightApp(view=make_view(), language="en")
    assert "Frontierwright Academy" in app._character_text()
    assert "Build an AI model from birth." in app._character_text()

    async with app.run_test(size=(120, 40)) as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)
        assert tabs.active == "character"

        for tab_id in (
            "character",
            "build",
            "paths",
            "resources",
            "data",
            "history",
            "candidates",
        ):
            assert app.query_one(f"#{tab_id}") is not None

        await pilot.press("2")
        await pilot.pause()
        assert tabs.active == "build"

        await pilot.press("l")
        await pilot.pause()
        assert tabs.active == "paths"

        await pilot.press("h")
        await pilot.pause()
        assert tabs.active == "build"

        await pilot.press("?")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)


def setup_candidate_project(root: Path) -> None:
    registry = Registry(root)
    registry.initialize("NOVA", ModelOrigin.ZERO)
    state = registry.read()
    champion = ModelState(
        model_id="champion",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint="champion",
        fingerprint="sha256:champion",
    )
    registry.register_candidate(champion)
    registry.promote_candidate(champion.model_id)

    for index in (1, 2):
        registry.register_candidate(
            ModelState(
                model_id=f"candidate-{index}",
                identity_id=state.project["identity_id"],
                origin=ModelOrigin.ZERO,
                checkpoint=f"candidate-{index}",
                fingerprint=f"sha256:candidate-{index}",
                parent_model_id=champion.model_id,
            )
        )


async def test_tui_candidate_keyboard_selection_and_compare_modal(tmp_path: Path) -> None:
    project = tmp_path / "project"
    setup_candidate_project(project)

    app = FrontierwrightApp(
        view=get_status(project),
        resources=get_resource_view(project),
        build=get_build_view(project),
        data=get_data_view(project),
        paths=get_paths_view(project),
        candidates=get_candidates_view(project),
        history=get_history_view(project),
        root=project,
        language="en",
    )

    async with app.run_test(size=(140, 45)) as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)

        await pilot.press("7")
        await pilot.pause()
        assert tabs.active == "candidates"
        assert app.candidate_index == 0

        await pilot.press("j")
        await pilot.pause()
        assert app.candidate_index == 1

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, CandidateScreen)
        assert app.screen.compare.candidate_model_id == "candidate-2"

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, CandidateScreen)

        await pilot.press("k")
        await pilot.pause()
        assert app.candidate_index == 0


async def test_tui_action_center_sets_build_intent_through_shared_service(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)

    app = FrontierwrightApp(
        view=get_status(project),
        resources=get_resource_view(project),
        build=get_build_view(project),
        data=get_data_view(project),
        paths=get_paths_view(project),
        candidates=get_candidates_view(project),
        history=get_history_view(project),
        root=project,
        language="en",
    )

    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ActionCenterScreen)

        # Detect resources -> Register dataset -> Set build intent.
        await pilot.press("j")
        await pilot.press("j")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, WorkflowFormScreen)

        await pilot.press("ctrl+s")
        await pilot.pause()
        assert not isinstance(app.screen, WorkflowFormScreen)

    build = get_build_view(project)
    assert build.configured is True
    assert build.archetype == "Balanced"
    assert build.priorities == {
        "coding": 1,
        "general": 1,
        "math": 1,
        "reasoning": 1,
    }


def test_tui_action_center_exposes_academy_birth_after_pretrain_data(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    source = tmp_path / "pretrain.txt"
    source.write_text("frontierwright academy\n" * 10, encoding="utf-8")

    from frontierwright.data import DatasetRole
    from frontierwright.service import add_local_dataset

    add_local_dataset(
        project,
        source,
        name="Pretrain",
        role=DatasetRole.PRETRAIN,
    )

    app = FrontierwrightApp(
        view=get_status(project),
        resources=get_resource_view(project),
        build=get_build_view(project),
        data=get_data_view(project),
        paths=get_paths_view(project),
        candidates=get_candidates_view(project),
        history=get_history_view(project),
        root=project,
        language="en",
    )
    action_ids = {item.action_id for item in app._action_items()}
    assert "birth_tokenizer" in action_ids
    assert "birth_zero" in action_ids


async def test_tui_can_initialize_academy_project_from_uninitialized_directory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "new-project"
    app = FrontierwrightApp(
        view=get_status(project),
        root=project,
        language="en",
    )

    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ActionCenterScreen)

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, WorkflowFormScreen)

        await pilot.press("ctrl+s")
        await pilot.pause()
        assert not isinstance(app.screen, WorkflowFormScreen)
        assert app.view.initialized is True
        assert app.view.origin == "ZERO"
        assert app.view.edition_profile == "ACADEMY"

    status = get_status(project)
    assert status.initialized is True
    assert status.project_name == "My Model"
    assert status.origin == "ZERO"
    assert status.edition_profile == "ACADEMY"


def test_tui_action_center_exposes_capability_v1_for_current_champion(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    setup_candidate_project(project)

    app = FrontierwrightApp(
        view=get_status(project),
        resources=get_resource_view(project),
        build=get_build_view(project),
        data=get_data_view(project),
        paths=get_paths_view(project),
        candidates=get_candidates_view(project),
        history=get_history_view(project),
        root=project,
        language="en",
    )

    actions = {item.action_id: item for item in app._action_items()}
    assert "capability_v1" in actions
    assert actions["capability_v1"].title == "Measure Capability v1"
