from pathlib import Path

from textual.widgets import TabbedContent

from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.registry import Registry
from frontierwright.service import (
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
from frontierwright.tui.app import CandidateScreen, HelpScreen


def make_view() -> StatusView:
    return StatusView(
        initialized=True,
        project_id="fw-test",
        project_name="NOVA",
        language="en",
        nickname="NOVA",
        origin="ZERO",
        history_confidence="COMPLETE",
        measurement_state="NOT_READY",
        build_mode="INTENT",
        strong_recommendation_allowed=True,
        stats={"general": None, "reasoning": None, "math": None, "coding": None},
    )


async def test_tui_starts_with_required_sections_and_keyboard_navigation() -> None:
    app = FrontierwrightApp(view=make_view(), language="en")

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
