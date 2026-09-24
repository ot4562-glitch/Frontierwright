from dataclasses import replace
from pathlib import Path

from textual.widgets import TabbedContent

import frontierwright.service as service_module
from frontierwright.capability_v1 import (
    CAPABILITY_V1_BUNDLE_ID,
    CAPABILITY_V1_BUNDLE_VERSION,
    CAPABILITY_V1_SCALE,
    CAPABILITY_V1_SCORING,
    capability_v1_bundle_hash,
    capability_v1_task_counts,
)
from frontierwright.domain import Axis, ModelOrigin, ModelState
from frontierwright.evaluations import EvaluationReceipt, RawMeasurement, apply_scale
from frontierwright.registry import Registry
from frontierwright.service import (
    CompareView,
    StatusView,
    WorkloadView,
    get_build_view,
    get_candidates_view,
    get_data_view,
    get_history_view,
    get_paths_view,
    get_resource_view,
    get_stats_view,
    get_status,
)
from frontierwright.tui import FrontierwrightApp
from frontierwright.tui.app import (
    ActionCenterScreen,
    CandidateScreen,
    HelpScreen,
    WorkflowFormScreen,
    _compare_text,
    _edition_help_text,
    _stat_line,
)


def test_candidate_compare_keeps_keyboard_shortcuts_visible() -> None:
    view = CompareView(
        champion_model_id="champion",
        candidate_model_id="candidate",
        candidate_status="PENDING",
        scale_comparable=True,
        champion_stats={"general": 50.0},
        candidate_stats={"general": 75.0},
        deltas={"general": 25.0},
    )

    rendered = _compare_text(view)
    assert "P Promote" in rendered
    assert "R Reject" in rendered
    assert "Esc Back" in rendered
    assert "[P]" not in rendered


def test_edition_help_and_stat_precision_are_intentionally_different() -> None:
    uncertainty: dict[str, object] = {
        "stat_lower": 55.0,
        "stat_upper": 90.0,
        "sample_size": 16,
    }

    academy_help = _edition_help_text("en", "ACADEMY")
    studio_help = _edition_help_text("en", "STUDIO")
    lab_help = _edition_help_text("en", "LAB")
    assert "UNDERSTAND BY DOING" in academy_help
    assert "FIT THE MODEL TO YOU" in studio_help
    assert "CONTROLLED FRONTIER DEVELOPMENT" in lab_help

    academy_stat = _stat_line(
        axis="general", value=75.0, uncertainty=uncertainty, edition_profile="ACADEMY"
    )
    studio_stat = _stat_line(
        axis="general", value=75.0, uncertainty=uncertainty, edition_profile="STUDIO"
    )
    lab_stat = _stat_line(
        axis="general", value=75.0, uncertainty=uncertainty, edition_profile="LAB"
    )
    assert "95% evidence range" in academy_stat
    assert "[55–90]" in studio_stat
    assert "CI95[55.0, 90.0] n=16 Wilson" in lab_stat


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
            "workload",
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


def test_tui_workload_experience_differs_by_edition() -> None:
    workload = WorkloadView(
        configured=True,
        profile_id="workload-demo",
        profile_hash="sha256:demo",
        profile_name="Korean economics work",
        source="EXPLICIT_USER",
        profile={
            "languages": ["ko", "en"],
            "domains": ["economics", "coding"],
            "task_weights": {"research": 2.0, "coding": 1.0},
            "context_tokens_p50": 2048,
            "context_tokens_p95": 8192,
            "max_latency_seconds": 2.0,
            "min_tokens_per_second": 15.0,
            "privacy": "PRIVATE",
            "critical_floors": {"general": 60.0},
        },
    )

    academy = FrontierwrightApp(view=make_view(), workload=workload)
    studio = FrontierwrightApp(
        view=replace(
            make_view(),
            edition_profile="STUDIO",
            edition_name="Frontierwright Studio",
        ),
        workload=workload,
    )
    lab = FrontierwrightApp(
        view=replace(
            make_view(),
            edition_profile="LAB",
            edition_name="Frontierwright Lab",
        ),
        workload=workload,
    )

    academy_text = academy._workload_text()
    studio_text = studio._workload_text()
    lab_text = lab._workload_text()

    assert "WHAT SHOULD THIS MODEL LEARN TO FIT?" in academy_text
    assert "fake capability points" in academy_text
    assert "Hash:" not in academy_text

    assert "YOUR WORKLOAD" in studio_text
    assert "Next evidence: profile the current Champion" in studio_text

    assert "WORKLOAD / SERVING REQUIREMENTS" in lab_text
    assert "Hash: sha256:demo" in lab_text
    assert "unknown constraints remain UNKNOWN" in lab_text


def test_tui_action_priority_differs_by_edition() -> None:
    academy = FrontierwrightApp(view=make_view())
    studio = FrontierwrightApp(view=replace(make_view(), edition_profile="STUDIO"))
    lab = FrontierwrightApp(view=replace(make_view(), edition_profile="LAB"))

    academy_ids = [item.action_id for item in academy._action_items()[:3]]
    studio_ids = [item.action_id for item in studio._action_items()[:3]]
    lab_ids = [item.action_id for item in lab._action_items()[:3]]

    assert academy_ids == ["add_dataset", "detect_resources", "set_workload"]
    assert studio_ids == ["detect_resources", "set_workload", "add_dataset"]
    assert lab_ids == ["set_workload", "detect_resources", "add_dataset"]


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


def activate_tui_capability_profile(root: Path) -> None:
    registry = Registry(root)
    model = registry.get_model("champion")
    receipt = EvaluationReceipt(
        receipt_id="receipt-tui-capability",
        model_id=model.model_id,
        model_fingerprint=model.fingerprint,
        evaluator_id="tui-capability-fixture",
        evaluator_version="1",
        conditions={},
        measurements=tuple(
            RawMeasurement(
                task_id=(f"{CAPABILITY_V1_BUNDLE_ID}.{axis.value.lower()}"),
                task_version=CAPABILITY_V1_BUNDLE_VERSION,
                metric="accuracy",
                value=0.5,
                higher_is_better=True,
            )
            for axis in Axis
        ),
    )
    registry.activate_capability_profile(
        receipt,
        CAPABILITY_V1_SCALE,
        apply_scale(receipt, CAPABILITY_V1_SCALE),
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

        await pilot.press("8")
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

        # Academy: dataset -> resources -> optional workload -> build intent.
        await pilot.press("j")
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


async def test_tui_keyboard_sets_scale_bound_measured_build_targets(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    setup_candidate_project(project)
    activate_tui_capability_profile(project)

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
    assert app.build.mode == "TARGETS_FLOORS"
    actions = app._action_items()
    action_index = [item.action_id for item in actions].index("build_targets")

    async with app.run_test(size=(140, 55)) as pilot:
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ActionCenterScreen)
        for _ in range(action_index):
            await pilot.press("j")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, WorkflowFormScreen)

        # First field is General target. Other target/floor fields stay intentionally blank.
        await pilot.press("1", "2", "0")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert not isinstance(app.screen, WorkflowFormScreen)

    build = get_build_view(project)
    assert build.mode == "TARGETS_FLOORS"
    assert build.configured is True
    assert build.targets == {"general": 120}
    assert build.floors == {}
    assert build.scale_bound is True
    assert build.scale_hash == CAPABILITY_V1_SCALE.sha256


async def test_tui_keyboard_measures_selected_candidate_capability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    setup_candidate_project(project)

    def fake_capability_backend(
        argv_template,
        *,
        environment_overrides,
        request_path,
        timeout_seconds,
    ):
        del argv_template, environment_overrides, timeout_seconds
        request = __import__("json").loads(request_path.read_text(encoding="utf-8"))
        assert request["operation"] == "capability_v1"
        axes = []
        for axis in ("general", "reasoning", "math", "coding"):
            axes.append(
                {
                    "axis": axis,
                    "task_id": f"{CAPABILITY_V1_BUNDLE_ID}.{axis}",
                    "task_version": CAPABILITY_V1_BUNDLE_VERSION,
                    "accuracy": 0.5,
                    "correct": 8,
                    "total": 16,
                    "mean_correct_margin_nats": 0.1,
                }
            )
        return {
            "schema_version": 1,
            "ok": True,
            "operation": "capability_v1",
            "metrics": {
                "backend_id": "frontierwright-reference-pytorch-v1",
                "bundle_id": CAPABILITY_V1_BUNDLE_ID,
                "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
                "bundle_hash": capability_v1_bundle_hash(),
                "scoring": CAPABILITY_V1_SCORING,
                "preset": "zero-8m",
                "device": "cpu",
                "axes": axes,
                "task_counts": capability_v1_task_counts(),
                "elapsed_seconds": 0.1,
                "parameter_count": 1,
                "tokenizer_fingerprint": "fixture-tokenizer",
                "python_version": "fixture",
                "torch_version": "fixture",
            },
        }

    monkeypatch.setattr(service_module, "run_structured_command", fake_capability_backend)

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
    actions = app._action_items()
    action_index = [item.action_id for item in actions].index("measure_candidate_capability")

    async with app.run_test(size=(140, 55)) as pilot:
        await pilot.press("a")
        await pilot.pause()
        for _ in range(action_index):
            await pilot.press("j")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, WorkflowFormScreen)
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, CandidateScreen)
        assert app.screen.compare.candidate_model_id == "candidate-1"

    candidate_stats = get_stats_view(project, "candidate-1")
    assert candidate_stats.measured is True
    assert candidate_stats.scale_hash == CAPABILITY_V1_SCALE.sha256
    assert candidate_stats.stats == {
        "general": 100.0,
        "reasoning": 100.0,
        "math": 100.0,
        "coding": 100.0,
    }
