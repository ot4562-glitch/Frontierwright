from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Static, TabbedContent, TabPane

from frontierwright.data import DatasetClassification, DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.editions import EditionProfile
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import HardBudgets, PermissionLevel
from frontierwright.i18n import tr
from frontierwright.paths import TrainingPathId
from frontierwright.reference_backend import backend_spec_payload
from frontierwright.service import (
    BuildView,
    CandidateView,
    CompareView,
    DataView,
    HistoryView,
    PathsView,
    ResourceView,
    StatusView,
    add_local_dataset,
    birth_zero_model,
    calibrate_training_plan,
    compare_candidate,
    compare_candidate_evaluation,
    create_training_plan,
    detect_resources,
    execute_training_plan,
    get_build_view,
    get_candidates_view,
    get_data_view,
    get_history_view,
    get_paths_view,
    get_resource_view,
    get_status,
    get_tokenizers_view,
    import_local_model,
    initialize_project,
    promote_candidate,
    reconcile_training_run,
    reject_candidate,
    set_build_intent,
    train_project_tokenizer,
)

TAB_KEYS = (
    ("1", "character"),
    ("2", "build"),
    ("3", "paths"),
    ("4", "resources"),
    ("5", "data"),
    ("6", "history"),
    ("7", "candidates"),
)


def _human_bytes(value: object) -> str:
    if not isinstance(value, int):
        return "UNKNOWN"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _compare_text(view: CompareView) -> str:
    lines = [
        "CANDIDATE COMPARISON",
        "",
        f"Champion  {view.champion_model_id}",
        f"Candidate {view.candidate_model_id} · {view.candidate_status}",
        f"Comparable {'YES' if view.scale_comparable else 'NO'}",
    ]
    if view.scale_reason:
        lines.append(view.scale_reason)
    lines.extend(["", "AXIS        CHAMPION  CANDIDATE  DELTA"])
    for axis in ("general", "reasoning", "math", "coding"):
        champion = view.champion_stats.get(axis)
        candidate = view.candidate_stats.get(axis)
        delta = view.deltas.get(axis)
        delta_text = "?" if delta is None else f"{delta:+g}"
        lines.append(
            f"{axis.title():10} "
            f"{str(champion) if champion is not None else '?':>8}  "
            f"{str(candidate) if candidate is not None else '?':>9}  "
            f"{delta_text:>5}"
        )
    if view.raw_evaluation_comparisons:
        lines.extend(["", "RAW EVALUATION"])
        for evidence in view.raw_evaluation_comparisons:
            lines.append(
                f"{evidence.get('pack_id')}@{evidence.get('pack_version')} · "
                f"dataset={evidence.get('dataset_id')}"
            )
            measurements = evidence.get("measurements")
            if isinstance(measurements, list):
                for item in measurements:
                    if not isinstance(item, dict):
                        continue
                    improvement = item.get("improvement_delta")
                    improvement_text = (
                        f"{improvement:+g}"
                        if isinstance(improvement, (int, float))
                        and not isinstance(improvement, bool)
                        else "?"
                    )
                    lines.append(
                        f"  {item.get('metric')}: "
                        f"{item.get('champion_value')} -> {item.get('candidate_value')} "
                        f"(improvement {improvement_text})"
                    )
    if view.build_constraints:
        lines.extend(["", "BUILD"])
        for item in view.build_constraints:
            lines.append(
                f"{str(item.get('axis')).title():10} {item.get('kind')} "
                f"{item.get('threshold')}  {item.get('status')}"
            )
    lines.extend(
        [
            "",
            f"Promotion eligible: {'YES' if view.promotion_eligible else 'NO'}",
        ]
    )
    for blocker in view.promotion_blockers:
        override = blocker.get("override")
        suffix = f" · {override}" if override else ""
        lines.append(f"Blocked: {blocker.get('code')}{suffix}")
    lines.extend(["", "[P] Promote   [R] Reject   [Esc] Back"])
    return "\n".join(lines)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
        Binding("enter", "dismiss", "Back"),
        Binding("?", "dismiss", "Back"),
    ]

    def __init__(self, language: str) -> None:
        super().__init__()
        self.language = language

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static(tr(self.language, "help"), id="help-text")


class CandidateScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Back"),
        Binding("p", "promote", "Promote"),
        Binding("u", "promote_unmeasured", "Promote unmeasured"),
        Binding("v", "promote_build_override", "Promote build override"),
        Binding("r", "reject", "Reject"),
    ]

    def __init__(self, compare: CompareView) -> None:
        super().__init__()
        self.compare = compare

    def compose(self) -> ComposeResult:
        with Vertical(id="candidate-dialog"):
            yield Static(_compare_text(self.compare), id="candidate-compare")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_promote(self) -> None:
        self.dismiss("promote")

    def action_promote_unmeasured(self) -> None:
        self.dismiss("promote_unmeasured")

    def action_promote_build_override(self) -> None:
        self.dismiss("promote_build_override")

    def action_reject(self) -> None:
        self.dismiss("reject")




@dataclass(frozen=True)
class ActionItem:
    action_id: str
    title: str
    description: str


@dataclass(frozen=True)
class FormField:
    key: str
    label: str
    default: str = ""
    placeholder: str = ""


class ActionCenterScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Back"),
        Binding("j", "next_item", "Next"),
        Binding("down", "next_item", "Next", show=False),
        Binding("k", "previous_item", "Prev"),
        Binding("up", "previous_item", "Prev", show=False),
        Binding("enter", "choose", "Choose"),
    ]

    def __init__(self, items: list[ActionItem]) -> None:
        super().__init__()
        self.items = items
        self.index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="action-dialog"):
            yield Static(self._text(), id="action-list")

    def _text(self) -> str:
        lines = [
            "ACTION CENTER",
            "",
            "Every action below uses the same Frontierwright core as CLI/JSON.",
            "",
        ]
        for index, item in enumerate(self.items):
            marker = ">" if index == self.index else " "
            lines.append(f"{marker} {item.title}")
            lines.append(f"    {item.description}")
        lines.extend(["", "j/k or ↑/↓ select · Enter choose · Esc back"])
        return "\n".join(lines)

    def _refresh(self) -> None:
        self.query_one("#action-list", Static).update(self._text())

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_next_item(self) -> None:
        if not self.items:
            return
        self.index = (self.index + 1) % len(self.items)
        self._refresh()

    def action_previous_item(self) -> None:
        if not self.items:
            return
        self.index = (self.index - 1) % len(self.items)
        self._refresh()

    def action_choose(self) -> None:
        if not self.items:
            self.dismiss(None)
            return
        self.dismiss(self.items[self.index].action_id)


class WorkflowFormScreen(ModalScreen[dict[str, str] | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Back"),
        Binding("ctrl+s", "submit", "Submit"),
    ]

    def __init__(
        self,
        *,
        title: str,
        description: str,
        fields: list[FormField],
    ) -> None:
        super().__init__()
        self.title_text = title
        self.description = description
        self.fields = fields

    def compose(self) -> ComposeResult:
        with Vertical(id="form-dialog"):
            yield Static(
                f"{self.title_text}\n\n{self.description}\n",
                id="form-title",
            )
            for index, field in enumerate(self.fields):
                yield Static(field.label)
                yield Input(
                    value=field.default,
                    placeholder=field.placeholder,
                    id=f"form-field-{index}",
                )
            yield Static(
                "\nTab/Shift+Tab move · Ctrl+S submit · Esc cancel",
                id="form-help",
            )

    def on_mount(self) -> None:
        if self.fields:
            self.query_one("#form-field-0", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        result: dict[str, str] = {}
        for index, field in enumerate(self.fields):
            result[field.key] = self.query_one(
                f"#form-field-{index}",
                Input,
            ).value.strip()
        self.dismiss(result)


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("enter", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "No"),
    ]

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.title_text = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(
                f"{self.title_text}\n\n{self.message}\n\n[Y/Enter] Confirm   [N/Esc] Cancel"
            )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class FrontierwrightApp(App[None]):
    """Keyboard-only human client over shared Frontierwright service views."""

    CSS = """
    #character-sheet, #build-view, #paths-view, #resources-view,
    #data-view, #history-view, #candidates-view {
        padding: 1 2;
    }
    #help-dialog, #candidate-dialog, #action-dialog, #form-dialog, #confirm-dialog {
        width: 88%;
        height: auto;
        max-height: 90%;
        border: round $accent;
        padding: 2;
        background: $surface;
        align: center middle;
    }
    #form-dialog Input {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("a", "action_center", "Actions"),
        Binding("?", "show_help", "Help"),
        Binding("escape", "back", "Back"),
        Binding("h", "previous_tab", "Prev"),
        Binding("l", "next_tab", "Next"),
        Binding("left", "previous_tab", "Prev", show=False),
        Binding("right", "next_tab", "Next", show=False),
        Binding("j", "candidate_next", "Next", show=False),
        Binding("down", "candidate_next", "Next", show=False),
        Binding("k", "candidate_previous", "Prev", show=False),
        Binding("up", "candidate_previous", "Prev", show=False),
        Binding("enter", "open_candidate", "Open", show=False),
        *[
            Binding(key, f"select_tab('{tab_id}')", f"{key}:{tab_id}", show=False)
            for key, tab_id in TAB_KEYS
        ],
    ]

    def __init__(
        self,
        *,
        view: StatusView,
        resources: ResourceView | None = None,
        build: BuildView | None = None,
        data: DataView | None = None,
        paths: PathsView | None = None,
        candidates: CandidateView | None = None,
        history: HistoryView | None = None,
        root: Path = Path("."),
        language: str = "en",
    ) -> None:
        super().__init__()
        self.root = root.resolve()
        self.view = view
        self.resources = resources or ResourceView()
        self.build = build or BuildView(mode=view.build_mode)
        self.data = data or DataView()
        self.paths = paths or PathsView()
        self.candidates = candidates or CandidateView()
        self.history = history or HistoryView()
        self.candidate_index = 0
        self.last_run_id: str | None = None
        self.language = language

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with TabbedContent(initial="character", id="main-tabs"):
            with TabPane(tr(self.language, "character"), id="character"):
                yield Static(self._character_text(), id="character-sheet")
            with TabPane(tr(self.language, "build"), id="build"):
                yield Static(self._build_text(), id="build-view")
            with TabPane(tr(self.language, "paths"), id="paths"):
                yield Static(self._paths_text(), id="paths-view")
            with TabPane(tr(self.language, "resources"), id="resources"):
                yield Static(self._resources_text(), id="resources-view")
            with TabPane(tr(self.language, "data"), id="data"):
                yield Static(self._data_text(), id="data-view")
            with TabPane(tr(self.language, "history"), id="history"):
                yield Static(self._history_text(), id="history-view")
            with TabPane(tr(self.language, "candidates"), id="candidates"):
                yield Static(self._candidates_text(), id="candidates-view")
        yield Footer()

    def _character_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")
        lines = [
            self.view.nickname or "?",
            f"Edition: {self.view.edition_name or self.view.edition_profile or '?'}",
        ]
        if self.view.edition_tagline:
            lines.append(self.view.edition_tagline)
        lines.extend(
            [
                f"Origin: {self.view.origin}",
                f"History: {self.view.history_confidence}",
                f"State: {self.view.measurement_state}",
            ]
        )
        if self.view.champion_model_id:
            lines.extend(
                [
                    f"Model: {self.view.champion_model_id}",
                    f"Format: {self.view.model_format}",
                    f"Trainable: {'YES' if self.view.trainable else 'NO'}",
                    f"Fingerprint: {self.view.model_fingerprint}",
                ]
            )
        lines.append("")
        for axis in ("general", "reasoning", "math", "coding"):
            value = self.view.stats.get(axis)
            lines.append(f"{axis.title():10} {value if value is not None else '?'}")
        return "\n".join(lines)

    def _build_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")

        lines = [f"Mode: {self.build.mode or 'NOT_READY'}"]
        if self.build.archetype:
            lines.append(f"Archetype: {self.build.archetype}")
        if self.build.priorities:
            lines.extend(["", "PRIORITIES"])
            for axis, value in sorted(self.build.priorities.items()):
                lines.append(f"{axis.title():10} {value}")
        if self.build.targets:
            lines.extend(["", "TARGETS"])
            for axis, value in sorted(self.build.targets.items()):
                lines.append(f"{axis.title():10} {value}")
        if self.build.floors:
            lines.extend(["", "FLOORS"])
            for axis, value in sorted(self.build.floors.items()):
                lines.append(f"{axis.title():10} {value}")
        if self.build.reason:
            lines.extend(["", self.build.reason])
        return "\n".join(lines)

    def _paths_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")

        lines: list[str] = []
        for item in self.paths.paths:
            lines.append(
                f"[{item.get('availability')}] {item.get('title')} "
                f"({item.get('path_id')})"
            )
            blockers = item.get("blockers")
            if isinstance(blockers, list):
                for blocker in blockers:
                    lines.append(f"  Missing: {blocker}")
            checks = item.get("next_checks")
            if isinstance(checks, list):
                for check in checks:
                    lines.append(f"  Next: {check}")
            ready_plan = item.get("ready_plan_id")
            if ready_plan:
                lines.append(f"  Ready plan: {ready_plan}")
            lines.append("")

        if self.paths.recommendation_reason:
            lines.append(self.paths.recommendation_reason)
        return "\n".join(lines).rstrip()

    def _resources_text(self) -> str:
        if not self.resources.available:
            return "NOT DETECTED\nRun: frontierwright resources detect"
        snapshot = self.resources.snapshot
        lines = [
            f"Provenance: {self.resources.provenance}",
            f"CPU: {snapshot.get('cpu_model', 'UNKNOWN')}",
            f"Logical CPUs: {snapshot.get('cpu_logical_count', 'UNKNOWN')}",
            (
                "RAM: "
                f"{_human_bytes(snapshot.get('ram_total_bytes'))} total · "
                f"{_human_bytes(snapshot.get('ram_available_bytes'))} available"
            ),
            (
                "Disk: "
                f"{_human_bytes(snapshot.get('disk_free_bytes'))} free / "
                f"{_human_bytes(snapshot.get('disk_total_bytes'))} total"
            ),
        ]
        gpus = snapshot.get("gpus")
        if isinstance(gpus, list) and gpus:
            for index, gpu in enumerate(gpus, start=1):
                if isinstance(gpu, dict):
                    lines.append(
                        f"GPU {index}: {gpu.get('vendor', '?')} {gpu.get('name', '?')} · "
                        f"{_human_bytes(gpu.get('memory_total_bytes'))} VRAM"
                    )
        else:
            lines.append("GPU: none detected")
        lines.extend(
            [
                f"Torch: {snapshot.get('torch_version') or 'not detected'}",
                f"CUDA toolkit: {snapshot.get('cuda_toolkit_version') or 'not detected'}",
                f"ROCm: {snapshot.get('rocm_version') or 'not detected'}",
                "bf16/fp16: UNKNOWN until backend-specific calibration",
            ]
        )
        return "\n".join(lines)

    def _data_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")
        if not self.data.datasets:
            return "No datasets registered.\nUse: frontierwright data add ..."

        lines: list[str] = []
        for item in self.data.datasets:
            lines.append(
                f"{item.get('name')} · {item.get('role')} · "
                f"{_human_bytes(item.get('total_bytes'))}"
            )
            lines.append(f"  Provenance: {item.get('provenance')}")
            lines.append(f"  Classification: {item.get('classification') or 'UNKNOWN'}")
            lines.append(f"  Fingerprint: {item.get('fingerprint')}")
            lines.append(f"  License: {item.get('license') or 'UNKNOWN'}")
            if item.get("managed"):
                lines.append(f"  Managed from: {item.get('source_dataset_id')}")
                lines.append(f"  Recipe: {item.get('preparation_recipe_id')}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _history_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")
        lines = [
            f"History confidence: {self.view.history_confidence}",
            "",
        ]
        if not self.history.events:
            lines.append("No development events.")
        else:
            for event in self.history.events[-30:]:
                lines.append(
                    f"{event.get('sequence'):>4}  {event.get('kind')}  "
                    f"{event.get('recorded_at')}"
                )
        return "\n".join(lines)

    def _candidates_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")
        if not self.candidates.candidates:
            return "No candidates.\nTraining never replaces the champion automatically."

        lines = [
            "j/k or ↑/↓ select · Enter compare",
            "",
        ]
        for index, item in enumerate(self.candidates.candidates):
            marker = ">" if index == self.candidate_index else " "
            stats = item.get("stats")
            stat_text = ""
            if isinstance(stats, dict):
                stat_text = " ".join(
                    f"{str(axis)[0].upper()}{value if value is not None else '?'}"
                    for axis, value in stats.items()
                )
            lines.append(
                f"{marker} {item.get('model_id')}  [{item.get('status')}] "
                f"{item.get('path_id') or ''}"
            )
            if stat_text:
                lines.append(f"    {stat_text}")
        return "\n".join(lines)

    def _tabs(self) -> TabbedContent:
        return self.query_one("#main-tabs", TabbedContent)

    def _on_candidates_tab(self) -> bool:
        return self._tabs().active == "candidates"

    def _refresh_all(self) -> None:
        self.view = get_status(self.root)
        self.resources = get_resource_view(self.root)
        self.build = get_build_view(self.root)
        self.data = get_data_view(self.root)
        self.paths = get_paths_view(self.root)
        self.candidates = get_candidates_view(self.root)
        self.history = get_history_view(self.root)
        if self.candidates.candidates:
            self.candidate_index = min(
                self.candidate_index,
                len(self.candidates.candidates) - 1,
            )
        else:
            self.candidate_index = 0
        self.query_one("#character-sheet", Static).update(self._character_text())
        self.query_one("#build-view", Static).update(self._build_text())
        self.query_one("#paths-view", Static).update(self._paths_text())
        self.query_one("#resources-view", Static).update(self._resources_text())
        self.query_one("#data-view", Static).update(self._data_text())
        self.query_one("#history-view", Static).update(self._history_text())
        self.query_one("#candidates-view", Static).update(self._candidates_text())

    def action_previous_tab(self) -> None:
        tabs = self._tabs()
        ids = [tab_id for _, tab_id in TAB_KEYS]
        current = tabs.active
        index = ids.index(current) if current in ids else 0
        tabs.active = ids[(index - 1) % len(ids)]

    def action_next_tab(self) -> None:
        tabs = self._tabs()
        ids = [tab_id for _, tab_id in TAB_KEYS]
        current = tabs.active
        index = ids.index(current) if current in ids else 0
        tabs.active = ids[(index + 1) % len(ids)]

    def action_select_tab(self, tab_id: str) -> None:
        if tab_id in {item[1] for item in TAB_KEYS}:
            self._tabs().active = tab_id

    def action_candidate_next(self) -> None:
        if not self._on_candidates_tab() or not self.candidates.candidates:
            return
        self.candidate_index = (self.candidate_index + 1) % len(
            self.candidates.candidates
        )
        self.query_one("#candidates-view", Static).update(self._candidates_text())

    def action_candidate_previous(self) -> None:
        if not self._on_candidates_tab() or not self.candidates.candidates:
            return
        self.candidate_index = (self.candidate_index - 1) % len(
            self.candidates.candidates
        )
        self.query_one("#candidates-view", Static).update(self._candidates_text())

    def action_open_candidate(self) -> None:
        if not self._on_candidates_tab() or not self.candidates.candidates:
            return
        item = self.candidates.candidates[self.candidate_index]
        model_id = item.get("model_id")
        if not isinstance(model_id, str):
            return
        try:
            compare = compare_candidate(self.root, model_id)
        except FrontierwrightError as exc:
            self.notify(str(exc), severity="error")
            return
        self.push_screen(
            CandidateScreen(compare),
            lambda result: self._candidate_result(model_id, result),
        )

    def _candidate_result(self, model_id: str, result: str | None) -> None:
        if result is None:
            return
        try:
            if result == "promote":
                promote_candidate(self.root, model_id)
                self.notify("Candidate promoted.")
            elif result == "promote_unmeasured":
                promote_candidate(
                    self.root,
                    model_id,
                    allow_unmeasured=True,
                )
                self.notify("Candidate promoted with explicit unmeasured override.")
            elif result == "promote_build_override":
                promote_candidate(
                    self.root,
                    model_id,
                    allow_build_violations=True,
                )
                self.notify("Candidate promoted with explicit build-violation override.")
            elif result == "reject":
                reject_candidate(self.root, model_id)
                self.notify("Candidate rejected.")
            else:
                return
        except FrontierwrightError as exc:
            self.notify(str(exc), severity="error")
            return
        self._refresh_all()


    def _first_dataset_id(self, role: DatasetRole | None = None) -> str:
        for item in self.data.datasets:
            if role is not None and item.get("role") != role.value:
                continue
            dataset_id = item.get("dataset_id")
            if isinstance(dataset_id, str):
                return dataset_id
        return ""

    def _selected_candidate_model_id(self) -> str:
        if not self.candidates.candidates:
            return ""
        index = min(self.candidate_index, len(self.candidates.candidates) - 1)
        model_id = self.candidates.candidates[index].get("model_id")
        return model_id if isinstance(model_id, str) else ""

    def _first_plannable_path(self) -> str:
        for item in self.paths.paths:
            if item.get("availability") == "PLANNABLE":
                path_id = item.get("path_id")
                if isinstance(path_id, str):
                    return path_id
        return ""

    def _first_ready_plan(self) -> str:
        for item in self.paths.paths:
            if item.get("availability") == "READY":
                plan_id = item.get("ready_plan_id")
                if isinstance(plan_id, str):
                    return plan_id
        return ""

    def _default_dataset_for_path(self, path_id: str) -> str:
        if path_id in {
            TrainingPathId.FROM_SCRATCH_PRETRAINING.value,
            TrainingPathId.CONTINUED_PRETRAINING.value,
        }:
            return self._first_dataset_id(DatasetRole.PRETRAIN)
        if path_id in {
            TrainingPathId.FULL_SFT.value,
            TrainingPathId.LORA_SFT.value,
            TrainingPathId.QLORA_SFT.value,
        }:
            return self._first_dataset_id(DatasetRole.SFT)
        return self._first_dataset_id()

    def _reference_backend_spec_path(self) -> Path:
        return self.root / ".frontierwright" / "backends" / "reference-pytorch-v1.json"

    def _write_reference_backend_spec(self, python_executable: str) -> Path:
        output = self._reference_backend_spec_path()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = backend_spec_payload(python_executable)
        output.write_text(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    def _reference_python_default(self) -> str:
        spec_path = self._reference_backend_spec_path()
        if spec_path.is_file():
            try:
                payload = json.loads(spec_path.read_text(encoding="utf-8"))
                argv = payload.get("train_argv") if isinstance(payload, dict) else None
                if (
                    isinstance(argv, list)
                    and argv
                    and isinstance(argv[0], str)
                    and argv[0]
                ):
                    return argv[0]
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        return sys.executable

    def _action_items(self) -> list[ActionItem]:
        if not self.view.initialized:
            return [
                ActionItem(
                    "initialize_project",
                    "Create Frontierwright project",
                    "Choose character name, origin, edition, and UI language.",
                )
            ]

        items = [
            ActionItem(
                "detect_resources",
                "Detect local resources",
                "Measure CPU/RAM/disk/GPU and refresh feasibility.",
            ),
            ActionItem(
                "add_dataset",
                "Register local dataset",
                "Add PRETRAIN, SFT, or PREFERENCE data with an explicit classification.",
            ),
        ]
        if self.build.mode == "INTENT":
            items.append(
                ActionItem(
                    "build_intent",
                    "Set build intent",
                    "Choose an archetype and real capability priorities before stats exist.",
                )
            )

        if self.view.champion_model_id is None:
            if self.view.origin == ModelOrigin.ZERO.value:
                if self._first_dataset_id(DatasetRole.PRETRAIN):
                    items.append(
                        ActionItem(
                            "birth_tokenizer",
                            "Train tokenizer",
                            "Train and freeze a tokenizer artifact from registered PRETRAIN data.",
                        )
                    )
                items.append(
                    ActionItem(
                        "birth_zero",
                        "Birth zero model",
                        "Materialize real initialized weights; training has not happened yet.",
                    )
                )
            elif self.view.origin in {
                ModelOrigin.IMPORTED_LOCAL.value,
                ModelOrigin.INTERNAL_LAB.value,
            }:
                items.append(
                    ActionItem(
                        "import_model",
                        "Import model",
                        "Fingerprint and register an existing local/trainable checkpoint.",
                    )
                )

        if self._first_plannable_path():
            items.append(
                ActionItem(
                    "reference_plan",
                    "Create + calibrate reference plan",
                    "Pin model/data/config/resources, then run representative calibration.",
                )
            )

        if self._first_ready_plan():
            items.append(
                ActionItem(
                    "train_ready",
                    "Dry-run + execute READY plan",
                    "Revalidate the plan, dry-run it, then ask before real training.",
                )
            )

        if self.last_run_id is not None:
            items.append(
                ActionItem(
                    "reconcile_run",
                    "Reconcile latest run",
                    "Read durable executor evidence without launching duplicate training.",
                )
            )

        if self.candidates.candidates and self.view.champion_model_id is not None:
            items.append(
                ActionItem(
                    "evaluate_candidate",
                    "Evaluate selected candidate vs champion",
                    "Run the same raw evaluation pack on both exact model fingerprints.",
                )
            )
        return items

    def action_action_center(self) -> None:
        self.push_screen(
            ActionCenterScreen(self._action_items()),
            self._action_center_result,
        )

    def _action_center_result(self, action_id: str | None) -> None:
        if action_id is None:
            return
        if action_id == "initialize_project":
            self.push_screen(
                WorkflowFormScreen(
                    title="CREATE FRONTIERWRIGHT PROJECT",
                    description=(
                        "Academy usually starts ZERO; Studio uses IMPORTED_LOCAL; "
                        "Lab uses INTERNAL_LAB. Profiles can change later without "
                        "rewriting lineage."
                    ),
                    fields=[
                        FormField("name", "Character / project name", "My Model"),
                        FormField("origin", "Origin", "ZERO"),
                        FormField("edition", "Edition", "ACADEMY"),
                        FormField("language", "UI language (en/ko)", self.language),
                    ],
                ),
                self._submit_initialize_project,
            )
            return

        if action_id == "detect_resources":
            try:
                detect_resources(self.root)
                self._refresh_all()
                self.notify("Resources detected.")
            except FrontierwrightError as exc:
                self.notify(str(exc), severity="error")
            return

        if action_id == "add_dataset":
            self.push_screen(
                WorkflowFormScreen(
                    title="REGISTER LOCAL DATASET",
                    description=(
                        "Nothing is uploaded. LOCAL_USER data defaults to PRIVATE. "
                        "Role must be PRETRAIN, SFT, or PREFERENCE."
                    ),
                    fields=[
                        FormField("source", "Source file/directory"),
                        FormField("name", "Display name", "Training data"),
                        FormField("role", "Role", "PRETRAIN"),
                        FormField("classification", "Classification", "PRIVATE"),
                    ],
                ),
                self._submit_add_dataset,
            )
            return

        if action_id == "build_intent":
            self.push_screen(
                WorkflowFormScreen(
                    title="SET BUILD INTENT",
                    description="Priorities are integer relative weights. They are not fake stats.",
                    fields=[
                        FormField("archetype", "Archetype", "Balanced"),
                        FormField("general", "General priority", "1"),
                        FormField("reasoning", "Reasoning priority", "1"),
                        FormField("math", "Math priority", "1"),
                        FormField("coding", "Coding priority", "1"),
                    ],
                ),
                self._submit_build_intent,
            )
            return

        if action_id == "birth_tokenizer":
            self.push_screen(
                WorkflowFormScreen(
                    title="TRAIN TOKENIZER",
                    description="Tokenizer birth must happen before zero-model birth.",
                    fields=[
                        FormField(
                            "dataset_id",
                            "PRETRAIN dataset ID",
                            self._first_dataset_id(DatasetRole.PRETRAIN),
                        ),
                        FormField("vocab_size", "Requested vocabulary size", "512"),
                    ],
                ),
                self._submit_tokenizer_birth,
            )
            return

        if action_id == "birth_zero":
            tokenizers = get_tokenizers_view(self.root).tokenizers
            tokenizer_id = ""
            if tokenizers:
                raw_id = tokenizers[-1].get("artifact_id")
                tokenizer_id = raw_id if isinstance(raw_id, str) else ""
            self.push_screen(
                WorkflowFormScreen(
                    title="BIRTH ZERO MODEL",
                    description=(
                        "This materializes a real root checkpoint. Capability remains unmeasured."
                    ),
                    fields=[
                        FormField("preset", "Preset", "zero-8m"),
                        FormField("seed", "Initialization seed", "42"),
                        FormField(
                            "python",
                            "Training Python executable",
                            self._reference_python_default(),
                        ),
                        FormField(
                            "tokenizer_artifact_id",
                            "Tokenizer artifact ID (optional)",
                            tokenizer_id,
                        ),
                    ],
                ),
                self._submit_zero_birth,
            )
            return

        if action_id == "import_model":
            self.push_screen(
                WorkflowFormScreen(
                    title="IMPORT LOCAL MODEL",
                    description=(
                        "The model is fingerprinted from local bytes. "
                        "History stays UNKNOWN/PARTIAL unless evidence is supplied."
                    ),
                    fields=[
                        FormField("source", "Model directory/file"),
                        FormField("history_manifest", "History manifest path (optional)"),
                    ],
                ),
                self._submit_import_model,
            )
            return

        if action_id == "reference_plan":
            path_id = self._first_plannable_path()
            self.push_screen(
                WorkflowFormScreen(
                    title="CREATE + CALIBRATE REFERENCE PLAN",
                    description=(
                        "Creates an EXECUTE_SINGLE plan with max_runs=1, then calibrates "
                        "the same reference workload. JSON config is pinned into plan identity."
                    ),
                    fields=[
                        FormField("path_id", "Training path", path_id),
                        FormField(
                            "dataset_id",
                            "Dataset ID",
                            self._default_dataset_for_path(path_id),
                        ),
                        FormField(
                            "python",
                            "Training Python executable",
                            self._reference_python_default(),
                        ),
                        FormField("config_json", "Backend config JSON", "{}"),
                    ],
                ),
                self._submit_reference_plan,
            )
            return

        if action_id == "train_ready":
            self.push_screen(
                WorkflowFormScreen(
                    title="TRAIN READY PLAN",
                    description=(
                        "Frontierwright first performs a dry-run. "
                        "Actual execution requires a second confirmation."
                    ),
                    fields=[
                        FormField("plan_id", "READY plan ID", self._first_ready_plan()),
                        FormField(
                            "python",
                            "Training Python executable",
                            self._reference_python_default(),
                        ),
                    ],
                ),
                self._submit_train_ready,
            )
            return

        if action_id == "reconcile_run":
            self._reconcile_latest_run()
            return

        if action_id == "evaluate_candidate":
            model_id = self._selected_candidate_model_id()
            self.push_screen(
                WorkflowFormScreen(
                    title="EVALUATE CANDIDATE VS CHAMPION",
                    description=(
                        "Runs the deterministic reference held-out LM pack on both models. "
                        "Raw evidence never promotes a candidate automatically."
                    ),
                    fields=[
                        FormField("candidate_model_id", "Candidate model ID", model_id),
                        FormField(
                            "dataset_id",
                            "Evaluation dataset ID",
                            self._first_dataset_id(),
                        ),
                        FormField(
                            "python",
                            "Evaluation Python executable",
                            self._reference_python_default(),
                        ),
                        FormField("device", "Device (auto/cpu/cuda)", "auto"),
                    ],
                ),
                self._submit_candidate_evaluation,
            )

    def _submit_initialize_project(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            language = values["language"].lower()
            if language not in {"en", "ko"}:
                raise ValueError("UI language must be en or ko.")
            view = initialize_project(
                self.root,
                name=values["name"] or "My Model",
                origin=ModelOrigin(values["origin"].upper()),
                language=language,
                edition_profile=EditionProfile(values["edition"].upper()),
            )
            self.language = view.language
            self._refresh_all()
            self.notify(f"Frontierwright project created: {view.project_name}")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_add_dataset(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            source = Path(values["source"]).expanduser()
            name = values["name"] or source.name or "Dataset"
            role = DatasetRole(values["role"].upper())
            classification = DatasetClassification(values["classification"].upper())
            add_local_dataset(
                self.root,
                source,
                name=name,
                role=role,
                classification=classification,
            )
            self._refresh_all()
            self.notify(f"Dataset registered: {name}")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_build_intent(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            priorities = {
                axis: int(values[axis])
                for axis in ("general", "reasoning", "math", "coding")
            }
            set_build_intent(
                self.root,
                archetype=values["archetype"] or "Balanced",
                priorities=priorities,
            )
            self._refresh_all()
            self.notify("Build intent updated.")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_tokenizer_birth(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            view = train_project_tokenizer(
                self.root,
                dataset_id=values["dataset_id"],
                vocab_size=int(values["vocab_size"]),
            )
            self._refresh_all()
            self.notify(f"Tokenizer ready: {view.artifact_id}")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_zero_birth(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            tokenizer_id = values.get("tokenizer_artifact_id") or None
            view = birth_zero_model(
                self.root,
                preset=values["preset"],
                seed=int(values["seed"]),
                python_executable=values["python"],
                tokenizer_artifact_id=tokenizer_id,
            )
            self._refresh_all()
            self.notify(f"Model born: {view.model_id}")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_import_model(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            if self.view.origin is None:
                raise ValueError("Project origin is unknown.")
            origin = ModelOrigin(self.view.origin)
            manifest = values.get("history_manifest") or None
            import_local_model(
                self.root,
                Path(values["source"]).expanduser(),
                origin=origin,
                history_manifest=Path(manifest).expanduser() if manifest else None,
            )
            self._refresh_all()
            self.notify("Model imported.")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_reference_plan(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            config_raw = json.loads(values["config_json"] or "{}")
            if not isinstance(config_raw, dict):
                raise ValueError("Backend config JSON must be an object.")
            config = {str(key): value for key, value in config_raw.items()}
            python_executable = values["python"]
            backend_spec = self._write_reference_backend_spec(python_executable)
            plan = create_training_plan(
                self.root,
                path_id=TrainingPathId(values["path_id"]),
                backend_spec_path=backend_spec,
                dataset_id=values["dataset_id"] or None,
                permission=PermissionLevel.EXECUTE_SINGLE,
                budgets=HardBudgets(max_runs=1),
                config=config,
            )
            if plan.plan_id is None:
                raise FrontierwrightError(
                    "PLAN_ID_MISSING",
                    "Created plan does not expose a durable plan ID.",
                    4,
                )
            calibrated = calibrate_training_plan(
                self.root,
                plan_id=plan.plan_id,
                backend_spec_path=backend_spec,
            )
            self._refresh_all()
            if calibrated.ready:
                self.notify(f"Plan READY: {calibrated.plan_id}")
            else:
                blockers = "; ".join(calibrated.blockers) or "unknown blockers"
                self.notify(f"Plan calibrated but not READY: {blockers}", severity="warning")
        except (
            FrontierwrightError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self.notify(str(exc), severity="error")

    def _submit_train_ready(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            backend_spec = self._write_reference_backend_spec(values["python"])
            dry_run = execute_training_plan(
                self.root,
                plan_id=values["plan_id"],
                backend_spec_path=backend_spec,
                dry_run=True,
                rerun=False,
            )
            self.notify(f"Dry-run passed for {dry_run.plan_id}.")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")
            return
        self.push_screen(
            ConfirmScreen(
                "EXECUTE TRAINING",
                (
                    "Create a real candidate now? "
                    "The current Champion will remain unchanged until explicit promotion."
                ),
            ),
            lambda confirmed: self._confirmed_train(values, confirmed),
        )

    def _confirmed_train(self, values: dict[str, str], confirmed: bool) -> None:
        if not confirmed:
            self.notify("Training cancelled.")
            return
        try:
            backend_spec = self._write_reference_backend_spec(values["python"])
            run = execute_training_plan(
                self.root,
                plan_id=values["plan_id"],
                backend_spec_path=backend_spec,
                dry_run=False,
                rerun=False,
            )
            self.last_run_id = run.run_id
            self._refresh_all()
            self.notify(
                f"Run {run.run_id}: {run.status}. "
                "Use Actions → Reconcile latest run after the worker finishes."
            )
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _reconcile_latest_run(self) -> None:
        if self.last_run_id is None:
            self.notify("No TUI-started run is available to reconcile.", severity="warning")
            return
        try:
            run = reconcile_training_run(self.root, self.last_run_id)
            self._refresh_all()
            if run.status == "COMPLETED":
                self.notify(f"Run completed. Candidate: {run.candidate_model_id}")
            elif run.status == "RUNNING":
                self.notify("Run is still active.")
            else:
                detail = run.error_message or run.error_code or run.status
                self.notify(f"Run {run.status}: {detail}", severity="warning")
        except FrontierwrightError as exc:
            self.notify(str(exc), severity="error")

    def _submit_candidate_evaluation(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            comparison = compare_candidate_evaluation(
                self.root,
                candidate_model_id=values["candidate_model_id"],
                pack_id="frontierwright.eval.reference-heldout-lm",
                dataset_id=values["dataset_id"],
                python_executable=values["python"],
                device=values["device"] or "auto",
            )
            self._refresh_all()
            self.notify(
                "Raw evaluation complete."
                if comparison.comparable
                else f"Evaluation completed but is not comparable: {comparison.reason}"
            )
            compare = compare_candidate(
                self.root,
                values["candidate_model_id"],
            )
            self.push_screen(
                CandidateScreen(compare),
                lambda result: self._candidate_result(
                    values["candidate_model_id"],
                    result,
                ),
            )
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen(self.language))

    async def action_back(self) -> None:
        return
