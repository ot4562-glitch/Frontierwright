from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from frontierwright.errors import FrontierwrightError
from frontierwright.i18n import tr
from frontierwright.service import (
    BuildView,
    CandidateView,
    CompareView,
    DataView,
    HistoryView,
    PathsView,
    ResourceView,
    StatusView,
    compare_candidate,
    get_build_view,
    get_candidates_view,
    get_data_view,
    get_history_view,
    get_paths_view,
    get_resource_view,
    get_status,
    promote_candidate,
    reject_candidate,
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

    def action_reject(self) -> None:
        self.dismiss("reject")


class FrontierwrightApp(App[None]):
    """Keyboard-only human client over shared Frontierwright service views."""

    CSS = """
    #character-sheet, #build-view, #paths-view, #resources-view,
    #data-view, #history-view, #candidates-view {
        padding: 1 2;
    }
    #help-dialog, #candidate-dialog {
        width: 88%;
        height: auto;
        max-height: 90%;
        border: round $accent;
        padding: 2;
        background: $surface;
        align: center middle;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
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
            elif result == "reject":
                reject_candidate(self.root, model_id)
                self.notify("Candidate rejected.")
            else:
                return
        except FrontierwrightError as exc:
            self.notify(str(exc), severity="error")
            return
        self._refresh_all()

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen(self.language))

    async def action_back(self) -> None:
        return
