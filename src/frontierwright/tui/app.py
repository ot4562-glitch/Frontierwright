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

from frontierwright.benchmark_sources import benchmark_source_payloads
from frontierwright.data import DatasetClassification, DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.editions import EditionProfile, policy_for
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import HardBudgets, PermissionLevel
from frontierwright.fit_planner import FitOpportunityPlan, plan_fit_opportunities
from frontierwright.i18n import tr
from frontierwright.observations import ObservationOutcome, ObservationSource, ObservationSummary
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
    WorkloadAcceptanceView,
    WorkloadFitView,
    WorkloadView,
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
    get_fit_opportunities,
    get_history_view,
    get_paths_view,
    get_resource_view,
    get_status,
    get_tokenizers_view,
    get_usage_observation_summary,
    get_workload_acceptance_view,
    get_workload_fit,
    get_workload_view,
    import_local_model,
    initialize_project,
    profile_reference_inference,
    promote_candidate,
    reconcile_training_run,
    record_usage_observation,
    reject_candidate,
    run_capability_v1,
    set_build_intent,
    set_build_targets,
    set_growth_goals,
    set_workload_profile,
    train_project_tokenizer,
)
from frontierwright.workloads import WorkloadProfile, human_fit_status

BRAND_BANNER = (
    "+-- FRONTIERWRIGHT ----------------------------------+",
    "|  MEASURE -> CHANGE -> PROVE -> KEEP THE CHAMPION  |",
    "+----------------------------------------------------+",
)

TAB_IDS = (
    "character",
    "build",
    "paths",
    "resources",
    "data",
    "workload",
    "history",
    "candidates",
)

EDITION_TAB_ORDERS: dict[EditionProfile, tuple[str, ...]] = {
    EditionProfile.ACADEMY: (
        "character",
        "data",
        "resources",
        "build",
        "paths",
        "workload",
        "candidates",
        "history",
    ),
    EditionProfile.STUDIO: (
        "character",
        "workload",
        "resources",
        "build",
        "paths",
        "data",
        "candidates",
        "history",
    ),
    EditionProfile.LAB: (
        "character",
        "workload",
        "data",
        "resources",
        "paths",
        "build",
        "candidates",
        "history",
    ),
}


def tab_order_for_edition(edition_profile: str | None) -> tuple[str, ...]:
    try:
        profile = EditionProfile(edition_profile or EditionProfile.STUDIO.value)
    except ValueError:
        profile = EditionProfile.STUDIO
    return EDITION_TAB_ORDERS[profile]


def _compact_identity(value: str | None, *, keep: int = 16) -> str:
    if not value:
        return "?"
    if len(value) <= keep:
        return value
    return value[:keep] + "…"


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
    paired = view.paired_capability_evidence
    if paired.get("available") is True:
        lines.extend(["", "PAIRED CAPABILITY ITEMS"])
        axes = paired.get("axes")
        if isinstance(axes, dict):
            for axis in ("general", "reasoning", "math", "coding"):
                evidence = axes.get(axis)
                if not isinstance(evidence, dict):
                    continue
                p_value = evidence.get("p_value_two_sided")
                p_text = (
                    f"{float(p_value):.4g}"
                    if isinstance(p_value, (int, float)) and not isinstance(p_value, bool)
                    else "?"
                )
                detect = (
                    "detectable@0.05"
                    if evidence.get("statistically_detectable_at_0_05") is True
                    else "inconclusive"
                )
                lines.append(
                    f"{axis.title():10} flips +{evidence.get('improvements', '?')} "
                    f"/-{evidence.get('regressions', '?')} · exact p={p_text} · {detect}"
                )
        overall = paired.get("overall")
        if isinstance(overall, dict):
            lines.append(
                f"Overall    flips +{overall.get('improvements', '?')} "
                f"/-{overall.get('regressions', '?')} · paired same-item evidence"
            )
    elif paired:
        reason = paired.get("reason")
        if reason:
            lines.extend(["", "PAIRED CAPABILITY ITEMS", f"Unavailable: {reason}"])

    if view.raw_evaluation_comparisons:
        lines.extend(["", "RAW EVALUATION"])
        for evidence in view.raw_evaluation_comparisons:
            if evidence.get("kind") == "EXTERNAL_LM_EVAL":
                lines.append(
                    f"lm-eval {evidence.get('evaluator_version')} · "
                    f"tasks={evidence.get('task_count')} · external raw evidence"
                )
            else:
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

    if view.workload_comparison.get("configured"):
        champion_fit = view.workload_comparison.get("champion")
        candidate_fit = view.workload_comparison.get("candidate")
        lines.extend(["", "WORKLOAD FIT"])
        if isinstance(champion_fit, dict) and isinstance(candidate_fit, dict):
            lines.append(
                f"Champion {champion_fit.get('overall_status')} -> "
                f"Candidate {candidate_fit.get('overall_status')}"
            )
        regressions = view.workload_comparison.get("regressions")
        if isinstance(regressions, list):
            for item in regressions:
                if isinstance(item, dict):
                    lines.append(
                        f"  ! {item.get('key')}: "
                        f"{item.get('champion_status')} -> {item.get('candidate_status')}"
                    )

    if view.pareto:
        lines.extend(["", f"EVIDENCE PARETO  {view.pareto.get('relation')}"])
        metrics = view.pareto.get("metrics")
        glyphs = {"BETTER": "▲", "WORSE": "▼", "SAME": "•"}
        if isinstance(metrics, list):
            for item in metrics:
                if not isinstance(item, dict):
                    continue
                relation = str(item.get("relation") or "UNKNOWN")
                if relation == "UNKNOWN":
                    continue
                glyph = glyphs.get(relation, "?")
                key = str(item.get("key") or "metric")
                lines.append(
                    f"  {glyph} {key}: {item.get('champion_value')} -> "
                    f"{item.get('candidate_value')}"
                )
        reason = view.pareto.get("inference_profile_reason")
        if reason:
            lines.append(f"Runtime: {reason}")
        utility = view.pareto.get("explicit_user_utility")
        if isinstance(utility, dict):
            status = str(utility.get("status") or "NOT_CONFIGURED")
            if status == "COMPLETE":
                delta = utility.get("utility_delta")
                delta_text = (
                    f"{delta:+.3f}"
                    if isinstance(delta, (int, float)) and not isinstance(delta, bool)
                    else "?"
                )
                lines.extend(["", f"USER UTILITY  {utility.get('relation')} · Δ {delta_text}"])
                contributions = utility.get("contributions")
                if isinstance(contributions, list):
                    for item in contributions:
                        if not isinstance(item, dict):
                            continue
                        contribution = item.get("contribution")
                        contribution_text = (
                            f"{contribution:+.3f}"
                            if isinstance(contribution, (int, float))
                            and not isinstance(contribution, bool)
                            else "?"
                        )
                        lines.append(
                            f"  {item.get('metric_key')}: {contribution_text} "
                            f"(w={item.get('weight')}, scale={item.get('scale')})"
                        )
            elif status == "INCOMPLETE":
                missing = utility.get("missing_metrics")
                lines.extend(["", "USER UTILITY  INCOMPLETE"])
                if isinstance(missing, list) and missing:
                    lines.append("Missing comparable evidence: " + ", ".join(map(str, missing)))
        utility = view.pareto.get("explicit_user_utility")
        if isinstance(utility, dict):
            status = str(utility.get("status") or "NOT_CONFIGURED")
            if status == "COMPLETE":
                delta = utility.get("utility_delta")
                delta_text = (
                    f"{delta:+.3f}"
                    if isinstance(delta, (int, float)) and not isinstance(delta, bool)
                    else "?"
                )
                lines.extend(
                    [
                        "",
                        f"USER UTILITY  {utility.get('relation')} · Δ {delta_text}",
                    ]
                )
                contributions = utility.get("contributions")
                if isinstance(contributions, list):
                    for item in contributions:
                        if not isinstance(item, dict):
                            continue
                        contribution = item.get("contribution")
                        contribution_text = (
                            f"{contribution:+.3f}"
                            if isinstance(contribution, (int, float))
                            and not isinstance(contribution, bool)
                            else "?"
                        )
                        lines.append(
                            f"  {item.get('metric_key')}: {contribution_text} "
                            f"(w={item.get('weight')}, scale={item.get('scale')})"
                        )
            elif status == "INCOMPLETE":
                missing = utility.get("missing_metrics")
                lines.extend(["", "USER UTILITY  INCOMPLETE"])
                if isinstance(missing, list) and missing:
                    lines.append("Missing comparable evidence: " + ", ".join(map(str, missing)))
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
    lines.extend(["", "P Promote   R Reject   Esc Back"])
    return "\n".join(lines)


def _ui(language: str, en: str, ko: str) -> str:
    return ko if language == "ko" else en


def _edition_help_text(language: str, edition_profile: str | None) -> str:
    try:
        profile = EditionProfile(edition_profile or EditionProfile.STUDIO.value)
    except ValueError:
        profile = EditionProfile.STUDIO
    keys = tr(language, "help")
    separator = chr(10) * 2

    if profile is EditionProfile.ACADEMY:
        if language == "ko":
            start_here = chr(10).join(
                [
                    "시작 순서",
                    "1. 학습 데이터를 등록",
                    "2. 토크나이저를 학습하고 첫 모델 생성",
                    "3. 학습·측정·비교 후 Candidate를 Champion으로 승격할지 결정",
                ]
            )
            return separator.join(
                [
                    "ACADEMY — 직접 만들면서 이해하기",
                    (
                        "실제 흐름을 따라갑니다: 데이터 → 토크나이저 → 모델 탄생 → 학습 → "
                        "평가 → Candidate → Champion. 측정하지 않은 근거는 모르는 상태로 "
                        "남고, 학습이 성공해도 Candidate 성능은 나빠질 수 있습니다."
                    ),
                    (
                        "Actions에서 다음 실제 작업을 실행하세요. 매 단계마다 모델에서 무엇이 "
                        "바뀌었는지, 어떤 근거를 측정했는지, 아직 무엇을 모르는지 확인합니다."
                    ),
                    start_here,
                    keys,
                ]
            )
        start_here = chr(10).join(
            [
                "START HERE",
                "1. Add learning data",
                "2. Train a tokenizer and create the model",
                "3. Train, measure, compare, then decide whether a Candidate becomes Champion",
            ]
        )
        return separator.join(
            [
                "ACADEMY — UNDERSTAND BY DOING",
                (
                    "Follow the real lifecycle: data → tokenizer → birth → training → "
                    "evaluation → candidate → champion. Unknown evidence stays unknown until "
                    "measured. Successful training can still produce a worse candidate."
                ),
                (
                    "Use Actions for the next real operation. Ask after each step what changed, "
                    "what was measured, and what is still unknown."
                ),
                start_here,
                keys,
            ]
        )

    if profile is EditionProfile.LAB:
        if language == "ko":
            start_here = chr(10).join(
                [
                    "시작 순서",
                    "1. 통제된 PRIVATE 인프라 연결",
                    "2. 모델·데이터·workload acceptance·평가기/보상·예산 고정",
                    "3. bounded experiment를 실행하고 비교 가능한 근거를 통해서만 승격",
                ]
            )
            return separator.join(
                [
                    "LAB — 통제된 프런티어 개발",
                    (
                        "모델·데이터셋·평가기·보상원·백엔드·자원 프로필·Candidate를 모두 "
                        "버전이 고정된 근거로 취급합니다. 비교 가능한 평가, 불확실성, 하드 예산, "
                        "재현 가능한 intervention을 우선하고 PRIVATE 경계는 하드 제약입니다."
                    ),
                    start_here,
                    keys,
                ]
            )
        start_here = chr(10).join(
            [
                "START HERE",
                "1. Connect controlled private infrastructure",
                "2. Pin model, data, workload acceptance, evaluator/reward, and budgets",
                "3. Run bounded experiments and promote only through comparable evidence",
            ]
        )
        return separator.join(
            [
                "LAB — CONTROLLED FRONTIER DEVELOPMENT",
                (
                    "Treat every model, dataset, evaluator, reward source, backend, resource "
                    "profile, and candidate as versioned evidence. Prefer comparable evals, "
                    "explicit uncertainty, hard budgets, and reproducible interventions. "
                    "Private boundaries are hard constraints."
                ),
                start_here,
                keys,
            ]
        )

    if language == "ko":
        start_here = chr(10).join(
            [
                "시작 순서",
                "1. 내가 가진 모델을 가져오거나 이어서 사용",
                "2. workload와 내 PC에서의 실제 동작을 측정",
                "3. 튜닝/최적화 → Candidate 비교 → Champion 승격 또는 거절",
            ]
        )
        return separator.join(
            [
                "STUDIO — 내 모델을 내 환경에 맞추기",
                (
                    "가지고 있는 로컬 모델에서 시작해 내 PC와 실제 사용 환경에서 측정하고, "
                    "원하는 성장 목표를 정한 뒤 Candidate를 반복해서 만듭니다. 시작 모델의 "
                    "성장 Stat은 0이고 이후 변화량이 올라가거나 내려갑니다."
                ),
                start_here,
                keys,
            ]
        )

    start_here = chr(10).join(
        [
            "START HERE",
            "1. Import or continue a model you control",
            "2. Describe your workload and measure model/machine fit",
            "3. Tune or optimize a Candidate, compare trade-offs, then promote or reject",
        ]
    )
    return separator.join(
        [
            "STUDIO — FIT THE MODEL TO YOU",
            (
                "Start from a local model you control, measure it on your machine, set growth "
                "goals, and iterate through Candidates. The origin model is growth Stat 0; "
                "later models move up or down relative to that permanent baseline."
            ),
            start_here,
            keys,
        ]
    )


def _growth_stat_line(
    *,
    language: str,
    axis: str,
    evidence: dict[str, object] | None,
    edition_profile: str | None,
) -> str:
    labels = {
        "knowledge": ("Knowledge", "지식"),
        "reasoning": ("Reasoning", "추론"),
        "math": ("Math", "수학"),
        "coding": ("Coding", "코딩"),
        "instruction": ("Instruction", "지시수행"),
        "language": ("Language", "언어"),
        "context": ("Context", "컨텍스트"),
    }
    en_label, ko_label = labels.get(axis, (axis.title(), axis))
    label = ko_label if language == "ko" else en_label
    label = f"{label:12}"
    if not isinstance(evidence, dict):
        return f"{label} ?  " + _ui(language, "UNMEASURED", "미측정")

    delta = evidence.get("display_delta")
    relation = str(evidence.get("relation") or "UNMEASURED")
    source = str(evidence.get("source") or "")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        return f"{label} ?  " + _ui(language, "UNMEASURED", "미측정")

    delta_text = f"{float(delta):+.1f}"
    if source == "ORIGIN_BASELINE":
        return f"{label} {delta_text}  " + _ui(
            language,
            "BASELINE (reference, not absolute capability)",
            "기준선 (절대 성능 점수가 아님)",
        )

    interval = evidence.get("delta_interval")
    interval_text = ""
    if (
        isinstance(interval, list)
        and len(interval) == 2
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in interval
        )
    ):
        interval_text = f"  Δ95[{float(interval[0]):+.1f}, {float(interval[1]):+.1f}]"

    try:
        profile = EditionProfile(edition_profile or EditionProfile.STUDIO.value)
    except ValueError:
        profile = EditionProfile.STUDIO
    if profile is EditionProfile.LAB:
        source_count = evidence.get("source_count")
        n_text = (
            f" n={source_count}"
            if isinstance(source_count, int) and not isinstance(source_count, bool)
            else ""
        )
        source_text = f" source={source}" if source else ""
        return f"{label} {delta_text}  {relation}{interval_text}{n_text}{source_text}"
    return f"{label} {delta_text}  {relation}{interval_text}"


def _stat_line(
    *,
    axis: str,
    value: float | None,
    uncertainty: dict[str, object] | None,
    edition_profile: str | None,
) -> str:
    label = f"{axis.title():10}"
    if value is None:
        return f"{label} ?"
    if not isinstance(uncertainty, dict):
        return f"{label} {value:g}"
    lower = uncertainty.get("stat_lower")
    upper = uncertainty.get("stat_upper")
    sample_size = uncertainty.get("sample_size")
    if not isinstance(lower, (int, float)) or isinstance(lower, bool):
        return f"{label} {value:g}"
    if not isinstance(upper, (int, float)) or isinstance(upper, bool):
        return f"{label} {value:g}"
    try:
        profile = EditionProfile(edition_profile or EditionProfile.STUDIO.value)
    except ValueError:
        profile = EditionProfile.STUDIO
    if profile is EditionProfile.LAB:
        n_text = (
            f" n={sample_size}"
            if isinstance(sample_size, int) and not isinstance(sample_size, bool)
            else ""
        )
        return f"{label} {value:g}  CI95[{float(lower):.1f}, {float(upper):.1f}]{n_text} Wilson"
    if profile is EditionProfile.ACADEMY:
        return f"{label} {value:g}  (95% evidence range {float(lower):.0f}–{float(upper):.0f})"
    return f"{label} {value:g}  [{float(lower):.0f}–{float(upper):.0f}]"


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
        Binding("enter", "dismiss", "Back"),
        Binding("?", "dismiss", "Back"),
    ]

    def __init__(self, language: str, edition_profile: str | None) -> None:
        super().__init__()
        self.language = language
        self.edition_profile = edition_profile

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static(
                _edition_help_text(self.language, self.edition_profile),
                id="help-text",
            )


class InfoScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
        Binding("enter", "dismiss", "Back"),
    ]

    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self.title_text = title
        self.body_text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static(
                self.title_text + chr(10) * 2 + self.body_text,
                id="help-text",
            )


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

    def __init__(
        self,
        items: list[ActionItem],
        *,
        title: str = "ACTION CENTER",
        intro: str = "Every action below uses the same Frontierwright core as CLI/JSON.",
        language: str = "en",
    ) -> None:
        super().__init__()
        self.items = items
        self.index = 0
        self.title_text = title
        self.intro = intro
        self.language = language

    def compose(self) -> ComposeResult:
        with Vertical(id="action-dialog"):
            yield Static(self._text(), id="action-list")

    def _text(self) -> str:
        lines = [
            self.title_text,
            "",
            self.intro,
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

    TITLE = "FRONTIERWRIGHT"

    CSS = """
    #character-sheet, #build-view, #paths-view, #resources-view,
    #data-view, #workload-view, #history-view, #candidates-view {
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
            Binding(
                str(index + 1),
                f"select_tab_index({index})",
                f"{index + 1}:tab",
                show=False,
            )
            for index in range(8)
        ],
    ]

    def __init__(
        self,
        *,
        view: StatusView,
        resources: ResourceView | None = None,
        build: BuildView | None = None,
        data: DataView | None = None,
        workload: WorkloadView | None = None,
        workload_acceptance: WorkloadAcceptanceView | None = None,
        workload_fit: WorkloadFitView | None = None,
        usage_observations: ObservationSummary | None = None,
        fit_opportunities: FitOpportunityPlan | None = None,
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
        self.workload = workload or WorkloadView()
        self.workload_acceptance = workload_acceptance or WorkloadAcceptanceView()
        self.workload_fit = workload_fit or WorkloadFitView()
        self.usage_observations = usage_observations or ObservationSummary(
            model_id=view.champion_model_id
        )
        self.fit_opportunities = fit_opportunities or plan_fit_opportunities(
            edition=EditionProfile.STUDIO,
            model_id=None,
            workload_configured=False,
            constraints=[],
            model_fit=None,
            workload_evaluation_coverage=None,
        )
        self.paths = paths or PathsView()
        self.candidates = candidates or CandidateView()
        self.history = history or HistoryView()
        self.candidate_index = 0
        self.last_run_id: str | None = None
        self.language = language
        self.tab_order = tab_order_for_edition(view.edition_profile)
        try:
            edition = EditionProfile(view.edition_profile or EditionProfile.STUDIO.value)
        except ValueError:
            edition = EditionProfile.STUDIO
        policy = policy_for(edition)
        self.title = "FRONTIERWRIGHT"
        self.sub_title = (
            f"{policy.display_name.removeprefix('Frontierwright ')}"
            + (f"  //  {view.nickname}" if view.nickname else "")
        )

    def compose(self) -> ComposeResult:
        renderers = {
            "character": ("character-sheet", self._character_text),
            "build": ("build-view", self._build_text),
            "paths": ("paths-view", self._paths_text),
            "resources": ("resources-view", self._resources_text),
            "data": ("data-view", self._data_text),
            "workload": ("workload-view", self._workload_text),
            "history": ("history-view", self._history_text),
            "candidates": ("candidates-view", self._candidates_text),
        }
        yield Header(show_clock=False)
        with TabbedContent(initial="character", id="main-tabs"):
            for tab_id in self.tab_order:
                static_id, renderer = renderers[tab_id]
                with TabPane(tr(self.language, tab_id), id=tab_id):
                    yield Static(renderer(), id=static_id)
        yield Footer()

    def _character_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")
        try:
            edition = EditionProfile(self.view.edition_profile or EditionProfile.STUDIO.value)
        except ValueError:
            edition = EditionProfile.STUDIO

        lines = [
            *BRAND_BANNER,
            "",
            self.view.nickname or "?",
            f"Edition: {self.view.edition_name or self.view.edition_profile or '?'}",
        ]
        if self.view.edition_tagline:
            lines.append(self.view.edition_tagline)
        pending_count = sum(
            1 for item in self.candidates.candidates if item.get("status") == "PENDING"
        )

        if edition is EditionProfile.ACADEMY:
            history_text = {
                "COMPLETE": "Project history: fully recorded",
                "VERIFIED": "Project history: verified",
                "PARTIAL": "Project history: partially known",
                "UNKNOWN": "Project history: prior training history not verified",
            }.get(
                self.view.history_confidence or "",
                "Project history: not yet established",
            )
            capability_text = (
                "Capability: measured"
                if self.view.measurement_state == "MEASURED"
                else "Capability: not measured yet"
            )
            model_text = (
                "Model: not created yet"
                if self.view.champion_model_id is None
                else "Model: root/Champion exists"
            )
            lines.extend(
                [
                    "",
                    "LEARNING STATE",
                    f"Origin: {self.view.origin}",
                    history_text,
                    model_text,
                    capability_text,
                ]
            )
            if self.view.champion_model_id is None:
                lines.extend(
                    [
                        "",
                        "NEXT CONCEPT",
                        "Tokenizer → model birth → pretraining are different real steps.",
                        "Use Actions to prepare data and create the first root model.",
                    ]
                )
            elif self.view.measurement_state != "MEASURED":
                lines.extend(
                    [
                        "",
                        "NEXT CONCEPT",
                        "A born/trained model has no capability stat until it is evaluated.",
                        "Measure Capability v1, then inspect what the evidence can and cannot say.",
                    ]
                )
            elif pending_count > 0:
                lines.extend(
                    [
                        "",
                        "NEXT CONCEPT",
                        "Training created a Candidate, not an automatic improvement.",
                        "Measure and compare it before deciding whether it becomes Champion.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "NEXT CONCEPT",
                        (
                            "Choose one measurable goal, create a Candidate, then inspect "
                            "gains and regressions."
                        ),
                    ]
                )
            if self.view.champion_model_id:
                lines.append("")
                lines.append(
                    "Current model: " + _compact_identity(self.view.champion_model_id, keep=18)
                )
        elif edition is EditionProfile.LAB:
            lines.extend(
                [
                    "",
                    "CONTROLLED MODEL / EVIDENCE / INFRASTRUCTURE",
                    f"Origin: {self.view.origin}",
                    f"History confidence: {self.view.history_confidence}",
                    f"Measurement state: {self.view.measurement_state}",
                ]
            )
            if self.view.champion_model_id:
                lines.extend(
                    [
                        f"Champion: {self.view.champion_model_id}",
                        f"Format: {self.view.model_format}",
                        f"Trainable: {'YES' if self.view.trainable else 'NO'}",
                        f"Fingerprint: {self.view.model_fingerprint}",
                    ]
                )
            lines.extend(
                [
                    "",
                    "EVIDENCE STATUS",
                    f"Workload fit: {human_fit_status(self.workload_fit.overall_status)}",
                    (
                        "Inference profile: MEASURED"
                        if self.resources.model_fit
                        else "Inference profile: MEASUREMENT NEEDED"
                    ),
                    f"Pending candidates: {pending_count}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "YOUR MODEL / YOUR MACHINE / YOUR WORKLOAD",
                    f"Origin: {self.view.origin}",
                    f"Capability: {self.view.measurement_state}",
                ]
            )
            if self.view.champion_model_id:
                lines.append("Champion: " + _compact_identity(self.view.champion_model_id, keep=20))
            lines.append(
                "Workload fit: "
                + (
                    human_fit_status(self.workload_fit.overall_status)
                    if self.workload.configured
                    else "NOT DEFINED"
                )
            )
            model_fit = self.resources.model_fit
            if model_fit:
                latency = model_fit.get("latency_seconds_p50")
                throughput = model_fit.get("tokens_per_second_p50")
                profile_parts: list[str] = []
                if isinstance(latency, (int, float)) and not isinstance(latency, bool):
                    profile_parts.append(f"p50 {float(latency):.3f}s")
                if isinstance(throughput, (int, float)) and not isinstance(throughput, bool):
                    profile_parts.append(f"{float(throughput):.1f} tok/s")
                free_vram = model_fit.get("cuda_memory_free_min_sampled_bytes")
                if isinstance(free_vram, int):
                    profile_parts.append(f"{_human_bytes(free_vram)} VRAM left")
                lines.append(
                    "Measured machine fit: "
                    + (" · ".join(profile_parts) if profile_parts else "PARTIAL EVIDENCE")
                )
            else:
                lines.append("Measured machine fit: MEASUREMENT NEEDED — profile the Champion")
            if self.workload_fit.configured and self.workload_fit.constraints:
                gap = next(
                    (
                        item
                        for item in self.workload_fit.constraints
                        if item.get("status") in {"FAIL", "UNKNOWN"}
                    ),
                    None,
                )
                if isinstance(gap, dict):
                    lines.append(
                        f"Next fit gap: {gap.get('key')} · "
                        f"{human_fit_status(str(gap.get('status') or 'UNKNOWN'))}"
                    )

        lines.extend(
            [
                "",
                _ui(
                    self.language,
                    "GROWTH STAT · ORIGIN MODEL = 0",
                    "성장 STAT · 시작 모델 = 0",
                ),
            ]
        )
        for axis in (
            "knowledge",
            "reasoning",
            "math",
            "coding",
            "instruction",
            "language",
            "context",
        ):
            lines.append(
                _growth_stat_line(
                    language=self.language,
                    axis=axis,
                    evidence=self.view.stat_v2.get(axis),
                    edition_profile=self.view.edition_profile,
                )
            )
        if edition is EditionProfile.LAB and self.view.measurement_state == "MEASURED":
            lines.extend(
                [
                    "",
                    "LEGACY CAPABILITY v1 BENCHMARK EVIDENCE",
                    "Raw absolute smoke-benchmark evidence; not the growth Stat scale.",
                ]
            )
            for axis in ("general", "reasoning", "math", "coding"):
                value = self.view.stats.get(axis)
                lines.append(
                    _stat_line(
                        axis=axis,
                        value=value,
                        uncertainty=self.view.stat_uncertainty.get(axis),
                        edition_profile=self.view.edition_profile,
                    )
                )
        return "\n".join(lines)

    def _build_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")

        lines = [f"Mode: {self.build.mode or 'NOT_READY'}"]
        if self.build.archetype:
            lines.append(f"Archetype: {self.build.archetype}")
        if self.build.priorities:
            lines.extend(["", "PRIORITIES"])
            for axis, priority_value in sorted(self.build.priorities.items()):
                lines.append(f"{axis.title():10} {priority_value}")
        if self.build.targets:
            lines.extend(["", "TARGETS"])
            for axis, target_value in sorted(self.build.targets.items()):
                lines.append(f"{axis.title():10} {target_value}")
        if self.build.floors:
            lines.extend(["", "FLOORS"])
            for axis, floor_value in sorted(self.build.floors.items()):
                lines.append(f"{axis.title():10} {floor_value}")
        if self.build.growth_goals:
            lines.extend(["", _ui(self.language, "GROWTH RULES", "성장 규칙")])
            for goal in self.build.growth_goals:
                kind = goal.get("kind")
                metric = goal.get("metric")
                goal_value = goal.get("value")
                unit = goal.get("unit")
                suffix = ""
                if goal_value is not None:
                    suffix = f"={goal_value}{'%' if unit == 'PERCENT' else ''}"
                lines.append(f"{kind} {metric}{(' ' + suffix) if suffix else ''}")
        if self.build.reason:
            lines.extend(["", self.build.reason])
        return "\n".join(lines)

    def _paths_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")

        lines: list[str] = []
        for item in self.paths.paths:
            lines.append(
                f"[{item.get('availability')}] {item.get('title')} ({item.get('path_id')})"
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
        headroom = self.resources.headroom
        try:
            profile = EditionProfile(self.view.edition_profile or EditionProfile.STUDIO.value)
        except ValueError:
            profile = EditionProfile.STUDIO

        if profile is EditionProfile.ACADEMY:
            lines = [
                "AVAILABLE ON THIS MACHINE NOW",
                "These are resources currently free. Model-specific fit needs a real profile.",
            ]
        elif profile is EditionProfile.LAB:
            lines = [
                "SYSTEM HEADROOM — POINT-IN-TIME EVIDENCE",
                f"Semantics: {headroom.get('semantics', 'UNKNOWN')} · model-specific=NO",
            ]
        else:
            lines = [
                "SYSTEM HEADROOM NOW",
                "Use a real model profile before treating this as post-load headroom.",
            ]
        if self.resources.detected_at:
            lines.append(f"Measured: {self.resources.detected_at}")
        lines.append("")

        lines.extend(
            [
                f"Provenance: {self.resources.provenance}",
                f"CPU: {snapshot.get('cpu_model', 'UNKNOWN')}",
                f"Logical CPUs: {snapshot.get('cpu_logical_count', 'UNKNOWN')}",
                (
                    "RAM: "
                    f"{_human_bytes(headroom.get('ram_available_bytes'))} available / "
                    f"{_human_bytes(headroom.get('ram_total_bytes'))} total"
                ),
                (
                    "Disk: "
                    f"{_human_bytes(headroom.get('disk_available_bytes'))} available / "
                    f"{_human_bytes(headroom.get('disk_total_bytes'))} total"
                ),
            ]
        )
        gpus = headroom.get("gpus")
        if isinstance(gpus, list) and gpus:
            for index, gpu in enumerate(gpus, start=1):
                if not isinstance(gpu, dict):
                    continue
                fraction = gpu.get("available_fraction")
                fraction_text = (
                    f" · {float(fraction) * 100:.0f}% free"
                    if isinstance(fraction, (int, float)) and not isinstance(fraction, bool)
                    else ""
                )
                lines.append(
                    f"GPU {index}: {gpu.get('vendor', '?')} {gpu.get('name', '?')} · "
                    f"{_human_bytes(gpu.get('memory_available_bytes'))} available / "
                    f"{_human_bytes(gpu.get('memory_total_bytes'))} VRAM{fraction_text}"
                )
        else:
            diagnostics = snapshot.get("diagnostics")
            gpu_diagnostics = (
                [
                    item
                    for item in diagnostics
                    if isinstance(item, dict)
                    and str(item.get("probe_id") or "").startswith("gpu.")
                    and item.get("status") != "MEASURED"
                ]
                if isinstance(diagnostics, list)
                else []
            )
            if gpu_diagnostics:
                status = (
                    "MEASUREMENT NEEDED"
                    if any(item.get("status") == "MEASUREMENT_NEEDED" for item in gpu_diagnostics)
                    else "UNSUPPORTED HERE"
                )
                lines.append(f"GPU availability: {status}")
                lines.append(
                    "No measured GPU record exists; this does not prove that no GPU is installed."
                )
                for item in gpu_diagnostics:
                    vendor = str(item.get("probe_id") or "gpu").split(".")[-1].upper()
                    lines.append(f"  {vendor}: {item.get('reason_code')} — {item.get('detail')}")
                    next_action = item.get("next_action")
                    if next_action:
                        lines.append(f"  Next: {next_action}")
            else:
                lines.append("GPU availability: MEASUREMENT NEEDED")
                lines.append(
                    "No measured GPU record exists; run resource diagnostics before concluding "
                    "that hardware is absent."
                )

        model_fit = self.resources.model_fit
        if model_fit:
            lines.append("")
            if profile is EditionProfile.ACADEMY:
                lines.append("WHEN THE CURRENT CHAMPION WAS ACTUALLY PROFILED")
                lines.append(
                    "These numbers came from running the model, not from its parameter count."
                )
            elif profile is EditionProfile.LAB:
                lines.append("CHAMPION MODEL-FIT RECEIPT")
                lines.append(
                    f"Scope: {model_fit.get('measurement_scope') or 'UNKNOWN'} · "
                    f"runs={model_fit.get('measured_runs') or 'UNKNOWN'}"
                )
            else:
                lines.append("CURRENT CHAMPION — MEASURED FIT")

            free_min = model_fit.get("cuda_memory_free_min_sampled_bytes")
            total_vram = model_fit.get("cuda_memory_total_bytes")
            if isinstance(free_min, int) and isinstance(total_vram, int):
                lines.append(
                    "VRAM left while profiled: "
                    f"{_human_bytes(free_min)} / {_human_bytes(total_vram)}"
                )
            peak_vram = model_fit.get("peak_vram_bytes")
            if isinstance(peak_vram, int):
                lines.append(f"PyTorch peak allocation: {_human_bytes(peak_vram)}")
            rss = model_fit.get("max_sampled_process_rss_bytes")
            if isinstance(rss, int):
                lines.append(f"Process RSS: {_human_bytes(rss)}")
            latency = model_fit.get("latency_seconds_p50")
            throughput = model_fit.get("tokens_per_second_p50")
            if isinstance(latency, (int, float)) and not isinstance(latency, bool):
                lines.append(f"Latency p50: {float(latency):.3f}s")
            if isinstance(throughput, (int, float)) and not isinstance(throughput, bool):
                lines.append(f"Throughput p50: {float(throughput):.1f} tok/s")
            if model_fit.get("workload_shape_bound") is False:
                lines.append(
                    "Reference profile shape: fixed prompt/output; not workload-shaped proof."
                )

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
                f"{item.get('name')} · {item.get('role')} · {_human_bytes(item.get('total_bytes'))}"
            )
            lines.append(f"  Provenance: {item.get('provenance')}")
            lines.append(f"  Classification: {item.get('classification') or 'UNKNOWN'}")
            lines.append(f"  Fingerprint: {item.get('fingerprint')}")
            lines.append(f"  License: {item.get('license') or 'UNKNOWN'}")
            token_count = item.get("token_count")
            if isinstance(token_count, int):
                lines.append(f"  Tokens: {token_count}")
            elif self.view.edition_profile == EditionProfile.ACADEMY.value:
                lines.append(
                    "  Tokens: not measured yet — token count depends on the tokenizer "
                    "bound to this model."
                )
            else:
                lines.append("  Tokens: MEASUREMENT NEEDED (tokenizer-dependent)")
            if item.get("classification") == "PRIVATE":
                lines.append(
                    "  PRIVATE: later operations must use a compatible data boundary; "
                    "registration itself does not upload the source."
                )
            if item.get("managed"):
                lines.append(f"  Managed from: {item.get('source_dataset_id')}")
                lines.append(f"  Recipe: {item.get('preparation_recipe_id')}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _real_use_lines(self, edition: EditionProfile) -> list[str]:
        observed = self.usage_observations
        lines = ["", "REAL USE"]
        if observed.total == 0:
            if edition is EditionProfile.ACADEMY:
                lines.append(
                    "No real-use outcomes recorded yet. This is optional evidence, not homework."
                )
            elif edition is EditionProfile.LAB:
                lines.append("No operational outcome evidence recorded for this exact model.")
            else:
                lines.append(
                    "No real-use outcomes yet. Record what works, fails, or needs correction."
                )
            return lines

        lines.append(
            f"{observed.total} uses · success {observed.direct_successes} · "
            f"failure {observed.failures} · corrected {observed.corrected} · "
            f"abstained {observed.abstained}"
        )
        if observed.direct_success_rate is not None and observed.direct_success_rate_ci95:
            lo, hi = observed.direct_success_rate_ci95
            lines.append(
                f"Direct success {observed.direct_success_rate:.1%} · "
                f"Wilson 95% CI {lo:.1%}–{hi:.1%}"
            )
        if observed.by_task:
            top = observed.by_task[0]
            if edition is EditionProfile.ACADEMY:
                lines.append(
                    f"Most useful lesson source: {top.task} · "
                    f"{top.failures + top.corrected} failures/corrections"
                )
                lines.append(
                    "These are observations, not RL rewards. Turn repeats into a real "
                    "eval/data set."
                )
            elif edition is EditionProfile.LAB:
                lines.append(
                    f"Highest observed issue task: {top.task} · n={top.total} · "
                    f"failure={top.failures} · corrected={top.corrected}"
                )
                lines.append(
                    "Operational observations require promotion into versioned "
                    "eval/data/reward evidence before training."
                )
            else:
                lines.append(
                    f"Most observed friction: {top.task} · "
                    f"{top.failures + top.corrected} failures/corrections"
                )
                lines.append(
                    "Use repeated failures to choose the next bounded experiment; "
                    "no gain is assumed."
                )
        return lines

    def _workload_text(self) -> str:
        if not self.view.initialized:
            return tr(self.language, "no_project")
        try:
            edition = EditionProfile(self.view.edition_profile or EditionProfile.STUDIO.value)
        except ValueError:
            edition = EditionProfile.STUDIO

        if not self.workload.configured:
            if edition is EditionProfile.ACADEMY:
                lines = [
                    "OPTIONAL LEARNING CONTEXT",
                    (
                        "Describe what you want the model to do. This is not a personality "
                        "quiz; it is evidence about real tasks."
                    ),
                    "",
                    (
                        "You can learn the birth/training/evaluation loop without it, then "
                        "add a workload when you want to study model trade-offs."
                    ),
                ]
            elif edition is EditionProfile.LAB:
                lines = [
                    "WORKLOAD / SERVING CONTRACT — NOT DEFINED",
                    (
                        "Define task mixture, serving constraints, privacy, and capability "
                        "floors before claiming model utility or frontier progress."
                    ),
                ]
            else:
                lines = [
                    "YOUR WORKLOAD — NOT DEFINED",
                    (
                        "Frontierwright can measure the model, but it cannot yet judge whether "
                        "the model fits your real work."
                    ),
                    (
                        "Define languages, domains, task weights, context, latency/throughput "
                        "needs, privacy, and hard capability floors."
                    ),
                ]
            lines.extend(self._real_use_lines(edition))
            return chr(10).join(lines)

        profile = self.workload.profile
        languages = profile.get("languages")
        domains = profile.get("domains")
        tasks = profile.get("task_weights")
        floors = profile.get("critical_floors")
        utility_weights = profile.get("utility_weights")
        utility_scales = profile.get("utility_scales")
        if edition is EditionProfile.ACADEMY:
            lines = [
                "WHAT SHOULD THIS MODEL LEARN TO FIT?",
                "These are real requirements. They do not add fake capability points.",
                "",
                f"Workload: {self.workload.profile_name or 'Unnamed'}",
            ]
        elif edition is EditionProfile.LAB:
            lines = [
                "WORKLOAD / SERVING REQUIREMENTS",
                f"Profile: {self.workload.profile_id}",
                f"Hash: {self.workload.profile_hash}",
                f"Source: {self.workload.source}",
                "",
            ]
        else:
            lines = [
                "YOUR WORKLOAD",
                "The Champion should be optimized for these real tasks and limits.",
                "",
                f"Profile: {self.workload.profile_name or 'Unnamed'}",
            ]

        if isinstance(languages, list) and languages:
            lines.append("Languages: " + ", ".join(str(item) for item in languages))
        if isinstance(domains, list) and domains:
            lines.append("Domains: " + ", ".join(str(item) for item in domains))
        if isinstance(tasks, dict) and tasks:
            rendered_tasks = []
            for name, weight in sorted(tasks.items()):
                if isinstance(weight, (int, float)) and not isinstance(weight, bool):
                    rendered_tasks.append(f"{name}×{weight:g}")
                else:
                    rendered_tasks.append(f"{name}×{weight}")
            lines.append("Task mix: " + ", ".join(rendered_tasks))
        p50 = profile.get("context_tokens_p50")
        p95 = profile.get("context_tokens_p95")
        if p50 is not None or p95 is not None:
            lines.append(f"Context: p50={p50 or '?'} · p95={p95 or '?'} tokens")
        latency = profile.get("max_latency_seconds")
        throughput = profile.get("min_tokens_per_second")
        if latency is not None:
            lines.append(f"Latency ceiling: {latency}s")
        if throughput is not None:
            lines.append(f"Throughput floor: {throughput} tok/s")
        privacy = profile.get("privacy")
        lines.append(f"Privacy: {privacy or 'UNKNOWN'}")
        if privacy == "PRIVATE":
            lines.append(
                "Privacy scope: Frontierwright application admission/data-boundary policy only."
            )
            lines.append(
                "It does not attest OS/process/kernel/network isolation or remote attestation."
            )
        if isinstance(floors, dict) and floors:
            lines.append("Capability floors:")
            for axis, value in sorted(floors.items()):
                lines.append(f"  {str(axis).title():10} >= {value}")
            lines.append(
                "  Point-estimate thresholds; use acceptance EXPLICIT_INTERVAL for CI gating."
            )
        if (
            edition is not EditionProfile.ACADEMY
            and isinstance(utility_weights, dict)
            and utility_weights
            and isinstance(utility_scales, dict)
        ):
            lines.append("Explicit decision utility:")
            for key, weight in sorted(utility_weights.items()):
                lines.append(f"  {key}: weight={weight} · scale={utility_scales.get(key)}")
            lines.append("  Missing comparable evidence keeps utility INCOMPLETE.")

        acceptance = self.workload_acceptance
        if edition is EditionProfile.ACADEMY:
            lines.extend(["", "WHAT COUNTS AS SUCCESS?"])
            if not acceptance.configured:
                lines.append(
                    "No measurable success criteria yet. Add thresholds so evaluation can answer "
                    "whether the model actually meets your goal."
                )
            else:
                lines.append("Success check: " + human_fit_status(acceptance.overall_status))
                contract_label = acceptance.contract_name or acceptance.contract_id
                lines.append(f"Criteria: {len(acceptance.criteria)} · contract {contract_label}")
                if acceptance.overall_status in {"UNKNOWN", "INCONCLUSIVE", "NOT_ASSESSED"}:
                    lines.append(
                        "Next lesson: collect the exact evidence needed to resolve the criteria."
                    )
        elif edition is EditionProfile.LAB:
            lines.extend(["", "ACCEPTANCE CONTRACT"])
            if not acceptance.configured:
                lines.append("NOT CONFIGURED — workload success claims remain unavailable.")
            else:
                active_label = "YES" if acceptance.active else "NO"
                lines.append(
                    f"Status: {human_fit_status(acceptance.overall_status)} · "
                    f"contract={acceptance.contract_id} · active={active_label}"
                )
                lines.append(f"Contract hash: {acceptance.contract_hash}")
                lines.append(f"Assessment: {acceptance.assessment_id or 'MEASUREMENT NEEDED'}")
        else:
            lines.extend(["", "SUCCESS CRITERIA"])
            if not acceptance.configured:
                lines.append(
                    "MEASUREMENT NEEDED — define the thresholds that make this model good enough "
                    "for your real work."
                )
            else:
                lines.append(
                    f"{human_fit_status(acceptance.overall_status)} · "
                    f"{len(acceptance.criteria)} measurable requirement(s)"
                )
                if acceptance.overall_status == "INCONCLUSIVE":
                    lines.append(
                        "More compatible samples are needed before the decision is stable."
                    )

        fit = self.workload_fit
        lines.extend(["", f"FIT EVIDENCE: {human_fit_status(fit.overall_status)}"])
        if fit.configured:
            counts = fit.counts
            lines.append(
                "PASS "
                f"{counts.get('PASS', 0)} · FAIL {counts.get('FAIL', 0)} · "
                f"MEASURE {counts.get('UNKNOWN', 0)} · MORE EVIDENCE "
                f"{counts.get('INCONCLUSIVE', 0)}"
            )
            glyphs = {"PASS": "✓", "FAIL": "×", "UNKNOWN": "◇", "INCONCLUSIVE": "…"}
            for item in fit.constraints:
                status = str(item.get("status") or "UNKNOWN")
                display_status = human_fit_status(status)
                glyph = glyphs.get(status, "?")
                key = str(item.get("key") or "constraint")
                observed = item.get("observed")
                required = item.get("requirement")
                if edition is EditionProfile.ACADEMY:
                    lines.append(f"  {glyph} {display_status} · {item.get('reason')}")
                elif edition is EditionProfile.LAB:
                    lines.append(
                        f"  {glyph} {display_status} {key} · required={required} · "
                        f"observed={observed} · evidence={item.get('evidence_source')}"
                    )
                else:
                    lines.append(
                        f"  {glyph} {display_status} {key} · required={required} · "
                        f"observed={observed}"
                    )
        elif fit.note:
            lines.append(fit.note)

        lines.extend(self._real_use_lines(edition))

        opportunities = self.fit_opportunities.opportunities
        if opportunities:
            if edition is EditionProfile.ACADEMY:
                heading = "WHAT TO TRY NEXT"
            elif edition is EditionProfile.LAB:
                heading = "EXPERIMENT QUEUE"
            else:
                heading = "NEXT EXPERIMENTS"
            lines.extend(["", heading])
            for index, opportunity in enumerate(opportunities[:3], start=1):
                lines.append(f"  {index}. {opportunity.title}")
                if edition is EditionProfile.ACADEMY:
                    lines.append(f"     Why: {opportunity.reason}")
                elif edition is EditionProfile.LAB:
                    evidence = ", ".join(opportunity.evidence_keys) or "measured Pareto evidence"
                    lines.append(
                        f"     {opportunity.opportunity_class.value} · evidence={evidence}"
                    )
                    lines.append(f"     success={opportunity.success_criterion}")
                else:
                    lines.append(f"     {opportunity.reason}")
            lines.append("  No gain is predicted; each item is a falsifiable experiment.")

        coverage = fit.workload_evaluation_coverage
        if coverage:
            lines.extend(["", "WORKLOAD EVALUATION COVERAGE"])
            lines.append("COMPLETE" if coverage.get("complete") is True else "INCOMPLETE")
            covered = coverage.get("covered")
            missing = coverage.get("missing")
            for kind in ("languages", "domains", "tasks"):
                if isinstance(covered, dict):
                    values = covered.get(kind)
                    if isinstance(values, list) and values:
                        lines.append(f"  ✓ {kind}: " + ", ".join(str(item) for item in values))
                if isinstance(missing, dict):
                    values = missing.get(kind)
                    if isinstance(values, list) and values:
                        lines.append(
                            f"  ? missing {kind}: " + ", ".join(str(item) for item in values)
                        )
            if edition is EditionProfile.LAB:
                receipt_ids = coverage.get("receipt_ids")
                binding_ids = coverage.get("binding_ids")
                if isinstance(receipt_ids, list) and receipt_ids:
                    lines.append("  Receipts: " + ", ".join(str(item) for item in receipt_ids))
                if isinstance(binding_ids, list) and binding_ids:
                    lines.append("  Bindings: " + ", ".join(str(item) for item in binding_ids))

        if edition is EditionProfile.STUDIO:
            lines.extend(
                [
                    "",
                    (
                        "Next evidence: profile the current Champion on this machine, then compare "
                        "measured latency/throughput and capability floors against this workload."
                    ),
                ]
            )
        elif edition is EditionProfile.LAB:
            lines.extend(
                [
                    "",
                    (
                        "Utility claims require comparable eval receipts and measured "
                        "serving/resource "
                        "evidence; unknown constraints remain UNKNOWN."
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    (
                        "Why this matters: a model can improve one skill while becoming slower, "
                        "larger, or worse at another skill. Workload requirements make that "
                        "trade-off explicit."
                    ),
                ]
            )
        return chr(10).join(lines)

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
                    f"{event.get('sequence'):>4}  {event.get('kind')}  {event.get('recorded_at')}"
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
        self.workload = get_workload_view(self.root)
        self.workload_acceptance = get_workload_acceptance_view(self.root)
        self.workload_fit = get_workload_fit(self.root)
        self.usage_observations = get_usage_observation_summary(self.root)
        self.fit_opportunities = get_fit_opportunities(self.root)
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
        self.query_one("#workload-view", Static).update(self._workload_text())
        self.query_one("#history-view", Static).update(self._history_text())
        self.query_one("#candidates-view", Static).update(self._candidates_text())

    def action_previous_tab(self) -> None:
        tabs = self._tabs()
        ids = list(self.tab_order)
        current = tabs.active
        index = ids.index(current) if current in ids else 0
        tabs.active = ids[(index - 1) % len(ids)]

    def action_next_tab(self) -> None:
        tabs = self._tabs()
        ids = list(self.tab_order)
        current = tabs.active
        index = ids.index(current) if current in ids else 0
        tabs.active = ids[(index + 1) % len(ids)]

    def action_select_tab_index(self, index: int) -> None:
        if 0 <= index < len(self.tab_order):
            self._tabs().active = self.tab_order[index]

    def action_candidate_next(self) -> None:
        if not self._on_candidates_tab() or not self.candidates.candidates:
            return
        self.candidate_index = (self.candidate_index + 1) % len(self.candidates.candidates)
        self.query_one("#candidates-view", Static).update(self._candidates_text())

    def action_candidate_previous(self) -> None:
        if not self._on_candidates_tab() or not self.candidates.candidates:
            return
        self.candidate_index = (self.candidate_index - 1) % len(self.candidates.candidates)
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
                self.push_screen(
                    WorkflowFormScreen(
                        title="REJECT CANDIDATE",
                        description=(
                            "Optionally record why this Candidate was rejected. Frontierwright "
                            "also pins the current Champion/evaluation/workload identity "
                            "in history."
                        ),
                        fields=[FormField("reason", "Reason (optional)", "")],
                    ),
                    lambda values, target=model_id: self._submit_candidate_rejection(
                        values, target
                    ),
                )
                return
            else:
                return
        except FrontierwrightError as exc:
            self.notify(str(exc), severity="error")
            return
        self._refresh_all()

    def _submit_candidate_rejection(
        self,
        values: dict[str, str] | None,
        model_id: str,
    ) -> None:
        if values is None:
            return
        try:
            reject_candidate(
                self.root,
                model_id,
                reason=values.get("reason") or None,
            )
            self._refresh_all()
            self.notify("Candidate rejected and audit context recorded.")
        except FrontierwrightError as exc:
            self.notify(str(exc), severity="error")

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
                if isinstance(argv, list) and argv and isinstance(argv[0], str) and argv[0]:
                    return argv[0]
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        return sys.executable

    def _action_items(self) -> list[ActionItem]:
        def copy(en: str, ko: str) -> str:
            return _ui(self.language, en, ko)

        if not self.view.initialized:
            return [
                ActionItem(
                    "initialize_project",
                    copy("Create Frontierwright project", "Frontierwright 프로젝트 만들기"),
                    copy(
                        "Choose character name, origin, edition, and UI language.",
                        "프로젝트 이름, origin, edition, UI 언어를 선택합니다.",
                    ),
                )
            ]

        try:
            edition = EditionProfile(self.view.edition_profile or EditionProfile.STUDIO.value)
        except ValueError:
            edition = EditionProfile.STUDIO

        resource_action = ActionItem(
            "detect_resources",
            copy("Detect local resources", "로컬 자원 측정"),
            copy(
                "Measure CPU/RAM/disk/GPU and refresh feasibility.",
                "CPU/RAM/디스크/GPU를 측정해 현재 실행 가능성을 갱신합니다.",
            ),
        )
        dataset_action = ActionItem(
            "add_dataset",
            copy("Register local dataset", "로컬 데이터셋 등록"),
            copy(
                "Add PRETRAIN, SFT, or PREFERENCE data with an explicit classification.",
                "PRETRAIN/SFT/PREFERENCE 데이터를 명시적 분류와 함께 등록합니다.",
            ),
        )
        if edition is EditionProfile.ACADEMY:
            workload_action = ActionItem(
                "set_workload",
                "Describe an optional learning workload",
                "Connect real tasks to the model-development concepts you are learning.",
            )
            items = [dataset_action, resource_action, workload_action]
        elif edition is EditionProfile.LAB:
            workload_action = ActionItem(
                "set_workload",
                "Define workload / serving contract",
                "Pin task mixture, serving constraints, privacy, and capability floors.",
            )
            items = [workload_action, resource_action, dataset_action]
        else:
            workload_action = ActionItem(
                "set_workload",
                copy("Define your workload", "내 workload 정의"),
                copy(
                    "Tell Frontierwright what this model must do well on your machine.",
                    "이 모델이 내 PC에서 어떤 일을 잘해야 하는지 정의합니다.",
                ),
            )
            items = [resource_action, workload_action, dataset_action]
        if self.view.champion_model_id is not None:
            observe_title = {
                EditionProfile.ACADEMY: "Record what happened in real use",
                EditionProfile.STUDIO: "Record real-use outcome",
                EditionProfile.LAB: "Record operational outcome evidence",
            }[edition]
            observe_description = {
                EditionProfile.ACADEMY: (
                    "Mark a real use as success, failure, corrected, or abstained so later "
                    "experiments can learn from actual mistakes without storing the prompt."
                ),
                EditionProfile.STUDIO: (
                    "Track a real task outcome against this exact Champion; prompt/response "
                    "content is not stored by this evidence record."
                ),
                EditionProfile.LAB: (
                    "Attach privacy-minimal production outcome metadata to the exact model and "
                    "active workload. This is operational evidence, not an RL reward."
                ),
            }[edition]
            items.append(ActionItem("record_observation", observe_title, observe_description))

        if self.build.mode == "INTENT":
            items.append(
                ActionItem(
                    "build_intent",
                    "Set build intent",
                    "Choose an archetype and real capability priorities before stats exist.",
                )
            )
        elif self.build.mode == "TARGETS_FLOORS":
            if edition is not EditionProfile.ACADEMY:
                items.append(
                    ActionItem(
                        "build_growth_goals",
                        copy("Set growth rules", "성장 규칙 설정"),
                        copy(
                            (
                                "Choose metrics to IMPROVE or PROTECT, plus allowed regression "
                                "tolerances and hard constraints. Promotion checks these rules."
                            ),
                            (
                                "향상할 지표, 보호할 지표, 허용 퇴보 범위와 하드 제약을 정합니다. "
                                "Candidate 승격 때 이 규칙을 검사합니다."
                            ),
                        ),
                    )
                )
            if edition is not EditionProfile.STUDIO:
                items.append(
                    ActionItem(
                        "build_targets",
                        copy(
                            "Set absolute benchmark targets",
                            "절대 benchmark 목표 설정",
                        ),
                        copy(
                            (
                                "Set explicit numeric targets and floors on the exact frozen "
                                "legacy capability scale."
                            ),
                            (
                                "고정된 legacy capability scale에 절대 수치 목표와 floor를 "
                                "설정합니다."
                            ),
                        ),
                    )
                )

        if self.view.champion_model_id is not None:
            items.append(
                ActionItem(
                    "capability_v1",
                    copy("Run legacy Capability v1 smoke benchmark", "Legacy Capability v1 측정"),
                    copy(
                        (
                            "Run the frozen 64-task smoke benchmark as raw comparison evidence. "
                            "It does not redefine the origin-relative growth Stat scale."
                        ),
                        (
                            "64-task smoke benchmark를 원시 비교 근거로 실행합니다. "
                            "시작 모델=0인 성장 Stat 기준선은 바뀌지 않습니다."
                        ),
                    ),
                )
            )
            profile_title = {
                EditionProfile.ACADEMY: "Measure speed and memory",
                EditionProfile.STUDIO: "Profile Champion on this machine",
                EditionProfile.LAB: "Profile Champion serving evidence",
            }[edition]
            profile_description = {
                EditionProfile.ACADEMY: (
                    "Measure how fast the real model runs and how much memory it uses; "
                    "this shows why hardware changes which models are practical."
                ),
                EditionProfile.STUDIO: (
                    "Measure latency, throughput, RAM, and VRAM so workload fit uses your "
                    "actual machine rather than parameter-count guesses."
                ),
                EditionProfile.LAB: (
                    "Record reproducible serving/runtime evidence for this exact Champion "
                    "under explicit profile conditions."
                ),
            }[edition]
            items.append(
                ActionItem(
                    "profile_champion_inference",
                    profile_title,
                    profile_description,
                )
            )
            items.append(
                ActionItem(
                    "benchmark_catalog",
                    copy("Inspect benchmark catalog", "Benchmark catalog 보기"),
                    copy(
                        (
                            "See available benchmark sources, integration status, licensing "
                            "checks, and the next action for each source."
                        ),
                        (
                            "사용 가능한 benchmark source, 통합 상태, 라이선스 확인 상태와 "
                            "각 source의 다음 작업을 봅니다."
                        ),
                    ),
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
                    "measure_candidate_capability",
                    "Measure selected candidate Capability v1",
                    (
                        "Give the candidate the same frozen four-axis stats used by "
                        "Build targets and promotion gates."
                    ),
                )
            )
            items.append(
                ActionItem(
                    "evaluate_candidate",
                    "Evaluate selected candidate vs champion",
                    "Run the same raw evaluation pack on both exact model fingerprints.",
                )
            )
            candidate_profile_title = {
                EditionProfile.ACADEMY: "Measure candidate speed and memory",
                EditionProfile.STUDIO: "Profile selected candidate on this machine",
                EditionProfile.LAB: "Profile candidate serving evidence",
            }[edition]
            items.append(
                ActionItem(
                    "profile_candidate_inference",
                    candidate_profile_title,
                    (
                        "Use the same measured runtime dimensions as the Champion so "
                        "Candidate comparison can expose speed/memory trade-offs."
                    ),
                )
            )
        return items

    def action_action_center(self) -> None:
        try:
            profile = EditionProfile(self.view.edition_profile or EditionProfile.STUDIO.value)
        except ValueError:
            profile = EditionProfile.STUDIO
        policy = policy_for(profile)
        if self.language == "ko":
            title = {
                EditionProfile.ACADEMY: "ACADEMY 가이드",
                EditionProfile.STUDIO: "STUDIO 작업대",
                EditionProfile.LAB: "LAB 컨트롤 플레인",
            }[profile]
            intro = {
                EditionProfile.ACADEMY: (
                    "실제 모델 상태를 바꾸거나 측정하는 작업을 순서대로 실행합니다."
                ),
                EditionProfile.STUDIO: (
                    "내 모델을 내 workload와 자원 환경에 맞게 측정·튜닝·비교합니다."
                ),
                EditionProfile.LAB: (
                    "버전이 고정된 근거, PRIVATE 경계, 예산, lineage와 회귀 gate를 기준으로 "
                    "실험을 운영합니다."
                ),
            }[profile]
        else:
            title = policy.action_center_title
            intro = policy.action_center_intro
        self.push_screen(
            ActionCenterScreen(
                self._action_items(),
                title=title,
                intro=intro,
                language=self.language,
            ),
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

        if action_id == "benchmark_catalog":
            sources = benchmark_source_payloads()
            lines: list[str] = []
            for source in sources:
                raw_axes = source.get("axes")
                axes = (
                    ", ".join(str(item) for item in raw_axes)
                    if isinstance(raw_axes, list) and raw_axes
                    else "-"
                )
                lines.extend(
                    [
                        f"{source.get('source_id')} · {source.get('title')}",
                        (
                            f"  {source.get('kind')} · {source.get('lifecycle')} · "
                            f"integration={source.get('integration_status')}"
                        ),
                        f"  axes={axes}",
                        f"  source={source.get('upstream_url')}",
                        (
                            f"  license={source.get('license_status')} · "
                            f"cost={source.get('cost_hint')}"
                        ),
                        f"  next={source.get('next_action')}",
                        "",
                    ]
                )
            self.push_screen(
                InfoScreen(
                    _ui(self.language, "BENCHMARK CATALOG", "BENCHMARK CATALOG"),
                    chr(10).join(lines).rstrip(),
                )
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

        if action_id == "record_observation":
            self.push_screen(
                WorkflowFormScreen(
                    title="RECORD REAL-USE OUTCOME",
                    description=(
                        "Record outcome metadata for this exact model. Prompt/response content is "
                        "not stored here, and this record is not an RL reward."
                    ),
                    fields=[
                        FormField("task", "Task/category", "general-use"),
                        FormField(
                            "outcome", "Outcome (SUCCESS/FAILURE/CORRECTED/ABSTAINED)", "SUCCESS"
                        ),
                        FormField("domain", "Domain (optional)", ""),
                        FormField("language", "Language (optional)", ""),
                        FormField("failure_category", "Failure/correction category (optional)", ""),
                        FormField("latency", "Latency seconds (optional)", ""),
                        FormField(
                            "source",
                            "Evidence source (HUMAN_CONFIRMED/SYNTHETIC_TEST/IMPORTED)",
                            "HUMAN_CONFIRMED",
                        ),
                    ],
                ),
                self._submit_observation,
            )
            return

        if action_id == "set_workload":
            try:
                edition = EditionProfile(self.view.edition_profile or EditionProfile.STUDIO.value)
            except ValueError:
                edition = EditionProfile.STUDIO
            workload_payload = self.workload.profile if self.workload.configured else {}
            current_languages = workload_payload.get("languages")
            current_domains = workload_payload.get("domains")
            current_tasks = workload_payload.get("task_weights")
            current_floors = workload_payload.get("critical_floors")
            current_utility_weights = workload_payload.get("utility_weights")
            current_utility_scales = workload_payload.get("utility_scales")
            workload_fields = [
                FormField(
                    "name",
                    "Workload name",
                    self.workload.profile_name or "My real workload",
                ),
                FormField(
                    "languages",
                    "Languages (comma separated)",
                    ", ".join(str(x) for x in current_languages)
                    if isinstance(current_languages, list)
                    else "",
                ),
                FormField(
                    "domains",
                    "Domains (comma separated)",
                    ", ".join(str(x) for x in current_domains)
                    if isinstance(current_domains, list)
                    else "",
                ),
                FormField(
                    "tasks",
                    "Task weights (task=weight, comma separated)",
                    ", ".join(f"{k}={v}" for k, v in sorted(current_tasks.items()))
                    if isinstance(current_tasks, dict)
                    else "",
                    "research=2, coding=1",
                ),
            ]
            if edition is not EditionProfile.ACADEMY:
                workload_fields.extend(
                    [
                        FormField(
                            "context_p50",
                            "Typical context tokens (p50)",
                            str(workload_payload.get("context_tokens_p50") or ""),
                        ),
                        FormField(
                            "context_p95",
                            "Long context tokens (p95)",
                            str(workload_payload.get("context_tokens_p95") or ""),
                        ),
                        FormField(
                            "max_latency",
                            "Maximum acceptable latency seconds",
                            str(workload_payload.get("max_latency_seconds") or ""),
                        ),
                        FormField(
                            "min_tps",
                            "Minimum throughput tok/s",
                            str(workload_payload.get("min_tokens_per_second") or ""),
                        ),
                        FormField(
                            "utility_weights",
                            "Decision weights (metric=weight, comma separated)",
                            ", ".join(
                                f"{k}={v}" for k, v in sorted(current_utility_weights.items())
                            )
                            if isinstance(current_utility_weights, dict)
                            else "",
                            "capability.coding=2, serving.latency_p50=1",
                        ),
                        FormField(
                            "utility_scales",
                            "Decision scales (same metric=meaningful delta)",
                            ", ".join(f"{k}={v}" for k, v in sorted(current_utility_scales.items()))
                            if isinstance(current_utility_scales, dict)
                            else "",
                            "capability.coding=10, serving.latency_p50=0.25",
                        ),
                    ]
                )
            workload_fields.extend(
                [
                    FormField(
                        "privacy",
                        "Privacy classification",
                        str(workload_payload.get("privacy") or "PRIVATE"),
                    ),
                    FormField(
                        "floors",
                        "Capability floors (axis=value, comma separated)",
                        ", ".join(f"{k}={v}" for k, v in sorted(current_floors.items()))
                        if isinstance(current_floors, dict)
                        else "",
                        "general=60, coding=55",
                    ),
                ]
            )
            description = {
                EditionProfile.ACADEMY: (
                    "Describe real tasks you care about. This does not give the model points; "
                    "it helps explain why different model trade-offs matter. Advanced serving "
                    "constraints are intentionally hidden here."
                ),
                EditionProfile.STUDIO: (
                    "Define the actual work this model should fit. Unknown requirements stay "
                    "unknown until measured; this profile is evidence, not a personality quiz."
                ),
                EditionProfile.LAB: (
                    "Define the workload/serving contract used to judge candidate utility. "
                    "Exact evaluator, reward, and runtime receipts remain separate evidence."
                ),
            }[edition]
            self.push_screen(
                WorkflowFormScreen(
                    title=(
                        "LEARNING WORKLOAD"
                        if edition is EditionProfile.ACADEMY
                        else "WORKLOAD / SERVING CONTRACT"
                        if edition is EditionProfile.LAB
                        else "YOUR WORKLOAD"
                    ),
                    description=description,
                    fields=workload_fields,
                ),
                self._submit_workload,
            )
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

        if action_id == "build_growth_goals":
            def metrics_for(kind: str) -> str:
                return ", ".join(
                    str(goal.get("metric"))
                    for goal in self.build.growth_goals
                    if goal.get("kind") == kind and isinstance(goal.get("metric"), str)
                )

            def values_for(kind: str) -> str:
                rendered: list[str] = []
                for goal in self.build.growth_goals:
                    if goal.get("kind") != kind or not isinstance(goal.get("metric"), str):
                        continue
                    value = goal.get("value")
                    if value is None:
                        continue
                    suffix = "%" if goal.get("unit") == "PERCENT" else ""
                    rendered.append(f"{goal['metric']}={value}{suffix}")
                return ", ".join(rendered)

            self.push_screen(
                WorkflowFormScreen(
                    title=_ui(self.language, "SET GROWTH RULES", "성장 규칙 설정"),
                    description=_ui(
                        self.language,
                        (
                            "Growth Stat is origin-relative (origin=0). IMPROVE requires a clear "
                            "measured gain; PROTECT forbids regression; TOLERANCE allows a bounded "
                            "regression such as serving.latency_p50=5%."
                        ),
                        (
                            "성장 Stat은 시작 모델=0 기준입니다. IMPROVE는 명확한 향상, "
                            "PROTECT는 퇴보 금지, TOLERANCE는 serving.latency_p50=5%처럼 "
                            "허용 가능한 퇴보 범위를 뜻합니다."
                        ),
                    ),
                    fields=[
                        FormField(
                            "improve",
                            _ui(
                                self.language,
                                "IMPROVE metrics (comma separated)",
                                "향상할 지표 (쉼표 구분)",
                            ),
                            metrics_for("IMPROVE"),
                            "capability.coding, serving.throughput_p50",
                        ),
                        FormField(
                            "protect",
                            _ui(
                                self.language,
                                "PROTECT metrics (comma separated)",
                                "보호할 지표 (쉼표 구분)",
                            ),
                            metrics_for("PROTECT"),
                            "capability.reasoning",
                        ),
                        FormField(
                            "tolerance",
                            _ui(
                                self.language,
                                "TOLERANCE metric=value or value%",
                                "허용 퇴보 metric=value 또는 value%",
                            ),
                            values_for("TOLERANCE"),
                            "serving.latency_p50=5%",
                        ),
                        FormField(
                            "require",
                            _ui(
                                self.language,
                                "Absolute requirement metric=value",
                                "절대 요구조건 metric=value",
                            ),
                            values_for("ABSOLUTE_REQUIREMENT"),
                            "serving.throughput_p50=20",
                        ),
                        FormField(
                            "hard",
                            _ui(
                                self.language,
                                "Hard constraint key=value",
                                "하드 제약 key=value",
                            ),
                            values_for("HARD_CONSTRAINT"),
                            "privacy=PRIVATE",
                        ),
                    ],
                ),
                self._submit_growth_goals,
            )
            return

        if action_id == "build_targets":
            fields: list[FormField] = []
            for axis in ("general", "reasoning", "math", "coding"):
                current = self.view.stats.get(axis)
                current_text = "?" if current is None else str(current)
                fields.append(
                    FormField(
                        f"target_{axis}",
                        f"{axis.title()} target (current {current_text})",
                        str(self.build.targets.get(axis, "")),
                        "leave blank to omit",
                    )
                )
                fields.append(
                    FormField(
                        f"floor_{axis}",
                        f"{axis.title()} floor (current {current_text})",
                        str(self.build.floors.get(axis, "")),
                        "optional",
                    )
                )
            self.push_screen(
                WorkflowFormScreen(
                    title="SET MEASURED BUILD TARGETS",
                    description=(
                        "Targets/floors bind to the current frozen capability scale. "
                        "Blank fields are omitted; at least one target is required."
                    ),
                    fields=fields,
                ),
                self._submit_build_targets,
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

        if action_id in {"profile_champion_inference", "profile_candidate_inference"}:
            try:
                edition = EditionProfile(self.view.edition_profile or EditionProfile.STUDIO.value)
            except ValueError:
                edition = EditionProfile.STUDIO
            model_id = (
                self.view.champion_model_id
                if action_id == "profile_champion_inference"
                else self._selected_candidate_model_id()
            )
            if not model_id:
                self.notify("No model is available for runtime profiling.", severity="warning")
                return
            if edition is EditionProfile.ACADEMY:
                title = "MEASURE SPEED + MEMORY"
                description = (
                    "Run the real model several times to measure latency, throughput, RAM, "
                    "and VRAM. These measurements explain why hardware changes which models fit."
                )
                fields = [FormField("device", "Run on (auto/cpu/cuda)", "auto")]
            elif edition is EditionProfile.LAB:
                title = "PROFILE SERVING EVIDENCE"
                description = (
                    "Record exact local serving evidence for this model. Use identical profile "
                    "conditions when comparing Champion and Candidate."
                )
                fields = [
                    FormField(
                        "python",
                        "Runtime Python executable",
                        self._reference_python_default(),
                    ),
                    FormField("device", "Device (auto/cpu/cuda)", "auto"),
                    FormField("max_new_tokens", "Generated tokens per run", "16"),
                    FormField("runs", "Measured runs", "3"),
                ]
            else:
                title = "PROFILE MODEL ON THIS MACHINE"
                description = (
                    "Measure the exact model on your machine so Frontierwright can compare "
                    "real speed and memory trade-offs instead of guessing from model size."
                )
                fields = [
                    FormField(
                        "python",
                        "Runtime Python executable",
                        self._reference_python_default(),
                    ),
                    FormField("device", "Device (auto/cpu/cuda)", "auto"),
                    FormField("max_new_tokens", "Generated tokens per run", "16"),
                    FormField("runs", "Measured runs", "3"),
                ]
            self.push_screen(
                WorkflowFormScreen(title=title, description=description, fields=fields),
                lambda values, target=model_id: self._submit_inference_profile(values, target),
            )
            return

        if action_id == "capability_v1":
            self.push_screen(
                WorkflowFormScreen(
                    title="MEASURE FRONTIERWRIGHT CAPABILITY v1",
                    description=(
                        "Runs the frozen 64-task local smoke benchmark. Raw accuracy and "
                        "uncertainty remain inspectable; growth Stat stays origin-relative."
                    ),
                    fields=[
                        FormField(
                            "python",
                            "Evaluation Python executable",
                            self._reference_python_default(),
                        ),
                        FormField("device", "Device (auto/cpu/cuda)", "auto"),
                    ],
                ),
                self._submit_capability_v1,
            )
            return

        if action_id == "measure_candidate_capability":
            model_id = self._selected_candidate_model_id()
            self.push_screen(
                WorkflowFormScreen(
                    title="MEASURE CANDIDATE CAPABILITY v1",
                    description=(
                        "Runs the same frozen legacy smoke benchmark on the selected candidate. "
                        "This adds comparison evidence and never promotes the candidate by itself."
                    ),
                    fields=[
                        FormField("candidate_model_id", "Candidate model ID", model_id),
                        FormField(
                            "python",
                            "Evaluation Python executable",
                            self._reference_python_default(),
                        ),
                        FormField("device", "Device (auto/cpu/cuda)", "auto"),
                    ],
                ),
                self._submit_candidate_capability_v1,
            )
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
                axis: int(values[axis]) for axis in ("general", "reasoning", "math", "coding")
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

    def _submit_build_targets(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            targets: dict[str, float] = {}
            floors: dict[str, float] = {}
            for axis in ("general", "reasoning", "math", "coding"):
                target_raw = values.get(f"target_{axis}", "").strip()
                floor_raw = values.get(f"floor_{axis}", "").strip()
                if target_raw:
                    targets[axis] = float(target_raw)
                if floor_raw:
                    floors[axis] = float(floor_raw)
            set_build_targets(
                self.root,
                targets=targets,
                floors=floors,
            )
            self._refresh_all()
            self.notify("Measured build targets updated.")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_growth_goals(self, values: dict[str, str] | None) -> None:
        if values is None:
            return

        def labels(raw: str) -> tuple[str, ...]:
            return tuple(item.strip() for item in raw.split(",") if item.strip())

        def assignments(raw: str) -> dict[str, str]:
            result: dict[str, str] = {}
            for item in labels(raw):
                if "=" not in item:
                    raise ValueError(f"Expected metric=value: {item}")
                key, raw_value = (part.strip() for part in item.split("=", 1))
                if not key or not raw_value:
                    raise ValueError(f"Expected nonempty metric=value: {item}")
                if key in result:
                    raise ValueError(f"Repeated metric: {key}")
                result[key] = raw_value
            return result

        try:
            tolerance: dict[str, tuple[float, str]] = {}
            for metric, raw_value in assignments(values.get("tolerance", "")).items():
                unit = "PERCENT" if raw_value.endswith("%") else "ABSOLUTE"
                numeric = raw_value[:-1].strip() if unit == "PERCENT" else raw_value
                tolerance[metric] = (float(numeric), unit)

            absolute_requirements = {
                metric: float(raw_value)
                for metric, raw_value in assignments(values.get("require", "")).items()
            }
            set_growth_goals(
                self.root,
                improve=labels(values.get("improve", "")),
                protect=labels(values.get("protect", "")),
                tolerance=tolerance,
                absolute_requirements=absolute_requirements,
                hard_constraints=assignments(values.get("hard", "")),
            )
            self._refresh_all()
            self.notify(
                _ui(
                    self.language,
                    "Growth rules updated.",
                    "성장 규칙을 갱신했습니다.",
                )
            )
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_observation(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            raw_latency = values.get("latency", "").strip()
            record_usage_observation(
                self.root,
                task=values["task"],
                outcome=ObservationOutcome(values["outcome"].strip().upper()),
                domain=values.get("domain") or None,
                language=values.get("language") or None,
                failure_category=values.get("failure_category") or None,
                latency_seconds=float(raw_latency) if raw_latency else None,
                source=ObservationSource(
                    values.get("source", "HUMAN_CONFIRMED").strip().upper()
                ),
            )
            self._refresh_all()
            self.notify("Real-use outcome recorded. No prompt/response content was stored.")
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_workload(self, values: dict[str, str] | None) -> None:
        if values is None:
            return

        def labels(raw: str) -> tuple[str, ...]:
            return tuple(item.strip() for item in raw.split(",") if item.strip())

        def weighted(raw: str) -> dict[str, float]:
            result: dict[str, float] = {}
            for item in labels(raw):
                if "=" not in item:
                    raise ValueError(f"Expected name=number: {item}")
                name, raw_value = item.split("=", 1)
                name = name.strip()
                if not name:
                    raise ValueError("Workload weight name cannot be empty.")
                result[name] = float(raw_value)
            return result

        def optional_int(raw: str) -> int | None:
            raw = raw.strip()
            return int(raw) if raw else None

        def optional_float(raw: str) -> float | None:
            raw = raw.strip()
            return float(raw) if raw else None

        try:
            privacy = DatasetClassification(values.get("privacy", "PRIVATE").upper())
            profile = WorkloadProfile(
                name=values["name"],
                languages=labels(values.get("languages", "")),
                domains=labels(values.get("domains", "")),
                task_weights=weighted(values.get("tasks", "")),
                context_tokens_p50=optional_int(values.get("context_p50", "")),
                context_tokens_p95=optional_int(values.get("context_p95", "")),
                max_latency_seconds=optional_float(values.get("max_latency", "")),
                min_tokens_per_second=optional_float(values.get("min_tps", "")),
                privacy=privacy,
                critical_floors=weighted(values.get("floors", "")),
                utility_weights=weighted(values.get("utility_weights", "")),
                utility_scales=weighted(values.get("utility_scales", "")),
            )
            set_workload_profile(self.root, profile)
            self._refresh_all()
            self.notify("Workload profile updated.")
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

    def _submit_inference_profile(
        self,
        values: dict[str, str] | None,
        model_id: str,
    ) -> None:
        if values is None:
            return
        try:
            python_executable = values.get("python") or self._reference_python_default()
            max_new_tokens = int(values.get("max_new_tokens") or "16")
            measured_runs = int(values.get("runs") or "3")
            result = profile_reference_inference(
                self.root,
                python_executable=python_executable,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                warmup_runs=1,
                measured_runs=measured_runs,
                device=(values.get("device") or "auto").lower(),
            )
            self._refresh_all()
            latency = result.metrics.get("latency_seconds_p50")
            throughput = result.metrics.get("tokens_per_second_p50")
            peak_vram = result.metrics.get("peak_vram_bytes")
            self.notify(
                "Runtime measured: "
                f"p50={latency}s · {throughput} tok/s · "
                f"peak VRAM={_human_bytes(peak_vram)}"
            )
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_capability_v1(self, values: dict[str, str] | None) -> None:
        if values is None:
            return
        try:
            result = run_capability_v1(
                self.root,
                python_executable=values["python"],
                device=(values.get("device") or "auto").lower(),
            )
            self._refresh_all()
            rendered = ", ".join(
                f"{axis.title()}={result.stats.get(axis, '?')}"
                for axis in ("general", "reasoning", "math", "coding")
            )
            self.notify("Capability v1 measured: " + rendered)
        except (FrontierwrightError, KeyError, ValueError) as exc:
            self.notify(str(exc), severity="error")

    def _submit_candidate_capability_v1(
        self,
        values: dict[str, str] | None,
    ) -> None:
        if values is None:
            return
        try:
            model_id = values["candidate_model_id"]
            result = run_capability_v1(
                self.root,
                model_id=model_id,
                python_executable=values["python"],
                device=(values.get("device") or "auto").lower(),
            )
            self._refresh_all()
            self.notify(f"Candidate Capability v1 measured: {result.model_id}")
            self.push_screen(
                CandidateScreen(compare_candidate(self.root, model_id)),
                lambda decision: self._candidate_result(model_id, decision),
            )
        except (FrontierwrightError, KeyError, ValueError) as exc:
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
        self.push_screen(HelpScreen(self.language, self.view.edition_profile))

    async def action_back(self) -> None:
        return
