"""Evidence-driven next experiments for fitting a model to a real workload.

This planner never predicts capability gains. It turns measured FAIL/UNKNOWN constraints
into the next falsifiable experiment or measurement that can close the evidence gap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from frontierwright.editions import EditionProfile


class OpportunityClass(StrEnum):
    BLOCKING_EVIDENCE = "BLOCKING_EVIDENCE"
    FIT_GAP = "FIT_GAP"
    HEADROOM_EXPERIMENT = "HEADROOM_EXPERIMENT"
    CONTINUAL_IMPROVEMENT = "CONTINUAL_IMPROVEMENT"


@dataclass(frozen=True)
class FitOpportunity:
    opportunity_id: str
    opportunity_class: OpportunityClass
    action: str
    title: str
    reason: str
    evidence_keys: tuple[str, ...]
    success_criterion: str
    improvement_prediction: None = None

    def to_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "opportunity_class": self.opportunity_class.value,
        }


@dataclass(frozen=True)
class FitOpportunityPlan:
    edition: EditionProfile
    model_id: str | None
    workload_configured: bool
    opportunities: tuple[FitOpportunity, ...]
    note: str

    def to_payload(self) -> dict[str, object]:
        return {
            "edition": self.edition.value,
            "model_id": self.model_id,
            "workload_configured": self.workload_configured,
            "opportunities": [item.to_payload() for item in self.opportunities],
            "note": self.note,
        }


def _edition_title(
    edition: EditionProfile,
    *,
    academy: str,
    studio: str,
    lab: str,
) -> str:
    if edition is EditionProfile.ACADEMY:
        return academy
    if edition is EditionProfile.LAB:
        return lab
    return studio


def _append_unique(
    items: list[FitOpportunity],
    seen: set[str],
    item: FitOpportunity,
) -> None:
    if item.opportunity_id in seen:
        return
    seen.add(item.opportunity_id)
    items.append(item)


def plan_fit_opportunities(
    *,
    edition: EditionProfile,
    model_id: str | None,
    workload_configured: bool,
    constraints: list[dict[str, object]],
    model_fit: dict[str, object] | None,
    workload_evaluation_coverage: dict[str, object] | None,
    observation_summary: dict[str, object] | None = None,
) -> FitOpportunityPlan:
    """Return falsifiable next experiments without forecasting improvement."""

    items: list[FitOpportunity] = []
    seen: set[str] = set()

    if not workload_configured:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="define-workload",
                opportunity_class=OpportunityClass.BLOCKING_EVIDENCE,
                action="frontierwright workload set",
                title=_edition_title(
                    edition,
                    academy="Optionally describe what you want the model to learn",
                    studio="Define your real workload before optimizing",
                    lab="Define the workload and serving contract",
                ),
                reason=(
                    "Without explicit task, latency, privacy, and capability requirements, "
                    "Frontierwright can measure a model but cannot judge user-specific fit."
                ),
                evidence_keys=("workload.profile",),
                success_criterion="A versioned Workload Profile exists.",
            ),
        )
        return FitOpportunityPlan(
            edition=edition,
            model_id=model_id,
            workload_configured=False,
            opportunities=tuple(items),
            note="No performance gain is predicted; the next step establishes the objective.",
        )

    if model_id is None:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="establish-champion",
                opportunity_class=OpportunityClass.BLOCKING_EVIDENCE,
                action="frontierwright model import OR frontierwright birth zero",
                title=_edition_title(
                    edition,
                    academy="Create a real model to measure",
                    studio="Import or establish your starting Champion",
                    lab="Register the exact controlled model under test",
                ),
                reason="Workload fit requires an exact model identity and fingerprint.",
                evidence_keys=("model.identity",),
                success_criterion="A current Champion model exists with an exact fingerprint.",
            ),
        )
        return FitOpportunityPlan(
            edition=edition,
            model_id=None,
            workload_configured=True,
            opportunities=tuple(items),
            note="No performance gain is predicted; establish the model under test first.",
        )

    by_status: dict[str, list[dict[str, object]]] = {
        "FAIL": [],
        "UNKNOWN": [],
        "INCONCLUSIVE": [],
        "PASS": [],
    }
    for constraint in constraints:
        status = str(constraint.get("status") or "UNKNOWN")
        by_status.setdefault(status, []).append(constraint)

    unknown_keys = {
        str(item.get("key"))
        for item in by_status.get("UNKNOWN", [])
        if isinstance(item.get("key"), str)
    }
    fail_keys = {
        str(item.get("key"))
        for item in by_status.get("FAIL", [])
        if isinstance(item.get("key"), str)
    }
    inconclusive_keys = {
        str(item.get("key"))
        for item in by_status.get("INCONCLUSIVE", [])
        if isinstance(item.get("key"), str)
    }

    if any(key.startswith("capability.") for key in unknown_keys):
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="measure-capability",
                opportunity_class=OpportunityClass.BLOCKING_EVIDENCE,
                action="frontierwright eval capability-v1",
                title=_edition_title(
                    edition,
                    academy="Measure what the model can do before changing it",
                    studio="Measure missing capability evidence",
                    lab="Close capability evidence gaps",
                ),
                reason="At least one hard capability floor lacks comparable measured evidence.",
                evidence_keys=tuple(
                    sorted(key for key in unknown_keys if key.startswith("capability."))
                ),
                success_criterion=(
                    "Every referenced capability floor has comparable measured evidence."
                ),
            ),
        )

    serving_unknown = sorted(key for key in unknown_keys if key.startswith("serving."))
    if serving_unknown:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="profile-model-fit",
                opportunity_class=OpportunityClass.BLOCKING_EVIDENCE,
                action="frontierwright operate profile",
                title=_edition_title(
                    edition,
                    academy="Run the model and measure how it fits this machine",
                    studio="Profile the Champion under the real serving contract",
                    lab="Capture comparable serving/resource evidence",
                ),
                reason=(
                    "Latency, throughput, context capacity, or memory fit has not been "
                    "measured under compatible conditions; parameter count alone is not enough "
                    "to claim feasibility."
                ),
                evidence_keys=tuple(serving_unknown),
                success_criterion=(
                    "A model-specific inference receipt supplies the missing serving and "
                    "resource evidence."
                ),
            ),
        )

    if "evaluation.workload_coverage" in unknown_keys:
        coverage = workload_evaluation_coverage or {}
        missing = coverage.get("missing")
        missing_parts: list[str] = []
        if isinstance(missing, dict):
            for kind in ("languages", "domains", "tasks"):
                values = missing.get(kind)
                if isinstance(values, list) and values:
                    missing_parts.append(f"{kind}=" + ",".join(str(value) for value in values))
        suffix = "; missing " + " | ".join(missing_parts) if missing_parts else ""
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="measure-workload-coverage",
                opportunity_class=OpportunityClass.BLOCKING_EVIDENCE,
                action="frontierwright eval import-lm-eval/import-lighteval + workload bind-eval",
                title=_edition_title(
                    edition,
                    academy="Test the model on the tasks you actually care about",
                    studio="Add workload-specific evaluation evidence",
                    lab="Bind exact evaluator evidence to workload requirements",
                ),
                reason=(
                    "Generic capability evidence does not prove performance on the declared "
                    "languages, domains, or task mixture" + suffix + "."
                ),
                evidence_keys=("evaluation.workload_coverage",),
                success_criterion=(
                    "Every declared workload language/domain/task is explicitly bound to exact "
                    "stored task/version/metric evidence."
                ),
            ),
        )

    if "evaluation.workload_acceptance" in unknown_keys:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="configure-workload-acceptance",
                opportunity_class=OpportunityClass.BLOCKING_EVIDENCE,
                action="frontierwright workload acceptance example --json",
                title=_edition_title(
                    edition,
                    academy="Choose what counts as success before deciding the next model",
                    studio="Turn your optimization goals into measurable success criteria",
                    lab="Pin the exact acceptance contract before trusting the experiment",
                ),
                reason=(
                    "Workload acceptance is mandatory decision evidence but no compatible "
                    "versioned success criteria are configured or assessed."
                ),
                evidence_keys=("evaluation.workload_acceptance",),
                success_criterion=(
                    "A versioned acceptance contract is created and assessed against exact "
                    "compatible evaluation evidence."
                ),
            ),
        )

    if "privacy.serving_boundary" in unknown_keys:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="measure-serving-boundary",
                opportunity_class=OpportunityClass.BLOCKING_EVIDENCE,
                action="profile/import serving evidence with an explicit execution boundary",
                title=_edition_title(
                    edition,
                    academy="Verify where the private model actually runs",
                    studio="Prove the serving boundary for private work",
                    lab="Attest the controlled serving boundary",
                ),
                reason=(
                    "The workload is non-public but compatible serving-boundary "
                    "evidence is missing."
                ),
                evidence_keys=("privacy.serving_boundary",),
                success_criterion=(
                    "Serving evidence proves LOCAL_MACHINE or CONTROLLED_PRIVATE execution."
                ),
            ),
        )

    if inconclusive_keys:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="reduce-decision-uncertainty",
                opportunity_class=OpportunityClass.BLOCKING_EVIDENCE,
                action=(
                    "collect additional compatible evaluation/profile samples, then reassess "
                    "the exact workload criteria"
                ),
                title=_edition_title(
                    edition,
                    academy="Measure a little more until the result is clear",
                    studio="Reduce uncertainty on the unresolved workload decision",
                    lab="Increase decision evidence under the same protocol",
                ),
                reason=(
                    "At least one measured criterion overlaps its decision threshold. "
                    "The result is not treated as a permanent unknown; gather more compatible "
                    "evidence until the criterion resolves or the measurement budget is exhausted."
                ),
                evidence_keys=tuple(sorted(inconclusive_keys)),
                success_criterion=(
                    "The configured uncertainty rule places every listed criterion wholly on one "
                    "side of its threshold, or the measurement budget is explicitly exhausted."
                ),
            ),
        )

    capability_fail = sorted(key for key in fail_keys if key.startswith("capability."))
    if capability_fail:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="specialize-capability-gap",
                opportunity_class=OpportunityClass.FIT_GAP,
                action="create a bounded CPT/SFT/LoRA/QLoRA/alignment candidate, then re-evaluate",
                title=_edition_title(
                    edition,
                    academy="Train one focused descendant and see what really changes",
                    studio="Run a specialist training experiment for the failed capability floor",
                    lab="Launch a controlled specialization/alignment experiment",
                ),
                reason=(
                    "Measured capability is below at least one declared hard floor. "
                    "The planner does not predict which intervention will improve it."
                ),
                evidence_keys=tuple(capability_fail),
                success_criterion=(
                    "A descendant is evaluated on the same capability/workload evidence and the "
                    "failed floor improves without unacceptable regressions."
                ),
            ),
        )

    serving_performance_fail = sorted(
        key for key in fail_keys if key in {"serving.latency_p50", "serving.throughput_p50"}
    )
    if serving_performance_fail:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="optimize-serving-gap",
                opportunity_class=OpportunityClass.FIT_GAP,
                action=(
                    "test quantization/distillation/runtime changes, then profile under "
                    "identical conditions"
                ),
                title=_edition_title(
                    edition,
                    academy="Try a smaller/faster descendant and measure the trade-off",
                    studio="Optimize the model for the failed latency/throughput contract",
                    lab="Run an efficiency frontier experiment",
                ),
                reason=(
                    "Measured serving performance violates the workload contract. "
                    "Only a same-condition re-profile can establish a real improvement."
                ),
                evidence_keys=tuple(serving_performance_fail),
                success_criterion=(
                    "A candidate satisfies the serving constraint under a comparable profile while "
                    "capability regressions remain visible."
                ),
            ),
        )

    if "serving.context_p95" in fail_keys:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="close-context-gap",
                opportunity_class=OpportunityClass.FIT_GAP,
                action=(
                    "test a context-capacity/runtime/model change, then verify the exact "
                    "context contract"
                ),
                title=_edition_title(
                    edition,
                    academy="Test why the model cannot yet cover your longer context",
                    studio="Close the workload context-capacity gap",
                    lab="Run a context-capacity architecture/runtime experiment",
                ),
                reason="Verified context capacity is below the workload p95 requirement.",
                evidence_keys=("serving.context_p95",),
                success_criterion=(
                    "A compatible receipt verifies context capacity at or above workload p95."
                ),
            ),
        )

    if "privacy.serving_boundary" in fail_keys:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="fix-serving-boundary",
                opportunity_class=OpportunityClass.FIT_GAP,
                action="move execution to LOCAL_MACHINE or CONTROLLED_PRIVATE and re-measure",
                title=_edition_title(
                    edition,
                    academy="Keep private work inside an allowed environment",
                    studio="Move serving inside the workload privacy boundary",
                    lab="Correct the execution/data-boundary violation",
                ),
                reason=(
                    "The measured serving boundary conflicts with the workload privacy "
                    "classification."
                ),
                evidence_keys=("privacy.serving_boundary",),
                success_criterion="Serving evidence passes the workload privacy constraint.",
            ),
        )

    observed = observation_summary or {}
    observed_failures = observed.get("failures")
    observed_corrected = observed.get("corrected")
    failure_count = (
        int(observed_failures)
        if isinstance(observed_failures, int) and not isinstance(observed_failures, bool)
        else 0
    )
    corrected_count = (
        int(observed_corrected)
        if isinstance(observed_corrected, int) and not isinstance(observed_corrected, bool)
        else 0
    )
    if failure_count + corrected_count > 0:
        categories = observed.get("failure_categories")
        category_text = ""
        if isinstance(categories, dict) and categories:
            top = sorted(
                (
                    (str(key), int(value))
                    for key, value in categories.items()
                    if isinstance(value, int) and not isinstance(value, bool) and value > 0
                ),
                key=lambda item: (-item[1], item[0].casefold()),
            )[:3]
            if top:
                category_text = "; top categories=" + ", ".join(
                    f"{name}:{count}" for name, count in top
                )
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="learn-from-observed-failures",
                opportunity_class=OpportunityClass.CONTINUAL_IMPROVEMENT,
                action=(
                    "frontierwright observe summary; turn repeated real-use failures into "
                    "versioned evaluation/data evidence, then run one bounded intervention"
                ),
                title=_edition_title(
                    edition,
                    academy="Turn real mistakes into the next lesson",
                    studio="Use real failures for the next targeted experiment",
                    lab="Promote production failures into controlled eval/data/reward evidence",
                ),
                reason=(
                    f"The exact model has {failure_count} failed and {corrected_count} "
                    f"user-corrected observed uses{category_text}. These observations are "
                    "operational evidence, not RL rewards and not proof of the best intervention."
                ),
                evidence_keys=(
                    "usage_observations.failures",
                    "usage_observations.corrected",
                ),
                success_criterion=(
                    "Repeated failures are represented in versioned evaluation/data evidence, and "
                    "a descendant is tested against the same real-use failure pattern with "
                    "regressions still visible."
                ),
            ),
        )

    model_fit_payload = model_fit or {}
    free_vram = model_fit_payload.get("cuda_memory_free_min_sampled_bytes")
    if (
        not by_status.get("FAIL")
        and not by_status.get("UNKNOWN")
        and isinstance(free_vram, int)
        and not isinstance(free_vram, bool)
        and free_vram > 0
    ):
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="calibrate-headroom-experiment",
                opportunity_class=OpportunityClass.HEADROOM_EXPERIMENT,
                action=(
                    "calibrate one higher-cost descendant/context/adapter experiment before "
                    "execution"
                ),
                title=_edition_title(
                    edition,
                    academy="Explore what the measured spare VRAM can safely support",
                    studio="Use measured headroom for the next controlled experiment",
                    lab="Probe the next point on the resource/capability frontier",
                ),
                reason=(
                    f"The last comparable profile observed {free_vram} bytes of minimum sampled "
                    "free VRAM. This is evidence of headroom under those conditions, not proof "
                    "that any particular larger model will fit."
                ),
                evidence_keys=("resource.cuda_memory_free_min_sampled_bytes",),
                success_criterion=(
                    "Calibration and a same-condition profile establish whether the higher-cost "
                    "experiment actually fits and improves user utility."
                ),
            ),
        )

    if not items:
        _append_unique(
            items,
            seen,
            FitOpportunity(
                opportunity_id="continue-pareto-experiment",
                opportunity_class=OpportunityClass.CONTINUAL_IMPROVEMENT,
                action=(
                    "create one bounded candidate and compare it on the same workload/resource "
                    "evidence"
                ),
                title=_edition_title(
                    edition,
                    academy="Try one change and learn from the measured trade-off",
                    studio="Explore the next user-value trade-off",
                    lab="Advance the measured candidate frontier",
                ),
                reason=(
                    "No explicit blocking evidence or failed constraint currently determines the "
                    "next move. Improvement remains an experiment, not a prediction."
                ),
                evidence_keys=(),
                success_criterion=(
                    "The candidate is compared against the Champion on comparable workload, "
                    "capability, and resource evidence before promotion."
                ),
            ),
        )

    return FitOpportunityPlan(
        edition=edition,
        model_id=model_id,
        workload_configured=True,
        opportunities=tuple(items),
        note=(
            "Opportunities are evidence-driven experiments. Frontierwright does not predict "
            "that any listed intervention will improve the model."
        ),
    )
