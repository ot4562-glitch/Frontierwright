"""Typer command surface for humans and automation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from frontierwright import __version__
from frontierwright.data import DatasetClassification, DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.editions import EditionProfile
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import REFERENCE_LM_PACK
from frontierwright.execution import BackendDataBoundary, HardBudgets, PermissionLevel
from frontierwright.paths import TrainingPathId
from frontierwright.recipes import (
    BYTE_SHARDS_PLUGIN_ID,
    PREFERENCE_JSONL_PLUGIN_ID,
    SNAPSHOT_COPY_PLUGIN_ID,
    TEXT_LINES_PLUGIN_ID,
)
from frontierwright.reference_backend import backend_spec_payload
from frontierwright.reference_tokenizer import DEFAULT_MAX_TRAINING_BYTES
from frontierwright.service import (
    ActionPreflightView,
    BirthView,
    BuildView,
    CandidateView,
    CompareView,
    DataView,
    EvaluationCompareView,
    EvaluationRunView,
    ExportVerifyView,
    ExportView,
    GenerationView,
    HistoryView,
    InferenceProfileView,
    InterventionsView,
    LabAdaptersView,
    MergeView,
    PathsView,
    PlanView,
    QuantizeView,
    ResourceView,
    RunView,
    StatsView,
    StatusView,
    TokenizersView,
    TokenizerView,
    WorkloadFitView,
    WorkloadView,
    add_local_dataset,
    bind_workload_evaluation,
    birth_zero_model,
    calibrate_training_plan,
    compare_candidate,
    compare_candidate_evaluation,
    connect_lab_adapter,
    create_training_plan,
    detect_resources,
    disconnect_lab_adapter,
    execute_training_plan,
    export_champion_bundle,
    generate_reference_text,
    get_birth_view,
    get_build_view,
    get_candidates_view,
    get_data_preparation_plugins,
    get_data_view,
    get_evaluation_packs,
    get_history_view,
    get_interventions_view,
    get_lab_adapters,
    get_paths_view,
    get_plan_view,
    get_resource_view,
    get_run_view,
    get_stats_view,
    get_status,
    get_tokenizers_view,
    get_workload_fit,
    get_workload_view,
    import_lighteval_evidence,
    import_lm_eval_evidence,
    import_local_model,
    import_vllm_serving_evidence,
    ingest_stats,
    initialize_project,
    merge_reference_models,
    preflight_capability_v1,
    preflight_evaluation_pack,
    preflight_export_champion_bundle,
    preflight_merge_reference_models,
    preflight_prepare_dataset,
    preflight_prepare_dataset_mixture,
    preflight_project_tokenizer,
    preflight_quantize_reference_model,
    preflight_training_calibration,
    preflight_zero_birth,
    prepare_dataset,
    prepare_dataset_mixture,
    profile_reference_inference,
    promote_candidate,
    quantize_reference_model,
    reconcile_training_run,
    reject_candidate,
    repair_run_receipt,
    run_capability_v1,
    run_evaluation_pack,
    set_build_intent,
    set_build_targets,
    set_project_edition,
    set_workload_profile,
    train_project_tokenizer,
    verify_export_bundle,
)
from frontierwright.workloads import WorkloadProfile

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Fit a model to your workload, machine, and goals.",
)
project_app = typer.Typer(help="Create and inspect Frontierwright project state.")
resources_app = typer.Typer(help="Detect and inspect project compute resources.")
workload_app = typer.Typer(help="Describe the user workload the model should fit.")
build_app = typer.Typer(help="Inspect and edit the desired model build.")
stats_app = typer.Typer(help="Inspect or ingest capability evaluation evidence.")
evaluation_app = typer.Typer(help="Run and inspect raw evaluation packs.")
data_app = typer.Typer(help="Register and inspect user/lab datasets.")
plan_app = typer.Typer(help="Create and inspect pinned training plans.")
backend_app = typer.Typer(help="Inspect and configure training backends.")
birth_app = typer.Typer(help="Materialize and inspect zero-model birth state.")
evolve_app = typer.Typer(help="Evolve models through artifact transforms.")
optimize_app = typer.Typer(help="Optimize model artifacts for deployment/inference.")
operate_app = typer.Typer(help="Export and operate accepted model artifacts.")
lab_app = typer.Typer(help="Connect and inspect controlled private Lab infrastructure.")
lab_adapters_app = typer.Typer(help="Manage Lab adapter manifests.")
lab_app.add_typer(lab_adapters_app, name="adapters")
app.add_typer(project_app, name="project")
app.add_typer(resources_app, name="resources")
app.add_typer(workload_app, name="workload")
app.add_typer(build_app, name="build")
app.add_typer(stats_app, name="stats")
app.add_typer(evaluation_app, name="eval")
app.add_typer(data_app, name="data")
app.add_typer(plan_app, name="plan")
app.add_typer(backend_app, name="backend")
app.add_typer(birth_app, name="birth")
app.add_typer(evolve_app, name="evolve")
app.add_typer(optimize_app, name="optimize")
app.add_typer(operate_app, name="operate")
app.add_typer(lab_app, name="lab")

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"frontierwright {__version__}")
        raise typer.Exit()


@app.callback()
def _root_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show Frontierwright version and exit.",
        ),
    ] = False,
) -> None:
    del version


def _emit_json(payload: dict[str, object]) -> None:
    typer.echo(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _fail(exc: FrontierwrightError, *, json_output: bool) -> NoReturn:
    if json_output:
        _emit_json(
            {
                "schema_version": 1,
                "ok": False,
                "error": {"code": exc.code, "message": str(exc)},
            }
        )
    else:
        console.print(f"[red]{exc.code}[/red]: {exc}")
    raise typer.Exit(code=exc.exit_code)


def _status_payload(view: StatusView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _birth_payload(view: BirthView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _tokenizer_payload(view: TokenizerView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _tokenizers_payload(view: TokenizersView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _resource_payload(view: ResourceView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _workload_payload(view: WorkloadView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _workload_fit_payload(view: WorkloadFitView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _build_payload(view: BuildView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _stats_payload(view: StatsView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _evaluation_payload(view: EvaluationRunView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _evaluation_compare_payload(
    view: EvaluationCompareView,
) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _data_payload(view: DataView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _paths_payload(view: PathsView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _interventions_payload(view: InterventionsView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _lab_adapters_payload(view: LabAdaptersView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _plan_payload(view: PlanView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _run_payload(view: RunView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _candidate_payload(view: CandidateView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _merge_payload(view: MergeView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _quantize_payload(view: QuantizeView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _export_payload(view: ExportView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _export_verify_payload(view: ExportVerifyView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _generation_payload(view: GenerationView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _inference_profile_payload(view: InferenceProfileView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _compare_payload(view: CompareView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _history_payload(view: HistoryView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _parse_permission(value: str) -> PermissionLevel:
    try:
        return PermissionLevel[value.strip().upper()]
    except KeyError as exc:
        allowed = ", ".join(item.name for item in PermissionLevel)
        raise FrontierwrightError(
            "INVALID_PERMISSION",
            f"Permission must be one of: {allowed}",
            2,
        ) from exc


def _load_config_json(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "INVALID_PLAN_CONFIG",
            "Plan config file must be valid UTF-8 JSON.",
            2,
        ) from exc
    if not isinstance(raw, dict):
        raise FrontierwrightError(
            "INVALID_PLAN_CONFIG",
            "Plan config JSON must be an object.",
            2,
        )
    return raw


def _parse_assignments(items: list[str], label: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise FrontierwrightError(
                "INVALID_BUILD_ARGUMENT",
                f"{label} must use axis=value syntax: {item}",
                2,
            )
        axis, raw_value = item.split("=", 1)
        axis = axis.strip().lower()
        if not axis:
            raise FrontierwrightError(
                "INVALID_BUILD_ARGUMENT",
                f"{label} axis cannot be empty.",
                2,
            )
        if axis in parsed:
            raise FrontierwrightError(
                "INVALID_BUILD_ARGUMENT",
                f"{label} repeats axis: {axis}",
                2,
            )
        try:
            parsed[axis] = int(raw_value)
        except ValueError as exc:
            raise FrontierwrightError(
                "INVALID_BUILD_ARGUMENT",
                f"{label} value must be an integer: {item}",
                2,
            ) from exc
    return parsed


def _parse_float_assignments(items: list[str], label: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise FrontierwrightError(
                "INVALID_WORKLOAD_ARGUMENT",
                f"{label} must use name=value syntax: {item}",
                2,
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise FrontierwrightError(
                "INVALID_WORKLOAD_ARGUMENT",
                f"{label} name cannot be empty.",
                2,
            )
        if key in parsed:
            raise FrontierwrightError(
                "INVALID_WORKLOAD_ARGUMENT",
                f"{label} repeats name: {key}",
                2,
            )
        try:
            parsed[key] = float(raw_value)
        except ValueError as exc:
            raise FrontierwrightError(
                "INVALID_WORKLOAD_ARGUMENT",
                f"{label} value must be numeric: {item}",
                2,
            ) from exc
    return parsed


def _print_workload(view: WorkloadView) -> None:
    console.print("[bold]WORKLOAD[/bold]")
    if not view.configured:
        console.print("No workload profile configured.")
        return
    profile = view.profile
    console.print(f"Name: {view.profile_name or profile.get('name') or 'UNKNOWN'}")
    console.print(f"Profile: {view.profile_id}")
    console.print(f"Hash: {view.profile_hash}")
    languages = profile.get("languages")
    domains = profile.get("domains")
    console.print(
        "Languages: "
        + (
            ", ".join(str(x) for x in languages)
            if isinstance(languages, list) and languages
            else "UNSPECIFIED"
        )
    )
    console.print(
        "Domains: "
        + (
            ", ".join(str(x) for x in domains)
            if isinstance(domains, list) and domains
            else "UNSPECIFIED"
        )
    )
    task_weights = profile.get("task_weights")
    if isinstance(task_weights, dict) and task_weights:
        console.print("Task weights:")
        for key, value in sorted(task_weights.items()):
            console.print(f"  {key}: {value}")
    console.print(
        f"Context p50/p95: {profile.get('context_tokens_p50') or 'UNKNOWN'} / "
        f"{profile.get('context_tokens_p95') or 'UNKNOWN'} tokens"
    )
    console.print(f"Max latency: {profile.get('max_latency_seconds') or 'UNSPECIFIED'} s")
    console.print(f"Min throughput: {profile.get('min_tokens_per_second') or 'UNSPECIFIED'} tok/s")
    console.print(f"Privacy: {profile.get('privacy') or 'UNKNOWN'}")
    floors = profile.get("critical_floors")
    if isinstance(floors, dict) and floors:
        console.print("Critical capability floors:")
        for axis, value in sorted(floors.items()):
            console.print(f"  {axis.title()}: {value}")
    utility_weights = profile.get("utility_weights")
    utility_scales = profile.get("utility_scales")
    if isinstance(utility_weights, dict) and utility_weights and isinstance(utility_scales, dict):
        console.print("Explicit user utility:")
        for key, value in sorted(utility_weights.items()):
            console.print(f"  {key}: weight={value} · scale={utility_scales.get(key)}")
        console.print("  Missing comparable evidence keeps the relative utility result INCOMPLETE.")


def _print_workload_fit(view: WorkloadFitView) -> None:
    console.print(f"[bold]WORKLOAD FIT: {view.overall_status}[/bold]")
    if not view.configured:
        console.print(view.note or "No workload profile configured.")
        return
    if view.model_id:
        console.print(f"Model: {view.model_id}")
    for item in view.constraints:
        console.print(
            f"[{item.get('status', 'UNKNOWN')}] {item.get('key')} · "
            f"required={item.get('requirement')} · observed={item.get('observed')}"
        )
        if item.get("reason"):
            console.print(f"  {item['reason']}")
    coverage = view.workload_evaluation_coverage
    if coverage:
        console.print("Workload evaluation coverage:")
        receipt_ids = coverage.get("receipt_ids")
        receipt_count = len(receipt_ids) if isinstance(receipt_ids, list) else 0
        console.print(
            "  "
            + ("COMPLETE" if coverage.get("complete") is True else "INCOMPLETE")
            + f" · receipts={receipt_count}"
        )
        missing = coverage.get("missing")
        if isinstance(missing, dict):
            for kind in ("languages", "domains", "tasks"):
                values = missing.get(kind)
                if isinstance(values, list) and values:
                    console.print(f"  Missing {kind}: " + ", ".join(str(x) for x in values))
    if view.note:
        console.print(view.note)


def _print_build(view: BuildView) -> None:
    console.print(f"[bold]BUILD[/bold] · {view.mode or 'NOT_READY'}")
    if view.archetype:
        console.print(f"Archetype: {view.archetype}")
    if view.priorities:
        console.print("Priorities:")
        for axis, value in sorted(view.priorities.items()):
            console.print(f"  {axis.title():10} {value}")
    if view.targets:
        console.print("Targets:")
        for axis, value in sorted(view.targets.items()):
            console.print(f"  {axis.title():10} {value}")
    if view.floors:
        console.print("Floors:")
        for axis, value in sorted(view.floors.items()):
            console.print(f"  {axis.title():10} {value}")
    if view.targets or view.floors:
        console.print(
            "Scale binding: "
            + (
                f"{view.scale_id} {view.scale_version} · {view.scale_hash}"
                if view.scale_bound
                else "UNBOUND"
            )
        )
    if view.reason:
        console.print(view.reason)


def _print_stats(view: StatsView) -> None:
    console.print("[bold]STATS[/bold]")
    if view.model_id:
        console.print(f"Model: {view.model_id}")
    if view.scale_id:
        console.print(f"Scale: {view.scale_id} {view.scale_version}")
        console.print(f"Scale hash: {view.scale_hash}")
    if view.receipt_id:
        console.print(f"Receipt: {view.receipt_id}")
        console.print(f"Receipt hash: {view.receipt_sha256}")
    if view.evaluator_id:
        console.print(f"Evaluator: {view.evaluator_id} {view.evaluator_version}")
    console.print("")
    for axis in ("general", "reasoning", "math", "coding"):
        value = view.stats.get(axis)
        console.print(f"{axis.title():10} {value if value is not None else '?'}")
    if view.reason:
        console.print("")
        console.print(view.reason)
    if view.raw_measurements:
        console.print("")
        console.print("RAW MEASUREMENTS")
        for item in view.raw_measurements:
            console.print(
                f"{item.get('task_id')}@{item.get('task_version')} "
                f"{item.get('metric')}={item.get('value')}"
            )


def _print_evaluation(view: EvaluationRunView) -> None:
    console.print("[bold]EVALUATION[/bold]")
    console.print(f"Pack: {view.pack_id}@{view.pack_version}")
    console.print(f"Model: {view.model_id}")
    console.print(f"Dataset: {view.dataset_id}")
    console.print(f"Receipt: {view.receipt_id}")
    console.print(f"Receipt hash: {view.receipt_sha256}")
    console.print(f"Replay: {'YES' if view.replayed else 'NO'}")
    console.print("")
    for item in view.measurements:
        console.print(
            f"{item.get('task_id')}@{item.get('task_version')} "
            f"{item.get('metric')}={item.get('value')}"
        )


def _print_evaluation_compare(view: EvaluationCompareView) -> None:
    console.print("[bold]RAW EVALUATION COMPARE[/bold]")
    console.print(f"Pack: {view.pack_id}@{view.pack_version}")
    console.print(f"Dataset: {view.dataset_id}")
    console.print(f"Champion: {view.champion_model_id}")
    console.print(f"Candidate: {view.candidate_model_id} [{view.candidate_status or 'UNKNOWN'}]")
    console.print(f"Comparable: {'YES' if view.comparable else 'NO'}")
    if view.reason:
        console.print(view.reason)
    if view.measurements:
        console.print("")
        for item in view.measurements:
            console.print(
                f"{item.get('metric')}: champion={item.get('champion_value')} "
                f"candidate={item.get('candidate_value')} "
                f"raw_delta={item.get('raw_delta')} "
                f"improvement_delta={item.get('improvement_delta')}"
            )


def _print_data(view: DataView) -> None:
    console.print("[bold]DATA[/bold]")
    if not view.datasets:
        console.print("No datasets registered.")
        return
    for item in view.datasets:
        console.print(
            f"{item.get('name')} · {item.get('role')} · {_human_bytes(item.get('total_bytes'))}"
        )
        console.print(f"  Fingerprint: {item.get('fingerprint')}")
        console.print(f"  Provenance: {item.get('provenance')}")
        console.print(f"  Classification: {item.get('classification') or 'UNKNOWN'}")
        console.print(f"  License: {item.get('license') or 'UNKNOWN'}")
        console.print(f"  Domain: {item.get('domain') or 'UNKNOWN'}")
        console.print(f"  Language: {item.get('language') or 'UNKNOWN'}")
        if item.get("managed"):
            console.print(f"  Managed: YES · source={item.get('source_dataset_id')}")
            console.print(
                f"  Recipe: {item.get('preparation_recipe_id')} "
                f"({item.get('preparation_recipe_hash')})"
            )
        token_count = item.get("token_count")
        console.print(f"  Tokens: {token_count if isinstance(token_count, int) else 'UNKNOWN'}")


def _print_paths(view: PathsView) -> None:
    console.print("[bold]TRAINING PATHS[/bold]")
    for item in view.paths:
        console.print(f"[{item.get('availability')}] {item.get('title')} ({item.get('path_id')})")
        blockers = item.get("blockers")
        if isinstance(blockers, list):
            for blocker in blockers:
                console.print(f"  Missing: {blocker}")
        checks = item.get("next_checks")
        if isinstance(checks, list):
            for check in checks:
                console.print(f"  Next check: {check}")
    if view.recommendation_reason:
        console.print("")
        console.print(view.recommendation_reason)


def _print_lab_adapters(view: LabAdaptersView) -> None:
    if not view.adapters:
        console.print("No connected Lab adapters.")
        return
    console.print("[bold]LAB ADAPTERS[/bold]")
    for item in view.adapters:
        console.print(
            f"{item.get('adapter_ref')} · {item.get('display_name')} · "
            f"{item.get('data_boundary')} · {item.get('network_scope')}"
        )
        kinds = item.get("kinds")
        kind_values = kinds if isinstance(kinds, list) else []
        console.print(f"  Kinds: {', '.join(str(value) for value in kind_values)}")
        console.print(f"  Manifest: {item.get('manifest_hash')}")


def _print_preflight(view: ActionPreflightView) -> None:
    console.print(f"[bold]DRY RUN[/bold] · {view.action}")
    console.print(f"Ready: {'YES' if view.ready else 'NO'}")
    console.print(f"Would replay: {'YES' if view.would_replay else 'NO'}")
    for key, value in sorted(view.details.items()):
        console.print(f"{key}: {value}")
    for blocker in view.blockers:
        console.print(f"Blocked: {blocker}")


def _print_plan(view: PlanView) -> None:
    console.print(f"[bold]PLAN[/bold] · {view.plan_id}")
    console.print(f"Path: {view.path_id}")
    console.print(
        f"Intervention: {view.intervention_id}@{view.intervention_version} "
        f"· {view.intervention_family}"
    )
    console.print(f"Backend: {view.backend_id}")
    console.print(f"Permission: {view.permission}")
    console.print(f"Dataset: {view.dataset_id}")
    console.print(
        f"Data policy: {view.dataset_classification or 'UNKNOWN'} -> "
        f"{view.backend_data_boundary or 'UNKNOWN'}"
    )
    if view.backend_adapter_ref:
        console.print(
            f"Lab adapter: {view.backend_adapter_ref} · {view.backend_adapter_hash or 'UNPINNED'}"
        )
    console.print(f"Resource profile: {view.resource_profile_id or 'UNPINNED'}")
    if view.budget_enforcement:
        console.print("Budget enforcement:")
        for key, value in sorted(view.budget_enforcement.items()):
            console.print(f"  {key}: {value}")
    console.print(f"Ready: {'YES' if view.ready else 'NO'}")
    if view.calibration:
        console.print("Calibration:")
        for key in (
            "calibration_id",
            "feasible",
            "step_time_seconds",
            "tokens_per_second",
            "peak_vram_bytes",
            "peak_ram_bytes",
            "projected_wall_seconds",
            "projected_storage_bytes",
        ):
            if key in view.calibration:
                console.print(f"  {key}: {view.calibration[key]}")
    for blocker in view.blockers:
        console.print(f"  Blocked: {blocker}")


def _print_run(view: RunView) -> None:
    console.print(f"[bold]RUN[/bold] · {view.run_id or 'DRY-RUN'}")
    console.print(f"Plan: {view.plan_id}")
    console.print(f"Status: {view.status}")
    if view.liveness_state:
        console.print(f"Liveness: {view.liveness_state}")
    if view.calibration_id:
        console.print(f"Calibration: {view.calibration_id}")
    if view.result_evidence_available:
        console.print("Durable result evidence: YES")
    if view.candidate_model_id:
        console.print(f"Candidate: {view.candidate_model_id}")
    if view.metrics:
        console.print("Metrics:")
        for key, value in sorted(view.metrics.items()):
            console.print(f"  {key}: {value}")
    if view.usage:
        console.print("Usage:")
        for key, value in sorted(view.usage.items()):
            console.print(f"  {key}: {value}")
    if view.error_code:
        console.print(f"Error: {view.error_code}: {view.error_message}")


def _print_candidates(view: CandidateView) -> None:
    console.print("[bold]CANDIDATES[/bold]")
    if not view.candidates:
        console.print("No candidates.")
        return
    for item in view.candidates:
        console.print(
            f"{item.get('model_id')} · {item.get('status')} · "
            f"{item.get('path_id') or 'unknown path'}"
        )
        stats = item.get("stats")
        if isinstance(stats, dict):
            rendered = " ".join(
                f"{axis[0].upper()}{value if value is not None else '?'}"
                for axis, value in stats.items()
            )
            console.print(f"  {rendered}")


def _print_merge(view: MergeView) -> None:
    console.print("[bold]MODEL MERGE[/bold]")
    console.print(f"Transform: {view.transform_id}")
    console.print(f"Primary: {view.primary_model_id} · weight={view.primary_weight}")
    console.print(f"Other: {view.other_model_id} · weight={view.other_weight}")
    console.print(f"Candidate: {view.candidate_model_id}")
    console.print(f"Fingerprint: {view.model_fingerprint}")
    console.print(f"Replay: {'YES' if view.replayed else 'NO'}")
    if view.checkpoint:
        console.print(f"Checkpoint: {view.checkpoint}")


def _print_quantize(view: QuantizeView) -> None:
    console.print("[bold]INT8 QUANTIZATION[/bold]")
    console.print(f"Transform: {view.transform_id}")
    console.print(f"Source: {view.source_model_id}")
    console.print(f"Candidate: {view.candidate_model_id}")
    console.print(f"Format: {view.model_format}")
    console.print(f"Trainable: {'YES' if view.trainable else 'NO'}")
    console.print(f"Fingerprint: {view.model_fingerprint}")
    console.print(f"Replay: {'YES' if view.replayed else 'NO'}")
    ratio = view.metrics.get("tensor_storage_ratio")
    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
        console.print(f"Tensor storage ratio: {float(ratio):.3f}")
    if view.checkpoint:
        console.print(f"Checkpoint: {view.checkpoint}")


def _print_export(view: ExportView) -> None:
    console.print("[bold]PORTABLE EXPORT[/bold]")
    console.print(f"Export: {view.export_id}")
    console.print(f"Model: {view.model_id}")
    console.print(f"Format: {view.model_format}")
    console.print(f"Trainable: {'YES' if view.trainable else 'NO'}")
    console.print(f"Fingerprint: {view.model_fingerprint}")
    console.print(f"Replay: {'YES' if view.replayed else 'NO'}")
    if view.destination:
        console.print(f"Destination: {view.destination}")
    if view.manifest_sha256:
        console.print(f"Manifest: {view.manifest_sha256}")


def _print_export_verify(view: ExportVerifyView) -> None:
    console.print("[bold]PORTABLE EXPORT VERIFY[/bold]")
    console.print(f"Valid: {'YES' if view.valid else 'NO'}")
    console.print(f"Export: {view.export_id}")
    console.print(f"Format: {view.model_format}")
    console.print(f"Trainable: {'YES' if view.trainable else 'NO'}")
    console.print(f"Fingerprint: {view.model_fingerprint}")
    console.print(f"Authenticity: {view.authenticity}")
    if view.manifest_sha256:
        console.print(f"Manifest: {view.manifest_sha256}")


def _print_generation(view: GenerationView) -> None:
    console.print("[bold]REFERENCE GENERATION[/bold]")
    console.print(f"Model: {view.model_id} · {view.model_format}")
    console.print(f"Fingerprint: {view.model_fingerprint}")
    console.print("")
    console.print(view.generated_text)


def _print_inference_profile(view: InferenceProfileView) -> None:
    console.print("[bold]REFERENCE INFERENCE PROFILE[/bold]")
    console.print(f"Model: {view.model_id} · {view.model_format}")
    console.print(f"Fingerprint: {view.model_fingerprint}")
    metrics = view.metrics
    console.print(f"Device: {metrics.get('device')}")
    latency = metrics.get("latency_seconds_p50")
    throughput = metrics.get("tokens_per_second_p50")
    if isinstance(latency, (int, float)) and not isinstance(latency, bool):
        console.print(f"Latency p50: {float(latency):.6f}s")
    if isinstance(throughput, (int, float)) and not isinstance(throughput, bool):
        console.print(f"Throughput p50: {float(throughput):.3f} tokens/s")
    console.print(f"Peak VRAM: {metrics.get('peak_vram_bytes')}")
    console.print(
        "Minimum sampled free VRAM: "
        f"{metrics.get('cuda_memory_free_min_sampled_bytes')} / "
        f"{metrics.get('cuda_memory_total_bytes')}"
    )
    console.print(f"Process RSS: {metrics.get('max_sampled_process_rss_bytes')}")


def _print_compare(view: CompareView) -> None:
    console.print("[bold]COMPARE[/bold]")
    console.print(f"Champion: {view.champion_model_id}")
    console.print(f"Candidate: {view.candidate_model_id} · {view.candidate_status}")
    console.print(f"Comparable: {'YES' if view.scale_comparable else 'NO'}")
    if view.scale_reason:
        console.print(view.scale_reason)
    console.print("")
    console.print("AXIS        CHAMPION   CANDIDATE   DELTA")
    for axis in ("general", "reasoning", "math", "coding"):
        champion = view.champion_stats.get(axis)
        candidate = view.candidate_stats.get(axis)
        delta = view.deltas.get(axis)
        delta_text = "?" if delta is None else f"{delta:+g}"
        console.print(
            f"{axis.title():10} "
            f"{champion if champion is not None else '?':>8}   "
            f"{candidate if candidate is not None else '?':>9}   "
            f"{delta_text:>5}"
        )
    if view.raw_evaluation_comparisons:
        console.print("")
        console.print("RAW EVALUATION")
        for evidence in view.raw_evaluation_comparisons:
            if evidence.get("kind") == "EXTERNAL_LM_EVAL":
                console.print(
                    f"lm-eval {evidence.get('evaluator_version')} · "
                    f"tasks={evidence.get('task_count')} · external raw evidence"
                )
            else:
                console.print(
                    f"{evidence.get('pack_id')}@{evidence.get('pack_version')} · "
                    f"dataset={evidence.get('dataset_id')}"
                )
            measurements = evidence.get("measurements")
            if isinstance(measurements, list):
                for item in measurements:
                    if not isinstance(item, dict):
                        continue
                    improvement = item.get("improvement_delta")
                    console.print(
                        f"  {item.get('task_id')} / {item.get('metric')}: "
                        f"{item.get('champion_value')} -> {item.get('candidate_value')} "
                        f"(improvement {improvement})"
                    )

    if view.build_constraints:
        console.print("")
        console.print("BUILD CONSTRAINTS")
        for item in view.build_constraints:
            console.print(
                f"{item.get('axis')} {item.get('kind')} "
                f"{item.get('threshold')} -> {item.get('status')}"
            )

    if view.workload_comparison.get("configured"):
        champion_fit = view.workload_comparison.get("champion")
        candidate_fit = view.workload_comparison.get("candidate")
        console.print("")
        console.print("WORKLOAD FIT")
        if isinstance(champion_fit, dict) and isinstance(candidate_fit, dict):
            console.print(
                f"Champion {champion_fit.get('overall_status')} -> "
                f"Candidate {candidate_fit.get('overall_status')}"
            )
        regressions = view.workload_comparison.get("regressions")
        if isinstance(regressions, list):
            for item in regressions:
                if isinstance(item, dict):
                    console.print(
                        f"  Regression: {item.get('key')} "
                        f"{item.get('champion_status')} -> {item.get('candidate_status')}"
                    )

    if view.pareto:
        console.print("")
        console.print(f"EVIDENCE PARETO: {view.pareto.get('relation')}")
        metrics = view.pareto.get("metrics")
        if isinstance(metrics, list):
            for item in metrics:
                if not isinstance(item, dict) or item.get("relation") == "UNKNOWN":
                    continue
                console.print(
                    f"  {item.get('key')}: {item.get('champion_value')} -> "
                    f"{item.get('candidate_value')} · {item.get('relation')}"
                )
        reason = view.pareto.get("inference_profile_reason")
        if reason:
            console.print(f"  Runtime evidence: {reason}")
        utility = view.pareto.get("explicit_user_utility")
        if isinstance(utility, dict):
            status = utility.get("status")
            if status == "COMPLETE":
                console.print(
                    f"  USER UTILITY: {utility.get('relation')} · "
                    f"delta={utility.get('utility_delta')}"
                )
                contributions = utility.get("contributions")
                if isinstance(contributions, list):
                    for item in contributions:
                        if isinstance(item, dict):
                            console.print(
                                f"    {item.get('metric_key')}: "
                                f"{item.get('contribution')} "
                                f"(weight={item.get('weight')}, scale={item.get('scale')})"
                            )
            elif status == "INCOMPLETE":
                missing = utility.get("missing_metrics")
                missing_text = (
                    ", ".join(str(item) for item in missing)
                    if isinstance(missing, list)
                    else "unknown"
                )
                console.print(f"  USER UTILITY: INCOMPLETE · missing {missing_text}")

    console.print("")
    console.print(f"Promotion eligible: {'YES' if view.promotion_eligible else 'NO'}")
    for blocker in view.promotion_blockers:
        override = blocker.get("override")
        suffix = f" · override {override}" if override else ""
        console.print(f"  Blocked: {blocker.get('code')} · {blocker.get('message')}{suffix}")


def _print_history(view: HistoryView) -> None:
    console.print("[bold]DEVELOPMENT HISTORY[/bold]")
    if not view.events:
        console.print("No history events.")
        return
    for event in view.events:
        console.print(
            f"{event.get('sequence'):>4}  {event.get('kind')}  {event.get('recorded_at')}"
        )
        details = event.get("details")
        if isinstance(details, dict) and details:
            console.print(f"      {json.dumps(details, ensure_ascii=False, sort_keys=True)}")


def _print_status(view: StatusView) -> None:
    console.print(f"[bold]{view.nickname}[/bold]")
    if view.edition_name:
        console.print(f"Edition: {view.edition_name}")
        if view.edition_tagline:
            console.print(view.edition_tagline)
    console.print(f"Origin: {view.origin}")
    console.print(f"History: {view.history_confidence}")
    if view.history_evidence_reason:
        console.print(f"History evidence: {view.history_evidence_reason}")
    console.print(f"State: {view.measurement_state}")
    console.print(f"Build mode: {view.build_mode}")
    if view.champion_model_id:
        console.print(f"Model: {view.champion_model_id}")
        console.print(f"Format: {view.model_format}")
        console.print(f"Trainable: {'YES' if view.trainable else 'NO'}")
        console.print(f"Fingerprint: {view.model_fingerprint}")
    for axis, value in view.stats.items():
        console.print(f"{axis.title():10} {value if value is not None else '?'}")


def _print_tokenizer(view: TokenizerView) -> None:
    console.print("[bold]TRAINABLE TOKENIZER[/bold]")
    console.print(f"Artifact: {view.artifact_id}")
    console.print(f"Fingerprint: {view.fingerprint}")
    console.print(f"Source dataset: {view.source_dataset_id}")
    console.print(f"Vocabulary: {view.vocab_size} / requested {view.requested_vocab_size}")
    console.print(f"Merges: {view.merge_count}")
    console.print(f"Training bytes: {view.training_bytes}")
    console.print(f"Replay: {'YES' if view.replayed else 'NO'}")
    if view.path:
        console.print(f"Artifact: {view.path}")


def _print_tokenizers(view: TokenizersView) -> None:
    if not view.tokenizers:
        console.print("No trained tokenizer artifacts.")
        return
    console.print("[bold]TRAINABLE TOKENIZERS[/bold]")
    for item in view.tokenizers:
        console.print(
            f"{item.get('artifact_id')} · vocab={item.get('vocab_size')} · "
            f"source={item.get('source_dataset_id')}"
        )
        console.print(f"  Fingerprint: {item.get('fingerprint')}")


def _print_birth(view: BirthView) -> None:
    if not view.born:
        console.print("No materialized zero-model root.")
        return
    console.print("[bold]MODEL BIRTH[/bold]")
    console.print(f"Preset: {view.preset}")
    console.print(f"Seed: {view.seed}")
    parameter_count = view.parameter_count if view.parameter_count is not None else "?"
    console.print(f"Parameters: {parameter_count}")
    if view.vocab_size is not None:
        console.print(f"Vocabulary: {view.vocab_size}")
    if view.tokenizer_artifact_id is not None:
        console.print(f"Tokenizer artifact: {view.tokenizer_artifact_id}")
        console.print(f"Tokenizer fingerprint: {view.tokenizer_fingerprint}")
    else:
        console.print("Tokenizer: built-in byte vocabulary")
    console.print(f"Model: {view.model_id}")
    console.print(f"Fingerprint: {view.model_fingerprint}")
    console.print(f"Checkpoint: {view.checkpoint}")
    console.print("Training steps: 0")


def _human_bytes(value: object) -> str:
    if not isinstance(value, int):
        return "UNKNOWN"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _print_resources(view: ResourceView) -> None:
    if not view.available:
        console.print("No detected resource profile. Run: frontierwright resources detect")
        return
    snapshot = view.snapshot
    console.print(f"[bold]RESOURCES[/bold] · {view.provenance}")
    console.print(f"CPU: {snapshot.get('cpu_model', 'UNKNOWN')}")
    console.print(f"Logical CPUs: {snapshot.get('cpu_logical_count', 'UNKNOWN')}")
    console.print(
        f"RAM: {_human_bytes(snapshot.get('ram_total_bytes'))} total · "
        f"{_human_bytes(snapshot.get('ram_available_bytes'))} available"
    )
    console.print(
        f"Disk: {_human_bytes(snapshot.get('disk_free_bytes'))} free / "
        f"{_human_bytes(snapshot.get('disk_total_bytes'))} total"
    )
    gpus = snapshot.get("gpus", [])
    if isinstance(gpus, list) and gpus:
        for index, gpu in enumerate(gpus, start=1):
            if not isinstance(gpu, dict):
                continue
            console.print(
                f"GPU {index}: {gpu.get('vendor', '?')} {gpu.get('name', '?')} · "
                f"{_human_bytes(gpu.get('memory_total_bytes'))} VRAM"
            )
    else:
        console.print("GPU: none detected")
    console.print(f"Torch: {snapshot.get('torch_version') or 'not detected'}")
    console.print(f"CUDA toolkit: {snapshot.get('cuda_toolkit_version') or 'not detected'}")
    console.print(f"ROCm: {snapshot.get('rocm_version') or 'not detected'}")
    console.print("bf16/fp16: UNKNOWN until backend-specific capability calibration")


@project_app.command("init")
def project_init(
    path: Annotated[Path, typer.Argument(help="Project directory.")] = Path("."),
    name: Annotated[str, typer.Option("--name", help="Project/character name.")] = "My Model",
    origin: Annotated[
        ModelOrigin,
        typer.Option(
            "--origin",
            case_sensitive=False,
            help="ZERO, IMPORTED_LOCAL, or INTERNAL_LAB.",
        ),
    ] = ModelOrigin.ZERO,
    language: Annotated[
        str,
        typer.Option("--lang", help="Human UI language: en or ko."),
    ] = "en",
    edition: Annotated[
        EditionProfile | None,
        typer.Option(
            "--edition",
            case_sensitive=False,
            help="STUDIO, ACADEMY, or LAB. Defaults from origin.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Stable machine-readable JSON."),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Never prompt for missing values."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Pre-authorize non-destructive confirmations."),
    ] = False,
) -> None:
    del non_interactive, yes
    try:
        view = initialize_project(
            path,
            name=name,
            origin=origin,
            language=language,
            edition_profile=edition,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_status_payload(view))
        return

    console.print(f"[bold]Frontierwright[/bold] initialized: {view.project_name}")
    _print_status(view)


@project_app.command("edition")
def project_edition(
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    set_to: Annotated[
        EditionProfile | None,
        typer.Option(
            "--set",
            case_sensitive=False,
            help="Change profile to STUDIO, ACADEMY, or LAB.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Stable machine-readable JSON."),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Never prompt."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Pre-authorize non-destructive confirmations."),
    ] = False,
) -> None:
    del non_interactive, yes
    try:
        view = set_project_edition(path, set_to) if set_to is not None else get_status(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_status_payload(view))
        return

    if view.edition_name:
        console.print(f"[bold]{view.edition_name}[/bold]")
    if view.edition_tagline:
        console.print(view.edition_tagline)
    if view.edition_starting_point:
        console.print(f"Starting point: {view.edition_starting_point}")


@lab_adapters_app.command("list")
def lab_adapters_list(
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_lab_adapters(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_lab_adapters_payload(view))
        return
    _print_lab_adapters(view)


@lab_adapters_app.command("connect")
def lab_adapters_connect(
    manifest: Annotated[Path, typer.Argument(help="Lab adapter manifest JSON.")],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = connect_lab_adapter(path, manifest)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_lab_adapters_payload(view))
        return
    _print_lab_adapters(view)


@lab_adapters_app.command("disconnect")
def lab_adapters_disconnect(
    adapter_ref: Annotated[str, typer.Argument(help="Adapter ref, e.g. id@version.")],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = disconnect_lab_adapter(path, adapter_ref)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_lab_adapters_payload(view))
        return
    _print_lab_adapters(view)


@birth_app.command("show")
def birth_show(
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_birth_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_birth_payload(view))
        return
    _print_birth(view)


@birth_app.command("tokenizer")
def birth_tokenizer(
    dataset_id: Annotated[
        str,
        typer.Argument(help="Registered PRETRAIN dataset ID used to learn the tokenizer."),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    vocab_size: Annotated[
        int,
        typer.Option(
            "--vocab-size",
            min=256,
            max=65536,
            help="Requested byte-BPE vocabulary size.",
        ),
    ] = 512,
    max_training_bytes: Annotated[
        int,
        typer.Option(
            "--max-training-bytes",
            min=1,
            help="Maximum source bytes used to learn merge rules.",
        ),
    ] = DEFAULT_MAX_TRAINING_BYTES,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate tokenizer birth and show replay identity without training it.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        if dry_run:
            preview = preflight_project_tokenizer(
                path,
                dataset_id=dataset_id,
                vocab_size=vocab_size,
                max_training_bytes=max_training_bytes,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return

        view = train_project_tokenizer(
            path,
            dataset_id=dataset_id,
            vocab_size=vocab_size,
            max_training_bytes=max_training_bytes,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_tokenizer_payload(view))
        return
    _print_tokenizer(view)


@birth_app.command("tokenizers")
def birth_tokenizers(
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_tokenizers_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_tokenizers_payload(view))
        return
    _print_tokenizers(view)


@birth_app.command("zero")
def birth_zero(
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    preset: Annotated[
        str,
        typer.Option("--preset", help="Built-in zero-model preset: zero-8m or zero-25m."),
    ] = "zero-8m",
    seed: Annotated[int, typer.Option("--seed", help="Initialization seed.")] = 42,
    tokenizer_artifact_id: Annotated[
        str | None,
        typer.Option(
            "--tokenizer-artifact",
            help=(
                "Managed tokenizer artifact ID to bind into the root model. "
                "Omit to use the built-in 256-byte vocabulary."
            ),
        ),
    ] = None,
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated PyTorch training environment.",
        ),
    ] = sys.executable,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", help="Maximum birth backend wall time in seconds."),
    ] = 300.0,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Validate zero-model birth and show replay identity without materializing weights."
            ),
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        if dry_run:
            preview = preflight_zero_birth(
                path,
                preset=preset,
                seed=seed,
                python_executable=python_executable,
                tokenizer_artifact_id=tokenizer_artifact_id,
                timeout_seconds=timeout_seconds,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return

        view = birth_zero_model(
            path,
            preset=preset,
            seed=seed,
            python_executable=python_executable,
            tokenizer_artifact_id=tokenizer_artifact_id,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_birth_payload(view))
        return
    _print_birth(view)


@evolve_app.command("merge")
def evolve_merge(
    other_model_id: Annotated[
        str,
        typer.Argument(help="Second registered model to merge with the current champion."),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    other_weight: Annotated[
        float,
        typer.Option(
            "--other-weight",
            min=0.0,
            max=1.0,
            help="Weight assigned to the second model; champion receives 1-weight.",
        ),
    ] = 0.5,
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated PyTorch transform environment.",
        ),
    ] = sys.executable,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", help="Maximum merge backend wall time in seconds."),
    ] = 300.0,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate merge identity without materializing it."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        if dry_run:
            preview = preflight_merge_reference_models(
                path,
                other_model_id=other_model_id,
                other_weight=other_weight,
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return
        view = merge_reference_models(
            path,
            other_model_id=other_model_id,
            other_weight=other_weight,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_merge_payload(view))
        return
    _print_merge(view)
    console.print("\nMerge created a PENDING candidate. Evaluate and compare it before promotion.")


@optimize_app.command("quantize")
def optimize_quantize(
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated PyTorch transform environment.",
        ),
    ] = sys.executable,
    timeout_seconds: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Maximum quantization backend wall time in seconds.",
        ),
    ] = 300.0,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate quantization identity without materializing it."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        if dry_run:
            preview = preflight_quantize_reference_model(
                path,
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return
        view = quantize_reference_model(
            path,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_quantize_payload(view))
        return
    _print_quantize(view)
    console.print(
        "\nQuantization created a PENDING optimized-model candidate. "
        "Evaluate and compare it before promotion."
    )


@operate_app.command("export")
def operate_export(
    destination: Annotated[
        Path,
        typer.Argument(help="New directory for the portable Frontierwright export bundle."),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate export identity and destination without copying."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        if dry_run:
            preview = preflight_export_champion_bundle(path, destination)
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return
        view = export_champion_bundle(path, destination)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_export_payload(view))
        return
    _print_export(view)


@operate_app.command("verify")
def operate_verify(
    destination: Annotated[
        Path,
        typer.Argument(help="Portable Frontierwright export bundle directory."),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = verify_export_bundle(destination)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_export_verify_payload(view))
        return
    _print_export_verify(view)


@operate_app.command("generate")
def operate_generate(
    prompt: Annotated[str, typer.Argument(help="Prompt text for the current Champion.")],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated PyTorch inference environment.",
        ),
    ] = sys.executable,
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", min=1, help="Number of byte tokens to generate."),
    ] = 64,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            min=0.0,
            help="0 for greedy decoding; positive values enable sampling.",
        ),
    ] = 0.0,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    device: Annotated[
        str,
        typer.Option("--device", help="auto, cpu, or cuda."),
    ] = "auto",
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", help="Maximum generation backend wall time in seconds."),
    ] = 60.0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = generate_reference_text(
            path,
            prompt=prompt,
            python_executable=python_executable,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            device=device,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_generation_payload(view))
        return
    _print_generation(view)


@operate_app.command("import-vllm-benchmark")
def operate_import_vllm_benchmark(
    result: Annotated[
        Path,
        typer.Argument(help="vLLM bench serve result JSON produced with --save-result."),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Exact model ID; defaults to current Champion."),
    ] = None,
    vllm_version: Annotated[
        str,
        typer.Option(
            "--vllm-version",
            help="Exact vLLM version or immutable revision used for the benchmark.",
        ),
    ] = "",
    execution_boundary: Annotated[
        str,
        typer.Option(
            "--execution-boundary",
            help="LOCAL_MACHINE, CONTROLLED_PRIVATE, or EXTERNAL.",
        ),
    ] = BackendDataBoundary.LOCAL_MACHINE.value,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        try:
            boundary = BackendDataBoundary(execution_boundary.upper())
        except ValueError as exc:
            raise FrontierwrightError(
                "SERVING_BOUNDARY_INVALID",
                "execution boundary must be LOCAL_MACHINE, CONTROLLED_PRIVATE, or EXTERNAL.",
                2,
            ) from exc
        if boundary is BackendDataBoundary.UNKNOWN:
            raise FrontierwrightError(
                "SERVING_BOUNDARY_INVALID",
                "UNKNOWN is not allowed for imported serving evidence.",
                2,
            )
        view = import_vllm_serving_evidence(
            path,
            result_path=result,
            model_id=model,
            vllm_version=vllm_version,
            execution_boundary=boundary,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    payload = {"ok": True, **view.to_dict()}
    if json_output:
        _emit_json(payload)
        return
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    console.print("[bold]VLLM SERVING EVIDENCE IMPORTED[/bold]")
    console.print(f"vLLM: {payload['vllm_version']}")
    console.print(f"Condition hash: {payload['profile_condition_hash']}")
    console.print(
        f"E2E p50: {metrics.get('latency_seconds_p50')}s · "
        f"TTFT p50: {metrics.get('ttft_seconds_p50')}s · "
        f"TPOT p50: {metrics.get('tpot_seconds_p50')}s/token"
    )
    console.print(
        "VRAM/RSS: UNKNOWN from client benchmark; import or run separate resource "
        "measurement before making memory-fit claims."
    )


@operate_app.command("profile")
def operate_profile(
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated PyTorch inference environment.",
        ),
    ] = sys.executable,
    model_id: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Exact model ID to profile; defaults to the current Champion.",
        ),
    ] = None,
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", min=1, help="Generated tokens per measured run."),
    ] = 16,
    warmup_runs: Annotated[
        int,
        typer.Option("--warmup-runs", min=1, help="Unmeasured warmup generations."),
    ] = 1,
    measured_runs: Annotated[
        int,
        typer.Option("--runs", min=1, help="Measured generation runs."),
    ] = 3,
    device: Annotated[
        str,
        typer.Option("--device", help="auto, cpu, or cuda."),
    ] = "auto",
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", help="Maximum profiling backend wall time in seconds."),
    ] = 120.0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = profile_reference_inference(
            path,
            python_executable=python_executable,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            device=device,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_inference_profile_payload(view))
        return
    _print_inference_profile(view)


@app.command("import")
def import_model(
    source: Annotated[Path, typer.Argument(help="Local model directory or GGUF file.")],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    origin: Annotated[
        ModelOrigin,
        typer.Option(
            "--origin",
            case_sensitive=False,
            help="IMPORTED_LOCAL or INTERNAL_LAB.",
        ),
    ] = ModelOrigin.IMPORTED_LOCAL,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Project/character name when creating a project."),
    ] = None,
    history_manifest: Annotated[
        Path | None,
        typer.Option(
            "--history-manifest",
            help="Optional frontierwright-lineage.json with verifiable evidence hashes.",
        ),
    ] = None,
    edition: Annotated[
        EditionProfile | None,
        typer.Option(
            "--edition",
            case_sensitive=False,
            help="STUDIO, ACADEMY, or LAB when creating the project.",
        ),
    ] = None,
    language: Annotated[str, typer.Option("--lang")] = "en",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = import_local_model(
            path,
            source,
            origin=origin,
            project_name=name,
            language=language,
            history_manifest=history_manifest,
            edition_profile=edition,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_status_payload(view))
        return

    console.print("[bold green]Model imported[/bold green]")
    _print_status(view)
    if not view.strong_recommendation_allowed:
        console.print(
            "[yellow]History-aware next-path recommendation is withheld until "
            "history becomes VERIFIED or COMPLETE.[/yellow]"
        )


@app.command("status")
def status(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Stable machine-readable JSON."),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Never prompt."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Pre-authorize non-destructive confirmations."),
    ] = False,
) -> None:
    del non_interactive, yes
    try:
        view = get_status(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if not view.initialized:
        error = FrontierwrightError(
            "NOT_INITIALIZED",
            "No Frontierwright project is initialized here.",
            10,
        )
        _fail(error, json_output=json_output)

    if json_output:
        _emit_json(_status_payload(view))
        return

    _print_status(view)


@resources_app.command("detect")
def resources_detect(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = detect_resources(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_resource_payload(view))
        return
    _print_resources(view)


@resources_app.command("show")
def resources_show(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_resource_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_resource_payload(view))
        return
    _print_resources(view)


@workload_app.command("show")
def workload_show(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_workload_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)
    if json_output:
        _emit_json(_workload_payload(view))
        return
    _print_workload(view)


@workload_app.command("fit")
def workload_fit(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model ID; defaults to current Champion."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_workload_fit(path, model)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)
    if json_output:
        _emit_json(_workload_fit_payload(view))
        return
    _print_workload_fit(view)


@workload_app.command("bind-eval")
def workload_bind_eval(
    manifest: Annotated[
        Path,
        typer.Argument(
            help=(
                "Workload-evaluation binding manifest JSON. It must reference exact stored "
                "receipt task/version/metric identities; task names are never inferred."
            )
        ),
    ],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Exact model ID; defaults to current Champion."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        binding = bind_workload_evaluation(
            path,
            manifest_path=manifest,
            model_id=model,
        )
        fit = get_workload_fit(path, model)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    payload = {
        "ok": True,
        **binding.to_payload(),
        "workload_evaluation_coverage": fit.workload_evaluation_coverage,
        "workload_fit_status": fit.overall_status,
    }
    if json_output:
        _emit_json(payload)
        return
    console.print("[bold]WORKLOAD EVALUATION EVIDENCE BOUND[/bold]")
    console.print(f"Model: {binding.model_id}")
    console.print(f"Receipt: {binding.receipt_id}")
    console.print(f"Evaluator: {binding.evaluator_id}@{binding.evaluator_version}")
    console.print(f"Binding: {binding.binding_id}")
    coverage = fit.workload_evaluation_coverage
    console.print("Coverage: " + ("COMPLETE" if coverage.get("complete") is True else "INCOMPLETE"))
    missing = coverage.get("missing")
    if isinstance(missing, dict):
        for kind in ("languages", "domains", "tasks"):
            values = missing.get(kind)
            if isinstance(values, list) and values:
                console.print(f"Missing {kind}: " + ", ".join(str(x) for x in values))


@workload_app.command("set")
def workload_set(
    name: Annotated[str, typer.Option("--name", help="Workload profile name.")],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    language: Annotated[
        list[str] | None,
        typer.Option("--language", help="Repeat for workload languages."),
    ] = None,
    domain: Annotated[
        list[str] | None,
        typer.Option("--domain", help="Repeat for workload domains."),
    ] = None,
    task: Annotated[
        list[str] | None,
        typer.Option("--task", help="Repeat task=weight, e.g. --task coding=2."),
    ] = None,
    context_p50: Annotated[int | None, typer.Option("--context-p50")] = None,
    context_p95: Annotated[int | None, typer.Option("--context-p95")] = None,
    max_latency: Annotated[float | None, typer.Option("--max-latency")] = None,
    min_tokens_per_second: Annotated[float | None, typer.Option("--min-tokens-per-second")] = None,
    privacy: Annotated[str, typer.Option("--privacy")] = "PRIVATE",
    floor: Annotated[
        list[str] | None,
        typer.Option("--floor", help="Repeat capability-axis=value hard floor."),
    ] = None,
    utility_weight: Annotated[
        list[str] | None,
        typer.Option(
            "--utility-weight",
            help="Repeat measured-metric=weight for explicit user utility.",
        ),
    ] = None,
    utility_scale: Annotated[
        list[str] | None,
        typer.Option(
            "--utility-scale",
            help="Repeat the same measured-metric=normalization-scale; required with weights.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        try:
            privacy_value = DatasetClassification(privacy.strip().upper())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in DatasetClassification)
            raise FrontierwrightError(
                "INVALID_WORKLOAD_ARGUMENT",
                f"privacy must be one of: {allowed}",
                2,
            ) from exc
        profile = WorkloadProfile(
            name=name,
            languages=tuple(language or ()),
            domains=tuple(domain or ()),
            task_weights=_parse_float_assignments(task or [], "task"),
            context_tokens_p50=context_p50,
            context_tokens_p95=context_p95,
            max_latency_seconds=max_latency,
            min_tokens_per_second=min_tokens_per_second,
            privacy=privacy_value,
            critical_floors=_parse_float_assignments(floor or [], "floor"),
            utility_weights=_parse_float_assignments(utility_weight or [], "utility-weight"),
            utility_scales=_parse_float_assignments(utility_scale or [], "utility-scale"),
        )
        view = set_workload_profile(path, profile)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)
    if json_output:
        _emit_json(_workload_payload(view))
        return
    _print_workload(view)


@build_app.command("show")
def build_show(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_build_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_build_payload(view))
        return
    _print_build(view)


@build_app.command("intent")
def build_intent_command(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    archetype: Annotated[str, typer.Option("--archetype")] = "Balanced",
    priority: Annotated[
        list[str] | None,
        typer.Option("--priority", help="Repeat axis=value, e.g. --priority coding=60."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        priorities = _parse_assignments(priority or [], "priority")
        view = set_build_intent(
            path,
            archetype=archetype,
            priorities=priorities,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_build_payload(view))
        return
    _print_build(view)


@build_app.command("set")
def build_set_command(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    target: Annotated[
        list[str] | None,
        typer.Option("--target", help="Repeat axis=value, e.g. --target coding=140."),
    ] = None,
    floor: Annotated[
        list[str] | None,
        typer.Option("--floor", help="Repeat axis=value, e.g. --floor general=120."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        targets = _parse_assignments(target or [], "target")
        floors = _parse_assignments(floor or [], "floor")
        view = set_build_targets(
            path,
            targets=targets,
            floors=floors,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_build_payload(view))
        return
    _print_build(view)


@stats_app.command("show")
def stats_show(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Inspect a specific champion/candidate model ID."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_stats_view(path, model)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_stats_payload(view))
        return
    _print_stats(view)


@stats_app.command("ingest")
def stats_ingest(
    receipt: Annotated[Path, typer.Option("--receipt", help="Evaluation receipt JSON.")],
    scale: Annotated[Path, typer.Option("--scale", help="Frozen capability scale JSON.")],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = ingest_stats(
            path,
            receipt_path=receipt,
            scale_path=scale,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_stats_payload(view))
        return
    _print_stats(view)


@evaluation_app.command("import-lm-eval")
def evaluation_import_lm_eval(
    result: Annotated[
        Path,
        typer.Argument(help="lm-evaluation-harness results JSON."),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Exact model ID; defaults to current Champion."),
    ] = None,
    harness_version: Annotated[
        str,
        typer.Option(
            "--harness-version",
            help="Exact lm-evaluation-harness version or immutable revision.",
        ),
    ] = "",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = import_lm_eval_evidence(
            path,
            result_path=result,
            model_id=model,
            harness_version=harness_version,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    payload = {"ok": True, **view.to_dict()}
    if json_output:
        _emit_json(payload)
        return
    console.print("[bold]LM-EVAL EVIDENCE IMPORTED[/bold]")
    console.print(f"Model: {payload['model_id']}")
    console.print(f"Evaluator: {payload['evaluator_id']}@{payload['evaluator_version']}")
    console.print(f"Receipt: {payload['receipt_id']}")
    console.print(f"Tasks: {payload['task_count']} · metrics: {payload['measurement_count']}")
    console.print("Capability stats activated: NO")
    console.print(str(payload["note"]))


@evaluation_app.command("import-lighteval")
def evaluation_import_lighteval(
    result: Annotated[
        Path,
        typer.Argument(help="LightEval saved results JSON."),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Exact model ID; defaults to current Champion."),
    ] = None,
    lighteval_version: Annotated[
        str,
        typer.Option(
            "--lighteval-version",
            help="Exact LightEval version or immutable revision used to create the result.",
        ),
    ] = "",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = import_lighteval_evidence(
            path,
            result_path=result,
            model_id=model,
            lighteval_version=lighteval_version,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    payload = {"ok": True, **view.to_dict()}
    if json_output:
        _emit_json(payload)
        return
    console.print("[bold]LIGHTEVAL EVIDENCE IMPORTED[/bold]")
    console.print(f"Model: {payload['model_id']}")
    console.print(f"Evaluator: {payload['evaluator_id']}@{payload['evaluator_version']}")
    console.print(f"Receipt: {payload['receipt_id']}")
    console.print(f"Tasks: {payload['task_count']} · metrics: {payload['measurement_count']}")
    console.print("Capability stats activated: NO")
    console.print(str(payload["note"]))


@evaluation_app.command("capability-v1")
def evaluation_capability_v1(
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model ID. Defaults to current champion."),
    ] = None,
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated evaluator environment.",
        ),
    ] = sys.executable,
    device: Annotated[
        str,
        typer.Option("--device", help="auto, cpu, or cuda."),
    ] = "auto",
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", min=0.001, help="Maximum evaluator wall time."),
    ] = 300.0,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate Capability v1 identity and replay status without evaluating.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        if dry_run:
            preview = preflight_capability_v1(
                path,
                model_id=model,
                python_executable=python_executable,
                device=device,
                timeout_seconds=timeout_seconds,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return

        view = run_capability_v1(
            path,
            model_id=model,
            python_executable=python_executable,
            device=device,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json({"ok": True, **view.to_dict()})
        return

    console.print("[bold]FRONTIERWRIGHT CAPABILITY v1[/bold]")
    console.print(f"Model: {view.model_id}")
    console.print(f"Bundle: {view.bundle_id}@{view.bundle_version}")
    console.print(f"Bundle hash: {view.bundle_hash}")
    console.print(f"Scale hash: {view.scale_hash}")
    console.print(f"Receipt: {view.receipt_id}")
    console.print(f"Replayed: {'YES' if view.replayed else 'NO'}")
    for axis in ("general", "reasoning", "math", "coding"):
        value = view.stats.get(axis)
        console.print(f"{axis.title():<10} {value if value is not None else '?'}")


@evaluation_app.command("packs")
def evaluation_packs(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    packs = get_evaluation_packs()
    if json_output:
        _emit_json(
            {
                "schema_version": 1,
                "ok": True,
                "packs": packs,
            }
        )
        return

    console.print("[bold]EVALUATION PACKS[/bold]")
    for item in packs:
        console.print(f"{item.get('pack_id')}@{item.get('pack_version')} · {item.get('title')}")
        metrics = item.get("metrics")
        if isinstance(metrics, list):
            console.print("  Metrics: " + ", ".join(str(value) for value in metrics))


@evaluation_app.command("run")
def evaluation_run(
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Registered dataset ID used as evaluation input."),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    pack: Annotated[
        str,
        typer.Option("--pack", help="Evaluation pack ID."),
    ] = REFERENCE_LM_PACK.pack_id,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model ID. Defaults to current champion."),
    ] = None,
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated evaluator environment.",
        ),
    ] = sys.executable,
    device: Annotated[
        str,
        typer.Option("--device", help="auto, cpu, or cuda."),
    ] = "auto",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Evaluation windows per batch."),
    ] = 4,
    max_batches: Annotated[
        int,
        typer.Option("--max-batches", min=1, help="Maximum deterministic batches."),
    ] = 16,
    max_dataset_bytes: Annotated[
        int,
        typer.Option(
            "--max-dataset-bytes",
            min=1,
            help="Maximum dataset bytes read by the evaluator.",
        ),
    ] = 64 * 1024 * 1024,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", min=0.001, help="Maximum evaluator wall time."),
    ] = 300.0,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate evaluation identity and replay status without evaluating.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        if dry_run:
            preview = preflight_evaluation_pack(
                path,
                pack_id=pack,
                dataset_id=dataset,
                model_id=model,
                python_executable=python_executable,
                device=device,
                batch_size=batch_size,
                max_batches=max_batches,
                max_dataset_bytes=max_dataset_bytes,
                timeout_seconds=timeout_seconds,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return

        view = run_evaluation_pack(
            path,
            pack_id=pack,
            dataset_id=dataset,
            model_id=model,
            python_executable=python_executable,
            device=device,
            batch_size=batch_size,
            max_batches=max_batches,
            max_dataset_bytes=max_dataset_bytes,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_evaluation_payload(view))
        return
    _print_evaluation(view)


@evaluation_app.command("compare")
def evaluation_compare(
    candidate: Annotated[
        str,
        typer.Option("--candidate", help="Candidate model ID to compare with champion."),
    ],
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Registered dataset ID used for both evaluations."),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="Frontierwright project directory."),
    ] = Path("."),
    pack: Annotated[
        str,
        typer.Option("--pack", help="Evaluation pack ID."),
    ] = REFERENCE_LM_PACK.pack_id,
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated evaluator environment.",
        ),
    ] = sys.executable,
    device: Annotated[
        str,
        typer.Option("--device", help="auto, cpu, or cuda."),
    ] = "auto",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Evaluation windows per batch."),
    ] = 4,
    max_batches: Annotated[
        int,
        typer.Option("--max-batches", min=1, help="Maximum deterministic batches."),
    ] = 16,
    max_dataset_bytes: Annotated[
        int,
        typer.Option(
            "--max-dataset-bytes",
            min=1,
            help="Maximum dataset bytes read by each evaluator.",
        ),
    ] = 64 * 1024 * 1024,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", min=0.001, help="Maximum wall time per evaluator."),
    ] = 300.0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = compare_candidate_evaluation(
            path,
            candidate_model_id=candidate,
            pack_id=pack,
            dataset_id=dataset,
            python_executable=python_executable,
            device=device,
            batch_size=batch_size,
            max_batches=max_batches,
            max_dataset_bytes=max_dataset_bytes,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_evaluation_compare_payload(view))
        return
    _print_evaluation_compare(view)


@data_app.command("show")
def data_show(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_data_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_data_payload(view))
        return
    _print_data(view)


@data_app.command("add")
def data_add(
    source: Annotated[Path, typer.Argument(help="Local dataset file or directory.")],
    role: Annotated[
        DatasetRole,
        typer.Option(
            "--role",
            case_sensitive=False,
            help="PRETRAIN, SFT, or PREFERENCE.",
        ),
    ],
    classification: Annotated[
        DatasetClassification | None,
        typer.Option(
            "--classification",
            case_sensitive=False,
            help="PUBLIC, INTERNAL, CONFIDENTIAL, or PRIVATE. Local data defaults PRIVATE.",
        ),
    ] = None,
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    name: Annotated[str | None, typer.Option("--name")] = None,
    license_name: Annotated[str | None, typer.Option("--license")] = None,
    domain: Annotated[str | None, typer.Option("--domain")] = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    token_count: Annotated[int | None, typer.Option("--token-count")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = add_local_dataset(
            path,
            source,
            name=name or source.stem or "Dataset",
            role=role,
            classification=classification,
            license_name=license_name,
            domain=domain,
            language=language,
            token_count=token_count,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_data_payload(view))
        return
    _print_data(view)


@data_app.command("recipes")
def data_recipes(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    plugins = get_data_preparation_plugins()
    if json_output:
        _emit_json(
            {
                "schema_version": 1,
                "ok": True,
                "plugins": plugins,
            }
        )
        return
    console.print("[bold]DATA PREPARATION RECIPES[/bold]")
    for plugin in plugins:
        console.print(f"{plugin['plugin_id']}@{plugin['plugin_version']} · {plugin['title']}")


@data_app.command("prepare")
def data_prepare(
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Source dataset ID to prepare."),
    ],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    name: Annotated[str | None, typer.Option("--name")] = None,
    recipe: Annotated[
        str,
        typer.Option(
            "--recipe",
            help=(
                "Preparation plugin ID. Aliases: snapshot-copy-v1 for a byte-preserving "
                "snapshot; text-lines-v1 for UTF-8 normalization and stable exact dedupe; "
                "preference-jsonl-v1 for canonical preference pairs; "
                "byte-shards-v1 for deterministic uint8 byte-ID shards."
            ),
        ),
    ] = "snapshot-copy-v1",
    bytes_per_shard: Annotated[
        int | None,
        typer.Option(
            "--bytes-per-shard",
            help="Shard size for byte-shards-v1. Omit to use the recipe default.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate recipe identity and replay status without materializing data.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    aliases = {
        "snapshot-copy-v1": SNAPSHOT_COPY_PLUGIN_ID,
        "text-lines-v1": TEXT_LINES_PLUGIN_ID,
        "preference-jsonl-v1": PREFERENCE_JSONL_PLUGIN_ID,
        "byte-shards-v1": BYTE_SHARDS_PLUGIN_ID,
    }
    plugin_id = aliases.get(recipe, recipe)
    if bytes_per_shard is not None and plugin_id != BYTE_SHARDS_PLUGIN_ID:
        _fail(
            FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                "--bytes-per-shard is only valid with byte-shards-v1.",
                2,
            ),
            json_output=json_output,
        )
    config: dict[str, object] | None = (
        {"bytes_per_shard": bytes_per_shard} if bytes_per_shard is not None else None
    )
    try:
        if dry_run:
            preview = preflight_prepare_dataset(
                path,
                dataset_id=dataset,
                plugin_id=plugin_id,
                config=config,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return

        view = prepare_dataset(
            path,
            dataset_id=dataset,
            plugin_id=plugin_id,
            config=config,
            name=name,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_data_payload(view))
        return
    _print_data(view)


@data_app.command("mix")
def data_mix(
    inputs: Annotated[
        list[str],
        typer.Option(
            "--input",
            help=(
                "Prepared text dataset in DATASET_ID:PARTS form. Repeat --input "
                "for every mixture source."
            ),
        ),
    ],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    name: Annotated[str | None, typer.Option("--name")] = None,
    max_output_bytes: Annotated[
        int,
        typer.Option(
            "--max-output-bytes",
            help="Hard preparation limit for the materialized mixed corpus.",
        ),
    ] = 4 * 1024**3,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate mixture identity and replay status without materializing data.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes

    parsed: list[tuple[str, int]] = []
    for raw in inputs:
        dataset_id, separator, parts_raw = raw.rpartition(":")
        if not separator or not dataset_id:
            _fail(
                FrontierwrightError(
                    "DATA_MIXTURE_INPUTS_INVALID",
                    f"Invalid --input {raw!r}; expected DATASET_ID:PARTS.",
                    2,
                ),
                json_output=json_output,
            )
        try:
            parts = int(parts_raw)
        except ValueError:
            _fail(
                FrontierwrightError(
                    "DATA_MIXTURE_INPUTS_INVALID",
                    f"Invalid mixture parts in --input {raw!r}.",
                    2,
                ),
                json_output=json_output,
            )
        parsed.append((dataset_id, parts))

    try:
        if dry_run:
            preview = preflight_prepare_dataset_mixture(
                path,
                inputs=parsed,
                max_output_bytes=max_output_bytes,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return

        view = prepare_dataset_mixture(
            path,
            inputs=parsed,
            name=name,
            max_output_bytes=max_output_bytes,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_data_payload(view))
        return
    _print_data(view)


@app.command("interventions")
def interventions_command(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    view = get_interventions_view()
    if json_output:
        _emit_json(_interventions_payload(view))
        return

    console.print("[bold]INTERVENTIONS[/bold]")
    for item in view.interventions:
        path_id = item.get("training_path_id")
        suffix = f" · path={path_id}" if path_id else ""
        console.print(
            f"{item.get('family')} / {item.get('surface')} · "
            f"{item.get('intervention_id')}@{item.get('version')}{suffix}"
        )
        console.print(f"  {item.get('title')}")


@app.command("paths")
def paths_command(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_paths_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_paths_payload(view))
        return
    _print_paths(view)


@backend_app.command("reference-spec")
def backend_reference_spec(
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated training environment.",
        ),
    ] = sys.executable,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Where to write the structured reference backend spec.",
        ),
    ] = Path(".frontierwright/backends/reference-pytorch-v1.json"),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    payload = backend_spec_payload(python_executable)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    response: dict[str, object] = {
        "schema_version": 1,
        "ok": True,
        "backend_spec_path": str(output),
        "backend": payload,
    }
    if json_output:
        _emit_json(response)
        return
    console.print("[bold green]Reference backend spec written[/bold green]")
    console.print(str(output))
    console.print(f"Python: {python_executable}")


@backend_app.command("doctor")
def backend_doctor(
    python_executable: Annotated[
        str,
        typer.Option(
            "--python",
            help="Python executable for the isolated training environment.",
        ),
    ] = sys.executable,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    script = (
        "import importlib.util,json,sys;"
        "p={'python':sys.executable,'python_version':sys.version.split()[0]};"
        "p['torch_available']=importlib.util.find_spec('torch') is not None;"
        "p['psutil_available']=importlib.util.find_spec('psutil') is not None;"
        "exec(\"if p['torch_available']:\\n import torch\\n "
        "p['torch_version']=torch.__version__\\n "
        "p['cuda_available']=torch.cuda.is_available()\\n "
        "p['cuda_device_count']=torch.cuda.device_count()\\n "
        "p['cuda_device_name']=(torch.cuda.get_device_name(0) "
        'if torch.cuda.is_available() else None)");'
        "print(json.dumps(p,sort_keys=True))"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", script],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        _fail(
            FrontierwrightError(
                "BACKEND_PYTHON_UNAVAILABLE",
                f"Could not execute training Python: {python_executable}",
                14,
            ),
            json_output=json_output,
        )

    if result.returncode != 0:
        _fail(
            FrontierwrightError(
                "BACKEND_DOCTOR_FAILED",
                result.stderr.strip() or result.stdout.strip() or "training Python probe failed",
                14,
            ),
            json_output=json_output,
        )

    try:
        payload_raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        _fail(
            FrontierwrightError(
                "BACKEND_DOCTOR_INVALID",
                "Training Python probe returned invalid JSON.",
                14,
            ),
            json_output=json_output,
        )

    if not isinstance(payload_raw, dict):
        _fail(
            FrontierwrightError(
                "BACKEND_DOCTOR_INVALID",
                "Training Python probe returned a non-object.",
                14,
            ),
            json_output=json_output,
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "ok": True,
        **payload_raw,
    }
    if json_output:
        _emit_json(payload)
        return

    console.print("[bold]REFERENCE BACKEND DOCTOR[/bold]")
    console.print(f"Python: {payload.get('python')}")
    console.print(f"Python version: {payload.get('python_version')}")
    console.print(f"Torch: {payload.get('torch_version') or 'not installed'}")
    console.print(f"psutil: {'YES' if payload.get('psutil_available') else 'NO'}")
    console.print(f"CUDA: {'YES' if payload.get('cuda_available') else 'NO'}")
    if payload.get("cuda_device_name"):
        console.print(f"GPU: {payload.get('cuda_device_name')}")


@plan_app.command("create")
def plan_create(
    path_id: Annotated[
        TrainingPathId,
        typer.Argument(help="Training path ID."),
    ],
    backend_spec: Annotated[
        Path,
        typer.Option("--backend-spec", help="Structured Command Backend spec JSON."),
    ],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    dataset: Annotated[str | None, typer.Option("--dataset")] = None,
    permission: Annotated[str, typer.Option("--permission")] = "PLAN",
    config_json: Annotated[Path | None, typer.Option("--config-json")] = None,
    max_wall_seconds: Annotated[float | None, typer.Option("--max-wall-seconds")] = None,
    max_gpu_hours: Annotated[float | None, typer.Option("--max-gpu-hours")] = None,
    max_runs: Annotated[int | None, typer.Option("--max-runs")] = None,
    max_storage_bytes: Annotated[int | None, typer.Option("--max-storage-bytes")] = None,
    max_money: Annotated[float | None, typer.Option("--max-money")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        parsed_permission = _parse_permission(permission)
        config = _load_config_json(config_json)
        budgets = HardBudgets(
            max_wall_seconds=max_wall_seconds,
            max_gpu_hours=max_gpu_hours,
            max_runs=max_runs,
            max_storage_bytes=max_storage_bytes,
            max_money=max_money,
        )
        view = create_training_plan(
            path,
            path_id=path_id,
            backend_spec_path=backend_spec,
            dataset_id=dataset,
            permission=parsed_permission,
            budgets=budgets,
            config=config,
        )
    except ValueError as exc:
        _fail(
            FrontierwrightError("INVALID_BUDGET", str(exc), 2),
            json_output=json_output,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_plan_payload(view))
        return
    _print_plan(view)


@plan_app.command("show")
def plan_show(
    plan_id: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_plan_view(path, plan_id)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_plan_payload(view))
        return
    _print_plan(view)


@app.command("calibrate")
def calibrate_command(
    plan_id: Annotated[str, typer.Argument()],
    backend_spec: Annotated[Path, typer.Option("--backend-spec")],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 300.0,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate calibration without executing the backend."),
    ] = False,
    rerun: Annotated[
        bool,
        typer.Option(
            "--rerun",
            help="Intentionally repeat calibration instead of replaying evidence.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        if dry_run:
            preview = preflight_training_calibration(
                path,
                plan_id=plan_id,
                backend_spec_path=backend_spec,
                timeout_seconds=timeout_seconds,
            )
            if json_output:
                _emit_json({"ok": True, **preview.to_dict()})
                return
            _print_preflight(preview)
            return
        view = calibrate_training_plan(
            path,
            plan_id=plan_id,
            backend_spec_path=backend_spec,
            timeout_seconds=timeout_seconds,
            rerun=rerun,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_plan_payload(view))
        return
    _print_plan(view)


@app.command("run")
def run_command(
    plan_id: Annotated[str, typer.Argument()],
    backend_spec: Annotated[Path, typer.Option("--backend-spec")],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    rerun: Annotated[bool, typer.Option("--rerun")] = False,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 86400.0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = execute_training_plan(
            path,
            plan_id=plan_id,
            backend_spec_path=backend_spec,
            dry_run=dry_run,
            rerun=rerun,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_run_payload(view))
        return
    _print_run(view)


@app.command("run-status")
def run_status(
    run_id: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_run_view(path, run_id)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_run_payload(view))
        return
    _print_run(view)


@app.command("run-reconcile")
def run_reconcile(
    run_id: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = reconcile_training_run(path, run_id)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_run_payload(view))
        return
    _print_run(view)


@app.command("run-repair-receipt")
def run_repair_receipt(
    run_id: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = repair_run_receipt(path, run_id)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_run_payload(view))
        return
    _print_run(view)


@app.command("train")
def train_command(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    plan_id: Annotated[
        str | None,
        typer.Option("--plan", help="Use a specific calibrated READY plan."),
    ] = None,
    backend_spec: Annotated[
        Path | None,
        typer.Option("--backend-spec", help="Pinned backend spec used by the plan."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually create a candidate; default is dry-run."),
    ] = False,
    rerun: Annotated[bool, typer.Option("--rerun")] = False,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 86400.0,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    try:
        paths = get_paths_view(path)
        ready = [
            item
            for item in paths.paths
            if item.get("availability") == "READY" and isinstance(item.get("ready_plan_id"), str)
        ]
        if plan_id is None:
            if not ready:
                _print_paths(paths)
                raise FrontierwrightError(
                    "NO_READY_PATH",
                    (
                        "No path is READY yet. Register data, create a plan, and run "
                        "calibration first."
                    ),
                    12,
                )
            if non_interactive:
                raise FrontierwrightError(
                    "PLAN_REQUIRED",
                    "--plan is required in --non-interactive mode.",
                    2,
                )
            console.print("[bold]READY TRAINING PATHS[/bold]")
            for index, item in enumerate(ready, start=1):
                console.print(f"[{index}] {item.get('title')} · {item.get('ready_plan_id')}")
            selected = typer.prompt(
                "Select path",
                type=int,
                default=1,
                show_default=True,
            )
            if selected < 1 or selected > len(ready):
                raise FrontierwrightError(
                    "INVALID_SELECTION",
                    "Selected training path number is out of range.",
                    2,
                )
            plan_id = str(ready[selected - 1]["ready_plan_id"])

        plan = get_plan_view(path, plan_id)
        if not plan.ready:
            _print_plan(plan)
            raise FrontierwrightError(
                "PLAN_NOT_READY",
                "Selected plan is not READY.",
                13,
            )

        if backend_spec is None:
            if non_interactive:
                raise FrontierwrightError(
                    "BACKEND_SPEC_REQUIRED",
                    "--backend-spec is required in --non-interactive mode.",
                    2,
                )
            entered = typer.prompt("Backend spec path")
            backend_spec = Path(entered)

        if execute and not yes:
            if non_interactive:
                raise FrontierwrightError(
                    "CONFIRMATION_REQUIRED",
                    "Use --yes with --execute in --non-interactive mode.",
                    2,
                )
            confirmed = typer.confirm(
                "Run training and create a candidate? Champion will remain unchanged.",
                default=False,
            )
            if not confirmed:
                console.print("Cancelled.")
                return

        result = execute_training_plan(
            path,
            plan_id=plan_id,
            backend_spec_path=backend_spec,
            dry_run=not execute,
            rerun=rerun,
            timeout_seconds=timeout_seconds,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=False)

    _print_run(result)
    if not execute:
        console.print("\nDry-run passed. Re-run with --execute to create a candidate.")


@app.command("candidates")
def candidates_command(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_candidates_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_candidate_payload(view))
        return
    _print_candidates(view)


@app.command("compare")
def compare_command(
    candidate_model_id: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = compare_candidate(path, candidate_model_id)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_compare_payload(view))
        return
    _print_compare(view)


@app.command("promote")
def promote_command(
    candidate_model_id: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    allow_unmeasured: Annotated[bool, typer.Option("--allow-unmeasured")] = False,
    allow_build_violations: Annotated[
        bool,
        typer.Option(
            "--allow-build-violations",
            help="Explicitly override build-floor or build-scale promotion blockers.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = promote_candidate(
            path,
            candidate_model_id,
            allow_unmeasured=allow_unmeasured,
            allow_build_violations=allow_build_violations,
        )
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_status_payload(view))
        return
    _print_status(view)


@app.command("reject")
def reject_command(
    candidate_model_id: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = reject_candidate(path, candidate_model_id)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_candidate_payload(view))
        return
    _print_candidates(view)


@app.command("history")
def history_command(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del non_interactive
    try:
        view = get_history_view(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_history_payload(view))
        return
    _print_history(view)


@app.command("play")
def play(
    path: Annotated[Path, typer.Option("--path", help="Project directory.")] = Path("."),
    language: Annotated[
        str | None,
        typer.Option("--lang", help="Override human UI language."),
    ] = None,
) -> None:
    from frontierwright.tui import FrontierwrightApp

    view = get_status(path)
    resources = get_resource_view(path)
    build = get_build_view(path)
    data = get_data_view(path)
    workload = get_workload_view(path)
    workload_fit = get_workload_fit(path)
    paths = get_paths_view(path)
    candidates = get_candidates_view(path)
    history = get_history_view(path)
    lang = language or view.language
    if lang not in {"en", "ko"}:
        raise typer.BadParameter("--lang must be en or ko")
    FrontierwrightApp(
        view=view,
        resources=resources,
        build=build,
        data=data,
        workload=workload,
        workload_fit=workload_fit,
        paths=paths,
        candidates=candidates,
        history=history,
        root=path,
        language=lang,
    ).run()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
