"""Typer command surface for humans and automation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from frontierwright.data import DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import HardBudgets, PermissionLevel
from frontierwright.paths import TrainingPathId
from frontierwright.reference_backend import backend_spec_payload
from frontierwright.registry import Registry
from frontierwright.service import (
    BuildView,
    CandidateView,
    CompareView,
    DataView,
    HistoryView,
    PathsView,
    PlanView,
    ResourceView,
    RunView,
    StatsView,
    StatusView,
    add_local_dataset,
    calibrate_training_plan,
    compare_candidate,
    create_training_plan,
    detect_resources,
    execute_training_plan,
    get_build_view,
    get_candidates_view,
    get_data_view,
    get_history_view,
    get_paths_view,
    get_plan_view,
    get_resource_view,
    get_run_view,
    get_stats_view,
    get_status,
    import_local_model,
    ingest_stats,
    promote_candidate,
    reconcile_training_run,
    reject_candidate,
    repair_run_receipt,
    set_build_intent,
    set_build_targets,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Build your LLM like a character.",
)
project_app = typer.Typer(help="Create and inspect Frontierwright project state.")
resources_app = typer.Typer(help="Detect and inspect project compute resources.")
build_app = typer.Typer(help="Inspect and edit the desired model build.")
stats_app = typer.Typer(help="Inspect or ingest capability evaluation evidence.")
data_app = typer.Typer(help="Register and inspect user/lab datasets.")
plan_app = typer.Typer(help="Create and inspect pinned training plans.")
backend_app = typer.Typer(help="Inspect and configure training backends.")
app.add_typer(project_app, name="project")
app.add_typer(resources_app, name="resources")
app.add_typer(build_app, name="build")
app.add_typer(stats_app, name="stats")
app.add_typer(data_app, name="data")
app.add_typer(plan_app, name="plan")
app.add_typer(backend_app, name="backend")

console = Console()


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


def _resource_payload(view: ResourceView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _build_payload(view: BuildView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _stats_payload(view: StatsView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _data_payload(view: DataView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _paths_payload(view: PathsView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _plan_payload(view: PlanView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _run_payload(view: RunView) -> dict[str, object]:
    return {"ok": True, **view.to_dict()}


def _candidate_payload(view: CandidateView) -> dict[str, object]:
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


def _print_data(view: DataView) -> None:
    console.print("[bold]DATA[/bold]")
    if not view.datasets:
        console.print("No datasets registered.")
        return
    for item in view.datasets:
        console.print(
            f"{item.get('name')} · {item.get('role')} · "
            f"{_human_bytes(item.get('total_bytes'))}"
        )
        console.print(f"  Fingerprint: {item.get('fingerprint')}")
        console.print(f"  Provenance: {item.get('provenance')}")
        console.print(f"  License: {item.get('license') or 'UNKNOWN'}")
        console.print(f"  Domain: {item.get('domain') or 'UNKNOWN'}")
        console.print(f"  Language: {item.get('language') or 'UNKNOWN'}")
        token_count = item.get("token_count")
        console.print(
            f"  Tokens: {token_count if isinstance(token_count, int) else 'UNKNOWN'}"
        )


def _print_paths(view: PathsView) -> None:
    console.print("[bold]TRAINING PATHS[/bold]")
    for item in view.paths:
        console.print(
            f"[{item.get('availability')}] {item.get('title')} "
            f"({item.get('path_id')})"
        )
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


def _print_plan(view: PlanView) -> None:
    console.print(f"[bold]PLAN[/bold] · {view.plan_id}")
    console.print(f"Path: {view.path_id}")
    console.print(f"Backend: {view.backend_id}")
    console.print(f"Permission: {view.permission}")
    console.print(f"Dataset: {view.dataset_id}")
    console.print(f"Resource profile: {view.resource_profile_id or 'UNPINNED'}")
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
    if view.build_constraints:
        console.print("")
        console.print("BUILD CONSTRAINTS")
        for item in view.build_constraints:
            console.print(
                f"{item.get('axis')} {item.get('kind')} "
                f"{item.get('threshold')} -> {item.get('status')}"
            )


def _print_history(view: HistoryView) -> None:
    console.print("[bold]DEVELOPMENT HISTORY[/bold]")
    if not view.events:
        console.print("No history events.")
        return
    for event in view.events:
        console.print(
            f"{event.get('sequence'):>4}  {event.get('kind')}  "
            f"{event.get('recorded_at')}"
        )
        details = event.get("details")
        if isinstance(details, dict) and details:
            console.print(f"      {json.dumps(details, ensure_ascii=False, sort_keys=True)}")


def _print_status(view: StatusView) -> None:
    console.print(f"[bold]{view.nickname}[/bold]")
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
        Registry(path).initialize(name, origin, language=language)
        view = get_status(path)
    except FrontierwrightError as exc:
        _fail(exc, json_output=json_output)

    if json_output:
        _emit_json(_status_payload(view))
        return

    console.print(f"[bold]Frontierwright[/bold] initialized: {view.project_name}")
    _print_status(view)


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
        typer.Option("--role", case_sensitive=False, help="PRETRAIN or SFT."),
    ],
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
        "if torch.cuda.is_available() else None)\");"
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
    json_output: Annotated[bool, typer.Option("--json")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    del non_interactive, yes
    try:
        view = calibrate_training_plan(
            path,
            plan_id=plan_id,
            backend_spec_path=backend_spec,
            timeout_seconds=timeout_seconds,
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
            if item.get("availability") == "READY"
            and isinstance(item.get("ready_plan_id"), str)
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
                console.print(
                    f"[{index}] {item.get('title')} · {item.get('ready_plan_id')}"
                )
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
        console.print(
            "\nDry-run passed. Re-run with --execute to create a candidate."
        )


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
