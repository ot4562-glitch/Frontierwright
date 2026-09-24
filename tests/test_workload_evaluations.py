from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.data import DatasetClassification
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.registry import Registry
from frontierwright.service import (
    bind_workload_evaluation,
    get_workload_fit,
    get_workload_view,
    import_lm_eval_evidence,
    set_workload_profile,
)
from frontierwright.workloads import WorkloadProfile

runner = CliRunner()


def _project(root: Path) -> tuple[Registry, ModelState]:
    registry = Registry(root)
    registry.initialize("FITMODEL", ModelOrigin.IMPORTED_LOCAL)
    state = registry.read()
    model = ModelState(
        model_id="model-workload-eval",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="synthetic-checkpoint",
        fingerprint="sha256:workload-eval-model",
    )
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)
    set_workload_profile(
        root,
        WorkloadProfile(
            name="Korean economics research",
            languages=("ko",),
            domains=("economics",),
            task_weights={"research": 1.0},
            privacy=DatasetClassification.PUBLIC,
        ),
    )
    return registry, model


def _lm_eval_result(path: Path) -> Path:
    payload = {
        "results": {
            "ko_understanding": {"acc,none": 0.73, "acc_stderr,none": 0.02},
            "economics_reasoning": {"acc,none": 0.64, "acc_stderr,none": 0.03},
            "research_synthesis": {"acc,none": 0.58, "acc_stderr,none": 0.04},
        },
        "versions": {
            "ko_understanding": 2,
            "economics_reasoning": "1.1",
            "research_synthesis": "2026-09",
        },
        "configs": {},
        "n-shot": {},
        "n-samples": {
            "ko_understanding": {"effective": 100},
            "economics_reasoning": {"effective": 80},
            "research_synthesis": {"effective": 60},
        },
        "higher_is_better": {
            "ko_understanding": {"acc": True},
            "economics_reasoning": {"acc": True},
            "research_synthesis": {"acc": True},
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _manifest(
    path: Path,
    *,
    profile_hash: str,
    receipt_id: str,
    include_research: bool = True,
    bad_metric: bool = False,
) -> Path:
    coverage: dict[str, object] = {
        "languages": {
            "ko": [
                {
                    "task_id": "lm-eval:ko_understanding",
                    "task_version": "2",
                    "metric": "missing,none" if bad_metric else "acc,none",
                }
            ]
        },
        "domains": {
            "economics": [
                {
                    "task_id": "lm-eval:economics_reasoning",
                    "task_version": "1.1",
                    "metric": "acc,none",
                }
            ]
        },
    }
    if include_research:
        coverage["tasks"] = {
            "research": [
                {
                    "task_id": "lm-eval:research_synthesis",
                    "task_version": "2026-09",
                    "metric": "acc,none",
                }
            ]
        }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workload_profile_hash": profile_hash,
                "receipt_id": receipt_id,
                "coverage": coverage,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_exact_external_eval_binding_completes_workload_coverage_and_is_idempotent(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, model = _project(project)
    imported = import_lm_eval_evidence(
        project,
        result_path=_lm_eval_result(tmp_path / "lm-eval.json"),
        model_id=model.model_id,
        harness_version="0.4.9@abcdef",
    )
    workload = get_workload_view(project)
    assert workload.profile_hash is not None

    manifest = _manifest(
        tmp_path / "binding.json",
        profile_hash=workload.profile_hash,
        receipt_id=imported.receipt.receipt_id,
    )
    first = bind_workload_evaluation(project, manifest_path=manifest)
    second = bind_workload_evaluation(project, manifest_path=manifest)

    assert first == second
    fit = get_workload_fit(project)
    assert fit.overall_status == "PASS"
    assert fit.workload_evaluation_coverage == {
        "complete": True,
        "profile_hash": workload.profile_hash,
        "model_id": model.model_id,
        "binding_ids": [first.binding_id],
        "receipt_ids": [imported.receipt.receipt_id],
        "covered": {
            "languages": ["ko"],
            "domains": ["economics"],
            "tasks": ["research"],
        },
        "missing": {"languages": [], "domains": [], "tasks": []},
    }
    event_count = sum(
        item["kind"] == "WORKLOAD_EVALUATION_BOUND" for item in registry.read().history
    )
    assert event_count == 1


def test_partial_binding_stays_unknown_and_cli_reports_missing_coverage(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, model = _project(project)
    imported = import_lm_eval_evidence(
        project,
        result_path=_lm_eval_result(tmp_path / "lm-eval.json"),
        model_id=model.model_id,
        harness_version="0.4.9",
    )
    workload = get_workload_view(project)
    assert workload.profile_hash is not None
    manifest = _manifest(
        tmp_path / "partial.json",
        profile_hash=workload.profile_hash,
        receipt_id=imported.receipt.receipt_id,
        include_research=False,
    )

    result = runner.invoke(
        app,
        [
            "workload",
            "bind-eval",
            str(manifest),
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["workload_fit_status"] == "UNKNOWN"
    assert payload["workload_evaluation_coverage"]["complete"] is False
    assert payload["workload_evaluation_coverage"]["missing"]["tasks"] == ["research"]

    fit_result = runner.invoke(
        app,
        ["workload", "fit", "--path", str(project), "--json", "--non-interactive"],
    )
    assert fit_result.exit_code == 0, fit_result.output
    fit_payload = json.loads(fit_result.stdout)
    assert fit_payload["workload_evaluation_coverage"]["missing"]["tasks"] == ["research"]


def test_binding_rejects_selector_not_present_in_exact_receipt(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, model = _project(project)
    imported = import_lm_eval_evidence(
        project,
        result_path=_lm_eval_result(tmp_path / "lm-eval.json"),
        model_id=model.model_id,
        harness_version="0.4.9",
    )
    workload = get_workload_view(project)
    assert workload.profile_hash is not None
    manifest = _manifest(
        tmp_path / "bad.json",
        profile_hash=workload.profile_hash,
        receipt_id=imported.receipt.receipt_id,
        bad_metric=True,
    )

    result = runner.invoke(
        app,
        [
            "workload",
            "bind-eval",
            str(manifest),
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "WORKLOAD_EVALUATION_MEASUREMENT_MISSING"


def test_binding_rejects_stale_workload_profile_hash(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, model = _project(project)
    imported = import_lm_eval_evidence(
        project,
        result_path=_lm_eval_result(tmp_path / "lm-eval.json"),
        model_id=model.model_id,
        harness_version="0.4.9",
    )
    manifest = _manifest(
        tmp_path / "stale.json",
        profile_hash="sha256:" + "0" * 64,
        receipt_id=imported.receipt.receipt_id,
    )

    result = runner.invoke(
        app,
        [
            "workload",
            "bind-eval",
            str(manifest),
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "WORKLOAD_EVALUATION_PROFILE_MISMATCH"
