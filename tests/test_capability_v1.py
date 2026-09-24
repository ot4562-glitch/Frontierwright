import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import frontierwright.service as service_module
from frontierwright.capability_v1 import (
    CAPABILITY_V1_BUNDLE_ID,
    CAPABILITY_V1_BUNDLE_VERSION,
    CAPABILITY_V1_EVALUATOR_ID,
    CAPABILITY_V1_EVALUATOR_VERSION,
    CAPABILITY_V1_FROZEN_BUNDLE_SHA256,
    CAPABILITY_V1_FROZEN_SCALE_SHA256,
    CAPABILITY_V1_SCALE,
    CAPABILITY_V1_SCORING,
    CAPABILITY_V1_TASKS,
    capability_v1_bundle_hash,
    capability_v1_task_counts,
)
from frontierwright.cli import app
from frontierwright.domain import Axis, ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import EvaluationReceipt, RawMeasurement, apply_scale
from frontierwright.registry import Registry
from frontierwright.service import (
    get_build_view,
    get_stats_view,
    import_local_model,
    run_capability_v1,
)

runner = CliRunner()


def make_reference_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "frontierwright_byte_causal_lm",
                "frontierwright_reference_backend": (
                    "frontierwright-reference-pytorch-v1"
                ),
                "preset": "zero-8m",
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    (root / "pytorch_model.bin").write_bytes(b"capability-v1-reference-model")
    return root


def setup_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_reference_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Capability",
    )
    return project


def fake_capability_result(
    argv_template: tuple[str, ...],
    *,
    environment_overrides: dict[str, str],
    request_path: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    del argv_template, environment_overrides, timeout_seconds
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["operation"] == "capability_v1"
    values = {
        "general": (8, 0.10),
        "reasoning": (4, -0.03),
        "math": (12, 0.22),
        "coding": (16, 0.41),
    }
    axes = []
    for axis in ("general", "reasoning", "math", "coding"):
        correct, margin = values[axis]
        axes.append(
            {
                "axis": axis,
                "task_id": f"{CAPABILITY_V1_BUNDLE_ID}.{axis}",
                "task_version": CAPABILITY_V1_BUNDLE_VERSION,
                "accuracy": correct / 16,
                "correct": correct,
                "total": 16,
                "mean_correct_margin_nats": margin,
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
            "elapsed_seconds": 1.0,
            "parameter_count": 7_521_280,
            "tokenizer_fingerprint": "fixture-tokenizer-fingerprint",
            "python_version": "3.12.10",
            "torch_version": "2.14.0+cpu",
        },
    }


def test_capability_v1_bundle_is_frozen_and_balanced() -> None:
    assert len(CAPABILITY_V1_TASKS) == 64
    assert capability_v1_task_counts() == {
        "general": 16,
        "reasoning": 16,
        "math": 16,
        "coding": 16,
    }
    assert len({task.item_id for task in CAPABILITY_V1_TASKS}) == 64
    assert all(len(task.choices) == 4 for task in CAPABILITY_V1_TASKS)
    assert CAPABILITY_V1_SCALE.frozen is True
    assert len(CAPABILITY_V1_SCALE.axes) == 4
    digest = capability_v1_bundle_hash()
    assert len(digest) == 64
    assert digest == CAPABILITY_V1_FROZEN_BUNDLE_SHA256
    assert CAPABILITY_V1_SCALE.sha256 == CAPABILITY_V1_FROZEN_SCALE_SHA256


def test_capability_v1_scale_has_explicit_non_capped_reference_anchor() -> None:
    measurements = []
    raw_by_axis = {
        Axis.GENERAL: 0.0,
        Axis.REASONING: 0.25,
        Axis.MATH: 0.5,
        Axis.CODING: 1.0,
    }
    for axis, value in raw_by_axis.items():
        measurements.append(
            RawMeasurement(
                task_id=f"{CAPABILITY_V1_BUNDLE_ID}.{axis.value.lower()}",
                task_version=CAPABILITY_V1_BUNDLE_VERSION,
                metric="accuracy",
                value=value,
                higher_is_better=True,
            )
        )
    receipt = EvaluationReceipt(
        receipt_id="receipt-scale-shape",
        model_id="model-scale-shape",
        model_fingerprint="fixture-model-fingerprint",
        evaluator_id=CAPABILITY_V1_EVALUATOR_ID,
        evaluator_version=CAPABILITY_V1_EVALUATOR_VERSION,
        conditions={},
        measurements=tuple(measurements),
    )

    stats = {item.axis: item.value for item in apply_scale(receipt, CAPABILITY_V1_SCALE)}

    assert stats[Axis.GENERAL] == 0.0
    assert stats[Axis.REASONING] == 50.0
    assert stats[Axis.MATH] == 100.0
    assert stats[Axis.CODING] == 200.0


def test_capability_v1_generates_receipt_activates_stats_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = setup_project(tmp_path)
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_capability_result(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", counted)

    first = run_capability_v1(
        project,
        python_executable="fixture-python",
        device="cpu",
    )
    second = run_capability_v1(
        project,
        python_executable="different-python-does-not-change-evidence",
        device="cpu",
    )

    assert calls == 1
    assert first.replayed is False
    assert second.replayed is True
    assert first.receipt_id == second.receipt_id
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.bundle_hash == capability_v1_bundle_hash()
    assert first.scale_hash == CAPABILITY_V1_SCALE.sha256
    assert first.stats == {
        "general": 100.0,
        "reasoning": 50.0,
        "math": 150.0,
        "coding": 200.0,
    }

    stats = get_stats_view(project)
    assert stats.measured is True
    assert stats.scale_id == CAPABILITY_V1_BUNDLE_ID
    assert stats.scale_version == CAPABILITY_V1_BUNDLE_VERSION
    assert stats.scale_hash == CAPABILITY_V1_SCALE.sha256
    assert stats.stats == first.stats

    build = get_build_view(project)
    assert build.mode == "TARGETS_FLOORS"

    assert first.receipt_id is not None
    stored = Registry(project).get_evaluation_receipt(first.receipt_id)
    assert stored is not None
    assert len(stored["measurements"]) == 8
    assert stored["conditions"]["bundle_hash"] == capability_v1_bundle_hash()
    assert stored["conditions"]["scale_hash"] == CAPABILITY_V1_SCALE.sha256


def test_capability_v1_rejects_bundle_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = setup_project(tmp_path)

    def wrong_bundle(*args, **kwargs):
        result = fake_capability_result(*args, **kwargs)
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        metrics["bundle_hash"] = "0" * 64
        return result

    monkeypatch.setattr(service_module, "run_structured_command", wrong_bundle)

    with pytest.raises(FrontierwrightError, match="bundle/scoring identity"):
        run_capability_v1(
            project,
            python_executable="fixture-python",
            device="cpu",
        )


def test_capability_v1_cli_json_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = setup_project(tmp_path)
    monkeypatch.setattr(
        service_module,
        "run_structured_command",
        fake_capability_result,
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "capability-v1",
            "--path",
            str(project),
            "--python",
            "fixture-python",
            "--device",
            "cpu",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["bundle_id"] == CAPABILITY_V1_BUNDLE_ID
    assert payload["stats"]["general"] == 100.0
    assert payload["stats"]["coding"] == 200.0
