import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import frontierwright.service as service_module
from frontierwright.cli import app
from frontierwright.domain import ModelOrigin
from frontierwright.registry import Registry
from frontierwright.service import import_local_model, profile_reference_inference

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
                "parameter_count": 8_000_000,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"reference-profile-weights")
    return root


def setup_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_reference_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Profiler",
    )
    return project


def profile_result(request: dict[str, object]) -> dict[str, object]:
    config = request["config"]
    assert isinstance(config, dict)
    return {
        "schema_version": 1,
        "ok": True,
        "operation": "profile",
        "metrics": {
            "backend_id": "frontierwright-reference-pytorch-v1",
            "preset": config["preset"],
            "device": "cpu",
            "measurement_scope": "steady_state_generation_excludes_model_load",
            "profile_prompt": "Frontierwright inference profile:",
            "prompt_bytes": 31,
            "max_new_tokens": config["max_new_tokens"],
            "warmup_runs": config["warmup_runs"],
            "measured_runs": config["measured_runs"],
            "latency_seconds_runs": [0.2, 0.1],
            "latency_seconds_mean": 0.15,
            "latency_seconds_p50": 0.15,
            "tokens_per_second_runs": [80.0, 160.0],
            "tokens_per_second_mean": 120.0,
            "tokens_per_second_p50": 120.0,
            "max_sampled_process_rss_bytes": 123456,
            "cuda_memory_allocated_bytes": None,
            "peak_vram_bytes": None,
            "parameter_count": 8_000_000,
            "sample_generated_token_ids": [65, 66],
            "sample_continuation_text": "PRIVATE PROFILE SAMPLE",
            "python_version": "fixture",
            "torch_version": "fixture",
        },
    }


def test_profile_records_runtime_evidence_without_sample_text_in_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = setup_project(tmp_path)
    captured: dict[str, object] = {}

    def fake_backend(
        argv_template: tuple[str, ...],
        *,
        environment_overrides: dict[str, str],
        request_path: Path,
        timeout_seconds: float,
    ) -> dict[str, object]:
        del argv_template, environment_overrides, timeout_seconds
        request = json.loads(request_path.read_text(encoding="utf-8"))
        captured["request"] = request
        captured["request_path"] = request_path
        return profile_result(request)

    monkeypatch.setattr(service_module, "run_structured_command", fake_backend)

    view = profile_reference_inference(
        project,
        python_executable="fixture-python",
        max_new_tokens=2,
        warmup_runs=1,
        measured_runs=2,
        device="cpu",
    )

    assert view.intervention_id == "frontierwright.operate.profile-reference"
    assert view.metrics["latency_seconds_p50"] == 0.15
    assert view.metrics["tokens_per_second_p50"] == 120.0
    request_path = captured["request_path"]
    assert isinstance(request_path, Path)
    assert not request_path.exists()

    history = Registry(project).read().history
    event = next(item for item in history if item["kind"] == "INFERENCE_PROFILE_MEASURED")
    serialized = json.dumps(event, ensure_ascii=False, sort_keys=True)
    assert "PRIVATE PROFILE SAMPLE" not in serialized
    assert "sample_generated_token_ids" not in serialized
    assert event["details"]["model_id"] == view.model_id
    assert event["details"]["metrics"]["tokens_per_second_p50"] == 120.0


def test_operate_profile_json_machine_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = setup_project(tmp_path)

    def fake_backend(
        argv_template: tuple[str, ...],
        *,
        environment_overrides: dict[str, str],
        request_path: Path,
        timeout_seconds: float,
    ) -> dict[str, object]:
        del argv_template, environment_overrides, timeout_seconds
        request = json.loads(request_path.read_text(encoding="utf-8"))
        return profile_result(request)

    monkeypatch.setattr(service_module, "run_structured_command", fake_backend)

    result = runner.invoke(
        app,
        [
            "operate",
            "profile",
            "--path",
            str(project),
            "--python",
            "fixture-python",
            "--max-new-tokens",
            "2",
            "--warmup-runs",
            "1",
            "--runs",
            "2",
            "--device",
            "cpu",
            "--json",
            "--non-interactive",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["intervention_id"] == "frontierwright.operate.profile-reference"
    assert payload["metrics"]["latency_seconds_p50"] == 0.15
    assert payload["metrics"]["tokens_per_second_p50"] == 120.0


def test_profile_rejects_invalid_structured_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = setup_project(tmp_path)

    def fake_backend(
        argv_template: tuple[str, ...],
        *,
        environment_overrides: dict[str, str],
        request_path: Path,
        timeout_seconds: float,
    ) -> dict[str, object]:
        del argv_template, environment_overrides, request_path, timeout_seconds
        return {
            "schema_version": 1,
            "ok": True,
            "operation": "profile",
            "metrics": {"latency_seconds_p50": -1},
        }

    monkeypatch.setattr(service_module, "run_structured_command", fake_backend)

    with pytest.raises(Exception, match="Inference profile metric"):
        profile_reference_inference(
            project,
            python_executable="fixture-python",
            max_new_tokens=2,
            warmup_runs=1,
            measured_runs=1,
            device="cpu",
        )
