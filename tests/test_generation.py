import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import frontierwright.service as service_module
from frontierwright.cli import app
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.registry import Registry
from frontierwright.service import generate_reference_text, import_local_model

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
    (root / "model.safetensors").write_bytes(b"reference-generation-weights")
    return root


def setup_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_reference_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Generator",
    )
    return project


def test_generation_is_ephemeral_and_not_recorded_in_history(
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
        prompt = str(request["prompt"])
        count = int(request["config"]["max_new_tokens"])
        tokens = [65] * count
        continuation = "A" * count
        return {
            "schema_version": 1,
            "ok": True,
            "operation": "generate",
            "prompt": prompt,
            "continuation_text": continuation,
            "generated_text": prompt + continuation,
            "generated_token_ids": tokens,
            "metrics": {
                "backend_id": "frontierwright-reference-pytorch-v1",
                "generated_tokens": count,
            },
        }

    monkeypatch.setattr(service_module, "run_structured_command", fake_backend)
    prompt = "PRIVATE PROMPT SENTINEL"
    view = generate_reference_text(
        project,
        prompt=prompt,
        python_executable="fixture-python",
        max_new_tokens=3,
        temperature=0.0,
        seed=7,
        device="cpu",
    )

    assert view.prompt == prompt
    assert view.continuation_text == "AAA"
    assert view.generated_text == prompt + "AAA"
    assert view.generated_token_ids == [65, 65, 65]
    request = captured["request"]
    assert isinstance(request, dict)
    assert request["prompt"] == prompt
    request_path = captured["request_path"]
    assert isinstance(request_path, Path)
    assert not request_path.exists()

    history_text = json.dumps(
        Registry(project).read().history,
        sort_keys=True,
        ensure_ascii=False,
    )
    assert prompt not in history_text


def test_generation_rejects_non_reference_huggingface_model(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    model = tmp_path / "ordinary-hf"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"ordinary"}',
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    import_local_model(
        project,
        model,
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Ordinary",
    )

    with pytest.raises(FrontierwrightError, match="not a Frontierwright reference"):
        generate_reference_text(
            project,
            prompt="hello",
            python_executable="fixture-python",
            max_new_tokens=1,
        )


def test_operate_generate_json_machine_surface(
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
        prompt = str(request["prompt"])
        count = int(request["config"]["max_new_tokens"])
        return {
            "schema_version": 1,
            "ok": True,
            "operation": "generate",
            "prompt": prompt,
            "continuation_text": "!" * count,
            "generated_text": prompt + ("!" * count),
            "generated_token_ids": [33] * count,
            "metrics": {"generated_tokens": count, "device": "cpu"},
        }

    monkeypatch.setattr(service_module, "run_structured_command", fake_backend)
    result = runner.invoke(
        app,
        [
            "operate",
            "generate",
            "hello",
            "--path",
            str(project),
            "--python",
            "fixture-python",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0",
            "--device",
            "cpu",
            "--json",
            "--non-interactive",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["intervention_id"] == "frontierwright.operate.generate-reference"
    assert payload["prompt"] == "hello"
    assert payload["continuation_text"] == "!!"
    assert payload["generated_token_ids"] == [33, 33]
