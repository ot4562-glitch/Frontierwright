import json
from pathlib import Path

import pytest

import frontierwright.service as service_module
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.registry import Registry
from frontierwright.service import birth_zero_model, get_birth_view, get_status


def fake_birth_backend(
    argv_template: tuple[str, ...],
    *,
    environment_overrides: dict[str, str],
    request_path: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    del argv_template, environment_overrides, timeout_seconds
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = Path(str(request["output_root"])) / "model"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(
            {
                "model_type": "frontierwright_byte_causal_lm",
                "preset": request["preset"],
                "parameter_count": 8_000_000,
            }
        ),
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    weights = f"root:{request['preset']}:{request['seed']}".encode()
    (output / "model.safetensors").write_bytes(weights)
    return {
        "schema_version": 1,
        "ok": True,
        "operation": "birth",
        "output_model_path": str(output),
        "metrics": {
            "backend_id": "frontierwright-reference-pytorch-v1",
            "preset": request["preset"],
            "seed": request["seed"],
            "parameter_count": 8_000_000,
            "python_version": "fixture",
            "torch_version": "fixture",
            "device": "cpu",
            "trained_steps": 0,
        },
    }


def test_zero_birth_materializes_root_and_becomes_current_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    monkeypatch.setattr(service_module, "run_structured_command", fake_birth_backend)

    born = birth_zero_model(
        project,
        preset="zero-8m",
        seed=42,
        python_executable="fixture-python",
    )

    assert born.born is True
    assert born.preset == "zero-8m"
    assert born.seed == 42
    assert born.parameter_count == 8_000_000
    assert born.model_id is not None
    assert born.model_fingerprint is not None
    assert born.checkpoint is not None
    assert Path(born.checkpoint).is_dir()

    state = Registry(project).read()
    assert state.champion is not None
    assert state.champion.model.model_id == born.model_id
    assert state.champion.model.parent_model_id is None
    assert state.champion.model.stats == ()

    birth_record = Registry(project).get_model_birth(born.model_id)
    assert birth_record is not None
    assert birth_record["preset"] == "zero-8m"
    assert birth_record["seed"] == 42

    status = get_status(project)
    assert status.champion_model_id == born.model_id
    assert status.measurement_state == "NOT_READY"
    assert status.build_mode == "INTENT"

    history = Registry(project).read().history
    assert any(
        event["kind"] == "MODEL_BORN"
        and event["details"]["model_id"] == born.model_id
        for event in history
    )


def test_same_birth_request_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    calls = 0

    def counted_backend(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_birth_backend(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", counted_backend)

    first = birth_zero_model(
        project,
        preset="zero-8m",
        seed=7,
        python_executable="fixture-python",
    )
    second = birth_zero_model(
        project,
        preset="zero-8m",
        seed=7,
        python_executable="fixture-python",
    )

    assert first == second
    assert calls == 1


def test_second_different_birth_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    monkeypatch.setattr(service_module, "run_structured_command", fake_birth_backend)

    birth_zero_model(
        project,
        preset="zero-8m",
        seed=1,
        python_executable="fixture-python",
    )

    with pytest.raises(FrontierwrightError, match="already has a materialized current model"):
        birth_zero_model(
            project,
            preset="zero-8m",
            seed=2,
            python_executable="fixture-python",
        )


def test_birth_is_zero_origin_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("Imported", ModelOrigin.IMPORTED_LOCAL)
    monkeypatch.setattr(service_module, "run_structured_command", fake_birth_backend)

    with pytest.raises(FrontierwrightError, match="only valid for ZERO-origin"):
        birth_zero_model(
            project,
            preset="zero-8m",
            seed=42,
            python_executable="fixture-python",
        )

    assert get_birth_view(project).born is False
