import json
import math
from pathlib import Path

import pytest

import frontierwright.service as service_module
from frontierwright.data import DatasetRole
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import REFERENCE_LM_PACK
from frontierwright.models import inspect_local_model
from frontierwright.registry import Registry
from frontierwright.service import (
    add_local_dataset,
    compare_candidate,
    compare_candidate_evaluation,
    import_local_model,
    promote_candidate,
    quantize_reference_model,
)


def make_hf_model(root: Path) -> Path:
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
    (root / "model.safetensors").write_bytes(b"full-precision-parent")
    return root


def fake_quantize_backend(
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
    (output / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "format": "frontierwright-symmetric-int8-v1",
        "backend_id": "frontierwright-reference-pytorch-v1",
        "preset": "zero-8m",
        "parameter_count": 8_000_000,
        "source_tensor_bytes": 32_000_000,
        "quantized_tensor_bytes": 8_000_000,
        "quantized_tensor_count": 10,
        "tensor_metadata": {},
    }
    (output / "frontierwright-quantized.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    (output / "quantized_model.pt").write_bytes(b"int8-packed-fixture")
    transform = {
        "schema_version": 1,
        "intervention_id": "frontierwright.optimize.symmetric-int8",
        "intervention_version": "1",
        "method": "symmetric_int8_post_training_quantization",
        "parent": request["parent"],
        "backend_id": "frontierwright-reference-pytorch-v1",
        "preset": "zero-8m",
        "parameter_count": 8_000_000,
        "source_tensor_bytes": 32_000_000,
        "quantized_tensor_bytes": 8_000_000,
        "quantized_file_bytes": 8_100_000,
        "tensor_storage_ratio": 0.25,
    }
    (output / "frontierwright-transform.json").write_text(
        json.dumps(transform, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "ok": True,
        "operation": "quantize",
        "output_model_path": str(output),
        "metrics": transform,
    }


def setup_project(tmp_path: Path) -> tuple[Path, Registry, str]:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_hf_model(tmp_path / "parent"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Quantize",
    )
    registry = Registry(project)
    champion_id = registry.read().project["champion_id"]
    assert isinstance(champion_id, str)
    return project, registry, champion_id


def test_quantize_creates_nontrainable_pending_candidate_and_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, registry, champion_id = setup_project(tmp_path)
    monkeypatch.setattr(service_module, "run_structured_command", fake_quantize_backend)

    view = quantize_reference_model(
        project,
        python_executable="fixture-python",
    )

    assert view.replayed is False
    assert view.source_model_id == champion_id
    assert view.candidate_model_id is not None
    assert view.model_format == "FRONTIERWRIGHT_QUANTIZED"
    assert view.trainable is False
    assert view.metrics["tensor_storage_ratio"] == 0.25

    candidate = registry.get_candidate(view.candidate_model_id)
    assert candidate.status.value == "PENDING"
    assert candidate.model.parent_model_id == champion_id
    assert candidate.model.trainable is False
    assert candidate.model.model_format.value == "FRONTIERWRIGHT_QUANTIZED"

    lineage = registry.get_model_lineage(candidate.model.model_id)
    transformed = [
        item for item in lineage if item["relation"] == "TRANSFORMED_FROM"
    ]
    assert len(transformed) == 1
    assert transformed[0]["parent_model_id"] == champion_id
    assert transformed[0]["details"]["weight"] == 1.0
    assert transformed[0]["details"]["intervention_id"] == (
        "frontierwright.optimize.symmetric-int8"
    )

    artifact = registry.get_model_artifact(candidate.model.model_id)
    assert artifact is not None
    assert artifact["trainable"] is False
    files = {item["relative_path"] for item in artifact["files"]}
    assert {
        "config.json",
        "tokenizer.json",
        "frontierwright-quantized.json",
        "frontierwright-transform.json",
        "quantized_model.pt",
    }.issubset(files)


def test_identical_quantization_replays_without_backend_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _, _ = setup_project(tmp_path)
    calls = 0

    def counted_backend(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_quantize_backend(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", counted_backend)

    first = quantize_reference_model(
        project,
        python_executable="fixture-python",
    )
    second = quantize_reference_model(
        project,
        python_executable="fixture-python",
    )

    assert calls == 1
    assert first.candidate_model_id == second.candidate_model_id
    assert first.replayed is False
    assert second.replayed is True


def test_quantized_artifact_tampering_blocks_compare_and_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, registry, _ = setup_project(tmp_path)
    monkeypatch.setattr(service_module, "run_structured_command", fake_quantize_backend)

    view = quantize_reference_model(
        project,
        python_executable="fixture-python",
    )
    assert view.candidate_model_id is not None
    candidate = registry.get_candidate(view.candidate_model_id)
    metadata = Path(candidate.model.checkpoint) / "frontierwright-quantized.json"
    metadata.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(FrontierwrightError, match="artifact"):
        compare_candidate(project, candidate.model.model_id)

    with pytest.raises(FrontierwrightError, match="artifact"):
        promote_candidate(
            project,
            candidate.model.model_id,
            allow_unmeasured=True,
        )




def test_quantization_rejects_untrained_birth_root(tmp_path: Path) -> None:
    project = tmp_path / "birth-project"
    registry = Registry(project)
    registry.initialize("Academy", ModelOrigin.ZERO)
    state = registry.read()
    source = make_hf_model(tmp_path / "birth-root")
    descriptor = inspect_local_model(source)
    model = ModelState(
        model_id="model-birth-root",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint=str(descriptor.source_path),
        fingerprint=descriptor.fingerprint,
        model_format=descriptor.model_format,
        trainable=descriptor.trainable,
    )
    registry.register_birth_model(
        model,
        descriptor,
        preset="zero-8m",
        seed=42,
        backend_id="fixture-birth",
        runtime={"trained_steps": 0, "parameter_count": 8_000_000},
    )

    with pytest.raises(FrontierwrightError, match="initial pretraining"):
        quantize_reference_model(
            project,
            python_executable="fixture-python",
        )


def test_quantization_rejects_nontrainable_champion(tmp_path: Path) -> None:
    project = tmp_path / "gguf-project"
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF-quantize-fixture")
    import_local_model(
        project,
        gguf,
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="GGUF",
    )

    with pytest.raises(FrontierwrightError, match="full trainable model checkpoint"):
        quantize_reference_model(
            project,
            python_executable="fixture-python",
        )
