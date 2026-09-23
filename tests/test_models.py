import json
from pathlib import Path

import pytest

from frontierwright.domain import HistoryConfidence, ModelFormat, ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.models import (
    discover_history_evidence,
    inspect_local_model,
    sha256_file,
)
from frontierwright.service import import_local_model


def fake_hf_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"model_type": "test", "hidden_size": 8}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text(
        json.dumps({"version": "1.0"}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"frontierwright-test-weights")
    return model


def evidence_item(root: Path, name: str, content: str) -> dict[str, str]:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return {"path": name, "sha256": f"sha256:{sha256_file(path)}"}


def test_huggingface_fingerprint_is_content_stable_and_sensitive(tmp_path: Path) -> None:
    model = fake_hf_model(tmp_path)
    first = inspect_local_model(model)
    second = inspect_local_model(model)

    assert first.model_format is ModelFormat.HUGGINGFACE
    assert first.trainable is True
    assert first.fingerprint == second.fingerprint
    assert first.total_bytes > 0

    (model / "model.safetensors").write_bytes(b"frontierwright-test-weights-changed")
    changed = inspect_local_model(model)
    assert changed.fingerprint != first.fingerprint


def test_training_metadata_is_partial_not_verified(tmp_path: Path) -> None:
    model = fake_hf_model(tmp_path)
    (model / "trainer_state.json").write_text("{}", encoding="utf-8")

    descriptor = inspect_local_model(model)
    evidence = discover_history_evidence(descriptor)

    assert evidence.confidence is HistoryConfidence.PARTIAL
    assert "trainer_state.json" in evidence.evidence_files


def test_verified_lineage_requires_hash_chain_and_evidence_files(tmp_path: Path) -> None:
    model = fake_hf_model(tmp_path)
    descriptor = inspect_local_model(model)

    config = evidence_item(model, "step-config.json", '{"lr": 1e-5}')
    dataset = evidence_item(model, "dataset-manifest.json", '{"fingerprint": "data-1"}')
    receipt = evidence_item(model, "run-receipt.json", '{"run": "run-1"}')

    manifest = {
        "schema_version": 1,
        "current_model_fingerprint": descriptor.fingerprint,
        "lineage": [
            {
                "input_fingerprint": "sha256:parent",
                "output_fingerprint": descriptor.fingerprint,
                "intervention": "SFT",
                "config": config,
                "dataset_manifest": dataset,
                "run_receipt": receipt,
            }
        ],
    }
    manifest_path = model / "frontierwright-lineage.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = discover_history_evidence(descriptor)
    assert evidence.confidence is HistoryConfidence.VERIFIED
    assert evidence.reason.startswith("Lineage chain")

    (model / "run-receipt.json").write_text('{"run": "tampered"}', encoding="utf-8")
    with pytest.raises(FrontierwrightError, match="hash mismatch"):
        discover_history_evidence(descriptor)


def test_imported_model_becomes_current_unmeasured_champion(tmp_path: Path) -> None:
    model = fake_hf_model(tmp_path / "source")
    project = tmp_path / "project"

    view = import_local_model(
        project,
        model,
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="NOVA",
    )

    assert view.champion_model_id is not None
    assert view.model_format == "HUGGINGFACE"
    assert view.trainable is True
    assert view.measurement_state == "NOT_READY"
    assert view.build_mode == "NOT_READY"
    assert view.history_confidence == "UNKNOWN"
    assert view.strong_recommendation_allowed is False
    assert view.model_fingerprint is not None


def test_gguf_is_inspectable_but_not_trainable(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF-test")

    descriptor = inspect_local_model(gguf)
    assert descriptor.model_format is ModelFormat.GGUF
    assert descriptor.trainable is False

    view = import_local_model(
        tmp_path / "project",
        gguf,
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="GGUF",
    )
    assert view.trainable is False
    assert view.model_format == "GGUF"
