import json
import socket
from pathlib import Path

import pytest

import frontierwright.service as service_module
from frontierwright.data import DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.evaluations import REFERENCE_LM_PACK
from frontierwright.service import (
    add_local_dataset,
    get_data_view,
    get_paths_view,
    get_status,
    import_local_model,
    run_evaluation_pack,
)


def _reference_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "frontierwright_byte_causal_lm",
                "frontierwright_reference_backend": "frontierwright-reference-pytorch-v1",
                "preset": "zero-8m",
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    (root / "pytorch_model.bin").write_bytes(b"offline-reference-model")
    return root


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "data.txt").write_text("offline local data\n" * 32, encoding="utf-8")
    return root


def test_core_local_workflow_operates_with_network_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Frontierwright core attempted a network operation")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", network_forbidden)

    project = tmp_path / "project"
    import_local_model(
        project,
        _reference_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="OfflineStudio",
    )
    add_local_dataset(
        project,
        _dataset(tmp_path / "dataset"),
        name="Private local data",
        role=DatasetRole.SFT,
    )

    status = get_status(project)
    data = get_data_view(project)
    paths = get_paths_view(project)

    assert status.initialized is True
    assert status.project_name == "OfflineStudio"
    assert status.edition_profile == "STUDIO"
    assert len(data.datasets) == 1
    assert data.datasets[0]["classification"] == "PRIVATE"
    assert {item["path_id"] for item in paths.paths} >= {
        "FULL_SFT",
        "LORA_SFT",
        "QLORA_SFT",
    }


def test_evaluation_backend_request_excludes_game_and_project_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        _reference_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="NOVA-character-name-must-not-leak",
    )
    data_view = add_local_dataset(
        project,
        _dataset(tmp_path / "eval"),
        name="Held out",
        role=DatasetRole.PRETRAIN,
    )
    dataset_id = str(data_view.datasets[0]["dataset_id"])

    captured: dict[str, object] = {}

    def fake_evaluator(
        argv_template: tuple[str, ...],
        *,
        environment_overrides: dict[str, str],
        request_path: Path,
        timeout_seconds: float,
    ) -> dict[str, object]:
        del argv_template, environment_overrides, timeout_seconds
        request = json.loads(request_path.read_text(encoding="utf-8"))
        captured.update(request)
        return {
            "schema_version": 1,
            "ok": True,
            "operation": "evaluate",
            "metrics": {
                "backend_id": "frontierwright-reference-pytorch-v1",
                "preset": "zero-8m",
                "device": "cpu",
                "cross_entropy_nats_per_token": 4.0,
                "perplexity": 54.598150033144236,
                "tokens_evaluated": 128,
                "windows_evaluated": 1,
                "batches_evaluated": 1,
                "elapsed_seconds": 0.1,
                "tokens_per_second": 1280.0,
                "parameter_count": 7_521_280,
                "python_version": "fixture",
                "torch_version": "fixture",
            },
        }

    monkeypatch.setattr(service_module, "run_structured_command", fake_evaluator)

    run_evaluation_pack(
        project,
        pack_id=REFERENCE_LM_PACK.pack_id,
        dataset_id=dataset_id,
        python_executable="fixture-python",
        device="cpu",
        batch_size=1,
        max_batches=1,
    )

    assert set(captured) == {
        "schema_version",
        "backend_id",
        "operation",
        "model_source_path",
        "dataset_source_path",
        "config",
    }
    serialized = json.dumps(captured, sort_keys=True).lower()
    for forbidden in (
        "nova-character-name-must-not-leak",
        "nickname",
        "archetype",
        "build",
        "target",
        "floor",
        "edition",
        "character",
        "history_confidence",
    ):
        assert forbidden not in serialized
