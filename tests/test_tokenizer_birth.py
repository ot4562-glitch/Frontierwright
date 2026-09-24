import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.data import DatasetRole
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.registry import Registry
from frontierwright.service import (
    add_local_dataset,
    get_tokenizers_view,
    train_project_tokenizer,
)

runner = CliRunner()


def setup_zero_with_pretrain(tmp_path: Path) -> tuple[Path, Path, str]:
    project = tmp_path / "project"
    Registry(project).initialize("ACADEMY", ModelOrigin.ZERO)
    data = tmp_path / "corpus"
    data.mkdir()
    (data / "train.txt").write_text(
        ("banana bandana frontierwright academy\n" * 40),
        encoding="utf-8",
    )
    view = add_local_dataset(
        project,
        data,
        name="Birth corpus",
        role=DatasetRole.PRETRAIN,
    )
    dataset_id = str(view.datasets[0]["dataset_id"])
    return project, data, dataset_id


def test_train_tokenizer_is_managed_and_idempotent(tmp_path: Path) -> None:
    project, _, dataset_id = setup_zero_with_pretrain(tmp_path)

    first = train_project_tokenizer(
        project,
        dataset_id=dataset_id,
        vocab_size=280,
        max_training_bytes=4096,
    )
    second = train_project_tokenizer(
        project,
        dataset_id=dataset_id,
        vocab_size=280,
        max_training_bytes=4096,
    )

    assert first.artifact_id is not None
    assert first.fingerprint is not None
    assert first.replayed is False
    assert second.artifact_id == first.artifact_id
    assert second.fingerprint == first.fingerprint
    assert second.replayed is True
    assert first.vocab_size is not None and first.vocab_size > 256
    assert first.merge_count == first.vocab_size - 256
    assert first.path is not None
    path = Path(first.path)
    assert path.is_file()
    assert path.parent.parent.name == "tokenizers"

    listed = get_tokenizers_view(project)
    assert len(listed.tokenizers) == 1
    assert listed.tokenizers[0]["artifact_id"] == first.artifact_id

    registry = Registry(project)
    rows = registry.list_tokenizer_artifacts()
    assert len(rows) == 1
    assert rows[0]["source_dataset_id"] == dataset_id
    assert any(event["kind"] == "TOKENIZER_TRAINED" for event in registry.read().history)


def test_tokenizer_birth_rejects_source_dataset_drift(tmp_path: Path) -> None:
    project, data, dataset_id = setup_zero_with_pretrain(tmp_path)
    (data / "train.txt").write_text("mutated after registration\n", encoding="utf-8")

    with pytest.raises(FrontierwrightError, match="no longer match"):
        train_project_tokenizer(
            project,
            dataset_id=dataset_id,
            vocab_size=272,
            max_training_bytes=4096,
        )


def test_tokenizer_birth_requires_pretrain_role(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("ACADEMY", ModelOrigin.ZERO)
    data = tmp_path / "sft"
    data.mkdir()
    (data / "train.txt").write_text("instruction answer\n" * 10, encoding="utf-8")
    view = add_local_dataset(
        project,
        data,
        name="SFT",
        role=DatasetRole.SFT,
    )

    with pytest.raises(FrontierwrightError, match="PRETRAIN"):
        train_project_tokenizer(
            project,
            dataset_id=str(view.datasets[0]["dataset_id"]),
            vocab_size=272,
        )


def test_tokenizer_change_is_blocked_after_model_birth_state_exists(tmp_path: Path) -> None:
    project, _, dataset_id = setup_zero_with_pretrain(tmp_path)
    registry = Registry(project)
    state = registry.read()
    model = ModelState(
        model_id="already-born",
        identity_id=str(state.project["identity_id"]),
        origin=ModelOrigin.ZERO,
        checkpoint="fixture",
        fingerprint="sha256:already-born",
    )
    registry.register_candidate(model)
    registry.promote_candidate(model.model_id)

    with pytest.raises(FrontierwrightError, match="before materializing"):
        train_project_tokenizer(
            project,
            dataset_id=dataset_id,
            vocab_size=272,
        )


def test_birth_tokenizer_cli_json_surface(tmp_path: Path) -> None:
    project, _, dataset_id = setup_zero_with_pretrain(tmp_path)

    created = runner.invoke(
        app,
        [
            "birth",
            "tokenizer",
            dataset_id,
            "--path",
            str(project),
            "--vocab-size",
            "276",
            "--max-training-bytes",
            "4096",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert created.exit_code == 0, created.output
    payload = json.loads(created.stdout)
    assert payload["ok"] is True
    assert payload["intervention_id"] == "frontierwright.birth.train-tokenizer"
    assert payload["source_dataset_id"] == dataset_id
    assert payload["vocab_size"] > 256

    listed = runner.invoke(
        app,
        [
            "birth",
            "tokenizers",
            "--path",
            str(project),
            "--json",
            "--non-interactive",
        ],
    )
    assert listed.exit_code == 0, listed.output
    list_payload = json.loads(listed.stdout)
    assert list_payload["ok"] is True
    assert len(list_payload["tokenizers"]) == 1
    assert list_payload["tokenizers"][0]["artifact_id"] == payload["artifact_id"]
