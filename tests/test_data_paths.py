from pathlib import Path

from frontierwright.data import DatasetRole, inspect_local_dataset
from frontierwright.domain import HistoryConfidence, ModelOrigin
from frontierwright.registry import Registry
from frontierwright.service import (
    add_local_dataset,
    get_data_view,
    get_paths_view,
    import_local_model,
)


def fake_hf_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"trainable-weights")
    return root


def make_dataset(root: Path, content: str = "sample\n") -> Path:
    root.mkdir(parents=True)
    (root / "data.jsonl").write_text(content, encoding="utf-8")
    return root


def by_id(view, path_id: str) -> dict[str, object]:
    return next(item for item in view.paths if item["path_id"] == path_id)


def test_dataset_fingerprint_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "data", "alpha\n")
    first = inspect_local_dataset(source)
    second = inspect_local_dataset(source)

    assert first.fingerprint == second.fingerprint
    assert first.file_count == 1

    (source / "data.jsonl").write_text("beta\n", encoding="utf-8")
    changed = inspect_local_dataset(source)
    assert changed.fingerprint != first.fingerprint


def test_same_local_dataset_role_is_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("ZERO", ModelOrigin.ZERO)
    source = make_dataset(tmp_path / "data")

    add_local_dataset(
        project,
        source,
        name="Pretrain",
        role=DatasetRole.PRETRAIN,
        license_name="Apache-2.0",
    )
    add_local_dataset(
        project,
        source,
        name="Duplicate name should not duplicate content",
        role=DatasetRole.PRETRAIN,
    )

    view = get_data_view(project)
    assert len(view.datasets) == 1
    assert view.datasets[0]["role"] == "PRETRAIN"
    assert view.datasets[0]["provenance"] == "LOCAL_USER"


def test_zero_model_pretraining_unlocks_to_plannable_with_user_data(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("ZERO", ModelOrigin.ZERO)

    before = get_paths_view(project)
    path = by_id(before, "FROM_SCRATCH_PRETRAINING")
    assert path["availability"] == "LOCKED"
    assert "pretraining dataset required" in path["blockers"]

    add_local_dataset(
        project,
        make_dataset(tmp_path / "pretrain"),
        name="My pretrain data",
        role=DatasetRole.PRETRAIN,
    )

    after = get_paths_view(project)
    path = by_id(after, "FROM_SCRATCH_PRETRAINING")
    assert path["availability"] == "PLANNABLE"
    assert path["blockers"] == []
    assert "training backend implementation" in path["next_checks"]
    assert "representative calibration" in path["next_checks"]


def test_sft_data_unlocks_trainable_imported_sft_paths_only_to_plannable(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        fake_hf_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Imported",
    )
    add_local_dataset(
        project,
        make_dataset(tmp_path / "sft"),
        name="My SFT",
        role=DatasetRole.SFT,
    )

    view = get_paths_view(project)
    for path_id in ("FULL_SFT", "LORA_SFT", "QLORA_SFT"):
        item = by_id(view, path_id)
        assert item["availability"] == "PLANNABLE"
        assert item["blockers"] == []
    assert by_id(view, "CONTINUED_PRETRAINING")["availability"] == "LOCKED"
    assert view.recommended_path is None
    assert "COMPLETE or VERIFIED" in (view.recommendation_reason or "")


def test_non_trainable_gguf_stays_locked_even_with_sft_data(tmp_path: Path) -> None:
    project = tmp_path / "project"
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF-fixture")
    import_local_model(
        project,
        gguf,
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="GGUF",
    )
    add_local_dataset(
        project,
        make_dataset(tmp_path / "sft"),
        name="SFT",
        role=DatasetRole.SFT,
    )

    view = get_paths_view(project)
    lora = by_id(view, "LORA_SFT")
    assert lora["availability"] == "LOCKED"
    assert "trainable model representation required" in lora["blockers"]


def test_verified_history_enables_recommendation_eligibility_but_no_fake_winner(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        fake_hf_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Imported",
    )
    Registry(project).set_history_confidence(HistoryConfidence.VERIFIED)
    add_local_dataset(
        project,
        make_dataset(tmp_path / "sft"),
        name="SFT",
        role=DatasetRole.SFT,
    )

    view = get_paths_view(project)
    lora = by_id(view, "LORA_SFT")
    assert lora["recommendation_eligible"] is True
    assert view.recommended_path is None
    assert "No effect/recommendation model" in (view.recommendation_reason or "")
