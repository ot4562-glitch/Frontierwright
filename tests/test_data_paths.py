from pathlib import Path

from frontierwright.data import DatasetRole, inspect_local_dataset
from frontierwright.domain import HistoryConfidence, ModelOrigin, ModelState
from frontierwright.models import inspect_local_model
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


def materialize_zero_birth(registry: Registry, root: Path) -> ModelState:
    source = fake_hf_model(root)
    descriptor = inspect_local_model(source)
    state = registry.read()
    model = ModelState(
        model_id="zero-root",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint=str(descriptor.source_path),
        fingerprint=descriptor.fingerprint,
        parent_model_id=None,
        stats=(),
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
    return model


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


def test_zero_model_pretraining_requires_birth_even_with_user_data(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("ZERO", ModelOrigin.ZERO)

    before = get_paths_view(project)
    path = by_id(before, "FROM_SCRATCH_PRETRAINING")
    assert path["availability"] == "LOCKED"
    assert "zero-model birth required before from-scratch pretraining" in path["blockers"]
    assert "pretraining dataset required" in path["blockers"]

    add_local_dataset(
        project,
        make_dataset(tmp_path / "pretrain"),
        name="My pretrain data",
        role=DatasetRole.PRETRAIN,
    )

    after = get_paths_view(project)
    path = by_id(after, "FROM_SCRATCH_PRETRAINING")
    assert path["availability"] == "LOCKED"
    assert path["blockers"] == ["zero-model birth required before from-scratch pretraining"]


def test_zero_birth_plus_pretrain_data_unlocks_from_scratch_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry = Registry(project)
    registry.initialize("ZERO", ModelOrigin.ZERO)
    add_local_dataset(
        project,
        make_dataset(tmp_path / "pretrain"),
        name="My pretrain data",
        role=DatasetRole.PRETRAIN,
    )

    materialize_zero_birth(registry, tmp_path / "birth-root")

    path = by_id(get_paths_view(project), "FROM_SCRATCH_PRETRAINING")
    assert path["availability"] == "PLANNABLE"
    assert path["blockers"] == []
    assert path["intervention_family"] == "LEARN"
    assert path["intervention_id"] == "frontierwright.learn.pretrain"


def test_zero_birth_root_cannot_skip_initial_pretraining_into_sft(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry = Registry(project)
    registry.initialize("ZERO", ModelOrigin.ZERO)
    materialize_zero_birth(registry, tmp_path / "birth-root")
    add_local_dataset(
        project,
        make_dataset(tmp_path / "sft"),
        name="SFT",
        role=DatasetRole.SFT,
    )

    view = get_paths_view(project)
    for path_id in ("FULL_SFT", "LORA_SFT", "QLORA_SFT"):
        item = by_id(view, path_id)
        assert item["availability"] == "LOCKED"
        assert (
            "born root has not completed initial pretraining; "
            "fine-tuning requires a trained descendant"
        ) in item["blockers"]


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


def test_preference_data_unlocks_dpo_without_unlocking_sft_paths(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        fake_hf_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Imported",
    )
    preference = tmp_path / "preference"
    preference.mkdir()
    (preference / "pairs.jsonl").write_text(
        '{"prompt":"Q: 1+1? A:","chosen":" 2","rejected":" 3"}\n',
        encoding="utf-8",
    )
    add_local_dataset(
        project,
        preference,
        name="Preference pairs",
        role=DatasetRole.PREFERENCE,
    )

    view = get_paths_view(project)
    dpo = by_id(view, "DPO")
    assert dpo["availability"] == "PLANNABLE"
    assert dpo["blockers"] == []
    assert dpo["intervention_family"] == "ALIGN"
    assert dpo["intervention_id"] == "frontierwright.align.dpo"

    for path_id in ("FULL_SFT", "LORA_SFT", "QLORA_SFT"):
        item = by_id(view, path_id)
        assert item["availability"] == "LOCKED"
        assert "SFT dataset required" in item["blockers"]
