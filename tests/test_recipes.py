import json
import sys
from pathlib import Path

import pytest

from frontierwright.data import DatasetClassification, DatasetRole
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import HardBudgets, PermissionLevel
from frontierwright.paths import TrainingPathId
from frontierwright.recipes import (
    BYTE_SHARDS_PLUGIN_ID,
    PREFERENCE_JSONL_PLUGIN_ID,
    TEXT_LINES_PLUGIN_ID,
)
from frontierwright.registry import Registry
from frontierwright.service import (
    add_local_dataset,
    create_training_plan,
    get_data_view,
    import_local_model,
    prepare_dataset,
    prepare_dataset_mixture,
    prepare_dataset_snapshot,
)


def make_dataset(root: Path, text: str = "alpha\nbeta\n") -> Path:
    root.mkdir(parents=True)
    (root / "train.txt").write_text(text, encoding="utf-8")
    return root


def make_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def write_backend_spec(root: Path) -> Path:
    root.mkdir(parents=True)
    script = root / "noop.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    spec = root / "backend.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": "recipe-test-backend",
                "supported_paths": ["LORA_SFT"],
                "calibrate_argv": [sys.executable, str(script), "{request_json}"],
                "train_argv": [sys.executable, str(script), "{request_json}"],
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    return spec


def test_snapshot_recipe_is_content_preserving_and_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    add_local_dataset(
        project,
        make_dataset(tmp_path / "data"),
        name="Raw pretrain",
        role=DatasetRole.PRETRAIN,
        license_name="Apache-2.0",
        language="en",
    )
    raw = get_data_view(project).datasets[0]

    first = prepare_dataset_snapshot(
        project,
        dataset_id=str(raw["dataset_id"]),
        name="Frozen pretrain",
    )
    second = prepare_dataset_snapshot(
        project,
        dataset_id=str(raw["dataset_id"]),
        name="Ignored duplicate name",
    )

    assert len(first.datasets) == 2
    assert len(second.datasets) == 2
    prepared = next(item for item in second.datasets if item["managed"] is True)
    assert prepared["fingerprint"] == raw["fingerprint"]
    assert prepared["source_dataset_id"] == raw["dataset_id"]
    assert prepared["provenance"] == "LOCAL_USER"
    assert prepared["license"] == "Apache-2.0"
    assert prepared["language"] == "en"
    assert isinstance(prepared["preparation_recipe_id"], str)
    assert str(prepared["preparation_recipe_hash"]).startswith("sha256:")
    assert Path(str(prepared["source_path"])).is_dir()
    assert (
        Path(str(prepared["source_path"])).read_bytes()
        if Path(str(prepared["source_path"])).is_file()
        else None
    ) is None

    recipe = Registry(project).get_data_recipe(str(prepared["preparation_recipe_id"]))
    assert recipe is not None
    assert recipe["plugin_id"] == "frontierwright.data.snapshot-copy"
    assert recipe["plugin_version"] == "1"
    assert recipe["source_dataset_id"] == raw["dataset_id"]
    assert recipe["source_fingerprint"] == raw["fingerprint"]
    assert recipe["config"] == {"mode": "byte_preserving_snapshot"}


def test_source_drift_blocks_snapshot_preparation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    source = make_dataset(tmp_path / "data", "original\n")
    add_local_dataset(
        project,
        source,
        name="Raw",
        role=DatasetRole.PRETRAIN,
    )
    raw = get_data_view(project).datasets[0]

    (source / "train.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(FrontierwrightError, match="content changed"):
        prepare_dataset_snapshot(project, dataset_id=str(raw["dataset_id"]))


def test_plan_prefers_single_prepared_dataset_and_pins_recipe(tmp_path: Path) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="NOVA",
    )
    add_local_dataset(
        project,
        make_dataset(tmp_path / "sft"),
        name="Raw SFT",
        role=DatasetRole.SFT,
    )
    raw = get_data_view(project).datasets[0]
    prepared_view = prepare_dataset_snapshot(
        project,
        dataset_id=str(raw["dataset_id"]),
        name="Prepared SFT",
    )
    prepared = next(item for item in prepared_view.datasets if item["managed"] is True)

    plan = create_training_plan(
        project,
        path_id=TrainingPathId.LORA_SFT,
        backend_spec_path=write_backend_spec(tmp_path / "backend"),
        dataset_id=None,
        permission=PermissionLevel.PLAN,
        budgets=HardBudgets(max_runs=1),
        config={"epochs": 1},
    )

    assert plan.dataset_id == prepared["dataset_id"]
    assert plan.dataset_recipe_id == prepared["preparation_recipe_id"]
    assert plan.dataset_recipe_hash == prepared["preparation_recipe_hash"]

    stored = Registry(project).get_plan(str(plan.plan_id))
    assert stored.dataset_id == prepared["dataset_id"]
    assert stored.dataset_recipe_id == prepared["preparation_recipe_id"]
    assert stored.dataset_recipe_hash == prepared["preparation_recipe_hash"]
    assert stored.request_payload()["dataset_recipe_hash"] == prepared["preparation_recipe_hash"]


def test_text_lines_recipe_normalizes_dedupes_and_is_idempotent(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)

    source = tmp_path / "text-data"
    source.mkdir()
    (source / "a.txt").write_text(" cafe\u0301 \nalpha\n\n", encoding="utf-8")
    (source / "b.txt").write_bytes("café\r\nalpha\r\n beta \r\n".encode())

    add_local_dataset(
        project,
        source,
        name="Raw text",
        role=DatasetRole.PRETRAIN,
        license_name="Apache-2.0",
        language="en",
    )
    raw = get_data_view(project).datasets[0]

    first = prepare_dataset(
        project,
        dataset_id=str(raw["dataset_id"]),
        plugin_id=TEXT_LINES_PLUGIN_ID,
        name="Normalized text",
    )
    second = prepare_dataset(
        project,
        dataset_id=str(raw["dataset_id"]),
        plugin_id=TEXT_LINES_PLUGIN_ID,
        name="Ignored duplicate name",
    )

    assert len(first.datasets) == 2
    assert len(second.datasets) == 2
    prepared = next(item for item in second.datasets if item["managed"] is True)
    assert prepared["source_dataset_id"] == raw["dataset_id"]
    assert prepared["classification"] == raw["classification"]
    assert prepared["fingerprint"] != raw["fingerprint"]

    prepared_root = Path(str(prepared["source_path"]))
    assert (prepared_root / "corpus.txt").read_text(encoding="utf-8") == (
        "café\nalpha\nbeta\n"
    )

    recipe = Registry(project).get_data_recipe(str(prepared["preparation_recipe_id"]))
    assert recipe is not None
    assert recipe["plugin_id"] == TEXT_LINES_PLUGIN_ID
    assert recipe["config"] == {
        "encoding": "utf-8",
        "unicode_normalization": "NFC",
        "line_endings": "LF",
        "strip_whitespace": True,
        "drop_empty": True,
        "dedupe": "stable_exact",
        "output": "corpus.txt",
    }

    manifest = json.loads(
        (prepared_root.parent / "recipe.json").read_text(encoding="utf-8")
    )
    assert manifest["transformation"] == {
        "input_files": 2,
        "input_lines": 6,
        "output_lines": 3,
        "empty_lines_removed": 1,
        "duplicate_lines_removed": 2,
        "output_bytes": len("café\nalpha\nbeta\n".encode()),
    }


def test_text_lines_recipe_config_changes_recipe_identity_and_output(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    add_local_dataset(
        project,
        make_dataset(tmp_path / "data", "alpha\nalpha\n"),
        name="Raw",
        role=DatasetRole.PRETRAIN,
    )
    raw = get_data_view(project).datasets[0]

    default_view = prepare_dataset(
        project,
        dataset_id=str(raw["dataset_id"]),
        plugin_id=TEXT_LINES_PLUGIN_ID,
        name="Deduped",
    )
    deduped = next(item for item in default_view.datasets if item["managed"] is True)

    no_dedupe_view = prepare_dataset(
        project,
        dataset_id=str(raw["dataset_id"]),
        plugin_id=TEXT_LINES_PLUGIN_ID,
        config={"dedupe": "none"},
        name="Not deduped",
    )
    prepared = [item for item in no_dedupe_view.datasets if item["managed"] is True]
    assert len(prepared) == 2
    no_dedupe = next(
        item
        for item in prepared
        if item["preparation_recipe_id"] != deduped["preparation_recipe_id"]
    )

    assert no_dedupe["preparation_recipe_hash"] != deduped["preparation_recipe_hash"]
    output = Path(str(no_dedupe["source_path"])) / "corpus.txt"
    assert output.read_text(encoding="utf-8") == "alpha\nalpha\n"


def test_text_lines_recipe_rejects_non_utf8_source(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    source = tmp_path / "binary-data"
    source.mkdir()
    (source / "bad.txt").write_bytes(b"\xff\xfe\x00")

    add_local_dataset(
        project,
        source,
        name="Bad text",
        role=DatasetRole.PRETRAIN,
    )
    raw = get_data_view(project).datasets[0]

    with pytest.raises(FrontierwrightError, match="requires UTF-8 input"):
        prepare_dataset(
            project,
            dataset_id=str(raw["dataset_id"]),
            plugin_id=TEXT_LINES_PLUGIN_ID,
        )


def test_weighted_text_mixture_binds_sources_parts_and_metadata(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)

    add_local_dataset(
        project,
        make_dataset(tmp_path / "raw-a", "alpha\nbeta\n"),
        name="Raw A",
        role=DatasetRole.PRETRAIN,
        classification=DatasetClassification.PUBLIC,
        license_name="Apache-2.0",
        domain="general",
        language="en",
    )
    raw_a = next(item for item in get_data_view(project).datasets if item["name"] == "Raw A")
    prepared_a_view = prepare_dataset(
        project,
        dataset_id=str(raw_a["dataset_id"]),
        plugin_id=TEXT_LINES_PLUGIN_ID,
        name="Prepared A",
    )
    prepared_a = next(
        item for item in prepared_a_view.datasets if item["name"] == "Prepared A"
    )

    add_local_dataset(
        project,
        make_dataset(tmp_path / "raw-b", "x\n"),
        name="Raw B",
        role=DatasetRole.PRETRAIN,
        classification=DatasetClassification.PRIVATE,
        license_name="MIT",
        domain="code",
        language="ko",
    )
    raw_b = next(item for item in get_data_view(project).datasets if item["name"] == "Raw B")
    prepared_b_view = prepare_dataset(
        project,
        dataset_id=str(raw_b["dataset_id"]),
        plugin_id=TEXT_LINES_PLUGIN_ID,
        name="Prepared B",
    )
    prepared_b = next(
        item for item in prepared_b_view.datasets if item["name"] == "Prepared B"
    )

    mixed_view = prepare_dataset_mixture(
        project,
        inputs=[
            (str(prepared_a["dataset_id"]), 2),
            (str(prepared_b["dataset_id"]), 1),
        ],
        name="Mixture",
        max_output_bytes=1024,
    )
    mixed = next(item for item in mixed_view.datasets if item["name"] == "Mixture")
    assert mixed["managed"] is True
    assert mixed["classification"] == "PRIVATE"
    assert mixed["license"] is None
    assert mixed["domain"] is None
    assert mixed["language"] is None
    assert mixed["token_count"] is None

    output = Path(str(mixed["source_path"])) / "corpus.txt"
    assert output.read_text(encoding="utf-8") == (
        "alpha\nbeta\nalpha\nbeta\nx\n"
    )

    recipe = Registry(project).get_data_recipe(str(mixed["preparation_recipe_id"]))
    assert recipe is not None
    assert recipe["plugin_id"] == "frontierwright.data.weighted-text-mixture"
    assert recipe["config"]["sources"] == [
        {
            "dataset_id": prepared_a["dataset_id"],
            "fingerprint": prepared_a["fingerprint"],
            "parts": 2,
        },
        {
            "dataset_id": prepared_b["dataset_id"],
            "fingerprint": prepared_b["fingerprint"],
            "parts": 1,
        },
    ]

    replay = prepare_dataset_mixture(
        project,
        inputs=[
            (str(prepared_a["dataset_id"]), 2),
            (str(prepared_b["dataset_id"]), 1),
        ],
        name="Ignored replay name",
        max_output_bytes=1024,
    )
    assert len(replay.datasets) == 5


def test_weighted_text_mixture_enforces_materialization_size_limit(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)

    prepared_ids: list[str] = []
    for index, text in enumerate(("alpha\n", "beta\n"), start=1):
        name = f"Raw {index}"
        add_local_dataset(
            project,
            make_dataset(tmp_path / f"raw-{index}", text),
            name=name,
            role=DatasetRole.PRETRAIN,
        )
        raw = next(
            item for item in get_data_view(project).datasets if item["name"] == name
        )
        prepared_name = f"Prepared {index}"
        view = prepare_dataset(
            project,
            dataset_id=str(raw["dataset_id"]),
            plugin_id=TEXT_LINES_PLUGIN_ID,
            name=prepared_name,
        )
        prepared = next(
            item for item in view.datasets if item["name"] == prepared_name
        )
        prepared_ids.append(str(prepared["dataset_id"]))

    with pytest.raises(FrontierwrightError, match="exceeding max_output_bytes"):
        prepare_dataset_mixture(
            project,
            inputs=[(prepared_ids[0], 1), (prepared_ids[1], 1)],
            max_output_bytes=3,
        )


def test_byte_shards_requires_prepared_text_and_records_exact_count(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    add_local_dataset(
        project,
        make_dataset(tmp_path / "raw", "café\nbeta\n"),
        name="Raw",
        role=DatasetRole.PRETRAIN,
    )
    raw = next(item for item in get_data_view(project).datasets if item["name"] == "Raw")

    with pytest.raises(FrontierwrightError, match="requires a managed prepared dataset"):
        prepare_dataset(
            project,
            dataset_id=str(raw["dataset_id"]),
            plugin_id=BYTE_SHARDS_PLUGIN_ID,
        )

    text_view = prepare_dataset(
        project,
        dataset_id=str(raw["dataset_id"]),
        plugin_id=TEXT_LINES_PLUGIN_ID,
        name="Text",
    )
    text = next(item for item in text_view.datasets if item["name"] == "Text")

    sharded_view = prepare_dataset(
        project,
        dataset_id=str(text["dataset_id"]),
        plugin_id=BYTE_SHARDS_PLUGIN_ID,
        config={"bytes_per_shard": 4},
        name="Byte shards",
    )
    sharded = next(item for item in sharded_view.datasets if item["name"] == "Byte shards")

    expected = "café\nbeta\n".encode()
    shard_root = Path(str(sharded["source_path"]))
    shard_paths = sorted(shard_root.glob("shard-*.bin"))
    assert [path.stat().st_size for path in shard_paths] == [4, 4, 3]
    assert b"".join(path.read_bytes() for path in shard_paths) == expected
    assert sharded["token_count"] == len(expected)

    recipe = Registry(project).get_data_recipe(str(sharded["preparation_recipe_id"]))
    assert recipe is not None
    manifest_path = shard_root.parent / "recipe.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["recipe_hash"] == recipe["recipe_hash"]
    assert manifest["transformation"]["representation"] == "uint8_byte_ids"
    assert manifest["transformation"]["byte_id_count"] == len(expected)
    assert manifest["transformation"]["shard_count"] == 3

    replay = prepare_dataset(
        project,
        dataset_id=str(text["dataset_id"]),
        plugin_id=BYTE_SHARDS_PLUGIN_ID,
        config={"bytes_per_shard": 4},
        name="Ignored replay name",
    )
    assert len(replay.datasets) == 3


def test_byte_shards_detects_published_artifact_tampering(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    add_local_dataset(
        project,
        make_dataset(tmp_path / "raw", "alpha\nbeta\n"),
        name="Raw",
        role=DatasetRole.PRETRAIN,
    )
    raw = next(item for item in get_data_view(project).datasets if item["name"] == "Raw")
    text_view = prepare_dataset(
        project,
        dataset_id=str(raw["dataset_id"]),
        plugin_id=TEXT_LINES_PLUGIN_ID,
        name="Text",
    )
    text = next(item for item in text_view.datasets if item["name"] == "Text")
    sharded_view = prepare_dataset(
        project,
        dataset_id=str(text["dataset_id"]),
        plugin_id=BYTE_SHARDS_PLUGIN_ID,
        config={"bytes_per_shard": 5},
        name="Byte shards",
    )
    sharded = next(item for item in sharded_view.datasets if item["name"] == "Byte shards")
    first_shard = sorted(Path(str(sharded["source_path"])).glob("shard-*.bin"))[0]
    first_shard.write_bytes(b"tampered")

    with pytest.raises(FrontierwrightError, match="recorded fingerprint"):
        prepare_dataset(
            project,
            dataset_id=str(text["dataset_id"]),
            plugin_id=BYTE_SHARDS_PLUGIN_ID,
            config={"bytes_per_shard": 5},
        )


def test_preference_jsonl_recipe_canonicalizes_dedupes_and_preserves_whitespace(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)

    source = tmp_path / "preference"
    source.mkdir()
    first = {
        "prompt": " cafe\u0301?",
        "chosen": " yes",
        "rejected": " no",
        "meta": {"source": 1},
    }
    duplicate = {
        "prompt": " café?",
        "chosen": " yes",
        "rejected": " no",
        "meta": {"source": 2},
    }
    second = {
        "prompt": "2+2?",
        "chosen": " 4",
        "rejected": " 5",
    }
    (source / "a.jsonl").write_text(
        json.dumps(first, ensure_ascii=False) + "\n\n",
        encoding="utf-8",
    )
    (source / "b.jsonl").write_text(
        json.dumps(duplicate, ensure_ascii=False)
        + "\n"
        + json.dumps(second, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    add_local_dataset(
        project,
        source,
        name="Raw preferences",
        role=DatasetRole.PREFERENCE,
    )
    raw = get_data_view(project).datasets[0]

    first_view = prepare_dataset(
        project,
        dataset_id=str(raw["dataset_id"]),
        plugin_id=PREFERENCE_JSONL_PLUGIN_ID,
        name="Canonical preferences",
    )
    second_view = prepare_dataset(
        project,
        dataset_id=str(raw["dataset_id"]),
        plugin_id=PREFERENCE_JSONL_PLUGIN_ID,
        name="Ignored duplicate name",
    )

    assert len(first_view.datasets) == 2
    assert len(second_view.datasets) == 2
    prepared = next(item for item in second_view.datasets if item["managed"] is True)
    assert prepared["role"] == "PREFERENCE"
    assert prepared["source_dataset_id"] == raw["dataset_id"]
    assert isinstance(prepared["preparation_recipe_hash"], str)

    data_root = Path(str(prepared["source_path"]))
    lines = (data_root / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["prompt"] == " café?"
    assert records[0]["chosen"] == " yes"
    assert records[0]["rejected"] == " no"
    assert records[0]["meta"] == {"source": 1}
    assert records[1]["chosen"] == " 4"

    recipe_id = str(prepared["preparation_recipe_id"])
    recipe = Registry(project).get_data_recipe(recipe_id)
    assert recipe is not None
    assert recipe["plugin_id"] == PREFERENCE_JSONL_PLUGIN_ID
    assert recipe["config"] == {
        "encoding": "utf-8",
        "unicode_normalization": "NFC",
        "dedupe": "stable_exact",
        "output": "pairs.jsonl",
    }

    manifest = json.loads((data_root.parent / "recipe.json").read_text(encoding="utf-8"))
    transformation = manifest["transformation"]
    assert transformation["input_records"] == 3
    assert transformation["output_records"] == 2
    assert transformation["blank_lines_removed"] == 1
    assert transformation["duplicate_pairs_removed"] == 1
    assert transformation["semantic_whitespace_preserved"] is True


def test_preference_jsonl_recipe_rejects_invalid_semantics(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)
    source = tmp_path / "preference"
    source.mkdir()
    (source / "pairs.jsonl").write_text(
        json.dumps(
            {
                "prompt": "Question:",
                "chosen": " same",
                "rejected": " same",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    add_local_dataset(
        project,
        source,
        name="Invalid preference",
        role=DatasetRole.PREFERENCE,
    )
    raw = get_data_view(project).datasets[0]

    with pytest.raises(FrontierwrightError, match="chosen and rejected"):
        prepare_dataset(
            project,
            dataset_id=str(raw["dataset_id"]),
            plugin_id=PREFERENCE_JSONL_PLUGIN_ID,
        )


def test_preference_recipe_role_compatibility_is_enforced(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Registry(project).initialize("NOVA", ModelOrigin.ZERO)

    pretrain = make_dataset(tmp_path / "pretrain")
    add_local_dataset(
        project,
        pretrain,
        name="Pretrain",
        role=DatasetRole.PRETRAIN,
    )
    pretrain_row = get_data_view(project).datasets[0]
    with pytest.raises(FrontierwrightError, match="requires a PREFERENCE dataset"):
        prepare_dataset(
            project,
            dataset_id=str(pretrain_row["dataset_id"]),
            plugin_id=PREFERENCE_JSONL_PLUGIN_ID,
        )

    preference = tmp_path / "preference-ok"
    preference.mkdir()
    (preference / "pairs.jsonl").write_text(
        json.dumps(
            {
                "prompt": "Q:",
                "chosen": " good",
                "rejected": " bad",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    add_local_dataset(
        project,
        preference,
        name="Preference",
        role=DatasetRole.PREFERENCE,
    )
    preference_row = next(
        item for item in get_data_view(project).datasets if item["role"] == "PREFERENCE"
    )
    with pytest.raises(FrontierwrightError, match="may only use"):
        prepare_dataset(
            project,
            dataset_id=str(preference_row["dataset_id"]),
            plugin_id=TEXT_LINES_PLUGIN_ID,
        )
