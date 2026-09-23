import json
from pathlib import Path

from frontierwright.reference_backend import (
    _load_config,
    _objective_for_path,
    _read_corpus,
    backend_spec_payload,
)


def test_reference_backend_declares_pretraining_and_full_sft() -> None:
    payload = backend_spec_payload("python")

    assert payload["supported_paths"] == [
        "FROM_SCRATCH_PRETRAINING",
        "CONTINUED_PRETRAINING",
        "FULL_SFT",
    ]


def test_reference_backend_objective_labels_are_path_specific() -> None:
    assert _objective_for_path("FROM_SCRATCH_PRETRAINING") == "causal_lm_pretraining"
    assert _objective_for_path("CONTINUED_PRETRAINING") == (
        "causal_lm_continued_pretraining"
    )
    assert _objective_for_path("FULL_SFT") == "full_parameter_causal_sft"


def test_reference_config_infers_preset_from_materialized_model(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "frontierwright_reference_backend": (
                    "frontierwright-reference-pytorch-v1"
                ),
                "preset": "zero-25m",
            }
        ),
        encoding="utf-8",
    )

    config = _load_config(
        {
            "model_source_path": str(model),
            "config": {
                "steps": 3,
                "calibration_steps": 1,
                "batch_size": 1,
                "device": "cpu",
            },
        }
    )

    assert config.preset.name == "zero-25m"
    assert config.steps == 3


def test_reference_backend_concatenates_byte_shards_without_separator(
    tmp_path: Path,
) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "shard-00000.bin").write_bytes(b"abcd")
    (shards / "shard-00001.bin").write_bytes(b"efgh")
    (shards / "shard-00002.bin").write_bytes(b"ijk")

    assert _read_corpus(shards, max_bytes=100) == b"abcdefghijk"
    assert _read_corpus(shards, max_bytes=6) == b"abcdef"


def test_reference_backend_still_separates_multiple_plain_files(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_bytes(b"alpha")
    (corpus / "b.txt").write_bytes(b"beta")

    assert _read_corpus(corpus, max_bytes=100) == b"alpha\nbeta"
