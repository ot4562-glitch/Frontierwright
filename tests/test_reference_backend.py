import json
from pathlib import Path

from frontierwright.reference_backend import (
    PRESETS,
    ReferenceConfig,
    _apply_qlora_parametrizations,
    _build_model,
    _load_config,
    _load_generation_config,
    _merge_lora_parametrizations,
    _objective_for_path,
    _parameter_count,
    _read_corpus,
    _read_preference_pairs,
    _read_rl_episodes,
    _trainable_parameter_count,
    backend_spec_payload,
)


def test_reference_backend_declares_pretraining_full_sft_and_lora() -> None:
    payload = backend_spec_payload("python")

    assert payload["supported_paths"] == [
        "FROM_SCRATCH_PRETRAINING",
        "CONTINUED_PRETRAINING",
        "FULL_SFT",
        "LORA_SFT",
        "QLORA_SFT",
        "DPO",
        "RL_POLICY_OPTIMIZATION",
        "DISTILL",
    ]


def test_reference_backend_objective_labels_are_path_specific() -> None:
    assert _objective_for_path("FROM_SCRATCH_PRETRAINING") == "causal_lm_pretraining"
    assert _objective_for_path("CONTINUED_PRETRAINING") == ("causal_lm_continued_pretraining")
    assert _objective_for_path("FULL_SFT") == "full_parameter_causal_sft"
    assert _objective_for_path("LORA_SFT") == "lora_causal_sft"
    assert _objective_for_path("QLORA_SFT") == "qlora_nf4_causal_sft"
    assert _objective_for_path("DPO") == "direct_preference_optimization"
    assert _objective_for_path("RL_POLICY_OPTIMIZATION") == "reinforce_verifiable_choice"
    assert _objective_for_path("DISTILL") == "knowledge_distillation"


def test_reference_config_infers_preset_from_materialized_model(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "frontierwright_reference_backend": ("frontierwright-reference-pytorch-v1"),
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


def test_reference_lora_config_is_path_scoped(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "frontierwright_reference_backend": ("frontierwright-reference-pytorch-v1"),
                "preset": "zero-8m",
            }
        ),
        encoding="utf-8",
    )

    lora = _load_config(
        {
            "path_id": "LORA_SFT",
            "model_source_path": str(model),
            "config": {"lora_rank": 4, "lora_alpha": 8.0},
        }
    )
    assert lora.lora_rank == 4
    assert lora.lora_alpha == 8.0

    qlora = _load_config(
        {
            "path_id": "QLORA_SFT",
            "model_source_path": str(model),
            "config": {"lora_rank": 2, "lora_alpha": 4.0},
        }
    )
    assert qlora.lora_rank == 2
    assert qlora.lora_alpha == 4.0

    import pytest

    with pytest.raises(ValueError, match="only valid for LORA_SFT or QLORA_SFT"):
        _load_config(
            {
                "path_id": "FULL_SFT",
                "model_source_path": str(model),
                "config": {"lora_rank": 4},
            }
        )


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


def test_reference_qlora_packs_nf4_and_merges_when_torch_available() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    preset = PRESETS["zero-8m"]
    config = ReferenceConfig(
        preset=preset,
        steps=1,
        calibration_steps=1,
        batch_size=1,
        learning_rate=3e-4,
        weight_decay=0.0,
        seed=7,
        device="cpu",
        max_dataset_bytes=1024,
        lora_rank=2,
        lora_alpha=4.0,
    )
    torch.manual_seed(config.seed)
    model = _build_model(torch, preset)
    base_parameter_count = _parameter_count(model)

    details = _apply_qlora_parametrizations(torch, model, config)

    packed = model.blocks[0].mlp[0].parametrizations.weight.original
    assert packed.dtype == torch.uint8
    assert packed.ndim == 1
    assert details["method"] == "qlora"
    assert details["quantization_type"] == "nf4"
    assert details["quantization_bits"] == 4
    assert (
        details["quantized_target_storage_bytes"] < details["full_precision_target_storage_bytes"]
    )
    assert _trainable_parameter_count(model) == details["trainable_parameter_count"]

    tokens = torch.randint(0, 256, (1, 8), dtype=torch.long)
    loss = model(tokens).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )

    _merge_lora_parametrizations(torch, model)
    assert _parameter_count(model) == base_parameter_count
    assert not hasattr(model.blocks[0].mlp[0], "parametrizations")


def test_reference_dpo_config_is_path_scoped(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "frontierwright_reference_backend": ("frontierwright-reference-pytorch-v1"),
                "preset": "zero-8m",
            }
        ),
        encoding="utf-8",
    )

    config = _load_config(
        {
            "path_id": "DPO",
            "model_source_path": str(model),
            "config": {"dpo_beta": 0.2},
        }
    )
    assert config.dpo_beta == 0.2

    import pytest

    with pytest.raises(ValueError, match="dpo_beta is only valid for DPO"):
        _load_config(
            {
                "path_id": "FULL_SFT",
                "model_source_path": str(model),
                "config": {"dpo_beta": 0.2},
            }
        )


def test_reference_distill_config_pins_smaller_student_preset(tmp_path: Path) -> None:
    model = tmp_path / "teacher"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "frontierwright_reference_backend": ("frontierwright-reference-pytorch-v1"),
                "preset": "zero-25m",
            }
        ),
        encoding="utf-8",
    )

    config = _load_config(
        {
            "path_id": "DISTILL",
            "model_source_path": str(model),
            "config": {
                "student_preset": "zero-8m",
                "distill_temperature": 3.0,
                "distill_alpha": 0.7,
            },
        }
    )
    assert config.preset.name == "zero-25m"
    assert config.student_preset is not None
    assert config.student_preset.name == "zero-8m"
    assert config.distill_temperature == 3.0
    assert config.distill_alpha == 0.7

    import pytest

    with pytest.raises(ValueError, match="DISTILL requires student_preset"):
        _load_config(
            {
                "path_id": "DISTILL",
                "model_source_path": str(model),
                "config": {},
            }
        )

    with pytest.raises(ValueError, match="only valid for DISTILL"):
        _load_config(
            {
                "path_id": "FULL_SFT",
                "model_source_path": str(model),
                "config": {"student_preset": "zero-8m"},
            }
        )


def test_reference_dpo_reads_jsonl_pairs_deterministically(tmp_path: Path) -> None:
    data = tmp_path / "preference"
    data.mkdir()
    (data / "pairs.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Question: 1+1? Answer:",
                        "chosen": " 2",
                        "rejected": " 3",
                    }
                ),
                json.dumps(
                    {
                        "prompt": "Opposite of cold:",
                        "chosen": " hot",
                        "rejected": " blue",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pairs = _read_preference_pairs(data, max_bytes=4096)
    assert len(pairs) == 2
    assert pairs[0].prompt == b"Question: 1+1? Answer:"
    assert pairs[0].chosen == b" 2"
    assert pairs[0].rejected == b" 3"


def test_reference_dpo_rejects_identical_preferences(tmp_path: Path) -> None:
    import pytest

    data = tmp_path / "preference"
    data.mkdir()
    (data / "pairs.jsonl").write_text(
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

    with pytest.raises(ValueError, match="chosen and rejected must differ"):
        _read_preference_pairs(data, max_bytes=4096)


def test_reference_generation_config_allows_greedy_and_rejects_negative_temperature(
    tmp_path: Path,
) -> None:
    import pytest

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "frontierwright_reference_backend": ("frontierwright-reference-pytorch-v1"),
                "preset": "zero-8m",
            }
        ),
        encoding="utf-8",
    )

    greedy = _load_generation_config(
        {
            "model_source_path": str(model),
            "config": {
                "max_new_tokens": 4,
                "temperature": 0.0,
                "seed": 9,
                "device": "cpu",
            },
        }
    )
    assert greedy.max_new_tokens == 4
    assert greedy.temperature == 0.0
    assert greedy.seed == 9
    assert greedy.preset.name == "zero-8m"

    with pytest.raises(ValueError, match="temperature"):
        _load_generation_config(
            {
                "model_source_path": str(model),
                "config": {"temperature": -0.1},
            }
        )


def test_reference_rl_config_and_episode_schema_are_explicit(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "frontierwright_reference_backend": "frontierwright-reference-pytorch-v1",
                "preset": "zero-8m",
            }
        ),
        encoding="utf-8",
    )
    config = _load_config(
        {
            "path_id": "RL_POLICY_OPTIMIZATION",
            "model_source_path": str(model),
            "config": {
                "steps": 2,
                "batch_size": 1,
                "rl": {
                    "schema_version": 1,
                    "algorithm_id": "reinforce",
                    "environment": {
                        "id": "choice-env",
                        "version": "1",
                        "kind": "VERIFIABLE_MULTIPLE_CHOICE",
                        "config": {},
                    },
                    "reward": {
                        "id": "exact-choice",
                        "version": "1",
                        "kind": "EXACT_CORRECT_CHOICE",
                        "config": {},
                    },
                    "algorithm_config": {"reward_baseline": 0.5},
                },
            },
        }
    )
    assert config.rl_spec is not None
    assert config.rl_spec.algorithm_id == "reinforce"

    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        '{"prompt":"2+2=","choices":["4","5"],"correct_index":0}\n',
        encoding="utf-8",
    )
    parsed = _read_rl_episodes(episodes, max_bytes=1024)
    assert len(parsed) == 1
    assert parsed[0].correct_index == 0
    assert parsed[0].choices == (b"4", b"5")
