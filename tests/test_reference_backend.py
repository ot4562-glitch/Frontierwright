import json
from pathlib import Path

from frontierwright.reference_backend import (
    _load_config,
    backend_spec_payload,
)


def test_reference_backend_declares_initial_and_continued_pretraining() -> None:
    payload = backend_spec_payload("python")

    assert payload["supported_paths"] == [
        "FROM_SCRATCH_PRETRAINING",
        "CONTINUED_PRETRAINING",
    ]


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
