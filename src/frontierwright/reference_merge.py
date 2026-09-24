"""Reference-model artifact transforms used by Frontierwright EVOLVE interventions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def merge_reference_models(
    *,
    torch: Any,
    backend_id: str,
    presets: dict[str, Any],
    build_model: Any,
    load_reference_model: Any,
    parameter_count: Any,
    native_path: Any,
    reported_child_path: Any,
    request: dict[str, Any],
) -> dict[str, object]:
    """Materialize a linear merge of two compatible Frontierwright reference models."""

    primary_raw = request.get("model_source_path")
    secondary_raw = request.get("other_model_source_path")
    output_root = request.get("output_root")
    other_weight_raw = request.get("other_weight", 0.5)

    if not isinstance(primary_raw, str) or not primary_raw:
        raise ValueError("model_source_path is required")
    if not isinstance(secondary_raw, str) or not secondary_raw:
        raise ValueError("other_model_source_path is required")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root is required")
    if (
        isinstance(other_weight_raw, bool)
        or not isinstance(other_weight_raw, (int, float))
        or not math.isfinite(float(other_weight_raw))
    ):
        raise ValueError("other_weight must be a finite number")

    other_weight = float(other_weight_raw)
    if not 0.0 < other_weight < 1.0:
        raise ValueError("other_weight must be strictly between 0 and 1")
    primary_weight = 1.0 - other_weight

    primary = native_path(primary_raw).expanduser().resolve()
    secondary = native_path(secondary_raw).expanduser().resolve()
    if primary == secondary:
        raise ValueError("merge inputs must be different model directories")

    primary_config = _read_json_object(primary / "config.json", "primary config")
    secondary_config = _read_json_object(
        secondary / "config.json",
        "secondary config",
    )
    if primary_config.get("frontierwright_reference_backend") != backend_id:
        raise ValueError("primary model is not a Frontierwright reference model")
    if secondary_config.get("frontierwright_reference_backend") != backend_id:
        raise ValueError("secondary model is not a Frontierwright reference model")

    primary_preset = primary_config.get("preset")
    secondary_preset = secondary_config.get("preset")
    if (
        not isinstance(primary_preset, str)
        or primary_preset not in presets
        or secondary_preset != primary_preset
    ):
        raise ValueError("merge inputs must use the same supported reference preset")
    preset = presets[primary_preset]
    primary_vocab_size = primary_config.get("vocab_size")
    secondary_vocab_size = secondary_config.get("vocab_size")
    if (
        isinstance(primary_vocab_size, bool)
        or not isinstance(primary_vocab_size, int)
        or primary_vocab_size <= 0
        or secondary_vocab_size != primary_vocab_size
    ):
        raise ValueError("merge inputs must use the same valid vocab_size")

    try:
        primary_tokenizer = (primary / "tokenizer.json").read_bytes()
        secondary_tokenizer = (secondary / "tokenizer.json").read_bytes()
    except OSError as exc:
        raise ValueError("merge inputs must contain tokenizer.json") from exc
    if primary_tokenizer != secondary_tokenizer:
        raise ValueError("merge inputs must use identical tokenizer artifacts")

    primary_model = load_reference_model(
        torch,
        preset,
        model_source_path=str(primary),
        device="cpu",
    )
    secondary_model = load_reference_model(
        torch,
        preset,
        model_source_path=str(secondary),
        device="cpu",
    )
    primary_state = primary_model.state_dict()
    secondary_state = secondary_model.state_dict()
    if tuple(primary_state) != tuple(secondary_state):
        raise ValueError("merge inputs do not have the same state-dict structure")

    merged_state: dict[str, Any] = {}
    merged_tensor_count = 0
    for key, left in primary_state.items():
        right = secondary_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"merge tensor mismatch for {key}")
        if left.is_floating_point() or left.is_complex():
            merged_state[key] = (
                left.detach().to(device="cpu") * primary_weight
                + right.detach().to(device="cpu") * other_weight
            )
            merged_tensor_count += 1
        else:
            if not torch.equal(left, right):
                raise ValueError(
                    f"non-floating merge tensor differs between parents: {key}"
                )
            merged_state[key] = left.detach().clone().to(device="cpu")

    model = build_model(torch, preset, primary_vocab_size).to("cpu")
    model.load_state_dict(merged_state, strict=True)
    total_parameters = int(parameter_count(model))

    output = native_path(output_root).expanduser().resolve() / "model"
    output.mkdir(parents=True, exist_ok=False)

    config_payload = dict(primary_config)
    config_payload["parameter_count"] = total_parameters
    (output / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_bytes(primary_tokenizer)
    torch.save(model.state_dict(), output / "pytorch_model.bin")

    parent_evidence = request.get("parents")
    if not isinstance(parent_evidence, list) or len(parent_evidence) != 2:
        raise ValueError("merge request requires exactly two parent evidence records")

    transform_payload: dict[str, object] = {
        "schema_version": 1,
        "intervention_id": "frontierwright.evolve.linear-merge",
        "intervention_version": "1",
        "method": "linear_weight_merge",
        "primary_weight": primary_weight,
        "other_weight": other_weight,
        "parents": parent_evidence,
        "backend_id": backend_id,
        "preset": preset.name,
        "parameter_count": total_parameters,
        "merged_tensor_count": merged_tensor_count,
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
    }
    (output / "frontierwright-transform.json").write_text(
        json.dumps(transform_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "schema_version": 1,
        "ok": True,
        "operation": "merge",
        "output_model_path": reported_child_path(output_root, "model"),
        "metrics": transform_payload,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object")
    return payload
