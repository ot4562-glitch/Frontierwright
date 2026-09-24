"""Reference post-training quantization for Frontierwright-managed model variants."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

QUANTIZATION_FORMAT = "frontierwright-symmetric-int8-v1"


def quantize_reference_model(
    *,
    torch: Any,
    backend_id: str,
    presets: dict[str, Any],
    load_reference_model: Any,
    parameter_count: Any,
    native_path: Any,
    reported_child_path: Any,
    request: dict[str, Any],
) -> dict[str, object]:
    source_raw = request.get("model_source_path")
    output_root = request.get("output_root")
    if not isinstance(source_raw, str) or not source_raw:
        raise ValueError("model_source_path is required")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root is required")

    source = native_path(source_raw).expanduser().resolve()
    config = _read_json(source / "config.json", "source config")
    if config.get("frontierwright_reference_backend") != backend_id:
        raise ValueError("source model is not a Frontierwright reference model")
    preset_name = config.get("preset")
    if not isinstance(preset_name, str) or preset_name not in presets:
        raise ValueError("source model has an unsupported reference preset")
    if not (source / "pytorch_model.bin").is_file():
        raise ValueError("quantization requires a full-precision reference checkpoint")
    tokenizer = source / "tokenizer.json"
    if not tokenizer.is_file():
        raise ValueError("source model must contain tokenizer.json")

    preset = presets[preset_name]
    model = load_reference_model(
        torch,
        preset,
        model_source_path=str(source),
        device="cpu",
    )
    state = model.state_dict()
    quantized_state: dict[str, Any] = {}
    tensor_meta: dict[str, dict[str, object]] = {}
    source_tensor_bytes = 0
    quantized_tensor_bytes = 0
    quantized_tensor_count = 0

    for key, tensor in state.items():
        value = tensor.detach().to(device="cpu")
        source_tensor_bytes += int(value.numel() * value.element_size())
        if value.is_floating_point():
            working = value.to(dtype=torch.float32)
            max_abs = float(working.abs().max().item()) if working.numel() else 0.0
            scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"invalid quantization scale for tensor {key}")
            packed = torch.round(working / scale).clamp(-127, 127).to(torch.int8)
            quantized_state[key] = packed
            quantized_tensor_bytes += int(packed.numel() * packed.element_size())
            quantized_tensor_count += 1
            tensor_meta[key] = {
                "quantized": True,
                "scale": scale,
                "source_dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        else:
            copied = value.clone()
            quantized_state[key] = copied
            quantized_tensor_bytes += int(copied.numel() * copied.element_size())
            tensor_meta[key] = {
                "quantized": False,
                "scale": None,
                "source_dtype": str(value.dtype),
                "shape": list(value.shape),
            }

    output = native_path(output_root).expanduser().resolve() / "model"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(config, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(tokenizer, output / "tokenizer.json")
    torch.save(quantized_state, output / "quantized_model.pt")

    parent = request.get("parent")
    if not isinstance(parent, dict):
        raise ValueError("quantization request requires parent evidence")

    quantized_metadata: dict[str, object] = {
        "schema_version": 1,
        "format": QUANTIZATION_FORMAT,
        "backend_id": backend_id,
        "preset": preset_name,
        "parameter_count": int(parameter_count(model)),
        "source_tensor_bytes": source_tensor_bytes,
        "quantized_tensor_bytes": quantized_tensor_bytes,
        "quantized_tensor_count": quantized_tensor_count,
        "tensor_metadata": tensor_meta,
    }
    (output / "frontierwright-quantized.json").write_text(
        json.dumps(quantized_metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    file_bytes = int((output / "quantized_model.pt").stat().st_size)
    transform_payload: dict[str, object] = {
        "schema_version": 1,
        "intervention_id": "frontierwright.optimize.symmetric-int8",
        "intervention_version": "1",
        "method": "symmetric_int8_post_training_quantization",
        "parent": parent,
        "backend_id": backend_id,
        "preset": preset_name,
        "parameter_count": int(parameter_count(model)),
        "source_tensor_bytes": source_tensor_bytes,
        "quantized_tensor_bytes": quantized_tensor_bytes,
        "quantized_file_bytes": file_bytes,
        "tensor_storage_ratio": (
            quantized_tensor_bytes / source_tensor_bytes
            if source_tensor_bytes > 0
            else 1.0
        ),
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
        "operation": "quantize",
        "output_model_path": reported_child_path(output_root, "model"),
        "metrics": transform_payload,
    }


def load_quantized_reference_state(torch: Any, source: Path) -> dict[str, Any]:
    metadata = _read_json(
        source / "frontierwright-quantized.json",
        "quantized metadata",
    )
    if metadata.get("format") != QUANTIZATION_FORMAT:
        raise ValueError("unsupported Frontierwright quantization format")
    tensor_meta = metadata.get("tensor_metadata")
    if not isinstance(tensor_meta, dict):
        raise ValueError("quantized metadata lacks tensor_metadata")

    raw_state = torch.load(
        source / "quantized_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw_state, dict):
        raise ValueError("quantized_model.pt must contain a state dictionary")
    if set(raw_state) != set(tensor_meta):
        raise ValueError("quantized tensor keys do not match metadata")

    restored: dict[str, Any] = {}
    for key, tensor in raw_state.items():
        info = tensor_meta.get(key)
        if not isinstance(info, dict):
            raise ValueError(f"quantized metadata is invalid for tensor {key}")
        expected_shape = info.get("shape")
        if not isinstance(expected_shape, list):
            raise ValueError(f"quantized metadata lacks shape for tensor {key}")
        if list(tensor.shape) != expected_shape:
            raise ValueError(f"quantized tensor shape mismatch for {key}")
        if info.get("quantized") is True:
            scale = info.get("scale")
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0.0
            ):
                raise ValueError(f"quantized tensor scale is invalid for {key}")
            restored[key] = tensor.to(dtype=torch.float32) * float(scale)
        elif info.get("quantized") is False:
            restored[key] = tensor
        else:
            raise ValueError(f"quantized flag is invalid for tensor {key}")
    return restored


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object")
    return payload
