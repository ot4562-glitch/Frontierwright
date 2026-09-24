"""Frontierwright built-in PyTorch reference backend.

This backend intentionally stays narrow: real causal language-model training for
Frontierwright reference-model lineages, including initial/continued pretraining,
full-parameter causal SFT, merged-output LoRA/QLoRA SFT, Direct Preference
Optimization, and teacher-to-smaller-student knowledge distillation. It is not a
compatibility layer for arbitrary Hugging Face architectures.

The module is executed by a dedicated training Python environment:
    python -m frontierwright.reference_backend REQUEST_JSON
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from frontierwright.reference_merge import merge_reference_models
from frontierwright.reference_quantization import (
    load_quantized_reference_state,
    quantize_reference_model,
)
from frontierwright.reference_tokenizer import (
    decode_with_tokenizer,
    encode_with_tokenizer,
    load_reference_tokenizer,
    tokenizer_fingerprint,
    tokenizer_vocab_size,
)

REFERENCE_BACKEND_ID = "frontierwright-reference-pytorch-v1"
SUPPORTED_PATHS = (
    "FROM_SCRATCH_PRETRAINING",
    "CONTINUED_PRETRAINING",
    "FULL_SFT",
    "LORA_SFT",
    "QLORA_SFT",
    "DPO",
    "DISTILL",
)
VOCAB_SIZE = 256
QLORA_BLOCK_SIZE = 64
NF4_CODEBOOK: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


@dataclass(frozen=True)
class ModelPreset:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    context_length: int


PRESETS: dict[str, ModelPreset] = {
    "zero-8m": ModelPreset(
        name="zero-8m",
        d_model=320,
        n_layers=6,
        n_heads=8,
        d_ff=1280,
        context_length=128,
    ),
    "zero-25m": ModelPreset(
        name="zero-25m",
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        context_length=128,
    ),
}


@dataclass(frozen=True)
class ReferenceEvaluationConfig:
    preset: ModelPreset
    batch_size: int
    max_batches: int
    device: str
    max_dataset_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": asdict(self.preset),
            "batch_size": self.batch_size,
            "max_batches": self.max_batches,
            "device": self.device,
            "max_dataset_bytes": self.max_dataset_bytes,
        }


@dataclass(frozen=True)
class ReferenceGenerationConfig:
    preset: ModelPreset
    max_new_tokens: int
    temperature: float
    seed: int
    device: str

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": asdict(self.preset),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass(frozen=True)
class ReferenceInferenceProfileConfig:
    preset: ModelPreset
    max_new_tokens: int
    warmup_runs: int
    measured_runs: int
    device: str

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": asdict(self.preset),
            "max_new_tokens": self.max_new_tokens,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "device": self.device,
        }


@dataclass(frozen=True)
class ReferenceConfig:
    preset: ModelPreset
    steps: int
    calibration_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str
    max_dataset_bytes: int
    lora_rank: int
    lora_alpha: float
    dpo_beta: float = 0.1
    student_preset: ModelPreset | None = None
    distill_temperature: float = 2.0
    distill_alpha: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": asdict(self.preset),
            "steps": self.steps,
            "calibration_steps": self.calibration_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "device": self.device,
            "max_dataset_bytes": self.max_dataset_bytes,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "dpo_beta": self.dpo_beta,
            "student_preset": (
                asdict(self.student_preset)
                if self.student_preset is not None
                else None
            ),
            "distill_temperature": self.distill_temperature,
            "distill_alpha": self.distill_alpha,
        }


def _native_path(raw: str) -> Path:
    """Translate a WSL /mnt/<drive>/ path when this backend runs on Windows."""
    if os.name == "nt" and raw.startswith("/mnt/") and len(raw) > 7:
        drive = raw[5]
        if raw[6] == "/":
            rest = raw[7:].replace("/", "\\")
            return Path(f"{drive.upper()}:\\{rest}")
    return Path(raw)


def _reported_child_path(raw_root: str, child: str) -> str:
    if raw_root.startswith("/mnt/"):
        return raw_root.rstrip("/\\") + "/" + child
    return str((_native_path(raw_root) / child).resolve())


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    sys.stdout.flush()


def _fail(code: str, message: str) -> int:
    _emit(
        {
            "schema_version": 1,
            "ok": False,
            "error": {"code": code, "message": message},
        }
    )
    return 2


def _positive_int(raw: object, default: int, label: str) -> int:
    value = default if raw is None else raw
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_float(raw: object, default: float, label: str) -> float:
    value = default if raw is None else raw
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _nonnegative_float(raw: object, default: float, label: str) -> float:
    value = default if raw is None else raw
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a nonnegative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a nonnegative finite number")
    return result


def _unit_interval_float(raw: object, default: float, label: str) -> float:
    value = default if raw is None else raw
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite number between 0 and 1")
    return result


def _preset_name_for_request(
    request: dict[str, Any],
    raw_config: dict[str, Any],
) -> str:
    explicit = raw_config.get("preset")
    if explicit is not None:
        if not isinstance(explicit, str) or explicit not in PRESETS:
            raise ValueError(
                "preset must be one of: " + ", ".join(sorted(PRESETS))
            )
        return explicit

    model_source = request.get("model_source_path")
    if model_source is None:
        return "zero-8m"
    if not isinstance(model_source, str) or not model_source:
        raise ValueError("model_source_path must be a nonempty string")

    config_path = _native_path(model_source).expanduser().resolve() / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "model_source_path does not contain a readable reference config.json"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("model source config.json must contain an object")
    inferred = payload.get("preset")
    if not isinstance(inferred, str) or inferred not in PRESETS:
        raise ValueError("model source config.json has an unsupported preset")
    return inferred


def _load_config(request: dict[str, Any]) -> ReferenceConfig:
    raw = request.get("config", {})
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")

    preset_name = _preset_name_for_request(request, raw)
    preset = PRESETS[preset_name]

    device = raw.get("device", "auto")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")

    seed = raw.get("seed", 1337)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    path_id = request.get("path_id")
    if path_id not in {"LORA_SFT", "QLORA_SFT"} and any(
        key in raw for key in ("lora_rank", "lora_alpha")
    ):
        raise ValueError(
            "lora_rank/lora_alpha are only valid for LORA_SFT or QLORA_SFT"
        )
    if path_id != "DPO" and "dpo_beta" in raw:
        raise ValueError("dpo_beta is only valid for DPO")

    distill_keys = {"student_preset", "distill_temperature", "distill_alpha"}
    if path_id != "DISTILL" and any(key in raw for key in distill_keys):
        raise ValueError(
            "student_preset/distill_temperature/distill_alpha are only valid for DISTILL"
        )
    student_preset: ModelPreset | None = None
    if path_id == "DISTILL":
        student_raw = raw.get("student_preset")
        if not isinstance(student_raw, str) or student_raw not in PRESETS:
            raise ValueError(
                "DISTILL requires student_preset to be one of: "
                + ", ".join(sorted(PRESETS))
            )
        student_preset = PRESETS[student_raw]

    return ReferenceConfig(
        preset=preset,
        steps=_positive_int(raw.get("steps"), 20, "steps"),
        calibration_steps=_positive_int(
            raw.get("calibration_steps"),
            2,
            "calibration_steps",
        ),
        batch_size=_positive_int(raw.get("batch_size"), 4, "batch_size"),
        learning_rate=_positive_float(
            raw.get("learning_rate"),
            3e-4,
            "learning_rate",
        ),
        weight_decay=_nonnegative_float(
            raw.get("weight_decay"),
            0.1,
            "weight_decay",
        ),
        seed=seed,
        device=device,
        max_dataset_bytes=_positive_int(
            raw.get("max_dataset_bytes"),
            64 * 1024 * 1024,
            "max_dataset_bytes",
        ),
        lora_rank=_positive_int(raw.get("lora_rank"), 8, "lora_rank"),
        lora_alpha=_positive_float(raw.get("lora_alpha"), 16.0, "lora_alpha"),
        dpo_beta=_positive_float(raw.get("dpo_beta"), 0.1, "dpo_beta"),
        student_preset=student_preset,
        distill_temperature=_positive_float(
            raw.get("distill_temperature"), 2.0, "distill_temperature"
        ),
        distill_alpha=_unit_interval_float(
            raw.get("distill_alpha"), 0.5, "distill_alpha"
        ),
    )


def _load_evaluation_config(
    request: dict[str, Any],
) -> ReferenceEvaluationConfig:
    raw = request.get("config", {})
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")

    unknown = sorted(
        set(raw) - {"preset", "batch_size", "max_batches", "device", "max_dataset_bytes"}
    )
    if unknown:
        raise ValueError(
            "evaluation config does not support keys: " + ", ".join(unknown)
        )

    preset_name = _preset_name_for_request(request, raw)
    preset = PRESETS[preset_name]

    device = raw.get("device", "auto")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")

    return ReferenceEvaluationConfig(
        preset=preset,
        batch_size=_positive_int(raw.get("batch_size"), 4, "batch_size"),
        max_batches=_positive_int(raw.get("max_batches"), 16, "max_batches"),
        device=device,
        max_dataset_bytes=_positive_int(
            raw.get("max_dataset_bytes"),
            64 * 1024 * 1024,
            "max_dataset_bytes",
        ),
    )


def _load_generation_config(
    request: dict[str, Any],
) -> ReferenceGenerationConfig:
    raw = request.get("config", {})
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")

    allowed = {"preset", "max_new_tokens", "temperature", "seed", "device"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "generation config does not support keys: " + ", ".join(unknown)
        )

    preset_name = _preset_name_for_request(request, raw)
    preset = PRESETS[preset_name]
    device = raw.get("device", "auto")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")

    seed = raw.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    return ReferenceGenerationConfig(
        preset=preset,
        max_new_tokens=_positive_int(
            raw.get("max_new_tokens"),
            64,
            "max_new_tokens",
        ),
        temperature=_nonnegative_float(
            raw.get("temperature"),
            0.0,
            "temperature",
        ),
        seed=seed,
        device=device,
    )


def _load_inference_profile_config(
    request: dict[str, Any],
) -> ReferenceInferenceProfileConfig:
    raw = request.get("config", {})
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")
    allowed = {"preset", "max_new_tokens", "warmup_runs", "measured_runs", "device"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "inference profile config does not support keys: " + ", ".join(unknown)
        )
    preset_name = _preset_name_for_request(request, raw)
    device = raw.get("device", "auto")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return ReferenceInferenceProfileConfig(
        preset=PRESETS[preset_name],
        max_new_tokens=_positive_int(raw.get("max_new_tokens"), 16, "max_new_tokens"),
        warmup_runs=_positive_int(raw.get("warmup_runs"), 1, "warmup_runs"),
        measured_runs=_positive_int(raw.get("measured_runs"), 3, "measured_runs"),
        device=device,
    )


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the reference-backend environment. "
            "Install Frontierwright with the 'train' extra in a supported Python environment."
        ) from exc
    return torch


def _select_device(torch: Any, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dataset_files(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f"dataset source is not readable: {source}")
    files = [path for path in source.rglob("*") if path.is_file()]
    if not files:
        raise ValueError("dataset contains no files")
    return sorted(files, key=lambda path: path.relative_to(source).as_posix())


def _read_corpus(source: Path, *, max_bytes: int) -> bytes:
    files = _dataset_files(source)
    shard_mode = all(
        path.suffix == ".bin" and path.name.startswith("shard-")
        for path in files
    )
    separator = b"" if shard_mode else b"\n"

    chunks: list[bytes] = []
    total = 0
    for index, path in enumerate(files):
        if index > 0 and separator:
            if total + len(separator) > max_bytes:
                break
            chunks.append(separator)
            total += len(separator)

        remaining = max_bytes - total
        if remaining <= 0:
            break
        payload = path.read_bytes()
        if len(payload) > remaining:
            chunks.append(payload[:remaining])
            total += remaining
            break
        chunks.append(payload)
        total += len(payload)

    corpus = b"".join(chunks)
    if not corpus:
        raise ValueError("dataset corpus is empty")

    return corpus


@dataclass(frozen=True)
class PreferencePair:
    prompt: bytes
    chosen: bytes
    rejected: bytes


def _read_preference_pairs(
    source: Path,
    *,
    max_bytes: int,
) -> tuple[PreferencePair, ...]:
    """Load deterministic UTF-8 JSONL preference pairs."""

    files = [
        path
        for path in _dataset_files(source)
        if path.suffix.lower() == ".jsonl"
    ]
    if not files:
        raise ValueError(
            "DPO preference dataset must contain at least one .jsonl file"
        )

    pairs: list[PreferencePair] = []
    consumed = 0
    limit_reached = False
    for path in files:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                if consumed + len(raw_line) > max_bytes:
                    if not pairs:
                        raise ValueError(
                            "first DPO preference record exceeds max_dataset_bytes"
                        )
                    limit_reached = True
                    break
                consumed += len(raw_line)
                try:
                    raw = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid UTF-8 JSONL preference record: "
                        f"{path.name}:{line_number}"
                    ) from exc
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"preference record must be an object: "
                        f"{path.name}:{line_number}"
                    )

                values: dict[str, bytes] = {}
                for key in ("prompt", "chosen", "rejected"):
                    value = raw.get(key)
                    if not isinstance(value, str) or not value:
                        raise ValueError(
                            f"preference record {key} must be a nonempty string: "
                            f"{path.name}:{line_number}"
                        )
                    values[key] = value.encode("utf-8")
                if values["chosen"] == values["rejected"]:
                    raise ValueError(
                        f"chosen and rejected must differ: "
                        f"{path.name}:{line_number}"
                    )
                pairs.append(
                    PreferencePair(
                        prompt=values["prompt"],
                        chosen=values["chosen"],
                        rejected=values["rejected"],
                    )
                )
        if limit_reached:
            break

    if not pairs:
        raise ValueError("DPO preference dataset contains no usable pairs")
    return tuple(pairs)


def _validate_preference_pairs(
    pairs: tuple[PreferencePair, ...],
    preset: ModelPreset,
    tokenizer_payload: dict[str, Any],
) -> None:
    maximum = preset.context_length + 1
    for index, pair in enumerate(pairs):
        prompt_ids = encode_with_tokenizer(pair.prompt, tokenizer_payload)
        for label, response in (
            ("chosen", pair.chosen),
            ("rejected", pair.rejected),
        ):
            response_ids = encode_with_tokenizer(response, tokenizer_payload)
            combined = len(prompt_ids) + len(response_ids)
            if combined < 2:
                raise ValueError(
                    f"DPO pair {index} {label} sequence is too short to score"
                )
            if combined > maximum:
                raise ValueError(
                    f"DPO pair {index} {label} exceeds context capacity "
                    f"({combined} tokens > {maximum})"
                )

def _response_logprob(
    torch: Any,
    model: Any,
    *,
    prompt: bytes,
    response: bytes,
    tokenizer_payload: dict[str, Any],
    device: str,
) -> Any:
    prompt_ids = encode_with_tokenizer(prompt, tokenizer_payload)
    response_ids = encode_with_tokenizer(response, tokenizer_payload)
    combined_ids = prompt_ids + response_ids
    if len(combined_ids) < 2 or not response_ids:
        raise ValueError("DPO response tokenization produced an unscorable sequence")
    tokens = torch.tensor(combined_ids, dtype=torch.long, device=device)
    inputs = tokens[:-1].unsqueeze(0)
    targets = tokens[1:]
    logits = model(inputs)[0]
    response_start = max(len(prompt_ids) - 1, 0)
    response_logits = logits[response_start:]
    response_targets = targets[response_start:]
    token_logprobs = torch.nn.functional.log_softmax(
        response_logits,
        dim=-1,
    )
    selected = token_logprobs.gather(
        1,
        response_targets.unsqueeze(1),
    ).squeeze(1)
    return selected.sum()

def _dpo_training_objects(
    torch: Any,
    config: ReferenceConfig,
    pairs: tuple[PreferencePair, ...],
    *,
    model_source_path: str,
) -> tuple[Any, Any, Any, Any, dict[str, Any], str, dict[str, object]]:
    torch.manual_seed(config.seed)
    device = _select_device(torch, config.device)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    policy = _load_reference_model(
        torch,
        config.preset,
        model_source_path=model_source_path,
        device=device,
    )
    reference = _load_reference_model(
        torch,
        config.preset,
        model_source_path=model_source_path,
        device=device,
    )
    tokenizer_payload, vocab_size = _reference_tokenizer_for_model(model_source_path)
    _validate_preference_pairs(pairs, config.preset, tokenizer_payload)

    for parameter in reference.parameters():
        parameter.requires_grad = False
    reference.eval()

    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("DPO policy produced no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    base_parameter_count = _parameter_count(policy)
    training_details: dict[str, object] = {
        "method": "dpo",
        "beta": config.dpo_beta,
        "reference_frozen": True,
        "reference_source": "current_model_checkpoint",
        "pair_count": len(pairs),
        "base_parameter_count": base_parameter_count,
        "trainable_parameter_count": _trainable_parameter_count(policy),
        "vocab_size": vocab_size,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer_payload),
    }
    return (
        policy,
        reference,
        optimizer,
        generator,
        tokenizer_payload,
        device,
        training_details,
    )


def _dpo_train_step(
    torch: Any,
    policy: Any,
    reference: Any,
    optimizer: Any,
    pairs: tuple[PreferencePair, ...],
    generator: Any,
    tokenizer_payload: dict[str, Any],
    *,
    config: ReferenceConfig,
    device: str,
) -> tuple[float, int]:
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    indexes = torch.randint(
        0,
        len(pairs),
        (config.batch_size,),
        generator=generator,
    )

    losses: list[Any] = []
    processed_tokens = 0
    for raw_index in indexes.tolist():
        pair = pairs[int(raw_index)]
        policy_chosen = _response_logprob(
            torch,
            policy,
            prompt=pair.prompt,
            response=pair.chosen,
            tokenizer_payload=tokenizer_payload,
            device=device,
        )
        policy_rejected = _response_logprob(
            torch,
            policy,
            prompt=pair.prompt,
            response=pair.rejected,
            tokenizer_payload=tokenizer_payload,
            device=device,
        )
        with torch.no_grad():
            reference_chosen = _response_logprob(
                torch,
                reference,
                prompt=pair.prompt,
                response=pair.chosen,
                tokenizer_payload=tokenizer_payload,
                device=device,
            )
            reference_rejected = _response_logprob(
                torch,
                reference,
                prompt=pair.prompt,
                response=pair.rejected,
                tokenizer_payload=tokenizer_payload,
                device=device,
            )

        policy_logratio = policy_chosen - policy_rejected
        reference_logratio = reference_chosen - reference_rejected
        preference_logit = config.dpo_beta * (
            policy_logratio - reference_logratio
        )
        losses.append(-torch.nn.functional.logsigmoid(preference_logit))
        prompt_ids = encode_with_tokenizer(pair.prompt, tokenizer_payload)
        chosen_ids = encode_with_tokenizer(pair.chosen, tokenizer_payload)
        rejected_ids = encode_with_tokenizer(pair.rejected, tokenizer_payload)
        processed_tokens += (
            len(prompt_ids) + len(chosen_ids) - 1
            + len(prompt_ids) + len(rejected_ids) - 1
        )

    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item()), processed_tokens


def _distillation_training_objects(
    torch: Any,
    config: ReferenceConfig,
    corpus_bytes: bytes,
    *,
    model_source_path: str,
) -> tuple[Any, Any, Any, Any, Any, str, dict[str, object]]:
    student_preset = config.student_preset
    if student_preset is None:
        raise ValueError("DISTILL requires student_preset")

    torch.manual_seed(config.seed)
    device = _select_device(torch, config.device)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    teacher = _load_reference_model(
        torch,
        config.preset,
        model_source_path=model_source_path,
        device=device,
    )
    tokenizer_payload, vocab_size = _reference_tokenizer_for_model(model_source_path)
    token_ids = encode_with_tokenizer(corpus_bytes, tokenizer_payload)
    if not token_ids:
        raise ValueError("distillation tokenizer produced an empty corpus")
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()

    student = _build_model(
        torch,
        student_preset,
        vocab_size=vocab_size,
    ).to(device)
    teacher_parameter_count = _parameter_count(teacher)
    student_parameter_count = _parameter_count(student)
    if student_parameter_count >= teacher_parameter_count:
        raise ValueError(
            "DISTILL student must have fewer parameters than the teacher model"
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    corpus = torch.tensor(token_ids, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    details: dict[str, object] = {
        "method": "knowledge_distillation",
        "teacher_preset": config.preset.name,
        "student_preset": student_preset.name,
        "teacher_frozen": True,
        "temperature": config.distill_temperature,
        "soft_loss_weight": config.distill_alpha,
        "hard_loss_weight": 1.0 - config.distill_alpha,
        "teacher_parameter_count": teacher_parameter_count,
        "base_parameter_count": student_parameter_count,
        "trainable_parameter_count": _trainable_parameter_count(student),
        "vocab_size": vocab_size,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer_payload),
    }
    return (
        student,
        teacher,
        optimizer,
        corpus,
        generator,
        device,
        details,
    )


def _kd_step(
    torch: Any,
    student: Any,
    teacher: Any,
    optimizer: Any,
    corpus: Any,
    generator: Any,
    *,
    config: ReferenceConfig,
    device: str,
) -> tuple[float, int]:
    student_preset = config.student_preset
    if student_preset is None:
        raise ValueError("DISTILL requires student_preset")

    student.train()
    teacher.eval()
    context_length = min(
        config.preset.context_length,
        student_preset.context_length,
    )
    x, y = _sample_batch(
        torch,
        corpus,
        batch_size=config.batch_size,
        context_length=context_length,
        generator=generator,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    student_logits = student(x)
    with torch.no_grad():
        teacher_logits = teacher(x)

    temperature = config.distill_temperature
    student_log_probs = torch.nn.functional.log_softmax(
        student_logits.reshape(-1, int(student_logits.shape[-1])) / temperature,
        dim=-1,
    )
    teacher_probs = torch.nn.functional.softmax(
        teacher_logits.reshape(-1, int(teacher_logits.shape[-1])) / temperature,
        dim=-1,
    )
    soft_loss = torch.nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature * temperature)
    hard_loss = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, int(student_logits.shape[-1])),
        y.reshape(-1),
    )
    loss = (
        config.distill_alpha * soft_loss
        + (1.0 - config.distill_alpha) * hard_loss
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item()), int(y.numel())


def _objective_for_path(path_id: object) -> str:
    if path_id == "FROM_SCRATCH_PRETRAINING":
        return "causal_lm_pretraining"
    if path_id == "CONTINUED_PRETRAINING":
        return "causal_lm_continued_pretraining"
    if path_id == "FULL_SFT":
        return "full_parameter_causal_sft"
    if path_id == "LORA_SFT":
        return "lora_causal_sft"
    if path_id == "QLORA_SFT":
        return "qlora_nf4_causal_sft"
    if path_id == "DPO":
        return "direct_preference_optimization"
    if path_id == "DISTILL":
        return "knowledge_distillation"
    raise ValueError(f"unsupported reference training path: {path_id!r}")


def _build_model(
    torch: Any,
    preset: ModelPreset,
    vocab_size: int = VOCAB_SIZE,
) -> Any:
    nn = torch.nn

    class Block(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(preset.d_model)
            self.attn = nn.MultiheadAttention(
                preset.d_model,
                preset.n_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.ln2 = nn.LayerNorm(preset.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(preset.d_model, preset.d_ff),
                nn.GELU(),
                nn.Linear(preset.d_ff, preset.d_model),
            )

        def forward(self, x: Any, mask: Any) -> Any:
            normalized = self.ln1(x)
            attended, _ = self.attn(
                normalized,
                normalized,
                normalized,
                attn_mask=mask,
                need_weights=False,
            )
            x = x + attended
            return x + self.mlp(self.ln2(x))

    class ByteCausalLM(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, preset.d_model)
            self.position_embedding = nn.Embedding(
                preset.context_length,
                preset.d_model,
            )
            self.blocks = nn.ModuleList([Block() for _ in range(preset.n_layers)])
            self.final_norm = nn.LayerNorm(preset.d_model)
            self.lm_head = nn.Linear(preset.d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight

            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, tokens: Any) -> Any:
            batch, sequence = tokens.shape
            if sequence > preset.context_length:
                raise ValueError("sequence exceeds configured context length")
            positions = torch.arange(sequence, device=tokens.device)
            hidden = self.token_embedding(tokens) + self.position_embedding(positions)
            mask = torch.triu(
                torch.ones(
                    sequence,
                    sequence,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                diagonal=1,
            )
            for block in self.blocks:
                hidden = block(hidden, mask)
            return self.lm_head(self.final_norm(hidden))

    return ByteCausalLM()


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))




def _training_detail_int(details: dict[str, object], key: str) -> int:
    value = details.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"training detail {key} must be a nonnegative integer")
    return value


def _trainable_parameter_count(model: Any) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def _apply_lora_parametrizations(
    torch: Any,
    model: Any,
    config: ReferenceConfig,
) -> dict[str, object]:
    """Freeze base weights and apply trainable low-rank deltas to transformer projections."""

    nn = torch.nn
    parametrize = torch.nn.utils.parametrize
    rank = config.lora_rank
    alpha = config.lora_alpha

    class LoRAWeight(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, out_features: int, in_features: int) -> None:
            super().__init__()
            if rank > min(out_features, in_features):
                raise ValueError(
                    "lora_rank exceeds the smallest targeted projection dimension"
                )
            self.a = nn.Parameter(torch.empty(rank, in_features))
            self.b = nn.Parameter(torch.zeros(out_features, rank))
            self.scale = alpha / rank
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

        def forward(self, original: Any) -> Any:
            return original + (self.b @ self.a) * self.scale

    for parameter in model.parameters():
        parameter.requires_grad = False

    targets: list[tuple[Any, str, str]] = []
    for block_index, block in enumerate(model.blocks):
        targets.append(
            (
                block.attn,
                "in_proj_weight",
                f"blocks.{block_index}.attn.in_proj_weight",
            )
        )
        targets.append(
            (
                block.attn.out_proj,
                "weight",
                f"blocks.{block_index}.attn.out_proj.weight",
            )
        )
        targets.append(
            (
                block.mlp[0],
                "weight",
                f"blocks.{block_index}.mlp.0.weight",
            )
        )
        targets.append(
            (
                block.mlp[2],
                "weight",
                f"blocks.{block_index}.mlp.2.weight",
            )
        )

    target_names: list[str] = []
    for module, parameter_name, target_name in targets:
        weight = getattr(module, parameter_name)
        if weight.ndim != 2:
            raise ValueError(f"LoRA target is not a matrix: {target_name}")
        out_features, in_features = weight.shape
        parametrization = LoRAWeight(int(out_features), int(in_features))
        parametrize.register_parametrization(
            module,
            parameter_name,
            parametrization,
            unsafe=False,
        )
        target_names.append(target_name)

    trainable = _trainable_parameter_count(model)
    if trainable <= 0:
        raise ValueError("LoRA produced no trainable parameters")

    return {
        "method": "lora",
        "rank": rank,
        "alpha": alpha,
        "target_count": len(target_names),
        "targets": target_names,
        "trainable_parameter_count": trainable,
        "merged_output": True,
    }


def _quantize_nf4_matrix(
    torch: Any,
    weight: Any,
    *,
    block_size: int = QLORA_BLOCK_SIZE,
) -> tuple[Any, Any]:
    """Pack a matrix into blockwise NF4 indices plus float32 absmax scales."""

    if weight.ndim != 2:
        raise ValueError("NF4 quantization requires a matrix")
    flat = weight.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    value_count = int(flat.numel())
    if value_count <= 0:
        raise ValueError("NF4 quantization requires a nonempty matrix")

    block_count = math.ceil(value_count / block_size)
    padded_count = block_count * block_size
    if padded_count != value_count:
        padded = torch.zeros(padded_count, dtype=torch.float32)
        padded[:value_count] = flat
    else:
        padded = flat

    blocks = padded.reshape(block_count, block_size)
    scales = blocks.abs().amax(dim=1)
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    normalized = blocks / scales[:, None]
    codebook = torch.tensor(NF4_CODEBOOK, dtype=torch.float32)
    distances = (normalized[:, :, None] - codebook[None, None, :]).abs()
    indices = distances.argmin(dim=2).to(dtype=torch.uint8).reshape(-1)[:value_count]

    if int(indices.numel()) % 2:
        indices = torch.cat(
            [indices, torch.tensor([7], dtype=torch.uint8)],
            dim=0,
        )

    low = indices[0::2]
    high = indices[1::2] << 4
    packed = low | high
    return packed.contiguous(), scales.contiguous()


def _apply_qlora_parametrizations(
    torch: Any,
    model: Any,
    config: ReferenceConfig,
) -> dict[str, object]:
    """Apply real frozen blockwise-NF4 base weights with trainable LoRA deltas.

    This reference path implements the defining QLoRA property used by the product:
    gradients update only low-rank adapters while targeted base projection matrices are
    stored as packed 4-bit NF4 values. It intentionally does not claim bitsandbytes
    double-quantization or paged-optimizer behavior.
    """

    nn = torch.nn
    parametrize = torch.nn.utils.parametrize
    rank = config.lora_rank
    alpha = config.lora_alpha
    codebook_values = NF4_CODEBOOK
    block_size = QLORA_BLOCK_SIZE

    class QLoRAWeight(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(
            self,
            out_features: int,
            in_features: int,
            scales: Any,
        ) -> None:
            super().__init__()
            if rank > min(out_features, in_features):
                raise ValueError(
                    "lora_rank exceeds the smallest targeted projection dimension"
                )
            self.out_features = out_features
            self.in_features = in_features
            self.value_count = out_features * in_features
            self.block_size = block_size
            self.a = nn.Parameter(torch.empty(rank, in_features))
            self.b = nn.Parameter(torch.zeros(out_features, rank))
            self.scale = alpha / rank
            self.register_buffer("nf4_scales", scales.to(dtype=torch.float32))
            self.register_buffer(
                "nf4_codebook",
                torch.tensor(codebook_values, dtype=torch.float32),
            )
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

        def _dequantize(self, packed: Any) -> Any:
            packed = packed.reshape(-1)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            indices = torch.empty(
                int(packed.numel()) * 2,
                dtype=torch.long,
                device=packed.device,
            )
            indices[0::2] = low.to(dtype=torch.long)
            indices[1::2] = high.to(dtype=torch.long)
            indices = indices[: self.value_count]
            values = self.nf4_codebook[indices]
            block_ids = torch.arange(
                self.value_count,
                dtype=torch.long,
                device=packed.device,
            ) // self.block_size
            base = values * self.nf4_scales[block_ids]
            return base.reshape(self.out_features, self.in_features)

        def forward(self, original: Any) -> Any:
            if original.dtype == torch.uint8 and original.ndim == 1:
                base = self._dequantize(original)
            else:
                # Registration may inspect the original full-precision tensor before
                # Frontierwright replaces it with packed NF4 storage.
                base = original
            delta = (self.b @ self.a) * self.scale
            return base.to(dtype=delta.dtype) + delta

    for parameter in model.parameters():
        parameter.requires_grad = False

    targets: list[tuple[Any, str, str]] = []
    for block_index, block in enumerate(model.blocks):
        targets.append(
            (
                block.attn,
                "in_proj_weight",
                f"blocks.{block_index}.attn.in_proj_weight",
            )
        )
        targets.append(
            (
                block.attn.out_proj,
                "weight",
                f"blocks.{block_index}.attn.out_proj.weight",
            )
        )
        targets.append(
            (
                block.mlp[0],
                "weight",
                f"blocks.{block_index}.mlp.0.weight",
            )
        )
        targets.append(
            (
                block.mlp[2],
                "weight",
                f"blocks.{block_index}.mlp.2.weight",
            )
        )

    target_names: list[str] = []
    quantized_storage_bytes = 0
    full_precision_target_bytes = 0
    for module, parameter_name, target_name in targets:
        weight = getattr(module, parameter_name)
        if weight.ndim != 2:
            raise ValueError(f"QLoRA target is not a matrix: {target_name}")
        out_features, in_features = weight.shape
        packed, scales = _quantize_nf4_matrix(
            torch,
            weight,
            block_size=block_size,
        )
        parametrization = QLoRAWeight(
            int(out_features),
            int(in_features),
            scales,
        )
        parametrize.register_parametrization(
            module,
            parameter_name,
            parametrization,
            unsafe=True,
        )
        parametrization_list = getattr(module.parametrizations, parameter_name)
        parametrization_list.original = nn.Parameter(
            packed,
            requires_grad=False,
        )
        quantized_storage_bytes += int(packed.numel() * packed.element_size())
        quantized_storage_bytes += int(scales.numel() * scales.element_size())
        quantized_storage_bytes += len(codebook_values) * 4
        full_precision_target_bytes += int(
            int(out_features) * int(in_features) * 4
        )
        target_names.append(target_name)

    trainable = _trainable_parameter_count(model)
    if trainable <= 0:
        raise ValueError("QLoRA produced no trainable parameters")

    return {
        "method": "qlora",
        "rank": rank,
        "alpha": alpha,
        "quantization_type": "nf4",
        "quantization_bits": 4,
        "quantization_block_size": block_size,
        "double_quantization": False,
        "paged_optimizer": False,
        "target_count": len(target_names),
        "targets": target_names,
        "trainable_parameter_count": trainable,
        "quantized_target_storage_bytes": quantized_storage_bytes,
        "full_precision_target_storage_bytes": full_precision_target_bytes,
        "merged_output": True,
    }


def _merge_lora_parametrizations(torch: Any, model: Any) -> None:
    parametrize = torch.nn.utils.parametrize
    nn = torch.nn

    def merge(module: Any, parameter_name: str) -> None:
        parametrizations = getattr(module.parametrizations, parameter_name)
        original = parametrizations.original
        if original.dtype == torch.uint8:
            # QLoRA stores the frozen base tensor as packed NF4. Materialize the
            # dequantized base plus trained LoRA delta before removing the packing
            # parametrization; PyTorch cannot set a float tensor into uint8 storage.
            materialized = getattr(module, parameter_name).detach().clone()
            parametrize.remove_parametrizations(
                module,
                parameter_name,
                leave_parametrized=False,
            )
            setattr(
                module,
                parameter_name,
                nn.Parameter(materialized, requires_grad=False),
            )
            return
        parametrize.remove_parametrizations(
            module,
            parameter_name,
            leave_parametrized=True,
        )

    for block in model.blocks:
        merge(block.attn, "in_proj_weight")
        merge(block.attn.out_proj, "weight")
        merge(block.mlp[0], "weight")
        merge(block.mlp[2], "weight")


def _sample_batch(
    torch: Any,
    corpus: Any,
    *,
    batch_size: int,
    context_length: int,
    generator: Any,
    device: str,
) -> tuple[Any, Any]:
    required = context_length + 1
    if corpus.numel() < required:
        repeats = math.ceil(required / int(corpus.numel())) + 1
        corpus = corpus.repeat(repeats)

    maximum = int(corpus.numel()) - required
    if maximum <= 0:
        starts = torch.zeros(batch_size, dtype=torch.long)
    else:
        starts = torch.randint(
            0,
            maximum + 1,
            (batch_size,),
            generator=generator,
        )

    xs: list[Any] = []
    ys: list[Any] = []
    for start in starts.tolist():
        chunk = corpus[start : start + required]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.stack(xs).to(device=device, non_blocking=True)
    y = torch.stack(ys).to(device=device, non_blocking=True)
    return x, y


def _process_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _reference_tokenizer_for_model(model_source_path: str) -> tuple[dict[str, Any], int]:
    source = _native_path(model_source_path).expanduser().resolve()
    tokenizer_path = source / "tokenizer.json"
    config_path = source / "config.json"
    if not tokenizer_path.is_file() or not config_path.is_file():
        raise ValueError("reference model must contain config.json and tokenizer.json")

    tokenizer_payload = load_reference_tokenizer(tokenizer_path)
    vocab_size = tokenizer_vocab_size(tokenizer_payload)
    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("reference model config.json is unreadable") from exc
    if not isinstance(config_payload, dict):
        raise ValueError("reference model config.json must contain an object")
    config_vocab_size = config_payload.get("vocab_size")
    if (
        isinstance(config_vocab_size, bool)
        or not isinstance(config_vocab_size, int)
        or config_vocab_size != vocab_size
    ):
        raise ValueError(
            "reference model config/tokenizer vocab_size mismatch"
        )
    return tokenizer_payload, vocab_size


def _encode_reference_corpus(
    corpus_bytes: bytes,
    *,
    model_source_path: str | None,
) -> tuple[list[int], dict[str, Any] | None, int]:
    if model_source_path is None:
        return list(corpus_bytes), None, VOCAB_SIZE
    tokenizer_payload, vocab_size = _reference_tokenizer_for_model(model_source_path)
    token_ids = encode_with_tokenizer(corpus_bytes, tokenizer_payload)
    if not token_ids:
        raise ValueError("tokenizer produced an empty corpus")
    return token_ids, tokenizer_payload, vocab_size


def _load_reference_model(
    torch: Any,
    preset: ModelPreset,
    *,
    model_source_path: str,
    device: str,
) -> Any:
    source = _native_path(model_source_path).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("model_source_path must be a model directory")
    weights_path = source / "pytorch_model.bin"
    quantized_weights_path = source / "quantized_model.pt"
    quantized_metadata_path = source / "frontierwright-quantized.json"
    config_path = source / "config.json"
    if not config_path.is_file():
        raise ValueError("reference model must contain config.json")

    root_config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(root_config, dict)
        or root_config.get("frontierwright_reference_backend") != REFERENCE_BACKEND_ID
        or root_config.get("preset") != preset.name
    ):
        raise ValueError(
            "model_source_path is not a compatible Frontierwright reference model "
            f"for preset {preset.name}"
        )

    _, vocab_size = _reference_tokenizer_for_model(model_source_path)
    model = _build_model(torch, preset, vocab_size=vocab_size)
    if weights_path.is_file():
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
    elif quantized_weights_path.is_file() and quantized_metadata_path.is_file():
        state_dict = load_quantized_reference_state(torch, source)
    else:
        raise ValueError(
            "reference model must contain pytorch_model.bin or a supported "
            "Frontierwright quantized checkpoint"
        )
    model.load_state_dict(state_dict)
    return model.to(device)


def _training_objects(
    torch: Any,
    config: ReferenceConfig,
    corpus_bytes: bytes,
    *,
    path_id: str,
    model_source_path: str | None = None,
) -> tuple[Any, Any, Any, Any, str, dict[str, object]]:
    torch.manual_seed(config.seed)
    device = _select_device(torch, config.device)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    token_ids, tokenizer_payload, vocab_size = _encode_reference_corpus(
        corpus_bytes,
        model_source_path=model_source_path,
    )
    if model_source_path is None:
        model = _build_model(torch, config.preset, vocab_size=vocab_size).to(device)
    else:
        model = _load_reference_model(
            torch,
            config.preset,
            model_source_path=model_source_path,
            device=device,
        )

    base_parameter_count = _parameter_count(model)
    training_details: dict[str, object] = {
        "method": "full_parameter",
        "base_parameter_count": base_parameter_count,
        "trainable_parameter_count": base_parameter_count,
        "vocab_size": vocab_size,
        "tokenizer_fingerprint": (
            tokenizer_fingerprint(tokenizer_payload)
            if tokenizer_payload is not None
            else None
        ),
    }
    if path_id == "LORA_SFT":
        training_details = {
            **_apply_lora_parametrizations(torch, model, config),
            "base_parameter_count": base_parameter_count,
        }
    elif path_id == "QLORA_SFT":
        training_details = {
            **_apply_qlora_parametrizations(torch, model, config),
            "base_parameter_count": base_parameter_count,
        }

    model = model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("training path produced no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    corpus = torch.tensor(list(corpus_bytes), dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    return model, optimizer, corpus, generator, device, training_details


def _train_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    corpus: Any,
    generator: Any,
    *,
    config: ReferenceConfig,
    device: str,
) -> tuple[float, int]:
    model.train()
    x, y = _sample_batch(
        torch,
        corpus,
        batch_size=config.batch_size,
        context_length=config.preset.context_length,
        generator=generator,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, int(logits.shape[-1])),
        y.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item()), int(y.numel())



def _evaluation_batches(
    torch: Any,
    token_ids: list[int],
    *,
    config: ReferenceEvaluationConfig,
    device: str,
) -> list[tuple[Any, Any]]:
    corpus = torch.tensor(token_ids, dtype=torch.long)
    required = config.preset.context_length + 1
    if int(corpus.numel()) < required:
        raise ValueError(
            "evaluation corpus is too small for one full context window; "
            f"need at least {required} tokens"
        )

    starts = list(
        range(
            0,
            int(corpus.numel()) - required + 1,
            config.preset.context_length,
        )
    )
    maximum_windows = config.max_batches * config.batch_size
    starts = starts[:maximum_windows]
    if not starts:
        raise ValueError("evaluation corpus produced no full context windows")

    batches: list[tuple[Any, Any]] = []
    for offset in range(0, len(starts), config.batch_size):
        batch_starts = starts[offset : offset + config.batch_size]
        xs: list[Any] = []
        ys: list[Any] = []
        for start in batch_starts:
            chunk = corpus[start : start + required]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        batches.append(
            (
                torch.stack(xs).to(device=device, non_blocking=True),
                torch.stack(ys).to(device=device, non_blocking=True),
            )
        )
    return batches


def _evaluate(
    request: dict[str, Any],
    config: ReferenceEvaluationConfig,
) -> dict[str, object]:
    torch = _import_torch()
    model_source_path = request.get("model_source_path")
    dataset_source = request.get("dataset_source_path")
    if not isinstance(model_source_path, str) or not model_source_path:
        raise ValueError("model_source_path is required")
    if not isinstance(dataset_source, str) or not dataset_source:
        raise ValueError("dataset_source_path is required")

    device = _select_device(torch, config.device)
    model = _load_reference_model(
        torch,
        config.preset,
        model_source_path=model_source_path,
        device=device,
    )
    model.eval()

    corpus_bytes = _read_corpus(
        _native_path(dataset_source),
        max_bytes=config.max_dataset_bytes,
    )
    tokenizer_payload, _ = _reference_tokenizer_for_model(model_source_path)
    token_ids = encode_with_tokenizer(corpus_bytes, tokenizer_payload)
    batches = _evaluation_batches(
        torch,
        token_ids,
        config=config,
        device=device,
    )

    total_negative_log_likelihood = 0.0
    tokens_evaluated = 0
    windows_evaluated = 0
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            loss_sum = torch.nn.functional.cross_entropy(
                logits.reshape(-1, int(logits.shape[-1])),
                y.reshape(-1),
                reduction="sum",
            )
            total_negative_log_likelihood += float(loss_sum.detach().cpu().item())
            tokens_evaluated += int(y.numel())
            windows_evaluated += int(y.shape[0])
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)

    if tokens_evaluated <= 0:
        raise ValueError("evaluation produced no scored tokens")
    cross_entropy = total_negative_log_likelihood / tokens_evaluated
    perplexity = math.exp(cross_entropy)
    if not math.isfinite(cross_entropy) or not math.isfinite(perplexity):
        raise ValueError("evaluation produced non-finite metrics")

    return {
        "schema_version": 1,
        "ok": True,
        "operation": "evaluate",
        "metrics": {
            "backend_id": REFERENCE_BACKEND_ID,
            "preset": config.preset.name,
            "device": device,
            "cross_entropy_nats_per_token": cross_entropy,
            "perplexity": perplexity,
            "tokens_evaluated": tokens_evaluated,
            "windows_evaluated": windows_evaluated,
            "batches_evaluated": len(batches),
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_evaluated / elapsed,
            "parameter_count": _parameter_count(model),
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
        },
    }


def _generate(
    request: dict[str, Any],
    config: ReferenceGenerationConfig,
) -> dict[str, object]:
    torch = _import_torch()
    model_source_path = request.get("model_source_path")
    prompt = request.get("prompt")
    if not isinstance(model_source_path, str) or not model_source_path:
        raise ValueError("model_source_path is required")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")

    prompt_bytes = prompt.encode("utf-8")
    if not prompt_bytes:
        raise ValueError("prompt must encode to at least one byte")
    tokenizer_payload, _ = _reference_tokenizer_for_model(model_source_path)
    prompt_token_ids = encode_with_tokenizer(prompt_bytes, tokenizer_payload)
    if not prompt_token_ids:
        raise ValueError("prompt tokenizer produced no tokens")

    device = _select_device(torch, config.device)
    model = _load_reference_model(
        torch,
        config.preset,
        model_source_path=model_source_path,
        device=device,
    )
    model.eval()

    generated_ids: list[int] = []
    token_history = list(prompt_token_ids)
    generator = None
    if config.temperature > 0.0:
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            context = token_history[-config.preset.context_length :]
            tokens = torch.tensor(
                [context],
                dtype=torch.long,
                device=device,
            )
            logits = model(tokens)[0, -1]
            if config.temperature == 0.0:
                next_token = int(torch.argmax(logits).item())
            else:
                probabilities = torch.softmax(logits / config.temperature, dim=-1)
                sampled = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
                next_token = int(sampled.item())
            generated_ids.append(next_token)
            token_history.append(next_token)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)
    continuation_bytes = decode_with_tokenizer(generated_ids, tokenizer_payload)
    continuation_text = continuation_bytes.decode("utf-8", errors="replace")

    return {
        "schema_version": 1,
        "ok": True,
        "operation": "generate",
        "prompt": prompt,
        "continuation_text": continuation_text,
        "generated_text": prompt + continuation_text,
        "generated_token_ids": generated_ids,
        "metrics": {
            "backend_id": REFERENCE_BACKEND_ID,
            "preset": config.preset.name,
            "device": device,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "seed": config.seed,
            "prompt_bytes": len(prompt_bytes),
            "prompt_tokens": len(prompt_token_ids),
            "context_tokens_used": min(
                len(prompt_token_ids),
                config.preset.context_length,
            ),
            "generated_tokens": len(generated_ids),
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(generated_ids) / elapsed,
            "parameter_count": _parameter_count(model),
            "stop_reason": "max_new_tokens",
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
        },
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _profile_inference(
    request: dict[str, Any],
    config: ReferenceInferenceProfileConfig,
) -> dict[str, object]:
    torch = _import_torch()
    model_source_path = request.get("model_source_path")
    if not isinstance(model_source_path, str) or not model_source_path:
        raise ValueError("model_source_path is required")

    device = _select_device(torch, config.device)
    model = _load_reference_model(
        torch,
        config.preset,
        model_source_path=model_source_path,
        device=device,
    )
    model.eval()
    profile_prompt = "Frontierwright inference profile:"
    prompt_bytes = profile_prompt.encode("utf-8")
    tokenizer_payload, _ = _reference_tokenizer_for_model(model_source_path)
    prompt_token_ids = encode_with_tokenizer(prompt_bytes, tokenizer_payload)

    def run_once() -> tuple[float, list[int]]:
        token_history = list(prompt_token_ids)
        generated_ids: list[int] = []
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                context = token_history[-config.preset.context_length :]
                tokens = torch.tensor(
                    [context],
                    dtype=torch.long,
                    device=device,
                )
                next_token = int(torch.argmax(model(tokens)[0, -1]).item())
                generated_ids.append(next_token)
                token_history.append(next_token)
        if device == "cuda":
            torch.cuda.synchronize()
        return max(time.perf_counter() - start, 1e-9), generated_ids

    for _ in range(config.warmup_runs):
        run_once()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    throughputs: list[float] = []
    rss_samples: list[int] = []
    sample_ids: list[int] = []
    for _ in range(config.measured_runs):
        elapsed, generated_ids = run_once()
        latencies.append(elapsed)
        throughputs.append(config.max_new_tokens / elapsed)
        rss = _process_rss_bytes()
        if rss is not None:
            rss_samples.append(rss)
        if not sample_ids:
            sample_ids = generated_ids

    mean_latency = sum(latencies) / len(latencies)
    mean_throughput = sum(throughputs) / len(throughputs)
    sample_text = decode_with_tokenizer(
        sample_ids, tokenizer_payload
    ).decode("utf-8", errors="replace")
    current_vram = (
        int(torch.cuda.memory_allocated()) if device == "cuda" else None
    )
    peak_vram = (
        int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    )

    return {
        "schema_version": 1,
        "ok": True,
        "operation": "profile",
        "metrics": {
            "backend_id": REFERENCE_BACKEND_ID,
            "preset": config.preset.name,
            "device": device,
            "measurement_scope": "steady_state_generation_excludes_model_load",
            "profile_prompt": profile_prompt,
            "prompt_bytes": len(prompt_bytes),
            "prompt_tokens": len(prompt_token_ids),
            "max_new_tokens": getattr(config, "max_" + "new_tokens"),
            "warmup_runs": config.warmup_runs,
            "measured_runs": config.measured_runs,
            "latency_seconds_runs": latencies,
            "latency_seconds_mean": mean_latency,
            "latency_seconds_p50": _median(latencies),
            "tokens_per_second_runs": throughputs,
            "tokens_per_second_mean": mean_throughput,
            "tokens_per_second_p50": _median(throughputs),
            "max_sampled_process_rss_bytes": (
                max(rss_samples) if rss_samples else None
            ),
            "cuda_memory_allocated_bytes": current_vram,
            "peak_vram_bytes": peak_vram,
            "parameter_count": _parameter_count(model),
            "sample_generated_token_ids": sample_ids,
            "sample_continuation_text": sample_text,
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
        },
    }


def _birth(request: dict[str, Any]) -> dict[str, object]:
    """Materialize exact initial bytes for a Frontierwright zero-model root."""

    torch = _import_torch()
    preset_name = request.get("preset", "zero-8m")
    seed = request.get("seed", 42)
    output_root = request.get("output_root")
    tok_path_raw = request.get("tokenizer_path")
    expected_tok_hash = request.get("tokenizer_fingerprint")

    if not isinstance(preset_name, str) or preset_name not in PRESETS:
        raise ValueError(f"unknown zero-model preset: {preset_name}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("birth seed must be a nonnegative integer")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root is required")
    if expected_tok_hash is not None and (
        not isinstance(expected_tok_hash, str) or not expected_tok_hash
    ):
        raise ValueError("tokenizer_fingerprint must be a nonempty string when supplied")

    tok_payload: dict[str, Any] | None = None
    tok_hash: str | None = None
    vocab_size = VOCAB_SIZE
    if tok_path_raw is not None:
        if not isinstance(tok_path_raw, str) or not tok_path_raw:
            raise ValueError("tokenizer_path must be a nonempty string when supplied")
        tok_path = _native_path(tok_path_raw).expanduser().resolve()
        tok_payload = load_reference_tokenizer(tok_path)
        tok_hash = tokenizer_fingerprint(tok_payload)
        if expected_tok_hash is not None and tok_hash != expected_tok_hash:
            raise ValueError("tokenizer artifact fingerprint does not match birth request")
        vocab_size = tokenizer_vocab_size(tok_payload)
    elif expected_tok_hash is not None:
        raise ValueError("tokenizer_fingerprint requires tokenizer_path")

    preset = PRESETS[preset_name]
    torch.manual_seed(seed)
    model = _build_model(torch, preset, vocab_size=vocab_size).to("cpu")
    parameter_count = _parameter_count(model)

    output = _native_path(output_root).expanduser().resolve() / "model"
    output.mkdir(parents=True, exist_ok=False)

    config_payload = {
        "architectures": ["FrontierwrightByteCausalLM"],
        "model_type": "frontierwright_byte_causal_lm",
        "frontierwright_reference_backend": REFERENCE_BACKEND_ID,
        "preset": preset.name,
        "vocab_size": vocab_size,
        "d_model": preset.d_model,
        "n_layers": preset.n_layers,
        "n_heads": preset.n_heads,
        "d_ff": preset.d_ff,
        "context_length": preset.context_length,
        "parameter_count": parameter_count,
    }
    (output / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2) + chr(10),
        encoding="utf-8",
    )
    tok_output = tok_payload or {
        "type": "frontierwright-byte-level",
        "version": 1,
        "vocab_size": VOCAB_SIZE,
        "mapping": "token id equals byte value 0..255",
    }
    (output / "tokenizer.json").write_text(
        json.dumps(tok_output, sort_keys=True, indent=2) + chr(10),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output / "pytorch_model.bin")
    birth_metadata = {
        "backend_id": REFERENCE_BACKEND_ID,
        "preset": preset.name,
        "seed": seed,
        "parameter_count": parameter_count,
        "vocab_size": vocab_size,
        "tokenizer_fingerprint": tok_hash,
        "torch_version": str(torch.__version__),
        "python_version": sys.version.split()[0],
        "device": "cpu",
        "trained_steps": 0,
    }
    (output / "birth_metadata.json").write_text(
        json.dumps(birth_metadata, sort_keys=True, indent=2) + chr(10),
        encoding="utf-8",
    )

    return {
        "schema_version": 1,
        "ok": True,
        "operation": "birth",
        "output_model_path": _reported_child_path(output_root, "model"),
        "metrics": birth_metadata,
    }
def _calibrate(request: dict[str, Any], config: ReferenceConfig) -> dict[str, object]:
    torch = _import_torch()
    path_id = request.get("path_id")
    if not isinstance(path_id, str):
        raise ValueError("path_id must be a string")
    objective = _objective_for_path(path_id)
    dataset_source = request.get("dataset_source_path")
    if not isinstance(dataset_source, str) or not dataset_source:
        raise ValueError("dataset_source_path is required")
    model_source_path = request.get("model_source_path")
    if model_source_path is not None and not isinstance(model_source_path, str):
        raise ValueError("model_source_path must be a string when supplied")

    if path_id == "DPO":
        if not isinstance(model_source_path, str) or not model_source_path:
            raise ValueError("DPO requires a materialized current model")
        preference_pairs = _read_preference_pairs(
            _native_path(dataset_source),
            max_bytes=config.max_dataset_bytes,
        )
        (
            model,
            reference_model,
            optimizer,
            generator,
            dpo_tokenizer_payload,
            device,
            training_details,
        ) = _dpo_training_objects(
            torch,
            config,
            preference_pairs,
            model_source_path=model_source_path,
        )
        _dpo_train_step(
            torch,
            model,
            reference_model,
            optimizer,
            preference_pairs,
            generator,
            dpo_tokenizer_payload,
            config=config,
            device=device,
        )
    elif path_id == "DISTILL":
        if not isinstance(model_source_path, str) or not model_source_path:
            raise ValueError("DISTILL requires a materialized teacher model")
        corpus_bytes = _read_corpus(
            _native_path(dataset_source),
            max_bytes=config.max_dataset_bytes,
        )
        (
            model,
            teacher_model,
            optimizer,
            corpus,
            generator,
            device,
            training_details,
        ) = _distillation_training_objects(
            torch,
            config,
            corpus_bytes,
            model_source_path=model_source_path,
        )
        _kd_step(
            torch,
            model,
            teacher_model,
            optimizer,
            corpus,
            generator,
            config=config,
            device=device,
        )
    else:
        corpus_bytes = _read_corpus(
            _native_path(dataset_source),
            max_bytes=config.max_dataset_bytes,
        )
        model, optimizer, corpus, generator, device, training_details = _training_objects(
            torch,
            config,
            corpus_bytes,
            path_id=path_id,
            model_source_path=model_source_path,
        )
        _train_step(
            torch,
            model,
            optimizer,
            corpus,
            generator,
            config=config,
            device=device,
        )
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rss_peak = _process_rss_bytes()
    token_count = 0
    start = time.perf_counter()
    for _ in range(config.calibration_steps):
        if path_id == "DPO":
            _, tokens = _dpo_train_step(
                torch,
                model,
                reference_model,
                optimizer,
                preference_pairs,
                generator,
                dpo_tokenizer_payload,
                config=config,
                device=device,
            )
        elif path_id == "DISTILL":
            _, tokens = _kd_step(
                torch,
                model,
                teacher_model,
                optimizer,
                corpus,
                generator,
                config=config,
                device=device,
            )
        else:
            _, tokens = _train_step(
                torch,
                model,
                optimizer,
                corpus,
                generator,
                config=config,
                device=device,
            )
        token_count += tokens
        current_rss = _process_rss_bytes()
        if current_rss is not None:
            rss_peak = current_rss if rss_peak is None else max(rss_peak, current_rss)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)
    step_time = elapsed / config.calibration_steps
    tokens_per_second = token_count / elapsed
    peak_vram = (
        int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    )
    parameter_count = _training_detail_int(training_details, "base_parameter_count")
    projected_storage = int(parameter_count * 4 * 1.05) + 16 * 1024
    calibration_preset = (
        config.student_preset
        if path_id == "DISTILL" and config.student_preset is not None
        else config.preset
    )

    return {
        "schema_version": 1,
        "ok": True,
        "feasible": True,
        "representative_steps": config.calibration_steps,
        "step_time_seconds": step_time,
        "tokens_per_second": tokens_per_second,
        "peak_vram_bytes": peak_vram,
        "peak_ram_bytes": rss_peak,
        "projected_storage_bytes": projected_storage,
        "projected_wall_seconds": step_time * config.steps,
        "gpu_count": int(torch.cuda.device_count()) if device == "cuda" else 0,
        "device": device,
        "parameter_count": parameter_count,
        "preset": calibration_preset.name,
        "objective": objective,
        "training": training_details,
    }


def _train(
    request: dict[str, Any],
    config: ReferenceConfig,
) -> dict[str, object]:
    torch = _import_torch()
    path_id = request.get("path_id")
    if not isinstance(path_id, str):
        raise ValueError("path_id must be a string")
    objective = _objective_for_path(path_id)
    dataset_source = request.get("dataset_source_path")
    output_root = request.get("output_root")
    if not isinstance(dataset_source, str) or not dataset_source:
        raise ValueError("dataset_source_path is required")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root is required")

    model_source_path = request.get("model_source_path")
    if model_source_path is not None and not isinstance(model_source_path, str):
        raise ValueError("model_source_path must be a string when supplied")

    if path_id == "DPO":
        if not isinstance(model_source_path, str) or not model_source_path:
            raise ValueError("DPO requires a materialized current model")
        preference_pairs = _read_preference_pairs(
            _native_path(dataset_source),
            max_bytes=config.max_dataset_bytes,
        )
        (
            model,
            reference_model,
            optimizer,
            generator,
            dpo_tokenizer_payload,
            device,
            training_details,
        ) = _dpo_training_objects(
            torch,
            config,
            preference_pairs,
            model_source_path=model_source_path,
        )
    elif path_id == "DISTILL":
        if not isinstance(model_source_path, str) or not model_source_path:
            raise ValueError("DISTILL requires a materialized teacher model")
        corpus_bytes = _read_corpus(
            _native_path(dataset_source),
            max_bytes=config.max_dataset_bytes,
        )
        (
            model,
            teacher_model,
            optimizer,
            corpus,
            generator,
            device,
            training_details,
        ) = _distillation_training_objects(
            torch,
            config,
            corpus_bytes,
            model_source_path=model_source_path,
        )
    else:
        corpus_bytes = _read_corpus(
            _native_path(dataset_source),
            max_bytes=config.max_dataset_bytes,
        )
        model, optimizer, corpus, generator, device, training_details = _training_objects(
            torch,
            config,
            corpus_bytes,
            path_id=path_id,
            model_source_path=model_source_path,
        )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses: list[float] = []
    token_count = 0
    start = time.perf_counter()
    for _ in range(config.steps):
        if path_id == "DPO":
            loss, tokens = _dpo_train_step(
                torch,
                model,
                reference_model,
                optimizer,
                preference_pairs,
                generator,
                dpo_tokenizer_payload,
                config=config,
                device=device,
            )
        elif path_id == "DISTILL":
            loss, tokens = _kd_step(
                torch,
                model,
                teacher_model,
                optimizer,
                corpus,
                generator,
                config=config,
                device=device,
            )
        else:
            loss, tokens = _train_step(
                torch,
                model,
                optimizer,
                corpus,
                generator,
                config=config,
                device=device,
            )
        losses.append(loss)
        token_count += tokens
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)

    if path_id in {"LORA_SFT", "QLORA_SFT"}:
        _merge_lora_parametrizations(torch, model)
    parameter_count = _parameter_count(model)
    expected_parameter_count = _training_detail_int(training_details, "base_parameter_count")
    if parameter_count != expected_parameter_count:
        raise ValueError(
            "merged adapter checkpoint parameter count differs from the base model"
        )

    output = (_native_path(output_root).expanduser().resolve() / "model")
    output.mkdir(parents=True, exist_ok=True)

    output_preset = (
        config.student_preset
        if path_id == "DISTILL" and config.student_preset is not None
        else config.preset
    )
    if not isinstance(model_source_path, str) or not model_source_path:
        raise ValueError("trained reference output requires a source tokenizer")
    output_tokenizer_payload, output_vocab_size = _reference_tokenizer_for_model(
        model_source_path
    )
    config_payload = {
        "architectures": ["FrontierwrightByteCausalLM"],
        "model_type": "frontierwright_byte_causal_lm",
        "frontierwright_reference_backend": REFERENCE_BACKEND_ID,
        "preset": output_preset.name,
        "vocab_size": output_vocab_size,
        "d_model": output_preset.d_model,
        "n_layers": output_preset.n_layers,
        "n_heads": output_preset.n_heads,
        "d_ff": output_preset.d_ff,
        "context_length": output_preset.context_length,
        "parameter_count": parameter_count,
    }
    (output / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        json.dumps(
            output_tokenizer_payload,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output / "pytorch_model.bin")
    (output / "training_metadata.json").write_text(
        json.dumps(
            {
                "backend_id": REFERENCE_BACKEND_ID,
                "path_id": request.get("path_id"),
                "run_id": request.get("run_id"),
                "dataset_fingerprint": request.get("dataset_fingerprint"),
                "config": config.to_dict(),
                "device": device,
                "objective": objective,
                "training": training_details,
                "steps": config.steps,
                "tokens_trained": token_count,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "schema_version": 1,
        "ok": True,
        "output_model_path": _reported_child_path(output_root, "model"),
        "metrics": {
            "backend_id": REFERENCE_BACKEND_ID,
            "preset": output_preset.name,
            "device": device,
            "objective": objective,
            "training": training_details,
            "parameter_count": parameter_count,
            "steps": config.steps,
            "tokens_trained": token_count,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed,
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device == "cuda"
                else None
            ),
            "peak_ram_bytes": _process_rss_bytes(),
        },
    }


def backend_spec_payload(python_executable: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend_id": REFERENCE_BACKEND_ID,
        "data_boundary": "LOCAL_MACHINE",
        "supported_paths": list(SUPPORTED_PATHS),
        "calibrate_argv": [
            python_executable,
            "-m",
            "frontierwright.reference_backend",
            "{request_json}",
        ],
        "train_argv": [
            python_executable,
            "-m",
            "frontierwright.reference_backend",
            "{request_json}",
        ],
        "environment": {
            "PYTHONUNBUFFERED": "1",
        },
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        return _fail(
            "REFERENCE_BACKEND_USAGE",
            "Usage: python -m frontierwright.reference_backend REQUEST_JSON",
        )

    try:
        request = json.loads(_native_path(arguments[0]).read_text(encoding="utf-8"))
        if not isinstance(request, dict) or request.get("schema_version") != 1:
            raise ValueError("request schema_version must be 1")
        if request.get("backend_id") != REFERENCE_BACKEND_ID:
            raise ValueError("request backend_id does not match reference backend")
        operation = request.get("operation")
        if operation == "birth":
            _emit(_birth(request))
            return 0
        if operation == "merge":
            _emit(
                merge_reference_models(
                    torch=_import_torch(),
                    backend_id=REFERENCE_BACKEND_ID,
                    presets=PRESETS,
                    build_model=_build_model,
                    load_reference_model=_load_reference_model,
                    parameter_count=_parameter_count,
                    native_path=_native_path,
                    reported_child_path=_reported_child_path,
                    request=request,
                )
            )
            return 0
        if operation == "quantize":
            _emit(
                quantize_reference_model(
                    torch=_import_torch(),
                    backend_id=REFERENCE_BACKEND_ID,
                    presets=PRESETS,
                    load_reference_model=_load_reference_model,
                    parameter_count=_parameter_count,
                    native_path=_native_path,
                    reported_child_path=_reported_child_path,
                    request=request,
                )
            )
            return 0
        if operation == "evaluate":
            model_source_path = request.get("model_source_path")
            if not isinstance(model_source_path, str) or not model_source_path:
                raise ValueError(
                    "reference evaluation requires a materialized model_source_path"
                )
            _emit(_evaluate(request, _load_evaluation_config(request)))
            return 0
        if operation == "generate":
            model_source_path = request.get("model_source_path")
            if not isinstance(model_source_path, str) or not model_source_path:
                raise ValueError(
                    "reference generation requires a materialized model_source_path"
                )
            _emit(_generate(request, _load_generation_config(request)))
            return 0
        if operation == "profile":
            model_source_path = request.get("model_source_path")
            if not isinstance(model_source_path, str) or not model_source_path:
                raise ValueError(
                    "reference profiling requires a materialized model_source_path"
                )
            _emit(_profile_inference(request, _load_inference_profile_config(request)))
            return 0
        path_id = request.get("path_id")
        if path_id not in SUPPORTED_PATHS:
            raise ValueError(
                "reference backend supports only: " + ", ".join(SUPPORTED_PATHS)
            )
        model_source_path = request.get("model_source_path")
        if not isinstance(model_source_path, str) or not model_source_path:
            raise ValueError(
                "reference backend training requires a materialized model_source_path"
            )
        config = _load_config(request)
        if operation == "calibrate":
            _emit(_calibrate(request, config))
            return 0
        if operation == "train":
            _emit(_train(request, config))
            return 0
        raise ValueError("operation must be birth, calibrate, or train")
    except Exception as exc:
        return _fail(type(exc).__name__.upper(), str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
