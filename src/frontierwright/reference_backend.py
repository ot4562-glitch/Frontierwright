"""Frontierwright built-in PyTorch reference backend.

This backend intentionally starts narrow: real from-scratch causal language-model
pretraining for the bundled zero-model path. It is not a compatibility layer for
arbitrary Hugging Face architectures.

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

REFERENCE_BACKEND_ID = "frontierwright-reference-pytorch-v1"
SUPPORTED_PATHS = (
    "FROM_SCRATCH_PRETRAINING",
    "CONTINUED_PRETRAINING",
)
VOCAB_SIZE = 256


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
    chunks: list[bytes] = []
    total = 0
    for path in _dataset_files(source):
        size = path.stat().st_size
        if total + size > max_bytes:
            remaining = max_bytes - total
            if remaining <= 0:
                break
            chunks.append(path.read_bytes()[:remaining])
            total += remaining
            break
        chunks.append(path.read_bytes())
        total += size
    corpus = b"\n".join(chunks)
    if not corpus:
        raise ValueError("dataset corpus is empty")
    return corpus


def _build_model(torch: Any, preset: ModelPreset) -> Any:
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
            self.token_embedding = nn.Embedding(VOCAB_SIZE, preset.d_model)
            self.position_embedding = nn.Embedding(
                preset.context_length,
                preset.d_model,
            )
            self.blocks = nn.ModuleList([Block() for _ in range(preset.n_layers)])
            self.final_norm = nn.LayerNorm(preset.d_model)
            self.lm_head = nn.Linear(preset.d_model, VOCAB_SIZE, bias=False)
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


def _training_objects(
    torch: Any,
    config: ReferenceConfig,
    corpus_bytes: bytes,
    *,
    model_source_path: str | None = None,
) -> tuple[Any, Any, Any, Any, str]:
    torch.manual_seed(config.seed)
    device = _select_device(torch, config.device)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = _build_model(torch, config.preset)
    if model_source_path is not None:
        source = _native_path(model_source_path).expanduser().resolve()
        if not source.is_dir():
            raise ValueError("model_source_path must be a model directory")
        weights_path = source / "pytorch_model.bin"
        config_path = source / "config.json"
        if not weights_path.is_file() or not config_path.is_file():
            raise ValueError(
                "born root model must contain config.json and pytorch_model.bin"
            )
        root_config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            not isinstance(root_config, dict)
            or root_config.get("frontierwright_reference_backend")
            != REFERENCE_BACKEND_ID
            or root_config.get("preset") != config.preset.name
        ):
            raise ValueError(
                "model_source_path is not a compatible Frontierwright birth root "
                f"for preset {config.preset.name}"
            )
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    corpus = torch.tensor(list(corpus_bytes), dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    return model, optimizer, corpus, generator, device


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
        logits.reshape(-1, VOCAB_SIZE),
        y.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item()), int(y.numel())



def _birth(request: dict[str, Any]) -> dict[str, object]:
    """Materialize exact initial bytes for a Frontierwright zero-model root."""

    torch = _import_torch()
    preset_name = request.get("preset", "zero-8m")
    seed = request.get("seed", 42)
    output_root = request.get("output_root")

    if not isinstance(preset_name, str) or preset_name not in PRESETS:
        raise ValueError(f"unknown zero-model preset: {preset_name}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("birth seed must be a nonnegative integer")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root is required")

    preset = PRESETS[preset_name]
    torch.manual_seed(seed)
    model = _build_model(torch, preset).to("cpu")
    parameter_count = _parameter_count(model)

    output = _native_path(output_root).expanduser().resolve() / "model"
    output.mkdir(parents=True, exist_ok=False)

    config_payload = {
        "architectures": ["FrontierwrightByteCausalLM"],
        "model_type": "frontierwright_byte_causal_lm",
        "frontierwright_reference_backend": REFERENCE_BACKEND_ID,
        "preset": preset.name,
        "vocab_size": VOCAB_SIZE,
        "d_model": preset.d_model,
        "n_layers": preset.n_layers,
        "n_heads": preset.n_heads,
        "d_ff": preset.d_ff,
        "context_length": preset.context_length,
        "parameter_count": parameter_count,
    }
    (output / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        json.dumps(
            {
                "type": "frontierwright-byte-level",
                "version": 1,
                "vocab_size": VOCAB_SIZE,
                "mapping": "token id equals byte value 0..255",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output / "pytorch_model.bin")
    (output / "birth_metadata.json").write_text(
        json.dumps(
            {
                "backend_id": REFERENCE_BACKEND_ID,
                "preset": preset.name,
                "seed": seed,
                "parameter_count": parameter_count,
                "torch_version": str(torch.__version__),
                "python_version": sys.version.split()[0],
                "device": "cpu",
                "trained_steps": 0,
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
        "operation": "birth",
        "output_model_path": _reported_child_path(output_root, "model"),
        "metrics": {
            "backend_id": REFERENCE_BACKEND_ID,
            "preset": preset.name,
            "seed": seed,
            "parameter_count": parameter_count,
            "torch_version": str(torch.__version__),
            "python_version": sys.version.split()[0],
            "device": "cpu",
            "trained_steps": 0,
        },
    }


def _calibrate(request: dict[str, Any], config: ReferenceConfig) -> dict[str, object]:
    torch = _import_torch()
    dataset_source = request.get("dataset_source_path")
    if not isinstance(dataset_source, str) or not dataset_source:
        raise ValueError("dataset_source_path is required")
    corpus_bytes = _read_corpus(
        _native_path(dataset_source),
        max_bytes=config.max_dataset_bytes,
    )
    model_source_path = request.get("model_source_path")
    if model_source_path is not None and not isinstance(model_source_path, str):
        raise ValueError("model_source_path must be a string when supplied")
    model, optimizer, corpus, generator, device = _training_objects(
        torch,
        config,
        corpus_bytes,
        model_source_path=model_source_path,
    )

    # Warm-up once so initialization/runtime setup is not charged to steady-state timing.
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
    parameter_count = _parameter_count(model)
    projected_storage = int(parameter_count * 4 * 1.05) + 16 * 1024

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
        "preset": config.preset.name,
    }


def _train(
    request: dict[str, Any],
    config: ReferenceConfig,
) -> dict[str, object]:
    torch = _import_torch()
    dataset_source = request.get("dataset_source_path")
    output_root = request.get("output_root")
    if not isinstance(dataset_source, str) or not dataset_source:
        raise ValueError("dataset_source_path is required")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root is required")

    corpus_bytes = _read_corpus(
        _native_path(dataset_source),
        max_bytes=config.max_dataset_bytes,
    )
    model_source_path = request.get("model_source_path")
    if model_source_path is not None and not isinstance(model_source_path, str):
        raise ValueError("model_source_path must be a string when supplied")
    model, optimizer, corpus, generator, device = _training_objects(
        torch,
        config,
        corpus_bytes,
        model_source_path=model_source_path,
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses: list[float] = []
    token_count = 0
    start = time.perf_counter()
    for _ in range(config.steps):
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

    output = (_native_path(output_root).expanduser().resolve() / "model")
    output.mkdir(parents=True, exist_ok=True)
    parameter_count = _parameter_count(model)

    config_payload = {
        "architectures": ["FrontierwrightByteCausalLM"],
        "model_type": "frontierwright_byte_causal_lm",
        "frontierwright_reference_backend": REFERENCE_BACKEND_ID,
        "preset": config.preset.name,
        "vocab_size": VOCAB_SIZE,
        "d_model": config.preset.d_model,
        "n_layers": config.preset.n_layers,
        "n_heads": config.preset.n_heads,
        "d_ff": config.preset.d_ff,
        "context_length": config.preset.context_length,
        "parameter_count": parameter_count,
    }
    (output / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        json.dumps(
            {
                "type": "frontierwright-byte-level",
                "version": 1,
                "vocab_size": VOCAB_SIZE,
                "mapping": "token id equals byte value 0..255",
            },
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
            "preset": config.preset.name,
            "device": device,
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
