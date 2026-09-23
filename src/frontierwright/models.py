"""Local model-source inspection, fingerprinting, and history-evidence verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from frontierwright.domain import HistoryConfidence, ModelFormat
from frontierwright.errors import FrontierwrightError

HASH_CHUNK_SIZE = 8 * 1024 * 1024

_HF_METADATA_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "adapter_config.json",
    "frontierwright-transform.json",
}
_PARTIAL_HISTORY_NAMES = {
    "trainer_state.json",
    "training_args.bin",
    "adapter_config.json",
    "README.md",
    "training_config.json",
    "run_config.json",
}
_LINEAGE_MANIFEST_NAME = "frontierwright-lineage.json"


@dataclass(frozen=True)
class FingerprintedFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ImportedModelDescriptor:
    source_path: Path
    model_format: ModelFormat
    trainable: bool
    fingerprint: str
    total_bytes: int
    files: tuple[FingerprintedFile, ...]


@dataclass(frozen=True)
class HistoryEvidenceResult:
    confidence: HistoryConfidence
    evidence_files: tuple[str, ...]
    reason: str


class ModelSourceAdapter(Protocol):
    """Private/internal model integrations can implement this protocol."""

    adapter_id: str

    def supports(self, source: Path) -> bool: ...

    def inspect(self, source: Path) -> ImportedModelDescriptor: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(root: Path, path: Path) -> str:
    root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise FrontierwrightError(
            "MODEL_PATH_ESCAPE",
            f"Model artifact escapes source directory: {path}",
            13,
        ) from exc
    return relative.as_posix()


def _is_hf_artifact(path: Path) -> bool:
    name = path.name
    if name in _HF_METADATA_NAMES:
        return True
    if name.startswith(("tokenizer", "vocab", "merges", "special_tokens", "added_tokens")):
        return path.suffix.lower() in {".json", ".txt", ".model"}
    if path.suffix.lower() in {".safetensors", ".bin"}:
        return True
    return False


def _fingerprint_files(
    root: Path,
    paths: list[Path],
) -> tuple[str, int, tuple[FingerprintedFile, ...]]:
    entries: list[FingerprintedFile] = []
    aggregate = hashlib.sha256()
    total = 0

    for path in sorted(paths, key=lambda item: _safe_relative(root, item)):
        relative = _safe_relative(root, path)
        if path.is_symlink():
            resolved = path.resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError as exc:
                raise FrontierwrightError(
                    "MODEL_PATH_ESCAPE",
                    f"Symlink escapes model source: {relative}",
                    13,
                ) from exc
        size = path.stat().st_size
        file_hash = sha256_file(path)
        entries.append(FingerprintedFile(relative, size, file_hash))
        total += size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\n")

    return f"sha256:{aggregate.hexdigest()}", total, tuple(entries)


class LocalHuggingFaceAdapter:
    adapter_id = "local-huggingface-v1"

    def supports(self, source: Path) -> bool:
        if not source.is_dir() or not (source / "config.json").is_file():
            return False
        return any(
            path.is_file() and path.suffix.lower() in {".safetensors", ".bin"}
            for path in source.rglob("*")
        )

    def inspect(self, source: Path) -> ImportedModelDescriptor:
        root = source.resolve()
        artifacts = [
            path
            for path in root.rglob("*")
            if path.is_file() and _is_hf_artifact(path)
        ]
        weight_files = [
            path for path in artifacts if path.suffix.lower() in {".safetensors", ".bin"}
        ]
        if not weight_files:
            raise FrontierwrightError(
                "MODEL_WEIGHTS_MISSING",
                "Hugging Face model directory has no supported local weight files.",
                12,
            )
        fingerprint, total, files = _fingerprint_files(root, artifacts)
        return ImportedModelDescriptor(
            source_path=root,
            model_format=ModelFormat.HUGGINGFACE,
            trainable=True,
            fingerprint=fingerprint,
            total_bytes=total,
            files=files,
        )


class LocalGGUFAdapter:
    adapter_id = "local-gguf-v1"

    def supports(self, source: Path) -> bool:
        return source.is_file() and source.suffix.lower() == ".gguf"

    def inspect(self, source: Path) -> ImportedModelDescriptor:
        resolved = source.resolve()
        file_hash = sha256_file(resolved)
        size = resolved.stat().st_size
        return ImportedModelDescriptor(
            source_path=resolved,
            model_format=ModelFormat.GGUF,
            trainable=False,
            fingerprint=f"sha256:{file_hash}",
            total_bytes=size,
            files=(FingerprintedFile(resolved.name, size, file_hash),),
        )


DEFAULT_ADAPTERS: tuple[ModelSourceAdapter, ...] = (
    LocalHuggingFaceAdapter(),
    LocalGGUFAdapter(),
)


def inspect_local_model(
    source: Path,
    *,
    adapters: tuple[ModelSourceAdapter, ...] = DEFAULT_ADAPTERS,
) -> ImportedModelDescriptor:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FrontierwrightError("MODEL_NOT_FOUND", f"Model source does not exist: {source}", 12)

    for adapter in adapters:
        if adapter.supports(source):
            return adapter.inspect(source)

    raise FrontierwrightError(
        "MODEL_FORMAT_UNSUPPORTED",
        "No installed model-source adapter recognizes this local model.",
        12,
    )


def _artifact_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            f"Evidence artifact escapes manifest directory: {relative}",
            12,
        ) from exc
    if not candidate.is_file():
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            f"Evidence artifact is missing: {relative}",
            12,
        )
    return candidate


def _verify_artifact(root: Path, item: object, label: str) -> str:
    if not isinstance(item, dict):
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            f"{label} must be an object with path and sha256.",
            12,
        )
    relative = item.get("path")
    expected = item.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            f"{label} must include string path and sha256.",
            12,
        )
    actual = sha256_file(_artifact_path(root, relative))
    expected_clean = expected.removeprefix("sha256:")
    if actual != expected_clean:
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            f"{label} hash mismatch for {relative}.",
            12,
        )
    return relative


def verify_history_manifest(
    manifest_path: Path,
    *,
    current_model_fingerprint: str,
) -> HistoryEvidenceResult:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            f"History manifest does not exist: {manifest_path}",
            12,
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            "History manifest is not valid UTF-8 JSON.",
            12,
        ) from exc

    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            "History manifest schema_version must be 1.",
            12,
        )
    if raw.get("current_model_fingerprint") != current_model_fingerprint:
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            "History manifest current_model_fingerprint does not match imported model.",
            12,
        )

    lineage = raw.get("lineage")
    if not isinstance(lineage, list) or not lineage:
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            "History manifest lineage must be a non-empty list.",
            12,
        )

    root = manifest_path.parent
    evidence_files: list[str] = [manifest_path.name]
    previous_output: str | None = None
    for index, step in enumerate(lineage):
        if not isinstance(step, dict):
            raise FrontierwrightError(
                "HISTORY_MANIFEST_INVALID",
                f"Lineage step {index} must be an object.",
                12,
            )
        input_fp = step.get("input_fingerprint")
        output_fp = step.get("output_fingerprint")
        intervention = step.get("intervention")
        if not all(
            isinstance(value, str) and value
            for value in (input_fp, output_fp, intervention)
        ):
            raise FrontierwrightError(
                "HISTORY_MANIFEST_INVALID",
                f"Lineage step {index} lacks input/output fingerprint or intervention.",
                12,
            )
        if previous_output is not None and input_fp != previous_output:
            raise FrontierwrightError(
                "HISTORY_MANIFEST_INVALID",
                f"Lineage hash chain breaks at step {index}.",
                12,
            )
        previous_output = output_fp

        evidence_files.extend(
            [
                _verify_artifact(root, step.get("config"), f"lineage[{index}].config"),
                _verify_artifact(
                    root,
                    step.get("dataset_manifest"),
                    f"lineage[{index}].dataset_manifest",
                ),
                _verify_artifact(
                    root,
                    step.get("run_receipt"),
                    f"lineage[{index}].run_receipt",
                ),
            ]
        )

    if previous_output != current_model_fingerprint:
        raise FrontierwrightError(
            "HISTORY_MANIFEST_INVALID",
            "Final lineage output does not match imported model fingerprint.",
            12,
        )

    return HistoryEvidenceResult(
        confidence=HistoryConfidence.VERIFIED,
        evidence_files=tuple(dict.fromkeys(evidence_files)),
        reason="Lineage chain and referenced evidence artifact hashes verified.",
    )


def discover_history_evidence(
    descriptor: ImportedModelDescriptor,
    *,
    manifest_path: Path | None = None,
) -> HistoryEvidenceResult:
    source = descriptor.source_path
    root = source if source.is_dir() else source.parent

    manifest = manifest_path
    if manifest is None:
        auto = root / _LINEAGE_MANIFEST_NAME
        manifest = auto if auto.is_file() else None

    if manifest is not None:
        return verify_history_manifest(
            manifest,
            current_model_fingerprint=descriptor.fingerprint,
        )

    partial = tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_file() and path.name in _PARTIAL_HISTORY_NAMES
        )
    )
    if partial:
        return HistoryEvidenceResult(
            confidence=HistoryConfidence.PARTIAL,
            evidence_files=partial,
            reason=(
                "Training-related metadata exists, but complete intervention lineage "
                "cannot be verified from it."
            ),
        )

    return HistoryEvidenceResult(
        confidence=HistoryConfidence.UNKNOWN,
        evidence_files=(),
        reason="No verifiable prior training lineage was supplied.",
    )
