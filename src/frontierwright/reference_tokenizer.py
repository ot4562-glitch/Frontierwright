"""Deterministic trainable byte-level BPE artifacts for Frontierwright Academy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontierwright.errors import FrontierwrightError

TOKENIZER_SCHEMA_VERSION = 1
TOKENIZER_TYPE = "frontierwright-byte-bpe"
BASE_VOCAB_SIZE = 256
DEFAULT_MAX_TRAINING_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class BPEMerge:
    left: int
    right: int
    token_id: int
    count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "right": self.right,
            "id": self.token_id,
            "count": self.count,
        }


@dataclass(frozen=True)
class TokenizerArtifact:
    artifact_id: str
    fingerprint: str
    path: Path
    source_dataset_id: str
    source_dataset_fingerprint: str
    requested_vocab_size: int
    vocab_size: int
    max_training_bytes: int
    training_bytes: int
    merges: tuple[BPEMerge, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "fingerprint": self.fingerprint,
            "path": str(self.path),
            "source_dataset_id": self.source_dataset_id,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "requested_vocab_size": self.requested_vocab_size,
            "vocab_size": self.vocab_size,
            "max_training_bytes": self.max_training_bytes,
            "training_bytes": self.training_bytes,
            "merge_count": len(self.merges),
        }


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def tokenizer_fingerprint(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def _safe_dataset_files(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.is_symlink():
            raise FrontierwrightError(
                "TOKENIZER_DATA_SYMLINK",
                "Tokenizer training source must not be a symbolic-link file.",
                13,
            )
        return [source]
    if not source.is_dir():
        raise FrontierwrightError(
            "TOKENIZER_DATA_NOT_FOUND",
            f"Tokenizer training source is not readable: {source}",
            12,
        )

    files: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(source)
        except ValueError as exc:
            raise FrontierwrightError(
                "TOKENIZER_DATA_PATH_ESCAPE",
                f"Tokenizer training file escapes dataset root: {path}",
                13,
            ) from exc
        if path.is_symlink():
            raise FrontierwrightError(
                "TOKENIZER_DATA_SYMLINK",
                f"Tokenizer training source contains symbolic link: {path}",
                13,
            )
        files.append(path)
    if not files:
        raise FrontierwrightError(
            "TOKENIZER_DATA_EMPTY",
            "Tokenizer training source contains no regular files.",
            12,
        )
    return sorted(files, key=lambda item: item.relative_to(source).as_posix())


def read_training_bytes(source: Path, *, max_training_bytes: int) -> bytes:
    if (
        isinstance(max_training_bytes, bool)
        or not isinstance(max_training_bytes, int)
        or max_training_bytes <= 0
    ):
        raise FrontierwrightError(
            "TOKENIZER_CONFIG_INVALID",
            "max_training_bytes must be a positive integer.",
            2,
        )

    chunks: list[bytes] = []
    total = 0
    for index, path in enumerate(_safe_dataset_files(source)):
        if index > 0 and total < max_training_bytes:
            chunks.append(b"\n")
            total += 1
        remaining = max_training_bytes - total
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
        raise FrontierwrightError(
            "TOKENIZER_DATA_EMPTY",
            "Tokenizer training corpus is empty after applying the byte limit.",
            12,
        )
    return corpus


def _replace_pair(tokens: list[int], pair: tuple[int, int], token_id: int) -> list[int]:
    left, right = pair
    output: list[int] = []
    index = 0
    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == left
            and tokens[index + 1] == right
        ):
            output.append(token_id)
            index += 2
        else:
            output.append(tokens[index])
            index += 1
    return output


def train_byte_bpe(corpus: bytes, *, requested_vocab_size: int) -> tuple[BPEMerge, ...]:
    if (
        isinstance(requested_vocab_size, bool)
        or not isinstance(requested_vocab_size, int)
        or requested_vocab_size < BASE_VOCAB_SIZE
        or requested_vocab_size > 65_536
    ):
        raise FrontierwrightError(
            "TOKENIZER_CONFIG_INVALID",
            "vocab_size must be an integer between 256 and 65536.",
            2,
        )
    if not corpus:
        raise FrontierwrightError(
            "TOKENIZER_DATA_EMPTY",
            "Tokenizer training corpus must be nonempty.",
            12,
        )

    tokens = list(corpus)
    merges: list[BPEMerge] = []
    next_token_id = BASE_VOCAB_SIZE

    while next_token_id < requested_vocab_size and len(tokens) >= 2:
        counts: dict[tuple[int, int], int] = {}
        for left, right in zip(tokens, tokens[1:], strict=False):
            pair = (left, right)
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        pair, count = min(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
        if count < 2:
            break
        merges.append(BPEMerge(pair[0], pair[1], next_token_id, count))
        tokens = _replace_pair(tokens, pair, next_token_id)
        next_token_id += 1

    return tuple(merges)


def tokenizer_payload(
    *,
    source_dataset_id: str,
    source_dataset_fingerprint: str,
    requested_vocab_size: int,
    max_training_bytes: int,
    training_bytes: int,
    merges: tuple[BPEMerge, ...],
) -> dict[str, object]:
    return {
        "schema_version": TOKENIZER_SCHEMA_VERSION,
        "type": TOKENIZER_TYPE,
        "base_vocab_size": BASE_VOCAB_SIZE,
        "requested_vocab_size": requested_vocab_size,
        "vocab_size": BASE_VOCAB_SIZE + len(merges),
        "source_dataset_id": source_dataset_id,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "max_training_bytes": max_training_bytes,
        "training_bytes": training_bytes,
        "merges": [merge.to_dict() for merge in merges],
        "encoding": "byte-level BPE; base token id equals byte value 0..255",
    }


def write_tokenizer_artifact(
    destination: Path,
    *,
    source_dataset_id: str,
    source_dataset_fingerprint: str,
    requested_vocab_size: int,
    max_training_bytes: int,
    corpus: bytes,
) -> TokenizerArtifact:
    merges = train_byte_bpe(corpus, requested_vocab_size=requested_vocab_size)
    payload = tokenizer_payload(
        source_dataset_id=source_dataset_id,
        source_dataset_fingerprint=source_dataset_fingerprint,
        requested_vocab_size=requested_vocab_size,
        max_training_bytes=max_training_bytes,
        training_bytes=len(corpus),
        merges=merges,
    )
    fingerprint = tokenizer_fingerprint(payload)
    artifact_id = "tokart-" + fingerprint.split(":", 1)[1][:32]

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "tokenizer.json"
    encoded = _canonical_json_bytes(payload) + b"\n"
    if path.exists():
        existing = path.read_bytes()
        if existing != encoded:
            raise FrontierwrightError(
                "TOKENIZER_ARTIFACT_CONFLICT",
                "Tokenizer artifact destination already contains different bytes.",
                13,
            )
    else:
        path.write_bytes(encoded)

    return TokenizerArtifact(
        artifact_id=artifact_id,
        fingerprint=fingerprint,
        path=path,
        source_dataset_id=source_dataset_id,
        source_dataset_fingerprint=source_dataset_fingerprint,
        requested_vocab_size=requested_vocab_size,
        vocab_size=BASE_VOCAB_SIZE + len(merges),
        max_training_bytes=max_training_bytes,
        training_bytes=len(corpus),
        merges=merges,
    )


def load_tokenizer_payload(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "TOKENIZER_ARTIFACT_INVALID",
            f"Tokenizer artifact is unreadable: {path}",
            13,
        ) from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != TOKENIZER_SCHEMA_VERSION
        or raw.get("type") != TOKENIZER_TYPE
    ):
        raise FrontierwrightError(
            "TOKENIZER_ARTIFACT_INVALID",
            "Tokenizer artifact schema/type is invalid.",
            13,
        )
    return raw


def encode_bytes(data: bytes, payload: dict[str, Any]) -> list[int]:
    tokens = list(data)
    merges = payload.get("merges")
    if not isinstance(merges, list):
        raise FrontierwrightError(
            "TOKENIZER_ARTIFACT_INVALID",
            "Tokenizer artifact merges must be a list.",
            13,
        )
    for item in merges:
        if not isinstance(item, dict):
            raise FrontierwrightError(
                "TOKENIZER_ARTIFACT_INVALID",
                "Tokenizer merge entry must be an object.",
                13,
            )
        left = item.get("left")
        right = item.get("right")
        token_id = item.get("id")
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
        ):
            raise FrontierwrightError(
                "TOKENIZER_ARTIFACT_INVALID",
                "Tokenizer merge IDs must be integers.",
                13,
            )
        tokens = _replace_pair(tokens, (left, right), token_id)
    return tokens


def decode_token_ids(token_ids: list[int], payload: dict[str, Any]) -> bytes:
    expansions: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    merges = payload.get("merges")
    if not isinstance(merges, list):
        raise FrontierwrightError(
            "TOKENIZER_ARTIFACT_INVALID",
            "Tokenizer artifact merges must be a list.",
            13,
        )
    for item in merges:
        if not isinstance(item, dict):
            raise FrontierwrightError(
                "TOKENIZER_ARTIFACT_INVALID",
                "Tokenizer merge entry must be an object.",
                13,
            )
        left = item.get("left")
        right = item.get("right")
        token_id = item.get("id")
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or left not in expansions
            or right not in expansions
        ):
            raise FrontierwrightError(
                "TOKENIZER_ARTIFACT_INVALID",
                "Tokenizer merge references invalid token IDs.",
                13,
            )
        expansions[token_id] = expansions[left] + expansions[right]

    output = bytearray()
    for token_id in token_ids:
        if token_id not in expansions:
            raise FrontierwrightError(
                "TOKENIZER_TOKEN_INVALID",
                f"Tokenizer cannot decode token id {token_id}.",
                13,
            )
        output.extend(expansions[token_id])
    return bytes(output)
