"""User-first local dataset inventory and reproducible content fingerprints."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from frontierwright.errors import FrontierwrightError
from frontierwright.models import sha256_file


class DatasetRole(StrEnum):
    PRETRAIN = "PRETRAIN"
    SFT = "SFT"
    PREFERENCE = "PREFERENCE"
    ROLLOUT = "ROLLOUT"


class DatasetProvenance(StrEnum):
    LOCAL_USER = "LOCAL_USER"
    INTERNAL_CONNECTED = "INTERNAL_CONNECTED"
    PUBLIC_DISCOVERED = "PUBLIC_DISCOVERED"


class DatasetClassification(StrEnum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    PRIVATE = "PRIVATE"


def default_dataset_classification(
    provenance: DatasetProvenance,
) -> DatasetClassification:
    if provenance is DatasetProvenance.PUBLIC_DISCOVERED:
        return DatasetClassification.PUBLIC
    if provenance is DatasetProvenance.INTERNAL_CONNECTED:
        return DatasetClassification.INTERNAL
    return DatasetClassification.PRIVATE


@dataclass(frozen=True)
class DatasetFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetDescriptor:
    source_path: Path
    fingerprint: str
    total_bytes: int
    files: tuple[DatasetFile, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    def manifest(self) -> list[dict[str, object]]:
        return [asdict(item) for item in self.files]


def _safe_relative(root: Path, path: Path) -> str:
    root = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise FrontierwrightError(
            "DATA_PATH_ESCAPE",
            f"Dataset artifact escapes source directory: {path}",
            13,
        ) from exc


def inspect_local_dataset(source: Path) -> DatasetDescriptor:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FrontierwrightError(
            "DATA_NOT_FOUND",
            f"Dataset source does not exist: {source}",
            12,
        )

    if source.is_file():
        root = source.parent
        paths = [source]
    elif source.is_dir():
        root = source
        paths = [path for path in source.rglob("*") if path.is_file()]
    else:
        raise FrontierwrightError(
            "DATA_UNSUPPORTED_SOURCE",
            "Dataset source must be a regular file or directory.",
            12,
        )

    if not paths:
        raise FrontierwrightError(
            "DATA_EMPTY",
            "Dataset source contains no regular files.",
            12,
        )

    entries: list[DatasetFile] = []
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
                    "DATA_PATH_ESCAPE",
                    f"Dataset symlink escapes source: {relative}",
                    13,
                ) from exc
        size = path.stat().st_size
        file_hash = sha256_file(path)
        entries.append(DatasetFile(relative, size, file_hash))
        total += size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\n")

    return DatasetDescriptor(
        source_path=source,
        fingerprint=f"sha256:{aggregate.hexdigest()}",
        total_bytes=total,
        files=tuple(entries),
    )
