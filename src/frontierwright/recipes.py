"""Reproducible dataset-preparation recipes and managed artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from frontierwright.data import DatasetDescriptor, inspect_local_dataset
from frontierwright.errors import FrontierwrightError

SNAPSHOT_COPY_PLUGIN_ID = "frontierwright.data.snapshot-copy"
SNAPSHOT_COPY_PLUGIN_VERSION = "1"
TEXT_LINES_PLUGIN_ID = "frontierwright.data.text-lines-normalize-dedupe"
TEXT_LINES_PLUGIN_VERSION = "1"
WEIGHTED_TEXT_MIXTURE_PLUGIN_ID = "frontierwright.data.weighted-text-mixture"
WEIGHTED_TEXT_MIXTURE_PLUGIN_VERSION = "1"
BYTE_SHARDS_PLUGIN_ID = "frontierwright.data.byte-shards"
BYTE_SHARDS_PLUGIN_VERSION = "1"
BYTE_VOCAB_ID = "frontierwright-byte-vocab-v1"


@dataclass(frozen=True)
class DataPreparationRecipe:
    recipe_id: str
    recipe_hash: str
    plugin_id: str
    plugin_version: str
    source_dataset_id: str
    source_fingerprint: str
    config: dict[str, object]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "source_dataset_id": self.source_dataset_id,
            "source_fingerprint": self.source_fingerprint,
            "config": self.config,
        }


@dataclass(frozen=True)
class DataSourceBinding:
    dataset_id: str
    fingerprint: str


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def make_snapshot_recipe(
    *,
    source_dataset_id: str,
    source_fingerprint: str,
) -> DataPreparationRecipe:
    payload: dict[str, object] = {
        "schema_version": 1,
        "plugin_id": SNAPSHOT_COPY_PLUGIN_ID,
        "plugin_version": SNAPSHOT_COPY_PLUGIN_VERSION,
        "source_dataset_id": source_dataset_id,
        "source_fingerprint": source_fingerprint,
        "config": {"mode": "byte_preserving_snapshot"},
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return DataPreparationRecipe(
        recipe_id=f"data-recipe-{digest[:32]}",
        recipe_hash=f"sha256:{digest}",
        plugin_id=SNAPSHOT_COPY_PLUGIN_ID,
        plugin_version=SNAPSHOT_COPY_PLUGIN_VERSION,
        source_dataset_id=source_dataset_id,
        source_fingerprint=source_fingerprint,
        config={"mode": "byte_preserving_snapshot"},
    )


def make_text_lines_recipe(
    *,
    source_dataset_id: str,
    source_fingerprint: str,
    config: dict[str, object] | None = None,
) -> DataPreparationRecipe:
    canonical_config = _text_lines_config(config)
    payload: dict[str, object] = {
        "schema_version": 1,
        "plugin_id": TEXT_LINES_PLUGIN_ID,
        "plugin_version": TEXT_LINES_PLUGIN_VERSION,
        "source_dataset_id": source_dataset_id,
        "source_fingerprint": source_fingerprint,
        "config": canonical_config,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return DataPreparationRecipe(
        recipe_id=f"data-recipe-{digest[:32]}",
        recipe_hash=f"sha256:{digest}",
        plugin_id=TEXT_LINES_PLUGIN_ID,
        plugin_version=TEXT_LINES_PLUGIN_VERSION,
        source_dataset_id=source_dataset_id,
        source_fingerprint=source_fingerprint,
        config=canonical_config,
    )


def _text_lines_config(config: dict[str, object] | None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "encoding": "utf-8",
        "unicode_normalization": "NFC",
        "line_endings": "LF",
        "strip_whitespace": True,
        "drop_empty": True,
        "dedupe": "stable_exact",
        "output": "corpus.txt",
    }
    supplied = dict(config or {})
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            f"text-lines recipe does not support config keys: {', '.join(unknown)}",
            2,
        )
    merged = {**defaults, **supplied}

    if merged["encoding"] != "utf-8":
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "text-lines v1 supports UTF-8 input only.",
            2,
        )
    normalization = merged["unicode_normalization"]
    if normalization not in ("NFC", "NFKC", "NFD", "NFKD", "NONE"):
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "unicode_normalization must be NFC, NFKC, NFD, NFKD, or NONE.",
            2,
        )
    if merged["line_endings"] != "LF":
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "text-lines v1 emits LF line endings only.",
            2,
        )
    for key in ("strip_whitespace", "drop_empty"):
        if not isinstance(merged[key], bool):
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                f"{key} must be true or false.",
                2,
            )
    if merged["dedupe"] not in ("stable_exact", "none"):
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "dedupe must be stable_exact or none.",
            2,
        )
    if merged["output"] != "corpus.txt":
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "text-lines v1 writes the fixed managed output corpus.txt.",
            2,
        )
    return merged



def _weighted_text_mixture_config(
    config: dict[str, object] | None,
    *,
    source_dataset_id: str | None = None,
    source_fingerprint: str | None = None,
) -> dict[str, object]:
    supplied = dict(config or {})
    raw_sources = supplied.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) < 2:
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "weighted-text-mixture requires at least two source entries.",
            2,
        )

    canonical_sources: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                "Each weighted-text-mixture source must be an object.",
                2,
            )
        unknown = sorted(set(raw) - {"dataset_id", "fingerprint", "parts"})
        if unknown:
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                "Mixture source has unsupported keys: " + ", ".join(unknown),
                2,
            )
        dataset_id = raw.get("dataset_id")
        fingerprint = raw.get("fingerprint")
        parts = raw.get("parts")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                f"Mixture source {index + 1} dataset_id must be nonempty.",
                2,
            )
        if dataset_id in seen_ids:
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                f"Mixture source is duplicated: {dataset_id}",
                2,
            )
        seen_ids.add(dataset_id)
        if (
            not isinstance(fingerprint, str)
            or not fingerprint.startswith("sha256:")
            or not fingerprint.strip()
        ):
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                f"Mixture source {dataset_id} requires a sha256 fingerprint.",
                2,
            )
        if isinstance(parts, bool) or not isinstance(parts, int) or parts <= 0:
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                f"Mixture source {dataset_id} parts must be a positive integer.",
                2,
            )
        if parts > 1000:
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                f"Mixture source {dataset_id} parts exceeds the v1 limit of 1000.",
                2,
            )
        canonical_sources.append(
            {
                "dataset_id": dataset_id,
                "fingerprint": fingerprint,
                "parts": parts,
            }
        )

    if source_dataset_id is not None:
        first = canonical_sources[0]
        if (
            first["dataset_id"] != source_dataset_id
            or first["fingerprint"] != source_fingerprint
        ):
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                "The first mixture source must match the recipe primary source.",
                2,
            )

    max_output_bytes = supplied.get("max_output_bytes", 4 * 1024**3)
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
    ):
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "max_output_bytes must be a positive integer.",
            2,
        )

    unknown_top = sorted(
        set(supplied) - {"sources", "max_output_bytes", "mode", "output"}
    )
    if unknown_top:
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "weighted-text-mixture does not support config keys: "
            + ", ".join(unknown_top),
            2,
        )
    if supplied.get("mode", "weighted_concat_text") != "weighted_concat_text":
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "weighted-text-mixture mode must be weighted_concat_text.",
            2,
        )
    if supplied.get("output", "corpus.txt") != "corpus.txt":
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "weighted-text-mixture v1 writes the fixed managed output corpus.txt.",
            2,
        )

    return {
        "mode": "weighted_concat_text",
        "sources": canonical_sources,
        "max_output_bytes": max_output_bytes,
        "output": "corpus.txt",
    }


def make_weighted_text_mixture_recipe(
    *,
    source_dataset_id: str,
    source_fingerprint: str,
    config: dict[str, object],
) -> DataPreparationRecipe:
    canonical_config = _weighted_text_mixture_config(
        config,
        source_dataset_id=source_dataset_id,
        source_fingerprint=source_fingerprint,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "plugin_id": WEIGHTED_TEXT_MIXTURE_PLUGIN_ID,
        "plugin_version": WEIGHTED_TEXT_MIXTURE_PLUGIN_VERSION,
        "source_dataset_id": source_dataset_id,
        "source_fingerprint": source_fingerprint,
        "config": canonical_config,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return DataPreparationRecipe(
        recipe_id=f"data-recipe-{digest[:32]}",
        recipe_hash=f"sha256:{digest}",
        plugin_id=WEIGHTED_TEXT_MIXTURE_PLUGIN_ID,
        plugin_version=WEIGHTED_TEXT_MIXTURE_PLUGIN_VERSION,
        source_dataset_id=source_dataset_id,
        source_fingerprint=source_fingerprint,
        config=canonical_config,
    )


def _copy_descriptor_files(
    descriptor: DatasetDescriptor,
    destination_root: Path,
) -> None:
    source = descriptor.source_path
    destination_root.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        if source.is_symlink():
            raise FrontierwrightError(
                "DATA_PREP_SYMLINK_REJECTED",
                "Managed dataset snapshots do not preserve symbolic links.",
                13,
            )
        shutil.copy2(source, destination_root / source.name)
        return

    for item in descriptor.files:
        relative = Path(item.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise FrontierwrightError(
                "DATA_PREP_PATH_ESCAPE",
                f"Dataset preparation contains unsafe path: {item.relative_path}",
                13,
            )
        source_file = source / relative
        if source_file.is_symlink():
            raise FrontierwrightError(
                "DATA_PREP_SYMLINK_REJECTED",
                f"Managed dataset snapshots do not preserve symbolic links: {item.relative_path}",
                13,
            )
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)


def _write_recipe_manifest(
    path: Path,
    recipe: DataPreparationRecipe,
    descriptor: DatasetDescriptor,
    *,
    transformation: dict[str, object] | None = None,
) -> None:
    payload = {
        **recipe.canonical_payload(),
        "recipe_id": recipe.recipe_id,
        "recipe_hash": recipe.recipe_hash,
        "prepared_fingerprint": descriptor.fingerprint,
        "total_bytes": descriptor.total_bytes,
        "file_count": descriptor.file_count,
        "files": descriptor.manifest(),
    }
    if transformation is not None:
        payload["transformation"] = transformation
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def materialize_snapshot_recipe(
    state_dir: Path,
    *,
    source: DatasetDescriptor,
    recipe: DataPreparationRecipe,
) -> DatasetDescriptor:
    """Create or verify a byte-preserving managed dataset snapshot."""

    if source.fingerprint != recipe.source_fingerprint:
        raise FrontierwrightError(
            "DATA_RECIPE_SOURCE_DRIFT",
            "Dataset content no longer matches the fingerprint pinned by the recipe.",
            13,
        )

    prepared_root = state_dir.resolve() / "data" / "prepared" / recipe.recipe_id
    data_root = prepared_root / "data"
    manifest_path = prepared_root / "recipe.json"

    if prepared_root.exists():
        if not manifest_path.is_file() or not data_root.is_dir():
            raise FrontierwrightError(
                "DATA_PREP_ARTIFACT_INVALID",
                "Existing prepared dataset artifact is incomplete.",
                13,
            )
        prepared = inspect_local_dataset(data_root)
        if prepared.fingerprint != source.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_ARTIFACT_TAMPERED",
                "Managed dataset snapshot no longer matches its source fingerprint.",
                13,
            )
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FrontierwrightError(
                "DATA_PREP_MANIFEST_INVALID",
                "Prepared dataset recipe manifest is unreadable.",
                13,
            ) from exc
        if not isinstance(raw, dict) or raw.get("recipe_hash") != recipe.recipe_hash:
            raise FrontierwrightError(
                "DATA_PREP_MANIFEST_INVALID",
                "Prepared dataset recipe manifest does not match the requested recipe.",
                13,
            )
        return prepared

    staging_parent = state_dir.resolve() / "data" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{recipe.recipe_id}-", dir=staging_parent)
    ).resolve()

    try:
        staging_data = staging_root / "data"
        _copy_descriptor_files(source, staging_data)
        prepared = inspect_local_dataset(staging_data)
        if prepared.fingerprint != source.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_COPY_MISMATCH",
                "Managed dataset snapshot changed the source content fingerprint.",
                13,
            )
        _write_recipe_manifest(staging_root / "recipe.json", recipe, prepared)

        prepared_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staging_root, prepared_root)
        except FileExistsError:
            shutil.rmtree(staging_root, ignore_errors=True)

        final = inspect_local_dataset(data_root)
        if final.fingerprint != source.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_COPY_MISMATCH",
                "Published dataset snapshot differs from the source fingerprint.",
                13,
            )
        return final
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise



def _text_source_files(source: DatasetDescriptor) -> tuple[Path, ...]:
    if source.source_path.is_file():
        if source.source_path.is_symlink():
            raise FrontierwrightError(
                "DATA_PREP_SYMLINK_REJECTED",
                "Text preparation does not read symbolic links.",
                13,
            )
        return (source.source_path,)

    paths: list[Path] = []
    for item in source.files:
        relative = Path(item.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise FrontierwrightError(
                "DATA_PREP_PATH_ESCAPE",
                f"Dataset preparation contains unsafe path: {item.relative_path}",
                13,
            )
        path = source.source_path / relative
        if path.is_symlink():
            raise FrontierwrightError(
                "DATA_PREP_SYMLINK_REJECTED",
                f"Text preparation does not read symbolic links: {item.relative_path}",
                13,
            )
        paths.append(path)
    return tuple(paths)


def _read_text_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise FrontierwrightError(
            "DATA_PREP_TEXT_ENCODING",
            f"Text-lines recipe requires UTF-8 input: {path}",
            13,
        ) from exc
    except OSError as exc:
        raise FrontierwrightError(
            "DATA_PREP_SOURCE_UNREADABLE",
            f"Text-lines recipe could not read source file: {path}",
            13,
        ) from exc


def _transform_text_lines(
    source: DatasetDescriptor,
    config: dict[str, object],
) -> tuple[bytes, dict[str, object]]:
    normalization = str(config["unicode_normalization"])
    strip_whitespace = bool(config["strip_whitespace"])
    drop_empty = bool(config["drop_empty"])
    dedupe = str(config["dedupe"])

    output_lines: list[str] = []
    seen: set[str] = set()
    input_lines = 0
    empty_removed = 0
    duplicates_removed = 0

    source_files = _text_source_files(source)
    for path in source_files:
        for raw_line in _read_text_lines(path):
            input_lines += 1
            line = raw_line
            if normalization == "NFC":
                line = unicodedata.normalize("NFC", line)
            elif normalization == "NFKC":
                line = unicodedata.normalize("NFKC", line)
            elif normalization == "NFD":
                line = unicodedata.normalize("NFD", line)
            elif normalization == "NFKD":
                line = unicodedata.normalize("NFKD", line)
            if strip_whitespace:
                line = line.strip()
            if drop_empty and not line:
                empty_removed += 1
                continue
            if dedupe == "stable_exact":
                if line in seen:
                    duplicates_removed += 1
                    continue
                seen.add(line)
            output_lines.append(line)

    payload = "\n".join(output_lines)
    if output_lines:
        payload += "\n"
    encoded = payload.encode("utf-8")
    transformation: dict[str, object] = {
        "input_files": len(source_files),
        "input_lines": input_lines,
        "output_lines": len(output_lines),
        "empty_lines_removed": empty_removed,
        "duplicate_lines_removed": duplicates_removed,
        "output_bytes": len(encoded),
    }
    return encoded, transformation


def _load_prepared_manifest(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "DATA_PREP_MANIFEST_INVALID",
            "Prepared dataset recipe manifest is unreadable.",
            13,
        ) from exc
    if not isinstance(raw, dict):
        raise FrontierwrightError(
            "DATA_PREP_MANIFEST_INVALID",
            "Prepared dataset recipe manifest must be an object.",
            13,
        )
    return raw


def materialize_text_lines_recipe(
    state_dir: Path,
    *,
    source: DatasetDescriptor,
    recipe: DataPreparationRecipe,
) -> DatasetDescriptor:
    """Normalize UTF-8 text lines and optionally remove exact duplicates."""

    if source.fingerprint != recipe.source_fingerprint:
        raise FrontierwrightError(
            "DATA_RECIPE_SOURCE_DRIFT",
            "Dataset content no longer matches the fingerprint pinned by the recipe.",
            13,
        )
    if recipe.plugin_id != TEXT_LINES_PLUGIN_ID:
        raise FrontierwrightError(
            "DATA_RECIPE_PLUGIN_MISMATCH",
            "Text-lines materializer received a recipe for a different plugin.",
            13,
        )

    config = _text_lines_config(recipe.config)
    prepared_root = state_dir.resolve() / "data" / "prepared" / recipe.recipe_id
    data_root = prepared_root / "data"
    manifest_path = prepared_root / "recipe.json"

    if prepared_root.exists():
        if not manifest_path.is_file() or not data_root.is_dir():
            raise FrontierwrightError(
                "DATA_PREP_ARTIFACT_INVALID",
                "Existing prepared dataset artifact is incomplete.",
                13,
            )
        prepared = inspect_local_dataset(data_root)
        manifest = _load_prepared_manifest(manifest_path)
        if manifest.get("recipe_hash") != recipe.recipe_hash:
            raise FrontierwrightError(
                "DATA_PREP_MANIFEST_INVALID",
                "Prepared dataset recipe manifest does not match the requested recipe.",
                13,
            )
        if manifest.get("prepared_fingerprint") != prepared.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_ARTIFACT_TAMPERED",
                "Prepared text dataset no longer matches its recorded fingerprint.",
                13,
            )
        return prepared

    staging_parent = state_dir.resolve() / "data" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{recipe.recipe_id}-", dir=staging_parent)
    ).resolve()

    try:
        staging_data = staging_root / "data"
        staging_data.mkdir(parents=True, exist_ok=False)
        output_bytes, transformation = _transform_text_lines(source, config)
        output_path = staging_data / str(config["output"])
        output_path.write_bytes(output_bytes)
        prepared = inspect_local_dataset(staging_data)
        _write_recipe_manifest(
            staging_root / "recipe.json",
            recipe,
            prepared,
            transformation=transformation,
        )

        prepared_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staging_root, prepared_root)
        except FileExistsError:
            shutil.rmtree(staging_root, ignore_errors=True)

        final = inspect_local_dataset(data_root)
        manifest = _load_prepared_manifest(manifest_path)
        if manifest.get("recipe_hash") != recipe.recipe_hash:
            raise FrontierwrightError(
                "DATA_PREP_MANIFEST_INVALID",
                "Published text preparation has the wrong recipe hash.",
                13,
            )
        if manifest.get("prepared_fingerprint") != final.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_COPY_MISMATCH",
                "Published text preparation differs from its recorded fingerprint.",
                13,
            )
        return final
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise



def _mixture_source_bindings(
    recipe: DataPreparationRecipe,
) -> tuple[DataSourceBinding, ...]:
    config = _weighted_text_mixture_config(
        recipe.config,
        source_dataset_id=recipe.source_dataset_id,
        source_fingerprint=recipe.source_fingerprint,
    )
    raw_sources = config["sources"]
    assert isinstance(raw_sources, list)
    bindings: list[DataSourceBinding] = []
    for raw in raw_sources:
        assert isinstance(raw, dict)
        dataset_id = raw["dataset_id"]
        fingerprint = raw["fingerprint"]
        assert isinstance(dataset_id, str)
        assert isinstance(fingerprint, str)
        bindings.append(
            DataSourceBinding(
                dataset_id=dataset_id,
                fingerprint=fingerprint,
            )
        )
    return tuple(bindings)


def _mixture_corpus_bytes(
    descriptor: DatasetDescriptor,
    *,
    dataset_id: str,
) -> bytes:
    if descriptor.source_path.is_file():
        raise FrontierwrightError(
            "DATA_MIXTURE_SOURCE_INVALID",
            f"Mixture source {dataset_id} must be a managed text dataset directory.",
            13,
        )
    corpus = descriptor.source_path / "corpus.txt"
    if corpus.is_symlink() or not corpus.is_file():
        raise FrontierwrightError(
            "DATA_MIXTURE_SOURCE_INVALID",
            f"Mixture source {dataset_id} does not contain managed corpus.txt.",
            13,
        )
    try:
        payload = corpus.read_bytes()
    except OSError as exc:
        raise FrontierwrightError(
            "DATA_PREP_SOURCE_UNREADABLE",
            f"Could not read mixture source corpus: {dataset_id}",
            13,
        ) from exc
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrontierwrightError(
            "DATA_PREP_TEXT_ENCODING",
            f"Mixture source {dataset_id} corpus.txt is not valid UTF-8.",
            13,
        ) from exc
    if payload and not payload.endswith(b"\n"):
        raise FrontierwrightError(
            "DATA_MIXTURE_SOURCE_INVALID",
            f"Mixture source {dataset_id} corpus.txt must end with LF.",
            13,
        )
    return payload


def materialize_weighted_text_mixture_recipe(
    state_dir: Path,
    *,
    sources: dict[str, DatasetDescriptor],
    recipe: DataPreparationRecipe,
) -> DatasetDescriptor:
    """Create one deterministic weighted concatenation of managed text corpora."""

    if recipe.plugin_id != WEIGHTED_TEXT_MIXTURE_PLUGIN_ID:
        raise FrontierwrightError(
            "DATA_RECIPE_PLUGIN_MISMATCH",
            "Weighted-mixture materializer received a recipe for a different plugin.",
            13,
        )
    config = _weighted_text_mixture_config(
        recipe.config,
        source_dataset_id=recipe.source_dataset_id,
        source_fingerprint=recipe.source_fingerprint,
    )
    bindings = _mixture_source_bindings(recipe)
    raw_sources = config["sources"]
    assert isinstance(raw_sources, list)

    prepared_root = state_dir.resolve() / "data" / "prepared" / recipe.recipe_id
    data_root = prepared_root / "data"
    manifest_path = prepared_root / "recipe.json"

    if prepared_root.exists():
        if not manifest_path.is_file() or not data_root.is_dir():
            raise FrontierwrightError(
                "DATA_PREP_ARTIFACT_INVALID",
                "Existing mixture artifact is incomplete.",
                13,
            )
        prepared = inspect_local_dataset(data_root)
        manifest = _load_prepared_manifest(manifest_path)
        if manifest.get("recipe_hash") != recipe.recipe_hash:
            raise FrontierwrightError(
                "DATA_PREP_MANIFEST_INVALID",
                "Prepared mixture manifest does not match the requested recipe.",
                13,
            )
        if manifest.get("prepared_fingerprint") != prepared.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_ARTIFACT_TAMPERED",
                "Prepared mixture no longer matches its recorded fingerprint.",
                13,
            )
        return prepared

    source_payloads: dict[str, bytes] = {}
    for binding in bindings:
        descriptor = sources.get(binding.dataset_id)
        if descriptor is None:
            raise FrontierwrightError(
                "DATA_MIXTURE_SOURCE_MISSING",
                f"Mixture source is not available: {binding.dataset_id}",
                13,
            )
        if descriptor.fingerprint != binding.fingerprint:
            raise FrontierwrightError(
                "DATA_RECIPE_SOURCE_DRIFT",
                f"Mixture source fingerprint drifted: {binding.dataset_id}",
                13,
            )
        source_payloads[binding.dataset_id] = _mixture_corpus_bytes(
            descriptor,
            dataset_id=binding.dataset_id,
        )

    projected_bytes = 0
    for raw in raw_sources:
        assert isinstance(raw, dict)
        dataset_id = raw["dataset_id"]
        parts = raw["parts"]
        assert isinstance(dataset_id, str)
        assert isinstance(parts, int)
        projected_bytes += len(source_payloads[dataset_id]) * parts

    max_output_bytes = config["max_output_bytes"]
    assert isinstance(max_output_bytes, int)
    if projected_bytes > max_output_bytes:
        raise FrontierwrightError(
            "DATA_PREP_OUTPUT_LIMIT",
            (
                f"Weighted mixture would create {projected_bytes} bytes, exceeding "
                f"max_output_bytes={max_output_bytes}."
            ),
            13,
        )

    staging_parent = state_dir.resolve() / "data" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{recipe.recipe_id}-", dir=staging_parent)
    ).resolve()

    try:
        staging_data = staging_root / "data"
        staging_data.mkdir(parents=True, exist_ok=False)
        output_path = staging_data / "corpus.txt"
        with output_path.open("wb") as handle:
            for raw in raw_sources:
                assert isinstance(raw, dict)
                dataset_id = raw["dataset_id"]
                parts = raw["parts"]
                assert isinstance(dataset_id, str)
                assert isinstance(parts, int)
                payload = source_payloads[dataset_id]
                for _ in range(parts):
                    handle.write(payload)

        prepared = inspect_local_dataset(staging_data)
        transformation: dict[str, object] = {
            "strategy": "weighted_concat_text",
            "sources": raw_sources,
            "output_bytes": prepared.total_bytes,
        }
        _write_recipe_manifest(
            staging_root / "recipe.json",
            recipe,
            prepared,
            transformation=transformation,
        )

        prepared_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staging_root, prepared_root)
        except FileExistsError:
            shutil.rmtree(staging_root, ignore_errors=True)

        final = inspect_local_dataset(data_root)
        manifest = _load_prepared_manifest(manifest_path)
        if manifest.get("recipe_hash") != recipe.recipe_hash:
            raise FrontierwrightError(
                "DATA_PREP_MANIFEST_INVALID",
                "Published mixture has the wrong recipe hash.",
                13,
            )
        if manifest.get("prepared_fingerprint") != final.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_COPY_MISMATCH",
                "Published mixture differs from its recorded fingerprint.",
                13,
            )
        return final
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _byte_shards_config(config: dict[str, object] | None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "vocab_id": BYTE_VOCAB_ID,
        "dtype": "uint8",
        "bytes_per_shard": 1024 * 1024,
        "source": "corpus.txt",
    }
    supplied = dict(config or {})
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "byte-shards recipe does not support config keys: " + ", ".join(unknown),
            2,
        )
    merged = {**defaults, **supplied}
    if merged["vocab_id"] != BYTE_VOCAB_ID:
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            f"byte-shards v1 requires vocab_id={BYTE_VOCAB_ID}.",
            2,
        )
    if merged["dtype"] != "uint8":
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "byte-shards v1 stores byte IDs as uint8 only.",
            2,
        )
    shard_size = merged["bytes_per_shard"]
    if (
        isinstance(shard_size, bool)
        or not isinstance(shard_size, int)
        or shard_size <= 0
    ):
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "bytes_per_shard must be a positive integer.",
            2,
        )
    if shard_size > 1024 * 1024 * 1024:
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "bytes_per_shard exceeds the v1 limit of 1 GiB.",
            2,
        )
    if merged["source"] != "corpus.txt":
        raise FrontierwrightError(
            "DATA_RECIPE_CONFIG_UNSUPPORTED",
            "byte-shards v1 consumes managed corpus.txt only.",
            2,
        )
    return merged


def make_byte_shards_recipe(
    *,
    source_dataset_id: str,
    source_fingerprint: str,
    config: dict[str, object] | None = None,
) -> DataPreparationRecipe:
    canonical_config = _byte_shards_config(config)
    payload: dict[str, object] = {
        "schema_version": 1,
        "plugin_id": BYTE_SHARDS_PLUGIN_ID,
        "plugin_version": BYTE_SHARDS_PLUGIN_VERSION,
        "source_dataset_id": source_dataset_id,
        "source_fingerprint": source_fingerprint,
        "config": canonical_config,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return DataPreparationRecipe(
        recipe_id=f"data-recipe-{digest[:32]}",
        recipe_hash=f"sha256:{digest}",
        plugin_id=BYTE_SHARDS_PLUGIN_ID,
        plugin_version=BYTE_SHARDS_PLUGIN_VERSION,
        source_dataset_id=source_dataset_id,
        source_fingerprint=source_fingerprint,
        config=canonical_config,
    )


def _managed_corpus_path(source: DatasetDescriptor) -> Path:
    if source.source_path.is_file():
        raise FrontierwrightError(
            "DATA_SHARDS_SOURCE_INVALID",
            "byte-shards requires a managed dataset directory containing corpus.txt.",
            13,
        )
    corpus = source.source_path / "corpus.txt"
    if corpus.is_symlink() or not corpus.is_file():
        raise FrontierwrightError(
            "DATA_SHARDS_SOURCE_INVALID",
            "byte-shards requires a managed corpus.txt source.",
            13,
        )
    return corpus


def materialize_byte_shards_recipe(
    state_dir: Path,
    *,
    source: DatasetDescriptor,
    recipe: DataPreparationRecipe,
) -> DatasetDescriptor:
    """Materialize deterministic uint8 byte-ID shards for the reference vocabulary."""

    if recipe.plugin_id != BYTE_SHARDS_PLUGIN_ID:
        raise FrontierwrightError(
            "DATA_RECIPE_PLUGIN_MISMATCH",
            "Byte-shards materializer received a recipe for a different plugin.",
            13,
        )
    if source.fingerprint != recipe.source_fingerprint:
        raise FrontierwrightError(
            "DATA_RECIPE_SOURCE_DRIFT",
            "Dataset content no longer matches the fingerprint pinned by the recipe.",
            13,
        )

    config = _byte_shards_config(recipe.config)
    prepared_root = state_dir.resolve() / "data" / "prepared" / recipe.recipe_id
    data_root = prepared_root / "data"
    manifest_path = prepared_root / "recipe.json"

    if prepared_root.exists():
        if not manifest_path.is_file() or not data_root.is_dir():
            raise FrontierwrightError(
                "DATA_PREP_ARTIFACT_INVALID",
                "Existing byte-shards artifact is incomplete.",
                13,
            )
        prepared = inspect_local_dataset(data_root)
        manifest = _load_prepared_manifest(manifest_path)
        if manifest.get("recipe_hash") != recipe.recipe_hash:
            raise FrontierwrightError(
                "DATA_PREP_MANIFEST_INVALID",
                "Prepared byte-shards manifest does not match the requested recipe.",
                13,
            )
        if manifest.get("prepared_fingerprint") != prepared.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_ARTIFACT_TAMPERED",
                "Prepared byte-shards no longer match their recorded fingerprint.",
                13,
            )
        return prepared

    corpus_path = _managed_corpus_path(source)
    try:
        payload = corpus_path.read_bytes()
    except OSError as exc:
        raise FrontierwrightError(
            "DATA_PREP_SOURCE_UNREADABLE",
            "Could not read managed corpus.txt for byte-shards preparation.",
            13,
        ) from exc
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrontierwrightError(
            "DATA_PREP_TEXT_ENCODING",
            "byte-shards requires corpus.txt to contain valid UTF-8.",
            13,
        ) from exc
    if not payload:
        raise FrontierwrightError(
            "DATA_EMPTY",
            "byte-shards cannot materialize an empty corpus.",
            12,
        )

    bytes_per_shard = config["bytes_per_shard"]
    assert isinstance(bytes_per_shard, int)

    staging_parent = state_dir.resolve() / "data" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{recipe.recipe_id}-", dir=staging_parent)
    ).resolve()

    try:
        staging_data = staging_root / "data"
        staging_data.mkdir(parents=True, exist_ok=False)
        shard_count = 0
        for offset in range(0, len(payload), bytes_per_shard):
            shard = payload[offset : offset + bytes_per_shard]
            shard_path = staging_data / f"shard-{shard_count:05d}.bin"
            shard_path.write_bytes(shard)
            shard_count += 1

        prepared = inspect_local_dataset(staging_data)
        transformation: dict[str, object] = {
            "representation": "uint8_byte_ids",
            "vocab_id": BYTE_VOCAB_ID,
            "dtype": "uint8",
            "byte_id_count": len(payload),
            "shard_count": shard_count,
            "bytes_per_shard": bytes_per_shard,
            "source": "corpus.txt",
        }
        _write_recipe_manifest(
            staging_root / "recipe.json",
            recipe,
            prepared,
            transformation=transformation,
        )

        prepared_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staging_root, prepared_root)
        except FileExistsError:
            shutil.rmtree(staging_root, ignore_errors=True)

        final = inspect_local_dataset(data_root)
        manifest = _load_prepared_manifest(manifest_path)
        if manifest.get("recipe_hash") != recipe.recipe_hash:
            raise FrontierwrightError(
                "DATA_PREP_MANIFEST_INVALID",
                "Published byte-shards artifact has the wrong recipe hash.",
                13,
            )
        if manifest.get("prepared_fingerprint") != final.fingerprint:
            raise FrontierwrightError(
                "DATA_PREP_COPY_MISMATCH",
                "Published byte-shards artifact differs from its recorded fingerprint.",
                13,
            )
        return final
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise


class DataPreparationPlugin(Protocol):
    """Contract for reproducible dataset preparation implementations."""

    @property
    def plugin_id(self) -> str: ...

    @property
    def plugin_version(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def accepts_prepared_source(self) -> bool: ...

    @property
    def requires_prepared_source(self) -> bool: ...

    def build_recipe(
        self,
        *,
        source_dataset_id: str,
        source_fingerprint: str,
        config: dict[str, object] | None = None,
    ) -> DataPreparationRecipe: ...

    def source_bindings(
        self,
        recipe: DataPreparationRecipe,
    ) -> tuple[DataSourceBinding, ...]: ...

    def materialize(
        self,
        state_dir: Path,
        *,
        sources: dict[str, DatasetDescriptor],
        recipe: DataPreparationRecipe,
    ) -> DatasetDescriptor: ...


@dataclass(frozen=True)
class SnapshotCopyPreparationPlugin:
    plugin_id: str = SNAPSHOT_COPY_PLUGIN_ID
    plugin_version: str = SNAPSHOT_COPY_PLUGIN_VERSION
    title: str = "Byte-preserving managed snapshot"
    accepts_prepared_source: bool = False
    requires_prepared_source: bool = False

    def build_recipe(
        self,
        *,
        source_dataset_id: str,
        source_fingerprint: str,
        config: dict[str, object] | None = None,
    ) -> DataPreparationRecipe:
        supplied = dict(config or {})
        if supplied not in ({}, {"mode": "byte_preserving_snapshot"}):
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                "snapshot-copy accepts no configurable transformation options.",
                2,
            )
        return make_snapshot_recipe(
            source_dataset_id=source_dataset_id,
            source_fingerprint=source_fingerprint,
        )

    def source_bindings(
        self,
        recipe: DataPreparationRecipe,
    ) -> tuple[DataSourceBinding, ...]:
        return (
            DataSourceBinding(
                dataset_id=recipe.source_dataset_id,
                fingerprint=recipe.source_fingerprint,
            ),
        )

    def materialize(
        self,
        state_dir: Path,
        *,
        sources: dict[str, DatasetDescriptor],
        recipe: DataPreparationRecipe,
    ) -> DatasetDescriptor:
        return materialize_snapshot_recipe(
            state_dir,
            source=sources[recipe.source_dataset_id],
            recipe=recipe,
        )


@dataclass(frozen=True)
class TextLinesPreparationPlugin:
    plugin_id: str = TEXT_LINES_PLUGIN_ID
    plugin_version: str = TEXT_LINES_PLUGIN_VERSION
    title: str = "Normalize and stable-dedupe UTF-8 text lines"
    accepts_prepared_source: bool = False
    requires_prepared_source: bool = False

    def build_recipe(
        self,
        *,
        source_dataset_id: str,
        source_fingerprint: str,
        config: dict[str, object] | None = None,
    ) -> DataPreparationRecipe:
        return make_text_lines_recipe(
            source_dataset_id=source_dataset_id,
            source_fingerprint=source_fingerprint,
            config=config,
        )

    def source_bindings(
        self,
        recipe: DataPreparationRecipe,
    ) -> tuple[DataSourceBinding, ...]:
        return (
            DataSourceBinding(
                dataset_id=recipe.source_dataset_id,
                fingerprint=recipe.source_fingerprint,
            ),
        )

    def materialize(
        self,
        state_dir: Path,
        *,
        sources: dict[str, DatasetDescriptor],
        recipe: DataPreparationRecipe,
    ) -> DatasetDescriptor:
        return materialize_text_lines_recipe(
            state_dir,
            source=sources[recipe.source_dataset_id],
            recipe=recipe,
        )


@dataclass(frozen=True)
class WeightedTextMixturePreparationPlugin:
    plugin_id: str = WEIGHTED_TEXT_MIXTURE_PLUGIN_ID
    plugin_version: str = WEIGHTED_TEXT_MIXTURE_PLUGIN_VERSION
    title: str = "Deterministic weighted mixture of managed UTF-8 corpora"
    accepts_prepared_source: bool = True
    requires_prepared_source: bool = True

    def build_recipe(
        self,
        *,
        source_dataset_id: str,
        source_fingerprint: str,
        config: dict[str, object] | None = None,
    ) -> DataPreparationRecipe:
        if config is None:
            raise FrontierwrightError(
                "DATA_RECIPE_CONFIG_UNSUPPORTED",
                "weighted-text-mixture requires explicit source configuration.",
                2,
            )
        return make_weighted_text_mixture_recipe(
            source_dataset_id=source_dataset_id,
            source_fingerprint=source_fingerprint,
            config=config,
        )

    def source_bindings(
        self,
        recipe: DataPreparationRecipe,
    ) -> tuple[DataSourceBinding, ...]:
        return _mixture_source_bindings(recipe)

    def materialize(
        self,
        state_dir: Path,
        *,
        sources: dict[str, DatasetDescriptor],
        recipe: DataPreparationRecipe,
    ) -> DatasetDescriptor:
        return materialize_weighted_text_mixture_recipe(
            state_dir,
            sources=sources,
            recipe=recipe,
        )



@dataclass(frozen=True)
class ByteShardsPreparationPlugin:
    plugin_id: str = BYTE_SHARDS_PLUGIN_ID
    plugin_version: str = BYTE_SHARDS_PLUGIN_VERSION
    title: str = "Materialize uint8 byte-ID training shards"
    accepts_prepared_source: bool = True
    requires_prepared_source: bool = True

    def build_recipe(
        self,
        *,
        source_dataset_id: str,
        source_fingerprint: str,
        config: dict[str, object] | None = None,
    ) -> DataPreparationRecipe:
        return make_byte_shards_recipe(
            source_dataset_id=source_dataset_id,
            source_fingerprint=source_fingerprint,
            config=config,
        )

    def source_bindings(
        self,
        recipe: DataPreparationRecipe,
    ) -> tuple[DataSourceBinding, ...]:
        return (
            DataSourceBinding(
                dataset_id=recipe.source_dataset_id,
                fingerprint=recipe.source_fingerprint,
            ),
        )

    def materialize(
        self,
        state_dir: Path,
        *,
        sources: dict[str, DatasetDescriptor],
        recipe: DataPreparationRecipe,
    ) -> DatasetDescriptor:
        return materialize_byte_shards_recipe(
            state_dir,
            source=sources[recipe.source_dataset_id],
            recipe=recipe,
        )


BUILTIN_DATA_PREPARATION_PLUGINS: tuple[DataPreparationPlugin, ...] = (
    SnapshotCopyPreparationPlugin(),
    TextLinesPreparationPlugin(),
    WeightedTextMixturePreparationPlugin(),
    ByteShardsPreparationPlugin(),
)

_DATA_PREPARATION_PLUGINS = {
    plugin.plugin_id: plugin for plugin in BUILTIN_DATA_PREPARATION_PLUGINS
}


def data_preparation_plugin(plugin_id: str) -> DataPreparationPlugin:
    try:
        return _DATA_PREPARATION_PLUGINS[plugin_id]
    except KeyError as exc:
        raise FrontierwrightError(
            "DATA_RECIPE_UNSUPPORTED",
            f"Unknown data preparation plugin: {plugin_id}",
            2,
        ) from exc
