"""Reproducible dataset-preparation recipes and managed artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from frontierwright.data import DatasetDescriptor, inspect_local_dataset
from frontierwright.errors import FrontierwrightError

SNAPSHOT_COPY_PLUGIN_ID = "frontierwright.data.snapshot-copy"
SNAPSHOT_COPY_PLUGIN_VERSION = "1"


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


class DataPreparationPlugin(Protocol):
    """Contract for reproducible dataset preparation implementations."""

    @property
    def plugin_id(self) -> str: ...

    @property
    def plugin_version(self) -> str: ...

    @property
    def title(self) -> str: ...

    def build_recipe(
        self,
        *,
        source_dataset_id: str,
        source_fingerprint: str,
        config: dict[str, object] | None = None,
    ) -> DataPreparationRecipe: ...

    def materialize(
        self,
        state_dir: Path,
        *,
        source: DatasetDescriptor,
        recipe: DataPreparationRecipe,
    ) -> DatasetDescriptor: ...


@dataclass(frozen=True)
class SnapshotCopyPreparationPlugin:
    plugin_id: str = SNAPSHOT_COPY_PLUGIN_ID
    plugin_version: str = SNAPSHOT_COPY_PLUGIN_VERSION
    title: str = "Byte-preserving managed snapshot"

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

    def materialize(
        self,
        state_dir: Path,
        *,
        source: DatasetDescriptor,
        recipe: DataPreparationRecipe,
    ) -> DatasetDescriptor:
        return materialize_snapshot_recipe(
            state_dir,
            source=source,
            recipe=recipe,
        )


BUILTIN_DATA_PREPARATION_PLUGINS: tuple[DataPreparationPlugin, ...] = (
    SnapshotCopyPreparationPlugin(),
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
