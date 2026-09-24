"""Portable Frontierwright model export bundles.

Exports are copy-only operations. They never mutate model/champion state and never
embed absolute source paths in the portable manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontierwright.errors import FrontierwrightError
from frontierwright.models import ImportedModelDescriptor, inspect_local_model, sha256_file

EXPORT_SCHEMA_VERSION = 1
EXPORT_MANIFEST_NAME = "frontierwright-export.json"


@dataclass(frozen=True)
class PortableExportBundle:
    export_id: str
    destination: Path
    manifest_path: Path
    manifest_sha256: str
    model_path: Path
    descriptor: ImportedModelDescriptor
    replayed: bool


@dataclass(frozen=True)
class PortableExportVerification:
    export_id: str
    destination: Path
    manifest_path: Path
    manifest_sha256: str
    model_path: Path
    descriptor: ImportedModelDescriptor


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest_sha256(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def _export_id_for(
    *,
    model_fingerprint: str,
    provenance: dict[str, object],
) -> str:
    identity_payload: dict[str, object] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "intervention_id": "frontierwright.operate.portable-export",
        "intervention_version": "1",
        "model_fingerprint": model_fingerprint,
        "provenance": provenance,
    }
    digest = hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest()
    return f"export-{digest[:32]}"


def _safe_bundle_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FrontierwrightError(
            "EXPORT_PATH_ESCAPE",
            "Portable export manifest contains an unsafe model path.",
            13,
        ) from exc
    return candidate


def _copy_model(
    descriptor: ImportedModelDescriptor,
    destination: Path,
) -> tuple[Path, str]:
    source = descriptor.source_path.resolve()
    model_root = destination / "model"
    model_root.mkdir(parents=True, exist_ok=False)

    if source.is_file():
        if source.is_symlink():
            raise FrontierwrightError(
                "EXPORT_SOURCE_SYMLINK",
                "Portable export refuses symbolic-link model sources.",
                13,
            )
        target = model_root / source.name
        shutil.copy2(source, target)
        return target, f"model/{source.name}"

    for item in descriptor.files:
        relative = Path(item.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise FrontierwrightError(
                "EXPORT_PATH_ESCAPE",
                f"Unsafe model artifact path: {item.relative_path}",
                13,
            )
        source_file = (source / relative).resolve()
        try:
            source_file.relative_to(source)
        except ValueError as exc:
            raise FrontierwrightError(
                "EXPORT_PATH_ESCAPE",
                f"Model artifact escapes source root: {item.relative_path}",
                13,
            ) from exc
        if not source_file.is_file():
            raise FrontierwrightError(
                "EXPORT_SOURCE_DRIFT",
                f"Model artifact disappeared before export: {item.relative_path}",
                13,
            )
        target = model_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
    return model_root, "model"


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            f"Portable export manifest is unreadable: {path}",
            13,
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export manifest schema is invalid.",
            13,
        )
    return raw


def _verify_existing(
    destination: Path,
    *,
    expected_export_id: str,
    expected_fingerprint: str,
) -> PortableExportBundle:
    manifest_path = destination / EXPORT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FrontierwrightError(
            "EXPORT_DESTINATION_CONFLICT",
            "Export destination already exists without a Frontierwright export manifest.",
            13,
        )
    manifest = _read_manifest(manifest_path)
    if manifest.get("export_id") != expected_export_id:
        raise FrontierwrightError(
            "EXPORT_DESTINATION_CONFLICT",
            "Export destination is bound to a different export identity.",
            13,
        )
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export manifest lacks artifact metadata.",
            13,
        )
    model_relative = artifact.get("bundle_model_path")
    if not isinstance(model_relative, str) or not model_relative:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export manifest lacks bundle_model_path.",
            13,
        )
    model_path = _safe_bundle_path(destination, model_relative)
    descriptor = inspect_local_model(model_path)
    if descriptor.fingerprint != expected_fingerprint:
        raise FrontierwrightError(
            "EXPORT_TAMPERED",
            "Exported model bytes no longer match the exported fingerprint.",
            13,
        )
    return PortableExportBundle(
        export_id=expected_export_id,
        destination=destination.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_manifest_sha256(manifest_path),
        model_path=model_path,
        descriptor=descriptor,
        replayed=True,
    )


def verify_portable_export(destination: Path) -> PortableExportVerification:
    """Verify bundle self-consistency and exact exported model bytes.

    This is integrity verification, not an authenticity/signature claim.
    """

    destination = destination.expanduser().resolve()
    manifest_path = destination / EXPORT_MANIFEST_NAME
    if not destination.is_dir() or not manifest_path.is_file():
        raise FrontierwrightError(
            "EXPORT_NOT_FOUND",
            "Portable Frontierwright export bundle was not found.",
            12,
        )

    manifest = _read_manifest(manifest_path)
    if manifest.get("intervention_id") != "frontierwright.operate.portable-export":
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export intervention identity is invalid.",
            13,
        )
    if manifest.get("intervention_version") != "1":
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export intervention version is unsupported.",
            13,
        )

    provenance = manifest.get("provenance")
    artifact = manifest.get("artifact")
    export_id = manifest.get("export_id")
    if not isinstance(provenance, dict) or not isinstance(artifact, dict):
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export manifest lacks provenance or artifact metadata.",
            13,
        )
    fingerprint = artifact.get("fingerprint")
    bundle_model_path = artifact.get("bundle_model_path")
    if not isinstance(export_id, str) or not export_id:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export manifest lacks export_id.",
            13,
        )
    if not isinstance(fingerprint, str) or not fingerprint:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export manifest lacks artifact fingerprint.",
            13,
        )
    if not isinstance(bundle_model_path, str) or not bundle_model_path:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_INVALID",
            "Portable export manifest lacks bundle_model_path.",
            13,
        )

    expected_export_id = _export_id_for(
        model_fingerprint=fingerprint,
        provenance=provenance,
    )
    if export_id != expected_export_id:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_TAMPERED",
            "Portable export identity does not match its manifest contents.",
            13,
        )

    model_path = _safe_bundle_path(destination, bundle_model_path)
    descriptor = inspect_local_model(model_path)
    if descriptor.fingerprint != fingerprint:
        raise FrontierwrightError(
            "EXPORT_TAMPERED",
            "Exported model bytes do not match the manifest fingerprint.",
            13,
        )

    expected_format = artifact.get("model_format")
    expected_trainable = artifact.get("trainable")
    expected_total_bytes = artifact.get("total_bytes")
    expected_files = artifact.get("files")
    actual_files = [
        {
            "relative_path": item.relative_path,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in descriptor.files
    ]
    if expected_format != descriptor.model_format.value:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_TAMPERED",
            "Exported model format does not match the manifest.",
            13,
        )
    if expected_trainable is not descriptor.trainable:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_TAMPERED",
            "Exported model trainability does not match the manifest.",
            13,
        )
    if expected_total_bytes != descriptor.total_bytes:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_TAMPERED",
            "Exported model size does not match the manifest.",
            13,
        )
    if expected_files != actual_files:
        raise FrontierwrightError(
            "EXPORT_MANIFEST_TAMPERED",
            "Exported model file manifest does not match the bundle.",
            13,
        )

    return PortableExportVerification(
        export_id=export_id,
        destination=destination,
        manifest_path=manifest_path,
        manifest_sha256=_manifest_sha256(manifest_path),
        model_path=model_path,
        descriptor=descriptor,
    )


def publish_portable_export(
    destination: Path,
    *,
    descriptor: ImportedModelDescriptor,
    provenance: dict[str, object],
) -> PortableExportBundle:
    """Atomically copy a model plus portable Frontierwright provenance evidence."""

    destination = destination.expanduser().resolve()
    source = descriptor.source_path.resolve()
    if source.is_dir():
        try:
            destination.relative_to(source)
        except ValueError:
            pass
        else:
            raise FrontierwrightError(
                "EXPORT_DESTINATION_UNSAFE",
                "Export destination must not be inside the source model directory.",
                2,
            )

    export_id = _export_id_for(
        model_fingerprint=descriptor.fingerprint,
        provenance=provenance,
    )

    if destination.exists():
        return _verify_existing(
            destination,
            expected_export_id=export_id,
            expected_fingerprint=descriptor.fingerprint,
        )

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.frontierwright-", dir=parent)
    ).resolve()

    try:
        model_path, bundle_model_path = _copy_model(descriptor, staging)
        copied = inspect_local_model(model_path)
        if copied.fingerprint != descriptor.fingerprint:
            raise FrontierwrightError(
                "EXPORT_COPY_MISMATCH",
                "Portable export model fingerprint changed during copy.",
                13,
            )

        manifest: dict[str, object] = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "export_id": export_id,
            "intervention_id": "frontierwright.operate.portable-export",
            "intervention_version": "1",
            "artifact": {
                "bundle_model_path": bundle_model_path,
                "model_format": descriptor.model_format.value,
                "trainable": descriptor.trainable,
                "fingerprint": descriptor.fingerprint,
                "total_bytes": descriptor.total_bytes,
                "files": [
                    {
                        "relative_path": item.relative_path,
                        "size_bytes": item.size_bytes,
                        "sha256": item.sha256,
                    }
                    for item in descriptor.files
                ],
            },
            "provenance": provenance,
        }
        manifest_path = staging / EXPORT_MANIFEST_NAME
        manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")

        try:
            os.replace(staging, destination)
        except FileExistsError:
            shutil.rmtree(staging, ignore_errors=True)
            return _verify_existing(
                destination,
                expected_export_id=export_id,
                expected_fingerprint=descriptor.fingerprint,
            )

        final_manifest = destination / EXPORT_MANIFEST_NAME
        final_model_path = _safe_bundle_path(destination, bundle_model_path)
        final_descriptor = inspect_local_model(final_model_path)
        if final_descriptor.fingerprint != descriptor.fingerprint:
            raise FrontierwrightError(
                "EXPORT_COPY_MISMATCH",
                "Published export bytes do not match the source model fingerprint.",
                13,
            )
        return PortableExportBundle(
            export_id=export_id,
            destination=destination,
            manifest_path=final_manifest,
            manifest_sha256=_manifest_sha256(final_manifest),
            model_path=final_model_path,
            descriptor=final_descriptor,
            replayed=False,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
