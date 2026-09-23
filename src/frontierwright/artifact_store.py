"""Managed candidate artifact sealing and integrity verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from frontierwright.errors import FrontierwrightError
from frontierwright.execution import TrainingPlan
from frontierwright.models import ImportedModelDescriptor, inspect_local_model, sha256_file

ARTIFACT_MANIFEST_SCHEMA = 1


@dataclass(frozen=True)
class SealedArtifact:
    model_id: str
    run_id: str
    artifact_root: Path
    model_path: Path
    manifest_path: Path
    manifest_sha256: str
    descriptor: ImportedModelDescriptor


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest_sha256(payload: dict[str, object]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(payload)).hexdigest()}"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_descriptor_files(
    descriptor: ImportedModelDescriptor,
    destination_root: Path,
) -> Path:
    source = descriptor.source_path
    if source.is_file():
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / source.name
        if source.is_symlink():
            raise FrontierwrightError(
                "ARTIFACT_SYMLINK_REJECTED",
                "Candidate artifact files must not be symbolic links.",
                13,
            )
        shutil.copy2(source, destination)
        return destination

    destination_root.mkdir(parents=True, exist_ok=True)
    for item in descriptor.files:
        relative = Path(item.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise FrontierwrightError(
                "ARTIFACT_PATH_ESCAPE",
                f"Candidate artifact contains unsafe path: {item.relative_path}",
                13,
            )
        source_file = source / relative
        if source_file.is_symlink():
            raise FrontierwrightError(
                "ARTIFACT_SYMLINK_REJECTED",
                f"Candidate artifact contains symbolic link: {item.relative_path}",
                13,
            )
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
    return destination_root


def _manifest_payload(
    *,
    model_id: str,
    run_id: str,
    plan: TrainingPlan,
    descriptor: ImportedModelDescriptor,
    model_relpath: str,
) -> dict[str, object]:
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "model_id": model_id,
        "run_id": run_id,
        "plan_id": plan.plan_id,
        "path_id": plan.path_id.value,
        "intervention_id": plan.intervention_id,
        "intervention_version": plan.intervention_version,
        "intervention_family": plan.intervention_family,
        "parent_model_id": plan.model_id,
        "dataset_id": plan.dataset_id,
        "dataset_fingerprint": plan.dataset_fingerprint,
        "dataset_recipe_id": plan.dataset_recipe_id,
        "dataset_recipe_hash": plan.dataset_recipe_hash,
        "backend_id": plan.backend_id,
        "backend_spec_hash": plan.backend_spec_hash,
        "effective_config": plan.config,
        "model_relpath": model_relpath,
        "model_format": descriptor.model_format.value,
        "trainable": descriptor.trainable,
        "model_fingerprint": descriptor.fingerprint,
        "total_bytes": descriptor.total_bytes,
        "files": [asdict(item) for item in descriptor.files],
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "ARTIFACT_MANIFEST_INVALID",
            f"Artifact manifest is unreadable: {path}",
            13,
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
        raise FrontierwrightError(
            "ARTIFACT_MANIFEST_INVALID",
            "Artifact manifest schema is invalid.",
            13,
        )
    return raw


def _sealed_from_existing(artifact_root: Path) -> SealedArtifact:
    manifest_path = artifact_root / "artifact-manifest.json"
    raw = _read_manifest(manifest_path)
    model_relpath = raw.get("model_relpath")
    if not isinstance(model_relpath, str) or not model_relpath:
        raise FrontierwrightError(
            "ARTIFACT_MANIFEST_INVALID",
            "Artifact manifest lacks model_relpath.",
            13,
        )
    model_path = (artifact_root / model_relpath).resolve()
    try:
        model_path.relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise FrontierwrightError(
            "ARTIFACT_PATH_ESCAPE",
            "Artifact manifest model path escapes managed artifact root.",
            13,
        ) from exc

    descriptor = inspect_local_model(model_path)
    expected_fingerprint = raw.get("model_fingerprint")
    if descriptor.fingerprint != expected_fingerprint:
        raise FrontierwrightError(
            "ARTIFACT_TAMPERED",
            "Managed candidate artifact fingerprint no longer matches its sealed manifest.",
            13,
        )
    expected_total = raw.get("total_bytes")
    if not isinstance(expected_total, int) or descriptor.total_bytes != expected_total:
        raise FrontierwrightError(
            "ARTIFACT_TAMPERED",
            "Managed candidate artifact size no longer matches its sealed manifest.",
            13,
        )

    expected_files = raw.get("files")
    actual_files = [asdict(item) for item in descriptor.files]
    if expected_files != actual_files:
        raise FrontierwrightError(
            "ARTIFACT_TAMPERED",
            "Managed candidate artifact file manifest has changed.",
            13,
        )

    model_id = raw.get("model_id")
    run_id = raw.get("run_id")
    if not isinstance(model_id, str) or not isinstance(run_id, str):
        raise FrontierwrightError(
            "ARTIFACT_MANIFEST_INVALID",
            "Artifact manifest lacks model_id or run_id.",
            13,
        )

    return SealedArtifact(
        model_id=model_id,
        run_id=run_id,
        artifact_root=artifact_root.resolve(),
        model_path=model_path,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=f"sha256:{sha256_file(manifest_path)}",
        descriptor=descriptor,
    )


def seal_training_artifact(
    state_dir: Path,
    *,
    source_path: Path,
    model_id: str,
    run_id: str,
    plan: TrainingPlan,
) -> SealedArtifact:
    """Copy a backend result into a Frontierwright-managed immutable-by-contract artifact."""

    descriptor = inspect_local_model(source_path)
    artifacts_root = state_dir.resolve() / "artifacts"
    final_root = artifacts_root / model_id

    if final_root.exists():
        sealed = _sealed_from_existing(final_root)
        if sealed.model_id != model_id or sealed.run_id != run_id:
            raise FrontierwrightError(
                "ARTIFACT_ID_CONFLICT",
                "Managed artifact path is already bound to a different model/run.",
                13,
            )
        if sealed.descriptor.fingerprint != descriptor.fingerprint:
            raise FrontierwrightError(
                "ARTIFACT_ID_CONFLICT",
                "Existing managed artifact differs from backend result.",
                13,
            )
        return sealed

    staging_parent = artifacts_root / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{model_id}-", dir=staging_parent)
    ).resolve()

    try:
        model_container = staging_root / "model"
        copied_model_path = _copy_descriptor_files(descriptor, model_container)
        sealed_descriptor = inspect_local_model(copied_model_path)
        if sealed_descriptor.fingerprint != descriptor.fingerprint:
            raise FrontierwrightError(
                "ARTIFACT_COPY_MISMATCH",
                "Copied candidate artifact fingerprint differs from backend output.",
                13,
            )

        model_relpath = copied_model_path.relative_to(staging_root).as_posix()
        payload = _manifest_payload(
            model_id=model_id,
            run_id=run_id,
            plan=plan,
            descriptor=sealed_descriptor,
            model_relpath=model_relpath,
        )
        manifest_path = staging_root / "artifact-manifest.json"
        _atomic_write_json(manifest_path, payload)

        final_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staging_root, final_root)
        except FileExistsError:
            # Another reconciler may have completed the same idempotent finalization.
            shutil.rmtree(staging_root, ignore_errors=True)

        sealed = _sealed_from_existing(final_root)
        if sealed.descriptor.fingerprint != descriptor.fingerprint:
            raise FrontierwrightError(
                "ARTIFACT_COPY_MISMATCH",
                "Final managed artifact differs from backend output.",
                13,
            )
        return sealed
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise


def verify_sealed_artifact(artifact_root: Path) -> SealedArtifact:
    return _sealed_from_existing(artifact_root.resolve())


def verify_manifest_digest(manifest_path: Path, expected_sha256: str) -> None:
    actual = f"sha256:{sha256_file(manifest_path)}"
    if actual != expected_sha256:
        raise FrontierwrightError(
            "ARTIFACT_MANIFEST_TAMPERED",
            "Managed artifact manifest hash no longer matches the registry.",
            13,
        )
