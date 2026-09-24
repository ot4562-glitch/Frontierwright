import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.domain import ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.models import inspect_local_model
from frontierwright.service import (
    export_champion_bundle,
    import_local_model,
    preflight_export_champion_bundle,
    verify_export_bundle,
)

runner = CliRunner()


def make_hf_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "frontierwright_byte_causal_lm",
                "frontierwright_reference_backend": (
                    "frontierwright-reference-pytorch-v1"
                ),
                "preset": "zero-8m",
                "parameter_count": 8_000_000,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"portable-export-weights")
    return root


def setup_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    source = make_hf_model(tmp_path / "source-model")
    import_local_model(
        project,
        source,
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Portable",
    )
    return project, source


def test_portable_export_preserves_model_fingerprint_without_source_paths(
    tmp_path: Path,
) -> None:
    project, source = setup_project(tmp_path)
    destination = tmp_path / "bundle"

    view = export_champion_bundle(project, destination)

    assert view.replayed is False
    assert view.export_id is not None
    assert view.model_fingerprint is not None
    assert view.manifest_sha256 is not None
    assert view.model_path is not None

    exported = inspect_local_model(Path(view.model_path))
    source_descriptor = inspect_local_model(source)
    assert exported.fingerprint == source_descriptor.fingerprint == view.model_fingerprint

    manifest_path = destination / "frontierwright-export.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert str(source.resolve()) not in manifest_text
    assert str(project.resolve()) not in manifest_text
    assert manifest["export_id"] == view.export_id
    assert manifest["artifact"]["fingerprint"] == source_descriptor.fingerprint
    assert manifest["artifact"]["bundle_model_path"] == "model"
    assert manifest["provenance"]["model"]["model_id"] == view.model_id
    assert manifest["provenance"]["project"]["name"] == "Portable"
    assert manifest["provenance"]["capability_evidence"] is None


def test_identical_portable_export_replays_existing_bundle(tmp_path: Path) -> None:
    project, _ = setup_project(tmp_path)
    destination = tmp_path / "bundle"

    first = export_champion_bundle(project, destination)
    second = export_champion_bundle(project, destination)

    assert first.export_id == second.export_id
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.replayed is False
    assert second.replayed is True




def test_export_dry_run_is_side_effect_free_and_predicts_replay(tmp_path: Path) -> None:
    project, _ = setup_project(tmp_path)
    destination = tmp_path / "bundle"

    preview = preflight_export_champion_bundle(project, destination)
    assert preview.action == "operate-export"
    assert preview.ready is True
    assert preview.would_replay is False
    assert not destination.exists()

    result = runner.invoke(
        app,
        [
            "operate", "export", str(destination),
            "--path", str(project),
            "--dry-run", "--json", "--non-interactive", "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["action"] == "operate-export"
    assert payload["would_replay"] is False
    assert not destination.exists()

    exported = export_champion_bundle(project, destination)
    replay = preflight_export_champion_bundle(project, destination)
    assert replay.would_replay is True
    assert replay.details["export_id"] == exported.export_id

def test_exported_model_tampering_is_detected_on_replay(tmp_path: Path) -> None:
    project, _ = setup_project(tmp_path)
    destination = tmp_path / "bundle"
    first = export_champion_bundle(project, destination)
    assert first.model_path is not None

    (Path(first.model_path) / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(FrontierwrightError, match="Exported model bytes"):
        export_champion_bundle(project, destination)


def test_export_destination_conflict_is_not_overwritten(tmp_path: Path) -> None:
    project, _ = setup_project(tmp_path)
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "keep.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FrontierwrightError, match="already exists"):
        export_champion_bundle(project, destination)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_export_destination_inside_source_model_is_rejected(tmp_path: Path) -> None:
    project, source = setup_project(tmp_path)

    with pytest.raises(FrontierwrightError, match="inside the source model"):
        export_champion_bundle(project, source / "portable-export")


def test_operate_export_json_machine_surface(tmp_path: Path) -> None:
    project, _ = setup_project(tmp_path)
    destination = tmp_path / "bundle"

    result = runner.invoke(
        app,
        [
            "operate",
            "export",
            str(destination),
            "--path",
            str(project),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["intervention_id"] == "frontierwright.operate.portable-export"
    assert payload["model_id"]
    assert payload["model_fingerprint"]
    assert payload["destination"] == str(destination.resolve())


def test_portable_export_verify_service_and_cli(tmp_path: Path) -> None:
    project, _ = setup_project(tmp_path)
    destination = tmp_path / "bundle"
    exported = export_champion_bundle(project, destination)

    verified = verify_export_bundle(destination)
    assert verified.valid is True
    assert verified.export_id == exported.export_id
    assert verified.model_fingerprint == exported.model_fingerprint
    assert verified.manifest_sha256 == exported.manifest_sha256
    assert verified.authenticity == "NOT_SIGNED"

    result = runner.invoke(
        app,
        [
            "operate",
            "verify",
            str(destination),
            "--json",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["valid"] is True
    assert payload["export_id"] == exported.export_id
    assert payload["authenticity"] == "NOT_SIGNED"


def test_portable_export_manifest_tampering_is_detected(tmp_path: Path) -> None:
    project, _ = setup_project(tmp_path)
    destination = tmp_path / "bundle"
    export_champion_bundle(project, destination)

    manifest_path = destination / "frontierwright-export.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["project"]["name"] = "tampered-name"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FrontierwrightError, match="identity does not match"):
        verify_export_bundle(destination)
