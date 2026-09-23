import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app

runner = CliRunner()


def test_cli_json_contract_is_stable_for_zero_project(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "project",
            "init",
            str(tmp_path),
            "--name",
            "NOVA",
            "--origin",
            "ZERO",
            "--lang",
            "en",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)

    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["initialized"] is True
    assert payload["project_name"] == "NOVA"
    assert payload["origin"] == "ZERO"
    assert payload["edition_profile"] == "ACADEMY"
    assert payload["edition_name"] == "Frontierwright Academy"
    assert payload["edition_starting_point"] == "BIRTH"
    assert payload["history_confidence"] == "COMPLETE"
    assert payload["measurement_state"] == "NOT_READY"
    assert payload["build_mode"] == "INTENT"
    assert payload["strong_recommendation_allowed"] is True
    assert payload["stats"] == {
        "coding": None,
        "general": None,
        "math": None,
        "reasoning": None,
    }

    status = runner.invoke(
        app,
        [
            "status",
            "--path",
            str(tmp_path),
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert status.exit_code == 0, status.output
    assert json.loads(status.stdout) == payload


def test_imported_local_json_withholds_recommendation(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "project",
            "init",
            str(tmp_path),
            "--name",
            "Imported",
            "--origin",
            "IMPORTED_LOCAL",
            "--json",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)

    assert payload["edition_profile"] == "STUDIO"
    assert payload["edition_name"] == "Frontierwright Studio"
    assert payload["history_confidence"] == "UNKNOWN"
    assert payload["strong_recommendation_allowed"] is False
    assert payload["build_mode"] == "NOT_READY"


def test_project_edition_command_changes_profile_without_reinit(tmp_path: Path) -> None:
    created = runner.invoke(
        app,
        [
            "project",
            "init",
            str(tmp_path),
            "--name",
            "NOVA",
            "--origin",
            "ZERO",
            "--edition",
            "ACADEMY",
            "--json",
            "--non-interactive",
        ],
    )
    assert created.exit_code == 0, created.output
    initial = json.loads(created.stdout)
    project_id = initial["project_id"]

    changed = runner.invoke(
        app,
        [
            "project",
            "edition",
            "--path",
            str(tmp_path),
            "--set",
            "STUDIO",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert changed.exit_code == 0, changed.output
    payload = json.loads(changed.stdout)

    assert payload["project_id"] == project_id
    assert payload["edition_profile"] == "STUDIO"
    assert payload["edition_name"] == "Frontierwright Studio"
    assert payload["origin"] == "ZERO"


def test_uninitialized_machine_error_has_stable_shape(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["status", "--path", str(tmp_path), "--json", "--non-interactive"],
    )
    assert result.exit_code == 10
    payload = json.loads(result.stdout)
    assert payload == {
        "error": {
            "code": "NOT_INITIALIZED",
            "message": "No Frontierwright project is initialized here.",
        },
        "ok": False,
        "schema_version": 1,
    }
