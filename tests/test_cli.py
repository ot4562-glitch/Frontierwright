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

    assert payload["history_confidence"] == "UNKNOWN"
    assert payload["strong_recommendation_allowed"] is False
    assert payload["build_mode"] == "NOT_READY"


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
