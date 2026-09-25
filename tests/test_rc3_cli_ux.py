import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app

runner = CliRunner()


def _init_academy(path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "project",
            "init",
            str(path),
            "--name",
            "NOVA",
            "--origin",
            "ZERO",
            "--edition",
            "ACADEMY",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output


def test_rc3_resources_diagnose_returns_actionable_probe_evidence(tmp_path: Path) -> None:
    _init_academy(tmp_path)

    result = runner.invoke(
        app,
        [
            "resources",
            "diagnose",
            "--path",
            str(tmp_path),
            "--json",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["detected_at"]
    diagnostics = payload["diagnostics"]
    assert diagnostics
    assert all(
        item["status"] in {"MEASURED", "MEASUREMENT_NEEDED", "UNSUPPORTED_HERE"}
        for item in diagnostics
    )
    gpu = [item for item in diagnostics if item["probe_id"].startswith("gpu.")]
    assert gpu
    assert all(item["physical_absence_proven"] is False for item in gpu)
    unresolved = [item for item in diagnostics if item["status"] != "MEASURED"]
    assert all(item["next_action"] for item in unresolved)


def test_rc3_project_init_human_output_gives_next_action(tmp_path: Path) -> None:
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
            "--edition",
            "ACADEMY",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Next:" in result.stdout
    assert "frontierwright play" in result.stdout


def test_rc3_root_help_explains_core_human_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output

    assert "Open the keyboard-first human interface" in result.stdout
    assert "Import an existing local model" in result.stdout
    assert "Execute a calibrated READY training plan" in result.stdout
