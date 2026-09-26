import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    ("origin", "edition", "action_title"),
    [
        ("ZERO", "ACADEMY", "ACADEMY GUIDE"),
        ("IMPORTED_LOCAL", "STUDIO", "STUDIO WORKBENCH"),
        ("INTERNAL_LAB", "LAB", "LAB CONTROL PLANE"),
    ],
)
def test_scripted_play_reaches_each_edition_action_center_without_a_pty(
    tmp_path: Path,
    origin: str,
    edition: str,
    action_title: str,
) -> None:
    project = tmp_path / edition.lower()
    created = runner.invoke(
        app,
        [
            "project",
            "init",
            str(project),
            "--name",
            f"{edition}-QA",
            "--origin",
            origin,
            "--edition",
            edition,
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert created.exit_code == 0, created.output

    script = tmp_path / f"{edition.lower()}-play.json"
    script.write_text(
        json.dumps(
            {"steps": [{"press": ["?"]}, {"press": ["escape", "a"]}]}
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "play",
            "--path",
            str(project),
            "--script",
            str(script),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)

    assert payload["mode"] == "SCRIPTED_TUI_QA"
    assert payload["snapshots"][1]["screen"] == "HelpScreen"
    assert payload["snapshots"][2]["screen"] == "ActionCenterScreen"
    assert action_title in payload["snapshots"][2]["text"]


def test_scripted_play_drives_real_tui_without_a_pty(tmp_path: Path) -> None:
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
            "--yes",
        ],
    )
    assert created.exit_code == 0, created.output

    script = tmp_path / "play.json"
    script.write_text(
        json.dumps(
            {
                "steps": [
                    {"press": ["?"]},
                    {"press": ["escape", "a"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "play",
            "--path",
            str(tmp_path),
            "--script",
            str(script),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["mode"] == "SCRIPTED_TUI_QA"
    assert len(payload["snapshots"]) == 3
    assert payload["snapshots"][0]["active_tab"] == "character"
    assert payload["snapshots"][1]["screen"] == "HelpScreen"
    assert payload["snapshots"][2]["screen"] == "ActionCenterScreen"
    assert "ACADEMY GUIDE" in payload["snapshots"][2]["text"]


def test_scripted_play_can_fill_a_visible_form(tmp_path: Path) -> None:
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
            "--yes",
        ],
    )
    assert created.exit_code == 0, created.output

    script = tmp_path / "form-play.json"
    script.write_text(
        json.dumps(
            {
                "steps": [
                    {"press": ["a", "enter"]},
                    {"set_input": {"id": "form-field-0", "value": "workspace/sample.txt"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "play",
            "--path",
            str(tmp_path),
            "--script",
            str(script),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)

    assert payload["snapshots"][-1]["focused_widget"] == "form-field-0"
    assert payload["snapshots"][-1]["inputs"]["form-field-0"] == "workspace/sample.txt"



def test_scripted_snapshot_text_contains_only_visible_active_tab(tmp_path: Path) -> None:
    created = runner.invoke(
        app,
        [
            "project",
            "init",
            str(tmp_path),
            "--name",
            "VISIBLE-QA",
            "--origin",
            "ZERO",
            "--edition",
            "ACADEMY",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert created.exit_code == 0, created.output

    script = tmp_path / "visible-play.json"
    script.write_text(json.dumps({"steps": []}), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "play",
            "--path",
            str(tmp_path),
            "--script",
            str(script),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    snapshot = json.loads(result.stdout)["snapshots"][0]

    assert snapshot["active_tab"] == "character"
    assert snapshot["visibility_semantics"] == "VISIBLE_ON_SCREEN_ONLY"
    assert "GROWTH STAT · ORIGIN MODEL = 0" in snapshot["text"]
    assert "No datasets registered." not in snapshot["text"]
    assert "No datasets registered." in snapshot["all_text"]


def test_scripted_play_renders_korean_help_and_action_center(tmp_path: Path) -> None:
    created = runner.invoke(
        app,
        [
            "project",
            "init",
            str(tmp_path),
            "--name",
            "KO-QA",
            "--origin",
            "ZERO",
            "--edition",
            "ACADEMY",
            "--lang",
            "ko",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert created.exit_code == 0, created.output

    script = tmp_path / "ko-play.json"
    script.write_text(
        json.dumps({"steps": [{"press": ["?"]}, {"press": ["escape", "a"]}]}),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "play",
            "--path",
            str(tmp_path),
            "--script",
            str(script),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    snapshots = json.loads(result.stdout)["snapshots"]

    assert "직접 만들면서 이해하기" in snapshots[1]["text"]
    assert "ACADEMY 가이드" in snapshots[2]["text"]
    assert "로컬 데이터셋 등록" in snapshots[2]["text"]
