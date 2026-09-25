"""Black-box smoke for rc3 non-PTY TUI play and resource diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(argv: list[str], *, cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"rc3 scripted-play smoke failed: {argv!r}\n"
            f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"rc3 scripted-play smoke produced no JSON: {argv!r}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"rc3 scripted-play smoke produced invalid JSON: {result.stdout}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("rc3 scripted-play smoke JSON must be an object")
    return payload


def _exercise_edition(
    *,
    root: Path,
    python: str,
    temp_root: Path,
    edition: str,
    origin: str,
    action_title: str,
) -> int:
    project = temp_root / edition.lower()
    created = _run(
        [
            python,
            "-m",
            "frontierwright",
            "project",
            "init",
            str(project),
            "--name",
            f"RC3-{edition}-SMOKE",
            "--origin",
            origin,
            "--edition",
            edition,
            "--json",
            "--non-interactive",
            "--yes",
        ],
        cwd=root,
    )
    if created.get("edition_profile") != edition:
        raise SystemExit(f"rc3 smoke did not initialize {edition}")

    script = temp_root / f"{edition.lower()}-play.json"
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
    played = _run(
        [
            python,
            "-m",
            "frontierwright",
            "play",
            "--path",
            str(project),
            "--script",
            str(script),
            "--json",
        ],
        cwd=root,
    )
    if played.get("mode") != "SCRIPTED_TUI_QA":
        raise SystemExit(f"rc3 smoke did not exercise scripted TUI QA for {edition}")
    snapshots = played.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 3:
        raise SystemExit(f"rc3 smoke expected three TUI snapshots for {edition}")
    final = snapshots[-1]
    if not isinstance(final, dict) or final.get("screen") != "ActionCenterScreen":
        raise SystemExit(f"rc3 smoke did not reach the {edition} Action Center")
    if action_title not in str(final.get("text") or ""):
        raise SystemExit(f"rc3 smoke lost {edition}-specific Action Center copy")
    return len(snapshots)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    with tempfile.TemporaryDirectory(prefix="frontierwright-rc3-") as raw:
        temp_root = Path(raw)
        editions = (
            ("ACADEMY", "ZERO", "ACADEMY GUIDE"),
            ("STUDIO", "IMPORTED_LOCAL", "STUDIO WORKBENCH"),
            ("LAB", "INTERNAL_LAB", "LAB CONTROL PLANE"),
        )
        snapshot_counts: dict[str, int] = {}
        for edition, origin, action_title in editions:
            snapshot_counts[edition] = _exercise_edition(
                root=root,
                python=py,
                temp_root=temp_root,
                edition=edition,
                origin=origin,
                action_title=action_title,
            )

        academy_project = temp_root / "academy"
        diagnosed = _run(
            [
                py,
                "-m",
                "frontierwright",
                "resources",
                "diagnose",
                "--path",
                str(academy_project),
                "--json",
                "--non-interactive",
            ],
            cwd=root,
        )
        diagnostics = diagnosed.get("diagnostics")
        if not isinstance(diagnostics, list) or not diagnostics:
            raise SystemExit("rc3 smoke produced no resource diagnostics")
        for item in diagnostics:
            if not isinstance(item, dict):
                raise SystemExit("rc3 smoke resource diagnostic must be an object")
            if item.get("status") not in {
                "MEASURED",
                "MEASUREMENT_NEEDED",
                "UNSUPPORTED_HERE",
            }:
                raise SystemExit(f"unexpected resource diagnostic status: {item!r}")
            if item.get("status") != "MEASURED" and not item.get("next_action"):
                raise SystemExit(f"unresolved resource diagnostic has no next action: {item!r}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": "SCRIPTED_TUI_QA",
                    "editions": list(snapshot_counts),
                    "snapshot_counts": snapshot_counts,
                    "diagnostic_count": len(diagnostics),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
