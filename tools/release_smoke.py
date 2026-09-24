"""Clean-wheel release smoke test for Frontierwright.

Build artifacts are produced separately. This script creates a fresh virtual
environment, installs exactly one wheel, then exercises the installed CLI
without importing the source checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _venv_frontierwright(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "frontierwright.exe"
    return root / "bin" / "frontierwright"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path, nargs="?")
    args = parser.parse_args()

    if args.wheel is None:
        wheels = sorted(Path("dist").glob("frontierwright-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected exactly one wheel in dist/, found {len(wheels)}")
        wheel = wheels[0].resolve()
    else:
        wheel = args.wheel.expanduser().resolve()
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")

    with tempfile.TemporaryDirectory(prefix="frontierwright-release-smoke-") as tmp:
        tmp_root = Path(tmp)
        venv_root = tmp_root / "venv"
        project_root = tmp_root / "project"

        _run([sys.executable, "-m", "venv", str(venv_root)])
        python = _venv_python(venv_root)
        cli = _venv_frontierwright(venv_root)

        _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(python), "-m", "pip", "install", str(wheel)])
        _run([str(cli), "--help"])

        init = _run(
            [
                str(cli),
                "project",
                "init",
                str(project_root),
                "--name",
                "ReleaseSmoke",
                "--origin",
                "ZERO",
                "--edition",
                "ACADEMY",
                "--json",
                "--non-interactive",
                "--yes",
            ]
        )
        init_payload = json.loads(init.stdout)
        if init_payload.get("ok") is not True:
            raise SystemExit("project init smoke did not return ok=true")

        status = _run(
            [
                str(cli),
                "status",
                "--path",
                str(project_root),
                "--json",
                "--non-interactive",
                "--yes",
            ]
        )
        status_payload = json.loads(status.stdout)
        if status_payload.get("initialized") is not True:
            raise SystemExit("installed CLI could not reopen initialized project")
        if status_payload.get("edition_profile") != "ACADEMY":
            raise SystemExit("installed CLI lost Academy edition profile")
        if status_payload.get("origin") != "ZERO":
            raise SystemExit("installed CLI lost ZERO origin")

        print(
            json.dumps(
                {
                    "ok": True,
                    "wheel": wheel.name,
                    "python": str(python),
                    "project": str(project_root),
                    "edition_profile": status_payload.get("edition_profile"),
                    "origin": status_payload.get("origin"),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
