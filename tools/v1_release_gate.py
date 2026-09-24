"""Frontierwright v1 candidate release-gate runner.

This is intentionally repository-side tooling, not a runtime product command.
It proves the canonical v1 gate through static checks, the complete automated
test suite, clean package installation, and a real PyTorch lifecycle smoke.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateStep:
    name: str
    argv: tuple[str, ...]
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "duration_seconds": self.duration_seconds,
        }


def _run(name: str, argv: list[str], *, cwd: Path, timeout: float) -> GateStep:
    started = time.monotonic()
    result = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    duration = time.monotonic() - started
    if result.returncode != 0:
        raise SystemExit(
            f"v1 gate failed: {name}\n"
            f"argv={argv!r}\n"
            f"exit={result.returncode}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    return GateStep(name=name, argv=tuple(argv), duration_seconds=duration)


def _require_file(root: Path, relative: str) -> None:
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"v1 gate required file missing: {relative}")


def _require_clean(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    if result.stdout.strip():
        raise SystemExit(
            "v1 gate requires a clean git worktree after the candidate commit:\n"
            + result.stdout
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-python",
        type=Path,
        required=True,
        help="Python executable with Frontierwright train dependencies installed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-step timeout in seconds.",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail unless the repository worktree is clean.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    training_python = args.training_python.expanduser()
    if not training_python.is_file():
        raise SystemExit(f"training Python not found: {training_python}")

    for relative in (
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        ".github/workflows/ci.yml",
        "product/V1_IMPLEMENTATION_BLUEPRINT_20260922.md",
        "tests/test_tui_release_e2e.py",
        "tests/test_release_gate.py",
        "tools/release_smoke.py",
        "tools/v1_e2e_smoke.py",
    ):
        _require_file(root, relative)

    if args.require_clean:
        _require_clean(root)

    py = sys.executable
    steps = [
        _run("git-diff-check", ["git", "diff", "--check"], cwd=root, timeout=args.timeout),
        _run("ruff", [py, "-m", "ruff", "check", "."], cwd=root, timeout=args.timeout),
        _run(
            "mypy",
            [py, "-m", "mypy", "src/frontierwright"],
            cwd=root,
            timeout=args.timeout,
        ),
        _run("pytest", [py, "-m", "pytest", "-q"], cwd=root, timeout=args.timeout),
        _run("build", [py, "-m", "build"], cwd=root, timeout=args.timeout),
        _run(
            "clean-wheel-smoke",
            [py, "tools/release_smoke.py"],
            cwd=root,
            timeout=args.timeout,
        ),
        _run(
            "real-training-e2e",
            [str(training_python), "tools/v1_e2e_smoke.py", "--timeout", str(args.timeout)],
            cwd=root,
            timeout=max(args.timeout, 180.0),
        ),
    ]

    print(
        json.dumps(
            {
                "ok": True,
                "gate": "Frontierwright canonical v1 candidate",
                "steps": [step.to_dict() for step in steps],
                "training_python": str(training_python),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
