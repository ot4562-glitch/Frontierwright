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

from frontierwright import __version__


@dataclass(frozen=True)
class GateCriterion:
    criterion_id: str
    requirement: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "criterion_id": self.criterion_id,
            "requirement": self.requirement,
            "status": "PASS",
            "evidence": list(self.evidence),
        }


CANONICAL_CRITERIA: tuple[GateCriterion, ...] = (
    GateCriterion(
        "V1-01",
        "frontierwright play launches the full-screen keyboard TUI",
        ("tests/test_tui.py", "tests/test_tui_release_e2e.py"),
    ),
    GateCriterion(
        "V1-02",
        "primary human loop is keyboard-only",
        ("tests/test_tui_release_e2e.py::test_keyboard_tui_academy_full_primary_loop",),
    ),
    GateCriterion(
        "V1-03",
        "Character/Build/Paths/Resources/Data/History/Candidates are reachable",
        ("tests/test_tui.py::test_tui_starts_with_required_sections_and_keyboard_navigation",),
    ),
    GateCriterion(
        "V1-04",
        "consequential TUI actions use shared core with machine equivalents",
        ("tests/test_tui_release_e2e.py", "tests/test_cli_features.py"),
    ),
    GateCriterion(
        "V1-05",
        "AI automation completes supported workflows through CLI/JSON",
        ("tools/v1_e2e_smoke.py",),
    ),
    GateCriterion(
        "V1-06",
        "real local model import immediately exposes a character sheet",
        (
            "tests/test_tui_release_e2e.py::test_keyboard_tui_studio_import_immediately_exposes_real_character_sheet",
        ),
    ),
    GateCriterion(
        "V1-07",
        "zero model reaches a first meaningful measured state",
        ("tools/v1_e2e_smoke.py", "tests/test_tui_release_e2e.py"),
    ),
    GateCriterion(
        "V1-08",
        "stats are backed by reproducible evaluation",
        ("tests/test_capability_v1.py", "src/frontierwright/capability_v1.py"),
    ),
    GateCriterion(
        "V1-09",
        "origin-aware Build editing works for zero/imported/internal-lab",
        ("tests/test_build.py",),
    ),
    GateCriterion(
        "V1-10",
        "history confidence gates strong recommendations",
        ("tests/test_models.py", "tests/test_data_paths.py"),
    ),
    GateCriterion(
        "V1-11",
        "real paths expose READY/LOCKED with reasons",
        ("tests/test_data_paths.py", "tests/test_execution.py"),
    ),
    GateCriterion(
        "V1-12",
        "resource and user-data-first requirements are inspectable",
        ("tests/test_resources.py", "tests/test_data_paths.py", "tests/test_tui.py"),
    ),
    GateCriterion(
        "V1-13",
        "real candidate can train/evaluate/compare/promote and update state",
        ("tools/v1_e2e_smoke.py",),
    ),
    GateCriterion(
        "V1-14", "rejected candidates cannot corrupt Champion", ("tests/test_candidates.py",)
    ),
    GateCriterion(
        "V1-15",
        "expensive lifecycle actions have dry-run and idempotency",
        (
            "tests/test_execution.py",
            "tests/test_birth.py",
            "tests/test_tokenizer_birth.py",
            "tests/test_evaluation_runner.py",
            "tests/test_capability_v1.py",
            "tests/test_merge.py",
            "tests/test_quantization.py",
            "tests/test_exporting.py",
            "tests/test_cli_features.py",
        ),
    ),
    GateCriterion(
        "V1-16",
        "expert/AI JSON and exit-code contracts are stable",
        ("tests/test_cli.py", "tests/test_cli_features.py"),
    ),
    GateCriterion(
        "V1-17",
        "core local workflow operates offline",
        ("tests/test_release_gate.py::test_core_local_workflow_operates_with_network_disabled",),
    ),
    GateCriterion(
        "V1-18",
        "game metadata does not contaminate evaluation",
        (
            "tests/test_release_gate.py::test_evaluation_backend_request_excludes_game_and_project_metadata",
        ),
    ),
    GateCriterion(
        "V1-19",
        "Apache-2.0 and third-party release notices are prepared",
        (
            "tests/test_release_gate.py::test_release_legal_notices_are_prepared",
            "tools/release_smoke.py",
        ),
    ),
    GateCriterion(
        "V1-20",
        "TUI/unit/integration/E2E verification passes",
        ("pytest", "tools/release_smoke.py", "tools/v1_e2e_smoke.py"),
    ),
)

if len(CANONICAL_CRITERIA) != 20:
    raise RuntimeError("canonical v1 release gate must contain exactly 20 criteria")


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
            "v1 gate requires a clean git worktree after the candidate commit:\n" + result.stdout
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
                "candidate_version": __version__,
                "criteria_passed": len(CANONICAL_CRITERIA),
                "criteria_total": len(CANONICAL_CRITERIA),
                "criteria": [criterion.to_dict() for criterion in CANONICAL_CRITERIA],
                "steps": [step.to_dict() for step in steps],
                "training_python": str(training_python),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
