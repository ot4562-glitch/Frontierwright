"""Frontierwright rc3 candidate integrity gate.

RC3 retains every rc2 user-fit/training/decision contract and adds explicit measurement
diagnostics, edition-specific first-run UX, and deterministic non-PTY Textual play.
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


RC3_CRITERIA: tuple[GateCriterion, ...] = (
    GateCriterion(
        "RC3-01",
        "versioned workload profiles and acceptance contracts separate coverage from success",
        (
            "tests/test_workloads.py",
            "tests/test_workload_acceptance.py",
            "tests/test_workload_acceptance_service.py",
        ),
    ),
    GateCriterion(
        "RC3-02",
        "missing or threshold-overlapping evidence becomes actionable measurement work",
        ("tests/test_workloads.py", "tests/test_fit_planner.py", "tests/test_tui.py"),
    ),
    GateCriterion(
        "RC3-03",
        "promotion revalidates exact versioned workload-fit gates before Champion mutation",
        ("tests/test_candidates.py", "tests/test_workload_evaluations.py"),
    ),
    GateCriterion(
        "RC3-04",
        "partial or mismatched evidence cannot become an unqualified superiority claim",
        ("tests/test_workloads.py", "tests/test_model_comparison.py"),
    ),
    GateCriterion(
        "RC3-05",
        "external evaluation evidence is pinned to exact model/evaluator/task/metric identity",
        ("tests/test_evaluator_adapters.py", "tests/test_workload_evaluations.py"),
    ),
    GateCriterion(
        "RC3-06",
        "serving and resource evidence composes only under compatible measured conditions",
        ("tests/test_serving_adapters.py", "tests/test_inference_profile.py"),
    ),
    GateCriterion(
        "RC3-07",
        "real-use observations are atomic, retry-safe, and workload-revision specific",
        ("tests/test_observations.py", "tests/test_migrations.py"),
    ),
    GateCriterion(
        "RC3-08",
        "real RL keeps explicit environment/reward identity and rejects direct observation rewards",
        ("tests/test_rl.py", "tools/rl_reference_smoke.py"),
    ),
    GateCriterion(
        "RC3-09",
        "edition workflows differ without changing shared evidence semantics",
        ("tests/test_tui.py", "src/frontierwright/editions.py"),
    ),
    GateCriterion(
        "RC3-10",
        "bounded controlled-private Slurm execution stays hash-pinned and scope-explicit",
        ("tests/test_slurm_executor.py", "src/frontierwright/slurm_executor.py"),
    ),
    GateCriterion(
        "RC3-11",
        "stock/open baselines and descendants compare through the same measured user-fit evidence",
        ("tests/test_model_comparison.py",),
    ),
    GateCriterion(
        "RC3-12",
        "package, clean-wheel install, real training, and real reference RL remain healthy",
        (
            "pytest",
            "tools/release_smoke.py",
            "tools/v1_e2e_smoke.py",
            "tools/rl_reference_smoke.py",
        ),
    ),
    GateCriterion(
        "RC3-13",
        (
            "resource probe failures retain reason codes and next actions without "
            "claiming hardware absence"
        ),
        ("tests/test_resources.py", "tests/test_rc3_cli_ux.py"),
    ),
    GateCriterion(
        "RC3-14",
        "Academy, Studio, and Lab expose distinct navigation and first-run explanations",
        ("tests/test_rc3_ux.py", "product/RC3_MEASUREMENT_AND_EDITION_UX_20260925.md"),
    ),
    GateCriterion(
        "RC3-15",
        "CodexPro and other non-PTY hosts can drive the actual Textual UI through scripted play",
        (
            "tests/test_scripted_play.py",
            "src/frontierwright/tui/scripted.py",
            "tools/rc3_scripted_play_smoke.py",
        ),
    ),
    GateCriterion(
        "RC3-16",
        "public onboarding uses real CLI routes and immediately identifies the next human action",
        ("tests/test_rc3_cli_ux.py", "README.md"),
    ),
)

if len(RC3_CRITERIA) != 16:
    raise RuntimeError("rc3 release gate must contain exactly 16 criteria")


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
            f"rc3 gate failed: {name}\n"
            f"argv={argv!r}\n"
            f"exit={result.returncode}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    return GateStep(name=name, argv=tuple(argv), duration_seconds=duration)


def _require_file(root: Path, relative: str) -> None:
    if not (root / relative).is_file():
        raise SystemExit(f"rc3 gate required file missing: {relative}")


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
            "rc3 gate requires a clean git worktree after the candidate commit:\n" + result.stdout
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

    required = {
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        ".github/workflows/ci.yml",
        "product/USER_FIT_OPTIMIZATION_AND_EDITION_EXPERIENCE_20260924.md",
        "product/RC3_MEASUREMENT_AND_EDITION_UX_20260925.md",
        "src/frontierwright/workload_acceptance.py",
        "src/frontierwright/slurm_executor.py",
        "src/frontierwright/tui/scripted.py",
        "tools/release_smoke.py",
        "tools/v1_e2e_smoke.py",
        "tools/rl_reference_smoke.py",
        "tools/rc3_scripted_play_smoke.py",
    }
    for criterion in RC3_CRITERIA:
        for evidence in criterion.evidence:
            if evidence == "pytest":
                continue
            required.add(evidence.split("::", 1)[0])
    for relative in sorted(required):
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
        _run(
            "scripted-tui-qa",
            [py, "tools/rc3_scripted_play_smoke.py"],
            cwd=root,
            timeout=args.timeout,
        ),
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
        _run(
            "real-reference-rl-smoke",
            [str(training_python), "tools/rl_reference_smoke.py"],
            cwd=root,
            timeout=max(args.timeout, 180.0),
        ),
    ]

    print(
        json.dumps(
            {
                "ok": True,
                "gate": "Frontierwright rc3 candidate integrity",
                "candidate_version": __version__,
                "criteria_passed": len(RC3_CRITERIA),
                "criteria_total": len(RC3_CRITERIA),
                "criteria": [criterion.to_dict() for criterion in RC3_CRITERIA],
                "steps": [step.to_dict() for step in steps],
                "training_python": str(training_python),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
