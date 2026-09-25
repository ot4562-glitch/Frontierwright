"""Frontierwright rc2 development integrity gate.

This gate supplements the frozen v1 candidate gate. It verifies the user-fit,
measurement, decision-integrity, continual-observation, RL-boundary, edition UX,
and bounded Lab execution contracts added on the rc2 development line.
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


RC2_CRITERIA: tuple[GateCriterion, ...] = (
    GateCriterion(
        "RC2-01",
        "versioned workload profiles and acceptance contracts separate coverage from success",
        (
            "tests/test_workloads.py",
            "tests/test_workload_acceptance.py",
            "tests/test_workload_acceptance_service.py",
        ),
    ),
    GateCriterion(
        "RC2-02",
        "missing or threshold-overlapping evidence becomes actionable measurement work",
        (
            "tests/test_workloads.py::test_human_fit_status_turns_uncertainty_into_actionable_measurement_states",
            "tests/test_fit_planner.py::test_inconclusive_fit_schedules_more_measurement_instead_of_stalling",
            "tests/test_tui.py",
        ),
    ),
    GateCriterion(
        "RC2-03",
        "promotion revalidates exact versioned workload-fit gates before Champion mutation",
        ("tests/test_candidates.py", "tests/test_workload_evaluations.py"),
    ),
    GateCriterion(
        "RC2-04",
        "partial or mismatched evidence cannot become an unqualified superiority claim",
        ("tests/test_workloads.py", "tests/test_model_comparison.py"),
    ),
    GateCriterion(
        "RC2-05",
        "external evaluation evidence is pinned to exact model/evaluator/task/metric identity",
        ("tests/test_evaluator_adapters.py", "tests/test_workload_evaluations.py"),
    ),
    GateCriterion(
        "RC2-06",
        "serving and resource evidence composes only under compatible measured conditions",
        ("tests/test_serving_adapters.py", "tests/test_inference_profile.py"),
    ),
    GateCriterion(
        "RC2-07",
        "real-use observations are atomic, retry-safe, and cohort-specific to workload revision",
        ("tests/test_observations.py", "tests/test_migrations.py"),
    ),
    GateCriterion(
        "RC2-08",
        "real RL keeps explicit environment/reward identity and rejects direct observation rewards",
        ("tests/test_rl.py", "tools/rl_reference_smoke.py"),
    ),
    GateCriterion(
        "RC2-09",
        "Academy, Studio, and Lab expose distinct workflows without changing evidence semantics",
        ("tests/test_tui.py", "src/frontierwright/editions.py"),
    ),
    GateCriterion(
        "RC2-10",
        "bounded controlled-private Slurm execution is hash-pinned and explicit about scope",
        ("tests/test_slurm_executor.py", "src/frontierwright/slurm_executor.py"),
    ),
    GateCriterion(
        "RC2-11",
        (
            "stock/open baselines and custom descendants compare through the same "
            "measured user-fit evidence"
        ),
        ("tests/test_model_comparison.py",),
    ),
    GateCriterion(
        "RC2-12",
        "the complete package, clean-wheel install, and real training/RL smoke remain healthy",
        (
            "pytest",
            "tools/release_smoke.py",
            "tools/v1_e2e_smoke.py",
            "tools/rl_reference_smoke.py",
        ),
    ),
)

if len(RC2_CRITERIA) != 12:
    raise RuntimeError("rc2 release gate must contain exactly 12 criteria")


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
            f"rc2 gate failed: {name}\n"
            f"argv={argv!r}\n"
            f"exit={result.returncode}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    return GateStep(name=name, argv=tuple(argv), duration_seconds=duration)


def _require_file(root: Path, relative: str) -> None:
    if not (root / relative).is_file():
        raise SystemExit(f"rc2 gate required file missing: {relative}")


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
            "rc2 gate requires a clean git worktree after the candidate commit:\n"
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

    required = {
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        ".github/workflows/ci.yml",
        "product/USER_FIT_OPTIMIZATION_AND_EDITION_EXPERIENCE_20260924.md",
        "src/frontierwright/workload_acceptance.py",
        "src/frontierwright/slurm_executor.py",
        "tests/test_workload_acceptance.py",
        "tests/test_observations.py",
        "tests/test_rl.py",
        "tests/test_slurm_executor.py",
        "tools/release_smoke.py",
        "tools/v1_e2e_smoke.py",
        "tools/rl_reference_smoke.py",
    }
    for criterion in RC2_CRITERIA:
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
                "gate": "Frontierwright rc2 development integrity",
                "candidate_version": __version__,
                "criteria_passed": len(RC2_CRITERIA),
                "criteria_total": len(RC2_CRITERIA),
                "criteria": [criterion.to_dict() for criterion in RC2_CRITERIA],
                "steps": [step.to_dict() for step in steps],
                "training_python": str(training_python),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
