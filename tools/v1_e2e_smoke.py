"""Frontierwright v1 end-to-end release smoke.

Run this from an environment installed with the train extra. The smoke uses only
local files and the installed CLI surface. It exercises a real zero-model Academy
lineage through birth, one-step training, frozen Capability v1 measurement,
candidate comparison, and explicit promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "frontierwright", *args]
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"argv={command!r}\n"
            f"exit={result.returncode}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    return result


def _json(
    args: list[str],
    *,
    timeout: float = 180.0,
    yes: bool = True,
) -> dict[str, Any]:
    suffix = ["--json", "--non-interactive"]
    if yes:
        suffix.append("--yes")
    result = _run([*args, *suffix], timeout=timeout)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {args!r}")
    if payload.get("ok") is False:
        raise RuntimeError(f"command returned ok=false: {payload!r}")
    return payload


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"missing nonempty {key}: {payload!r}")
    return value


def _wait_for_candidate(project: Path, run_id: str, timeout_seconds: float) -> str:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = _json(
            [
                "run-reconcile",
                run_id,
                "--path",
                str(project),
            ],
            timeout=60.0,
            yes=False,
        )
        status = last.get("status")
        if status == "COMPLETED":
            return _required_str(last, "candidate_model_id")
        if status in {"FAILED", "INCOMPLETE"}:
            raise RuntimeError(f"training terminated without candidate: {last!r}")
        time.sleep(0.25)
    raise RuntimeError(f"training did not finish before timeout; last={last!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-root",
        type=Path,
        help="Use this directory instead of an automatically removed temporary directory.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum wait for the one-step training run.",
    )
    args = parser.parse_args()

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="frontierwright-v1-e2e-")
        root = Path(temporary.name)
    else:
        root = args.work_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

    try:
        project = root / "project"
        corpus = root / "corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "train.txt").write_text(
            (
                "Frontierwright develops real language models from reproducible state. "
                "A model is measured, trained, compared, and explicitly promoted.\n"
            )
            * 64,
            encoding="utf-8",
        )

        init = _json(
            [
                "project",
                "init",
                str(project),
                "--name",
                "V1Smoke",
                "--origin",
                "ZERO",
                "--edition",
                "ACADEMY",
            ]
        )
        if init.get("edition_profile") != "ACADEMY" or init.get("origin") != "ZERO":
            raise RuntimeError(f"unexpected project identity: {init!r}")

        data = _json(
            [
                "data",
                "add",
                str(corpus),
                "--role",
                "PRETRAIN",
                "--classification",
                "PRIVATE",
                "--name",
                "V1 smoke pretrain",
                "--path",
                str(project),
            ]
        )
        datasets = data.get("datasets")
        if not isinstance(datasets, list) or len(datasets) != 1:
            raise RuntimeError(f"expected one registered dataset: {data!r}")
        dataset = datasets[0]
        if not isinstance(dataset, dict):
            raise RuntimeError(f"invalid dataset payload: {data!r}")
        dataset_id = _required_str(dataset, "dataset_id")

        _json(
            [
                "build",
                "intent",
                "--path",
                str(project),
                "--archetype",
                "Balanced",
                "--priority",
                "general=1",
                "--priority",
                "reasoning=1",
                "--priority",
                "math=1",
                "--priority",
                "coding=1",
            ]
        )

        tokenizer = _json(
            [
                "birth",
                "tokenizer",
                dataset_id,
                "--path",
                str(project),
                "--vocab-size",
                "384",
            ]
        )
        tokenizer_id = _required_str(tokenizer, "artifact_id")

        born = _json(
            [
                "birth",
                "zero",
                "--path",
                str(project),
                "--preset",
                "zero-8m",
                "--seed",
                "42",
                "--tokenizer-artifact",
                tokenizer_id,
                "--python",
                sys.executable,
            ],
            timeout=120.0,
        )
        root_model_id = _required_str(born, "model_id")

        _json(["resources", "detect", "--path", str(project)])

        root_capability = _json(
            [
                "eval",
                "capability-v1",
                "--path",
                str(project),
                "--model",
                root_model_id,
                "--python",
                sys.executable,
                "--device",
                "cpu",
            ],
            timeout=180.0,
        )
        root_stats = root_capability.get("stats")
        if not isinstance(root_stats, dict) or any(
            root_stats.get(axis) is None
            for axis in ("general", "reasoning", "math", "coding")
        ):
            raise RuntimeError(f"birth root did not reach measured state: {root_capability!r}")

        backend_spec = root / "reference-backend.json"
        _json(
            [
                "backend",
                "reference-spec",
                "--python",
                sys.executable,
                "--output",
                str(backend_spec),
            ]
        )
        config_path = root / "train-config.json"
        config_path.write_text(
            json.dumps(
                {
                    "steps": 1,
                    "calibration_steps": 1,
                    "batch_size": 1,
                    "learning_rate": 0.0003,
                    "weight_decay": 0.0,
                    "seed": 7,
                    "device": "cpu",
                    "max_dataset_bytes": 16384,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        plan = _json(
            [
                "plan",
                "create",
                "FROM_SCRATCH_PRETRAINING",
                "--path",
                str(project),
                "--backend-spec",
                str(backend_spec),
                "--dataset",
                dataset_id,
                "--permission",
                "EXECUTE_SINGLE",
                "--config-json",
                str(config_path),
                "--max-runs",
                "1",
                "--max-wall-seconds",
                "120",
                "--max-storage-bytes",
                str(256 * 1024 * 1024),
            ]
        )
        plan_id = _required_str(plan, "plan_id")

        calibrated = _json(
            [
                "calibrate",
                plan_id,
                "--path",
                str(project),
                "--backend-spec",
                str(backend_spec),
                "--timeout-seconds",
                "120",
            ],
            timeout=180.0,
        )
        if calibrated.get("ready") is not True:
            raise RuntimeError(f"calibrated plan is not READY: {calibrated!r}")

        dry = _json(
            [
                "run",
                plan_id,
                "--path",
                str(project),
                "--backend-spec",
                str(backend_spec),
                "--dry-run",
                "--timeout-seconds",
                "120",
            ]
        )
        if dry.get("status") != "DRY_RUN" or dry.get("run_id") is not None:
            raise RuntimeError(f"training dry-run contract failed: {dry!r}")

        started = _json(
            [
                "run",
                plan_id,
                "--path",
                str(project),
                "--backend-spec",
                str(backend_spec),
                "--timeout-seconds",
                "120",
            ]
        )
        run_id = _required_str(started, "run_id")
        candidate_model_id = _wait_for_candidate(project, run_id, args.timeout)

        candidate_capability = _json(
            [
                "eval",
                "capability-v1",
                "--path",
                str(project),
                "--model",
                candidate_model_id,
                "--python",
                sys.executable,
                "--device",
                "cpu",
            ],
            timeout=180.0,
        )
        candidate_stats = candidate_capability.get("stats")
        if not isinstance(candidate_stats, dict) or any(
            candidate_stats.get(axis) is None
            for axis in ("general", "reasoning", "math", "coding")
        ):
            raise RuntimeError(
                f"candidate did not receive Capability v1 stats: {candidate_capability!r}"
            )

        comparison = _json(
            [
                "compare",
                candidate_model_id,
                "--path",
                str(project),
            ],
            yes=False,
        )
        if comparison.get("candidate_model_id") != candidate_model_id:
            raise RuntimeError(f"candidate compare identity mismatch: {comparison!r}")

        promoted = _json(
            [
                "promote",
                candidate_model_id,
                "--path",
                str(project),
            ]
        )
        if promoted.get("champion_model_id") != candidate_model_id:
            raise RuntimeError(f"candidate was not promoted: {promoted!r}")

        status = _json(["status", "--path", str(project)], yes=False)
        if status.get("champion_model_id") != candidate_model_id:
            raise RuntimeError(f"status did not reflect promoted champion: {status!r}")
        if status.get("measurement_state") != "MEASURED":
            raise RuntimeError(f"promoted champion did not retain measured state: {status!r}")

        history = _json(["history", "--path", str(project)], yes=False)
        events = history.get("events")
        if not isinstance(events, list):
            raise RuntimeError("history surface did not return events")
        kinds = {event.get("kind") for event in events if isinstance(event, dict)}
        required_kinds = {
            "MODEL_BORN",
            "RUN_COMPLETED",
            "EVALUATION_RECEIPT_GENERATED",
            "CANDIDATE_PROMOTED",
        }
        if not required_kinds.issubset(kinds):
            raise RuntimeError(
                f"history is missing lifecycle evidence: {required_kinds - kinds!r}"
            )

        print(
            json.dumps(
                {
                    "ok": True,
                    "project": str(project),
                    "root_model_id": root_model_id,
                    "candidate_model_id": candidate_model_id,
                    "run_id": run_id,
                    "plan_id": plan_id,
                    "root_stats": root_stats,
                    "candidate_stats": candidate_stats,
                    "history_event_count": len(events),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
