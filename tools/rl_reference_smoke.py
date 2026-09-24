"""Real PyTorch smoke for Frontierwright's built-in verifiable-reward RL path."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from frontierwright.reference_backend import _birth, _load_config, _train

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".rl-reference-smoke"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rl_config() -> dict[str, object]:
    return {
        "steps": 3,
        "calibration_steps": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "seed": 23,
        "device": "cpu",
        "rl": {
            "schema_version": 1,
            "algorithm_id": "reinforce",
            "environment": {
                "id": "frontierwright.reference.choice-env",
                "version": "1",
                "kind": "VERIFIABLE_MULTIPLE_CHOICE",
                "config": {"episode_semantics": "contextual_bandit"},
            },
            "reward": {
                "id": "frontierwright.reference.exact-choice",
                "version": "1",
                "kind": "EXACT_CORRECT_CHOICE",
                "config": {"correct": 1.0, "incorrect": 0.0},
            },
            "rollout_temperature": 1.0,
            "entropy_coefficient": 0.01,
            "algorithm_config": {"reward_baseline": 0.5},
        },
    }


def main() -> int:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    try:
        birth_root = WORK / "birth"
        born = _birth(
            {
                "preset": "zero-8m",
                "seed": 7,
                "output_root": str(birth_root),
            }
        )
        birth_model = Path(str(born["output_model_path"]))

        corpus = WORK / "corpus.txt"
        corpus.write_text(
            (
                "Frontierwright trains real model descendants from measured evidence. "
                "A candidate is evaluated before promotion. "
            )
            * 8,
            encoding="utf-8",
        )
        pretrain_root = WORK / "pretrain"
        pretrain_request: dict[str, object] = {
            "path_id": "FROM_SCRATCH_PRETRAINING",
            "model_source_path": str(birth_model),
            "dataset_source_path": str(corpus),
            "dataset_fingerprint": "smoke-pretrain",
            "output_root": str(pretrain_root),
            "run_id": "smoke-pretrain",
            "config": {
                "steps": 1,
                "calibration_steps": 1,
                "batch_size": 1,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "seed": 11,
                "device": "cpu",
            },
        }
        pretrained = _train(pretrain_request, _load_config(pretrain_request))
        policy_model = Path(str(pretrained["output_model_path"]))
        before_hash = _sha256(policy_model / "pytorch_model.bin")

        episodes = WORK / "episodes.jsonl"
        records = [
            {"prompt": "2 + 2 =", "choices": [" 4", " 5"], "correct_index": 0},
            {
                "prompt": "Capital of France:",
                "choices": [" Paris", " Berlin"],
                "correct_index": 0,
            },
            {
                "prompt": "Opposite of hot:",
                "choices": [" cold", " tall"],
                "correct_index": 0,
            },
            {
                "prompt": "3 * 3 =",
                "choices": [" 9", " 8"],
                "correct_index": 0,
            },
        ]
        episodes.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
        )

        rl_root = WORK / "rl"
        request: dict[str, object] = {
            "path_id": "RL_POLICY_OPTIMIZATION",
            "model_source_path": str(policy_model),
            "dataset_source_path": str(episodes),
            "dataset_fingerprint": "smoke-rollouts",
            "output_root": str(rl_root),
            "run_id": "smoke-rl",
            "config": _rl_config(),
        }
        result = _train(request, _load_config(request))
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        candidate_model = Path(str(result["output_model_path"]))
        after_hash = _sha256(candidate_model / "pytorch_model.bin")

        assert before_hash != after_hash
        assert metrics["objective"] == "reinforce_verifiable_choice"
        assert metrics["rollout_episodes"] == 6
        assert metrics["successful_episodes"] >= 0
        assert 0.0 <= float(metrics["reward_mean"]) <= 1.0
        training = metrics["training"]
        assert isinstance(training, dict)
        assert training["method"] == "reinforcement_learning"
        assert training["algorithm"] == "reinforce"
        assert training["contextual_bandit_episode"] is True
        assert isinstance(training["rl_spec_hash"], str)

        print(
            json.dumps(
                {
                    "ok": True,
                    "objective": metrics["objective"],
                    "rollout_episodes": metrics["rollout_episodes"],
                    "successful_episodes": metrics["successful_episodes"],
                    "reward_mean": metrics["reward_mean"],
                    "reward_initial": metrics["reward_initial"],
                    "reward_final": metrics["reward_final"],
                    "weights_changed": before_hash != after_hash,
                    "rl_spec_hash": training["rl_spec_hash"],
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
