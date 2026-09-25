import json
from pathlib import Path

import pytest

from frontierwright.data import DatasetRole
from frontierwright.domain import HistoryConfidence, ModelOrigin
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import HardBudgets, PermissionLevel
from frontierwright.interventions import (
    InterventionFamily,
    intervention_for_training_path,
    training_intervention_for_path,
)
from frontierwright.paths import PathContext, TrainingPathId
from frontierwright.rl import (
    REFERENCE_RL_ENVIRONMENT_KIND,
    REFERENCE_RL_REWARD_KIND,
    RLExperimentSpec,
    RLIdentity,
    rl_spec_from_config,
    validate_reference_rl_spec,
)
from frontierwright.service import add_local_dataset, create_training_plan, import_local_model


def _rl_config(*, reward_version: str = "1") -> dict[str, object]:
    return {
        "rl": {
            "schema_version": 1,
            "algorithm_id": "reinforce",
            "environment": {
                "id": "fixture-choice-environment",
                "version": "1",
                "kind": REFERENCE_RL_ENVIRONMENT_KIND,
                "config": {"episode_semantics": "one_prompt_one_choice"},
            },
            "reward": {
                "id": "fixture-exact-choice-reward",
                "version": reward_version,
                "kind": REFERENCE_RL_REWARD_KIND,
                "config": {"correct": 1.0, "incorrect": 0.0},
            },
            "rollout_temperature": 1.0,
            "entropy_coefficient": 0.01,
            "algorithm_config": {"reward_baseline": 0.5},
        }
    }


def _fake_hf_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"type":"fixture"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"fixture-weights")
    return root


def _backend_spec(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": "fixture-rl-backend",
                "data_boundary": "LOCAL_MACHINE",
                "supported_paths": ["RL_POLICY_OPTIMIZATION"],
                "calibrate_argv": ["fixture", "calibrate", "{request_json}"],
                "train_argv": ["fixture", "train", "{request_json}"],
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_rl_spec_has_stable_identity_and_rejects_fake_rl() -> None:
    first = rl_spec_from_config(_rl_config())
    second = rl_spec_from_config(_rl_config())

    assert first.spec_hash == second.spec_hash
    assert first.algorithm_id == "reinforce"
    validate_reference_rl_spec(first)

    changed = rl_spec_from_config(_rl_config(reward_version="2"))
    assert changed.spec_hash != first.spec_hash

    with pytest.raises(FrontierwrightError, match="config.rl object"):
        rl_spec_from_config({"dpo_beta": 0.1})


def test_reference_rl_contract_rejects_non_rl_reward_or_algorithm() -> None:
    environment = RLIdentity(
        "env",
        "1",
        REFERENCE_RL_ENVIRONMENT_KIND,
        {},
    )
    reward = RLIdentity("reward", "1", REFERENCE_RL_REWARD_KIND, {})

    validate_reference_rl_spec(
        RLExperimentSpec(
            algorithm_id="reinforce",
            environment=environment,
            reward=reward,
        )
    )

    with pytest.raises(FrontierwrightError, match="algorithm_id=reinforce"):
        validate_reference_rl_spec(
            RLExperimentSpec(
                algorithm_id="grpo",
                environment=environment,
                reward=reward,
            )
        )


def test_rl_is_builtin_align_path_and_requires_rollout_evidence() -> None:
    intervention = intervention_for_training_path(TrainingPathId.RL_POLICY_OPTIMIZATION)
    assert intervention.intervention_id == "frontierwright.align.reinforcement-learning"
    assert intervention.family is InterventionFamily.ALIGN

    without_rollout = PathContext(
        origin=ModelOrigin.IMPORTED_LOCAL,
        champion_present=True,
        champion_trainable=True,
        history_confidence=HistoryConfidence.VERIFIED,
        resource_profile_available=True,
        dataset_roles=frozenset(),
    )
    locked = training_intervention_for_path(TrainingPathId.RL_POLICY_OPTIMIZATION).assess(
        without_rollout
    )
    assert locked.availability.value == "LOCKED"
    assert "ROLLOUT environment/task dataset required" in locked.blockers

    with_rollout = PathContext(
        origin=ModelOrigin.IMPORTED_LOCAL,
        champion_present=True,
        champion_trainable=True,
        history_confidence=HistoryConfidence.VERIFIED,
        resource_profile_available=True,
        dataset_roles=frozenset({DatasetRole.ROLLOUT}),
    )
    plannable = training_intervention_for_path(TrainingPathId.RL_POLICY_OPTIMIZATION).assess(
        with_rollout
    )
    assert plannable.availability.value == "PLANNABLE"
    assert plannable.blockers == ()
    assert "pin reward source identity and version" in plannable.next_checks


def test_rl_plan_pins_environment_reward_and_rollout_dataset(tmp_path: Path) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        _fake_hf_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="RL",
    )
    data = tmp_path / "rollouts"
    data.mkdir()
    (data / "episodes.jsonl").write_text(
        '{"prompt":"2+2=","choices":["4","5"],"correct_index":0}\n',
        encoding="utf-8",
    )
    data_view = add_local_dataset(
        project,
        data,
        name="Verifier episodes",
        role=DatasetRole.ROLLOUT,
    )
    rollout = next(item for item in data_view.datasets if item["role"] == "ROLLOUT")
    backend = _backend_spec(tmp_path / "backend.json")

    first = create_training_plan(
        project,
        path_id=TrainingPathId.RL_POLICY_OPTIMIZATION,
        backend_spec_path=backend,
        dataset_id=str(rollout["dataset_id"]),
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=1),
        config=_rl_config(),
    )
    changed_reward = create_training_plan(
        project,
        path_id=TrainingPathId.RL_POLICY_OPTIMIZATION,
        backend_spec_path=backend,
        dataset_id=str(rollout["dataset_id"]),
        permission=PermissionLevel.EXECUTE_SINGLE,
        budgets=HardBudgets(max_runs=1),
        config=_rl_config(reward_version="2"),
    )

    assert first.path_id == "RL_POLICY_OPTIMIZATION"
    assert first.intervention_id == "frontierwright.align.reinforcement-learning"
    assert first.dataset_id == rollout["dataset_id"]
    assert first.config == _rl_config()
    assert first.idempotency_key != changed_reward.idempotency_key
    assert "representative calibration required" in first.blockers


def test_rl_plan_refuses_missing_reward_environment_identity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    import_local_model(
        project,
        _fake_hf_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="RL",
    )
    data = tmp_path / "rollouts"
    data.mkdir()
    (data / "episodes.jsonl").write_text(
        '{"prompt":"2+2=","choices":["4","5"],"correct_index":0}\n',
        encoding="utf-8",
    )
    add_local_dataset(
        project,
        data,
        name="Verifier episodes",
        role=DatasetRole.ROLLOUT,
    )

    with pytest.raises(FrontierwrightError, match="config.rl object"):
        create_training_plan(
            project,
            path_id=TrainingPathId.RL_POLICY_OPTIMIZATION,
            backend_spec_path=_backend_spec(tmp_path / "backend.json"),
            dataset_id=None,
            permission=PermissionLevel.EXECUTE_SINGLE,
            budgets=HardBudgets(max_runs=1),
            config={},
        )


def test_rl_contract_rejects_usage_observation_as_direct_reward() -> None:
    environment = RLIdentity(
        "env",
        "1",
        REFERENCE_RL_ENVIRONMENT_KIND,
        {},
    )
    direct_observation_reward = RLIdentity(
        "usage-outcome",
        "1",
        "USAGE_OBSERVATION",
        {"positive": "SUCCESS", "negative": "FAILURE"},
    )
    with pytest.raises(
        FrontierwrightError,
        match="cannot be used directly as RL rewards",
    ) as direct_exc:
        RLExperimentSpec(
            algorithm_id="reinforce",
            environment=environment,
            reward=direct_observation_reward,
        )
    assert direct_exc.value.code == "RL_OBSERVATION_REWARD_REQUIRES_TRANSFORM"

    disguised_source = RLIdentity(
        "derived-but-unversioned",
        "1",
        "CUSTOM_REWARD",
        {"source_kind": "OPERATIONAL_OUTCOME"},
    )
    with pytest.raises(FrontierwrightError) as source_exc:
        RLExperimentSpec(
            algorithm_id="reinforce",
            environment=environment,
            reward=disguised_source,
        )
    assert source_exc.value.code == "RL_OBSERVATION_REWARD_REQUIRES_TRANSFORM"
