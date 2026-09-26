from __future__ import annotations

import inspect
import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app, plan_create
from frontierwright.data import DatasetClassification
from frontierwright.domain import ModelOrigin
from frontierwright.service import initialize_project, set_workload_profile
from frontierwright.workloads import WorkloadProfile

runner = CliRunner()


def test_benchmark_source_catalog_is_public_json() -> None:
    result = runner.invoke(app, ["benchmarks", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["stat_axes"] == [
        "KNOWLEDGE",
        "REASONING",
        "MATH",
        "CODING",
        "INSTRUCTION",
        "LANGUAGE",
        "CONTEXT",
    ]
    ids = {item["source_id"] for item in payload["sources"]}
    assert {"livebench", "livecodebench", "scicode", "longbench-v2"} <= ids


def test_acceptance_schema_and_bound_example_are_discoverable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    initialize_project(project, name="RC4", origin=ModelOrigin.IMPORTED_LOCAL)
    workload = set_workload_profile(
        project,
        WorkloadProfile(
            name="My workload",
            privacy=DatasetClassification.PRIVATE,
            languages=("en",),
            domains=("coding",),
            task_weights={"coding": 1.0},
        ),
    )

    schema_result = runner.invoke(app, ["workload", "acceptance", "schema", "--json"])
    assert schema_result.exit_code == 0, schema_result.output
    schema = json.loads(schema_result.stdout)
    criterion = schema["properties"]["criteria"]["items"]["properties"]
    assert criterion["operator"]["enum"] == ["AT_LEAST", "AT_MOST"]

    example_result = runner.invoke(
        app,
        ["workload", "acceptance", "example", "--path", str(project), "--json"],
    )
    assert example_result.exit_code == 0, example_result.output
    example = json.loads(example_result.stdout)
    assert example["workload_profile_hash"] == workload.profile_hash
    assert example["criteria"][0]["evidence_rule"]["kind"] == "EXPLICIT_INTERVAL"


def test_machine_json_is_ascii_safe_for_non_ascii_user_text(tmp_path: Path) -> None:
    project = tmp_path / "unicode-project"
    result = runner.invoke(
        app,
        [
            "project",
            "init",
            str(project),
            "--name",
            "모델—테스트",
            "--origin",
            "ZERO",
            "--edition",
            "ACADEMY",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert all(ord(character) < 128 for character in result.stdout)
    assert json.loads(result.stdout)["ok"] is True



def test_lab_contract_schemas_and_examples_are_cli_discoverable() -> None:
    adapter_schema_result = runner.invoke(app, ["lab", "adapters", "schema", "--json"])
    assert adapter_schema_result.exit_code == 0, adapter_schema_result.output
    adapter_schema = json.loads(adapter_schema_result.stdout)
    assert adapter_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "adapter_id" in adapter_schema["required"]
    assert "EVALUATOR" in adapter_schema["properties"]["kinds"]["items"]["enum"]

    adapter_example_result = runner.invoke(app, ["lab", "adapters", "example", "--json"])
    assert adapter_example_result.exit_code == 0, adapter_example_result.output
    adapter_example = json.loads(adapter_example_result.stdout)
    assert adapter_example["schema_version"] == 1
    assert adapter_example["data_boundary"] == "CONTROLLED_PRIVATE"

    slurm_schema_result = runner.invoke(app, ["lab", "slurm", "schema", "--json"])
    assert slurm_schema_result.exit_code == 0, slurm_schema_result.output
    slurm_schema = json.loads(slurm_schema_result.stdout)
    assert "slurm" in slurm_schema["required"]
    assert slurm_schema["properties"]["slurm"]["properties"]["shared_filesystem"] == {
        "const": True
    }

    slurm_example_result = runner.invoke(app, ["lab", "slurm", "example", "--json"])
    assert slurm_example_result.exit_code == 0, slurm_example_result.output
    slurm_example = json.loads(slurm_example_result.stdout)
    assert slurm_example["kinds"] == ["TRAINER", "CLUSTER_EXECUTOR"]
    assert "{request_json}" in slurm_example["slurm"]["train_worker_argv"]

    rl_schema_result = runner.invoke(app, ["plan", "rl-schema", "--json"])
    assert rl_schema_result.exit_code == 0, rl_schema_result.output
    rl_schema = json.loads(rl_schema_result.stdout)
    assert rl_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "bounded verifiable-choice REINFORCE" in rl_schema["description"]
    assert "not an open-ended RLHF" in rl_schema["description"]
    assert "rl" in rl_schema["required"]
    assert "environment" in rl_schema["properties"]["rl"]["required"]

    rl_example_result = runner.invoke(app, ["plan", "rl-example", "--json"])
    assert rl_example_result.exit_code == 0, rl_example_result.output
    rl_example = json.loads(rl_example_result.stdout)
    assert rl_example["rl"]["algorithm_id"] == "reinforce"
    assert rl_example["rl"]["environment"]["kind"] == "VERIFIABLE_MULTIPLE_CHOICE"
    assert rl_example["rl"]["reward"]["kind"] == "EXACT_CORRECT_CHOICE"


def test_plan_create_defaults_to_single_execution_permission() -> None:
    default = inspect.signature(plan_create).parameters["permission"].default
    assert default == "EXECUTE_SINGLE"

    result = runner.invoke(app, ["plan", "create", "--help"])
    assert result.exit_code == 0, result.output
    assert "EXECUTE_SINGLE" in result.output



def test_root_help_describes_core_lifecycle_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command, phrase in (
        ("status", "growth stats."),
        ("interventions", "versioned intervention surfaces"),
        ("paths", "factual training-path availability"),
        ("calibrate", "pinned training plan"),
        ("run", "pinned permissions and budgets."),
        ("candidates", "Candidate models and lifecycle status."),
        ("promote", "evidence gates"),
        ("reject", "audit reason."),
        ("history", "decision"),
    ):
        assert command in result.output
        assert phrase in result.output
