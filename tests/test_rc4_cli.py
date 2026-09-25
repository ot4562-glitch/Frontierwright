from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app
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
