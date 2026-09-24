import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import frontierwright.service as service_module
from frontierwright.cli import app
from frontierwright.domain import ModelFormat, ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.registry import Registry
from frontierwright.service import (
    compare_candidate,
    merge_reference_models,
    preflight_merge_reference_models,
    promote_candidate,
)


def make_model_dir(root: Path, label: str) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "frontierwright_byte_causal_lm",
                "frontierwright_reference_backend": "frontierwright-reference-pytorch-v1",
                "preset": "zero-8m",
                "parameter_count": 8_000_000,
                "label": label,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(f"weights:{label}".encode())
    return root


def setup_project(tmp_path: Path) -> tuple[Path, Registry, ModelState, ModelState]:
    project = tmp_path / "project"
    registry = Registry(project)
    registry.initialize("NOVA", ModelOrigin.ZERO)
    state = registry.read()

    primary_path = make_model_dir(tmp_path / "primary", "primary")
    other_path = make_model_dir(tmp_path / "other", "other")
    primary = ModelState(
        model_id="model-primary",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint=str(primary_path),
        fingerprint="sha256:primary-fixture",
        model_format=ModelFormat.HUGGINGFACE,
        trainable=True,
    )
    other = ModelState(
        model_id="model-other",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint=str(other_path),
        fingerprint="sha256:other-fixture",
        model_format=ModelFormat.HUGGINGFACE,
        trainable=True,
    )
    registry.register_candidate(primary)
    registry.promote_candidate(primary.model_id)
    registry.register_candidate(other)
    return project, registry, primary, other


def fake_merge_backend(
    argv_template: tuple[str, ...],
    *,
    environment_overrides: dict[str, str],
    request_path: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    del argv_template, environment_overrides, timeout_seconds
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = Path(str(request["output_root"])) / "model"
    output.mkdir(parents=True, exist_ok=False)

    (output / "config.json").write_text(
        json.dumps(
            {
                "model_type": "frontierwright_byte_causal_lm",
                "frontierwright_reference_backend": "frontierwright-reference-pytorch-v1",
                "preset": "zero-8m",
                "parameter_count": 8_000_000,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    parents = request["parents"]
    (output / "model.safetensors").write_bytes(
        json.dumps(parents, sort_keys=True).encode()
    )
    transform = {
        "schema_version": 1,
        "intervention_id": "frontierwright.evolve.linear-merge",
        "intervention_version": "1",
        "method": "linear_weight_merge",
        "primary_weight": 1.0 - float(request["other_weight"]),
        "other_weight": float(request["other_weight"]),
        "parents": parents,
        "backend_id": "frontierwright-reference-pytorch-v1",
        "preset": "zero-8m",
        "parameter_count": 8_000_000,
        "merged_tensor_count": 4,
    }
    (output / "frontierwright-transform.json").write_text(
        json.dumps(transform, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "ok": True,
        "operation": "merge",
        "output_model_path": str(output),
        "metrics": transform,
    }


def test_merge_creates_pending_candidate_and_multi_parent_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, registry, primary, other = setup_project(tmp_path)
    monkeypatch.setattr(service_module, "run_structured_command", fake_merge_backend)

    view = merge_reference_models(
        project,
        other_model_id=other.model_id,
        other_weight=0.25,
        python_executable="fixture-python",
    )

    assert view.replayed is False
    assert view.primary_model_id == primary.model_id
    assert view.other_model_id == other.model_id
    assert view.primary_weight == 0.75
    assert view.other_weight == 0.25
    assert view.candidate_model_id is not None

    candidate = registry.get_candidate(view.candidate_model_id)
    assert candidate.status.value == "PENDING"
    assert candidate.model.parent_model_id == primary.model_id
    assert Path(candidate.model.checkpoint).is_relative_to(
        (registry.state_dir / "transforms").resolve()
    )

    lineage = registry.get_model_lineage(candidate.model.model_id)
    merge_edges = [item for item in lineage if item["relation"] == "MERGED_FROM"]
    assert [(item["parent_model_id"], item["ordinal"]) for item in merge_edges] == [
        (primary.model_id, 0),
        (other.model_id, 1),
    ]
    assert merge_edges[0]["details"]["weight"] == 0.75
    assert merge_edges[1]["details"]["weight"] == 0.25

    artifact = registry.get_model_artifact(candidate.model.model_id)
    assert artifact is not None
    files = {item["relative_path"] for item in artifact["files"]}
    assert "frontierwright-transform.json" in files


def test_identical_merge_replays_without_backend_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _, _, other = setup_project(tmp_path)
    calls = 0

    def counted_backend(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_merge_backend(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", counted_backend)

    first = merge_reference_models(
        project,
        other_model_id=other.model_id,
        other_weight=0.4,
        python_executable="fixture-python",
    )
    second = merge_reference_models(
        project,
        other_model_id=other.model_id,
        other_weight=0.4,
        python_executable="fixture-python",
    )

    assert calls == 1
    assert first.candidate_model_id == second.candidate_model_id
    assert first.replayed is False
    assert second.replayed is True




def test_merge_dry_run_is_side_effect_free_and_predicts_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, registry, _, other = setup_project(tmp_path)

    def must_not_execute(*args, **kwargs):
        del args, kwargs
        raise AssertionError("merge dry-run executed backend")

    monkeypatch.setattr(service_module, "run_structured_command", must_not_execute)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evolve", "merge", other.model_id,
            "--path", str(project),
            "--other-weight", "0.4",
            "--python", "fixture-python",
            "--dry-run", "--json", "--non-interactive", "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["action"] == "evolve-merge"
    assert payload["would_replay"] is False
    assert len(registry.read().candidates) == 2

    monkeypatch.setattr(service_module, "run_structured_command", fake_merge_backend)
    created = merge_reference_models(
        project,
        other_model_id=other.model_id,
        other_weight=0.4,
        python_executable="fixture-python",
    )
    replay = preflight_merge_reference_models(
        project,
        other_model_id=other.model_id,
        other_weight=0.4,
        python_executable="different-python",
    )
    assert replay.would_replay is True
    assert replay.details["candidate_model_id"] == created.candidate_model_id

def test_merge_artifact_tampering_blocks_compare_and_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, registry, _, other = setup_project(tmp_path)
    monkeypatch.setattr(service_module, "run_structured_command", fake_merge_backend)

    view = merge_reference_models(
        project,
        other_model_id=other.model_id,
        other_weight=0.5,
        python_executable="fixture-python",
    )
    assert view.candidate_model_id is not None
    candidate = registry.get_candidate(view.candidate_model_id)
    transform_path = Path(candidate.model.checkpoint) / "frontierwright-transform.json"
    transform_path.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(FrontierwrightError, match="artifact"):
        compare_candidate(project, candidate.model.model_id)

    with pytest.raises(FrontierwrightError, match="artifact"):
        promote_candidate(
            project,
            candidate.model.model_id,
            allow_unmeasured=True,
        )


def test_merge_rejects_same_parent_and_invalid_weight(tmp_path: Path) -> None:
    project, _, primary, other = setup_project(tmp_path)

    with pytest.raises(FrontierwrightError, match="strictly between 0 and 1"):
        merge_reference_models(
            project,
            other_model_id=other.model_id,
            other_weight=1.0,
            python_executable="fixture-python",
        )

    with pytest.raises(FrontierwrightError, match="two different model identities"):
        merge_reference_models(
            project,
            other_model_id=primary.model_id,
            other_weight=0.5,
            python_executable="fixture-python",
        )
