import json
from pathlib import Path

import pytest

import frontierwright.service as service_module
from frontierwright.data import DatasetRole
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.evaluations import REFERENCE_LM_PACK, EvaluationReceipt, RawMeasurement
from frontierwright.registry import Registry
from frontierwright.service import (
    add_local_dataset,
    compare_candidate_evaluation,
    get_evaluation_packs,
    get_stats_view,
    import_local_model,
    run_evaluation_pack,
)


def make_reference_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "frontierwright_byte_causal_lm",
                "frontierwright_reference_backend": (
                    "frontierwright-reference-pytorch-v1"
                ),
                "preset": "zero-8m",
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        '{"type":"frontierwright-byte-level","vocab_size":256}',
        encoding="utf-8",
    )
    (root / "pytorch_model.bin").write_bytes(b"reference-model-weights")
    return root


def make_eval_data(root: Path, text: str = "held out evaluation bytes\n") -> Path:
    root.mkdir(parents=True)
    (root / "eval.txt").write_text(text * 20, encoding="utf-8")
    return root


def setup_project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project"
    import_local_model(
        project,
        make_reference_model(tmp_path / "model"),
        origin=ModelOrigin.IMPORTED_LOCAL,
        project_name="Reference",
    )
    add_local_dataset(
        project,
        make_eval_data(tmp_path / "eval"),
        name="Held out",
        role=DatasetRole.PRETRAIN,
    )
    dataset = Registry(project).read().datasets[0]
    return project, str(dataset["dataset_id"])


def fake_evaluator_result(
    argv_template: tuple[str, ...],
    *,
    environment_overrides: dict[str, str],
    request_path: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    del argv_template, environment_overrides, timeout_seconds
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["operation"] == "evaluate"
    return {
        "schema_version": 1,
        "ok": True,
        "operation": "evaluate",
        "metrics": {
            "backend_id": "frontierwright-reference-pytorch-v1",
            "preset": "zero-8m",
            "device": "cpu",
            "cross_entropy_nats_per_token": 4.25,
            "perplexity": 70.10541234668786,
            "tokens_evaluated": 512,
            "windows_evaluated": 4,
            "batches_evaluated": 2,
            "elapsed_seconds": 0.5,
            "tokens_per_second": 1024.0,
            "parameter_count": 7_521_280,
            "python_version": "3.12.10",
            "torch_version": "2.14.0+cpu",
        },
    }


def test_builtin_reference_evaluation_pack_is_discoverable() -> None:
    packs = get_evaluation_packs()
    assert packs == [REFERENCE_LM_PACK.to_dict()]
    assert packs[0]["metrics"] == [
        "cross_entropy_nats_per_token",
        "perplexity",
    ]


def test_reference_evaluation_generates_raw_receipt_without_capability_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, dataset_id = setup_project(tmp_path)
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_evaluator_result(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", counted)

    view = run_evaluation_pack(
        project,
        pack_id=REFERENCE_LM_PACK.pack_id,
        dataset_id=dataset_id,
        python_executable="fixture-python",
        device="cpu",
        batch_size=2,
        max_batches=2,
    )

    assert calls == 1
    assert view.replayed is False
    assert view.pack_id == REFERENCE_LM_PACK.pack_id
    assert view.model_id is not None
    assert view.receipt_id is not None
    assert [item["metric"] for item in view.measurements] == [
        "cross_entropy_nats_per_token",
        "perplexity",
    ]
    assert all(item["higher_is_better"] is False for item in view.measurements)

    registry = Registry(project)
    stored = registry.get_evaluation_receipt(view.receipt_id)
    assert stored is not None
    assert stored["model_id"] == view.model_id
    assert stored["conditions"]["dataset_id"] == dataset_id

    stats = get_stats_view(project)
    assert stats.measured is False
    assert stats.profile_id is None

    receipt_file = (
        registry.state_dir
        / "evaluations"
        / "receipts"
        / f"{view.receipt_id}.json"
    )
    assert receipt_file.is_file()

    generated_events = [
        event
        for event in registry.read().history
        if event["kind"] == "EVALUATION_RECEIPT_GENERATED"
    ]
    assert len(generated_events) == 1
    assert generated_events[0]["details"]["receipt_id"] == view.receipt_id


def test_reference_evaluation_replays_identical_receipt_without_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, dataset_id = setup_project(tmp_path)
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_evaluator_result(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", counted)

    first = run_evaluation_pack(
        project,
        pack_id=REFERENCE_LM_PACK.pack_id,
        dataset_id=dataset_id,
        python_executable="fixture-python",
        device="cpu",
        batch_size=2,
        max_batches=2,
    )
    second = run_evaluation_pack(
        project,
        pack_id=REFERENCE_LM_PACK.pack_id,
        dataset_id=dataset_id,
        python_executable="different-python-does-not-change-evidence",
        device="cpu",
        batch_size=2,
        max_batches=2,
    )

    assert calls == 1
    assert first.receipt_id == second.receipt_id
    assert first.receipt_sha256 == second.receipt_sha256
    assert second.replayed is True


def test_evaluation_config_change_creates_distinct_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, dataset_id = setup_project(tmp_path)
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_evaluator_result(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", counted)

    first = run_evaluation_pack(
        project,
        pack_id=REFERENCE_LM_PACK.pack_id,
        dataset_id=dataset_id,
        python_executable="fixture-python",
        max_batches=2,
    )
    second = run_evaluation_pack(
        project,
        pack_id=REFERENCE_LM_PACK.pack_id,
        dataset_id=dataset_id,
        python_executable="fixture-python",
        max_batches=3,
    )

    assert calls == 2
    assert first.receipt_id != second.receipt_id


def test_reference_evaluation_rejects_dataset_drift_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, dataset_id = setup_project(tmp_path)
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        return fake_evaluator_result(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", should_not_run)
    (tmp_path / "eval" / "eval.txt").write_text(
        "changed after registration\n" * 20,
        encoding="utf-8",
    )

    with pytest.raises(FrontierwrightError, match="content changed"):
        run_evaluation_pack(
            project,
            pack_id=REFERENCE_LM_PACK.pack_id,
            dataset_id=dataset_id,
            python_executable="fixture-python",
        )

    assert called is False


def test_unknown_evaluation_pack_is_rejected(tmp_path: Path) -> None:
    project, dataset_id = setup_project(tmp_path)
    with pytest.raises(FrontierwrightError, match="not executable"):
        run_evaluation_pack(
            project,
            pack_id="frontierwright.eval.unknown",
            dataset_id=dataset_id,
            python_executable="fixture-python",
        )


def test_invalid_evaluation_provenance_is_rejected_before_idempotent_replay(
    tmp_path: Path,
) -> None:
    project, _ = setup_project(tmp_path)
    registry = Registry(project)
    champion = registry.read().champion
    assert champion is not None
    receipt = EvaluationReceipt(
        receipt_id="receipt-provenance",
        model_id=champion.model.model_id,
        model_fingerprint=champion.model.fingerprint,
        evaluator_id="fixture",
        evaluator_version="1",
        conditions={},
        measurements=(
            RawMeasurement(
                task_id="fixture",
                task_version="1",
                metric="loss",
                value=1.0,
                higher_is_better=False,
            ),
        ),
    )

    registry.store_evaluation_receipt(receipt, provenance="GENERATED")
    with pytest.raises(FrontierwrightError, match="provenance"):
        registry.store_evaluation_receipt(receipt, provenance="INVALID")


def test_raw_candidate_compare_respects_metric_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, dataset_id = setup_project(tmp_path)
    registry = Registry(project)
    state = registry.read()
    assert state.champion is not None
    candidate = ModelState(
        model_id="model-eval-candidate",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="candidate-eval-checkpoint",
        fingerprint="sha256:candidate-eval",
        parent_model_id=state.champion.model.model_id,
        stats=(),
        trainable=True,
    )
    registry.register_candidate(candidate)

    def model_sensitive_evaluator(
        argv_template: tuple[str, ...],
        *,
        environment_overrides: dict[str, str],
        request_path: Path,
        timeout_seconds: float,
    ) -> dict[str, object]:
        del argv_template, environment_overrides, timeout_seconds
        request = json.loads(request_path.read_text(encoding="utf-8"))
        is_candidate = request["model_source_path"] == "candidate-eval-checkpoint"
        cross_entropy = 3.0 if is_candidate else 4.0
        perplexity = 20.085536923187668 if is_candidate else 54.598150033144236
        return {
            "schema_version": 1,
            "ok": True,
            "operation": "evaluate",
            "metrics": {
                "backend_id": "frontierwright-reference-pytorch-v1",
                "preset": "zero-8m",
                "device": "cpu",
                "cross_entropy_nats_per_token": cross_entropy,
                "perplexity": perplexity,
                "tokens_evaluated": 512,
                "windows_evaluated": 4,
                "batches_evaluated": 2,
                "elapsed_seconds": 0.5,
                "tokens_per_second": 1024.0,
                "parameter_count": 7_521_280,
                "python_version": "3.12.10",
                "torch_version": "2.14.0+cpu",
            },
        }

    monkeypatch.setattr(
        service_module,
        "run_structured_command",
        model_sensitive_evaluator,
    )

    view = compare_candidate_evaluation(
        project,
        candidate_model_id=candidate.model_id,
        pack_id=REFERENCE_LM_PACK.pack_id,
        dataset_id=dataset_id,
        python_executable="fixture-python",
        device="cpu",
        batch_size=2,
        max_batches=2,
    )

    assert view.comparable is True
    assert view.candidate_status == "PENDING"
    assert view.champion_receipt_id != view.candidate_receipt_id
    assert [item["metric"] for item in view.measurements] == [
        "cross_entropy_nats_per_token",
        "perplexity",
    ]
    for item in view.measurements:
        assert item["higher_is_better"] is False
        assert item["raw_delta"] < 0
        assert item["improvement_delta"] > 0


def test_deterministic_evaluation_replay_rejects_poisoned_receipt_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, dataset_id = setup_project(tmp_path)
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_evaluator_result(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_structured_command", counted)
    first = run_evaluation_pack(
        project,
        pack_id=REFERENCE_LM_PACK.pack_id,
        dataset_id=dataset_id,
        python_executable="fixture-python",
        device="cpu",
        max_batches=2,
    )
    assert first.receipt_id is not None
    registry = Registry(project)
    stored = registry.get_evaluation_receipt(first.receipt_id)
    assert stored is not None
    poisoned_conditions = dict(stored["conditions"])
    poisoned_conditions["dataset_fingerprint"] = "sha256:poisoned"
    with registry.connect(write=True) as connection:
        connection.execute(
            "UPDATE evaluation_receipts SET conditions_json = ? WHERE receipt_id = ?",
            (
                json.dumps(poisoned_conditions, sort_keys=True),
                first.receipt_id,
            ),
        )

    with pytest.raises(FrontierwrightError, match="occupied by different"):
        run_evaluation_pack(
            project,
            pack_id=REFERENCE_LM_PACK.pack_id,
            dataset_id=dataset_id,
            python_executable="fixture-python",
            device="cpu",
            max_batches=2,
        )

    assert calls == 1
