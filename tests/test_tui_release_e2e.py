import asyncio
import json
import sys
from pathlib import Path

import pytest
from textual.widgets import Input

import frontierwright.service as service_module
from frontierwright.capability_v1 import (
    CAPABILITY_V1_BUNDLE_ID,
    CAPABILITY_V1_BUNDLE_VERSION,
    CAPABILITY_V1_SCORING,
    CAPABILITY_V1_TASKS,
    capability_v1_bundle_hash,
    capability_v1_task_counts,
)
from frontierwright.service import get_history_view, get_status
from frontierwright.tui import FrontierwrightApp
from frontierwright.tui.app import (
    ActionCenterScreen,
    CandidateScreen,
    ConfirmScreen,
    WorkflowFormScreen,
)


def _write_training_backend(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    script = root / "backend.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if request["operation"] == "calibrate":
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "feasible": True,
        "representative_steps": 1,
        "step_time_seconds": 0.01,
        "tokens_per_second": 1000.0,
        "peak_vram_bytes": 0,
        "peak_ram_bytes": 1024,
        "projected_storage_bytes": 128,
        "projected_wall_seconds": 1.0,
        "gpu_count": 1
    }))
elif request["operation"] == "train":
    output = Path(request["output_root"]) / "model"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps({
            "model_type": "frontierwright_byte_causal_lm",
            "preset": "zero-8m",
            "vocab_size": 276
        }),
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        json.dumps({"type": "frontierwright-byte-level", "vocab_size": 276}),
        encoding="utf-8",
    )
    (output / "model.safetensors").write_bytes(b"tui-e2e-candidate")
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "output_model_path": str(output),
        "metrics": {"steps": 1, "loss": 0.5}
    }))
else:
    raise SystemExit(2)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec = root / "backend.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": "tui-release-e2e-backend",
                "data_boundary": "LOCAL_MACHINE",
                "supported_paths": ["FROM_SCRATCH_PRETRAINING"],
                "calibrate_argv": [sys.executable, str(script), "{request_json}"],
                "train_argv": [sys.executable, str(script), "{request_json}"],
                "environment": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return spec


def _fake_reference_command(
    argv_template: tuple[str, ...],
    *,
    environment_overrides: dict[str, str],
    request_path: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    del argv_template, environment_overrides, timeout_seconds
    request = json.loads(request_path.read_text(encoding="utf-8"))
    operation = request.get("operation")

    if operation == "birth":
        output = Path(str(request["output_root"])) / "model"
        output.mkdir(parents=True, exist_ok=False)
        tokenizer_path = request.get("tokenizer_path")
        if isinstance(tokenizer_path, str):
            tokenizer_payload = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
            vocab_size = int(tokenizer_payload["vocab_size"])
        else:
            tokenizer_payload = {
                "type": "frontierwright-byte-level",
                "vocab_size": 256,
            }
            vocab_size = 256
        (output / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "frontierwright_byte_causal_lm",
                    "preset": request["preset"],
                    "parameter_count": 8_000_000,
                    "vocab_size": vocab_size,
                }
            ),
            encoding="utf-8",
        )
        (output / "tokenizer.json").write_text(
            json.dumps(tokenizer_payload, sort_keys=True),
            encoding="utf-8",
        )
        (output / "model.safetensors").write_bytes(b"tui-e2e-root")
        return {
            "schema_version": 1,
            "ok": True,
            "operation": "birth",
            "output_model_path": str(output),
            "metrics": {
                "backend_id": "frontierwright-reference-pytorch-v1",
                "preset": request["preset"],
                "seed": request["seed"],
                "parameter_count": 8_000_000,
                "vocab_size": vocab_size,
                "tokenizer_fingerprint": request.get("tokenizer_fingerprint"),
                "python_version": "fixture",
                "torch_version": "fixture",
                "device": "cpu",
                "trained_steps": 0,
            },
        }

    if operation == "capability_v1":
        task_counts = capability_v1_task_counts()
        axes = []
        item_results = []
        for axis in ("general", "reasoning", "math", "coding"):
            total = task_counts[axis]
            correct = total // 2
            axes.append(
                {
                    "axis": axis,
                    "task_id": f"{CAPABILITY_V1_BUNDLE_ID}.{axis}",
                    "task_version": CAPABILITY_V1_BUNDLE_VERSION,
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                    "mean_correct_margin_nats": 0.1,
                }
            )
            axis_tasks = [task for task in CAPABILITY_V1_TASKS if task.axis.value.lower() == axis]
            for index, task in enumerate(axis_tasks):
                item_results.append(
                    {
                        "item_id": task.item_id,
                        "axis": axis,
                        "correct": index < correct,
                        "correct_margin_nats": 0.1,
                    }
                )
        return {
            "schema_version": 1,
            "ok": True,
            "operation": "capability_v1",
            "metrics": {
                "backend_id": "frontierwright-reference-pytorch-v1",
                "bundle_id": CAPABILITY_V1_BUNDLE_ID,
                "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
                "bundle_hash": capability_v1_bundle_hash(),
                "scoring": CAPABILITY_V1_SCORING,
                "task_counts": task_counts,
                "axes": axes,
                "item_results": item_results,
                "preset": "zero-8m",
                "device": "cpu",
                "parameter_count": 8_000_000,
                "tokenizer_fingerprint": None,
                "elapsed_seconds": 0.01,
                "python_version": "fixture",
                "torch_version": "fixture",
            },
        }

    raise AssertionError(f"unexpected structured operation: {operation!r}")


async def _choose_action(app: FrontierwrightApp, pilot, action_id: str) -> None:
    items = app._action_items()
    ids = [item.action_id for item in items]
    assert action_id in ids, (action_id, ids)
    index = ids.index(action_id)
    await pilot.press("a")
    await pilot.pause()
    assert isinstance(app.screen, ActionCenterScreen)
    for _ in range(index):
        await pilot.press("j")
    await pilot.press("enter")
    await pilot.pause()


async def _submit_form(
    app: FrontierwrightApp,
    pilot,
    values: dict[str, str] | None = None,
) -> None:
    assert isinstance(app.screen, WorkflowFormScreen)
    screen = app.screen
    values = values or {}
    for index, field in enumerate(screen.fields):
        if field.key in values:
            screen.query_one(f"#form-field-{index}", Input).value = values[field.key]
    await pilot.press("ctrl+s")
    await pilot.pause()


@pytest.mark.asyncio
async def test_keyboard_tui_primary_lifecycle_reaches_measured_promoted_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.txt").write_text(
        "Frontierwright TUI release lifecycle.\n" * 64,
        encoding="utf-8",
    )
    backend_spec = _write_training_backend(tmp_path / "backend")

    monkeypatch.setattr(service_module, "run_structured_command", _fake_reference_command)
    monkeypatch.setattr(
        FrontierwrightApp,
        "_write_reference_backend_spec",
        lambda self, python_executable: backend_spec,
    )

    app = FrontierwrightApp(
        view=get_status(project),
        root=project,
        language="en",
    )

    async with app.run_test(size=(150, 60)) as pilot:
        await _choose_action(app, pilot, "initialize_project")
        await _submit_form(app, pilot)
        assert app.view.initialized is True

        await _choose_action(app, pilot, "detect_resources")
        assert app.resources.available is True

        await _choose_action(app, pilot, "add_dataset")
        await _submit_form(
            app,
            pilot,
            {
                "source": str(corpus),
                "name": "TUI pretrain",
                "role": "PRETRAIN",
                "classification": "PRIVATE",
            },
        )
        assert len(app.data.datasets) == 1

        await _choose_action(app, pilot, "build_intent")
        await _submit_form(app, pilot)
        assert app.build.configured is True

        await _choose_action(app, pilot, "birth_tokenizer")
        await _submit_form(app, pilot, {"vocab_size": "276"})
        assert app.view.champion_model_id is None

        await _choose_action(app, pilot, "birth_zero")
        await _submit_form(app, pilot, {"python": "fixture-python"})
        root_model_id = app.view.champion_model_id
        assert isinstance(root_model_id, str)

        await _choose_action(app, pilot, "capability_v1")
        await _submit_form(app, pilot, {"python": "fixture-python", "device": "cpu"})
        assert app.view.measurement_state == "MEASURED"

        await _choose_action(app, pilot, "build_targets")
        await _submit_form(app, pilot, {"target_general": "100"})
        assert app.build.targets == {"general": 100}

        await _choose_action(app, pilot, "reference_plan")
        await _submit_form(app, pilot, {"python": sys.executable, "config_json": "{}"})
        assert any(item.get("availability") == "READY" for item in app.paths.paths)

        await _choose_action(app, pilot, "train_ready")
        await _submit_form(app, pilot, {"python": sys.executable})
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        await pilot.pause()
        assert app.last_run_id is not None

        for _ in range(40):
            await asyncio.sleep(0.05)
            await _choose_action(app, pilot, "reconcile_run")
            if app.candidates.candidates:
                break
        assert len(app.candidates.candidates) == 1
        candidate_id = str(app.candidates.candidates[0]["model_id"])
        assert candidate_id != root_model_id

        await _choose_action(app, pilot, "measure_candidate_capability")
        await _submit_form(app, pilot, {"python": "fixture-python", "device": "cpu"})
        assert isinstance(app.screen, CandidateScreen)
        assert app.screen.compare.candidate_model_id == candidate_id
        assert app.screen.compare.promotion_eligible is True

        await pilot.press("p")
        await pilot.pause()

        assert app.view.champion_model_id == candidate_id
        assert app.view.measurement_state == "MEASURED"

    status = get_status(project)
    assert status.champion_model_id == candidate_id
    assert status.measurement_state == "MEASURED"
    kinds = {event["kind"] for event in get_history_view(project).events}
    assert {
        "MODEL_BORN",
        "TOKENIZER_TRAINED",
        "RUN_COMPLETED",
        "EVALUATION_RECEIPT_GENERATED",
        "CANDIDATE_PROMOTED",
    }.issubset(kinds)


@pytest.mark.asyncio
async def test_keyboard_tui_studio_import_immediately_exposes_real_character_sheet(
    tmp_path: Path,
) -> None:
    project = tmp_path / "studio-project"
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "fixture-causal-lm", "hidden_size": 64}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text(
        json.dumps({"type": "fixture", "vocab_size": 256}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"studio-import-weights")

    app = FrontierwrightApp(
        view=get_status(project),
        root=project,
        language="en",
    )

    async with app.run_test(size=(140, 45)) as pilot:
        await _choose_action(app, pilot, "initialize_project")
        await _submit_form(
            app,
            pilot,
            {
                "name": "STUDIO-NOVA",
                "origin": "IMPORTED_LOCAL",
                "edition": "STUDIO",
                "language": "en",
            },
        )
        assert app.view.edition_profile == "STUDIO"
        assert app.view.origin == "IMPORTED_LOCAL"
        assert app.view.champion_model_id is None

        await _choose_action(app, pilot, "import_model")
        await _submit_form(app, pilot, {"source": str(model)})

        assert isinstance(app.view.champion_model_id, str)
        assert app.view.model_format == "HUGGINGFACE"
        assert app.view.trainable is True
        assert isinstance(app.view.model_fingerprint, str)
        assert app.view.model_fingerprint.startswith("sha256:")
        assert app.view.measurement_state == "NOT_READY"
        assert app.view.history_confidence == "UNKNOWN"

        sheet = app._character_text()
        assert "STUDIO-NOVA" in sheet
        assert "Frontierwright Studio" in sheet
        assert "YOUR MODEL / YOUR MACHINE / YOUR WORKLOAD" in sheet
        assert "Champion: " in sheet
        assert app.view.champion_model_id[:20] in sheet
        assert "Measured machine fit: UNKNOWN" in sheet
        assert "Workload fit: NOT DEFINED" in sheet
        assert "General    ?" in sheet
        assert "Reasoning  ?" in sheet
        assert "Math       ?" in sheet
        assert "Coding     ?" in sheet
