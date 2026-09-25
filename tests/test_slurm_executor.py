import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.errors import FrontierwrightError
from frontierwright.execution import load_command_backend_spec
from frontierwright.slurm_executor import (
    inspect_slurm_availability,
    load_slurm_executor_profile,
    run_slurm_request,
    slurm_backend_spec_payload,
)

runner = CliRunner()


def _write_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(0o700)
    return path


def _fake_cluster(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    worker_body = r"""import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.request.read_text(encoding="utf-8"))
if request["operation"] == "calibrate":
    payload = {
        "schema_version": 1,
        "ok": True,
        "feasible": True,
        "representative_steps": 2,
        "step_time_seconds": 0.25,
        "tokens_per_second": 1234.0,
        "peak_vram_bytes": 4096,
        "peak_ram_bytes": 8192,
        "projected_storage_bytes": 16384,
        "projected_wall_seconds": 12.0,
    }
else:
    output_root = Path(request["output_root"])
    model = output_root / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "weights.bin").write_bytes(b"frontierwright-slurm-test")
    payload = {
        "schema_version": 1,
        "ok": True,
        "output_model_path": str(model.resolve()),
        "metrics": {"loss": 0.5, "worker": "fake-slurm"},
    }
print(json.dumps(payload, sort_keys=True))
"""
    sbatch_body = r"""import subprocess
import sys
from pathlib import Path

argv = sys.argv[1:]

def value(flag):
    return argv[argv.index(flag) + 1]

stdout_pattern = value("--output")
stderr_pattern = value("--error")
script = Path(argv[-1])
job_id = "4242"
print(job_id, flush=True)
result = subprocess.run([str(script)], capture_output=True, check=False)
Path(stdout_pattern.replace("%j", job_id)).write_bytes(result.stdout)
Path(stderr_pattern.replace("%j", job_id)).write_bytes(result.stderr)
raise SystemExit(result.returncode)
"""
    scancel_body = r"""from pathlib import Path
import sys

Path(__file__).with_suffix(".calls").write_text(
    " ".join(sys.argv[1:]),
    encoding="utf-8",
)
"""
    sacct_body = "print('4242|COMPLETED|0:0|2|cpu=8,gres/gpu=2|1G|2G|10')\n"

    worker = _write_script(tmp_path / "worker.py", worker_body)
    sbatch = _write_script(tmp_path / "sbatch.py", sbatch_body)
    scancel = _write_script(tmp_path / "scancel.py", scancel_body)
    sacct = _write_script(tmp_path / "sacct.py", sacct_body)
    return worker, sbatch, scancel, sacct

def _profile_payload(
    worker: Path,
    sbatch: Path,
    scancel: Path,
    sacct: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "adapter_id": "lab.slurm.test",
        "adapter_version": "1",
        "display_name": "Private Slurm Test",
        "kinds": ["TRAINER", "CLUSTER_EXECUTOR"],
        "data_boundary": "CONTROLLED_PRIVATE",
        "network_scope": "PRIVATE_ONLY",
        "capabilities": ["slurm", "full-sft"],
        "slurm": {
            "schema_version": 1,
            "supported_paths": ["FULL_SFT"],
            "calibrate_worker_argv": [
                sys.executable,
                str(worker),
                "--request",
                "{request_json}",
            ],
            "train_worker_argv": [
                sys.executable,
                str(worker),
                "--request",
                "{request_json}",
            ],
            "sbatch_argv": [sys.executable, str(sbatch)],
            "scancel_argv": [sys.executable, str(scancel)],
            "sacct_argv": [sys.executable, str(sacct)],
            "partition": "private",
            "nodes": 1,
            "gpus_per_node": 2,
            "cpus_per_task": 8,
            "memory_mb": 32768,
            "time_limit_minutes": 15,
            "max_queue_wait_seconds": 30,
            "shared_filesystem": True,
        },
    }


def _write_profile(tmp_path: Path) -> Path:
    worker, sbatch, scancel, sacct = _fake_cluster(tmp_path)
    path = tmp_path / "slurm-profile.json"
    path.write_text(
        json.dumps(_profile_payload(worker, sbatch, scancel, sacct), sort_keys=True),
        encoding="utf-8",
    )
    return path


def _request(
    profile_ref: str, *, operation: str, output_root: Path | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "operation": operation,
        "plan_id": "plan-slurm-test",
        "path_id": "FULL_SFT",
        "backend_data_boundary": "CONTROLLED_PRIVATE",
        "backend_adapter_ref": profile_ref,
        "budgets": {"max_wall_seconds": 120.0},
    }
    if output_root is not None:
        payload["output_root"] = str(output_root.resolve())
    return payload


@pytest.mark.skipif(os.name == "nt", reason="Slurm bridge requires a POSIX submit host")
def test_slurm_bridge_executes_structured_worker_and_records_accounting(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    profile = load_slurm_executor_profile(profile_path)
    readiness = inspect_slurm_availability(profile)
    assert readiness["ready"] is True

    calibration_request = tmp_path / "calibrate.json"
    calibration_request.write_text(
        json.dumps(_request(profile.adapter_ref, operation="calibrate")),
        encoding="utf-8",
    )
    calibration = run_slurm_request(
        profile_path,
        expected_profile_hash=profile.sha256,
        request_path=calibration_request,
    )
    assert calibration["feasible"] is True
    assert calibration["representative_steps"] == 2
    scheduler = calibration["scheduler"]
    assert scheduler["kind"] == "SLURM"
    assert scheduler["job_id"] == "4242"
    assert scheduler["accounting"]["job"]["state"] == "COMPLETED"
    assert scheduler["accounting"]["job"]["elapsed_raw_seconds"] == 2

    output_root = tmp_path / "train-output"
    train_request = tmp_path / "train.json"
    train_request.write_text(
        json.dumps(_request(profile.adapter_ref, operation="train", output_root=output_root)),
        encoding="utf-8",
    )
    trained = run_slurm_request(
        profile_path,
        expected_profile_hash=profile.sha256,
        request_path=train_request,
    )
    model_path = Path(str(trained["output_model_path"]))
    assert model_path.is_dir()
    assert (model_path / "weights.bin").read_bytes() == b"frontierwright-slurm-test"
    assert trained["metrics"]["scheduler"]["job_id"] == "4242"


def test_slurm_profile_and_generated_backend_spec_are_hash_pinned(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    profile = load_slurm_executor_profile(profile_path)
    payload = slurm_backend_spec_payload(profile_path, profile, python_executable=sys.executable)
    spec_path = tmp_path / "backend.json"
    spec_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    spec = load_command_backend_spec(spec_path)

    assert spec.data_boundary.value == "CONTROLLED_PRIVATE"
    assert spec.provider_adapter_ref == profile.adapter_ref
    assert profile.sha256 in spec.calibrate_argv
    assert profile.sha256 in spec.train_argv
    assert spec.sha256.startswith("sha256:")

    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["slurm"]["nodes"] = 2
    profile_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    changed = load_slurm_executor_profile(profile_path)
    assert changed.sha256 != profile.sha256


@pytest.mark.skipif(os.name == "nt", reason="Slurm bridge requires a POSIX submit host")
def test_slurm_profile_drift_is_blocked_before_submission(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    profile = load_slurm_executor_profile(profile_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(_request(profile.adapter_ref, operation="calibrate")),
        encoding="utf-8",
    )
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["slurm"]["cpus_per_task"] = 16
    profile_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")

    with pytest.raises(FrontierwrightError) as exc_info:
        run_slurm_request(
            profile_path,
            expected_profile_hash=profile.sha256,
            request_path=request_path,
        )
    assert exc_info.value.code == "SLURM_PROFILE_DRIFT"


def test_slurm_profile_rejects_secrets_and_unsafe_boundary(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["slurm"]["credentials"] = {"token": "do-not-store"}
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(FrontierwrightError) as secret_exc:
        load_slurm_executor_profile(profile_path)
    assert secret_exc.value.code == "SLURM_PROFILE_SECRET_FORBIDDEN"

    profile_path = _write_profile(tmp_path / "boundary")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["data_boundary"] = "EXTERNAL"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(FrontierwrightError) as boundary_exc:
        load_slurm_executor_profile(profile_path)
    assert boundary_exc.value.code == "SLURM_PROFILE_INVALID"


def test_slurm_profile_requires_shared_filesystem_v1(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["slurm"]["shared_filesystem"] = False
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(FrontierwrightError) as exc_info:
        load_slurm_executor_profile(profile_path)
    assert exc_info.value.code == "SLURM_PROFILE_INVALID"
    assert "shared filesystem" in str(exc_info.value).lower()


def test_slurm_cli_inspect_and_backend_spec_are_machine_readable(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    profile = load_slurm_executor_profile(profile_path)

    inspected = runner.invoke(
        app,
        [
            "lab",
            "slurm",
            "inspect",
            str(profile_path),
            "--json",
            "--non-interactive",
        ],
    )
    assert inspected.exit_code == 0, inspected.output
    inspected_payload = json.loads(inspected.stdout)
    assert inspected_payload["profile_hash"] == profile.sha256
    assert inspected_payload["adapter_ref"] == profile.adapter_ref
    assert inspected_payload["maturity"] == "BOUNDED_V1"
    assert inspected_payload["availability"]["profile_hash"] == profile.sha256

    output = tmp_path / "generated-backend.json"
    generated = runner.invoke(
        app,
        [
            "lab",
            "slurm",
            "backend-spec",
            str(profile_path),
            "--output",
            str(output),
            "--python-executable",
            sys.executable,
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert generated.exit_code == 0, generated.output
    generated_payload = json.loads(generated.stdout)
    assert generated_payload["profile_hash"] == profile.sha256
    assert Path(generated_payload["output"]) == output.resolve()
    spec = load_command_backend_spec(output)
    assert spec.provider_adapter_ref == profile.adapter_ref
    assert profile.sha256 in spec.calibrate_argv
    assert profile.sha256 in spec.train_argv
