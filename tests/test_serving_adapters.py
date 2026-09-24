from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.registry import Registry
from frontierwright.service import (
    compare_candidate,
    get_resource_view,
    import_serving_resource_evidence,
)

runner = CliRunner()


def setup_pair(root: Path) -> tuple[Registry, ModelState, ModelState]:
    registry = Registry(root)
    registry.initialize("NOVA", ModelOrigin.IMPORTED_LOCAL)
    state = registry.read()
    champion = ModelState(
        model_id="champion-vllm",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="champion-checkpoint",
        fingerprint="sha256:" + "a" * 64,
    )
    candidate = ModelState(
        model_id="candidate-vllm",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="candidate-checkpoint",
        fingerprint="sha256:" + "b" * 64,
        parent_model_id=champion.model_id,
    )
    registry.register_candidate(champion)
    registry.promote_candidate(champion.model_id)
    registry.register_candidate(candidate)
    return registry, champion, candidate


def write_vllm_result(
    path: Path,
    *,
    source_model: str,
    e2e_ms: float,
    ttft_ms: float,
    tpot_ms: float,
    itl_ms: float,
    output_throughput: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "backend": "vllm",
                "model_id": source_model,
                "tokenizer_id": source_model,
                "num_prompts": 80,
                "request_rate": "inf",
                "burstiness": 1.0,
                "max_concurrency": 1,
                "dataset_name": "custom",
                "completed": 80,
                "failed": 0,
                "request_throughput": 2.0,
                "output_throughput": output_throughput,
                "total_token_throughput": output_throughput + 10.0,
                "median_ttft_ms": ttft_ms,
                "median_tpot_ms": tpot_ms,
                "median_itl_ms": itl_ms,
                "median_e2el_ms": e2e_ms,
                "date": "2026-09-24T00:00:00Z",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def import_result(project: Path, model_id: str, result: Path) -> dict[str, object]:
    cli = runner.invoke(
        app,
        [
            "operate",
            "import-vllm-benchmark",
            str(result),
            "--path",
            str(project),
            "--model",
            model_id,
            "--vllm-version",
            "0.12.0@abcdef",
            "--execution-boundary",
            "LOCAL_MACHINE",
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert cli.exit_code == 0, cli.output
    return json.loads(cli.stdout)


def test_vllm_import_preserves_latency_semantics_and_condition_identity(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, champion, candidate = setup_pair(project)
    champion_result = write_vllm_result(
        tmp_path / "champion.json",
        source_model="org/stock-model",
        e2e_ms=500.0,
        ttft_ms=120.0,
        tpot_ms=20.0,
        itl_ms=21.0,
        output_throughput=40.0,
    )
    candidate_result = write_vllm_result(
        tmp_path / "candidate.json",
        source_model="local/custom-descendant",
        e2e_ms=430.0,
        ttft_ms=90.0,
        tpot_ms=18.0,
        itl_ms=19.0,
        output_throughput=35.0,
    )

    champion_payload = import_result(project, champion.model_id, champion_result)
    candidate_payload = import_result(project, candidate.model_id, candidate_result)
    assert champion_payload["profile_condition_hash"] == candidate_payload["profile_condition_hash"]
    champion_metrics = champion_payload["metrics"]
    assert champion_metrics["latency_seconds_p50"] == 0.5
    assert champion_metrics["ttft_seconds_p50"] == 0.12
    assert champion_metrics["tpot_seconds_p50"] == 0.02
    assert champion_metrics["tokens_per_second_p50"] is None
    assert champion_metrics["peak_vram_bytes"] is None

    comparison = compare_candidate(project, candidate.model_id)
    assert comparison.pareto["inference_profile_comparable"] is True
    assert comparison.pareto["relation"] == "TRADEOFF"
    metrics = {item["key"]: item for item in comparison.pareto["metrics"]}
    assert metrics["serving.latency_p50"]["relation"] == "BETTER"
    assert metrics["serving.ttft_p50"]["relation"] == "BETTER"
    assert metrics["serving.tpot_p50"]["relation"] == "BETTER"
    assert metrics["serving.output_throughput_aggregate"]["relation"] == "WORSE"
    assert metrics["resource.peak_vram"]["relation"] == "UNKNOWN"

    # Re-importing identical evidence is idempotent.
    again = import_result(project, champion.model_id, champion_result)
    assert again["source_sha256"] == champion_payload["source_sha256"]
    imported_events = [
        item
        for item in registry.read().history
        if item["kind"] == "INFERENCE_PROFILE_MEASURED"
        and item["details"].get("provenance") == "IMPORTED_VLLM_BENCH_SERVE"
    ]
    assert len(imported_events) == 2


def test_vllm_client_result_does_not_invent_model_memory_headroom(tmp_path: Path) -> None:
    project = tmp_path / "project"
    registry, champion, _ = setup_pair(project)
    result = write_vllm_result(
        tmp_path / "result.json",
        source_model="org/model",
        e2e_ms=400.0,
        ttft_ms=100.0,
        tpot_ms=15.0,
        itl_ms=16.0,
        output_throughput=50.0,
    )
    import_result(project, champion.model_id, result)

    # ResourceView needs a system resource snapshot to exist; the imported serving event
    # still must never claim server-process VRAM/RSS from client-only evidence.
    state = registry.read()
    model_fit = next(
        item["details"]["metrics"]
        for item in reversed(state.history)
        if item["kind"] == "INFERENCE_PROFILE_MEASURED"
    )
    assert model_fit["peak_vram_bytes"] is None
    assert model_fit["max_sampled_process_rss_bytes"] is None
    assert get_resource_view(project).available is False


def write_resource_manifest(
    path: Path,
    *,
    model_fingerprint: str,
    profile_condition_hash: str,
    peak_vram_bytes: int,
    process_rss_bytes: int,
    runtime_version: str = "0.12.0@abcdef",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_fingerprint": model_fingerprint,
                "runtime_id": "vllm",
                "runtime_version": runtime_version,
                "profile_condition_hash": profile_condition_hash,
                "measurement_scope": "server_process_resource_profile",
                "execution_boundary": "LOCAL_MACHINE",
                "metrics": {
                    "peak_vram_bytes": peak_vram_bytes,
                    "max_sampled_process_rss_bytes": process_rss_bytes,
                    "cuda_memory_total_bytes": 12 * 1024**3,
                    "cuda_memory_free_min_sampled_bytes": 2 * 1024**3,
                    "cuda_memory_free_after_profile_bytes": 3 * 1024**3,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def import_resource_manifest(project: Path, model_id: str, manifest: Path) -> dict[str, object]:
    result = runner.invoke(
        app,
        [
            "operate",
            "import-serving-resources",
            str(manifest),
            "--path",
            str(project),
            "--model",
            model_id,
            "--json",
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_serving_resource_receipts_compose_only_under_exact_runtime_conditions(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    registry, champion, candidate = setup_pair(project)
    champion_result = write_vllm_result(
        tmp_path / "champion.json",
        source_model="org/stock-model",
        e2e_ms=500.0,
        ttft_ms=120.0,
        tpot_ms=20.0,
        itl_ms=21.0,
        output_throughput=40.0,
    )
    candidate_result = write_vllm_result(
        tmp_path / "candidate.json",
        source_model="local/custom-descendant",
        e2e_ms=430.0,
        ttft_ms=90.0,
        tpot_ms=18.0,
        itl_ms=19.0,
        output_throughput=35.0,
    )
    champion_client = import_result(project, champion.model_id, champion_result)
    candidate_client = import_result(project, candidate.model_id, candidate_result)
    condition_hash = str(champion_client["profile_condition_hash"])
    assert candidate_client["profile_condition_hash"] == condition_hash

    champion_resource = write_resource_manifest(
        tmp_path / "champion-resource.json",
        model_fingerprint=champion.fingerprint,
        profile_condition_hash=condition_hash,
        peak_vram_bytes=8 * 1024**3,
        process_rss_bytes=10 * 1024**3,
    )
    candidate_resource = write_resource_manifest(
        tmp_path / "candidate-resource.json",
        model_fingerprint=candidate.fingerprint,
        profile_condition_hash=condition_hash,
        peak_vram_bytes=6 * 1024**3,
        process_rss_bytes=8 * 1024**3,
    )
    champion_memory = import_resource_manifest(project, champion.model_id, champion_resource)
    candidate_memory = import_resource_manifest(project, candidate.model_id, candidate_resource)
    assert champion_memory["runtime_id"] == "vllm"
    assert candidate_memory["profile_condition_hash"] == condition_hash

    comparison = compare_candidate(project, candidate.model_id)
    assert comparison.pareto["inference_profile_comparable"] is True
    metrics = {item["key"]: item for item in comparison.pareto["metrics"]}
    assert metrics["serving.latency_p50"]["relation"] == "BETTER"
    assert metrics["serving.ttft_p50"]["relation"] == "BETTER"
    assert metrics["resource.peak_vram"]["relation"] == "BETTER"
    assert metrics["resource.process_rss"]["relation"] == "BETTER"

    # The merged Champion profile records both independent evidence sources.
    state = registry.read()
    from frontierwright.service import _latest_model_fit_from_state

    fit = _latest_model_fit_from_state(state, champion.model_id)
    assert fit["latency_seconds_p50"] == 0.5
    assert fit["peak_vram_bytes"] == 8 * 1024**3
    sources = fit["evidence_sources"]
    assert isinstance(sources, list)
    assert len(sources) == 2


def test_serving_resource_condition_mismatch_does_not_cross_fill_client_metrics(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _, champion, candidate = setup_pair(project)
    result = write_vllm_result(
        tmp_path / "candidate.json",
        source_model="local/custom-descendant",
        e2e_ms=430.0,
        ttft_ms=90.0,
        tpot_ms=18.0,
        itl_ms=19.0,
        output_throughput=35.0,
    )
    client = import_result(project, candidate.model_id, result)
    wrong_condition = "sha256:" + "c" * 64
    assert client["profile_condition_hash"] != wrong_condition
    manifest = write_resource_manifest(
        tmp_path / "candidate-resource.json",
        model_fingerprint=candidate.fingerprint,
        profile_condition_hash=wrong_condition,
        peak_vram_bytes=6 * 1024**3,
        process_rss_bytes=8 * 1024**3,
    )
    import_resource_manifest(project, candidate.model_id, manifest)

    from frontierwright.service import _latest_model_fit_from_state

    fit = _latest_model_fit_from_state(Registry(project).read(), candidate.model_id)
    assert fit["profile_condition_hash"] == wrong_condition
    assert fit["peak_vram_bytes"] == 6 * 1024**3
    assert fit["latency_seconds_p50"] is None
    assert len(fit["evidence_sources"]) == 1


def test_serving_resource_manifest_rejects_wrong_model_fingerprint(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _, champion, _ = setup_pair(project)
    manifest = write_resource_manifest(
        tmp_path / "wrong-model.json",
        model_fingerprint="sha256:" + "f" * 64,
        profile_condition_hash="sha256:" + "d" * 64,
        peak_vram_bytes=1,
        process_rss_bytes=1,
    )
    with pytest.raises(FrontierwrightError) as exc_info:
        import_serving_resource_evidence(
            project, manifest_path=manifest, model_id=champion.model_id
        )
    assert exc_info.value.code == "SERVING_RESOURCE_MODEL_MISMATCH"
