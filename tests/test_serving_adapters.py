from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from frontierwright.cli import app
from frontierwright.domain import ModelOrigin, ModelState
from frontierwright.registry import Registry
from frontierwright.service import compare_candidate, get_resource_view

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
        fingerprint="sha256:champion-vllm",
    )
    candidate = ModelState(
        model_id="candidate-vllm",
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.IMPORTED_LOCAL,
        checkpoint="candidate-checkpoint",
        fingerprint="sha256:candidate-vllm",
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
