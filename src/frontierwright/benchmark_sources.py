"""Curated external benchmark/source catalog for Frontierwright RC4.

The catalog is metadata only. Frontierwright does not vendor third-party benchmark
datasets or code here, and a catalog entry is not evidence that a benchmark is
installed, licensed for redistribution, or runnable on the current machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class StatAxisV2(StrEnum):
    KNOWLEDGE = "KNOWLEDGE"
    REASONING = "REASONING"
    MATH = "MATH"
    CODING = "CODING"
    INSTRUCTION = "INSTRUCTION"
    LANGUAGE = "LANGUAGE"
    CONTEXT = "CONTEXT"


class BenchmarkKind(StrEnum):
    MODEL = "MODEL"
    SYSTEM = "SYSTEM"
    FRAMEWORK = "FRAMEWORK"


class BenchmarkLifecycle(StrEnum):
    ACTIVE = "ACTIVE"
    STABLE = "STABLE"
    SOURCE_FRAMEWORK = "SOURCE_FRAMEWORK"
    WATCH = "WATCH"


@dataclass(frozen=True)
class BenchmarkSource:
    source_id: str
    title: str
    kind: BenchmarkKind
    lifecycle: BenchmarkLifecycle
    axes: tuple[StatAxisV2, ...]
    upstream_url: str
    runner_hint: str
    contamination_note: str
    redistribution: str = "VERIFY_UPSTREAM_BEFORE_VENDORING"
    integration_status: str = "CATALOG_ONLY"
    license_status: str = "VERIFY_UPSTREAM"
    cost_hint: str = "VARIES_BY_MODEL_AND_SELECTED_SUITE"

    @property
    def next_action(self) -> str:
        if self.kind is BenchmarkKind.FRAMEWORK:
            return (
                "Use this framework to import/run a pinned task; "
                "Frontierwright does not score the framework."
            )
        if self.kind is BenchmarkKind.SYSTEM:
            return "Connect a system evaluator and keep results separate from pure model Stat v2."
        return "Pin an exact upstream version/task and import compatible evaluation evidence."

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["lifecycle"] = self.lifecycle.value
        payload["axes"] = [axis.value for axis in self.axes]
        payload["next_action"] = self.next_action
        return payload


BENCHMARK_SOURCES: tuple[BenchmarkSource, ...] = (
    BenchmarkSource(
        source_id="livebench",
        title="LiveBench",
        kind=BenchmarkKind.MODEL,
        lifecycle=BenchmarkLifecycle.ACTIVE,
        axes=(
            StatAxisV2.REASONING,
            StatAxisV2.CODING,
            StatAxisV2.MATH,
            StatAxisV2.LANGUAGE,
            StatAxisV2.INSTRUCTION,
        ),
        upstream_url="https://livebench.ai/",
        runner_hint="official LiveBench runner/adapter",
        contamination_note="Prefer pinned dated releases; upstream refreshes tasks over time.",
    ),
    BenchmarkSource(
        source_id="mmlu-pro",
        title="MMLU-Pro",
        kind=BenchmarkKind.MODEL,
        lifecycle=BenchmarkLifecycle.STABLE,
        axes=(StatAxisV2.KNOWLEDGE,),
        upstream_url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        runner_hint="lm-eval/OpenCompass/official implementation",
        contamination_note="Use as broad knowledge evidence, not a sole frontier discriminator.",
    ),
    BenchmarkSource(
        source_id="ifeval",
        title="IFEval",
        kind=BenchmarkKind.MODEL,
        lifecycle=BenchmarkLifecycle.STABLE,
        axes=(StatAxisV2.INSTRUCTION,),
        upstream_url=(
            "https://github.com/google-research/google-research/"
            "tree/master/instruction_following_eval"
        ),
        runner_hint="lm-eval or upstream evaluator",
        contamination_note="Pin evaluator/version and strict metric identity.",
    ),
    BenchmarkSource(
        source_id="livecodebench",
        title="LiveCodeBench",
        kind=BenchmarkKind.MODEL,
        lifecycle=BenchmarkLifecycle.ACTIVE,
        axes=(StatAxisV2.CODING,),
        upstream_url="https://github.com/LiveCodeBench/LiveCodeBench",
        runner_hint="official LiveCodeBench runner",
        contamination_note=(
            "Prefer release windows newer than the model/training corpus when possible."
        ),
    ),
    BenchmarkSource(
        source_id="scicode",
        title="SciCode",
        kind=BenchmarkKind.MODEL,
        lifecycle=BenchmarkLifecycle.STABLE,
        axes=(StatAxisV2.CODING, StatAxisV2.REASONING, StatAxisV2.KNOWLEDGE),
        upstream_url="https://github.com/scicode-bench/SciCode",
        runner_hint="Inspect/OpenCompass/official implementation",
        contamination_note="Scientific-coding evidence should remain separately inspectable.",
    ),
    BenchmarkSource(
        source_id="longbench-v2",
        title="LongBench v2",
        kind=BenchmarkKind.MODEL,
        lifecycle=BenchmarkLifecycle.STABLE,
        axes=(StatAxisV2.CONTEXT, StatAxisV2.REASONING),
        upstream_url="https://longbench2.github.io/",
        runner_hint="official dataset/runner or compatible adapter",
        contamination_note="Record exact context length and task subset with every receipt.",
    ),
    BenchmarkSource(
        source_id="terminal-bench",
        title="Terminal-Bench",
        kind=BenchmarkKind.SYSTEM,
        lifecycle=BenchmarkLifecycle.WATCH,
        axes=(StatAxisV2.CODING,),
        upstream_url="https://github.com/harbor-framework/terminal-bench",
        runner_hint="system/agent evaluation; never merge directly into pure model stats",
        contamination_note="Measures model+agent+tools+sandbox system capability.",
    ),
    BenchmarkSource(
        source_id="lm-eval",
        title="EleutherAI lm-evaluation-harness",
        kind=BenchmarkKind.FRAMEWORK,
        lifecycle=BenchmarkLifecycle.SOURCE_FRAMEWORK,
        axes=(),
        upstream_url="https://github.com/EleutherAI/lm-evaluation-harness",
        runner_hint="evaluation framework and task/plugin source",
        contamination_note="Framework source; task identities remain independently versioned.",
    ),
    BenchmarkSource(
        source_id="inspect-ai",
        title="Inspect AI",
        kind=BenchmarkKind.FRAMEWORK,
        lifecycle=BenchmarkLifecycle.SOURCE_FRAMEWORK,
        axes=(),
        upstream_url="https://inspect.aisi.org.uk/",
        runner_hint="frontier evaluation framework and pre-built eval source",
        contamination_note=(
            "Framework source; do not treat aggregate framework coverage as a score."
        ),
    ),
    BenchmarkSource(
        source_id="opencompass",
        title="OpenCompass",
        kind=BenchmarkKind.FRAMEWORK,
        lifecycle=BenchmarkLifecycle.SOURCE_FRAMEWORK,
        axes=(),
        upstream_url="https://doc.opencompass.org.cn/",
        runner_hint="benchmark framework/catalog source",
        contamination_note="Framework source; pin each selected dataset/config independently.",
    ),
)


def benchmark_source_payloads() -> list[dict[str, object]]:
    return [source.to_payload() for source in BENCHMARK_SOURCES]
