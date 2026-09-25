# Frontierwright RC4 — Stat v2, Benchmark Sources, and Studio Growth

Date: 2026-09-26  
Status: RC4 FOUNDATION IMPLEMENTED / CANDIDATE HARDENING

## Product decision lock

RC4 adopts a permanent project-origin stat baseline.

- The first model that establishes a Frontierwright project identity is the permanent origin.
- Every player-facing model stat is displayed relative to that origin.
- The origin is exactly 0.0 on every Stat v2 axis.
- Promoting a Candidate to Champion never re-zeros stats.
- A later Champion therefore remains directly comparable with the model that started the project.
- Raw benchmark measurements remain stored beside the relative display stat; the relative stat never replaces scientific evidence.

This is intentionally compatible with Frontierwright's existing identity/level metaphor: model development is cumulative growth, not a sequence of unrelated local baselines.

## Stat v2 model axes

The RC4 model-level axes are:

1. Knowledge
2. Reasoning
3. Math
4. Coding
5. Instruction
6. Language
7. Context

These axes are deliberately broader than any one benchmark suite. External benchmark tasks are mapped into these stable Frontierwright categories only through explicit, versioned mappings.

System/agent/runtime measurements are not silently folded into pure model stats. Examples kept separate include:

- agentic/task-environment success;
- tool-use/system success;
- latency;
- throughput;
- TTFT/TPOT/ITL;
- memory/VRAM;
- artifact storage.

A model can therefore gain a better agent scaffold or faster runtime without Frontierwright falsely claiming that its model weights gained Coding or Reasoning capability.

## Capability v1 compatibility

Capability v1 remains frozen and readable.

It is reclassified conceptually as a small deterministic smoke/regression microbenchmark:

- 64 total four-choice items;
- 16 each for General, Reasoning, Math, Coding;
- frozen bundle and frozen presentation scale;
- useful for cheap local sanity checks;
- not the target RC4 Lab benchmark engine.

RC4 must not silently change Capability v1 tasks or its frozen score mapping. New measurement semantics belong to Stat v2 / Capability v2.

## Capability v2 direction

Capability v2 is a multi-benchmark evidence layer rather than one small handcrafted quiz.

Requirements:

- substantially larger evidence surface than Capability v1;
- exact benchmark/version/runner identity;
- raw metrics retained;
- sample counts retained;
- confidence or uncertainty evidence retained when supported;
- no arbitrary conversion from unrelated benchmark units into one score;
- benchmark-specific evidence can remain auxiliary rather than being forced into a composite;
- model-level and system-level benchmark evidence remain distinguishable.

The default player-facing normalized delta for a [0, 1] accuracy/pass-rate metric is percentage-point change from the permanent origin:

    display_delta = direction_normalized(current - origin) * 100

This is a presentation rule, not a replacement for the raw result.

## UNCERTAIN is a first-class state

Stat v2 uses:

- BETTER
- WORSE
- SAME
- UNCERTAIN
- UNMEASURED

A nonzero point estimate alone is not sufficient to declare BETTER/WORSE when uncertainty overlaps zero.

Additional measurement follows the estimand:

### Deterministic capability/evaluation

Do not rerun the same deterministic item and pretend it is a new sample.

When more evidence is available, open additional unused held-out items or a compatible additional benchmark slice.

Possible progression:

    64 -> 128 -> 256 -> 512 -> larger Lab budget

### Runtime/resource measurements

Runtime measurements may legitimately gain precision from repetition, but should use comparable conditions and interleaved Champion/Candidate ordering where possible to reduce time-varying host effects.

Examples:

    A B B A A B B A

The exact runtime/environment identity remains part of the evidence.

### Consent

Additional measurement consumes compute. Frontierwright should recommend it, estimate the extra work when possible, and ask for user consent before starting a materially larger measurement.

Studio copy example:

    Coding is still uncertain.
    Frontierwright recommends measuring 128 more held-out items.
    Continue?

Lab can expose the full protocol and budget before execution.

## Studio product direction

Studio is for hobbyists and existing local-model users who want:

- visible growth;
- real performance improvement;
- customization for their own use;
- a satisfying model-development loop.

Studio may lean toward a game-like presentation, but rewards must be evidence-backed.

Good rewards:

- permanent stat growth from measured improvement;
- personal bests;
- level progression grounded in measured model development;
- build traits/specialization;
- clear trade-offs;
- a visible lineage of Champions and Candidates.

Do not award fake capability XP merely because training completed. Training success is not model improvement.

## Lab product direction

Lab uses the same evidence core as Studio but exposes a professional research surface.

Lab prioritizes:

- benchmark source/version identity;
- reproducibility;
- sample counts;
- uncertainty;
- raw metrics;
- evaluator/grader identity;
- hardware/runtime conditions;
- long-running measurements when useful;
- benchmark lifecycle/freshness;
- contamination and saturation awareness;
- exact experiment budgets;
- frontier-oriented external evaluators and infrastructure adapters.

Studio can later inherit mature Lab capabilities behind a simpler experience.

## User optimization goal contract

The intended optimization language is user-directed rather than Frontierwright inventing a universal objective.

Core rule types:

1. IMPROVE — increase this stat/metric.
2. PROTECT — this stat/metric must not regress.
3. TOLERANCE — regression is allowed only within a declared amount or percentage.
4. ABSOLUTE_REQUIREMENT — meet an absolute capability/runtime/resource threshold.
5. HARD_CONSTRAINT — satisfy a categorical rule such as privacy/data boundary.

Typical Studio setup:

    IMPROVE     Coding
    IMPROVE     Instruction
    PROTECT     Reasoning
    TOLERANCE   Latency <= +5%
    TOLERANCE   Memory <= +10%
    HARD        Private data stays local

RC4 should converge Build goals and Workload Acceptance so the user does not have to describe the same success condition twice.

## Acceptance UX

RC3 proved that the acceptance implementation existed but the public creation contract was too difficult to discover.

RC4 requires:

- a machine-readable public JSON Schema;
- a complete minimal example bound to the active workload profile;
- exact invalid-field diagnostics;
- allowed enum values in errors;
- a direct `workload next` action when acceptance is missing.

Studio should eventually synthesize the acceptance contract from the user's Build/goal UI. Lab keeps full explicit schema control.

## External benchmark/source registry

RC4 introduces a source catalog rather than copying benchmark datasets into Frontierwright.

Initial source pool:

- LiveBench — refreshed objective model evaluation; useful across reasoning, coding, math, language and instruction-following.
- MMLU-Pro — broad knowledge source.
- IFEval — instruction-following source.
- LiveCodeBench — contamination-aware coding source.
- SciCode — realistic scientific coding/reasoning source.
- LongBench v2 — long-context and long-document reasoning source.
- Terminal-Bench — system/agent evaluation only; never silently merged into pure model stats.
- lm-evaluation-harness — evaluation framework/task source.
- Inspect AI — frontier evaluation framework/source.
- OpenCompass — broad benchmark framework/catalog source.

Source metadata does not mean Frontierwright vendors or redistributes upstream code/data.

Before integrating an external benchmark Frontierwright must record or verify:

- source URL;
- exact release/version/commit where possible;
- runner/evaluator identity;
- license for code and dataset separately where relevant;
- redistribution permission;
- contamination/freshness notes;
- metric/grader identity;
- task subset;
- model/system boundary;
- required context/tools/runtime;
- exact receipt provenance.

## Benchmark lifecycle

External benchmark suites move.

Frontierwright must not overwrite old evidence when a source changes. New benchmark releases receive new identities.

If a Stat v2 mapping migrates to a new source generation and continuity matters, the permanent origin model and a relevant current model should be re-evaluated under the new suite rather than stitching incomparable raw scores together.

## External research anchors

The RC4 direction was cross-checked against current public evaluation ecosystems on 2026-09-26:

- LiveBench: https://livebench.ai/
- Stanford HELM capabilities: https://crfm.stanford.edu/2025/03/20/helm-capabilities.html
- LiveCodeBench: https://github.com/LiveCodeBench/LiveCodeBench
- SciCode: https://github.com/scicode-bench/SciCode
- LongBench v2: https://longbench2.github.io/
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- Inspect AI: https://inspect.aisi.org.uk/
- OpenCompass: https://doc.opencompass.org.cn/
- Google IFEval: https://github.com/google-research/google-research/tree/master/instruction_following_eval

These references guide source selection and measurement design. They are not copied into Frontierwright by this document.

## RC4 implementation slices

### Slice A — implemented in RC4

- Stat v2 stable axes.
- Origin-relative stat primitive.
- Permanent `origin_model_id` persisted in registry schema v26 and preserved across Champion changes.
- Best-effort v25 -> v26 origin backfill from the unique root model or earliest root model event.
- UNCERTAIN relation.
- Conservative repeated-runtime observed-range uncertainty for latency/throughput Pareto evidence.
- External benchmark source metadata registry with model/system/framework separation.
- Public acceptance JSON Schema.
- Public acceptance example bound to the active workload hash.
- Field-specific acceptance enum/numeric diagnostics.
- Acceptance missing -> explicit `workload next` action.
- Plan surfaces separate `calibration_ready`, `new_attempt_allowed`, `remaining_runs`, and `replay_available`.
- A consumed `max_runs` allowance can no longer advertise a new training attempt as `READY`.
- Machine JSON escapes non-ASCII text so Windows CP949 hosts cannot crash on valid Unicode payloads.

### Slice B — next implementation

- multi-source executable Stat v2 receipts and frozen mappings;
- capability additional-held-out-item measurement;
- runtime interleaved remeasurement and user-approved larger sampling budgets;
- consent/budget surface for additional measurement;
- user growth-goal persistence integrated with existing Build/Workload authority;
- Build -> Acceptance contract synthesis;
- statistical runtime intervals beyond the conservative observed-range guard.

### Slice C — Lab depth

- executable benchmark adapters for selected upstream sources;
- benchmark lifecycle/version pinning;
- rigorous Lab suites and source selection/mixing;
- real external-system/agent benchmark separation;
- reproducibility and long-run measurement reports.

## Non-goals for this RC4 foundation

- Do not claim that the source catalog means all listed benchmarks are already runnable.
- Do not vendor third-party benchmark data without license review.
- Do not replace raw metrics with a synthetic intelligence number.
- Do not call a noisy point-estimate difference an improvement.
- Do not reset stats when Champion changes.
- Do not turn training completion into fake progression.
