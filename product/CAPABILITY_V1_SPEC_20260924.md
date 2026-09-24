# Frontierwright Capability v1

Date: 2026-09-24
Status: FROZEN v1 MEASUREMENT CONTRACT
Scope: Official Frontierwright v1 player-facing capability stats for supported reference-model evaluation.

## Purpose

Frontierwright Capability v1 turns reproducible local evaluation evidence into the four core visible stats:

- General
- Reasoning
- Math
- Coding

It is a fixed product microbenchmark, not a claim of general intelligence, a moving leaderboard, or a substitute for domain-specific evaluation.

Raw evidence remains stored and inspectable separately from the displayed stats.

## Frozen task bundle

Bundle ID:

`frontierwright.capability.v1`

Bundle version:

`1`

The bundle contains 64 four-choice tasks:

- General: 16
- Reasoning: 16
- Math: 16
- Coding: 16

The canonical task definitions live in:

`src/frontierwright/capability_v1.py`

Frozen bundle SHA-256:

`52ae4fc38dca0bda9e7d8d2845d985e38afeb9530317d9f716f88cabd25261c8`

Changing any task, choice, answer, axis assignment, or canonical scoring identity requires a new bundle version and new frozen hash.

## Scoring

Scoring ID:

`mean-conditional-logprob-v1`

For every item, Frontierwright:

1. tokenizes the frozen prompt with the exact tokenizer bound to the model;
2. tokenizes each of the four answer continuations with that same tokenizer;
3. scores each answer by its mean conditional token log-probability;
4. selects the answer with the highest score;
5. uses the lowest choice index only as a deterministic exact-tie breaker.

The evaluator records, for each axis:

- accuracy;
- correct count;
- total count;
- mean correct-answer margin over the strongest incorrect answer, in nats.

The margin is raw evidence. It does not currently contribute to the player-facing stat.

## Frozen scale

Scale ID:

`frontierwright.capability.v1`

Scale version:

`1`

Frozen scale SHA-256:

`34ca954fc3dd6fae6f9e322517518e628491a17700ac1aa4d74c3d0b16073ce5`

Each axis uses its 16-task accuracy as the only v1 scale input.

The frozen mapping is:

```text
0% accuracy   ->   0
25% accuracy  ->  50   (four-choice chance reference)
50% accuracy  -> 100   (Frontierwright v1 reference anchor)
100% accuracy -> 200
```

Equivalently:

`display_stat = 200 * accuracy`

The value 100 is a reference anchor, not a maximum. Frontierwright capability stats are not capped at 100 by product semantics.

## Evidence identity

A generated Capability v1 receipt is pinned to:

- exact model ID;
- exact model fingerprint;
- bundle ID/version/hash;
- scale ID/version/hash;
- scoring ID;
- evaluator ID/version;
- device request;
- exact per-axis raw results;
- tokenizer fingerprint;
- model preset/parameter count where available;
- Python and PyTorch runtime versions.

An identical model/config request replays the durable receipt rather than rerunning the benchmark.

If model bytes change, the model fingerprint changes and prior evidence cannot silently apply to the new model.

## Product surfaces

Machine/human CLI:

```text
frontierwright eval capability-v1
```

The command supports stable JSON/non-interactive output.

The human TUI Action Center exposes:

`Measure Capability v1`

After successful measurement the active character sheet uses the resulting frozen-scale stats and measured models become eligible for numeric Build targets/floors.

## Interpretation limits

Capability v1 deliberately stays small, deterministic, transparent, and local so that the v1 product can provide real stats without inventing measurements.

It should not be interpreted as:

- a comprehensive benchmark of all model capability;
- a secure anti-gaming leaderboard;
- an industry-standard intelligence score;
- a replacement for raw held-out loss, task-specific benchmarks, safety evaluation, robustness evaluation, or user-domain evaluation.

Future Stat Packs may add richer measurements without silently changing the frozen Capability v1 definition.
