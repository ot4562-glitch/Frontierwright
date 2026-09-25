# Frontierwright

[![CI](https://github.com/ot4562-glitch/Frontierwright/actions/workflows/ci.yml/badge.svg)](https://github.com/ot4562-glitch/Frontierwright/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **Don't fit yourself to a model. Fit the model to you.**

Frontierwright is a local/private model-development environment for turning open or
user-controlled language models into models that fit a specific **workload, machine,
budget, and set of goals**.

A stock checkpoint is a starting point, not the finished product. Frontierwright
measures the model and the machine, applies real training or optimization
interventions, creates descendants, evaluates the trade-offs, and keeps exact lineage
so the model can keep evolving as the user's needs change.

**Fixed open models are starting points, not finished products.**

---

## Why Frontierwright?

Open models ship in discrete sizes and configurations. Real users do not.

Your machine may have spare VRAM but a latency limit. Your work may need Korean,
economics, code, long context, or private domain knowledge in a mixture no public
leaderboard was designed for. A larger stock model may be wasteful; a smaller,
specialized descendant may be better for *your* work.

Frontierwright changes the question from:

> Which downloadable model is best?

to:

> **Which reachable model gives me the most useful capability inside my actual
> resource and privacy constraints?**

The optimization target is user utility, not parameter count and not 100% hardware
utilization.

---

## The loop

```text
measure model + machine + workload
              ↓
        choose a real goal
              ↓
 plan / calibrate / dry-run
              ↓
 train · align · distill · merge · optimize
              ↓
           Candidate
              ↓
 comparable eval + resource evidence
              ↓
       promote or reject
              ↓
           Champion
              ↓
       use → observe → repeat
```

A successful training run is **not** automatically an improvement. Frontierwright
keeps the current Champion unchanged until a Candidate is measured and explicitly
accepted.

---

## One core, three different experiences

| Edition | Built for | Default experience |
| --- | --- | --- |
| **Frontierwright Studio** | People who already control an open/local model | Fit the model to **your machine and workload**: headroom, goals, specialist branches, compression, candidate trade-offs, continual improvement |
| **Frontierwright Academy** | People learning how models are actually built | **Understand by doing**: guided data → tokenizer → birth → training → evaluation, progressive disclosure, explanations tied to real state |
| **Frontierwright Lab** | Teams developing large private/internal models | **Controlled frontier development**: exact evidence, private infrastructure, experiment matrices, large-scale training/RL adapters, regression and budget gates |

The editions share the same model lineage and raw evidence. Switching editions never
rewrites history or invents different stats; it changes the **workflow, information
density, defaults, explanations, and policies**.

---

## Real evidence, not game stats

Frontierwright never awards fake XP.

A visible stat must trace back to a versioned evaluation receipt. A resource number
must come from detection, calibration, or runtime measurement. A Candidate is an
actual model artifact.

Capability v1 is deliberately a small frozen local microbenchmark, not a claim of
general intelligence. Frontierwright exposes the **sample count and 95% Wilson
uncertainty interval** with each axis so a 16-item estimate is not presented with
fake precision. New v2 evaluator receipts also retain same-item correctness/margin
evidence. Candidate comparison reports improvement/regression flips and an exact
two-sided McNemar test; this is diagnostic evidence, never an automatic promotion rule,
and a non-significant result is treated as inconclusive rather than proof of equivalence.

Resource evidence similarly distinguishes:

- **system availability now** — RAM, disk, and GPU VRAM reported free at detection;
- **model-specific headroom** — requires an actual model/runtime profile before it can
  be claimed.

Internally, missing evidence remains conservative. In the human UI it becomes an actionable
**MEASUREMENT NEEDED** state, and threshold-overlapping uncertainty becomes **MORE EVIDENCE
NEEDED**. Frontierwright should measure away uncertainty whenever a bounded compatible
measurement is available instead of leaving the user in a permanent unknown state.

---

## What a real trade-off looks like

A direct RC self-play with the tiny built-in reference model produced this comparison:

| Axis | Champion | Candidate | Delta |
| --- | ---: | ---: | ---: |
| General | 50.0 | 75.0 | +25.0 |
| Reasoning | 50.0 | 62.5 | +12.5 |
| Math | 50.0 | 37.5 | -12.5 |
| Coding | 50.0 | 50.0 | 0.0 |

The Build target was **General ≥ 60**, so the target was reached — but Math visibly
regressed. Promotion remained an explicit decision.

This is an integration example from Frontierwright's tiny local benchmark/model, **not
a claim that Frontierwright universally improves models by these amounts**. The point
is that gains, regressions, and resource trade-offs stay visible.

---

## What Frontierwright can do today

The current core supports real, reproducible lifecycle operations including:

- model import, fingerprinting, lineage, Candidate/Champion state;
- Academy tokenizer training and deterministic `zero-8m` / `zero-25m` Birth;
- pretraining, continued pretraining, full SFT, LoRA, NF4 QLoRA, and DPO;
- knowledge distillation, deterministic linear merge, and symmetric int8 optimization;
- frozen Capability v1 evaluation plus held-out LM evaluation;
- raw comparable Candidate-vs-Champion evidence, practical-margin decision claims, and promotion gates;
- versioned Workload Acceptance contracts with PASS / FAIL / INCONCLUSIVE / UNKNOWN internal decisions and actionable measurement states;
- local generation, inference profiling, portable export and verification;
- privacy-minimal real-use observations with atomic retry semantics and exact workload-revision cohorts;
- private-data boundaries, Lab adapter contracts, and a bounded controlled-private Slurm executor bridge;
- hard run/storage/time budgets, durable execution receipts, recovery and idempotency;
- keyboard-first TUI for humans and stable CLI/JSON surfaces for automation.

The long engineering inventory and known gaps live in
[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

---

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/ot4562-glitch/Frontierwright.git
cd Frontierwright

python -m venv .venv
python -m pip install -e ".[dev]"
```

For real reference-model training/evaluation, install the training extra in a Python
environment with a PyTorch build appropriate for your machine:

```bash
python -m pip install -e ".[train]"
```

Create an Academy project and launch the human interface:

```bash
frontierwright project init . --name NOVA --origin ZERO --edition ACADEMY
frontierwright play --path .
```

Or start from an existing trainable model in Studio:

```bash
frontierwright project init . --name MYMODEL --origin IMPORTED_LOCAL --edition STUDIO
frontierwright model import /path/to/model --path .
frontierwright resources detect --path .
frontierwright play --path .
```

Compare a stock/open baseline with any registered descendant even after the descendant
has already become Champion:

```bash
frontierwright workload compare-models STOCK_MODEL_ID CUSTOM_MODEL_ID --path .
```

This comparison is not tied to Candidate status. It reuses the same capability, workload,
serving/resource, artifact-size, paired-item, Pareto, and explicit user-utility evidence
that Frontierwright has actually measured; missing material dimensions block an unqualified
superiority claim and become measurement work to close.

Close the loop with privacy-minimal real-use evidence. These records are **not RL rewards** and do not store prompt/response content by default:

```bash
frontierwright observe record coding --outcome FAILURE --failure-category tool-selection --path .
frontierwright observe summary --path .
frontierwright workload next --path .
```

Repeated failures/corrections become evidence for the next bounded experiment. Frontierwright still requires them to be promoted into versioned evaluation/data/reward evidence before training or RL can claim to learn from them.

AI agents and automation should use the non-interactive CLI/JSON contract instead of
driving the TUI:

```text
--json
--non-interactive
--yes
stable exit codes
versioned schemas
idempotency for expensive actions
```

---

## User-fit development layer

The rc2 development line is already moving beyond stock-model selection:

- **Workload Profiles** pin task/language/context/latency/privacy requirements and hard
  capability floors as versioned evidence.
- **Measured model fit** links real inference receipts to the exact model instead of
  treating currently-free system memory as post-load headroom. Client latency/throughput
  and server VRAM/RSS receipts may be composed only when exact model fingerprint, runtime,
  execution boundary, and serving-condition hash match.
- **Evidence-driven next experiments** turn FAIL constraints into bounded interventions,
  missing measurements into concrete measurement work, and INCONCLUSIVE thresholds into
  additional compatible sampling. Frontierwright never predicts that a listed experiment
  will improve the model; success is defined by comparable evidence.
- **Pareto Candidate comparison** keeps capability, latency, throughput, VRAM/RAM and
  artifact-size gains/regressions visible side by side.
- **Explicit user utility** is optional: a user must provide both a weight and a
  normalization scale for every metric they want combined. Missing evidence makes the
  result INCOMPLETE; Frontierwright never invents cross-unit conversion or hides the
  underlying Pareto trade-off.
- **External evidence adapters** can import exact-version lm-evaluation-harness, LightEval,
  vLLM, and framework-neutral evaluation manifests without automatically turning arbitrary
  external scores into Frontierwright stats. Generic manifests must pin exact model
  fingerprint, evaluator/version, task/version/metric identity, value, metric direction,
  and may preserve sample count, standard error, and confidence intervals. Workload
  language/domain/task coverage becomes PASS only through an explicit binding to exact
  stored receipt identities; Frontierwright never infers coverage from benchmark names.
- **Real RL foundations** include a rollout/reward/policy-optimization contract and a
  reference verifier-driven policy-gradient path; DPO remains correctly labeled as
  preference optimization. Operational observations are explicitly rejected as direct RL
  rewards unless a separate versioned reward/verifier transformation is introduced.
- **Bounded Lab Slurm execution** validates one explicit controlled-private scheduler profile,
  hash-pins the submit contract, records scheduler accounting, and exposes CLI commands to
  inspect the submit host and generate a Command Backend spec. It is deliberately not a
  claim of generic cluster/framework support.

The next major layers are broader arbitrary-open-model trainer adapters, production external
RL backends such as verl/OpenRLHF behind the same reward/evidence boundary, richer distributed
Lab topologies/checkpoint recovery, and validated adaptive evaluation.

---

## Product contracts

Canonical direction, in precedence order:

1. [Product decisions](product/PRODUCT_DECISIONS_OVERRIDE_20260922.md)
2. [User-fit optimization & edition experience](product/USER_FIT_OPTIMIZATION_AND_EDITION_EXPERIENCE_20260924.md)
3. [Edition architecture](product/EDITION_ARCHITECTURE_DECISION_20260923.md)
4. [v1 implementation blueprint](product/V1_IMPLEMENTATION_BLUEPRINT_20260922.md)
5. [Capability v1 spec](product/CAPABILITY_V1_SPEC_20260924.md)

External projects and papers being evaluated for interoperability are tracked in
[External technical references](product/EXTERNAL_TECHNICAL_REFERENCES_20260924.md).

---

## Current status

The previously certified v1 candidate is `1.0.0rc1`. The current rc2 candidate is
**`1.0.0rc2`**. It includes the user-fit optimization, measurement-integrity,
continual-observation, edition-experience, and bounded Lab execution work described above.

Run the verification suite:

```bash
python -m ruff check .
python -m mypy src/frontierwright
python -m pytest
python -m build
```

The frozen v1 candidate gate remains in `tools/v1_release_gate.py`. The rc2 candidate
integrity gate lives in `tools/rc2_release_gate.py` and adds workload-acceptance,
measurement-resolution, observation, RL-boundary, edition-UX, and bounded-Slurm criteria
while retaining clean-wheel and real PyTorch lifecycle/RL smokes.

---

## License

Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
