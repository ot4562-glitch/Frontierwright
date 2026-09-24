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
fake precision.

Resource evidence similarly distinguishes:

- **system availability now** — RAM, disk, and GPU VRAM reported free at detection;
- **model-specific headroom** — requires an actual model/runtime profile before it can
  be claimed.

Unknown stays **UNKNOWN** until measured.

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
- raw comparable Candidate-vs-Champion evidence and promotion gates;
- local generation, inference profiling, portable export and verification;
- private-data boundaries and Lab adapter contracts;
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

## Development directions

The next major layers are centered on **user-fit optimization**, not cosmetic game
mechanics:

1. **Workload Profiles** — versioned task/language/context/latency/privacy mixtures
   built from explicit user input or permitted local observations.
2. **Measured model fit** — combine inference profiles with resource snapshots to show
   actual post-load VRAM/RAM/latency/throughput headroom.
3. **Broader evaluation adapters** — ingest mature suites while preserving exact task,
   evaluator, runtime, and raw evidence identity.
4. **Pareto candidate selection** — capability, reliability, latency, throughput,
   VRAM/RAM/storage and cost without hiding trade-offs in one magic score.
5. **Real RL adapters** — real rollout/reward/policy-optimization workflows for Lab;
   preference optimization is not mislabeled as RL.
6. **Hardware-fit descendants** — measured pruning, distillation, quantization and
   architecture transforms instead of forcing users to choose only stock released
   sizes.

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

The previously certified v1 candidate is `1.0.0rc1`. The current development version
is **`1.0.0rc2.dev0`**, which starts the user-fit optimization and edition-experience
pass described above.

Run the verification suite:

```bash
python -m ruff check .
python -m mypy src/frontierwright
python -m pytest
python -m build
```

The canonical candidate gate lives in `tools/v1_release_gate.py` and includes package
installation smoke plus a real local PyTorch Birth → train → Capability v1 → compare
→ promote lifecycle.

---

## License

Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
