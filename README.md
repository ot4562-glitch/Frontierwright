# Frontierwright

[![CI](https://github.com/ot4562-glitch/Frontierwright/actions/workflows/ci.yml/badge.svg)](https://github.com/ot4562-glitch/Frontierwright/actions/workflows/ci.yml)
[![GitHub Release](https://img.shields.io/github/v/release/ot4562-glitch/Frontierwright)](https://github.com/ot4562-glitch/Frontierwright/releases)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

```text
+======================================================================+
|                         F R O N T I E R W R I G H T                  |
|                                                                      |
|      MEASURE  ->  CHANGE  ->  PROVE  ->  KEEP THE CHAMPION          |
|                                                                      |
|  ORIGIN 0 ========> CANDIDATE ========> CHAMPION ========> REPEAT   |
+======================================================================+
```

> **Don't fit yourself to a model. Fit the model to you.**

Frontierwright is a keyboard-first local model-development environment for people who
**control the model** and want to improve it without losing the evidence.

It measures the model, machine, workload, and constraints; runs real training or model
transformations; creates descendants; compares gains and regressions; and keeps exact
lineage so a training run never silently becomes an “improvement”.

**v1.0.0 ships three experiences over one evidence core: Studio, Academy, and Lab.**

---

## 30-second start

Requires **Python 3.11+**.

### Install from the v1 release source

```bash
git clone https://github.com/ot4562-glitch/Frontierwright.git
cd Frontierwright
git checkout v1.0.0

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -e .
```

For the built-in PyTorch reference model, real training, local evaluation, generation,
and profiling:

```bash
python -m pip install -e ".[train]"
```

Then choose an edition.

### Studio — grow a model you already own

```bash
frontierwright import /path/to/model --path ./my-model --edition STUDIO --name MYMODEL
frontierwright resources detect --path ./my-model
frontierwright play --path ./my-model
```

### Academy — build a real model from zero

```bash
frontierwright project init ./academy --name NOVA --origin ZERO --edition ACADEMY
frontierwright play --path ./academy
```

### Lab — controlled private model development

```bash
frontierwright import /path/to/internal-model --path ./lab \
  --origin INTERNAL_LAB --edition LAB --name FRONTIER-LAB
frontierwright play --path ./lab
```

Everything visible in the TUI is backed by the same service layer used by the CLI/JSON
API. Automation should use `--json --non-interactive --yes` instead of driving the TUI.

---

## The idea

```text
                         YOUR REAL CONSTRAINTS
                   workload · machine · privacy · budget
                                  |
                                  v
+-----------+      +----------+      +-----------+      +------------+
|  ORIGIN   | ---> |  PLAN    | ---> | CANDIDATE | ---> |  COMPARE   |
| baseline0 |      | calibrate|      | real bytes|      | evidence   |
+-----------+      +----------+      +-----------+      +------------+
      ^                                                         |
      |                                                         v
      |                                                +----------------+
      +------------------- REPEAT <--------------------| PROMOTE/REJECT |
                                                       +----------------+
                                                               |
                                                               v
                                                          CHAMPION
```

A successful training job means **a Candidate exists**. It does not mean the Candidate
is better. Frontierwright keeps the Champion unchanged until measured evidence and an
explicit decision say otherwise.

The permanent project Origin is display baseline **0**. Promoting a new Champion never
resets the growth history.

---

## Three editions, one model history

| Edition | Best for | What the interface emphasizes |
| --- | --- | --- |
| **Studio** | Local/open-model owners and hobbyists | Growth, specialization, real trade-offs, machine fit, build goals |
| **Academy** | Learning by actually building a model | Data → tokenizer → birth → training → evaluation → Candidate → Champion |
| **Lab** | Startups/teams developing controlled private models | Exact evidence, workload acceptance, adapters, budgets, RL and scheduler boundaries |

Changing editions changes workflow density and guidance. It does **not** rewrite model
identity, lineage, measurements, or history.

### Studio: “grow my model”

Studio supports origin-relative growth rules:

- `IMPROVE` — this metric should measurably improve;
- `PROTECT` — this metric must not regress;
- `TOLERANCE` — bounded regression is acceptable;
- absolute requirements and hard constraints for advanced workflows.

```bash
frontierwright build goals --path ./my-model \
  --improve capability.coding \
  --protect capability.reasoning \
  --tolerance serving.latency_p50=10% \
  --hard privacy=PRIVATE
```

### Academy: real state, not a tutorial simulator

Academy can train a tokenizer, materialize a deterministic zero-model root, pretrain it,
evaluate it, create descendants, and promote a measured Candidate.

```bash
frontierwright data add ./corpus.txt --role PRETRAIN --classification PRIVATE --path ./academy
frontierwright birth tokenizer DATASET_ID --path ./academy --vocab-size 384
frontierwright birth zero --path ./academy --preset zero-8m --tokenizer-artifact TOKENIZER_ID
```

### Lab: exact evidence before claims

Lab adds explicit workload acceptance contracts, controlled-private adapter identities,
scheduler contracts, and a bounded verifier-driven RL path.

```bash
frontierwright workload acceptance schema --json
frontierwright workload acceptance example --path ./lab --json
frontierwright lab adapters schema --json
frontierwright lab slurm schema --json
frontierwright plan rl-schema --json
```

`PRIVATE` means Frontierwright's application/data-boundary policy. It is **not** an
attestation of OS, process, kernel, hypervisor, or network isolation.

---

## What v1 can actually do

### Model lifecycle

- import local trainable model directories with fingerprints and history confidence;
- Candidate / Champion lifecycle with explicit promote/reject decisions;
- permanent Origin identity and origin-relative Stat v2 primitives;
- exact lineage for training and transformations;
- deterministic export + independent verification.

### Real built-in training paths

The PyTorch reference backend executes real parameter updates for:

- from-scratch pretraining;
- continued pretraining;
- full-parameter causal SFT;
- merged-output LoRA SFT;
- bounded reference QLoRA SFT;
- DPO;
- knowledge distillation;
- bounded verifier-driven `RL_POLICY_OPTIMIZATION` using REINFORCE.

The built-in RL path is intentionally narrow: verifiable multiple-choice episodes with an
exact reward. It is **not** presented as open-ended RLHF, arbitrary agentic RL, or a
production distributed RL stack.

### Model transformations

- deterministic weighted linear merge;
- symmetric int8 deployment artifact generation;
- local reference generation;
- portable export and verification.

Quantization v1 claims the measured artifact-storage transformation. The reference loader
may dequantize for execution, so it does not invent RAM/latency gains.

### Evaluation and evidence

- frozen 64-item Capability v1 smoke benchmark;
- held-out causal-LM evaluation packs;
- exact Champion/Candidate comparable evidence;
- Wilson uncertainty intervals and paired McNemar diagnostics;
- external evidence import from lm-evaluation-harness, LightEval, vLLM, generic manifests,
  and exact serving-resource receipts;
- explicit workload language/domain/task coverage binding;
- versioned Workload Acceptance contracts with point or interval-aware rules;
- privacy-minimal real-use observations with explicit provenance.

Capability v1 is a **small frozen smoke benchmark**, not a claim of general intelligence.
Its role is regression/integration evidence.

### Resource and execution control

- CPU/RAM/disk/GPU discovery without pretending an unavailable GPU was measured;
- reference-model latency/throughput/process-memory profiling;
- uncertainty-aware runtime comparison;
- consent-triggered interleaved Champion/Candidate runtime remeasurement;
- hard run-count, wall-time, and storage gates;
- durable worker/result/usage receipts and reconciliation;
- idempotent replay for expensive requests;
- repairable exported run receipts.

### Lab infrastructure

- controlled-private adapter manifests;
- exact adapter identity and manifest-drift guards;
- one bounded Slurm bridge with hash-pinned scheduler/worker profile;
- scheduler availability inspection and backend-spec generation.

The Slurm bridge requires a POSIX submit host and real `sbatch`/`scancel` tools for
actual submission. Frontierwright reports unsupported hosts instead of pretending a
cluster exists.

---

## Stats: growth without fake precision

Frontierwright v1 defines these long-lived model axes:

```text
KNOWLEDGE   REASONING   MATH   CODING   INSTRUCTION   LANGUAGE   CONTEXT
```

System evidence stays separate:

```text
AGENTIC / TOOL USE / LATENCY / THROUGHPUT / MEMORY / STORAGE
```

That separation matters: a better scaffold, tool runner, or runtime must not silently
become a claim that model weights improved.

Stat v2 stores raw benchmark evidence and derives an origin-relative display. If evidence
is not compatible, the result stays `UNMEASURED`. If uncertainty overlaps the decision
boundary, the result stays `UNCERTAIN`.

---

## Benchmark registry

`frontierwright benchmarks list --json` ships a metadata registry for external benchmark
sources, currently including model, system, and framework sources such as LiveBench,
MMLU-Pro, IFEval, LiveCodeBench, SciCode, LongBench v2, Terminal-Bench,
lm-evaluation-harness, Inspect AI, and OpenCompass.

The registry is deliberately **not** a bundled benchmark-data dump. Each source keeps its
own version, runner, license, contamination, and redistribution considerations. v1 does
not claim every cataloged benchmark is already executable through Frontierwright.

---

## Terminal UX

Launch the keyboard interface:

```bash
frontierwright play --path .
```

The v1 shell is edition-aware and starts with the same identity everywhere:

```text
+-- FRONTIERWRIGHT ----------------------------------+
|  MEASURE -> CHANGE -> PROVE -> KEEP THE CHAMPION  |
+----------------------------------------------------+

[CHARACTER] [WORKLOAD] [RESOURCES] [BUILD] [PATHS] [DATA] [CANDIDATES] [HISTORY]

GROWTH STAT · ORIGIN MODEL = 0
Knowledge      +0.0  BASELINE
Reasoning      +6.2  UNCERTAIN
Math           +4.1  SAME
Coding        +18.4  BETTER
Instruction      ?   UNMEASURED
Language         ?   UNMEASURED
Context          ?   UNMEASURED
```

Core navigation is visible in the footer; `?` opens help and `a` opens the
edition-specific Action Center. Scripted Textual QA uses the same widgets and reports
only visible active-tab text by default.

---

## CLI for humans and automation

Every public v1 command has a one-line help description.

```bash
frontierwright --help
frontierwright workload --help
frontierwright plan --help
frontierwright lab slurm --help
```

Machine surfaces use JSON and stable error codes:

```bash
frontierwright status --path . --json --non-interactive
frontierwright workload next --path . --json --non-interactive
frontierwright candidates --path . --json --non-interactive
```

Expensive state-changing operations support explicit confirmation, dry-run/calibration
where applicable, durable identity, and replay semantics.

---

## Final v1 validation

Before the `v1.0.0` tag, the release candidate was played through all three editions
using the public CLI/TUI:

- **Academy** — zero-model birth, real PyTorch pretraining, evaluation, Candidate compare,
  and successful Champion promotion;
- **Studio** — continued pretraining, full SFT, LoRA, QLoRA, DPO, int8 transform, merge,
  external eval/serving evidence, and full TUI tab walk;
- **Lab** — exact interval-aware acceptance PASS, workload coverage PASS, controlled adapter
  connect/disconnect, bounded Slurm host inspection, real verifier-driven RL, compare/reject,
  and full TUI tab walk;
- **CLI discovery** — all 81 public command help surfaces executed successfully.

The repository also runs Ruff, strict mypy, the full pytest suite, package build, clean-wheel
smoke, real reference-training lifecycle smoke, scripted Textual smoke, and reference-RL smoke.

Detailed implementation boundaries are kept in
[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

---

## Deliberate v1 boundaries

Frontierwright v1 does **not** claim:

- arbitrary Hugging Face architecture training support;
- executable adapters for every benchmark in the catalog;
- production verl/OpenRLHF integration;
- generic multi-node/distributed training guarantees beyond the bounded Slurm bridge;
- OS/hypervisor/network privacy attestation;
- universal model improvement.

These are product boundaries, not hidden “coming soon” behavior. Unknown evidence stays
unknown.

---

## Development verification

```bash
python -m pip install -e ".[train,dev]"
python -m ruff check .
python -m mypy src/frontierwright
python -m pytest -q
python -m build
python tools/release_smoke.py
python tools/v1_e2e_smoke.py --timeout 180
python tools/rl_reference_smoke.py
python tools/rc3_scripted_play_smoke.py
```

On a POSIX host with a real Slurm client, run the Slurm integration path separately.
Windows intentionally reports that submission is unavailable.

---

## Product and engineering documents

- [Implementation status](docs/IMPLEMENTATION_STATUS.md)
- [Product decisions](product/PRODUCT_DECISIONS_OVERRIDE_20260922.md)
- [RC4 Stat v2 / benchmark / Studio growth design](product/RC4_STAT_V2_BENCHMARK_AND_STUDIO_GROWTH_20260926.md)
- [User-fit optimization and edition UX](product/USER_FIT_OPTIMIZATION_AND_EDITION_EXPERIENCE_20260924.md)
- [Edition architecture](product/EDITION_ARCHITECTURE_DECISION_20260923.md)
- [Capability v1 specification](product/CAPABILITY_V1_SPEC_20260924.md)
- [External technical references](product/EXTERNAL_TECHNICAL_REFERENCES_20260924.md)

---

## License

Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
