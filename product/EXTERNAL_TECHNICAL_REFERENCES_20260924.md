# Frontierwright External Technical References — 2026-09-24

Purpose: local engineering research notes for evaluation accuracy, real performance improvement, resource-aware model optimization, RL, and README/product presentation.

This file records **what Frontierwright should learn**, not code to vendor blindly. Licenses and compatibility must be reviewed before importing any implementation.

## Evaluation / measurement

### EleutherAI lm-evaluation-harness
https://github.com/EleutherAI/lm-evaluation-harness

Why it matters:
- large established benchmark registry;
- public prompt/task reproducibility;
- local/HF/vLLM/adapters support;
- plugin support is especially relevant to Frontierwright's evaluator adapter architecture.

Frontierwright lesson:
- do not hardcode all benchmarks into core;
- add evaluator/task plugins with exact version identity;
- preserve sample-level evidence for debugging.

### Hugging Face LightEval
https://github.com/huggingface/lighteval

Why it matters:
- 1000+ tasks across knowledge/math/code/chat/multilingual/long-context;
- custom task/metric support;
- multiple execution backends.

Frontierwright lesson:
- external task suites can feed Frontierwright receipts;
- Frontierwright should own identity/provenance/stats mapping, not reimplement every benchmark.

Caution:
- current README states Windows is not officially supported; use adapters/subprocess/containerization rather than making it a hard core dependency on Windows.

### Stanford HELM
https://github.com/stanford-crfm/helm

Why it matters:
- holistic evaluation beyond accuracy;
- efficiency, bias, toxicity, robustness and domain suites;
- strong provenance/reproducibility model.

Frontierwright lesson:
- capability alone is insufficient;
- future Stat Packs should include efficiency/reliability/robustness dimensions.

Note:
- HELM announced maintenance mode in 2026, so use the design principles and interoperable result ingestion rather than making HELM itself a strategic single dependency.

### LiveBench
https://github.com/LiveBench/LiveBench

Why it matters:
- contamination-resistant design;
- objective ground truth;
- continuously refreshed questions.

Frontierwright lesson:
- public benchmark scores need contamination/freshness metadata;
- dynamic external benchmarks should remain separate from frozen longitudinal local stats.

### LiveCodeBench
https://github.com/LiveCodeBench/LiveCodeBench

Why it matters:
- execution-oriented, contamination-aware coding evaluation.

Frontierwright lesson:
- Coding stat should evolve from toy multiple choice toward isolated executable tests, with sandbox identity and resource limits.

### EvalPlus
https://github.com/evalplus/evalplus

Why it matters:
- far more tests per generated program than baseline HumanEval/MBPP;
- demonstrates how apparently-correct code can fail under stronger test suites.

Frontierwright lesson:
- coding evaluation needs hidden/expanded execution tests and robustness evidence, not surface answer matching.

### tinyBenchmarks
https://github.com/felipemaiapolo/tinyBenchmarks
Paper: https://proceedings.mlr.press/v235/maia-polo24a.html

Why it matters:
- uses IRT-based anchor selection to estimate large benchmark performance with far fewer examples;
- reports low empirical estimation errors on several benchmarks.

Frontierwright lesson:
- IRT/adaptive measurement is promising for cheap repeated per-candidate evaluation;
- it should enter only after validation against Frontierwright model populations and workloads.

### PSN-IRT / “Lost in Benchmarks?”
DOI: https://doi.org/10.1609/aaai.v40i41.40814

Why it matters:
- 2026 benchmark analysis reports measurement-quality problems across mainstream benchmarks;
- uses enhanced IRT to model item/model characteristics.

Frontierwright lesson:
- a stat needs measurement quality, not only benchmark popularity;
- future Capability versions should evaluate item discrimination/difficulty and model separability.

## Real RL / post-training

### verl
https://github.com/verl-project/verl

Why it matters:
- production-oriented RL for LLMs;
- PPO/GRPO workflows;
- FSDP/Megatron/vLLM/SGLang integrations;
- flexible device mapping and multi-GPU LoRA RL.

Frontierwright lesson:
- Lab RL should be an adapter to mature rollout/training stacks;
- Frontierwright owns reward/environment identity, budgets, lineage, receipts, candidate gates.

### OpenRLHF
https://github.com/OpenRLHF/OpenRLHF

Why it matters:
- PPO, REINFORCE++, GRPO, RLOO and agent-based RL;
- single/multi-turn reward-driven workflows.

Frontierwright lesson:
- “real RL” intervention contract should be algorithm-agnostic;
- reward source and rollout environment are first-class immutable evidence.

### NVIDIA Megatron-LM / Megatron RL
https://github.com/NVIDIA/Megatron-LM

Why it matters:
- large-scale transformer training building blocks;
- TP/PP/DP/EP/CP;
- post-training and RL;
- model scales into hundreds of billions of parameters.

Frontierwright lesson:
- Lab should not replace Megatron;
- Lab should coordinate model/data/config/cluster/eval/reward lineage around it.

## Resource-aware serving / fitting

### vLLM
https://github.com/vllm-project/vllm

Why it matters:
- established inference engine;
- benchmark surfaces for latency, serving throughput and offline throughput;
- useful model/resource execution evidence.

Frontierwright lesson:
- “fits my machine” needs measured TTFT, latency, throughput and memory, not parameter count heuristics.

### NVIDIA Minitron / Model Optimizer
https://github.com/nvlabs/minitron
https://github.com/NVIDIA/Model-Optimizer

Why it matters:
- structured width/depth pruning followed by distillation;
- examples explicitly target different parameter counts and latency/accuracy trade-offs;
- current Model Optimizer docs show search over architecture dimensions.

Frontierwright lesson:
- a stock 8B model can become a measured 7B/6B/etc descendant rather than forcing the user to choose only released sizes;
- pruning must be followed by re-evaluation and often distillation;
- this directly supports resource-envelope optimization.

### Wanda
https://github.com/locuslab/wanda

Why it matters:
- weight+activation pruning;
- lightweight route for sparsity experiments.

Frontierwright lesson:
- useful as an OPTIMIZE plugin family candidate, especially for experimental Lab workflows.

### AWQ / AutoAWQ
https://github.com/casper-hansen/autoawq

Why it matters:
- activation-aware 4-bit weight quantization;
- shows inference speed/memory trade-offs and kernel dependence.

Frontierwright lesson:
- quantization must report storage **and measured runtime behavior on the user's backend**;
- “4-bit” alone is not enough to claim speed.

## README / product presentation references

### uv
https://github.com/astral-sh/uv

Pattern worth copying:
- one-line value proposition immediately;
- visual proof/benchmark early;
- short “Highlights”;
- installation immediately available;
- deep implementation detail lives in docs.

### FastAPI
https://github.com/fastapi/fastapi

Pattern worth copying:
- outcome-focused tagline;
- quick working example;
- proof of value before architecture detail.

### Hugging Face Transformers
https://github.com/huggingface/transformers

Pattern worth copying:
- broad capability communicated without dumping every internal implementation detail;
- ecosystem/integration framing.

### ComfyUI
https://github.com/Comfy-Org/ComfyUI

Pattern worth copying:
- clear identity and extensibility;
- user mental model is the workflow, not the internal engine.

### OpenAI Evals
https://github.com/openai/evals

Pattern worth copying:
- focuses README around why evals matter and how users create/use them;
- reinforces private/custom evaluation as first-class.

## Product-experience observation: Meta Muse

Current September 2026 reporting describes Meta's Muse as a consumer assistant focused on ordinary-life execution such as email/subscription/shopping/phone-call tasks rather than merely presenting a benchmark-leading model.

Sources consulted:
- Wired, September 2026
- Business Insider, September 2026
- Axios, September 2026

Frontierwright lesson:
- raw model technology does not define the complete product experience;
- each Frontierwright edition should center a distinct user job.

Do **not** copy Muse's privacy model. Frontierwright's private/local-first data boundary remains a product differentiator.

## Engineering adoption policy

Before integrating any external project:
1. confirm license;
2. pin version/commit;
3. wrap behind Frontierwright adapter contract;
4. record exact tool identity in receipts;
5. keep raw external metrics;
6. do not translate external scores into Frontierwright stats without a frozen mapping;
7. re-run Frontierwright regression/evaluation after every transform/training operation.

## Serving-benchmark precision addendum — 2026-09-25

Current vLLM benchmark documentation explicitly warns that prefix-cache reuse can inflate
throughput across repeated runs and distinguishes client-visible TTFT, request TPOT and ITL
measurement points. Frontierwright therefore stores serving-condition identity and now
composes client latency evidence with server VRAM/RSS receipts only when exact model
fingerprint, runtime/version, execution boundary and condition hash match.
