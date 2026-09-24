# Frontierwright — User-Fit Optimization and Edition Experience Contract

Date: 2026-09-24
Status: CANONICAL
Precedence: This document extends and, where necessary, overrides older game-first wording. It does not change the one-core / three-edition architecture.

## 1. Product north star

Frontierwright is **not** an LLM-raising game.

It is a lifelong model-development system that turns:
- an open/trainable starting model,
- the user's hardware and budget,
- the user's own workload and data,
- observed failures and preferences,
- real evaluation evidence,
- and available training/optimization methods

into a model that is continuously optimized for one specific user or organization.

Human-facing game-like language is a presentation layer for making this process intuitive. It must never replace real ML state, real evaluation, real reinforcement signals, real resource accounting, or real lineage.

Primary product statement:

> **Don't fit yourself to a model. Fit the model to you.**

Engineering statement:

> **Fixed open models are starting points, not finished products.**

## 2. The inefficiency Frontierwright attacks

Open models are distributed in discrete architectures, parameter counts, quantizations, context windows, and serving formats. A user's resource envelope and workload are continuous and highly specific.

The common failure mode is therefore:
1. choose the nearest available model;
2. accept unused VRAM/RAM/storage or excessive latency;
3. accept capabilities the user does not need;
4. accept weak capabilities in the user's actual domain;
5. repeatedly work around the model instead of changing it.

Frontierwright must invert this relationship.

The key question is not:

> Which downloadable model is best?

It is:

> Given this user's workload, data, hardware, latency constraints, privacy boundary, and improvement budget, which reachable descendant model has the highest user utility?

## 3. User-fit optimization objective

Frontierwright treats model development as constrained multi-objective optimization.

Conceptually:

    maximize   U(model | workload, preferences, observed failures)
    subject to VRAM <= available VRAM
               RAM <= available RAM
               storage <= available storage
               latency <= user limit
               throughput >= user floor
               training cost <= budget
               privacy/data-boundary constraints
               minimum capability/reliability floors

Important consequence:

**Resource utilization is not the objective. User utility is.**

If 400 MiB of unused VRAM has no meaningful marginal benefit, leaving it unused can be optimal.

If a smaller specialized model is materially better for the user's real tasks while cheaper/faster, it can be a better Frontierwright model even if it ranks lower on broad public leaderboards.

## 4. Workload is a first-class object

Frontierwright must eventually persist a versioned **Workload Profile** that can contain:
- task/domain mixture;
- languages;
- prompt/context length distribution;
- output length distribution;
- tool-use frequency;
- latency sensitivity;
- concurrency/batch pattern;
- privacy classification;
- user correction/failure categories;
- critical capability floors;
- reliability/safety requirements;
- serving environment.

A workload profile is evidence, not a personality quiz.

The profile may be:
- explicitly authored;
- imported from logs;
- inferred from private local usage only with explicit permission;
- updated over time.

## 5. Utility profile is separate from capability

Capability is what the model can do.
Utility is how valuable that capability is to this user under this workload and cost envelope.

A future Utility Profile may combine:
- user-defined capability weights;
- hard floors;
- latency/throughput preferences;
- memory/storage costs;
- training/spend budgets;
- reliability penalties;
- task success rates from real use.

Frontierwright must preserve the raw measurements behind every derived utility score.

A single overall utility number, if introduced, must never hide Pareto trade-offs.

## 6. Resource headroom is actionable evidence

Resource detection must evolve from “what hardware exists?” into:

- total resource;
- currently available resource;
- measured current-model peak;
- headroom after the current model;
- calibrated headroom for candidate operations;
- confidence/provenance of each measurement.

Examples of useful headroom-driven actions:
- increase context size;
- increase batch size;
- raise adapter rank;
- use a larger descendant;
- keep the same model and add a specialist adapter;
- quantize to free memory for longer context;
- prune/distill to hit a latency/VRAM target;
- reject an expansion because marginal utility is too low.

Frontierwright must not claim feasibility from parameter count alone. Calibration or measured backend evidence is preferred.

## 7. Evaluation truthfulness requirements

Displayed stats must never imply more precision than the evidence supports.

Every capability result should, where technically possible, expose:
- point estimate;
- sample/task count;
- uncertainty interval or equivalent reliability estimate;
- exact benchmark/task bundle version;
- scoring method;
- contamination status/limitations;
- evaluator/runtime identity;
- raw evidence.

Capability v1 remains a frozen local microbenchmark and is not “general intelligence.”

Future evaluation architecture should support:
- public reproducible suites;
- user-private workload evals;
- edition/domain Stat Packs;
- contamination-resistant/live suites;
- robust coding execution tests;
- reliability/calibration/robustness metrics;
- IRT/adaptive or efficient benchmarking only when statistically validated.

## 8. Actual performance improvement

A successful run is not an improvement.

The improvement contract is:

    Champion evidence
        -> intervention
        -> Candidate
        -> same/comparable evaluation
        -> workload/resource measurement
        -> explicit promote/reject

Frontierwright should make regressions as visible as gains.

The product should optimize user-relevant **Pareto fronts**:
- capability;
- latency;
- throughput;
- VRAM;
- RAM;
- storage;
- reliability;
- training cost.

## 9. Real reinforcement learning

Frontierwright must reserve the term “reinforcement learning” for training loops with real rollout/reward/policy optimization semantics.

DPO is preference optimization, not renamed RL.

Future RL support should be adapter/plugin based and capable of connecting to mature RL systems rather than reimplementing them all.

Reward sources can include:
- deterministic math/code verifiers;
- unit/integration tests;
- tool-task completion;
- environment success/failure;
- user preference/reward signals;
- lab reward models;
- bounded human feedback.

Every reward source needs identity, version, provenance, and anti-reward-hacking diagnostics where possible.

## 10. Continual content comes from real use

Frontierwright does not need fake game content.

The endless development loop is supplied by reality:

    use
      -> observe successes/failures
      -> collect evidence
      -> update workload/data/reward set
      -> diagnose weakness
      -> propose intervention
      -> train Candidate
      -> evaluate
      -> promote/reject
      -> deploy/use again

New open models, new hardware, new training algorithms, new data, and changing user requirements are all new development content.

## 11. Three editions: one core, genuinely different experiences

Edition remains a UX/policy profile. Model identity and lineage remain edition-independent.

### Frontierwright Academy — understand by doing

Primary user need:
- learn how real language models are created and changed.

Experience goals:
- every major action explains **what changed in the real model**;
- progressive disclosure: beginner summary first, exact IDs/config/runtime in Details;
- concepts are introduced only when needed;
- before/after visualizations connect action to evidence;
- mistakes become teaching moments with recovery guidance;
- no fake XP or simulated training.

Academy should emphasize:
- tokenizer -> model birth -> pretraining distinction;
- loss/perplexity/grad norm/LR/tokens/throughput/VRAM;
- why evaluation differs from training;
- why successful training can still yield a rejected candidate;
- why resource limits shape architecture and intervention choices.

Default UX:
- guided lifecycle;
- explanatory cards;
- compact presets;
- “Why?” and “What changed?” affordances;
- advanced settings collapsed.

Success criterion:
A user should leave with a more accurate mental model of training, evaluation, lineage, and trade-offs.

### Frontierwright Studio — own and shape your personal model

Primary user need:
- turn an available open/trainable model into the best model for their own tasks and machine.

Experience goals:
- strongest sense of ownership and iteration;
- fast path from import to workload fit analysis;
- visible hardware headroom;
- model-vs-user fit gaps;
- specialist builds/adapters/merges/compression;
- clear trade-offs and Pareto comparisons;
- low configuration friction.

Studio is where game-like presentation can be most motivating, but every progression element is backed by real evidence.

Default UX:
- “Your model / Your machine / Your workload” dashboard;
- headroom and bottleneck cards;
- next highest-value experiment;
- candidate trade-off comparison;
- lineage and specialist branches;
- model-use feedback -> next intervention loop.

Success criterion:
The user should prefer their Frontierwright descendant to the nearest stock open model for their own workload.

### Frontierwright Lab — push controlled models toward the frontier

Primary user need:
- maximize capability/efficiency/reliability on large private models and infrastructure.

Experience goals:
- professional research and training control;
- no childish language;
- exact provenance, configs, metrics, confidence, reproducibility;
- cluster-aware execution and cost accounting;
- strong benchmark/eval packs;
- large-scale SFT, preference optimization, real RL, distillation, pruning, quantization, architecture migration;
- private reward/eval/data adapters;
- scalable experiment matrices and Pareto frontier analysis.

Lab should integrate mature external training stacks through adapters:
- Megatron / distributed training;
- verl / OpenRLHF / equivalent RL stacks;
- vLLM/SGLang inference;
- robust evaluation suites.

Default UX:
- experiment matrix;
- evaluation confidence and contamination notes;
- resource topology;
- job/cluster state;
- candidate frontier;
- regression and statistical significance view.

Success criterion:
A lab can use Frontierwright as the control/evidence/lineage layer while retaining its own high-performance trainer, evaluator, reward system, storage, and cluster.

## 12. Edition UX must not be cosmetic

Changing edition must alter:
- onboarding;
- default information density;
- help content;
- default action ordering;
- terminology;
- which details are collapsed/expanded;
- recommended views;
- warnings and policy gates;
- success feedback.

Changing edition must **not**:
- rewrite model history;
- change raw evidence;
- change model weights;
- invent different stats.

## 13. Immediate engineering priorities from this contract

P0:
1. measurement uncertainty in Capability v1;
2. edition-specific TUI help/experience copy;
3. fix candidate keyboard hint rendering;
4. user-facing resource headroom view;
5. compact README focused on value and quick proof instead of implementation inventory.

P1:
1. Workload Profile schema + local/private ingestion;
2. Utility constraints/weights + Pareto candidate comparison;
3. benchmark adapter layer for lm-eval/LightEval/OpenCompass/HELM-class suites;
4. real inference benchmark receipts (TTFT, throughput, latency, VRAM/RAM);
5. structured pruning/distillation/quantization adapters for hardware-fit descendants.

P2:
1. real RL intervention family backed by external RL adapters;
2. reward/environment contracts;
3. adaptive/IRT-backed evaluation research track;
4. Lab distributed executor + Megatron/Slurm/private cluster adapters;
5. continual observe -> data/reward -> intervention loop.

## 14. Non-goals

Frontierwright will not:
- promise that arbitrary leftover VRAM can be converted smoothly into arbitrary parameter count;
- claim that a compressed/quantized model is faster without measured runtime evidence;
- call preference optimization “RL” when there is no RL loop;
- hide regression behind a single score;
- upload private data just to classify or optimize it;
- optimize benchmark score while ignoring the user's actual workload.
