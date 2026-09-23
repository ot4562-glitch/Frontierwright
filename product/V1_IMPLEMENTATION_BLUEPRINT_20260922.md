# Frontierwright v1 Implementation Blueprint

Date: 2026-09-22
Status: PRODUCT-FIRST / GAME-LIKE LLM DEVELOPMENT / IMPLEMENTATION CONTRACT
Project name: **Frontierwright**
Public product name: **Frontierwright**
License: **Apache-2.0**
Canonical decision overlay: `PRODUCT_DECISIONS_OVERRIDE_20260922.md` has higher precedence if any older wording conflicts.

## 0. Product core

Frontierwright exists to make owning, creating, training, and improving an LLM feel as intuitive as developing a game character.

The product is centered on the user's relationship with their model: understanding it, choosing a build, training it, and watching it grow.

The primary experience is:

```text
MY MODEL
  ->
see its character sheet
  ->
choose what kind of model I want it to become
  ->
see which training paths are available or locked
  ->
see what hardware/data/time each path needs
  ->
train
  ->
watch stats change
  ->
accept or reject the new build
  ->
keep developing the same model identity
```

The model is the character.

Everything game-like must correspond to real LLM state:

```text
character        = the actual model
nickname         = persistent model identity
stats            = measured capability values
rating/power     = current champion capability state; historical peaks stay in history
build            = desired capability allocation
class/archetype  = descriptive shape of the stat profile, never a hidden modifier
training path    = a real training/fine-tuning/pretraining route
resources        = real compute, memory, time, data, storage
locked path      = actually infeasible with current resources/requirements
unlocked path    = actually feasible
candidate        = a real trained descendant/checkpoint
champion         = the currently accepted model
history          = actual lineage and interventions
```

No fake XP, random buffs, arbitrary rarity, fake quests, or decorative mechanics may affect model evaluation or training.

## 1. One-sentence definition

> **Frontierwright is a CLI-first LLM development game-like tool that lets a user create or import an LLM, see it as a persistent character with understandable capability stats, choose a desired build, discover feasible ways to train toward that build, and watch the same model grow through real checkpoints and real training.**

## 2. The v1 success test

A new user should be able to understand the current model and decide what to do next without needing to understand the internal ML stack first.

A competent ML user should still be able to inspect every underlying technical detail.

The product must therefore have two simultaneous layers:

### Player layer

Readable, compact, intuitive:

```text
NOVA
Stage        POST-TRAINING
Build        Code Specialist

General       125
Reasoning     120
Math          115
Coding        130

Next goal
Coding        130 -> 140
General       keep >= 120

Available paths
LoRA SFT      READY       ~4.8 h
QLoRA SFT     READY       ~6.1 h
Full SFT      LOCKED      needs +43 GB VRAM
CPT           LOCKED      dataset missing
```

### Expert layer

Every visible conclusion can be expanded:

```text
frontierwright explain route-003
frontierwright inspect --technical
frontierwright stats --raw
frontierwright resource show --details
frontierwright history --technical
```

The game UX is a view over real ML state, never a replacement for it.

## 3. Primary user journeys

Frontierwright v1 has three equal entry paths.

### 3.1 I already have an LLM

```text
import model
-> inspect
-> measure stats
-> create identity
-> choose build
-> see training paths
-> train candidate
-> compare
-> promote or reject
```

Example:

```bash
frontierwright import ./my-model
frontierwright play
```

### 3.2 I want to build my own LLM from zero

```text
create starter
-> random initialization ("birth")
-> choose/name identity
-> prepare data
-> pretrain
-> first measurable checkpoint
-> first measured character sheet
-> continue growing
```

Example:

```bash
frontierwright new nova --starter zero-25m
frontierwright play
```

### 3.3 I am an experienced LLM developer

The same project state is exposed without the playful wording:

```bash
frontierwright project init --mode expert
frontierwright model inspect --json
frontierwright plan create --json
frontierwright run <plan-id> --non-interactive
```

Game-like presentation is optional. Game-like state semantics are not.

## 4. The main screen: Character Sheet

The most important Frontierwright command is:

```bash
frontierwright play
```

or equivalently:

```bash
frontierwright status
```

Default human output:

```text
╭──────────────── FRONTIERWRIGHT ────────────────╮
│ NOVA                                            │
│ Qwen-derived 14B · POST-TRAINING               │
│                                               │
│ CHARACTER                                     │
│ General       125  ████████████░               │
│ Reasoning     120  ████████████                │
│ Math          115  ███████████░                │
│ Coding        130  █████████████               │
│                                               │
│ BUILD                                         │
│ Code Specialist                               │
│ Coding target         140                      │
│ General floor         120                      │
│                                               │
│ RESOURCES                                     │
│ RTX 4090 x1 · 24 GB VRAM · 64 GB RAM          │
│                                               │
│ NEXT PATHS                                    │
│ [1] LoRA SFT        READY       ~4.8 h         │
│ [2] QLoRA SFT       READY       ~6.1 h         │
│ [3] Full SFT        LOCKED      +43 GB VRAM    │
│ [4] CPT             LOCKED      data required  │
│                                               │
│ Last growth: Coding +2                         │
╰───────────────────────────────────────────────╯
```

This screen is the product.

Everything else exists to make this screen truthful and actionable.

## 5. Interaction model

Frontierwright should feel closer to a character builder than a collection of unrelated ML subcommands.

Recommended top-level commands:

```text
frontierwright play
frontierwright new
frontierwright import
frontierwright stats
frontierwright build
frontierwright paths
frontierwright train
frontierwright compare
frontierwright history
frontierwright resources
frontierwright data
frontierwright explain
```

Advanced/automation namespaces can expose:

```text
frontierwright project ...
frontierwright model ...
frontierwright plan ...
frontierwright run ...
frontierwright candidate ...
frontierwright backend ...
```

Humans should not need the advanced vocabulary for ordinary use.

## 6. The "Build" concept

A build is the model the user is trying to create.

Example:

```bash
frontierwright build edit
```

Interactive UI:

```text
CURRENT BUILD

General       125    keep >= [120]
Reasoning     120    keep >= [115]
Math          115    keep >= [110]
Coding        130    target  [140]

What do you want to change?
> Coding target 145
```

Or:

```bash
frontierwright build set coding=145 --keep general=120 --keep math=110
```

The product must never assume a universally best LLM.

A user may intentionally make a highly specialized model.

Frontierwright shows the consequences and technical requirements; it does not punish specialization.

## 7. Build archetypes

Archetypes are descriptive labels derived from the current or target stat shape.

Examples:

```text
Balanced
Code Specialist
Math Specialist
Science Specialist
Generalist
Sharp Specialist
Custom
```

Rules:

- archetypes do not change training;
- archetypes do not give bonuses;
- archetypes do not alter evaluation;
- they are simply readable summaries of the build vector.

Example:

```text
Build shape: Code Specialist
Concentration: High
```

Do not frame "Balanced" as better than "Specialized."

## 8. Capability stats

Core v1 player-facing axes:

```text
General
Reasoning
Math
Coding
```

Possible experimental axes:

```text
Science
Reliability
Multilingual
```

Stats must be:

- derived from actual evaluation;
- versioned;
- comparable only under declared evaluation conditions;
- able to exceed 100;
- accompanied by underlying raw metrics in expert view.

100 is a reference anchor, not a maximum.

Example:

```text
Coding 151
```

is valid.

Do not show a stat when evidence is insufficient.

Use:

```text
Coding   ?
Status   NOT MEASURED
```

rather than inventing a number.

## 9. Stat scale v1

The display scale exists to make model capability understandable, not to claim that one number captures intelligence.

Each stat is backed by a frozen evaluation bundle.

Conceptually:

```text
raw task measurements
-> normalized task values
-> frozen axis aggregation
-> display stat
```

Requirements:

- frozen scale version;
- frozen task versions;
- frozen task weights;
- frozen reference anchors;
- exact evaluation receipt;
- no moving leaderboard normalization;
- historical score remains historical.

State:

```text
Scale: Frontierwright Capability v1
```

Expert view must expose every raw contributing measurement.

## 10. Identity

Each model lineage has a persistent identity.

Example:

```text
Nickname      NOVA
Identity ID   fw-id-01...
Born/imported 2026-09-22
State        current champion
Champion      model-0042
```

Nickname can be:

- entered by the user;
- chosen by the model;
- changed later.

Nickname is metadata only.

It must never enter benchmark prompts or change model training.

## 11. Current-state rating and progression

The old personal-best accumulation level rule is removed.

Frontierwright must describe the model the user has **now**, meaning the current accepted champion.

Canonical rules:

- all visible capability stats come from the current champion;
- historical bests are stored only as history/milestones;
- a weaker current champion must look weaker even if an older checkpoint was stronger;
- run count, elapsed time, money spent, checkpoint count, and historical-best accumulation do not inflate the current model;
- candidate comparison is always against the current champion.

A single overall `Rating` or `Power` value is optional for v1 until its aggregation formula is frozen and justified.

If implemented, it must be derived only from the current champion's current frozen-scale stats and therefore may rise or fall with the current model.

Individual capability stats remain authoritative.

## 12. From-scratch lifecycle

For a newly created model:

```text
NOVA
Stage       PRETRAINING
Parameters  25M
Tokens      0
Weights     RANDOM INIT
Stats       NOT READY
Rating      NOT READY
```

The model receives its first meaningful current-state character sheet at the first checkpoint eligible for the frozen capability evaluation.

This gives a real game-like lifecycle:

```text
birth
-> first language competence
-> first measurable stats
-> first measured state
-> real training
-> stat growth
-> specialization
-> new builds
```

v1 starter models:

```text
zero-8m
zero-25m
```

These are real small decoder-only models generated from config and deterministic seed.

## 13. Training paths as a skill/path screen

The user should not first encounter "backend configuration."

They should encounter available paths:

```bash
frontierwright paths
```

Example:

```text
TRAINING PATHS

[READY]
LoRA SFT
  Purpose       specialize current model cheaply
  Time          ~4.8 h
  Peak VRAM     ~18.3 GB
  Data          ready
  Keeps base    yes

[READY]
QLoRA SFT
  Purpose       specialize under tighter VRAM
  Time          ~6.1 h
  Peak VRAM     ~13.9 GB
  Data          ready
  Keeps base    yes

[LOCKED]
Full SFT
  Missing       +43 GB VRAM

[LOCKED]
Continued Pretraining
  Missing       pretraining dataset
```

Paths are unlocked only by actual feasibility.

No artificial progression lock.

## 14. v1 real training paths

Required:

```text
From-scratch pretraining
Continued pretraining
Full SFT
LoRA SFT
QLoRA SFT
```

Later additions can include:

```text
DPO
ORPO
GRPO
adapter variants
context extension
model merging
architecture expansion
RAG/tool/no-train paths
```

v1 should not pretend to support a route unless it can actually plan, execute, and record it.

## 15. Training screen

Human-facing command:

```bash
frontierwright train
```

Flow:

```text
Select a path
> LoRA SFT

TARGET BUILD
Coding       130 -> target 140
General      keep >= 120

RESOURCE CHECK
GPU          RTX 4090 x1
VRAM         24 GB
Peak est.    18.3 GB
Status       READY

DATA
Code SFT     81M tokens
Status       READY

ESTIMATED RUN
Wall time    4.8-5.2 h
Storage      +16 GB

Create candidate?
[Y/n]
```

Advanced users can bypass interaction with exact flags/JSON.

## 16. Candidate/champion gameplay

Training never silently replaces the user's current model.

Every run creates a candidate.

```text
NOVA

Champion
General     125
Reasoning   120
Math        115
Coding      130

Candidate #43
General     123   -2
Reasoning   121   +1
Math        116   +1
Coding      137   +7
```

Then:

```text
Build constraints
General >= 120    PASS
Math    >= 110    PASS

Current-state changes
General   -2
Reasoning +1
Math      +1
Coding    +7

[Promote candidate]
[Keep champion]
[Inspect details]
```

If promoted:

```text
NOVA champion updated
Current Coding: 137
Current General: 123
```

If rejected, the current champion and its displayed stats remain unchanged.

## 17. Comparison screen

```bash
frontierwright compare
```

Example:

```text
                CHAMPION    CANDIDATE
General             125          123   -2
Reasoning           120          121   +1
Math                115          116   +1
Coding              130          137   +7

Peak VRAM             -         18.2G
Training time          -          4.9h
New storage            -         15.7G

Build constraints
General >=120                     PASS
Math >=110                        PASS
```

This is more important to v1 than an abstract route optimizer.

## 18. History as a real character timeline

```bash
frontierwright history
```

Example:

```text
NOVA · DEVELOPMENT HISTORY

IMPORT
       Initial measured build
       G126 R118 M114 C121

SFT
       Coding +5
       Reasoning +3
       PROMOTED

       QLoRA
       Coding +2
       General -9
       REJECTED

CPT + SFT
       Coding +4
       Math +1
       PROMOTED

LoRA
       No new personal best
       REJECTED
```

The history is generated from actual model lineage.

## 19. Resources as inventory, but real

Resources may visually behave like an inventory screen, while remaining literal hardware/data.

```bash
frontierwright resources
```

Example:

```text
RESOURCES

COMPUTE
RTX 4090 x1
VRAM      24 GB
RAM       64 GB
Disk      1.2 TB free

DATA
General pretrain    9.2B tokens
Code SFT            81M tokens
Math SFT            34M tokens

LIMITS
GPU budget          20 h
Storage budget      500 GB

OTHER PROFILES
lab-a100            A100 80GB x4
whatif-h100         H100 80GB x8 (simulation)
```

The user should immediately understand why a training path is locked.

## 20. Resource detection

Automatically detect where possible:

- GPU model/count;
- VRAM total/free;
- CPU;
- RAM;
- disk;
- CUDA/ROCm;
- Torch/runtime;
- bf16/fp16 availability;
- model size;
- dataset size/token count;
- measured throughput after calibration.

Resource values carry provenance:

```text
DETECTED
MEASURED
DECLARED
CONNECTED
ESTIMATED
HYPOTHETICAL
```

## 21. Calibration

Before a long training run, Frontierwright can run a short representative calibration.

```bash
frontierwright calibrate
```

Player-facing result:

```text
CALIBRATION COMPLETE

Peak VRAM      18.2 GB
Speed          18.4k tokens/s
Step time      1.82 s

Projected path
LoRA SFT
Time           4.8-5.2 h
Status         READY
```

This turns vague hardware estimation into something concrete.

## 22. Cost

Cost is shown as understandable resources rather than one synthetic score.

v1:

```text
Time
GPU-hours
Peak VRAM
RAM
Storage
Tokens
Optional money estimate
```

Example:

```text
COST

Time         4.8-5.2 h
GPU usage    4.8-5.2 GPU-h
Peak VRAM    18.2 GB
Storage      +15.7 GB
Tokens       81M
Cloud cost   not configured
```

## 23. Data screen

```bash
frontierwright data
```

Example:

```text
DATA LIBRARY

Code-SFT
  81M tokens
  License      Apache-2.0
  Domain       code
  Ready        YES

Math-SFT
  34M tokens
  License      mixed/inspect
  Domain       math
  Ready        REVIEW

Pretrain-General
  9.2B tokens
  Ready        YES
```

Data is part of the player's build resources.

The product tracks provenance, token count, preprocessing, and known license metadata.

## 24. Human-first and agent-friendly are separate concerns

Frontierwright is human-first in interaction design.

It must also expose stable machine interfaces so AI assistants can operate it.

Every important command supports:

```text
--json
--non-interactive
--no-color
--yes
```

But JSON design must never dictate the human UX.

Human command:

```bash
frontierwright train
```

Agent/expert command:

```bash
frontierwright run plan_01K... --json --non-interactive
```

Same engine, different surface.

## 25. Permission model

AI automation must not turn the game-like experience into uncontrolled expensive execution.

Permission levels:

```text
INSPECT_ONLY
PLAN
DRY_RUN
EXECUTE_SINGLE
EXECUTE_BOUNDED
RECURSIVE_EXECUTE
```

Default:

```text
PLAN
```

A recursive mode requires explicit hard budgets and user permission.

## 26. Hard budgets

Possible limits:

```text
max GPU-hours
max wall time
max runs
max storage
optional max monetary spend
```

If a limit is hit:

```text
RUN STOPPED
Reason: GPU-hour budget reached
Candidate status: INCOMPLETE
Champion unchanged
```

A failed or budget-stopped run never becomes a successful candidate.

## 27. Persistent project state

Use one authoritative local project registry plus immutable run receipts.

Recommended:

```text
.frontierwright/
  project.toml
  registry.sqlite
  profiles/
  builds/
  plans/
  runs/
  candidates/
  exports/
```

The player should never need to edit this directly.

Every important state transition is recorded.

## 28. Lineage

Every model state has:

```text
model ID
parent model ID
identity ID
intervention
dataset
config
checkpoint
resource profile
run ID
evaluation snapshot
promotion status
```

This lets the user follow one LLM character for months or years without losing its development history.

## 29. Model fingerprinting

A model is identified by content, not nickname.

Fingerprint includes:

- model config;
- tokenizer;
- weights or immutable source revision;
- adapters;
- relevant generation/evaluation metadata.

The same nickname may exist in different projects.

The identity ID and model fingerprint are authoritative.

## 30. Idempotency

Expensive training must not launch twice because a human or AI retries a command.

Each resolved training plan receives an idempotency fingerprint.

Default:

- completed identical plan -> show existing run;
- running identical plan -> show current run;
- intentional repeat -> explicit `--rerun`.

## 31. Evaluation

Evaluation is what turns a checkpoint into visible stats.

Requirements:

- frozen task/evaluator version;
- deterministic settings where appropriate;
- prompt/config hashes;
- raw metric preservation;
- contamination-risk metadata;
- optional private/user probes.

Nickname and game metadata are never included in evaluation prompts.

## 32. Coding evaluation safety

Generated code must not execute directly on the host simply because Coding is a stat.

Coding evaluation requires an isolated execution backend.

If safe execution is unavailable:

```text
Coding   ?
Reason   safe evaluator unavailable
```

Safety is preferable to a fake or unsafe stat.

## 33. Trainer architecture

Frontierwright should not become a monolithic training framework.

It owns the user experience and model-development state.

Training implementation is adapter-based.

v1:

### Reference backend

Used for:

- zero-8m / zero-25m;
- small pretraining;
- integration/E2E tests;
- simple continued training;
- simple SFT/LoRA.

### External production backend

Use a mature training system through an adapter for larger/real workloads.

### Command backend

Advanced users can connect an internal trainer through a structured argv/config/result contract.

Frontierwright remains the character/build/history layer regardless of trainer.

## 34. Route plugin contract

Each training path plugin exposes:

```text
name
description
requirements
feasibility check
cost estimate
execution materialization
result parser
candidate creation
```

The player-facing description must be understandable without knowing the backend.

Example:

```text
LoRA SFT
"Specialize selected abilities by training small adapter weights.
Low memory cost and easy rollback."
```

Technical detail remains expandable.

## 35. CLI hierarchy

Recommended final v1 surface:

```text
frontierwright play
frontierwright new
frontierwright import
frontierwright stats
frontierwright build
frontierwright paths
frontierwright train
frontierwright compare
frontierwright history
frontierwright resources
frontierwright data
frontierwright explain
```

Advanced:

```text
frontierwright project
frontierwright model
frontierwright profile
frontierwright plan
frontierwright run
frontierwright candidate
frontierwright backend
frontierwright export
```

This is deliberately different from exposing ML implementation vocabulary first.

## 36. Mandatory v1 keyboard TUI

The full-screen terminal UI is a required v1 surface, not a future add-on.

`frontierwright play` MUST launch the keyboard-only roguelike-style human interface defined by the canonical product decisions.

Required v1 tabs/screens:

```text
CHARACTER
BUILD
PATHS
RESOURCES
DATA
HISTORY
CANDIDATES
```

Required navigation:

- Arrow keys or `hjkl`: move;
- Enter: select/confirm;
- Esc: back;
- number keys: direct visible actions where useful;
- `?`: contextual help;
- no mouse required.

Use Textual as the preferred v1 full-screen TUI framework, with Rich rendering reused where practical.

The TUI is only a client of the shared core engine and MUST contain no unique business logic. Every consequential TUI action must have a stable machine/CLI equivalent.

A separate graphical desktop/web GUI is not required for v1.

## 37. Education mode

Education mode uses the exact same model/training engine.

It adds explanations.

Example:

```text
Why didn't parameter count increase?

You trained the existing 25M parameters.
Training changed their values; it did not add new layers or width.
```

A user can learn LLM development by actually developing a real small LLM.

This is a first-class feature, not a toy simulator.

## 38. Expert mode

Expert users can disable explanatory text and access exact technical details.

```bash
frontierwright play --mode expert
```

Example:

```text
LoRA
rank              32
target modules    q_proj,k_proj,v_proj,o_proj
trainable params  83,886,080
optimizer         adamw
precision         bf16
peak calibrated   18.21 GiB
throughput        18,420 tok/s
```

Player and expert mode operate on the same project.

## 39. No false gamification

Forbidden:

- XP for merely running commands;
- daily rewards;
- arbitrary rarity;
- random stat bonuses;
- fake equipment bonuses;
- model personalities affecting benchmark scores;
- rating/power based on time;
- rating/power based on money;
- deliberately obscure technical limitations for "game feel."

Allowed:

- nickname;
- persistent identity;
- character sheet;
- current-state rating/power derived only from the current champion's measured stats, if a justified aggregate is implemented;
- build archetype;
- locked/unlocked real paths;
- growth notifications;
- real history/timeline;
- comparison screens;
- understandable resource inventory.

## 40. v1 repository shape

```text
frontierwright/
  pyproject.toml
  README.md
  docs/
  schemas/
  src/frontierwright/
    cli/
      play.py
      new.py
      import_model.py
      stats.py
      build.py
      paths.py
      train.py
      compare.py
      history.py
      resources.py
      data.py
      explain.py
    core/
      ids.py
      schemas/
      hashing.py
      events.py
    project/
    identity/
    models/
    profile/
    build/
    resources/
    data/
    routes/
    feasibility/
    cost/
    backends/
    execution/
    evaluation/
    lineage/
    candidates/
    export/
    agent/
  tests/
    unit/
    integration/
    e2e/
```

The directory structure should mirror product concepts, not academic concepts.

## 41. v1 implementation order

1. Freeze product vocabulary: Character, Stats, Build, Path, Candidate, Champion, Current State, History.
2. Scaffold package, project state, schemas, SQLite registry.
3. Implement the shared application/service layer so human and machine surfaces use the same business logic.
4. Implement the mandatory Textual keyboard-only `frontierwright play` TUI shell and navigation.
5. Implement model import, origin classification, history-confidence state, and fingerprinting.
6. Implement identity and current character-sheet rendering.
7. Implement resource detection and Resources screen.
8. Implement raw evaluation/profile system.
9. Freeze Frontierwright Capability v1 presentation scale.
10. Implement Stats screen.
11. Implement origin-aware Build UX: intent/priorities for unmeasured zero models; targets/floors for measured models.
12. Implement route/path plugin protocol and history-aware recommendation gating.
13. Implement the five real v1 paths.
14. Implement feasibility and lock/unlock reasons.
15. Implement calibration and cost estimates.
16. Implement user-data-first Data inventory/discovery/preparation flow.
17. Implement Train flow and dry-run.
18. Implement candidate/champion immutable lifecycle.
19. Implement Compare screen.
20. Implement current-state Rating/Power only if its frozen aggregation is justified; otherwise keep individual stats authoritative.
21. Implement History timeline.
22. Implement from-scratch zero-8m and zero-25m paths.
23. Implement the separate expert/AI CLI+JSON surface with stable schemas and exit codes.
24. Add hard budgets, permissions, idempotency, reproducibility export, and offline/privacy protections.
25. Run all TUI, CLI/JSON, integration, and E2E scenarios.
26. Only then release v1.

## 42. Required E2E scenarios

### E2E 1 — Existing model onboarding

```text
import model
-> auto inspect
-> resource detect
-> evaluate
-> nickname
-> first measured character sheet
-> build screen
-> paths screen
```

A first-time user must reach a useful game-like character sheet without reading implementation documentation.

### E2E 2 — Real model growth

```text
champion
-> choose Coding target
-> LoRA path
-> dry-run
-> train
-> evaluate
-> compare
-> promote
-> stats update
-> current champion stats update to the promoted candidate's measured state
```

### E2E 3 — Bad specialization candidate

```text
Coding improves
General falls below user's floor
-> candidate clearly shows the trade
-> promotion is blocked/requires changing the user's declared rule
-> champion remains unchanged
```

### E2E 4 — Locked path becomes unlocked

```text
Full SFT locked on local GPU
-> user adds external resource profile
-> path becomes READY under that profile
```

### E2E 5 — From zero to first measured state

```text
new zero-8m
-> random init
-> real pretraining
-> first measurable checkpoint
-> capability evaluation
-> identity baseline
-> first measured state
```

### E2E 6 — No fake stat

```text
required evaluator unavailable
-> stat shown as unknown
-> no synthesized number
```

### E2E 7 — No duplicate expensive run

```text
same training plan submitted twice
-> one real execution
-> second command points to existing run
```

## 43. v1 release gate

Do not call it 1.0.0 until:

- `frontierwright play` launches the required full-screen keyboard-only roguelike-style TUI;
- the complete primary human loop is usable with keyboard navigation only and requires no mouse;
- Character, Build, Paths, Resources, Data, History, and Candidates are reachable from the v1 TUI;
- consequential TUI actions use the shared core and have equivalent non-interactive CLI/JSON operations;
- AI automation can complete supported workflows through stable CLI/JSON without screen-scraping or keyboard simulation;
- a user can import a real model and immediately understand its character sheet;
- a user can create a small model from zero and reach a first meaningful measured state;
- stats are backed by reproducible evaluation;
- origin-aware Build editing works for zero, imported, and internal/lab model cases;
- history confidence gates strong path recommendations correctly;
- real training paths show READY/LOCKED with reasons;
- resource and user-data-first data requirements are understandable;
- at least one real candidate can be trained, evaluated, compared, promoted, and reflected in current stats/history;
- rejected candidates cannot corrupt the champion;
- all expensive actions support dry-run and idempotency;
- expert/AI JSON output and exit codes are stable;
- offline mode works;
- no game metadata contaminates evaluation;
- Apache-2.0 release files and third-party notice obligations are prepared;
- all TUI, unit, integration, and E2E tests pass.

## 44. Public positioning

Do not lead with:

- experiment orchestration;
- Pareto optimization;
- control plane;
- training economics;
- benchmark framework;
- fine-tuning framework.

Those are implementation/technical properties.

Lead with:

> **Build your LLM like a character. See what it can do, choose what you want it to become, train it through real development paths, and watch the same model grow.**

Technical subtitle:

> A local-first, CLI-first full-lifecycle LLM development environment with real capability stats, builds, resource-aware training paths, model lineage, candidates, and reproducible progression.

## 45. Canonical product rule

Whenever a feature is proposed, ask:

> Does this make developing *my model* more understandable, intentional, and satisfying while still corresponding to real ML state?

If yes, it belongs near the product core.

If it only adds platform complexity without improving the model-as-character development loop, it is secondary.

## 46. Final identity

Frontierwright is about the ownership and development relationship between a user and an LLM.

The user should be able to say:

```text
This is NOVA.
I started it from this model/checkpoint.
These are its stats.
This is the build I am aiming for.
These paths are available on my hardware and data.
I trained this candidate.
Here is exactly what changed.
I kept this version and rejected that one.
This is how the same model has grown over time.
```

That is the product core.

Everything in v1 should reinforce that experience.
