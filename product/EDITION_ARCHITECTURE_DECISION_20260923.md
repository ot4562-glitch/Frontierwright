# Frontierwright Edition Architecture Decision

Date: 2026-09-23
Status: FROZEN PRODUCT DECISION
Precedence: This document is authoritative for Frontierwright edition structure, naming, distribution boundaries, and shared-core architecture.

## 1. Decision

Frontierwright ships as one product family with one shared core and three official editions:

1. **Frontierwright Studio**
2. **Frontierwright Academy**
3. **Frontierwright Lab**

These are not three forks and not three independent products.

They are distribution and UX profiles over the same Frontierwright Core, state model, registry, recipe system, execution engine, evaluation engine, artifact model, lineage, and candidate/champion lifecycle.

The GitHub source repository remains:

`ot4562-glitch/Frontierwright`

## 2. Product family

### Frontierwright Studio

Purpose:

Develop an existing user-controlled model.

Primary users:

- local-model users;
- independent developers;
- model builders;
- researchers working with local/trainable checkpoints;
- users who already have a Hugging Face-compatible or otherwise supported trainable model.

Typical starting point:

```text
Existing local model
    ↓
Import
    ↓
Inspect / fingerprint
    ↓
Evaluate
    ↓
Develop
```

Primary workflow families:

- continued pretraining;
- domain-adaptive pretraining;
- full SFT;
- parameter-efficient tuning;
- alignment;
- model merging;
- distillation;
- quantization / deployment optimization;
- evaluation;
- comparison;
- explicit promotion/rejection;
- continued model evolution.

Studio is the default general-purpose Frontierwright experience.

### Frontierwright Academy

Purpose:

Build and understand a real model from birth.

Primary users:

- students;
- educators;
- AI/ML beginners;
- developers learning model training;
- workshops and courses;
- researchers prototyping small models from scratch.

Typical starting point:

```text
Corpus
  ↓
Tokenizer recipe
  ↓
Tokenizer artifact
  ↓
Architecture recipe
  ↓
Initialization recipe
  ↓
MODEL BIRTH
  ↓
Root checkpoint
  ↓
Pretraining
  ↓
Evaluation
  ↓
Specialization / alignment / optimization
```

Academy uses the same real execution engine as Studio.

It must not simulate training state.

Displayed loss, perplexity, tokens seen, learning rate, gradient norm, throughput, memory usage, checkpoints, evaluation results, and lineage must come from actual model development state.

Academy may expose more explanation, guided defaults, visual teaching aids, and small-model presets such as `zero-8m` and `zero-25m`.

Academy projects remain normal Frontierwright projects and may later be opened through Studio without conversion or lineage loss.

### Frontierwright Lab

Purpose:

Develop private/internal models using controlled private infrastructure and internal evidence.

Primary users:

- companies;
- research labs;
- university labs;
- internal foundation-model teams;
- teams with private checkpoints, private datasets, trainer configs, logs, manifests, and custom infrastructure.

Typical environment:

```text
Private checkpoints
Private tokenizer
Private datasets
Internal trainer
Internal evaluator
Internal storage
Cluster / Slurm / private servers
        ↓
Frontierwright Lab
```

Lab must support private adapters without requiring public model weights, public datasets, public APIs, or external uploads.

Lab may operate with external network access disabled by policy.

Private/internal integrations should normally be implemented as private plugins/adapters rather than merged into the public core repository.

## 3. Shared Frontierwright Core

All editions share the same core lifecycle:

```text
Model State
    ↓
Recipe / Intervention
    ↓
Plan
    ↓
Calibration
    ↓
Durable Execution
    ↓
Sealed Artifact
    ↓
Pending Candidate
    ↓
Evaluation
    ↓
Compare
    ↓
Explicit Promote / Reject
    ↓
New Current Model State
```

The common core owns:

- model identity;
- lineage;
- recipes;
- intervention history;
- project registry;
- datasets and prepared-data identity;
- tokenizer artifacts;
- plans;
- calibration;
- permissions and budgets;
- durable runs;
- execution receipts;
- sealed artifacts;
- candidates;
- champions;
- evaluations;
- build targets/floors;
- comparison;
- promotion/rejection;
- deployment profiles;
- history.

## 4. Editions are profiles, not model identities

A project must not become structurally incompatible because it was created in a particular edition.

Do not create separate model types such as:

```text
AcademyModel
StudioModel
LabModel
```

Do not create independent incompatible project formats for each edition.

Edition is a product/UX/policy profile.

Conceptually:

```text
Project
  edition_profile: STUDIO | ACADEMY | LAB
```

The project, model identity, lineage, artifacts, recipes, and evaluation evidence remain edition-independent.

A model born in Academy can continue in Studio.

A model developed in Studio can be moved into a Lab environment if its storage, security, and adapter requirements are satisfied.

Edition transitions must not rewrite model history.

## 5. One codebase

The following architecture is prohibited:

```text
frontierwright-studio/
frontierwright-academy/
frontierwright-lab/
```

as three copied or independently evolving implementations.

The required architecture is one codebase:

```text
frontierwright/
├── core/
├── recipes/
├── execution/
├── evaluation/
├── artifacts/
├── interventions/
├── adapters/
├── editions/
│   ├── studio
│   ├── academy
│   └── lab
└── tui/
```

Exact module names may evolve, but the architectural rule does not:

**one core, three edition profiles.**

## 6. Intervention architecture

The existing five v1 paths remain valid built-in implementations:

- from-scratch pretraining;
- continued pretraining;
- full SFT;
- LoRA SFT;
- QLoRA SFT.

They are not the permanent top-level ontology.

Long-term Frontierwright development is organized around extensible intervention families:

```text
BIRTH
LEARN
SPECIALIZE
ALIGN
EVOLVE
OPTIMIZE
EVALUATE
OPERATE
```

Examples:

### BIRTH
- tokenizer training/import;
- architecture selection;
- initialization;
- root checkpoint materialization.

### LEARN
- pretraining;
- continued pretraining;
- domain-adaptive pretraining;
- curriculum and mixture training.

### SPECIALIZE
- full SFT;
- adapters;
- prefix tuning;
- LoRA;
- QLoRA;
- DoRA;
- future PEFT methods.

### ALIGN
- instruction tuning;
- preference optimization;
- DPO;
- reward-model/RLHF backends;
- future alignment methods.

### EVOLVE
- merging;
- distillation;
- continual learning;
- architecture/tokenizer migration where supported.

### OPTIMIZE
- quantization;
- pruning;
- deployment optimization.

### EVALUATE
- capability;
- regression;
- robustness;
- efficiency;
- custom evaluation packs.

### OPERATE
- export;
- inference profile;
- deployment;
- observation;
- feedback/data collection;
- next intervention.

New training/tuning research should normally enter Frontierwright as an intervention plugin or backend capability rather than requiring a product rewrite.

## 7. Academy-specific policy

Academy is not a toy simulator.

All model-development operations must remain real.

Education-specific behavior belongs in:

- explanations;
- guided defaults;
- teaching views;
- progressive disclosure;
- safe small-model presets;
- curriculum-style UI;
- inspection tools.

It must not invent fake:

- XP;
- capability growth;
- training progress;
- benchmark values;
- resources;
- model states.

Academy may explain real quantities more aggressively than Studio or Lab.

## 8. Lab-specific policy

Lab extends the same core with controlled integrations.

Expected private adapter families include:

- model source adapters;
- dataset/storage adapters;
- trainer adapters;
- executor adapters;
- evaluator adapters;
- cluster/Slurm adapters;
- internal artifact stores.

Lab must support local/private-first analysis.

Private data must not be uploaded to an external service merely to classify, plan, calibrate, evaluate, or recommend an intervention.

Future Lab policy may support data classifications such as:

```text
PUBLIC
INTERNAL
CONFIDENTIAL
PRIVATE
```

Executor/data-policy incompatibility should result in a factual hard lock.

Example:

```text
LOCKED
Private dataset cannot be sent to the selected executor.
```

## 9. Distribution

The public repository remains a single repository.

Recommended package/distribution model:

```text
frontierwright
frontierwright[academy]
private lab adapter packages
```

Binary/installable releases may present three branded installers or launch profiles while still sharing the same underlying core.

Examples:

```text
Frontierwright Studio
Frontierwright Academy
Frontierwright Lab
```

Private company/lab integrations may live in separately controlled repositories/packages.

## 10. CLI and machine interface

Human entrypoint may support:

```text
frontierwright play
frontierwright play --edition studio
frontierwright play --edition academy
frontierwright play --edition lab
```

Initial interactive setup may offer:

```text
Studio
Bring a model.

Academy
Birth a model.

Lab
Bring your infrastructure.
```

Machine/agent interfaces must remain stable and edition-independent wherever practical.

AI agents must not need to drive separate TUI implementations for each edition.

## 11. Product continuity

Frontierwright must remain useful after initial training or tuning.

The intended lifecycle is not:

```text
Create → Train → Tune → Done
```

It is:

```text
Birth / Import
  ↓
Learn
  ↓
Measure
  ↓
Specialize
  ↓
Measure
  ↓
Align
  ↓
Measure
  ↓
Deploy / Use
  ↓
Observe
  ↓
Collect new evidence/data
  ↓
New intervention
  ↓
Candidate
  ↓
Evaluate / Compare
  ↓
Champion
  ↓
Repeat
```

There is no required terminal "Done" state.

The champion is the current stable model state, not the end of the product lifecycle.

## 12. Branding

Frozen official edition names:

- **Frontierwright Studio**
- **Frontierwright Academy**
- **Frontierwright Lab**

Brand roles:

**Frontierwright Studio**
> Develop models you control.

**Frontierwright Academy**
> Build an AI model from birth.

**Frontierwright Lab**
> Develop private models inside controlled infrastructure.

The parent brand remains **Frontierwright**.

## 13. Core product definition

Frontierwright is a lifelong model-development environment.

It allows users to birth or import a model, teach it, specialize it, align it, evaluate it, evolve descendants, optimize it, deploy it, observe it, and continue developing it while preserving real reproducible model state and lineage.

The game-like character metaphor remains a human UX layer over real ML state.

## 14. Non-negotiable invariants

1. One shared core; no three-way code fork.
2. Edition is a profile, not model identity.
3. Academy uses real training, not simulation.
4. Lab can operate entirely inside private infrastructure.
5. Studio remains the general existing-model workflow.
6. Academy-created models can continue through Studio without migration loss.
7. All editions share recipe, plan, calibration, durable execution, sealed candidate, evaluation, and explicit promotion semantics.
8. The five initial training paths remain supported but must not limit future intervention families.
9. New training/tuning techniques should be pluggable.
10. Frontierwright remains useful throughout the model lifecycle, not only during initial training/tuning.

## 15. Precedence

If older planning documents imply:

- separate codebases for the editions;
- Academy as a simulated educational mode;
- Lab requiring public access;
- edition-specific incompatible model/project identities;
- a fixed permanent list of only five training methods;
- or a lifecycle that ends after training/tuning;

this document overrides them.
