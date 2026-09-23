# Frontierwright

Frontierwright is a keyboard-first, game-like LLM development environment for models the user actually controls.

> **Build your LLM like a character: inspect its real stats, choose the build you want, see which development paths are feasible, train real candidates, compare them with the current champion, and keep the model's actual development history.**

The local workspace/folder name and public product name are both **Frontierwright**.

## Product principles

- The model is the character.
- Stats come from real evaluation.
- The visible state describes the current accepted model, not a historical peak.
- Builds express what the user wants the model to become.
- Training paths are real pretraining/fine-tuning interventions.
- Resources are real compute, data, memory, time, and storage.
- Training creates candidates; the user accepts or rejects them.
- History is real lineage and intervention history.
- Human UX is a keyboard-only roguelike-style terminal UI.
- AI/automation uses a separate stable CLI/JSON interface.
- User/lab-owned data is preferred before public data discovery.
- Strong path recommendations require sufficiently trustworthy model history.
- API-only hosted models without user-controlled trainable state are not trainable Frontierwright characters.
- Frontierwright is an independent product. It has no dependency on a separate research project.

## Official editions

Frontierwright is one shared core with three official product profiles:

- **Frontierwright Studio** — develop existing user-controlled models.
- **Frontierwright Academy** — build a real model from birth with education-oriented guidance.
- **Frontierwright Lab** — develop private/internal models inside controlled infrastructure.

Edition is a UX/policy profile, not model identity. Projects can move between edition profiles without rewriting lineage.

## Canonical product documents

Read these in this order:

1. `product/PRODUCT_DECISIONS_OVERRIDE_20260922.md` — **highest-precedence product decisions**
2. `product/EDITION_ARCHITECTURE_DECISION_20260923.md` — **frozen Studio / Academy / Lab edition architecture and lifecycle boundary**
3. `product/V1_IMPLEMENTATION_BLUEPRINT_20260922.md` — v1 implementation contract

If an older planning document conflicts with any canonical document above, the canonical documents win within their stated scope.

A supporting engineering note, `product/MODEL_DEVELOPMENT_AS_RECIPE_20260923.md`, explains why Frontierwright treats a checkpoint as the current result of a reproducible training recipe rather than as a set of cleanly swappable capability modules.

Superseded brainstorming documents are kept only in the local development archive and are intentionally excluded from the public repository so they cannot be mistaken for the current product contract.

## Primary human UX

```bash
frontierwright play
```

This launches the keyboard-first full-screen TUI.

The interaction must be usable without a mouse and should feel closer to a clean roguelike character/status interface than a sequence of shell prompts.

## AI / automation UX

AI agents must not operate the TUI.

They use stable non-interactive commands and schemas:

```text
--json
--non-interactive
--yes
stable exit codes
versioned schemas
idempotency for expensive actions
```

## Current development state

The first executable v1 vertical slice is implemented.

Implemented now:

- installable Python package and `frontierwright` entrypoint;
- Apache-2.0 `LICENSE` and `NOTICE`;
- immutable domain primitives for model origin, capability evidence, candidates/champion, build mode, and history confidence;
- authoritative `.frontierwright/registry.sqlite` project state plus human-readable `project.toml`;
- automatic registry migration through schema v12, including persisted edition profiles, zero-model birth provenance, and data-preparation recipes;
- current-champion semantics: historical model states never inflate the current displayed stats;
- local Hugging Face model import with content-based SHA-256 fingerprinting of config/tokenizer/weight artifacts;
- GGUF inspection/import as explicitly non-trainable rather than pretending an inference artifact is trainable;
- history evidence states `UNKNOWN / PARTIAL / VERIFIED / COMPLETE`;
- hash-checked `frontierwright-lineage.json` verification for imported model history;
- recommendation gating: only `COMPLETE / VERIFIED` history is eligible for a strong history-aware recommendation;
- actual local CPU/RAM/disk/NVIDIA GPU detection with `DETECTED` provenance;
- detected resource profiles persisted in SQLite and shown through CLI/JSON/TUI;
- bf16/fp16 kept `UNKNOWN` until backend-specific capability calibration instead of hardware-name guessing;
- raw benchmark `EvaluationReceipt` ingestion with exact model-id/fingerprint matching;
- immutable frozen capability-scale manifests with content hash and explicit task/version/weight/anchor mapping;
- capability profiles persisted separately from model weights and backed by exact raw measurements;
- missing frozen tasks leave an axis unmeasured rather than inventing a number;
- current profile replacement semantics: a newer lower current score is displayed lower instead of preserving a historical peak;
- origin-aware Build persistence:
  - unmeasured Frontierwright zero model -> `INTENT`;
  - unmeasured inserted model -> `NOT_READY`;
  - measured model -> `TARGETS_FLOORS`;
- persisted build archetypes/relative priorities plus numeric targets/floors with validation;
- user/lab-first local dataset inventory with content fingerprints, role, license/domain/language metadata, and optional token count;
- reproducible DataPreparationPlugin contract with discoverable recipe registry;
- built-in byte-preserving managed snapshot recipe via `frontierwright data prepare`, preserving source provenance while freezing exact training-input bytes under Frontierwright state;
- prepared datasets retain source-dataset ID plus recipe ID/hash, and training plans pin the recipe hash in addition to dataset bytes;
- extensible intervention taxonomy spanning `BIRTH / LEARN / SPECIALIZE / ALIGN / EVOLVE / OPTIMIZE / EVALUATE / OPERATE`;
- the five v1 training paths are registered built-in interventions rather than the permanent top-level ontology;
- factual path prerequisite evaluation using current model trainability, local data inventory, resources, birth state, and history confidence;
- Academy zero-model birth via `frontierwright birth zero`, with deterministic `zero-8m` / `zero-25m` root checkpoint materialization, exact fingerprinting, runtime provenance, and idempotent repeated birth requests;
- from-scratch pretraining now requires and pins a materialized zero-model birth root instead of silently reinitializing weights;
- hard-missing prerequisites show `LOCKED`; calibrated accepted plans can project `READY` only after pinned model/data/backend/resource checks;
- immutable `TrainingPlan` identity with pinned model fingerprint, dataset fingerprint, backend spec hash, resource profile, config, permission, and hard budgets;
- representative calibration receipts with step time, throughput, peak memory, projected storage, and projected wall time;
- idempotent durable local run attempts with worker/process identity, liveness reconciliation, durable executor results, explicit rerun semantics, and repairable run-receipt export;
- built-in PyTorch reference backend with real decoder-only `zero-8m` and `zero-25m` from-scratch pretraining/calibration;
- successful training output is copied into a Frontierwright-managed sealed artifact before candidate registration;
- sealed artifact manifests bind the run, plan, dataset, dataset-preparation recipe, backend, effective config, model fingerprint, and exact file hashes;
- evaluation, comparison, and promotion revalidate managed candidate bytes; artifact tampering blocks promotion even with an unmeasured override;
- training creates a `PENDING` candidate and never silently changes the champion;
- candidate compare/promote/reject surfaces preserve explicit human ownership of champion selection;
- stable JSON/non-interactive machine surfaces for project/model/birth/resource/build/stats/data/path/plan/run/candidate operations;
- mandatory Textual keyboard TUI shell with CHARACTER / BUILD / PATHS / RESOURCES / DATA / HISTORY / CANDIDATES;
- English/Korean human-string structure;
- automated domain, migration, evaluation, execution recovery/idempotency, artifact integrity, candidate, model fingerprint/history, resource, data, path, build, CLI JSON, lineage, and TUI keyboard tests.

Not implemented yet:

- an official shipped Frontierwright Capability v1 benchmark/task/anchor bundle and benchmark runner;
- production built-in training backends for all five paths beyond the narrow from-scratch reference backend and structured external command adapter;
- full execution-time enforcement/accounting for every hard-budget dimension across calibration and repeated attempts;
- public dataset discovery/download adapters and transformative recipes such as normalize/dedupe/tokenize/mixture;
- remote/server/Slurm executor implementations;
- full candidate evaluation-eligibility policy and build-floor enforcement transaction.

Those surfaces remain explicitly `NOT_READY` / `UNKNOWN` rather than using fake data.

### Development quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/frontierwright project init . --name NOVA --origin ZERO --edition ACADEMY
.venv/bin/frontierwright birth zero --path . --preset zero-8m --seed 42 --python /path/to/training-python
.venv/bin/frontierwright project edition --path . --set STUDIO --json --non-interactive --yes
.venv/bin/frontierwright status --json --non-interactive --yes
.venv/bin/frontierwright play
```

Verification:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src/frontierwright
.venv/bin/python -m build
```

## License

Frontierwright is released under the **Apache License 2.0**.

The repository includes the standard Apache-2.0 `LICENSE` text and a `NOTICE` file. Third-party notice obligations must be reviewed as distributable dependencies/assets are added.
