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

## Canonical product documents

Read these in this order:

1. `product/PRODUCT_DECISIONS_OVERRIDE_20260922.md` — **highest-precedence product decisions**
2. `product/V1_IMPLEMENTATION_BLUEPRINT_20260922.md` — v1 implementation contract

If an older planning document conflicts with either file above, the two canonical documents win.

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
- automatic registry migration through schema v9;
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
- v1 training-path plugin protocol for From-scratch pretraining, Continued pretraining, Full SFT, LoRA SFT, and QLoRA SFT;
- factual path prerequisite evaluation using current model trainability, local data inventory, resources, and history confidence;
- hard-missing prerequisites show `LOCKED`; calibrated accepted plans can project `READY` only after pinned model/data/backend/resource checks;
- immutable `TrainingPlan` identity with pinned model fingerprint, dataset fingerprint, backend spec hash, resource profile, config, permission, and hard budgets;
- representative calibration receipts with step time, throughput, peak memory, projected storage, and projected wall time;
- idempotent durable local run attempts with worker/process identity, liveness reconciliation, durable executor results, explicit rerun semantics, and repairable run-receipt export;
- built-in PyTorch reference backend with real decoder-only `zero-8m` and `zero-25m` from-scratch pretraining/calibration;
- successful training output is copied into a Frontierwright-managed sealed artifact before candidate registration;
- sealed artifact manifests bind the run, plan, dataset, backend, effective config, model fingerprint, and exact file hashes;
- evaluation, comparison, and promotion revalidate managed candidate bytes; artifact tampering blocks promotion even with an unmeasured override;
- training creates a `PENDING` candidate and never silently changes the champion;
- candidate compare/promote/reject surfaces preserve explicit human ownership of champion selection;
- stable JSON/non-interactive machine surfaces for project/model/resource/build/stats/data/path/plan/run/candidate operations;
- mandatory Textual keyboard TUI shell with CHARACTER / BUILD / PATHS / RESOURCES / DATA / HISTORY / CANDIDATES;
- English/Korean human-string structure;
- automated domain, migration, evaluation, execution recovery/idempotency, artifact integrity, candidate, model fingerprint/history, resource, data, path, build, CLI JSON, lineage, and TUI keyboard tests.

Not implemented yet:

- an official shipped Frontierwright Capability v1 benchmark/task/anchor bundle and benchmark runner;
- production built-in training backends for all five paths beyond the narrow from-scratch reference backend and structured external command adapter;
- full execution-time enforcement/accounting for every hard-budget dimension across calibration and repeated attempts;
- explicit zero-model birth/materialization as a root checkpoint before first training;
- prepared-dataset recipe artifacts and public dataset discovery/download adapters;
- remote/server/Slurm executor implementations;
- full candidate evaluation-eligibility policy and build-floor enforcement transaction.

Those surfaces remain explicitly `NOT_READY` / `UNKNOWN` rather than using fake data.

### Development quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/frontierwright project init . --name NOVA --origin ZERO
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
