# Frontierwright — Product Decisions Override

Date: 2026-09-22
Status: CANONICAL PRODUCT DECISIONS
Precedence: This document overrides earlier train-for-llm / Frontierwright product notes wherever they conflict.

## 0. Product boundary

Frontierwright is an independent product.

It is a game-like, keyboard-first LLM development environment for models the user actually controls: a bundled zero model, a local/open trainable model, or an internal model being developed by a professional lab/company.

API-only hosted models whose trainable state is not controlled by the user are out of scope as trainable Frontierwright characters.

The product core is:

- model = character;
- real evaluations = stats;
- desired specialization = build;
- real pretraining/fine-tuning routes = development paths;
- actual compute/data/time = resources;
- trained descendant = candidate;
- accepted current model = champion;
- real lineage = history.

No fake game mechanic may alter evaluation or training.

## 1. Entry-origin-aware build UX

The Build UI MUST change according to the origin and maturity of the inserted model.

### 1.1 Bundled zero model / from scratch

A random-init model has no meaningful capability stats yet.

Do NOT ask the user to set precise target stats before they exist.

Start with build intent and relative priorities such as:

- Balanced;
- Code-focused;
- Math-focused;
- General-focused;
- Custom priority allocation.

After meaningful evaluations exist, unlock exact targets/floors.

### 1.2 Imported local model

First:

1. inspect;
2. evaluate;
3. produce current character sheet;
4. then allow target/floor editing relative to measured current stats.

### 1.3 In-development internal/frontier model

Ingest:

- checkpoint/model state;
- architecture/config;
- tokenizer;
- trainer configuration;
- run logs;
- dataset manifests;
- previous intervention records;
- lineage/checkpoints;
- resource records;

when supplied.

Internal/private model formats must be supported through adapters so a professional lab can use Frontierwright entirely inside its own environment.

## 2. Keyboard-only roguelike TUI + separate AI interface

Human UX:

`frontierwright play` launches a full-screen terminal UI with a roguelike feel.

Mandatory keyboard interaction:

- Arrow keys or hjkl: move;
- Enter: select/confirm;
- Esc: back;
- number keys: direct visible action selection where useful;
- ?: contextual help;
- no mouse required.

The TUI should feel like a persistent character/status game screen, not a sequence of shell prompts.

Preferred implementation direction: Textual for the full-screen TUI, Rich for reusable rendering.

The TUI MUST contain no unique business logic.

AI/machine UX is separate and first-class:

- stable CLI subcommands;
- `--json`;
- `--non-interactive`;
- `--yes`;
- stable exit codes;
- versioned schemas;
- idempotency for expensive actions.

AI agents MUST NOT screen-scrape the TUI or simulate keyboard navigation.

## 3. Current state, not historical-best level

The old rule "level = cumulative new personal-best stat points" is REJECTED.

The visible model state must describe the current accepted champion.

Canonical rules:

- current stats come from the current champion;
- candidate comparisons are current champion vs candidate;
- historical bests are history/milestones only;
- a historical peak never makes the current model appear stronger;
- if the accepted champion becomes weaker on an axis, the displayed current state must show that loss;
- run count, time spent, money spent, and old personal bests do not inflate current capability.

A single overall Rating/Power value is optional until its aggregation formula is justified and frozen.

Individual capability stats are authoritative.

If a single summary is implemented, it MUST derive only from the current champion's current frozen-scale stats and may go down when the current model gets worse.

Do not call historical achievement a "level".

## 4. Recommendation requires trustworthy history

Frontierwright may provide a strong "Recommended next path" only when the model's relevant development history is sufficiently known.

History confidence enum:

```text
COMPLETE
VERIFIED
PARTIAL
UNKNOWN
```

Definitions:

- COMPLETE: Frontierwright managed the lineage end-to-end.
- VERIFIED: imported logs/configs/manifests/lineage evidence establish the relevant prior interventions.
- PARTIAL: some history is known but important gaps remain.
- UNKNOWN: only the current checkpoint/model state is trustworthy.

Recommendation policy:

- COMPLETE / VERIFIED -> history-aware path recommendation allowed.
- PARTIAL / UNKNOWN -> show feasible paths, costs, resource requirements, and facts, but do NOT claim one is the history-aware recommended next intervention.

Reason:

Weights alone cannot reliably prove all previous training operations. In particular, a merged LoRA, prior data mixture, optimizer history, or other interventions may be impossible to reconstruct with certainty from final weights alone.

If the user supplies missing logs/manifests later, history confidence may be upgraded.

## 5. Data policy: user's data first

Data acquisition priority:

1. user/lab-owned datasets already available;
2. explicitly connected internal storage/datasets;
3. public datasets suggested by Frontierwright only when suitable user data is unavailable or the user asks for help.

For professional labs:

- assume they may already possess proprietary/internal datasets;
- never upload private data to an external discovery service merely to classify it;
- local/private analysis first;
- external discovery is opt-in.

For hobby/education users:

Frontierwright should help discover suitable public data when needed.

Discovery provider architecture may search curated/public dataset registries and return:

- license;
- provenance;
- size/token estimate;
- domain;
- language;
- format;
- known quality/duplication metadata.

Download/preprocessing requires explicit approval.

Preparation may include:

- download;
- normalization;
- deduplication;
- filtering;
- train/validation split;
- tokenization;
- manifest/fingerprint creation.

Public data is assistance, not a default replacement for user-owned data.

## 6. Supported model ownership boundary

v1 primary trainable targets:

- Frontierwright zero models;
- local Hugging Face-style trainable checkpoints;
- local/open trainable models;
- private/internal models under active development through adapters.

Architecture rule:

Frontierwright must not assume the model is public.

A company such as a frontier lab could run Frontierwright fully inside its own environment using private adapters without exposing model weights, datasets, or trainer internals.

API-only models are excluded as trainable characters when the user cannot control the underlying trainable state.

GGUF or other inference-oriented artifacts may be inspectable/profile-able where practical, but training routes require access to a compatible trainable representation.

## 7. Remaining previously proposed defaults

Unless contradicted by a later explicit product decision, use these defaults:

### Capability axes

Core v1:

- General
- Reasoning
- Math
- Coding

Additional axes use versioned Stat Packs, e.g.:

- Science
- Multilingual
- Reliability
- domain-specific/user-defined packs

A from-scratch model may use readiness/language-acquisition probes before full core stats are meaningful; those probes are not automatically permanent global core stats.

### Candidate promotion

Training always creates a candidate.

Default promotion is explicit user approval.

Automatic promotion is opt-in and requires explicit user rules.

### Compute environments

Design resource/executor adapters for:

- local machine;
- user-owned remote/server resources;
- professional lab/cluster resources.

Validate local execution most deeply first.

Avoid hard-coding the product to one cloud provider.

### Language/UI tone

- English is the canonical machine/schema language.
- Human UI architecture is i18n-ready from the start.
- Korean is a first-class human UI language.
- Tone: game-like and intuitive, but not childish/fantasy-heavy.

## 8. Implementation consequence

The primary human implementation loop is now:

```text
launch keyboard TUI
-> choose/create/import character
-> inspect current state
-> establish history confidence
-> evaluate current stats
-> choose/edit build appropriate to model maturity
-> inspect resources and user's existing data
-> show feasible development paths
-> recommend one only if history is COMPLETE/VERIFIED
-> dry-run
-> train candidate
-> evaluate candidate
-> compare current champion vs candidate
-> user promotes/rejects
-> current character sheet updates
-> history records what actually happened
```

This loop has higher priority than generic experiment-platform features.

## 9. License — FINAL

Frontierwright will be released under the **Apache License 2.0**.

This decision is final for v1 unless the user explicitly changes it later.

Release requirements:

- include the standard Apache-2.0 `LICENSE` text at repository root;
- include a `NOTICE` file and preserve notices where Apache-2.0 requires it;
- retain third-party license/notice obligations for bundled dependencies or redistributed assets;
- do not describe Apache-2.0 as preventing forks or commercial use;
- use the Frontierwright name/branding policy separately from the source-code license if brand protection is desired.

Do not substitute MIT, AGPL, or a custom license during implementation or release preparation without an explicit later product decision.
