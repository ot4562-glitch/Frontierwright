# Frontierwright RC3 — Measurement Completeness, Edition UX, and Non-PTY Play

Date: 2026-09-25

## Purpose

RC3 does not redefine Frontierwright as a game. It tightens the product around the real
goal: continuously develop a model that fits one user's or lab's workload, resources,
privacy boundary, and evidence requirements.

The rc3 change is deliberately narrow enough to verify:

1. missing machine evidence becomes a diagnosable measurement problem rather than a bare
   UNKNOWN or a false claim of hardware absence;
2. Academy, Studio, and Lab expose different first-run information architecture while
   sharing the same model lineage and raw evidence;
3. non-PTY environments such as CodexPro can exercise the actual Textual human interface
   deterministically rather than falling back to source inspection or pretending CLI JSON
   is equivalent to human play;
4. public onboarding makes the correct edition and first action discoverable before the
   engineering inventory.

## Measurement completeness contract

UNKNOWN remains valid internally when evidence is genuinely absent, but it is not a
successful end-user state by itself.

A user-facing unresolved measurement should carry:

- the probe or estimand that is missing;
- a status such as MEASUREMENT NEEDED or UNSUPPORTED HERE;
- a reason code;
- a short explanation of what was and was not established;
- a concrete next action when one exists;
- the timestamp/conditions of point-in-time resource evidence.

GPU detection is an important example. Failure to find or execute nvidia-smi does not
prove physical GPU absence. rc3 distinguishes missing probe tools, failed probes,
unparseable responses, no visible device rows, and measured devices. AMD/ROCm evidence is
handled separately. Frontierwright must not fill unmeasured VRAM or precision support from
model names or hardware-name heuristics.

System availability and model-specific headroom remain separate estimands.

## Edition experience contract

The editions are profiles over one evidence core, not code forks.

### Academy

Goal: understand real model development by changing and measuring real state.

Default navigation follows the learning sequence more closely:

CHARACTER → DATA → RESOURCES → BUILD → PATHS → WORKLOAD → CANDIDATES → HISTORY

Academy translates registry language into learner language where possible. For example,
a complete project-history record and an unmeasured capability state are explained as
separate facts instead of displaying apparently contradictory raw enums.

The first-run help explicitly teaches a sequence and data/tokenizer/model-birth
distinctions. PRIVATE and tokenizer-dependent token-count semantics are explained in the
Data view.

### Studio

Goal: fit an existing model to the user's own machine and workload.

Default navigation prioritizes workload and machine fit:

CHARACTER → WORKLOAD → RESOURCES → BUILD → PATHS → DATA → CANDIDATES → HISTORY

The first-run question is whether the existing Champion is already a good fit and which
measured trade-off justifies an intervention.

### Lab

Goal: run controlled private model-development experiments with frontier-oriented
measurement rigor.

Default navigation prioritizes workload contract, private data, and resources:

CHARACTER → WORKLOAD → DATA → RESOURCES → PATHS → BUILD → CANDIDATES → HISTORY

Lab first-run guidance starts with controlled infrastructure and pins model/data/workload
acceptance/evaluator-or-reward/budget identities before an experiment is trusted.

## Scripted TUI QA contract

Normal AI automation continues to use stable CLI/JSON operations and must not scrape the
TUI.

Human-interface QA is different: it needs to prove that the keyboard UI itself works.
The command frontierwright play --script PLAY.json --json therefore runs the same
FrontierwrightApp with Textual's test pilot, without requiring a host PTY.

A play script can:

- press real TUI keys;
- set a visible Textual Input by widget ID;
- pause for UI work;
- capture after every step:
  - screen class;
  - active tab;
  - focused widget;
  - visible static text;
  - visible input values.

This is a QA interface, not a second product state engine. Consequential UI actions still
call the shared service layer.

The virtual terminal is bounded and deterministic. Invalid scripts fail before silently
skipping an action.

## Public onboarding contract

The README shows Studio, Academy, and Lab routes before the detailed engineering
inventory. The documented Studio import command must match the real root frontierwright
import command.

Human project initialization prints the next frontierwright play --path action.
Core root commands have purpose-oriented help text.

## Non-goals

RC3 does not claim:

- physical GPU absence when a probe is unavailable;
- production-quality generic ROCm GPU parsing;
- arbitrary-Hugging-Face production training;
- generic multi-node distributed training;
- production external RL framework integration;
- empirical proof that Academy improves human learning outcomes;
- that scripted TUI QA replaces real terminal visual review.

Those remain separately testable follow-up work.
