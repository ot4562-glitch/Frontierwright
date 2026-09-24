# Frontierwright Implementation Status

Updated: 2026-09-24

This document keeps the detailed engineering inventory out of the public README.
The README explains the product and fastest path to trying it; this file records the
implemented surface and known gaps.

## Detailed v1 / RC implementation inventory

Implemented now:

- installable Python package and `frontierwright` entrypoint;
- Apache-2.0 `LICENSE` and `NOTICE`;
- immutable domain primitives for model origin, capability evidence, candidates/champion, build mode, and history confidence;
- authoritative `.frontierwright/registry.sqlite` project state plus human-readable `project.toml`;
- automatic registry migration through schema v20, including persisted edition profiles, zero-model birth provenance, trainable tokenizer artifacts, data-preparation recipes, intervention identity backfill, frozen-scale Build binding, dataset/backend data-boundary policy, Lab adapter manifests, durable run-usage receipts, preference-dataset role support, and multi-parent model lineage edges;
- current-champion semantics: historical model states never inflate the current displayed stats;
- local Hugging Face model import with content-based SHA-256 fingerprinting of config/tokenizer/weight artifacts;
- GGUF inspection/import as explicitly non-trainable rather than pretending an inference artifact is trainable;
- history evidence states `UNKNOWN / PARTIAL / VERIFIED / COMPLETE`;
- hash-checked `frontierwright-lineage.json` verification for imported model history;
- native multi-parent lineage graph with `DERIVED_FROM / MERGED_FROM / DISTILLED_FROM / TRANSFORMED_FROM` relations while retaining the primary-parent compatibility field;
- recommendation gating: only `COMPLETE / VERIFIED` history is eligible for a strong history-aware recommendation;
- actual local CPU/RAM/disk/NVIDIA GPU detection with `DETECTED` provenance;
- detected resource profiles persisted in SQLite and shown through CLI/JSON/TUI;
- versioned Workload Profiles persist explicit languages/domains/task mixture, context demand, latency/throughput constraints, privacy boundary, hard capability floors, and optional explicit decision-utility weights/scales;
- Workload Fit evaluates each declared requirement as PASS / FAIL / UNKNOWN from exact current-model evidence and refuses to treat generic capability as proof of user-domain coverage;
- an edition-aware evidence planner maps UNKNOWN constraints to the next required measurement and measured FAIL constraints to bounded, falsifiable experiments; it exposes the same logic through `frontierwright workload next` and the TUI while deliberately returning no predicted performance gain;
- Candidate comparison includes a raw Pareto surface across comparable capability, serving latency/TTFT/TPOT/throughput, measured VRAM/RSS, and artifact-storage evidence; unknown or mismatched runtime conditions remain unknown;
- optional user utility is computed only when the user supplies both a positive weight and normalization scale for every selected measured metric; missing evidence yields INCOMPLETE, raw Pareto evidence remains visible, and utility never auto-promotes a Candidate;
- bf16/fp16 kept `UNKNOWN` until backend-specific capability calibration instead of hardware-name guessing;
- raw benchmark `EvaluationReceipt` ingestion with exact model-id/fingerprint matching;
- strict lm-evaluation-harness result import pins exact harness revision, source hash, task/version/metric identity, standard-error metadata and metric direction while keeping external results as raw evidence until an explicit frozen scale maps them;
- Hugging Face LightEval saved-result import pins an explicit evaluator version/revision, source hash, declared task version and metric direction from `config_tasks`, skips the mixed aggregate `all` row, preserves standard-error evidence, and imports only aggregate result JSON rather than copying detail Parquet prompts/responses;
- a framework-neutral external-evaluation manifest importer requires the exact target model fingerprint plus explicit evaluator/version/task/version/metric/value/direction evidence, preserves optional sample-count/standard-error/confidence-interval metadata, and therefore supports private or third-party suites without teaching Frontierwright to guess benchmark semantics;
- workload-evaluation binding manifests connect exact stored evaluation receipt task/version/metric identities to declared workload languages/domains/tasks; stale profile hashes, wrong models, missing measurements, or name-only inference are rejected, and Workload Fit remains UNKNOWN until every declared workload dimension has explicit evidence coverage;
- vLLM serve-benchmark import pins exact runtime version and comparable workload-condition hash, imports latency/TTFT/TPOT/ITL/aggregate-throughput evidence, and deliberately refuses to invent server RSS/VRAM or per-request p50 throughput from client-only results;
- framework-neutral serving-resource manifests pin exact model fingerprint, runtime/version, execution boundary, and serving-condition hash; compatible server-side VRAM/RSS/headroom receipts are composed with client latency evidence only under exact matching conditions, while mismatches remain separate/UNKNOWN;
- immutable frozen capability-scale manifests with content hash and explicit task/version/weight/anchor mapping;
- capability profiles persisted separately from model weights and backed by exact raw measurements;
- missing frozen tasks leave an axis unmeasured rather than inventing a number;
- current profile replacement semantics: a newer lower current score is displayed lower instead of preserving a historical peak;
- origin-aware Build persistence:
  - unmeasured Frontierwright zero model -> `INTENT`;
  - unmeasured inserted model -> `NOT_READY`;
  - measured model -> `TARGETS_FLOORS`;
- persisted build archetypes/relative priorities plus numeric targets/floors with validation;
- user/lab-first local dataset inventory with content fingerprints, `PRETRAIN / SFT / PREFERENCE` roles, license/domain/language metadata, optional token count, and explicit `PUBLIC / INTERNAL / CONFIDENTIAL / PRIVATE` classification;
- local user data defaults to `PRIVATE`; prepared datasets inherit the source classification instead of silently weakening policy;
- backend specs declare a pinned data boundary: `LOCAL_MACHINE / CONTROLLED_PRIVATE / EXTERNAL / UNKNOWN`; non-public data is hard-blocked from `EXTERNAL` and `UNKNOWN` boundaries;
- plan identity pins both dataset classification and backend boundary, and re-checks classification drift before calibration/execution;
- legacy backend specs without a boundary remain hash-compatible and are interpreted as `LOCAL_MACHINE`; new explicit boundary declarations participate in the backend-spec hash;
- Lab adapter manifests declare adapter ID/version, trainer/executor/data/evaluator/artifact-store kinds, network scope, capabilities, and data boundary without storing credentials;
- `frontierwright lab adapters connect/list/disconnect` provides a stable machine/human surface for private adapter registration;
- `CONTROLLED_PRIVATE` backends must bind a connected Lab adapter; plans pin both adapter ref and immutable manifest hash, and disconnect/drift makes an existing plan stale before calibration or execution;
- adapter ref/hash evidence is propagated into plan identity, durable execution requests, and sealed candidate manifests;
- reproducible DataPreparationPlugin contract with discoverable recipe registry;
- built-in byte-preserving managed snapshot recipe via `frontierwright data prepare`, preserving source provenance while freezing exact training-input bytes under Frontierwright state;
- built-in `text-lines-v1` preparation normalizes UTF-8 text, canonicalizes line endings/Unicode, removes empty lines, and performs deterministic stable exact dedupe while recording transformation counts;
- built-in `preference-jsonl-v1` preparation validates PREFERENCE JSONL records, canonicalizes JSON/Unicode, preserves semantically meaningful field whitespace, stable-dedupes exact prompt/chosen/rejected triples, and emits managed `pairs.jsonl` with transformation evidence;
- built-in weighted text-mixture preparation composes multiple managed text corpora with exact source fingerprints/parts, conservative metadata/classification propagation, deterministic output, and a hard materialization-size limit;
- built-in `byte-shards-v1` chained preparation converts managed `corpus.txt` into deterministic uint8 byte-ID shards, records the exact training-unit count, and lets the reference backend reconstruct shard streams without inserting synthetic separators;
- prepared-source chaining is explicit per plugin: immutable prepared data is never silently reprocessed by a recipe that did not declare support;
- prepared datasets retain source-dataset ID plus recipe ID/hash, and training plans pin the recipe hash in addition to dataset bytes; `data prepare --dry-run` and `data mix --dry-run` validate recipe identity/replay without materializing output;
- extensible intervention taxonomy spanning `BIRTH / LEARN / SPECIALIZE / ALIGN / EVOLVE / OPTIMIZE / EVALUATE / OPERATE`;
- discoverable InterventionPlugin registry with explicit execution surfaces; built-in training interventions wrap their real path-assessment plugins rather than acting as display-only labels;
- `frontierwright interventions --json` exposes stable intervention ID/provider/version/family/surface metadata to automation;
- the five v1 training paths remain registered built-in interventions; post-v1 built-ins now include `frontierwright.align.dpo`, `frontierwright.evolve.distill`, the EVOLVE artifact transform `frontierwright.evolve.linear-merge`, the OPTIMIZE artifact transform `frontierwright.optimize.symmetric-int8`, and OPERATE operations for portable export and reference-model generation; the list is not the permanent top-level ontology;
- factual path prerequisite evaluation using current model trainability, local data inventory, resources, birth state, and history confidence;
- Academy tokenizer birth via `frontierwright birth tokenizer DATASET_ID`, training deterministic byte-level BPE merge rules from an exact registered PRETRAIN dataset fingerprint, publishing an immutable managed tokenizer artifact with replay semantics, and blocking dataset drift or tokenizer changes after model birth;
- Academy zero-model birth via `frontierwright birth zero`, with deterministic `zero-8m` / `zero-25m` root checkpoint materialization, exact fingerprinting, runtime provenance, idempotent repeated birth requests, and optional `--tokenizer-artifact` binding that changes the real embedding/lm-head vocabulary size while pinning tokenizer identity into birth provenance;
- from-scratch pretraining now requires and pins a materialized zero-model birth root instead of silently reinitializing weights;
- hard-missing prerequisites show `LOCKED`; calibrated accepted plans can project `READY` only after pinned model/data/backend/resource checks;
- immutable `TrainingPlan` identity with pinned intervention ID/version/family, model fingerprint, dataset/recipe fingerprint, dataset classification, backend spec hash/data boundary, resource profile, config, permission, and hard budgets;
- representative calibration receipts with step time, throughput, peak memory, projected storage, and projected wall time;
- idempotent durable local run attempts with worker/process identity, liveness reconciliation, durable executor results, explicit rerun semantics, and repairable run-receipt export;
- durable run-usage receipts record measured wall time and output bytes, plus provenance-labelled accounted GPU-hours when GPU count is available; unknown cost dimensions remain unknown instead of being fabricated;
- budget enforcement modes are explicit in plan views: run-count registry admission, wall-time timeout, accounted GPU-hour timeout, storage calibration admission plus a 100 ms local output-tree watchdog and finalization gate, and currently unavailable runtime money enforcement;
- max_storage_bytes is enforced during local command execution by a 100 ms output-tree watchdog that terminates the backend process group after an observed overrun, then rechecked from durable usage evidence and candidate bytes at finalization; this is not an OS/filesystem quota, so writes between polls may temporarily overshoot the configured byte limit;
- `max_money` keeps a plan non-ready until an executor with real runtime cost enforcement exists; a projected price alone is not treated as a hard budget;
- built-in PyTorch reference backend with real decoder-only `zero-8m` and `zero-25m` birth, tokenizer-aware from-scratch pretraining, continued pretraining, full-parameter causal SFT, merged-output LoRA SFT, merged-output QLoRA SFT, Direct Preference Optimization (DPO), and teacher-to-smaller-student knowledge distillation for Frontierwright reference-model lineages; trained descendants inherit the exact parent tokenizer artifact;
- reference QLoRA freezes the base model, packs targeted transformer projection matrices as blockwise NF4 4-bit values with float32 absmax scales, trains only LoRA parameters, records quantized-vs-full target storage evidence, and materializes a normal merged reference checkpoint after training; it does not claim bitsandbytes double quantization or paged optimizers;
- reference DPO consumes UTF-8 JSONL `{prompt, chosen, rejected}` preference pairs, keeps an exact frozen reference copy of the current checkpoint, optimizes the policy with the pairwise log-ratio objective and pinned `dpo_beta`, and runs the same objective during calibration so readiness reflects the real alignment workload;
- real RL has a distinct `RL_POLICY_OPTIMIZATION` path requiring a ROLLOUT dataset plus immutable environment/reward identities; the reference backend implements a verifier-driven REINFORCE path with actual sampled rollouts, rewards, policy-gradient updates and reward/episode receipts, while external mature RL systems remain adapter targets rather than being mislabeled or reimplemented blindly;
- reference distillation freezes the current Champion as teacher, initializes a separately pinned smaller student preset, minimizes a temperature-scaled teacher-logit KL term mixed with hard-token cross-entropy, requires the student to have fewer parameters than the teacher, calibrates the same teacher+student workload, and records `DISTILLED_FROM` lineage on the resulting PENDING candidate;
- `frontierwright evolve merge` performs deterministic linear weight merging for two compatible Frontierwright reference models, requires identical preset/tokenizer/state structure, fingerprints transform provenance into the model artifact, creates a PENDING candidate, and records both parents as weighted `MERGED_FROM` lineage edges; identical transform requests replay the existing candidate instead of recomputing;
- `frontierwright optimize quantize` performs deterministic per-tensor symmetric int8 post-training quantization of a trained Frontierwright reference Champion, preserves the full model as its parent, creates a non-trainable PENDING optimized-model candidate with `TRANSFORMED_FROM` lineage, and records exact source/quantized tensor-storage evidence; the reference evaluator can reload the quantized artifact for before/after raw evaluation, while an untrained birth root is explicitly rejected; the current reference loader dequantizes to float32 for execution, so Frontierwright claims the measured storage reduction only—not runtime RAM, latency, or throughput gains;
- `frontierwright operate export DEST` atomically creates a portable model bundle containing exact model bytes plus a path-free Frontierwright manifest with model identity/fingerprint, edition/project identity, immediate lineage, history-confidence evidence, and active capability receipt/scale identifiers; identical exports replay, conflicting destinations are never overwritten, and exported-model tampering is detected on replay; `frontierwright operate verify BUNDLE` independently rechecks manifest identity, exact file hashes/shape, model fingerprint, format, and trainability without requiring the original project, while reporting authenticity as `NOT_SIGNED` until a future signature/trust layer exists;
- `frontierwright operate generate PROMPT` runs the current Champion locally through the reference backend for both full and Frontierwright-int8 artifacts, encodes/decodes through the model-bound tokenizer, supports deterministic greedy decoding or seeded temperature sampling, returns exact generated token IDs and runtime evidence, and does not append the prompt to registry/history; the plaintext local request file exists only for the duration of the backend call and is unlinked afterward (not a secure-erasure claim);
- `frontierwright operate profile` measures steady-state reference generation after warmup, returning per-run/p50 latency, token throughput, process RSS and CUDA memory evidence; the project history stores model/config/runtime summaries but deliberately omits sampled continuation text/token IDs;
- successful training output is copied into a Frontierwright-managed sealed artifact before candidate registration;
- sealed artifact manifests bind the run, plan, intervention ID/version/family, dataset, dataset-preparation recipe, backend, effective config, model fingerprint, and exact file hashes;
- built-in evaluation-pack registry exposed through `frontierwright eval packs`; the first pack runs deterministic held-out causal-LM evaluation for Frontierwright reference models;
- `frontierwright eval run` produces raw cross-entropy/perplexity receipts pinned to exact model and dataset fingerprints, preparation-recipe evidence, evaluation config, and Python/PyTorch runtime versions;
- identical model/data/config evaluation requests replay the existing durable receipt without rerunning the evaluator, while dataset/model drift is rejected before execution;
- generated raw evaluation evidence remains separate from capability stats: Frontierwright does not invent a player-facing stat until an explicit frozen capability scale maps the evidence;
- official Frontierwright Capability v1 ships as a frozen local 64-task four-choice bundle with 16 General, 16 Reasoning, 16 Math, and 16 Coding probes; the exact task bundle and scale are hash-locked and versioned rather than silently changing under the same stat name;
- Capability v1 scores each choice by mean conditional token log-probability using the exact tokenizer bound to the model, stores raw accuracy/count/margin evidence, and maps each axis with the frozen anchors 0% -> 0, 25% chance -> 50, 50% -> 100, and 100% -> 200, so 100 remains a reference anchor rather than a cap;
- Capability v1 evaluator evidence schema v2 additionally stores every frozen item outcome and correct-answer margin; Candidate comparison reports same-item improvement/regression flips and exact two-sided McNemar evidence, while aggregate-only v1 receipts remain readable but cannot provide paired diagnostics;
- each 16-item Capability axis exposes a 95% Wilson interval, and paired p-values are diagnostic evidence only: non-significance is treated as inconclusive rather than model equivalence and never auto-promotes a Candidate;
- `frontierwright eval capability-v1` generates/replays the durable Capability v1 receipt and activates the frozen-scale character stats; the keyboard TUI Action Center exposes the same operation through the shared service layer;
- `frontierwright eval compare` evaluates champion and candidate under the same pack/data/config and reports raw deltas plus direction-normalized `improvement_delta` without choosing or promoting a winner; stored comparable raw receipts are also surfaced automatically in the normal Candidate compare view/TUI;
- deterministic evaluation replay revalidates model/data/config/evaluator identity, so a conflicting receipt cannot silently occupy a generated receipt ID;
- evaluation, comparison, and promotion revalidate managed candidate bytes; artifact tampering blocks promotion even with an unmeasured override;
- training creates a `PENDING` candidate and never silently changes the champion;
- candidate compare/promote/reject surfaces preserve explicit human ownership of champion selection;
- `frontierwright workload compare-models` compares any two registered models, including a historical stock/open baseline and an already-promoted descendant, without requiring Candidate status; it reuses measured capability/workload/serving/resource/storage/Pareto/user-utility evidence and never fills missing dimensions from parameter-count guesses;
- privacy-minimal Usage Observations bind SUCCESS/FAILURE/CORRECTED/ABSTAINED outcomes to the exact model fingerprint and active workload hash without storing prompt/response content; optional idempotency keys make connector retries safe while repeated real uses remain distinct evidence;
- real-use summaries expose task/failure-category counts and Wilson uncertainty for direct-success rates, and repeated failures/corrections feed the evidence-driven fit planner without being mislabeled as RL rewards or predicted gains;
- promotion re-checks current champion/build/evaluation state transactionally, enforces frozen-scale build floors by default, and records explicit override evidence when a user intentionally accepts an unmeasured or build-violating candidate;
- stable JSON/non-interactive machine surfaces for project/model/birth/evolve/optimize/operate/export/generation/resource/build/stats/evaluation/data/path/plan/run/candidate/Lab-adapter operations;
- mandatory Textual keyboard TUI shell with CHARACTER / BUILD / PATHS / RESOURCES / DATA / HISTORY / CANDIDATES;
- English/Korean human-string structure;
- automated domain, migration, evaluation, execution recovery/idempotency, artifact integrity, candidate, model fingerprint/history, resource, data, path, build, CLI JSON, lineage, and TUI keyboard tests.

Not implemented yet:

- broad arbitrary-Hugging-Face production adapters; the built-in QLoRA path is intentionally a narrow Frontierwright-reference implementation rather than a claim of arbitrary-architecture compatibility;
- OS/filesystem-enforced storage quotas with zero polling overshoot, real GPU-utilization telemetry beyond accounted GPU-hours, and runtime monetary metering/enforcement;
- public dataset discovery/download adapters and richer prepared tokenization/sharding formats beyond the implemented uint8 byte-ID shards;
- remote/server/Slurm executor implementations and concrete private Lab adapters;
- stronger executor-boundary attestation beyond the current adapter-declared trust contract.

Those surfaces remain explicitly `NOT_READY` / `UNKNOWN` rather than using fake data.

## Verification

The canonical candidate gate is implemented in `tools/v1_release_gate.py`.
See the root README for current development status and verification commands.

