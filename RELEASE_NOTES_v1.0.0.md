# Frontierwright v1.0.0

Frontierwright v1 is the first stable release of the keyboard-first local model-development
environment.

## Three editions

- **Studio** — import and grow a model you already control.
- **Academy** — build a real reference model from tokenizer/birth through training and evaluation.
- **Lab** — controlled private experiments with exact acceptance evidence, adapters, budgets, RL,
  and a bounded Slurm bridge.

All three share one model identity, lineage, evidence, Candidate/Champion, and execution core.

## Highlights

- Permanent project Origin with origin-relative growth stats.
- Explicit Candidate/Champion lifecycle: training never auto-promotes.
- Real built-in PyTorch paths for pretraining, continued pretraining, full SFT, LoRA, QLoRA,
  DPO, distillation, and bounded verifier-driven REINFORCE.
- Deterministic linear merge and symmetric-int8 model transformations.
- Workload profiles, persisted growth rules, exact workload coverage, and versioned acceptance.
- External lm-eval, LightEval, generic evaluation, vLLM, and serving-resource evidence import.
- Uncertainty-aware comparison and consent-triggered interleaved runtime remeasurement.
- Durable calibration/run/result/usage receipts, hard budgets, idempotent replay, and recovery.
- Controlled-private Lab adapter identities and a bounded hash-pinned Slurm submission contract.
- Edition-specific Textual interface with a shared Action Center and full CLI/JSON automation.
- Curated benchmark metadata registry without vendoring third-party benchmark datasets.

## Final release playtest

The release tree was played through the public surfaces before tagging:

- Academy: real zero-model birth → pretraining → evaluation → Candidate → Champion.
- Studio: continued pretraining, full SFT, LoRA, QLoRA, DPO, quantization, merge, external
  evaluation/serving evidence, and full TUI traversal.
- Lab: exact 95% interval acceptance PASS, complete workload coverage, private adapter lifecycle,
  Slurm host inspection, real bounded REINFORCE, Candidate comparison/rejection, and full TUI traversal.
- All 81 public command help surfaces returned valid help without tracebacks.
- Windows correctly reported real Slurm submission unavailable because it is not a POSIX submit host
  and does not have sbatch/scancel/sacct.

## Boundaries

v1 does not claim arbitrary Hugging Face architecture training, production external RL stacks,
generic distributed training, OS/network privacy attestation, or executable integrations for every
benchmark listed in the registry.

Apache-2.0.
