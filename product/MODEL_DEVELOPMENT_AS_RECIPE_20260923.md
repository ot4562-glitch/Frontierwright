# Model Development as Recipe, Not Assembly

Date: 2026-09-23

## Core observation

A trained LLM is closer to a cooked or fermented result than to a robot assembled from cleanly separable parts.

Its architecture and tensors can be inspected, and activations, features, ablations, pruning, and mechanistic-interpretability probes can reveal partial structure. But useful capabilities are generally distributed across many parameters and representations. There is usually no reliable post-hoc decomposition such as "these weights are Python" or "this module is economic reasoning."

## Development implication

Do not frame model improvement primarily as disassembling a finished model and swapping good parts.

A stronger framing is:

1. Evaluate the current trained model.
2. Identify behavioral failures, inefficiencies, and capability gaps.
3. Form hypotheses about the training conditions that produced them.
4. Change data, objectives, curriculum, optimization, architecture, post-training, or inference conditions.
5. Train or continue-train a candidate.
6. Measure whether those interventions produced a better model.

The finished model is evidence about the training process, not a perfectly reversible blueprint of that process.

## Frontierwright implication

Frontierwright should treat a model as a state produced by an experimental recipe. Its core unit of improvement should therefore include the recipe and intervention history, not only the final checkpoint.

Checkpoint-level analysis remains useful, but primarily as measurement and evidence for deciding what training-process intervention to try next.

## Caution

This is an engineering heuristic, not a claim that trained models are unanalyzable. Interpretability methods can recover meaningful internal structure; the point is that the structure is distributed and only partially recoverable, so model development should not assume clean component-level reversibility.
