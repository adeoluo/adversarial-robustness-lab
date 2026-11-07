# ADR-0002: Pixel-space threat model with normalization inside the model, verified on every batch

- **Status:** Accepted
- **Date:** 2026-11-07

## Context
Robust accuracy is only meaningful relative to a precise threat model: the set of inputs
the attacker may submit. Published numbers for CIFAR-10 use the L-infinity ball of
radius 8/255 and the L2 ball of radius 0.5 around images with pixels in [0, 1]. A common
evaluation bug is to attack the standardized tensor (after subtracting the mean and
dividing by the standard deviation), which silently changes the budget per channel by a
factor of about four, or to forget the [0, 1] box. A second class of bugs comes from
attacks that return points outside their ball (a missing projection after momentum, a
step taken after the last projection), which inflates attack success.

## Decision
- Every model takes pixels in [0, 1] and standardizes them in its first layer
  (`Normalize`), so attacks, training and certification all work in pixel space and
  `eps = 8/255` means what the literature means.
- One module (`attacks/geometry.py`) owns the projections onto the L-infinity and L2
  balls intersected with the box, the random starts and the steepest-ascent directions.
  Every attack uses it.
- `evaluate_attack` checks every adversarial batch before scoring it: finite values,
  inside [0, 1], and within `eps` (plus 1e-5) in the attack's norm. A violation raises
  `ThreatModelError`; it is never scored. The HTTP service applies the same check.
- Attacks never change the model's train or eval mode. Evaluation runs in eval mode.
  Adversarial training generates examples in the mode each paper's code uses (train mode
  for PGD-AT as in Rice et al., eval mode for the TRADES inner maximization).

## Alternatives considered
- **Normalization in the data pipeline**: standard for clean training, but every attack
  would then need to convert budgets per channel, and checkpoints would not be usable by
  a third-party attack library without the same transform.
- **Trusting attack implementations**: cheaper by a few milliseconds per batch, but the
  torchattacks comparison runs third-party code through the same scorer, so the check
  is what makes the two libraries comparable.

## Consequences
- The torchattacks attacks run on our models unchanged (they also assume [0, 1] inputs).
- C&W, a minimum-norm attack, is made budget-aware: examples farther than `eps` are
  replaced by the clean input before scoring, so its robust accuracy at a budget is
  comparable with the bounded attacks.
- The check costs one extra reduction per batch, negligible next to an attack step.
