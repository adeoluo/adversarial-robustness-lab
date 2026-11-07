# ADR-0003: Report the worst case over an AutoAttack-style ensemble, not a single attack

- **Status:** Accepted
- **Date:** 2026-11-07

## Context
A single fixed-step PGD attack overestimates robustness when its step size is badly
tuned or when the loss surface is flat or noisy: Croce and Hein (2020) re-evaluated more than 50 published models with AutoAttack
and lowered the reported robust accuracy of many of them, for some by more than 10
points. The community standard is AutoAttack: APGD-CE, APGD-T (targeted DLR), FAB-T
and Square, each run on the points that survived the previous ones, with no
per-defense tuning.

## Decision
- Implement APGD (step-size schedule with checkpoints, momentum, restart from the best
  point) with the CE, DLR and targeted DLR losses, and the Square attack (score-based,
  so it is not fooled by masked gradients), following the papers and the reference code.
- Report as the headline robust accuracy the worst case over an ensemble run
  sequentially: APGD-CE (100 steps), APGD-T (100 steps per target class), Square.
- FAB-T is not implemented. To measure what that and any other difference costs, the
  same models are attacked with torchattacks' `AutoAttack(version="standard")`, which
  includes FAB-T, and both numbers are reported side by side.
- On the full 10,000-image test set the ensemble uses 3 target classes for APGD-T and
  1,000 Square queries (instead of 9 and 5,000) to fit the compute budget; the
  standard-settings comparison on 1,000 images shows how much this changes the result.

## Alternatives considered
- **PGD-20 or PGD-100 only**: the common choice in older papers; reported in the tables
  for comparison, but not as the headline.
- **Calling the `autoattack` package directly**: it is not on PyPI and the point of this
  project is to implement the attacks; torchattacks ports it and serves as the reference.

## Consequences
- The headline number is an upper bound on the true robust accuracy: a stronger attack
  can only lower it. The sanity checks (ADR 0005) look for signs that it is too high.
- Evaluation cost is dominated by APGD-T and Square on the points that remain robust,
  so it grows with the model's robustness.
