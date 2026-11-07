# Evaluation report

Every number here comes from a script in `benchmarks/` run on the machine below; raw
results (JSON summaries and per-image outcomes) are in `benchmarks/results/`, and
`make report` prints every table in this file from them.

## Setup

- **Data:** CIFAR-10 (Krizhevsky, 2009). Training uses all 50,000 training images;
  clean and robust accuracy use all 10,000 test images. Costly comparisons use the first
  1,000 test images (AutoAttack pair: first 500) and certification every 20th test image
  (500 images). Subsets are fixed and stated with each table. See
  [DATA_CARD.md](DATA_CARD.md).
- **Models:** trained here, one seed (0) each; see [MODEL_CARD.md](MODEL_CARD.md) and
  [ADR-0004](adr/0004-adversarial-training-budget.md) for the budget.
- **Threat model:** L-infinity, eps = 8/255 in pixel units, unless stated. Every
  adversarial batch is checked to lie in [0, 1] and within eps before it is scored
  ([ADR-0002](adr/0002-pixel-space-threat-model-and-verification.md)); the largest
  perturbation seen per attack is recorded in the results (`max_linf`).
- **Metric:** robust accuracy = fraction of test images classified correctly on the
  clean image and on the adversarial example. The ensemble (APGD-CE 100 steps, APGD-T
  over the 3 most likely wrong classes, Square with 1,000 queries, each run on the
  survivors of the previous one) reports the per-image worst case
  ([ADR-0003](adr/0003-autoattack-style-ensemble.md)).
- **Uncertainty:** 95% Wilson intervals over test images for every accuracy. Attack
  comparisons on the same images use the exact two-sided binomial (sign) test on the
  images only one of the two attacks broke. Training-seed variance is not measured (one
  seed per model; see threats to validity).
- **Hardware:** Apple M1 (8 cores, 8 GB unified memory), PyTorch MPS backend, float32,
  shared with other jobs; MPS memory capped at about 2 GB per process. Wall-clock times
  are those of this shared machine.
- **Reproduce:** `make train-all`, `make evaluation`, `make compare`, `make certify`,
  `make sanity`, `uv run python benchmarks/ablations.py`, then `make report`.
  `scripts/eval_queue.sh` runs all of it in bounded, resumable steps.

<!-- TABLES -->

## Threats to validity

- **Upper bounds.** Robust accuracy is an upper bound: it holds only for the attacks
  run. The ensemble omits FAB-T from AutoAttack's standard sequence. The sanity checks
  look for the usual signs of gradient masking but cannot prove the bound is tight.
- **One training seed.** Each model is trained once; differences between training
  methods of a point or two may be within seed-to-seed variation, which is not measured.
  The intervals cover test-set sampling only.
- **Short schedules and small models.** 10 epochs on a 0.69 M parameter WideResNet-16-2;
  published PGD-AT and TRADES results use 100 to 200 epochs on WideResNet-34-10, so the
  absolute numbers are lower and the published ones were not rerun here.
- **Certification subset and budget.** 500 images with n = 10,000 Monte Carlo samples
  against Cohen et al.'s 10,000 images with n = 100,000. A smaller n lowers the
  Clopper-Pearson bound, so the largest certifiable radius at alpha = 0.001 is smaller
  (see the sampling budget table); certified accuracy at large radii is underestimated
  relative to the cited protocol.
- **Shared hardware.** Times are measured on a machine running other jobs; they compare
  libraries run under the same conditions, not peak speeds.
- **Test-set near-duplicates.** CIFAR-10 has near-duplicates between train and test
  (Barz and Denzler, 2020); not corrected, as in the published numbers.
