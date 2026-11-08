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
- **Models:** trained here, one seed (0) each, every checkpoint from its final epoch;
  see [MODEL_CARD.md](MODEL_CARD.md) and
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
  images only one of the two attacks broke; the same test compares training methods on
  their paired per-image outcomes. Training-seed variance is not measured (one seed per
  model; see threats to validity).
- **Hardware and software:** Apple M1 (8 cores, 8 GB unified memory), macOS 27.0.1,
  PyTorch 2.15.0 on the MPS backend in float32, Python 3.12.14, torchattacks 3.5.1;
  every results file records these. The machine was shared with other jobs and MPS
  memory was capped at 1.92 GiB per process (`--memory-fraction 0.36`); wall-clock
  times are those of this shared machine.
- **Reproduce:** `make train-all`, `make evaluation`, `make compare`, `make certify`,
  `make sanity`, `uv run python benchmarks/ablations.py`, then `make report`.
  `scripts/eval_queue.sh` runs all of it in bounded, resumable steps.

<!-- TABLES -->
## Results

Printed by `make report` from `benchmarks/results/`.

### Headline

| Model | Training | Parameters | Epochs | Training time | Clean | FGSM | PGD-20 | Ensemble (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Small CNN | standard | 0.59 M | 20 | 8 min | 85.8% | 3.9% | 0.0% | **0.0%** (0.0% to 0.0%) |
| WRN-16-2 | standard | 0.69 M | 10 | 18 min | 88.7% | 4.4% | 0.0% | **0.0%** (0.0% to 0.0%) |
| PreAct ResNet-18 | standard | 11.17 M | 10 | 70 min | 91.2% | 5.2% | 0.0% | **0.0%** (0.0% to 0.0%) |
| WRN-16-2 | FGSM-RS (fast AT) | 0.69 M | 10 | 20 min | 72.7% | 44.4% | 39.7% | **36.0%** (35.1% to 37.0%) |
| WRN-16-2 | PGD-AT (5 steps) | 0.69 M | 10 | 49 min | 70.7% | 43.0% | 39.1% | **35.5%** (34.6% to 36.4%) |
| WRN-16-2 | TRADES (beta 6) | 0.69 M | 10 | 84 min | 70.5% | 42.6% | 39.8% | **35.4%** (34.5% to 36.4%) |

### Ensemble breakdown

| Model | Clean | After APGD-CE | After APGD-T | After Square | Time (APGD-CE / APGD-T / Square) |
|---|---:|---:|---:|---:|---|
| Small CNN standard | 85.81% | 0.00% | 0.00% | 0.00% | 744 s / 0 s / 0 s |
| WRN-16-2 standard | 88.67% | 0.00% | 0.00% | 0.00% | 1794 s / 0 s / 0 s |
| PreAct ResNet-18 standard | 91.18% | 0.00% | 0.00% | 0.00% | 4263 s / 0 s / 0 s |
| WRN-16-2 FGSM-RS (fast AT) | 72.73% | 39.11% | 36.04% | 36.04% | 1150 s / 2006 s / 2147 s |
| WRN-16-2 PGD-AT (5 steps) | 70.66% | 38.54% | 35.50% | 35.50% | 1852 s / 1199 s / 1168 s |
| WRN-16-2 TRADES (beta 6) | 70.54% | 39.52% | 35.42% | 35.42% | 2760 s / 1746 s / 1788 s |

### C&W margin loss

| Model | PGD-20 (CE) | PGD-20 (C&W margin) |
|---|---:|---:|
| Small CNN standard | 0.0% | 0.0% |
| WRN-16-2 standard | 0.0% | 0.0% |
| PreAct ResNet-18 standard | 0.0% | 0.0% |
| WRN-16-2 FGSM-RS (fast AT) | 39.7% | 38.3% |
| WRN-16-2 PGD-AT (5 steps) | 39.1% | 37.3% |
| WRN-16-2 TRADES (beta 6) | 39.8% | 36.1% |

### Comparison with torchattacks

wrn16_2_pgd (torchattacks 3.5.1):

| Attack | Images | Ours: robust acc | torchattacks: robust acc | Broken only by ours / only by torchattacks | Verdict | Time ours / torchattacks |
|---|---:|---:|---:|---:|---|---:|
| FGSM | 10,000 | 43.02% | 43.02% | 0 / 0 | tie (identical outcomes) | 20 s / 15 s |
| PGD-20 | 10,000 | 39.11% | 39.05% | 13 / 19 | tie (p = 0.38) | 153 s / 141 s |
| PGD-L2-20 | 10,000 | 53.24% | 53.01% | 1 / 24 | torchattacks stronger | 192 s / 174 s |
| C&W-L2 (c=1, 100 steps) | 1,000 | 51.90% | 51.90% | 0 / 0 | tie (identical outcomes) | 128 s / 113 s |
| APGD-CE-100 | 1,000 | 40.20% | 40.20% | 0 / 0 | tie (identical outcomes) | 86 s / 73 s |
| APGD-DLR-100 | 1,000 | 39.30% | 39.30% | 0 / 0 | tie (identical outcomes) | 101 s / 107 s |
| Square-1000 | 1,000 | 47.00% | 46.70% | 3 / 6 | tie (p = 0.51) | 420 s / 367 s |
| AutoAttack standard | 500 | 36.80% | 36.80% | 0 / 0 | tie (identical outcomes) | 1019 s / 1039 s |

wrn16_2_standard (torchattacks 3.5.1):

| Attack | Images | Ours: robust acc | torchattacks: robust acc | Broken only by ours / only by torchattacks | Verdict | Time ours / torchattacks |
|---|---:|---:|---:|---:|---|---:|
| FGSM | 10,000 | 4.44% | 4.45% | 4 / 3 | tie (p = 1.00) | 23 s / 19 s |
| PGD-20 | 10,000 | 0.00% | 0.00% | 0 / 0 | tie (identical outcomes) | 249 s / 279 s |
| PGD-L2-20 | 10,000 | 0.01% | 0.01% | 0 / 0 | tie (identical outcomes) | 201 s / 252 s |
| C&W-L2 (c=1, 100 steps) | 1,000 | 0.30% | 1.00% | 7 / 0 | ours stronger | 191 s / 109 s |
| APGD-CE-100 | 1,000 | 0.00% | 0.00% | 0 / 0 | tie (identical outcomes) | 116 s / 195 s |
| APGD-DLR-100 | 1,000 | 0.00% | 0.00% | 0 / 0 | tie (identical outcomes) | 163 s / 162 s |
| Square-1000 | 1,000 | 0.30% | 0.20% | 0 / 1 | tie (p = 1.00) | 125 s / 115 s |
| AutoAttack standard | 500 | 0.00% | 0.00% | 0 / 0 | tie (identical outcomes) | 104 s / 16 s |

### Certification

| Base model | sigma | Images | n | r = 0.00 | r = 0.25 | r = 0.50 | r = 0.75 | r = 1.00 | Abstain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WRN-16-2, noise 0.25 | 0.25 | 500 | 10,000 | 71.4% | 56.2% | 38.2% | 18.2% | 0.0% | 9.6% |
| WRN-16-2, noise 0.5 | 0.5 | 500 | 10,000 | 59.0% | 50.0% | 38.4% | 29.0% | 18.0% | 18.6% |
| Cohen et al. ResNet-110 (cited) | 0.25 | 10,000 | 100,000 | 77% | 61% | 42% | 25% | 0% | n/a |
| Cohen et al. ResNet-110 (cited) | 0.5 | 10,000 | 100,000 | 66% | 55% | 43% | 32% | 22% | n/a |

### Sampling budget

| sigma | n | Certified at r = 0 | r = 0.25 | r = 0.5 | Abstain | Largest certifiable radius |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 100 | 65.0% | 47.4% | 0.0% | 22.2% | 0.38 |
| 0.25 | 1,000 | 70.6% | 54.8% | 33.8% | 12.8% | 0.62 |
| 0.25 | 10,000 | 71.4% | 56.2% | 38.2% | 9.6% | 0.80 |
| 0.5 | 100 | 51.6% | 40.4% | 28.0% | 33.6% | 0.75 |
| 0.5 | 1,000 | 58.0% | 48.0% | 36.4% | 20.8% | 1.23 |
| 0.5 | 10,000 | 59.0% | 50.0% | 38.4% | 18.6% | 1.60 |

### Sanity checks

| Check | wrn16_2_pgd | wrn16_2_trades | wrn16_2_fgsm_rs | wrn16_2_pgd+quantize3 |
|---|---|---|---|---|
| iterative_beats_single_step | pass: PGD-20 0.4070 vs FGSM 0.4400 | pass: PGD-20 0.4080 vs FGSM 0.4340 | pass: PGD-20 0.4040 vs FGSM 0.4440 | pass: PGD-20 0.7050 vs FGSM 0.7140 |
| white_box_beats_black_box | pass: PGD-20 0.4070 vs Square-1000 0.4700 | pass: PGD-20 0.4080 vs Square-1000 0.4570 | pass: PGD-20 0.4040 vs Square-1000 0.4740 | **FAIL**: PGD-20 0.7050 vs Square-1000 0.4380 |
| unbounded_attack_succeeds | pass: PGD-50 at eps=1: 0.0000 | pass: PGD-50 at eps=1: 0.0000 | pass: PGD-50 at eps=1: 0.0000 | **FAIL**: PGD-50 at eps=1: 0.1760 |
| monotone_in_eps | pass: accuracy at eps x (1/4, 1/2, 1, 2): 0.6520, 0.5690, 0.4070, 0.1080 | pass: accuracy at eps x (1/4, 1/2, 1, 2): 0.6480, 0.5700, 0.4080, 0.1480 | pass: accuracy at eps x (1/4, 1/2, 1, 2): 0.6580, 0.5740, 0.4040, 0.1070 | pass: accuracy at eps x (1/4, 1/2, 1, 2): 0.7130, 0.7100, 0.7050, 0.6930 |
| more_steps_help_attacker | pass: PGD-100 0.4070 vs PGD-20 0.4070 | pass: PGD-100 0.4080 vs PGD-20 0.4080 | pass: PGD-100 0.4050 vs PGD-20 0.4040 | pass: PGD-100 0.7050 vs PGD-20 0.7050 |
| gradients_informative | pass: all-zero gradients 0.0000, non-finite 0.0000 | pass: all-zero gradients 0.0000, non-finite 0.0000 | pass: all-zero gradients 0.0000, non-finite 0.0000 | **FAIL**: all-zero gradients 1.0000, non-finite 0.0000 |
| random_noise_weaker | pass: uniform noise 0.7080 vs PGD-20 0.4070 | pass: uniform noise 0.7110 vs PGD-20 0.4080 | pass: uniform noise 0.7320 vs PGD-20 0.4040 | pass: uniform noise 0.7050 vs PGD-20 0.7050 |
| transfer_weaker_than_white_box | pass: transfer 0.6900 vs white-box 0.4070 | pass: transfer 0.6990 vs white-box 0.4080 | pass: transfer 0.7170 vs white-box 0.4040 | **FAIL**: transfer 0.6900 vs white-box 0.7050 |

Quantized model (wrn16_2_pgd+quantize3): clean 71.4%, PGD-20 70.5%, PGD-20 with BPDA 40.5%, Square-1000 43.8%.

### Ablations

PGD steps (wrn16_2_pgd, first 1000 test images):

| Steps | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Robust accuracy | 44.0% | 42.0% | 40.9% | 40.8% | 40.7% | 40.7% | 40.7% |

Restarts (PGD-20): 1: 40.4%, 2: 40.4%, 5: 40.4%

| Iterations | PGD | APGD-CE |
|---:|---:|---:|
| 10 | 40.8% | 40.4% |
| 25 | 40.7% | 40.3% |
| 50 | 40.7% | 40.2% |
| 100 | 40.7% | 40.2% |

Square queries: 10: 64.6%, 30: 62.5%, 100: 59.8%, 300: 53.8%, 1000: 47.8%, 3000: 43.2%, 5000: 41.3%
median queries to success: 176.0

| Model | 1/255 | 2/255 | 4/255 | 6/255 | 8/255 | 10/255 | 12/255 | 16/255 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wrn16_2_standard | 21.8% | 1.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| wrn16_2_fgsm_rs | 70.6% | 65.8% | 57.4% | 48.5% | 40.4% | 32.7% | 25.0% | 10.7% |
| wrn16_2_pgd | 67.6% | 65.2% | 56.9% | 48.5% | 40.7% | 33.0% | 24.5% | 10.8% |
| wrn16_2_trades | 68.0% | 64.8% | 57.0% | 48.8% | 40.8% | 33.1% | 26.3% | 14.8% |
<!-- /TABLES -->

## Findings

- **Adversarial training works at this scale, and the method matters less than the
  budget.** FGSM-RS, PGD-AT and TRADES reach 36.0%, 35.5% and 35.4% ensemble robust
  accuracy on all 10,000 test images. The three standard models are at 0.0% under
  the ensemble whatever their size: the small CNN (0.59 M parameters, 85.8% clean),
  WideResNet-16-2 (0.69 M, 88.7%) and PreAct ResNet-18 (11.17 M, 91.2%). On the paired
  per-image
  outcomes none of the robust accuracy differences is significant (sign test p = 0.056
  for FGSM-RS against PGD-AT, 0.059 against TRADES, 0.83 for PGD-AT against TRADES);
  FGSM-RS keeps 2.1 points more clean accuracy than PGD-AT (72.7% against 70.7%,
  p < 0.001) and trained in 20 minutes against 49 (PGD-AT) and 84 (TRADES). With one
  seed per model and the short 10-epoch schedule, FGSM-RS showed no catastrophic
  overfitting. Robustness against epsilon is the same for the robust models within 0.5
  points from 4/255 to 10/255; TRADES holds up better at 16/255.
- **PGD-20 overstates robustness by 3.6 to 4.4 points.** APGD-CE and APGD-T remove those
  images; Square then finds nothing more on any of the three robust models, which
  together with the sanity checks is evidence that the white-box attacks are not
  hampered by masking. The C&W margin loss alone already closes 1.3 to 3.7 points of
  that gap.
- **The attacks match torchattacks.** Fourteen of sixteen pairs on the PGD-trained and
  the standard WideResNet-16-2 are ties by the sign test, ten of them image for image.
  The one loss is PGD-L2-20 on the PGD-trained model, where torchattacks breaks 24
  images ours does not and ours breaks 1 it does not (53.24% against 53.01%). The two
  differ in the L2 random start (uniform in the ball's volume here, a uniform radius in
  torchattacks); this is the likely cause and was not isolated. The one win is C&W on the
  standard model (7 images against 0, 0.30% against 1.00%), also not isolated.
  torchattacks 3.5.1 runs APGD-CE and APGD-T inside `AutoAttack` with 10 iterations (its
  default) rather than the reference 100, so its AutoAttack row is a weaker and faster
  configuration than ours; on both models the outcomes were still identical. Summed over
  the fourteen non-AutoAttack pairs the two libraries take 2,169 s and 2,122 s.
- **Certification is 3 to 7 points below Cohen et al. at every radius**, with a model 2.5
  times smaller and 500 images certified with a tenth of their Monte Carlo samples. The
  sampling budget table shows how much of the gap at larger radii is the budget: going
  from n = 1,000 to n = 10,000 raises certified accuracy at r = 0.5 from 33.8% to 38.2%
  (sigma 0.25), and the largest certifiable radius from 0.62 to 0.80.
- **The masking checks do their job.** All three adversarially trained models pass all
  eight checks; the PGD-AT model behind 3-bit quantization fails four, appears 29.8
  points more robust under PGD-20 (70.5% against 40.7%), and BPDA brings it back to
  40.5%.

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
- **Resumed runs.** Long steps save per-image progress and resume. Progress is keyed on
  the settings and a SHA-256 prefix of every checkpoint used; an earlier version keyed
  on the settings only, and one evaluation (PreAct ResNet-18, never committed) resumed
  from predictions of an older checkpoint; it was rerun from scratch after the fix. The
  clean predictions in every committed robustness file were checked against a fresh CPU
  pass over all 10,000 test images: they agree image for image.
- **Test-set near-duplicates.** CIFAR-10 has near-duplicates between train and test
  (Barz and Denzler, 2020); not corrected, as in the published numbers.
