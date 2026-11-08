# Model card: the CIFAR-10 model zoo

## Overview

Seven image classifiers trained from scratch in this repository on the 50,000 CIFAR-10
training images, to be attacked, defended and certified by the code here. They are
evaluation subjects, not products.

| Name | Architecture | Parameters | Training | Epochs | Training time (M1, MPS) |
|---|---|---:|---|---:|---:|
| `small_cnn_standard` | four-convolution CNN | 591,274 | standard | 20 | 8.5 min |
| `wrn16_2_standard` | WideResNet-16-2 | 691,674 | standard | 10 | 17.7 min |
| `wrn16_2_fgsm_rs` | WideResNet-16-2 | 691,674 | FGSM with random start (Wong et al., 2020), eps 8/255 | 10 | 19.8 min |
| `wrn16_2_pgd` | WideResNet-16-2 | 691,674 | PGD-AT (Madry et al., 2018), eps 8/255, 5 steps of 2/255 | 10 | 49.3 min |
| `wrn16_2_trades` | WideResNet-16-2 | 691,674 | TRADES (Zhang et al., 2019), beta 6, 5 steps | 10 | 84.1 min |
| `wrn16_2_noise025` | WideResNet-16-2 | 691,674 | Gaussian noise, sigma 0.25 (base model for smoothing) | 10 | 17.0 min |
| `wrn16_2_noise050` | WideResNet-16-2 | 691,674 | Gaussian noise, sigma 0.50 (base model for smoothing) | 10 | 24.8 min |

All use SGD with Nesterov momentum 0.9, weight decay 5e-4, batch size 128, a one-cycle
learning rate schedule, random crop and horizontal flip, seed 0. Inputs are pixels in
[0, 1]; per-channel standardization is the first layer of each network. Configurations
and per-epoch histories: `benchmarks/train_zoo.py`, `benchmarks/results/train/*.json`.
A PreAct ResNet-18 (11.2 M parameters, standard training) was also configured; its run
ran out of MPS memory under the 1.92 GiB cap during epoch 4 (64.3% clean after epoch 3,
about 7.3 minutes per epoch), so it is not part of the results.
Checkpoints are written to `../.bench-data/adversarial-robustness-lab/runs/` and are
not committed; `make train-all` recreates them.

## Intended use

- Subjects for measuring attacks, adversarial training and certification on CIFAR-10.
- Reproducing the tables in the README and [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

## Out of scope

- Any real classification task, and any claim about robustness outside CIFAR-10 at the
  stated budgets.
- Comparisons with published robust models as equals: published PGD-AT and TRADES
  results use 100 to 200 epochs on WideResNet-34-10 (46M parameters); these models use
  10 epochs on 0.7M parameters, so their clean and robust accuracies are lower.

## Performance

Full 10,000-image test set, L-infinity eps = 8/255, 95% Wilson intervals in the README
and [EVALUATION_REPORT.md](EVALUATION_REPORT.md):

<!-- PERF -->
| Model | Clean | FGSM | PGD-20 | Ensemble (APGD-CE, APGD-T, Square) |
|---|---:|---:|---:|---:|
| `wrn16_2_standard` | 88.7% | 4.4% | 0.0% | 0.0% |
| `wrn16_2_pgd` | 70.7% | 43.0% | 39.1% | 35.5% |
| `wrn16_2_trades` | 70.5% | 42.6% | 39.8% | 35.4% |
<!-- /PERF -->

Smoothed classifiers (500 test images, n = 10,000, alpha = 0.001), certified L2
accuracy at radius 0 / 0.25 / 0.5 / 0.75 / 1.0: `wrn16_2_noise025` at sigma 0.25:
71.4% / 56.2% / 38.2% / 18.2% / 0.0%; `wrn16_2_noise050` at sigma 0.5: 59.0% / 50.0% /
38.4% / 29.0% / 18.0%.

## Risks and limitations

- Short schedules: the adversarially trained models are under-trained compared with
  the literature, and their robustness is lower than published numbers for the same
  method.
- The models inherit CIFAR-10's biases (10 everyday object classes, 32 x 32 images).
- Robust accuracy is an upper bound measured by the attacks implemented here; a
  stronger attack could lower it. The sanity checks are designed to catch the common
  ways this bound is loose (gradient masking), not to prove it is tight.

## Footprint

Training the seven models took 3.7 hours on an Apple M1 (8 GB) with the MPS backend,
shared with other work; evaluation times are recorded in each results file.
