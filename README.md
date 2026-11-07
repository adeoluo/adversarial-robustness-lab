<h1 align="center">Adversarial Robustness Lab</h1>

<p align="center">
  <em>Evasion attacks, adversarial training and randomized smoothing certification for CIFAR-10 classifiers, measured on the full test set.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/adversarial-robustness-lab/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/adversarial-robustness-lab/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/adversarial-robustness-lab/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/adversarial-robustness-lab"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

A robustness claim is only as good as the strongest attack it was tested against. Many
published defenses looked robust under PGD and fell to a stronger or adaptive attack,
usually because their gradients were masked rather than because the model was robust.
This repository implements the standard evaluation toolkit in one place and holds it to
the same bar a reviewer would: every attack is checked against a reference library on
the same models, every adversarial batch is verified to lie inside the threat set, and
gradient-masking checks run next to the robust accuracy numbers.

It covers the three sides of the problem on CIFAR-10: attacks (FGSM, PGD in L-infinity
and L2, Carlini-Wagner, APGD-CE, APGD-DLR, APGD-T, Square and an AutoAttack-style
ensemble), defenses by adversarial training (PGD-AT and TRADES, with fast FGSM training
as a baseline), and certified robustness by randomized smoothing (Cohen et al., 2019,
with exact Clopper-Pearson bounds). All models are trained here on an 8 GB Apple M1,
and the training times are reported with the accuracies.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="860">
</p>

<sub>Legend: amber = entry points and external systems, indigo = core logic, green = data and state, blue = validation and control, grey = observability and outputs.</sub>

The CLI and the HTTP service pass images and budgets (eps, norm, steps, queries) to the
attack layer, where every attack is built on one set of projections onto the L-infinity
or L2 ball and the pixel box. Models come from the training layer (standard, PGD-AT,
TRADES, fast FGSM, Gaussian noise for smoothing) trained on the 50,000 CIFAR-10
training images; all of them take pixels in [0, 1] and normalize inside the network,
so eps is always in pixel units. The ensemble runs APGD-CE, APGD-T and Square in turn
on the images still robust and keeps the worst case per image. Every adversarial batch
passes a threat-model check before it is scored, and the masking checks and reports
read the per-image outcomes. Module by module walkthrough: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Results

RESULTS_PLACEHOLDER

## Quickstart

```bash
git clone https://github.com/adeoluo/adversarial-robustness-lab && cd adversarial-robustness-lab
make setup                       # uv sync with the validation group, pre-commit hooks
make data                        # CIFAR-10 (163 MiB, MD5 checked) to ../.bench-data/adversarial-robustness-lab
uv run python -m adversarial_robustness_lab train --arch wrn_16_2 --method pgd --epochs 10 --out runs/wrn_pgd.pt
uv run python -m adversarial_robustness_lab autoattack -c runs/wrn_pgd.pt --n 1000
```

Or with Docker: `docker compose up --build` serves a checkpoint over HTTP (see
`docker-compose.yml`). `ARL_DEVICE` selects `cpu`, `cuda` or `mps`; the default prefers
CUDA, then Apple MPS, then CPU.

## Usage

### Command line

```bash
python -m adversarial_robustness_lab info                                   # versions, device, attacks
python -m adversarial_robustness_lab evaluate -c model.pt --attacks fgsm pgd apgd-ce --eps 8/255
python -m adversarial_robustness_lab autoattack -c model.pt --version standard --norm linf
python -m adversarial_robustness_lab attack -c model.pt --attack pgd --n 6   # per-image predictions
python -m adversarial_robustness_lab certify -c noise025.pt --n-samples 10000 --skip 20
python -m adversarial_robustness_lab sanity -c model.pt --strict            # gradient-masking checks
```

Every command accepts `--n`, `--start`, `--json` and `--out`; epsilons accept fractions
such as `8/255`.

### Library

```python
from adversarial_robustness_lab.attacks import APGD, PGD, AutoAttackEnsemble
from adversarial_robustness_lab.evaluation import evaluate_attack
from adversarial_robustness_lab.smoothing import Smooth

result = evaluate_attack(model, PGD(8 / 255, 2 / 255, 20), x, y, eps=8 / 255, norm="linf")
print(result.robust_accuracy, result.robust_ci())

ensemble = AutoAttackEnsemble.from_version(8 / 255, "linf", "standard")
worst = ensemble.run(model, x, y)            # robust mask and which attack broke each image

cert = Smooth(model, num_classes=10, sigma=0.25).certify(x[0], n0=100, n=10_000, alpha=0.001)
print(cert.prediction, cert.radius)
```

### HTTP service

```bash
python -m adversarial_robustness_lab serve -c model.pt --port 8000
curl -s localhost:8000/v1/attack -d '{"index": 3, "attack": "apgd-ce", "eps": 0.0314}'
curl -s localhost:8000/v1/certify -d '{"index": 3, "sigma": 0.25, "n": 1000}'
curl -s localhost:8000/metrics
```

Routes: `GET /health`, `GET /ready`, `GET /metrics` (Prometheus text), `POST /v1/predict`,
`POST /v1/attack`, `POST /v1/certify`. Work per request is bounded (steps, queries,
samples and body size).

### Reproducing the benchmarks

| Step | Command | Output |
|---|---|---|
| Train the zoo | `make train-all` (or `make train NAME=wrn16_2_pgd`) | `benchmarks/results/train/` |
| Robust accuracy, full test set | `make evaluation` | `benchmarks/results/robustness/` |
| Against torchattacks | `make compare` | `benchmarks/results/compare/` |
| Certification | `make certify` | `benchmarks/results/certify/` |
| Masking checks, ablations | `make sanity`, `uv run python benchmarks/ablations.py` | `benchmarks/results/sanity/`, `ablations/` |
| Tables in this README | `make report` | stdout |

`scripts/eval_queue.sh` runs every step in order, one at a time; long steps save their
progress and stop after about 35 minutes, and the queue resumes them until they finish.

### Demos

Recorded from real runs with [vhs](https://github.com/charmbracelet/vhs); each tape is in `docs/demo/`.

PGD at 8/255 against the standard and the PGD-trained WideResNet-16-2 on the first six test images.

<p align="center">
  <img src="docs/demo/attack.gif" alt="PGD at 8/255 against the standard and the PGD-trained WideResNet-16-2" width="860">
</p>

Gradient-masking checks flag a defense that quantizes its input; the BPDA attack then breaks it.

<p align="center">
  <img src="docs/demo/sanity.gif" alt="Gradient-masking checks flag a quantizing defense; BPDA breaks it" width="860">
</p>

Randomized smoothing certificates (sigma 0.25) with their Clopper-Pearson lower bounds and L2 radii.

<p align="center">
  <img src="docs/demo/certify.gif" alt="Randomized smoothing certificates with Clopper-Pearson bounds and L2 radii" width="860">
</p>

## Design decisions

- **Pixel-space threat model, verified on every batch.** Normalization is the first
  layer of every model and each adversarial batch is checked for the box and the budget
  before scoring. [ADR-0002](docs/adr/0002-pixel-space-threat-model-and-verification.md)
- **Report the worst case over an ensemble.** APGD-CE, APGD-T and Square in sequence on
  the survivors, rather than PGD alone. [ADR-0003](docs/adr/0003-autoattack-style-ensemble.md)
- **Training sized for the machine.** One architecture (WideResNet-16-2) for every
  training method, 10 epochs each, so differences come from the method; times stated.
  [ADR-0004](docs/adr/0004-adversarial-training-budget.md)
- **Exact certification statistics in the package.** Clopper-Pearson via the incomplete
  beta function, tested against SciPy and statsmodels; masking checks next to every
  number. [ADR-0005](docs/adr/0005-exact-certification-statistics.md)
- **PyTorch only, batched.** Attacks are tensor programs on the model's device; a
  compiled extension would not help when the cost is the network's forward and backward
  passes. [ADR-0006](docs/adr/0006-pytorch-without-compiled-core.md)

## Testing

```bash
make test           # 231 tests, about 45 s on CPU
make coverage       # with the 80% gate (currently 99% line coverage)
make validation     # the parity group alone: torchattacks, SciPy, statsmodels
```

- **Parity with reference implementations** (53 tests): FGSM, PGD, PGD-L2, C&W, APGD
  and Square against torchattacks on the same seeded models; Clopper-Pearson, the
  incomplete beta, binomial test and normal quantile against SciPy and statsmodels.
- **Property-based tests** (Hypothesis): projections onto the balls and the box, the
  attack losses, seeded batching and the interval statistics on generated inputs.
- **Attack invariants**: every attack stays in its threat set and is deterministic for
  a seed; FGSM matches its closed form on a linear model; restarts are never worse; PGD
  is at least as strong as FGSM and APGD at least as strong as PGD; Square uses no
  gradients; the ensemble's worst case is no better than any member.
- **Training**: each method reduces its loss, checkpoints round-trip, a resumed run
  matches the schedule of a single run.
- **Integration**: every CLI command and HTTP route on a synthetic split, including
  malformed and oversized requests.

CI runs lint, mypy, the tests with the coverage gate on Python 3.11 and 3.12, and builds
the package and the Docker image; CodeQL runs on main and weekly.

## Responsible AI

- **Model card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **Data card:** [docs/DATA_CARD.md](docs/DATA_CARD.md)
- **Threat model:** [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)
- **Evaluation report:** [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md)

The attacks here are published, widely available techniques for evaluating image
classifiers on a public benchmark; they are meant for measuring defenses, not for
attacking deployed systems. The models are research artifacts for CIFAR-10 and are not
suitable for any real decision.

## Limitations

LIMITATIONS_PLACEHOLDER

## Roadmap

- [ ] FAB-T, to complete the AutoAttack standard sequence.
- [ ] Longer schedules (100+ epochs) and larger WideResNets on a GPU machine.
- [ ] Certification of all 10,000 test images with n = 100,000.
- [ ] L2 adversarial training and the L2 ensemble on the full test set.

## License

Released under the [MIT License](LICENSE).
