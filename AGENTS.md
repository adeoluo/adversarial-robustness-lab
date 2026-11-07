# AGENTS.md

Working notes for coding agents and contributors in **Adversarial Robustness Lab**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Evasion attacks, adversarial training and randomized smoothing certification for CIFAR-10 classifiers, measured on the full test set.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/adversarial_robustness_lab/__init__.py`: Adversarial Robustness Lab - package root.
- `src/adversarial_robustness_lab/attacks/__init__.py`: Evasion attacks: FGSM, PGD (L-infinity and L2), C&W L2, APGD, APGD-T, Square, ensembles.
- `src/adversarial_robustness_lab/attacks/apgd.py`: Auto-PGD: APGD-CE, APGD-DLR and targeted APGD-T (Croce and Hein, 2020).
- `src/adversarial_robustness_lab/attacks/autoattack.py`: An AutoAttack-style ensemble: run attacks in sequence on the points still robust.
- `src/adversarial_robustness_lab/attacks/base.py`: Common interface for evasion attacks.
- `src/adversarial_robustness_lab/attacks/cw.py`: Carlini-Wagner L2 attack (Carlini and Wagner, 2017).
- `src/adversarial_robustness_lab/attacks/geometry.py`: Threat-model geometry shared by every attack: norms, projections and random starts.
- `src/adversarial_robustness_lab/attacks/losses.py`: Objective functions maximized by the attacks, all returned per sample.
- `src/adversarial_robustness_lab/attacks/pgd.py`: Gradient attacks with a fixed step: FGSM and projected gradient descent (PGD).
- `src/adversarial_robustness_lab/attacks/square.py`: Square attack, L-infinity (Andriushchenko et al., 2020): a score-based black-box attack.
- `src/adversarial_robustness_lab/cli.py`: Command line: ``python -m adversarial_robustness_lab <command>``.
- `src/adversarial_robustness_lab/data.py`: CIFAR-10 loading, caching and on-device augmentation.
- `src/adversarial_robustness_lab/evaluation.py`: Robust accuracy evaluation with threat-model verification.
- `src/adversarial_robustness_lab/masking.py`: A deliberately gradient-masking defense, and the adaptive attack that breaks it.
- `src/adversarial_robustness_lab/models.py`: Classifiers for CIFAR-10: PreAct ResNet-18, WideResNet and a small CNN.
- `src/adversarial_robustness_lab/runtime.py`: Device selection, seeding, memory limits and checkpoints.
- `src/adversarial_robustness_lab/sanity.py`: Gradient-masking sanity checks (Athalye et al., 2018; Carlini et al., 2019).
- `src/adversarial_robustness_lab/service.py`: HTTP service: classify, attack and certify CIFAR-10 images with a trained checkpoint.
- `src/adversarial_robustness_lab/smoothing.py`: Randomized smoothing certification (Cohen, Rosenfeld and Kolter, ICML 2019).
- `src/adversarial_robustness_lab/stats.py`: Statistics for robustness claims: exact binomial tools and confidence intervals.
- `src/adversarial_robustness_lab/training.py`: Training: standard, PGD adversarial training, TRADES, fast FGSM training and noise.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies (with the validation group) and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make validation` | Parity tests against torchattacks, SciPy and statsmodels |
| `make data` | Download and verify CIFAR-10 (163 MiB, outside the repository) |
| `make train` | Train one zoo model: make train NAME=wrn16_2_pgd |
| `make train-all` | Train the whole zoo, one run at a time |
| `make evaluation` | Robust accuracy of every model on the full test set |
| `make compare` | Our attacks against torchattacks on the same models |
| `make certify` | Randomized smoothing certification |
| `make sanity` | Gradient-masking sanity checks |
| `make report` | Print the README tables from benchmarks/results |
| `make run` | Run the HTTP service (CHECKPOINT=path) |
| `make diagram` | Render the architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose (ARL_CHECKPOINT=path) |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
