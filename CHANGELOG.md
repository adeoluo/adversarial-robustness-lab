# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.0] 2026-11-08

### Features

- Attacks, adversarial training, smoothing certification, CLI and service (#1)
- Report script, training results and demo GIFs (#3)

### Benchmarks

- Full CIFAR-10 test set results for the standard WideResNet-16-2 (#8)
- Full test set robustness of the PGD-trained WideResNet-16-2 (#10)
- Full test set robustness of the TRADES-trained WideResNet-16-2 (#11)
- Randomized smoothing certification on 500 test images (#12)
- Attack strength against torchattacks on the PGD-trained WideResNet-16-2 (#14)
- Gradient-masking sanity checks on the robust models and a quantizing defense (#16)
- Ablations of steps, restarts, query budget and epsilon (#17)
- Full test set robustness of the FGSM-RS WRN-16-2 and the small CNN (#19)
- APGD and AutoAttack pairs on the standard WRN-16-2, PreAct ResNet-18 back in the queue (#21)
- Full test set robustness of the standard PreAct ResNet-18 (#27)

### Bug fixes

- Keep per-stage ensemble times in the robustness results (#9)
- Smaller batches for the torchattacks AutoAttack comparison (#13)
- Batch the zero-gradient sanity check to bound memory (#15)
- Key resumable benchmark progress on a checkpoint fingerprint (#23)

### Performance

- Resumable time-bounded evaluation, certification and training steps (#4)
- Resumable parts for sanity checks, ablations and the torchattacks comparison (#7)

### Documentation

- Architecture decision records, diagram, threat model and data card (#2)
- README, evaluation report and model card with full test set results (#6)
- Map the threat model to MITRE ATLAS v2026.10, OWASP and NIST AI 100-2 (#22)
- Results for the FGSM-RS model, the small CNN and the standard-model comparison (#24)

### Maintenance

- Scaffold project
- Release tooling, agent guide and changelog (#5)
- Drop the PreAct ResNet-18 run from the queues (#18)
- Latest toolchain and CI versions (#20)
- Citation metadata, llms.txt and release script from current repository data (#25)
- List benchmark pull requests in the changelog (#26)
