# ADR-0004: Adversarial training sized for an 8 GB Apple M1 shared with other work

- **Status:** Accepted
- **Date:** 2026-11-07

## Context
The reference CIFAR-10 robust models (Madry et al.; TRADES) are WideResNet-34-10
(46 M parameters) trained for 100 or more epochs with 7 to 10 PGD steps per batch, many
GPU-days of compute. This project trains on one Apple M1 (8 GPU cores, 8 GB unified
memory, shared with other jobs), with a memory cap of about 2 GB per process. Measured
training throughput with plain PyTorch on MPS under that load: PreAct ResNet-18
(11.2 M parameters) about 150 to 230 images per second, WideResNet-16-2 (0.69 M) about
450 to 570. A PGD-10 epoch of PreAct ResNet-18 would take close to an hour.

## Decision
- Compare the training methods on one architecture, WideResNet-16-2, so differences
  come from the method, not the model: standard, fast FGSM with a random start (Wong et
  al., 2020), PGD-AT with 5 steps of 2/255 (Madry et al.), TRADES with beta 6 and the
  same inner attack (Zhang et al., 2019), and Gaussian noise augmentation for smoothing
  (sigma 0.25 and 0.5).
- 10 epochs each with a one-cycle learning-rate schedule (peak at 40% of steps), SGD
  with Nesterov momentum 0.9, weight decay 5e-4, batch 128, random crop and flip. Short
  cyclic schedules are what made fast adversarial training practical (Wong et al.);
  with 10 epochs, robust overfitting (Rice et al., 2020) is not a concern.
- Train PreAct ResNet-18 and a small CNN with standard training to show the effect of
  model size on clean accuracy and on (lack of) robustness. Outcome: the PreAct
  ResNet-18 run ran out of memory under the MPS cap during epoch 4 and was dropped; the
  small CNN and the standard WideResNet-16-2 cover the comparison.
- Keep the 50,000 training images on the device as uint8 (154 MB instead of 614 MB as
  float32) and augment on the device; cap the MPS allocator with
  `torch.mps.set_per_process_memory_fraction`.
- Checkpoint every epoch and resume, so a long run survives interruptions.

## Alternatives considered
- **WideResNet-34-10 or PreAct ResNet-18 with PGD-10**: infeasible in the budget.
- **Mixed precision on MPS**: measured speedups were inconsistent under contention and
  float16 changes gradient attacks; everything runs in float32.
- **Pretrained robust models (RobustBench)**: would reproduce known numbers, but the
  project is about training as well as attacking, and RobustBench downloads need the
  `robustbench` package and up to several hundred MB per model.

## Consequences
- Absolute robust accuracy is well below the published WideResNet-34-10 numbers (TRADES
  WideResNet-34-10 is listed at 53.08% AutoAttack accuracy on RobustBench, not rerun); the EVALUATION_REPORT compares orders, gaps and
  costs, not absolute values, with the literature.
- Wall-clock times are stated per run and depend on the shared machine's load.
