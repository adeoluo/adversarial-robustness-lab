# Threat model

## What is evaluated

An **evasion** attacker who changes a test image within a small budget to change the
classifier's prediction:

| Setting | Budget | Knowledge | Attacks |
|---|---|---|---|
| White box, L-infinity | eps = 8/255 per pixel | weights and gradients | FGSM, PGD, PGD with margin loss, APGD-CE, APGD-DLR, APGD-T |
| White box, L2 | eps = 0.5 (pixel units) | weights and gradients | PGD-L2, C&W L2 |
| Black box, L-infinity | eps = 8/255 | output scores only | Square (up to 5,000 queries) |
| Adaptive | eps = 8/255 | knows the preprocessing defense | BPDA through input quantization |
| Certified, L2 | any radius r | none needed | randomized smoothing certificate holds for all attacks |

Images have pixels in [0, 1]; every adversarial example is checked to lie in the box
and within the budget before it is scored. Robust accuracy is the fraction of test
images classified correctly at the clean image and at every adversarial example an
attack found; the ensemble reports the worst case over its attacks per image.

## What the numbers mean and do not mean

- **Empirical robust accuracy is an upper bound.** A stronger or adaptive attack can
  lower it. The ensemble, the comparison with torchattacks and the gradient-masking
  checks reduce, but cannot remove, that risk.
- **Certified accuracy is a lower bound** for the smoothed classifier, valid with
  probability at least 1 - alpha (alpha = 0.001) per image, for L2 perturbations only.
  It says nothing about L-infinity budgets beyond the inclusion
  ||d||_2 <= sqrt(3072) ||d||_inf, which is far too loose to be useful at 8/255.
- **Out of scope:** physical-world perturbations, spatial transformations, poisoning
  and backdoors (training-time attacks), model extraction, membership inference, and
  attacks on the HTTP service itself beyond input validation.

## Service hardening

The HTTP service runs attacks and certification for a client, so it bounds the work
per request (at most 200 steps, 5,000 Square queries, 100,000 smoothing samples, 1 MB
bodies), validates shapes, ranges and types, serializes model access with a lock, and
binds to 127.0.0.1 by default. Checkpoints are loaded with `torch.load(weights_only=True)`,
which refuses pickled code; a test confirms a checkpoint carrying an object with a
custom `__reduce__` is rejected.

## Misuse

The attacks here are standard published algorithms, implemented for evaluation. They
apply to classifiers the user controls or is authorized to test. Nothing in the
repository targets a third-party system.
