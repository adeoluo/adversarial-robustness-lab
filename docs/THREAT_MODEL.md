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

## Mapping to standards

Technique and mitigation IDs are from MITRE ATLAS v2026.10 (`ATLAS-2026.10.yaml` in the
mitre-atlas/atlas-data release of 2026-10-28); risk IDs from the OWASP Machine Learning
Security Top 10 (2023, the current edition) and the attack classes of NIST AI 100-2
E2025 (Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and
Mitigations).

| What this repository does | MITRE ATLAS v2026.10 | OWASP | NIST AI 100-2 E2025 |
|---|---|---|---|
| FGSM, PGD (L-infinity, L2), C&W, APGD-CE, APGD-DLR, APGD-T with the weights | AML.T0043.000 Craft Adversarial Data: White-Box Optimization, given AML.T0044 Full AI Model Access; goal AML.T0015 Evade AI Model | ML01:2023 Input Manipulation Attack | Evasion, white box |
| Square with output scores only | AML.T0043.001 Black-Box Optimization through AML.T0040 AI Model Inference API Access | ML01:2023 | Evasion, black box (score based) |
| Transfer check: PGD on the standard model as a surrogate, scored on the robust model | AML.T0043.002 Black-Box Transfer | ML01:2023 | Evasion, transfer |
| BPDA through 3-bit input quantization (adaptive attack) | AML.T0043.000 against the mitigation AML.M0010 Predictive AI Input Restoration | ML01:2023 | Evasion, adaptive attack on a preprocessing defense |
| Robust accuracy under the ensemble, comparison with torchattacks, gradient-masking checks | AML.M0008 Validate AI Model (measures what AML.T0042 Verify Attack would find) | ML01:2023 (testing the prevention) | Evaluation of mitigations |
| PGD-AT, TRADES, FGSM-RS adversarial training; randomized smoothing certificates | AML.M0003 Predictive AI Model Hardening | ML01:2023 prevention: adversarial training, robust models | Mitigations for evasion: adversarial training, certified defenses |
| Checkpoints loaded with `weights_only=True`, pickled code refused | AML.T0011.000 User Execution: Unsafe AI Artifacts and AML.T0010.003 Supply Chain Compromise: Model Repository; partial AML.M0014 Verify AI Artifacts (no signature or hash check) | ML06:2023 AI Supply Chain Attacks | Supply chain attacks on model files |
| Per-request bounds on steps, queries, samples and body size in the HTTP service | AML.T0029 Denial of AI Service and AML.T0034.001 Cost Harvesting: Resource-Intensive Queries; partial AML.M0004 Limit AI Service Query Volume and Rate (work per request is bounded, request rate is not) | n/a in the ML Top 10 | Availability |

Out of scope, with their IDs for reference: training data poisoning and backdoors
(AML.T0020 Training Data Poisoning, AML.T0043.004 Insert Backdoor Trigger, OWASP ML02:2023
and ML10:2023), model theft and replication (AML.T0005, AML.T0024 Exfiltration via AI
Inference API, ML05:2023), and model inversion and membership inference (ML03:2023,
ML04:2023). The OWASP Top 10 for LLM Applications 2026 and the OWASP Top 10 for Agentic
Applications 2026 address LLM and agent applications and do not apply to an image
classifier; the nearest LLM entries are LLM04:2026 Supply Chain for the checkpoint
loading and LLM06:2026 Unbounded Consumption for the service bounds.

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
