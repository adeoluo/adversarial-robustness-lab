# ADR-0005: Exact Clopper-Pearson bounds implemented in the package, and gradient-masking checks

- **Status:** Accepted
- **Date:** 2026-11-07

## Context
A randomized-smoothing certificate is only as sound as the confidence bound on the top
class probability. Cohen et al. use the one-sided Clopper-Pearson lower bound at
alpha = 0.001 and an exact two-sided binomial test for abstention. Normal
approximations are anti-conservative when the count is close to n, which is exactly
the regime of large certified radii. Empirical robustness has the opposite risk: a
model whose gradients are masked looks robust to gradient attacks.

## Decision
- Implement the regularized incomplete beta function (continued fraction, Lentz),
  its inverse (bisection to machine precision), Clopper-Pearson intervals, the exact
  binomial test and the normal quantile in `stats.py` with no SciPy dependency at run
  time. Validate against SciPy (`betainc`, `beta.ppf`, `binomtest`) and statsmodels
  (`proportion_confint(method="beta")`) to 1e-9 or better, including n = 100,000.
- Follow Cohen et al.'s CERTIFY and PREDICT exactly: separate selection (n0 = 100) and
  estimation samples, abstain when the lower bound is at most 1/2.
- Add gradient-masking sanity checks (`sanity.py`) from Athalye et al. (2018) and
  Carlini et al. (2019), and a deliberately masking defense (input quantization) with
  the BPDA attack that breaks it, as a positive control for the checks.

## Alternatives considered
- **SciPy at run time**: fine, but the core would then depend on SciPy for one function;
  keeping it as a test-time reference makes the parity explicit.
- **Wilson or normal intervals for certification**: not exact; used only to report
  confidence intervals on accuracies, never for certificates.

## Consequences
- Certificates hold with probability at least 1 - alpha per image, as in the paper.
- The maximum certifiable radius is bounded by the sample count: with n = 10,000 and
  alpha = 0.001 it is sigma times Phi^-1(0.001^(1/10000)), about 3.4 sigma.
