"""Statistics for robustness claims: exact binomial tools and confidence intervals.

Randomized smoothing turns Monte Carlo counts into a certificate, so its correctness
rests on these functions. They are implemented with the standard library and NumPy and
checked against SciPy and statsmodels in the validation tests.

* ``betainc(a, b, x)``: the regularized incomplete beta function
  ``I_x(a, b) = B(x; a, b) / B(a, b)``, evaluated with the continued fraction of
  Numerical Recipes (modified Lentz algorithm), using the symmetry
  ``I_x(a, b) = 1 - I_{1-x}(b, a)`` where the fraction converges faster.
* ``beta_ppf(p, a, b)``: its inverse in ``x`` by bisection (60 halvings reach double
  precision on ``[0, 1]``).
* Clopper-Pearson: for ``k`` successes in ``n`` trials, the exact interval
  ``[Beta.ppf(alpha/2; k, n-k+1), Beta.ppf(1-alpha/2; k+1, n-k)]``; the one-sided lower bound
  used by Cohen et al. is ``Beta.ppf(alpha; k, n-k+1)``.
* ``binom_test``: exact two-sided binomial test, summing the probabilities of all
  outcomes no more likely than the observed one (SciPy's definition).
* Wilson score interval and a percentile bootstrap for accuracies.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_FPMIN = 1e-300
_EPS = 1e-15


def _betacf(a: float, b: float, x: float, max_iter: int = 10_000) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > _FPMIN else _FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > _FPMIN else _FPMIN
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    raise RuntimeError("betainc continued fraction did not converge")  # pragma: no cover


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` for ``a, b > 0``, ``x`` in ``[0, 1]``."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0 or x == 1.0:
        return x
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse of ``x -> I_x(a, b)``: the ``p`` quantile of a Beta(a, b) distribution."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-16:
            break
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided ``1 - alpha`` confidence interval for a binomial proportion."""
    _check_counts(k, n)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    lower = 0.0 if k == 0 else beta_ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else beta_ppf(1 - alpha / 2, k + 1, n - k)
    return lower, upper


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.001) -> float:
    """One-sided ``1 - alpha`` lower confidence bound (Cohen et al.'s ``pA`` lower bound)."""
    _check_counts(k, n)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return 0.0 if k == 0 else beta_ppf(alpha, k, n - k + 1)


def binom_pmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    """Binomial probability mass, computed in log space for large ``n``."""
    k = np.asarray(k, dtype=np.float64)
    if p in (0.0, 1.0):
        target = 0 if p == 0.0 else n
        return (k == target).astype(np.float64)
    lg = np.vectorize(math.lgamma)
    log_pmf = (
        lg(n + 1.0) - lg(k + 1.0) - lg(n - k + 1.0) + k * math.log(p) + (n - k) * math.log1p(-p)
    )
    return np.asarray(np.exp(log_pmf))


def binom_test(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (SciPy's ``binomtest`` definition)."""
    _check_counts(k, n)
    if n == 0:
        return 1.0
    pmf = binom_pmf(np.arange(n + 1), n, p)
    observed = pmf[k]
    return float(min(1.0, pmf[pmf <= observed * (1 + 1e-7)].sum()))


def norm_ppf(q: float) -> float:
    """Standard normal quantile ``Phi^{-1}(q)``."""
    return NormalDist().inv_cdf(q)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion (good coverage near 0 and 1)."""
    _check_counts(k, n)
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    lower = 0.0 if k == 0 else max(0.0, centre - half)
    upper = 1.0 if k == n else min(1.0, centre + half)
    return lower, upper


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of ``values``."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    tail = (1 - confidence) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def _check_counts(k: int, n: int) -> None:
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
