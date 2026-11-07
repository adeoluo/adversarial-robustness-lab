from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from adversarial_robustness_lab import stats as S


@pytest.mark.parametrize(
    ("a", "b", "x", "expected"),
    [
        (1.0, 1.0, 0.3, 0.3),  # uniform
        (2.0, 1.0, 0.5, 0.25),  # x^2
        (1.0, 2.0, 0.5, 0.75),  # 1 - (1 - x)^2
        (0.5, 0.5, 0.5, 0.5),  # arcsine, symmetric
        (2.0, 3.0, 0.4, 0.5248),  # binomial identity: P(Bin(4, 0.4) >= 2)
    ],
)
def test_betainc_known_values(a: float, b: float, x: float, expected: float) -> None:
    assert S.betainc(a, b, x) == pytest.approx(expected, abs=1e-12)


@settings(max_examples=60, deadline=None)
@given(st.floats(0.2, 300), st.floats(0.2, 300), st.floats(0.0, 1.0))
def test_betainc_symmetry_and_range(a: float, b: float, x: float) -> None:
    v = S.betainc(a, b, x)
    assert 0.0 <= v <= 1.0
    assert v + S.betainc(b, a, 1.0 - x) == pytest.approx(1.0, abs=1e-9)


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 2000), st.integers(1, 2000), st.floats(1e-6, 1 - 1e-6))
def test_beta_ppf_inverts_betainc(a: int, b: int, p: float) -> None:
    x = S.beta_ppf(p, a, b)
    assert S.betainc(a, b, x) == pytest.approx(p, abs=1e-9)


def test_betainc_binomial_tail_identity() -> None:
    # P(Bin(n, p) >= k) = I_p(k, n - k + 1)
    n, p = 30, 0.37
    for k in range(1, n + 1):
        tail = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
        assert S.betainc(k, n - k + 1, p) == pytest.approx(tail, abs=1e-12)


def test_betainc_validation() -> None:
    with pytest.raises(ValueError):
        S.betainc(0, 1, 0.5)
    with pytest.raises(ValueError):
        S.betainc(1, 1, 1.5)
    with pytest.raises(ValueError):
        S.beta_ppf(1.5, 1, 1)
    assert S.betainc(2, 2, 0.0) == 0.0 and S.betainc(2, 2, 1.0) == 1.0
    assert S.beta_ppf(0.0, 2, 2) == 0.0 and S.beta_ppf(1.0, 2, 2) == 1.0


def test_clopper_pearson_edges_and_known_value() -> None:
    assert S.clopper_pearson(0, 10)[0] == 0.0
    assert S.clopper_pearson(10, 10)[1] == 1.0
    # Zero successes in n: the upper bound is 1 - (alpha / 2)^(1 / n).
    assert S.clopper_pearson(0, 20, 0.05)[1] == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-12)
    # All successes: the one-sided lower bound is alpha^(1 / n) (the smoothing ceiling).
    assert S.clopper_pearson_lower(1000, 1000, 0.001) == pytest.approx(
        0.001 ** (1 / 1000), abs=1e-12
    )
    assert S.clopper_pearson_lower(0, 10) == 0.0


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 500), st.data())
def test_clopper_pearson_contains_estimate(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(0, n))
    lo, hi = S.clopper_pearson(k, n, 0.05)
    assert lo <= k / n <= hi
    assert S.clopper_pearson_lower(k, n, 0.025) == pytest.approx(lo, abs=1e-10)


def test_clopper_pearson_coverage_by_simulation() -> None:
    rng = np.random.default_rng(0)
    p, n, trials = 0.3, 40, 2000
    ks = rng.binomial(n, p, size=trials)
    covered = 0
    for k in ks:
        lo, hi = S.clopper_pearson(int(k), n, 0.1)
        covered += lo <= p <= hi
    assert covered / trials >= 0.9  # exact intervals are conservative


def test_count_validation() -> None:
    with pytest.raises(ValueError):
        S.clopper_pearson(5, 3)
    with pytest.raises(ValueError):
        S.clopper_pearson(1, 3, alpha=0)
    with pytest.raises(ValueError):
        S.clopper_pearson_lower(1, 3, alpha=1)
    with pytest.raises(ValueError):
        S.wilson_interval(-1, 3)


def test_binom_test_known_values() -> None:
    # Two-sided exact test of 7 heads in 10 fair flips: P = 0.34375.
    assert S.binom_test(7, 10) == pytest.approx(0.34375, abs=1e-12)
    assert S.binom_test(5, 10) == pytest.approx(1.0)
    assert S.binom_test(0, 0) == 1.0
    assert S.binom_test(10, 10) == pytest.approx(2 * 0.5**10)


def test_binom_test_asymmetric_p() -> None:
    pmf = S.binom_pmf(np.arange(6), 5, 0.2)
    assert pmf.sum() == pytest.approx(1.0)
    expected = pmf[pmf <= pmf[3] * (1 + 1e-7)].sum()
    assert S.binom_test(3, 5, 0.2) == pytest.approx(expected)
    assert S.binom_pmf(np.array([0, 3]), 3, 0.0).tolist() == [1.0, 0.0]
    assert S.binom_pmf(np.array([0, 3]), 3, 1.0).tolist() == [0.0, 1.0]


def test_norm_ppf() -> None:
    assert S.norm_ppf(0.5) == pytest.approx(0.0)
    assert S.norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)


def test_wilson_interval() -> None:
    lo, hi = S.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    assert S.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = S.wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35


def test_bootstrap_ci() -> None:
    v = np.r_[np.ones(600), np.zeros(400)]
    lo, hi = S.bootstrap_ci(v, 500, seed=1)
    assert lo < 0.6 < hi and hi - lo < 0.08
    with pytest.raises(ValueError):
        S.bootstrap_ci(np.array([]))
