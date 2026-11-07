"""Randomized smoothing certification (Cohen, Rosenfeld and Kolter, ICML 2019).

Given a base classifier ``f``, the smoothed classifier predicts the class that ``f``
returns most often under Gaussian noise::

    g(x) = argmax_c P(f(x + e) = c),   e ~ N(0, sigma^2 I)

Theorem 1 of Cohen et al.: if the top class ``A`` has probability at least ``pA`` and
``pA > 1/2``, then ``g`` returns ``A`` for every ``x'`` with ``||x' - x||_2 < R``, where
``R = sigma * Phi^{-1}(pA)``.

``certify`` follows the paper's Monte Carlo procedure (``CERTIFY``): guess the top class
from ``n0`` noisy samples, estimate a one-sided ``1 - alpha`` Clopper-Pearson lower
bound on its probability from ``n`` fresh samples, and either abstain (bound at most 1/2)
or return the class with radius ``sigma * Phi^{-1}(lower bound)``. With probability at
least ``1 - alpha`` over the sampling, a returned certificate is correct. ``predict``
(``PREDICT``) abstains unless a two-sided binomial test rejects a tie between the two
most frequent classes at level ``alpha``.

The certified accuracy at radius ``r`` is the fraction of test points that are
certified with the correct class and a radius of at least ``r``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from adversarial_robustness_lab.stats import binom_test, clopper_pearson_lower, norm_ppf

ABSTAIN = -1


@dataclass(frozen=True)
class Certificate:
    """Outcome of ``certify`` for one input."""

    prediction: int
    radius: float
    p_lower: float
    count: int
    seconds: float


class Smooth:
    """A smoothed classifier around ``base`` with noise level ``sigma``."""

    def __init__(
        self,
        base: nn.Module,
        num_classes: int,
        sigma: float,
        batch_size: int = 500,
        seed: int = 0,
    ) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.base = base
        self.num_classes = num_classes
        self.sigma = sigma
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def sample_counts(self, x: torch.Tensor, num: int) -> np.ndarray:
        """Class counts of ``base`` over ``num`` noisy copies of the single image ``x``."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        remaining = num
        while remaining > 0:
            b = min(self.batch_size, remaining)
            batch = x.unsqueeze(0).expand(b, *x.shape)
            noise = torch.randn(batch.shape, generator=self.generator).to(x.device)
            preds = self.base(batch + self.sigma * noise).argmax(1)
            counts += np.bincount(preds.cpu().numpy(), minlength=self.num_classes)
            remaining -= b
        return counts

    def predict(self, x: torch.Tensor, n: int, alpha: float) -> int:
        """``PREDICT``: the top class, or ``ABSTAIN`` when it is not significantly ahead."""
        counts = self.sample_counts(x, n)
        top2 = counts.argsort()[::-1][:2]
        n_a, n_b = int(counts[top2[0]]), int(counts[top2[1]])
        if binom_test(n_a, n_a + n_b, 0.5) > alpha:
            return ABSTAIN
        return int(top2[0])

    def certify(self, x: torch.Tensor, n0: int, n: int, alpha: float) -> Certificate:
        """``CERTIFY``: class and L2 radius certified with probability at least ``1 - alpha``."""
        t0 = time.perf_counter()
        guess = int(np.argmax(self.sample_counts(x, n0)))
        counts = self.sample_counts(x, n)
        n_a = int(counts[guess])
        p_lower = clopper_pearson_lower(n_a, n, alpha)
        seconds = time.perf_counter() - t0
        if p_lower < 0.5:
            return Certificate(ABSTAIN, 0.0, p_lower, n_a, seconds)
        return Certificate(guess, self.sigma * norm_ppf(p_lower), p_lower, n_a, seconds)


def certified_accuracy(
    predictions: np.ndarray, radii: np.ndarray, labels: np.ndarray, radius: float
) -> float:
    """Fraction certified correct with radius at least ``radius``."""
    ok = (predictions == labels) & (radii >= radius)
    return float(ok.mean()) if ok.size else 0.0


def max_certifiable_radius(n: int, alpha: float, sigma: float) -> float:
    """The largest radius ``certify`` can return with ``n`` samples (all agree)."""
    return sigma * norm_ppf(clopper_pearson_lower(n, n, alpha))
