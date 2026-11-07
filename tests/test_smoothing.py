from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from adversarial_robustness_lab.smoothing import (
    ABSTAIN,
    Smooth,
    certified_accuracy,
    max_certifiable_radius,
)
from adversarial_robustness_lab.stats import clopper_pearson_lower, norm_ppf


class Constant(nn.Module):
    def __init__(self, label: int) -> None:
        super().__init__()
        self.label = label
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], 10)
        out[:, self.label] = 1
        return out


class HalfPlane(nn.Module):
    """Class 0 if the mean pixel is below 0.5, class 1 otherwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.flatten(1).mean(1) - 0.5
        return torch.stack([-s, s] + [torch.full_like(s, -1e9)] * 8, 1)


def test_constant_classifier_gets_maximum_radius() -> None:
    smooth = Smooth(Constant(3), 10, sigma=0.5, batch_size=64)
    cert = smooth.certify(torch.rand(3, 32, 32), n0=10, n=500, alpha=0.001)
    assert cert.prediction == 3
    assert cert.count == 500
    expected = 0.5 * norm_ppf(clopper_pearson_lower(500, 500, 0.001))
    assert cert.radius == pytest.approx(expected)
    assert cert.radius == pytest.approx(max_certifiable_radius(500, 0.001, 0.5))


def test_coin_flip_classifier_abstains() -> None:
    smooth = Smooth(HalfPlane(), 10, sigma=0.5, batch_size=100)
    x = torch.full((3, 32, 32), 0.5)
    assert smooth.certify(x, 50, 400, 0.001).prediction == ABSTAIN
    assert smooth.predict(x, 400, 0.001) == ABSTAIN


def test_radius_matches_closed_form_for_linear_boundary() -> None:
    # For a half-space, P(class 1) = Phi(d / sigma) where d is the L2 distance to the boundary.
    sigma = 0.25
    d = 0.2
    x = torch.full((3, 32, 32), 0.5 + d / (3 * 32 * 32) ** 0.5)
    cert = Smooth(HalfPlane(), 10, sigma, 1000, seed=0).certify(x, 100, 20_000, 0.001)
    assert cert.prediction == 1
    # Certified radius is a lower bound on the true distance, close for large n.
    assert 0.15 < cert.radius <= d + 1e-6


def test_predict_returns_majority_class() -> None:
    smooth = Smooth(HalfPlane(), 10, sigma=0.1, batch_size=50)
    assert smooth.predict(torch.full((3, 32, 32), 0.9), 200, 0.001) == 1
    assert smooth.predict(torch.full((3, 32, 32), 0.1), 200, 0.001) == 0


def test_sample_counts_total_and_seed() -> None:
    a = Smooth(HalfPlane(), 10, 0.5, batch_size=7, seed=1).sample_counts(
        torch.full((3, 4, 4), 0.5), 50
    )
    b = Smooth(HalfPlane(), 10, 0.5, batch_size=7, seed=1).sample_counts(
        torch.full((3, 4, 4), 0.5), 50
    )
    assert a.sum() == 50
    np.testing.assert_array_equal(a, b)


def test_sigma_validation() -> None:
    with pytest.raises(ValueError):
        Smooth(HalfPlane(), 10, 0.0)


def test_certified_accuracy() -> None:
    preds = np.array([1, 2, ABSTAIN, 0])
    radii = np.array([0.5, 0.1, 0.0, 1.0])
    labels = np.array([1, 2, 0, 1])
    assert certified_accuracy(preds, radii, labels, 0.0) == 0.5
    assert certified_accuracy(preds, radii, labels, 0.25) == 0.25
    assert certified_accuracy(preds, radii, labels, 2.0) == 0.0
    assert certified_accuracy(np.array([]), np.array([]), np.array([]), 0.0) == 0.0
