"""Parity with reference implementations (needs the validation dependency group)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from adversarial_robustness_lab import stats as S
from adversarial_robustness_lab.attacks import APGD, FGSM, PGD, SquareAttack, predict
from adversarial_robustness_lab.attacks import losses as L
from adversarial_robustness_lab.models import SmallCNN

pytestmark = pytest.mark.validation
torchattacks = pytest.importorskip("torchattacks")
scipy_stats = pytest.importorskip("scipy.stats")
statsmodels_proportion = pytest.importorskip("statsmodels.stats.proportion")


def acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((predict(model, x) == y).float().mean())


def test_fgsm_matches_torchattacks(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    ref = torchattacks.FGSM(trained_cnn, eps=8 / 255)(x, y)
    torch.testing.assert_close(FGSM(8 / 255)(trained_cnn, x, y), ref, atol=1e-6, rtol=0)


def test_pgd_linf_matches_torchattacks(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    ref = torchattacks.PGD(trained_cnn, eps=8 / 255, alpha=2 / 255, steps=10, random_start=False)(
        x, y
    )
    ours = PGD(8 / 255, 2 / 255, 10, "linf", random_init=False)(trained_cnn, x, y)
    # Sign steps can flip on ties at float precision; allow a handful of coordinates.
    mismatch = ((ours - ref).abs() > 1e-5).float().mean()
    assert float(mismatch) < 1e-3


def test_pgd_l2_matches_torchattacks(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    # An untrained network keeps gradients far above torchattacks' 1e-10 division guard,
    # which it applies to the batch-mean loss (ours uses the sum).
    torch.manual_seed(0)
    model = SmallCNN(width=8).eval()
    x, y = batch
    ref = torchattacks.PGDL2(model, eps=0.5, alpha=0.1, steps=10, random_start=False)(x, y)
    ours = PGD(0.5, 0.1, 10, "l2", random_init=False)(model, x, y)
    # Float rounding differs (sum versus mean loss) and is amplified at ReLU and box edges.
    diff = (ours - ref).abs()
    assert float(diff.max()) < 0.01
    assert float(diff.mean()) < 1e-4


def test_apgd_strength_matches_torchattacks(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    eps = 12 / 255
    ref = torchattacks.APGD(trained_cnn, eps=eps, steps=20, loss="ce", seed=0)(x, y)
    ours = APGD(eps, steps=20, loss="ce")(trained_cnn, x, y)
    assert abs(acc(trained_cnn, ours, y) - acc(trained_cnn, ref, y)) <= 0.08
    ref_dlr = torchattacks.APGD(trained_cnn, eps=eps, steps=20, loss="dlr", seed=0)(x, y)
    ours_dlr = APGD(eps, steps=20, loss="dlr")(trained_cnn, x, y)
    assert abs(acc(trained_cnn, ours_dlr, y) - acc(trained_cnn, ref_dlr, y)) <= 0.08


def test_square_strength_matches_torchattacks(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    eps = 16 / 255
    ref = torchattacks.Square(trained_cnn, eps=eps, n_queries=1000, p_init=0.8, seed=0)(x, y)
    ours = SquareAttack(eps, 1000, 0.8)(trained_cnn, x, y)
    assert abs(acc(trained_cnn, ours, y) - acc(trained_cnn, ref, y)) <= 0.1


def test_dlr_matches_reference_formula() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(32, 10, generator=g)
    y = torch.randint(0, 10, (32,), generator=g)
    # The autoattack package's DLR, written out.
    x_sorted, ind_sorted = logits.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(32)
    ref = -(logits[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1.0 - ind)) / (
        x_sorted[:, -1] - x_sorted[:, -3] + 1e-12
    )
    torch.testing.assert_close(L.dlr(logits, y), ref)


@pytest.mark.parametrize(
    ("a", "b"), [(0.5, 0.5), (1, 1), (3, 7), (50, 2), (999, 2), (5000, 5000), (99_999, 2)]
)
def test_betainc_matches_scipy(a: float, b: float) -> None:
    from scipy.special import betainc

    for x in np.linspace(0.001, 0.999, 23):
        assert S.betainc(a, b, float(x)) == pytest.approx(float(betainc(a, b, x)), abs=1e-9)


@pytest.mark.parametrize(
    ("k", "n"),
    [(0, 10), (1, 10), (5, 10), (10, 10), (93, 100), (9_990, 10_000), (100_000, 100_000)],
)
@pytest.mark.parametrize("alpha", [0.001, 0.05])
def test_clopper_pearson_matches_statsmodels(k: int, n: int, alpha: float) -> None:
    lo, hi = statsmodels_proportion.proportion_confint(k, n, alpha=alpha, method="beta")
    ours = S.clopper_pearson(k, n, alpha)
    assert ours[0] == pytest.approx(lo, abs=1e-10)
    assert ours[1] == pytest.approx(hi, abs=1e-10)
    # Cohen et al.'s one-sided bound is the two-sided lower bound at 2 * alpha.
    ref_lower = statsmodels_proportion.proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]
    assert S.clopper_pearson_lower(k, n, alpha) == pytest.approx(ref_lower, abs=1e-10)


@pytest.mark.parametrize("q", [0.001, 0.1, 0.5, 0.9, 0.999])
@pytest.mark.parametrize(("a", "b"), [(2, 3), (100, 1), (1, 100), (7000, 3000)])
def test_beta_ppf_matches_scipy(q: float, a: float, b: float) -> None:
    assert S.beta_ppf(q, a, b) == pytest.approx(float(scipy_stats.beta.ppf(q, a, b)), abs=1e-10)


@pytest.mark.parametrize(
    ("k", "n", "p"), [(7, 10, 0.5), (0, 20, 0.5), (60, 100, 0.5), (3, 5, 0.2), (480, 1000, 0.5)]
)
def test_binom_test_matches_scipy(k: int, n: int, p: float) -> None:
    ref = scipy_stats.binomtest(k, n, p).pvalue
    assert S.binom_test(k, n, p) == pytest.approx(ref, rel=1e-9, abs=1e-15)


def test_norm_ppf_matches_scipy() -> None:
    for q in (1e-6, 0.01, 0.3, 0.5, 0.97, 1 - 1e-6):
        assert S.norm_ppf(q) == pytest.approx(float(scipy_stats.norm.ppf(q)), abs=1e-9)
