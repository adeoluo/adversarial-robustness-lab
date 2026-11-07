from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from adversarial_robustness_lab.attacks import geometry as G
from adversarial_robustness_lab.attacks import losses as L

eps_st = st.floats(0.001, 2.0)


@settings(max_examples=30, deadline=None)
@given(eps_st, st.sampled_from(G.NORMS), st.integers(0, 1000))
def test_projection_lands_in_ball(eps: float, norm: G.Norm, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    delta = torch.randn(5, 3, 4, 4, generator=g) * 3
    proj = G.project(delta, eps, norm)
    assert float(G.flat_norm(proj, norm).max()) <= eps * (1 + 1e-5)


@settings(max_examples=30, deadline=None)
@given(eps_st, st.sampled_from(G.NORMS), st.integers(0, 1000))
def test_projection_is_idempotent(eps: float, norm: G.Norm, seed: int) -> None:
    delta = torch.randn(4, 3, 4, 4, generator=torch.Generator().manual_seed(seed))
    once = G.project(delta, eps, norm)
    torch.testing.assert_close(G.project(once, eps, norm), once)


def test_projection_keeps_interior_points() -> None:
    delta = torch.full((2, 3, 2, 2), 0.01)
    torch.testing.assert_close(G.project(delta, 1.0, "l2"), delta)
    torch.testing.assert_close(G.project(delta, 0.5, "linf"), delta)


def test_projection_with_per_sample_eps() -> None:
    delta = torch.ones(2, 1, 2, 2)
    eps = torch.tensor([0.1, 0.5])
    out = G.project(delta, eps, "linf")
    assert float(out[0].max()) == pytest.approx(0.1)
    assert float(out[1].max()) == pytest.approx(0.5)
    out2 = G.project(delta, eps, "l2")
    torch.testing.assert_close(G.flat_norm(out2, "l2"), eps)


def test_l2_projection_is_radial() -> None:
    delta = torch.tensor([[[[3.0, 4.0]]]])
    torch.testing.assert_close(G.project(delta, 1.0, "l2"), torch.tensor([[[[0.6, 0.8]]]]))


def test_project_to_threat_respects_box() -> None:
    x = torch.rand(3, 3, 8, 8)
    x_adv = x + torch.randn_like(x)
    out = G.project_to_threat(x, x_adv, 0.1, "linf")
    assert float(out.min()) >= 0 and float(out.max()) <= 1
    assert float((out - x).abs().max()) <= 0.1 + 1e-6
    assert torch.equal(G.clip_to_box(x, x_adv - x) + x, x_adv.clamp(0, 1))


@pytest.mark.parametrize("norm", G.NORMS)
def test_random_start_in_threat_set(norm: G.Norm) -> None:
    x = torch.rand(16, 3, 8, 8)
    out = G.random_start(x, 0.3, norm, torch.Generator().manual_seed(1))
    assert float(G.flat_norm(out - x, norm).max()) <= 0.3 + 1e-6
    assert float(out.min()) >= 0 and float(out.max()) <= 1
    assert not torch.equal(out, x)


def test_l2_random_start_radius_distribution() -> None:
    # Uniform in the ball of dimension d: radius r has CDF (r/eps)^d, so most mass is near eps.
    x = torch.full((400, 1, 4, 4), 0.5)
    out = G.random_start(x, 0.1, "l2", torch.Generator().manual_seed(0))
    radii = G.flat_norm(out - x, "l2")
    assert float(radii.median()) == pytest.approx(0.1 * 0.5 ** (1 / 16), rel=0.03)


def test_normalize_gradient() -> None:
    g = torch.randn(4, 3, 4, 4)
    assert torch.equal(G.normalize_gradient(g, "linf"), g.sign())
    torch.testing.assert_close(G.flat_norm(G.normalize_gradient(g, "l2"), "l2"), torch.ones(4))


def test_check_norm() -> None:
    assert G.check_norm("l2") == "l2"
    with pytest.raises(ValueError):
        G.check_norm("l1")


def test_margin_sign_matches_misclassification() -> None:
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    y = torch.tensor([0, 0])
    m = L.margin(logits, y)
    assert m.tolist() == [-2.0, 2.0]
    assert L.margin(logits, y, kappa=1.0).tolist() == [-2.0, 1.0]
    assert L.other_max(logits, y).tolist() == [1.0, 2.0]


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000), st.floats(0.1, 50.0), st.floats(-20.0, 20.0))
def test_dlr_is_shift_and_scale_invariant(seed: int, scale: float, shift: float) -> None:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(6, 10, generator=g, dtype=torch.float64)
    y = torch.randint(0, 10, (6,), generator=g)
    base = L.dlr(logits, y)
    torch.testing.assert_close(L.dlr(logits * scale + shift, y), base, rtol=1e-6, atol=1e-6)
    t = (y + 1) % 10
    torch.testing.assert_close(
        L.dlr_targeted(logits * scale + shift, y, t),
        L.dlr_targeted(logits, y, t),
        rtol=1e-6,
        atol=1e-6,
    )


def test_cross_entropy_is_not_scale_invariant() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    y = torch.tensor([0])
    assert float(L.cross_entropy(logits * 10, y)) < float(L.cross_entropy(logits, y))


def test_dlr_known_value() -> None:
    logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]])
    # -(4 - 3) / (4 - 1)
    assert float(L.dlr(logits, torch.tensor([0]))) == pytest.approx(-1 / 3)
    # targeted to class 2: -(4 - 1) / (4 - (1 + 0) / 2)
    assert float(L.dlr_targeted(logits, torch.tensor([0]), torch.tensor([2]))) == pytest.approx(
        -3 / 3.5
    )


def test_dlr_needs_classes() -> None:
    with pytest.raises(ValueError):
        L.dlr(torch.zeros(1, 2), torch.tensor([0]))
    with pytest.raises(ValueError):
        L.dlr_targeted(torch.zeros(1, 3), torch.tensor([0]), torch.tensor([1]))


def test_get_loss() -> None:
    assert L.get_loss("ce") is L.cross_entropy
    with pytest.raises(ValueError):
        L.get_loss("hinge")
    assert math.isfinite(float(L.get_loss("margin")(torch.zeros(1, 3), torch.tensor([0]))))
