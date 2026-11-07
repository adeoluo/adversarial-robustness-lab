from __future__ import annotations

import pytest
import torch
from torch import nn

from adversarial_robustness_lab.attacks import (
    APGD,
    ATTACK_NAMES,
    FGSM,
    PGD,
    APGDTargeted,
    CarliniWagnerL2,
    SquareAttack,
    attack_norm,
    make_attack,
    predict,
)
from adversarial_robustness_lab.attacks.apgd import checkpoints
from adversarial_robustness_lab.attacks.geometry import flat_norm
from adversarial_robustness_lab.attacks.square import p_selection
from adversarial_robustness_lab.evaluation import check_threat_model

EPS = 8 / 255


def acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((predict(model, x) == y).float().mean())


def test_fgsm_closed_form_on_linear_model(
    trained_linear: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    xg = x.clone().requires_grad_(True)
    nn.functional.cross_entropy(trained_linear(xg), y, reduction="sum").backward()
    assert xg.grad is not None
    expected = (x + EPS * xg.grad.sign()).clamp(0, 1)
    torch.testing.assert_close(FGSM(EPS)(trained_linear, x, y), expected)


@pytest.mark.parametrize("name", ATTACK_NAMES)
def test_every_attack_stays_in_threat_set(
    name: str, trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    x, y = x[:16], y[:16]
    eps = 0.5 if attack_norm(name) == "l2" else EPS
    attack = make_attack(name, eps, steps=5, queries=50)
    x_adv = attack(trained_cnn, x, y)
    check_threat_model(x, x_adv, eps, attack_norm(name))  # type: ignore[arg-type]
    assert attack.queries > 0


@pytest.mark.parametrize("name", ["fgsm", "pgd", "apgd-ce", "apgd-dlr", "apgd-t", "cw-linf"])
def test_attacks_reduce_accuracy(
    name: str, trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    clean = acc(trained_cnn, x, y)
    adv = acc(trained_cnn, make_attack(name, 16 / 255, steps=10)(trained_cnn, x, y), y)
    assert clean > 0.5
    assert adv < clean


def test_unknown_attack() -> None:
    with pytest.raises(ValueError):
        make_attack("deepfool", EPS)


def test_pgd_default_step_and_validation() -> None:
    assert PGD(0.1, steps=10).alpha == pytest.approx(0.025)
    with pytest.raises(ValueError):
        PGD(0.1, steps=0)
    with pytest.raises(ValueError):
        PGD(-0.1)
    assert "PGD" in repr(PGD(0.1))


def test_pgd_is_deterministic_given_seed(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    a = PGD(EPS, steps=3, seed=7)(trained_cnn, x, y)
    b = PGD(EPS, steps=3, seed=7)(trained_cnn, x, y)
    c = PGD(EPS, steps=3, seed=8)(trained_cnn, x, y)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_pgd_without_random_start_starts_at_x(
    trained_linear: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    one = PGD(EPS, alpha=EPS, steps=1, random_init=False)(trained_linear, x, y)
    torch.testing.assert_close(one, FGSM(EPS)(trained_linear, x, y))


def test_pgd_restarts_never_worse(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    one = acc(trained_cnn, PGD(4 / 255, steps=3, restarts=1)(trained_cnn, x, y), y)
    three = acc(trained_cnn, PGD(4 / 255, steps=3, restarts=3)(trained_cnn, x, y), y)
    assert three <= one


def test_pgd_l2_budget(trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = batch
    x_adv = PGD(0.25, steps=5, norm="l2")(trained_cnn, x, y)
    assert float(flat_norm(x_adv - x, "l2").max()) <= 0.25 + 1e-5


def test_pgd_stronger_than_fgsm(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    pgd = acc(trained_cnn, PGD(EPS, steps=10)(trained_cnn, x, y), y)
    fgsm = acc(trained_cnn, FGSM(EPS)(trained_cnn, x, y), y)
    assert pgd <= fgsm


def test_apgd_checkpoints_match_paper() -> None:
    assert checkpoints(100) == [0, 22, 41, 57, 70, 80, 87, 93, 99]
    assert checkpoints(10) == [0, *range(2, 11)]


def test_apgd_at_least_as_strong_as_pgd(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    pgd = acc(trained_cnn, PGD(4 / 255, steps=20)(trained_cnn, x, y), y)
    apgd = acc(trained_cnn, APGD(4 / 255, steps=20)(trained_cnn, x, y), y)
    assert apgd <= pgd + 0.05


def test_apgd_validation_and_names() -> None:
    assert APGD(0.1, loss="dlr").name == "apgd-dlr"
    assert APGDTargeted(0.1).name == "apgd-t"
    with pytest.raises(ValueError):
        APGD(0.1, steps=0)
    with pytest.raises(ValueError):
        APGDTargeted(0.1, n_target_classes=0)


def test_apgd_margin_loss_and_l2(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    x_adv = APGD(3.0, "l2", 10, "margin", restarts=2)(trained_cnn, x, y)
    check_threat_model(x, x_adv, 3.0, "l2")
    assert acc(trained_cnn, x_adv, y) < acc(trained_cnn, x, y)


def test_apgd_returns_misclassified_inputs_unchanged(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    wrong_y = (y + 1) % 10
    out = APGD(EPS, steps=5)(trained_cnn, x, wrong_y)
    already = predict(trained_cnn, x) != wrong_y
    assert torch.equal(out[already], x[already])


def test_apgd_targeted_hits_targets(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    x_adv = APGDTargeted(16 / 255, steps=10, n_target_classes=3)(trained_cnn, x, y)
    assert acc(trained_cnn, x_adv, y) < acc(trained_cnn, x, y)


def test_square_schedule() -> None:
    assert p_selection(0.8, 0, 10000) == 0.8
    assert p_selection(0.8, 11, 10000) == 0.4
    assert p_selection(0.8, 51, 10000) == 0.2
    assert p_selection(0.8, 9000, 10000) == pytest.approx(0.8 / 512)
    assert p_selection(0.8, 50, 100) == pytest.approx(0.8 / 128)


def test_square_needs_no_gradients(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    attack = SquareAttack(16 / 255, n_queries=200, seed=0)
    with torch.no_grad():
        x_adv = attack(trained_cnn, x, y)
    check_threat_model(x, x_adv, 16 / 255, "linf")
    assert acc(trained_cnn, x_adv, y) < acc(trained_cnn, x, y)
    assert attack.query_counts is not None
    assert int(attack.query_counts.max()) <= 200
    # Every coordinate sits at a vertex of the ball unless clipped by the pixel box.
    delta = (x_adv - x).abs()
    at_vertex = (delta - 16 / 255).abs() < 1e-5
    clipped = (x_adv == 0) | (x_adv == 1)
    assert bool((at_vertex | clipped).all())


def test_square_stops_on_fooled_samples(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    wrong = (predict(trained_cnn, x) + 1) % 10
    attack = SquareAttack(EPS, n_queries=50)
    attack(trained_cnn, x, wrong)
    assert attack.query_counts is not None
    assert int(attack.query_counts.max()) <= 2


def test_square_validation() -> None:
    with pytest.raises(ValueError):
        SquareAttack(0.1, n_queries=0)
    with pytest.raises(ValueError):
        SquareAttack(0.1, p_init=0)


def test_cw_finds_small_perturbations(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    x, y = x[:16], y[:16]
    attack = CarliniWagnerL2(c=1.0, steps=30, lr=0.05, binary_search_steps=3)
    x_adv = attack(trained_cnn, x, y)
    fooled = predict(trained_cnn, x_adv) != y
    assert bool(fooled.any())
    assert attack.last_distance is not None
    dist = flat_norm(x_adv - x, "l2")
    torch.testing.assert_close(dist[fooled], attack.last_distance[fooled], rtol=1e-4, atol=1e-4)
    assert bool((dist[~fooled] == 0).all())


def test_cw_budget_discards_far_examples(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    x_adv = CarliniWagnerL2(eps=0.05, steps=20, lr=0.05, binary_search_steps=2)(trained_cnn, x, y)
    assert float(flat_norm(x_adv - x, "l2").max()) <= 0.05 + 1e-6


def test_cw_validation() -> None:
    with pytest.raises(ValueError):
        CarliniWagnerL2(steps=0)


def test_attack_norm_names() -> None:
    assert attack_norm("pgd-l2") == "l2"
    assert attack_norm("square") == "linf"


def test_predict_handles_empty_input(trained_cnn: nn.Module) -> None:
    assert predict(trained_cnn, torch.empty(0, 3, 32, 32)).numel() == 0
