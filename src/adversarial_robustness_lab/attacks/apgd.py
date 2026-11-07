"""Auto-PGD: APGD-CE, APGD-DLR and targeted APGD-T (Croce and Hein, 2020).

APGD removes PGD's fixed step size. With step size ``eta`` (initially ``2 eps``) and
momentum ``a = 0.75``, each iteration does::

    z^{k+1} = P_S( x^k + eta * d(grad L(x^k)) )
    x^{k+1} = P_S( x^k + a (z^{k+1} - x^k) + (1 - a)(x^k - x^{k-1}) )

(the first iteration uses ``a = 1``). The run is split at checkpoints ``w_j = ceil(p_j N)``
with ``p_0 = 0``, ``p_1 = 0.22`` and ``p_{j+1} = p_j + max(p_j - p_{j-1} - 0.03, 0.06)``.
At each checkpoint the step size of a sample is halved, and the sample restarts from its
best point so far, when either

1. the loss increased in at most ``rho = 0.75`` of the iterations since the previous
   checkpoint (the walk oscillates), or
2. the step size was not reduced at the previous checkpoint and the best loss has not
   improved since then (the walk stalls).

The iterate with the highest loss is tracked per sample and every misclassified iterate
is kept, so the attack never returns a worse point than it visited. The loss can be
cross-entropy (APGD-CE), the scale-invariant DLR loss (APGD-DLR), or the targeted DLR
loss for each of the top ``n_target_classes`` wrong classes in turn (APGD-T).

The structure (checkpoint lengths, oscillation test, initialization scaled to the ball
boundary) follows the authors' reference implementation in the ``autoattack`` package.
"""

from __future__ import annotations

import torch
from torch import nn

from adversarial_robustness_lab.attacks.base import Attack
from adversarial_robustness_lab.attacks.geometry import flat_norm, project_to_threat
from adversarial_robustness_lab.attacks.losses import cross_entropy, dlr, dlr_targeted, get_loss


def checkpoints(n_iter: int) -> list[int]:
    """Iterations at which APGD checks progress (the ``w_j`` of the paper).

    Integer form of the reference code: the first interval is ``int(0.22 N)``, each next
    one is ``int(0.03 N)`` shorter, never below ``int(0.06 N)`` (all at least 1).
    """
    k = max(int(0.22 * n_iter), 1)
    shortest = max(int(0.06 * n_iter), 1)
    decrease = max(int(0.03 * n_iter), 1)
    marks = [0]
    while marks[-1] + k <= n_iter:
        marks.append(marks[-1] + k)
        k = max(k - decrease, shortest)
    return marks


def _view(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return v.view(-1, *([1] * (like.dim() - 1)))


class APGD(Attack):
    """APGD with the cross-entropy or DLR loss, untargeted, with optional restarts."""

    name = "apgd"

    def __init__(
        self,
        eps: float,
        norm: str = "linf",
        steps: int = 100,
        loss: str = "ce",
        restarts: int = 1,
        rho: float = 0.75,
        seed: int = 0,
    ) -> None:
        super().__init__(eps, norm, seed)
        if steps < 1 or restarts < 1:
            raise ValueError("steps and restarts must be at least 1")
        self.steps = steps
        self.loss_name = loss
        self.restarts = restarts
        self.rho = rho
        self.name = f"apgd-{loss}"

    # The per-sample objective; APGD-T overrides it with a fixed target.
    def objective(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "ce":
            return cross_entropy(logits, y)
        if self.loss_name == "dlr":
            return dlr(logits, y)
        return get_loss(self.loss_name)(logits, y)

    def _init(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        if self.norm == "linf":
            t = 2 * torch.rand(x.shape, generator=gen, dtype=x.dtype).to(x.device) - 1
            delta = self.eps * t / _view(flat_norm(t, "linf"), x)
        else:
            t = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
            delta = self.eps * t / _view(flat_norm(t, "l2") + 1e-12, x)
        return (x + delta).clamp(0.0, 1.0)

    def _step(
        self,
        x: torch.Tensor,
        x_adv: torch.Tensor,
        grad: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        if self.norm == "linf":
            return x_adv + _view(eta, x) * grad.sign()
        g = grad / _view(flat_norm(grad, "l2") + 1e-12, grad)
        return x_adv + _view(eta, x) * g

    def _grad(
        self, model: nn.Module, x_adv: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = self.objective(logits, y)
        (grad,) = torch.autograd.grad(loss.sum(), x_adv)
        self.queries += 2 * x_adv.shape[0]
        return loss.detach(), logits.detach(), grad.detach()

    def attack_single_run(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, gen: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One APGD run. Returns ``(best_adv, fooled, best_loss)`` per sample."""
        n = x.shape[0]
        marks = set(checkpoints(self.steps)[1:])
        x_adv = self._init(x, gen)
        loss, logits, grad = self._grad(model, x_adv, y)
        fooled = logits.argmax(1) != y
        x_best_adv = x_adv.clone()
        loss_best = loss.clone()
        x_best, grad_best = x_adv.clone(), grad.clone()
        eta = torch.full((n,), 2.0 * self.eps, device=x.device)
        x_prev = x_adv.clone()
        increases = torch.zeros(n, device=x.device)
        loss_prev = loss.clone()
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones(n, dtype=torch.bool, device=x.device)
        last_mark = 0
        for k in range(self.steps):
            a = 0.75 if k > 0 else 1.0
            with torch.no_grad():
                z = project_to_threat(x, self._step(x, x_adv, grad, eta), self.eps, self.norm)
                x_new = x_adv + a * (z - x_adv) + (1 - a) * (x_adv - x_prev)
                x_new = project_to_threat(x, x_new, self.eps, self.norm)
                x_prev = x_adv
                x_adv = x_new
            loss, logits, grad = self._grad(model, x_adv, y)
            with torch.no_grad():
                wrong = logits.argmax(1) != y
                x_best_adv[wrong] = x_adv[wrong]
                fooled |= wrong
                increases += (loss > loss_prev).float()
                loss_prev = loss
                improved = loss > loss_best
                x_best[improved] = x_adv[improved]
                grad_best[improved] = grad[improved]
                loss_best = torch.where(improved, loss, loss_best)
                if k + 1 in marks:
                    span = k + 1 - last_mark
                    oscillating = increases <= self.rho * span
                    stalled = ~reduced_last_check & (loss_best_last_check >= loss_best)
                    reduce = oscillating | stalled
                    reduced_last_check = reduce
                    loss_best_last_check = loss_best.clone()
                    eta = torch.where(reduce, eta / 2, eta)
                    x_adv[reduce] = x_best[reduce]
                    grad[reduce] = grad_best[reduce]
                    increases.zero_()
                    last_mark = k + 1
        # Samples never fooled return their highest-loss point.
        x_best_adv[~fooled] = x_best[~fooled]
        return x_best_adv, fooled, loss_best

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gen = self.generator()
        adv = x.clone()
        with torch.no_grad():
            fooled = model(x).argmax(1) != y
        self.queries += x.shape[0]
        for _ in range(self.restarts):
            todo = (~fooled).nonzero().flatten()
            if todo.numel() == 0:
                break
            best, ok, _ = self.attack_single_run(model, x[todo], y[todo], gen)
            adv[todo] = best
            fooled[todo] = ok
        return adv


class APGDTargeted(APGD):
    """APGD-T: targeted DLR towards each of the top ``n_target_classes`` wrong classes."""

    name = "apgd-t"

    def __init__(
        self,
        eps: float,
        norm: str = "linf",
        steps: int = 100,
        n_target_classes: int = 9,
        restarts: int = 1,
        rho: float = 0.75,
        seed: int = 0,
    ) -> None:
        super().__init__(eps, norm, steps, "dlr", restarts, rho, seed)
        if n_target_classes < 1:
            raise ValueError("n_target_classes must be at least 1")
        self.n_target_classes = n_target_classes
        self.name = "apgd-t"
        self._target: torch.Tensor | None = None

    def objective(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._target is None:  # pragma: no cover - set before every run
            raise RuntimeError("target not set")
        return dlr_targeted(logits, y, self._target[: logits.shape[0]])

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gen = self.generator()
        adv = x.clone()
        with torch.no_grad():
            logits = model(x)
        self.queries += x.shape[0]
        fooled = logits.argmax(1) != y
        order = logits.sort(dim=1).indices  # ascending
        for rank in range(2, self.n_target_classes + 2):
            if rank > logits.shape[1]:
                break
            for _ in range(self.restarts):
                todo = (~fooled).nonzero().flatten()
                if todo.numel() == 0:
                    return adv
                self._target = order[todo, -rank]
                best, ok, _ = self.attack_single_run(model, x[todo], y[todo], gen)
                adv[todo[ok]] = best[ok]
                fooled[todo] = ok
        self._target = None
        return adv
