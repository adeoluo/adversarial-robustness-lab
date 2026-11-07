"""Threat-model geometry shared by every attack: norms, projections and random starts.

An attack searches the set ``S(x) = {x' : ||x' - x||_p <= eps, 0 <= x' <= 1}``. For the
two norms used here the Euclidean projection onto the norm ball has a closed form:

* L-infinity: clip every coordinate of ``delta`` to ``[-eps, eps]``.
* L2: rescale, ``delta * min(1, eps / ||delta||_2)``.

Intersecting with the pixel box ``[0, 1]`` afterwards (a clip of ``x + delta``) gives a
point of ``S(x)``; for L-infinity this is the exact projection onto the intersection
because both sets are axis-aligned boxes, for L2 it is the standard approximation used
by Madry et al. and the reference libraries.
"""

from __future__ import annotations

from typing import Literal

import torch

Norm = Literal["linf", "l2"]
NORMS: tuple[Norm, ...] = ("linf", "l2")
_TINY = 1e-12


def check_norm(norm: str) -> Norm:
    """Validate a norm name and return it typed."""
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    return norm  # type: ignore[return-value]


def flat_norm(t: torch.Tensor, norm: Norm) -> torch.Tensor:
    """Per-sample norm of a batch, shape ``(n,)``."""
    flat = t.reshape(t.shape[0], -1)
    if norm == "linf":
        return flat.abs().amax(dim=1)
    return flat.norm(p=2, dim=1)


def _view(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return v.view(-1, *([1] * (like.dim() - 1)))


def project(delta: torch.Tensor, eps: float | torch.Tensor, norm: Norm) -> torch.Tensor:
    """Project perturbations onto the ``norm`` ball of radius ``eps`` (per sample)."""
    if norm == "linf":
        if isinstance(eps, torch.Tensor):
            e = _view(eps, delta)
            return torch.maximum(torch.minimum(delta, e), -e)
        return delta.clamp(-eps, eps)
    norms = flat_norm(delta, "l2")
    e = eps if isinstance(eps, torch.Tensor) else torch.full_like(norms, float(eps))
    factor = torch.clamp(e / (norms + _TINY), max=1.0)
    return delta * _view(factor, delta)


def clip_to_box(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Return ``delta`` adjusted so that ``x + delta`` lies in ``[0, 1]``."""
    return (x + delta).clamp(0.0, 1.0) - x


def project_to_threat(
    x: torch.Tensor, x_adv: torch.Tensor, eps: float | torch.Tensor, norm: Norm
) -> torch.Tensor:
    """Map ``x_adv`` into ``S(x)``: project ``x_adv - x`` to the ball, then clip to pixels."""
    return (x + project(x_adv - x, eps, norm)).clamp(0.0, 1.0)


def normalize_gradient(g: torch.Tensor, norm: Norm, eps_div: float = 1e-10) -> torch.Tensor:
    """Steepest-ascent direction under ``norm``: ``sign(g)`` for L-infinity, ``g / ||g||`` for L2.

    ``eps_div`` matches the constant torchattacks adds before dividing (1e-10), so the
    L2 step is directly comparable in the parity tests.
    """
    if norm == "linf":
        return g.sign()
    return g / (_view(flat_norm(g, "l2"), g) + eps_div)


def random_start(
    x: torch.Tensor, eps: float, norm: Norm, generator: torch.Generator | None = None
) -> torch.Tensor:
    """A random point of ``S(x)``.

    L-infinity: ``delta ~ U[-eps, eps]^d``. L2: a uniform point of the ball, sampled as a
    Gaussian direction scaled by ``eps * u^(1/d)`` with ``u ~ U[0, 1]``.
    """
    if norm == "linf":
        noise = torch.rand(x.shape, generator=generator, dtype=x.dtype).to(x.device)
        delta = (2 * noise - 1) * eps
    else:
        d = x[0].numel()
        direction = torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)
        direction = direction / _view(flat_norm(direction, "l2") + _TINY, x)
        radius = torch.rand(x.shape[0], generator=generator, dtype=x.dtype).to(x.device)
        delta = direction * _view(eps * radius.pow(1.0 / d), x)
    return (x + delta).clamp(0.0, 1.0)
