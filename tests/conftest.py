"""Shared fixtures: synthetic CIFAR-shaped data and small models trained on it (CPU)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from adversarial_robustness_lab.data import ImageSplit, synthetic_split
from adversarial_robustness_lab.models import SmallCNN

torch.set_num_threads(2)


class LinearModel(nn.Module):
    """A linear classifier on flattened pixels: gradients are exact and cheap."""

    def __init__(self, seed: int = 0, num_classes: int = 10) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(num_classes, 3 * 32 * 32, generator=g) * 0.05)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1) @ self.weight.T + self.bias


def _fit(model: nn.Module, split: ImageSplit, steps: int = 100, lr: float = 3e-3) -> nn.Module:
    torch.manual_seed(0)
    x, y = split.tensors()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for i in range(steps):
        idx = torch.arange(i * 64, (i + 1) * 64) % x.shape[0]
        loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@pytest.fixture(scope="session")
def train_split() -> ImageSplit:
    return synthetic_split(512, seed=0)


@pytest.fixture(scope="session")
def test_split() -> ImageSplit:
    return synthetic_split(64, seed=1)


@pytest.fixture(scope="session")
def trained_cnn(train_split: ImageSplit) -> nn.Module:
    return _fit(SmallCNN(width=8), train_split)


@pytest.fixture(scope="session")
def trained_linear(train_split: ImageSplit) -> nn.Module:
    return _fit(LinearModel(), train_split, steps=80, lr=1e-2)


@pytest.fixture()
def batch(test_split: ImageSplit) -> tuple[torch.Tensor, torch.Tensor]:
    return test_split.tensors()
