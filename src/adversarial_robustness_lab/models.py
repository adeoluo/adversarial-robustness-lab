"""Classifiers for CIFAR-10: PreAct ResNet-18, WideResNet and a small CNN.

Every model takes images in ``[0, 1]`` and standardizes them in its first layer, so the
threat model (an L-infinity or L2 ball around the pixel image) never depends on how the
network was trained.

* **PreAct ResNet-18** (He et al., 2016, "Identity Mappings in Deep Residual Networks"):
  batch norm and ReLU come before each convolution and the shortcut is a pure identity.
  It is the standard model of the "Fast is better than free" (Wong et al., 2020) and
  "Overfitting in adversarially robust deep learning" (Rice et al., 2020) baselines.
* **WideResNet-d-k** (Zagoruyko and Komodakis, 2016): ``(d - 4) / 6`` pre-activation
  basic blocks per stage, stage widths ``16k, 32k, 64k``. WRN-34-10 is the model of
  Madry et al. and TRADES; WRN-16-4 is the scaled-down version that fits the M1 budget.
* **SmallCNN**: four convolutions and two dense layers, used for tests, sanity checks
  and the quick demos.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from adversarial_robustness_lab.data import CIFAR10_MEAN, CIFAR10_STD


class Normalize(nn.Module):
    """Per-channel standardization ``(x - mean) / std`` as a fixed layer."""

    def __init__(
        self,
        mean: tuple[float, ...] = CIFAR10_MEAN,
        std: tuple[float, ...] = CIFAR10_STD,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class PreActBlock(nn.Module):
    """Pre-activation basic block: BN, ReLU, conv3x3, BN, ReLU, conv3x3, plus shortcut."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.dropout = dropout
        self.shortcut: nn.Module | None = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class PreActResNet(nn.Module):
    """Pre-activation ResNet for 32 x 32 inputs (four stages, widths 64 to 512)."""

    def __init__(self, num_blocks: tuple[int, int, int, int], num_classes: int = 10) -> None:
        super().__init__()
        self.normalize = Normalize()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.normalize(x))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


class WideResNet(nn.Module):
    """WideResNet-depth-width with pre-activation blocks (Zagoruyko and Komodakis, 2016)."""

    def __init__(
        self, depth: int = 16, width: int = 4, num_classes: int = 10, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must be 6n + 4")
        n = (depth - 4) // 6
        widths = [16, 16 * width, 32 * width, 64 * width]
        self.normalize = Normalize()
        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        stages: list[nn.Module] = []
        in_planes = widths[0]
        for stage, planes in enumerate(widths[1:]):
            blocks: list[nn.Module] = []
            for i in range(n):
                stride = (1 if stage == 0 else 2) if i == 0 else 1
                blocks.append(PreActBlock(in_planes, planes, stride, dropout))
                in_planes = planes
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.bn = nn.BatchNorm2d(widths[3])
        self.linear = nn.Linear(widths[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stages(self.conv1(self.normalize(x)))
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


class SmallCNN(nn.Module):
    """A four-convolution network (about 0.3 M parameters) for tests and demos."""

    def __init__(self, num_classes: int = 10, width: int = 32) -> None:
        super().__init__()
        self.normalize = Normalize()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * width * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(self.normalize(x)))


def preact_resnet18(num_classes: int = 10) -> PreActResNet:
    return PreActResNet((2, 2, 2, 2), num_classes)


def wrn_16_2(num_classes: int = 10) -> WideResNet:
    return WideResNet(16, 2, num_classes)


def wrn_16_4(num_classes: int = 10) -> WideResNet:
    return WideResNet(16, 4, num_classes)


def wrn_16_8(num_classes: int = 10) -> WideResNet:
    return WideResNet(16, 8, num_classes)


def small_cnn(num_classes: int = 10) -> SmallCNN:
    return SmallCNN(num_classes)


ARCHITECTURES: dict[str, Callable[[int], nn.Module]] = {
    "preact_resnet18": preact_resnet18,
    "wrn_16_2": wrn_16_2,
    "wrn_16_4": wrn_16_4,
    "wrn_16_8": wrn_16_8,
    "small_cnn": small_cnn,
}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Instantiate a registered architecture by name."""
    try:
        factory = ARCHITECTURES[name]
    except KeyError as exc:
        raise ValueError(f"unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)}") from exc
    return factory(num_classes)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
