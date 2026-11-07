from __future__ import annotations

import pytest
import torch

from adversarial_robustness_lab import models as M


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("small_cnn", 591_274),
        ("wrn_16_2", 691_674),
        ("wrn_16_4", 2_748_890),
        ("preact_resnet18", 11_172_170),
    ],
)
def test_parameter_counts(name: str, params: int) -> None:
    assert M.count_parameters(M.build_model(name)) == params


@pytest.mark.parametrize("name", sorted(M.ARCHITECTURES))
def test_forward_shape(name: str) -> None:
    model = M.build_model(name).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_wrn_16_8_and_custom_classes() -> None:
    assert M.count_parameters(M.wrn_16_8()) > M.count_parameters(M.wrn_16_4())
    assert M.SmallCNN(num_classes=3)(torch.rand(1, 3, 32, 32)).shape == (1, 3)


def test_wideresnet_depth_validation() -> None:
    with pytest.raises(ValueError):
        M.WideResNet(depth=15)


def test_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        M.build_model("vgg")


def test_normalize_layer() -> None:
    norm = M.Normalize()
    x = torch.rand(4, 3, 32, 32)
    mean = torch.tensor(M.CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(M.CIFAR10_STD).view(1, 3, 1, 1)
    torch.testing.assert_close(norm(x), (x - mean) / std)


def test_normalize_is_not_trainable() -> None:
    assert all(p is not None for p in M.Normalize().buffers())
    assert list(M.Normalize().parameters()) == []


def test_preact_block_dropout_and_shortcut() -> None:
    block = M.PreActBlock(16, 32, stride=2, dropout=0.5).train()
    assert block.shortcut is not None
    assert block(torch.rand(2, 16, 8, 8)).shape == (2, 32, 4, 4)
    same = M.PreActBlock(8, 8)
    assert same.shortcut is None
