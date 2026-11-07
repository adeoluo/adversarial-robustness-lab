from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from adversarial_robustness_lab import data as D


def _fake_archive(path: Path, n: int = 4) -> None:
    rng = np.random.default_rng(0)
    with tarfile.open(path, "w:gz") as tar:
        names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
        for k, name in enumerate(names):
            batch = {
                b"data": rng.integers(0, 256, size=(n, 3072), dtype=np.uint8),
                b"labels": [int(v) for v in (np.arange(n) + k) % 10],
            }
            raw = pickle.dumps(batch)
            info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))


def test_read_archive_shapes_and_order(tmp_path: Path) -> None:
    archive = tmp_path / D.ARCHIVE_NAME
    _fake_archive(archive)
    data = D.read_archive(archive)
    assert data.train.images.shape == (20, 3, 32, 32)
    assert data.test.images.shape == (4, 3, 32, 32)
    assert data.train.labels[:4].tolist() == [0, 1, 2, 3]
    assert data.train.labels[4:8].tolist() == [1, 2, 3, 4]
    assert data.test.labels.tolist() == [5, 6, 7, 8]


def test_channel_major_decoding(tmp_path: Path) -> None:
    archive = tmp_path / D.ARCHIVE_NAME
    _fake_archive(archive)
    with tarfile.open(archive) as tar:
        member = tar.extractfile("cifar-10-batches-py/test_batch")
        assert member is not None
        raw = pickle.loads(member.read(), encoding="bytes")
    data = D.read_archive(archive)
    # The first 1024 bytes of a row are the red plane in row-major order.
    np.testing.assert_array_equal(data.test.images[0, 0].reshape(-1), raw[b"data"][0, :1024])
    np.testing.assert_array_equal(data.test.images[0, 2].reshape(-1), raw[b"data"][0, 2048:])


def test_load_builds_cache_and_reuses_it(tmp_path: Path) -> None:
    _fake_archive(tmp_path / D.ARCHIVE_NAME)
    first = D.load_cifar10(tmp_path, verify=False)
    assert (tmp_path / "cifar10.npz").exists()
    (tmp_path / D.ARCHIVE_NAME).unlink()
    second = D.load_cifar10(tmp_path)
    np.testing.assert_array_equal(first.train.images, second.train.images)
    np.testing.assert_array_equal(first.test.labels, second.test.labels)


def test_load_verifies_md5(tmp_path: Path) -> None:
    _fake_archive(tmp_path / D.ARCHIVE_NAME)
    with pytest.raises(ValueError, match="MD5"):
        D.load_cifar10(tmp_path, verify=True)


def test_missing_data_explains_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        D.load_cifar10(tmp_path)


def test_md5sum(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert D.md5sum(p) == "900150983cd24fb0d6963f7d28e17f72"


def test_image_split_validation() -> None:
    with pytest.raises(ValueError):
        D.ImageSplit(np.zeros((2, 32, 32, 3), np.uint8), np.zeros(2, np.uint8))
    with pytest.raises(ValueError):
        D.ImageSplit(np.zeros((2, 3, 32, 32), np.float32), np.zeros(2, np.uint8))
    with pytest.raises(ValueError):
        D.ImageSplit(np.zeros((2, 3, 32, 32), np.uint8), np.zeros(3, np.uint8))


def test_tensors_float_and_uint8() -> None:
    split = D.synthetic_split(8)
    x, y = split.tensors()
    assert x.dtype == torch.float32 and float(x.min()) >= 0 and float(x.max()) <= 1
    assert y.dtype == torch.int64
    xu, _ = split.tensors(as_uint8=True)
    assert xu.dtype == torch.uint8
    torch.testing.assert_close(D.to_float(xu), x)
    assert D.to_float(x) is x


def test_subset_copies() -> None:
    split = D.synthetic_split(10)
    sub = split.subset(range(2, 5))
    assert len(sub) == 3
    sub.images[0] = 0
    assert split.images[2].any()
    assert len(split.subset([0, 9])) == 2


def _reference_augment(
    x: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, flip: torch.Tensor, p: int
) -> torch.Tensor:
    padded = torch.nn.functional.pad(x, (p, p, p, p))
    out = []
    for i in range(x.shape[0]):
        crop = padded[i, :, int(dy[i]) : int(dy[i]) + 32, int(dx[i]) : int(dx[i]) + 32]
        out.append(crop.flip(-1) if bool(flip[i]) else crop)
    return torch.stack(out)


def test_augment_matches_loop_reference() -> None:
    x, _ = D.synthetic_split(16).tensors()
    g1 = torch.Generator().manual_seed(3)
    out = D.augment(x, 4, generator=g1)
    g2 = torch.Generator().manual_seed(3)
    dx = torch.randint(0, 9, (16,), generator=g2)
    dy = torch.randint(0, 9, (16,), generator=g2)
    flip = torch.rand(16, generator=g2) < 0.5
    torch.testing.assert_close(out, _reference_augment(x, dx, dy, flip, 4))


def test_augment_without_padding_is_flip_or_identity() -> None:
    x, _ = D.synthetic_split(8).tensors()
    out = D.augment(x, 0, generator=torch.Generator().manual_seed(0))
    for i in range(8):
        assert torch.equal(out[i], x[i]) or torch.equal(out[i], x[i].flip(-1))


@settings(max_examples=20, deadline=None)
@given(st.integers(1, 40), st.integers(1, 17), st.booleans(), st.booleans())
def test_iterate_batches_covers_everything(n: int, bs: int, shuffle: bool, drop_last: bool) -> None:
    x = torch.arange(n).float().view(n, 1)
    y = torch.arange(n)
    seen = [
        int(v)
        for _, yb in D.iterate_batches(
            x, y, bs, shuffle, torch.Generator().manual_seed(0), drop_last
        )
        for v in yb
    ]
    expected = n - (n % bs) if drop_last else n
    assert len(seen) == expected
    assert len(set(seen)) == expected
    if not shuffle:
        assert seen == list(range(expected))


def test_iterate_batches_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        list(D.iterate_batches(torch.zeros(2, 1), torch.zeros(2), 0))


def test_synthetic_split_is_deterministic() -> None:
    a, b = D.synthetic_split(32, seed=5), D.synthetic_split(32, seed=5)
    np.testing.assert_array_equal(a.images, b.images)
    assert set(np.unique(a.labels)).issubset(set(range(10)))


def test_constants() -> None:
    assert len(D.CLASSES) == 10
    assert D.ARCHIVE_URL.startswith("https://")
