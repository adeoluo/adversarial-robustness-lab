"""CIFAR-10 loading, caching and on-device augmentation.

The official ``cifar-10-python.tar.gz`` archive (Krizhevsky, 2009) holds five training
batches of 10,000 images and one test batch of 10,000 images, each a pickled dict with a
``uint8`` array of shape ``(10000, 3072)`` in channel-major order. We decode it once into
a compressed ``.npz`` cache of ``uint8`` arrays in NCHW layout (about 180 MB in memory)
and keep pixels in ``[0, 1]`` after conversion: every attack in this package works in
pixel space, and per-channel standardization lives inside the model (see
:class:`adversarial_robustness_lab.models.Normalize`), so epsilon is always measured in
the units of the image a person would see (for example 8/255).

Augmentation (random crop with 4 pixels of zero padding, horizontal flip) runs as tensor
operations on the training device, which is several times faster than a ``DataLoader``
with per-image transforms on Apple silicon and keeps peak memory predictable.
"""

from __future__ import annotations

import hashlib
import pickle
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2471, 0.2435, 0.2616)
ARCHIVE_NAME = "cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
ARCHIVE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[3] / ".bench-data" / "adversarial-robustness-lab"
)


@dataclass(frozen=True)
class ImageSplit:
    """A split of images as ``uint8`` NCHW and integer labels."""

    images: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError(f"images must be N x 3 x H x W, got {self.images.shape}")
        if self.images.dtype != np.uint8:
            raise ValueError(f"images must be uint8, got {self.images.dtype}")
        if self.labels.shape != (self.images.shape[0],):
            raise ValueError("labels must have one entry per image")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def subset(self, indices: np.ndarray | list[int] | range) -> ImageSplit:
        """Return the rows at ``indices`` (copied)."""
        idx = np.asarray(list(indices) if isinstance(indices, range) else indices)
        return ImageSplit(self.images[idx].copy(), self.labels[idx].copy())

    def tensors(
        self, device: torch.device | str = "cpu", as_uint8: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Images as float32 in ``[0, 1]`` (or raw ``uint8``) and labels as int64, on ``device``.

        Keeping the 50,000 training images on the device as ``uint8`` takes 154 MB instead
        of 614 MB as float32; :func:`to_float` converts one batch at a time.
        """
        x = torch.from_numpy(self.images).to(device)
        if not as_uint8:
            x = to_float(x)
        y = torch.from_numpy(self.labels.astype(np.int64)).to(device)
        return x, y


@dataclass(frozen=True)
class Cifar10:
    """The full CIFAR-10 dataset: 50,000 training and 10,000 test images."""

    train: ImageSplit
    test: ImageSplit


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """Hex MD5 of a file, read in chunks."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _decode_batch(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    # The archive is the published dataset; the pickles only hold numpy arrays and lists.
    batch = pickle.loads(raw, encoding="bytes")
    data = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"labels"], dtype=np.uint8)
    return data, labels


def read_archive(archive: Path) -> Cifar10:
    """Decode the official python archive into train and test splits."""
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    test: tuple[np.ndarray, np.ndarray] | None = None
    with tarfile.open(archive, "r:gz") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for i in range(1, 6):
            handle = tar.extractfile(members[f"data_batch_{i}"])
            if handle is None:  # pragma: no cover - only for corrupt archives
                raise ValueError(f"data_batch_{i} missing from {archive}")
            x, y = _decode_batch(handle.read())
            train_x.append(x)
            train_y.append(y)
        handle = tar.extractfile(members["test_batch"])
        if handle is None:  # pragma: no cover
            raise ValueError(f"test_batch missing from {archive}")
        test = _decode_batch(handle.read())
    return Cifar10(
        train=ImageSplit(np.concatenate(train_x), np.concatenate(train_y)),
        test=ImageSplit(*test),
    )


def load_cifar10(data_dir: Path | str = DEFAULT_DATA_DIR, verify: bool = True) -> Cifar10:
    """Load CIFAR-10 from ``data_dir``, building the ``cifar10.npz`` cache on first use.

    Raises ``FileNotFoundError`` with the download command when neither the cache nor
    the archive is present.
    """
    root = Path(data_dir)
    cache = root / "cifar10.npz"
    if cache.exists():
        with np.load(cache) as z:
            return Cifar10(
                ImageSplit(z["train_x"], z["train_y"]), ImageSplit(z["test_x"], z["test_y"])
            )
    archive = root / ARCHIVE_NAME
    if not archive.exists():
        raise FileNotFoundError(
            f"{archive} not found; run `make data` (downloads {ARCHIVE_URL}, 163 MiB)"
        )
    if verify and md5sum(archive) != ARCHIVE_MD5:
        raise ValueError(f"{archive} does not match the published MD5 {ARCHIVE_MD5}")
    data = read_archive(archive)
    np.savez_compressed(
        cache,
        train_x=data.train.images,
        train_y=data.train.labels,
        test_x=data.test.images,
        test_y=data.test.labels,
    )
    return data


def to_float(x: torch.Tensor) -> torch.Tensor:
    """``uint8`` pixels to float32 in ``[0, 1]``; float input is returned unchanged."""
    if x.dtype == torch.uint8:
        return x.to(torch.float32).div_(255.0)
    return x


def augment(
    x: torch.Tensor, padding: int = 4, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Random crop after zero padding, then random horizontal flip, for a batch on any device.

    Each image gets its own offset in ``[0, 2 * padding]`` along both axes and its own
    flip, drawn on the CPU generator so runs are reproducible across devices.
    """
    n, _, h, w = x.shape
    padded = torch.nn.functional.pad(x, (padding, padding, padding, padding))
    dx = torch.randint(0, 2 * padding + 1, (n,), generator=generator).to(x.device)
    dy = torch.randint(0, 2 * padding + 1, (n,), generator=generator).to(x.device)
    flip = (torch.rand(n, generator=generator) < 0.5).to(x.device)
    rows = torch.arange(h, device=x.device).view(1, h) + dy.view(n, 1)
    cols = torch.arange(w, device=x.device).view(1, w) + dx.view(n, 1)
    cols = torch.where(flip.view(n, 1), cols.flip(1), cols)
    batch = torch.arange(n, device=x.device).view(n, 1, 1)
    # Gather rows then columns: (n, c, h, W+2p) -> (n, c, h, w)
    out = padded.permute(0, 2, 3, 1)[batch, rows.view(n, h, 1), cols.view(n, 1, w)]
    return out.permute(0, 3, 1, 2).contiguous()


def iterate_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    drop_last: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(x, y)`` mini-batches, optionally in a seeded random order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = x.shape[0]
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    order = order.to(x.device)
    stop = n - (n % batch_size) if drop_last else n
    for start in range(0, stop, batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def synthetic_split(n: int = 64, seed: int = 0, num_classes: int = 10) -> ImageSplit:
    """A small learnable stand-in for CIFAR-10 used by tests and the CI demo.

    Each class is a distinct mean colour plus a class-specific stripe, with noise, so a
    small network separates it in a few steps.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n).astype(np.uint8)
    # Class prototypes are fixed; only labels and noise depend on the seed.
    base = np.random.default_rng(1234).uniform(40, 215, size=(num_classes, 3, 1, 1))
    images = np.repeat(np.repeat(base[labels], 32, axis=2), 32, axis=3)
    for i, label in enumerate(labels):
        images[i, :, int(label) * 3 : int(label) * 3 + 2, :] += 60
    images = images + rng.normal(0, 12, size=images.shape)
    return ImageSplit(np.clip(images, 0, 255).astype(np.uint8), labels)
