# ADR-0006: PyTorch for every attack, no compiled extension

- **Status:** Accepted
- **Date:** 2026-11-07

## Context
Several projects in this portfolio compile a Rust core because their hot loops are
scalar code (tokenization, hashing, n-gram scoring). Here the hot loops are forward and
backward passes of convolutional networks: an attack step is one forward and one
backward pass, and the rest (projection, sign, momentum, step-size bookkeeping) is a
handful of elementwise tensor operations on the same device.

## Decision
- Implement every attack, training method and the smoothing sampler with batched
  PyTorch tensor operations on the model's device (MPS on the M1, CUDA or CPU
  elsewhere); avoid per-sample Python loops. Square draws its windows for the whole
  batch with masks instead of looping over images, as the original NumPy code does.
- Keep the scalar statistics (incomplete beta, Clopper-Pearson) in Python: they run once
  per certified image, microseconds against seconds of sampling.

## Alternatives considered
- **A Rust or C++ extension for the projection and step logic**: would move data off
  the GPU and back, slower than the fused tensor operations it replaces.
- **torch.compile**: not reliable on MPS at the time of writing; revisit on CUDA.

## Consequences
- Attack cost is reported per model in GPU seconds and matches torchattacks within the
  noise of the shared machine (see the comparison table).
- No toolchain beyond Python is needed to install the package.
