# Data card: CIFAR-10

| Field | Value |
|---|---|
| Source | Krizhevsky (2009), "Learning Multiple Layers of Features from Tiny Images"; https://www.cs.toronto.edu/~kriz/cifar.html |
| File | `cifar-10-python.tar.gz`, 170,498,071 bytes, MD5 `c58f30108f718f92721af3b95e74349a` (checked on load) |
| Size | 50,000 training and 10,000 test images, 32 x 32 RGB, 10 classes, balanced (5,000 and 1,000 per class) |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Location | `../.bench-data/adversarial-robustness-lab/` (outside the repository; never committed) |
| Cache | `cifar10.npz`, uint8 NCHW, built on first load |

## Use in this project

- **Training:** all 50,000 training images, random crop (4-pixel zero padding) and
  horizontal flip. No validation split is carved out: models are trained for a fixed
  schedule and the last epoch is evaluated; there is no early stopping or model
  selection on test data.
- **Evaluation:** all 10,000 test images for clean and robust accuracy. Costly
  comparisons use the first 1,000 test images and certification every 20th test image
  (500 images; Cohen et al. certified the full CIFAR-10 test set with 100,000 samples
  per image, 200 times this sampling budget). The subsets are fixed and stated with
  each table.
- **Tests and CI:** a synthetic, CIFAR-shaped stand-in (`synthetic_split`), so CI needs
  no download.

## Known issues

- CIFAR-10 contains near-duplicate images between the training and test sets (Barz and
  Denzler, 2020, report about 3.3% of test images with a near-duplicate in training);
  this affects all published CIFAR-10 numbers equally and is not corrected here.
- Images are low resolution; robustness at 8/255 on 32 x 32 images does not transfer to
  higher-resolution domains.
- No personal data; the images were collected from the web by the dataset authors.
