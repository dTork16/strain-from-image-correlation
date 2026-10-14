import numpy as np
import pytest

from src.field import StretchWithBulge
from src.speckle import speckle_image, warp, add_noise


class Zero:
    def displacement(self, x, y):
        return np.zeros_like(np.asarray(x, dtype=float)), \
               np.zeros_like(np.asarray(y, dtype=float))


class Rigid:
    def __init__(self, dx, dy):
        self.dx, self.dy = dx, dy

    def displacement(self, x, y):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, self.dx), np.full_like(x, self.dy)


def test_image_has_contrast_and_sits_in_range():
    img = speckle_image(size=161, n_blobs=1500, seed=3)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.std() > 0.05


def test_same_seed_repeats_and_different_seed_does_not():
    a = speckle_image(size=81, n_blobs=400, seed=7)
    b = speckle_image(size=81, n_blobs=400, seed=7)
    c = speckle_image(size=81, n_blobs=400, seed=8)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_zero_field_leaves_the_image_alone():
    img = speckle_image(size=121, n_blobs=900, seed=4)
    out = warp(img, Zero())
    inner = (slice(6, -6), slice(6, -6))
    assert np.abs(out[inner] - img[inner]).max() < 1e-6


def test_rigid_translation_shifts_by_the_right_whole_pixels():
    """out[y, x] samples the reference at (x - ux, y - uy), so check that."""
    n, pad = 121, 10
    ux, uy = 3.0, -2.0
    img = speckle_image(size=n, n_blobs=900, seed=5)
    out = warp(img, Rigid(ux, uy))

    inner_out = out[pad:n - pad, pad:n - pad]
    y0, x0 = pad - int(uy), pad - int(ux)
    inner_ref = img[y0:y0 + (n - 2 * pad), x0:x0 + (n - 2 * pad)]
    assert inner_out.shape == inner_ref.shape
    assert np.abs(inner_out - inner_ref).max() < 1e-6


def test_warp_actually_changes_the_image_for_a_real_field():
    img = speckle_image(size=161, n_blobs=1500, seed=6)
    out = warp(img, StretchWithBulge())
    inner = (slice(10, -10), slice(10, -10))
    assert np.abs(out[inner] - img[inner]).max() > 0.05


def test_noise_has_the_requested_spread():
    img = speckle_image(size=161, n_blobs=1500, seed=9)
    noisy = add_noise(img, 0.02, seed=11)
    assert (noisy - img).std() == pytest.approx(0.02, rel=0.05)
