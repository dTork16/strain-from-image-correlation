import numpy as np
import pytest

from src.field import StretchWithBulge
from src.speckle import speckle_image, warp, make_interpolator
from src.dic import integer_search, refine_subset, correlate, _zero_norm


class Rigid:
    def __init__(self, dx, dy):
        self.dx, self.dy = dx, dy

    def displacement(self, x, y):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, self.dx), np.full_like(x, self.dy)


def test_identical_images_give_zero_displacement():
    img = speckle_image(size=121, n_blobs=900, seed=12)
    coords, ux, uy, conv = correlate(img, img, hw=10, spacing=20)
    assert conv == 1.0
    assert np.abs(ux).max() < 1e-3
    assert np.abs(uy).max() < 1e-3


def test_integer_search_finds_a_whole_pixel_shift():
    img = speckle_image(size=121, n_blobs=900, seed=13)
    shifted = warp(img, Rigid(2.0, -1.0))
    dx, dy, c = integer_search(img, shifted, 60, 60, hw=12, search=3)
    assert (dx, dy) == (2.0, -1.0)
    assert c > 0.9


def test_subpixel_translation_recovered_to_a_hundredth_of_a_pixel():
    """No noise and no gradient, so there is nothing left to blame."""
    img = speckle_image(size=161, n_blobs=1500, seed=14)
    shifted = warp(img, Rigid(0.37, -0.62))
    coords, ux, uy, conv = correlate(img, shifted, hw=14, spacing=25)
    assert conv == 1.0
    assert np.abs(ux - 0.37).max() < 0.01
    assert np.abs(uy + 0.62).max() < 0.01


def test_full_field_recovered_to_better_than_a_twentieth_of_a_pixel():
    f = StretchWithBulge()
    ref = speckle_image(size=241, n_blobs=3500, seed=15)
    defm = warp(ref, f)
    coords, ux, uy, conv = correlate(ref, defm, hw=14, spacing=20)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    tx, ty = f.displacement(xx, yy)
    assert conv == 1.0
    assert np.abs(ux - tx).max() < 0.05
    assert np.abs(uy - ty).max() < 0.05


def test_grid_geometry_is_consistent():
    img = speckle_image(size=161, n_blobs=1500, seed=16)
    coords, ux, uy, _ = correlate(img, img, hw=12, spacing=15)
    assert ux.shape == uy.shape == (coords.size, coords.size)
    assert coords.min() >= 12
    assert coords.max() <= 161 - 12 - 1


def test_subset_too_big_for_the_image_is_refused():
    img = speckle_image(size=81, n_blobs=400, seed=17)
    with pytest.raises(ValueError):
        correlate(img, img, hw=35, spacing=12)


def test_zero_norm_returns_a_unit_vector_and_its_scale():
    a = np.array([[1.0, 2.0], [3.0, 10.0]])
    z, n = _zero_norm(a)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert np.sqrt((z * z).sum()) == pytest.approx(1.0)
    assert n == pytest.approx(np.sqrt(((a - a.mean()) ** 2).sum()))


def test_refinement_converges_from_a_deliberately_wrong_start():
    img = speckle_image(size=161, n_blobs=1500, seed=18)
    shifted = warp(img, Rigid(0.8, 0.5))
    gy, gx = np.gradient(img)
    hw = 16
    x0 = y0 = 80
    sl = (slice(y0 - hw, y0 + hw + 1), slice(x0 - hw, x0 + hw + 1))
    fhat, dnorm = _zero_norm(img[sl])
    off = np.arange(-hw, hw + 1, dtype=float)
    ox, oy = np.meshgrid(off, off, indexing="xy")
    spl = make_interpolator(shifted)
    px, py, ok = refine_subset(fhat, gx[sl], gy[sl], dnorm, spl,
                               x0 + ox, y0 + oy, (0.0, 0.0))
    assert ok
    assert px == pytest.approx(0.8, abs=0.02)
    assert py == pytest.approx(0.5, abs=0.02)
