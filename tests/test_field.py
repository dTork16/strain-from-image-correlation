import numpy as np
import pytest

from src.field import StretchWithBulge


def test_analytic_strain_matches_numerical_derivative():
    """The strain I differentiate by hand has to agree with finite differences."""
    f = StretchWithBulge()
    h = 1e-4
    pts = [(80.0, 90.0), (150.0, 150.0), (160.0, 140.0), (240.0, 60.0)]
    for x, y in pts:
        ux_p, _ = f.displacement(x + h, y)
        ux_m, _ = f.displacement(x - h, y)
        dux_dx = (ux_p - ux_m) / (2 * h)

        _, uy_p = f.displacement(x, y + h)
        _, uy_m = f.displacement(x, y - h)
        duy_dy = (uy_p - uy_m) / (2 * h)

        ux_yp, _ = f.displacement(x, y + h)
        ux_ym, _ = f.displacement(x, y - h)
        dux_dy = (ux_yp - ux_ym) / (2 * h)

        exx, eyy, exy = f.strain(x, y)
        assert exx == pytest.approx(dux_dx, abs=1e-8)
        assert eyy == pytest.approx(duy_dy, abs=1e-8)
        assert exy == pytest.approx(0.5 * dux_dy, abs=1e-8)


def test_far_from_bulge_is_uniform_stretch():
    f = StretchWithBulge(e0=0.002, nu=0.33)
    exx, eyy, exy = f.strain(5.0, 5.0)
    assert exx == pytest.approx(0.002, abs=1e-6)
    assert eyy == pytest.approx(-0.33 * 0.002, abs=1e-12)
    assert exy == pytest.approx(0.0, abs=1e-6)


def test_concentration_is_well_above_background():
    """If the bulge barely perturbs the strain there is nothing to measure."""
    f = StretchWithBulge()
    grid = np.arange(20.0, 282.0, 4.0)
    peak = f.peak_exx(grid)
    assert peak > 4 * f.e0


def test_displacements_stay_small_enough_for_the_search_window():
    f = StretchWithBulge()
    g = np.arange(0.0, 301.0, 5.0)
    xx, yy = np.meshgrid(g, g, indexing="xy")
    ux, uy = f.displacement(xx, yy)
    assert np.abs(ux).max() < 3.0
    assert np.abs(uy).max() < 3.0
