import numpy as np
import pytest

from src.strain import pointwise_least_squares, rms


def _linear_grid(spacing=12.0, n=9, a=0.01, b=0.02, c=0.005, d=-0.03):
    coords = np.arange(n) * spacing + 20.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    ux = 3.0 + a * xx + b * yy
    uy = -1.0 + c * xx + d * yy
    return coords, ux, uy, (a, b, c, d)


def test_exactly_linear_displacements_give_the_exact_gradients():
    """A plane fit to a plane has to come back with the plane's slopes."""
    coords, ux, uy, (a, b, c, d) = _linear_grid()
    exx, eyy, exy = pointwise_least_squares(coords, ux, uy, 5)
    ok = np.isfinite(exx)
    assert np.allclose(exx[ok], a, atol=1e-10)
    assert np.allclose(eyy[ok], d, atol=1e-10)
    assert np.allclose(exy[ok], 0.5 * (b + c), atol=1e-10)


def test_window_size_does_not_change_the_answer_on_a_plane():
    coords, ux, uy, (a, _, _, d) = _linear_grid(n=13)
    for w in (3, 5, 7, 9):
        exx, eyy, _ = pointwise_least_squares(coords, ux, uy, w)
        ok = np.isfinite(exx)
        assert np.allclose(exx[ok], a, atol=1e-10)
        assert np.allclose(eyy[ok], d, atol=1e-10)


def test_border_is_nan_exactly_as_wide_as_the_half_window():
    coords, ux, uy, _ = _linear_grid(n=9)
    exx, _, _ = pointwise_least_squares(coords, ux, uy, 5)
    assert np.isnan(exx[:2, :]).all()
    assert np.isnan(exx[-2:, :]).all()
    assert np.isnan(exx[:, :2]).all()
    assert np.isnan(exx[:, -2:]).all()
    assert np.isfinite(exx[2:-2, 2:-2]).all()


def test_bad_windows_are_refused():
    coords, ux, uy, _ = _linear_grid(n=9)
    for bad in (2, 4, 1, 0):
        with pytest.raises(ValueError):
            pointwise_least_squares(coords, ux, uy, bad)
    with pytest.raises(ValueError):
        pointwise_least_squares(coords, ux, uy, 11)


def test_smoothing_flattens_a_peak_rather_than_preserving_it():
    """Sanity check on the mechanism the experiment is measuring."""
    spacing = 12.0
    coords = np.arange(21) * spacing
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    s = 30.0
    ux = 0.4 * np.exp(-((xx - 120.0) ** 2 + (yy - 120.0) ** 2) / (2 * s ** 2))
    uy = np.zeros_like(ux)
    peaks = []
    for w in (3, 7, 11):
        exx, _, _ = pointwise_least_squares(coords, ux, uy, w)
        peaks.append(np.nanmax(np.abs(exx)))
    assert peaks[0] > peaks[1] > peaks[2]


def test_rms_ignores_nan_and_is_zero_for_a_perfect_match():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.0, 99.0, 4.0])
    assert rms(a, b) == pytest.approx(0.0)
    c = np.array([1.0, 3.0, np.nan, 4.0])
    assert rms(a, c) == pytest.approx(np.sqrt(1.0 / 3.0))
    assert np.isnan(rms(np.array([np.nan]), np.array([1.0])))
