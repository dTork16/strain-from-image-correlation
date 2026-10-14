"""Strain from a measured displacement field.

Strain is the derivative of displacement, and differentiating measured data
amplifies whatever noise is in it. The standard way round that is to fit a
local plane to the displacements over a window of grid points and take the
gradient of the fitted plane instead of differencing neighbours. The window
size is the smoothing knob, and this module exists so that knob can be turned.
"""

import numpy as np


def pointwise_least_squares(coords, ux, uy, window):
    """Local plane fit for the displacement gradients.

    window is the number of grid points on a side and must be odd and >= 3.
    Points too close to the edge to carry a full window come back as NaN.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and at least 3")

    half = window // 2
    ny, nx = ux.shape
    if window > min(ny, nx):
        raise ValueError("window larger than the measurement grid")

    exx = np.full((ny, nx), np.nan)
    eyy = np.full((ny, nx), np.nan)
    exy = np.full((ny, nx), np.nan)

    off = np.arange(-half, half + 1)
    step = coords[1] - coords[0]
    dX, dY = np.meshgrid(off * step, off * step, indexing="xy")
    A = np.column_stack([np.ones(dX.size), dX.ravel(), dY.ravel()])
    pinv = np.linalg.pinv(A)

    for j in range(half, ny - half):
        for i in range(half, nx - half):
            bx = ux[j - half:j + half + 1, i - half:i + half + 1].ravel()
            by = uy[j - half:j + half + 1, i - half:i + half + 1].ravel()
            cx = pinv @ bx
            cy = pinv @ by
            exx[j, i] = cx[1]
            eyy[j, i] = cy[2]
            exy[j, i] = 0.5 * (cx[2] + cy[1])

    return exx, eyy, exy


def rms(a, b):
    """RMS difference over the entries where both are finite."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.nan
    d = a[ok] - b[ok]
    return float(np.sqrt((d * d).mean()))
