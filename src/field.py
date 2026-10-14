"""The displacement field used as ground truth.

A uniform stretch in x with a Poisson contraction in y, plus a localised
bulge that creates a strain concentration in the middle of the frame.
Everything here has an analytic derivative, so the strain that the image
correlation is trying to recover is known exactly rather than estimated.
"""

import numpy as np


class StretchWithBulge:
    def __init__(self, e0=0.002, nu=0.33, amp=0.4, sigma=30.0,
                 centre=(150.0, 150.0), rigid=(1.3, -0.7)):
        self.e0 = e0
        self.nu = nu
        self.amp = amp
        self.sigma = sigma
        self.centre = centre
        self.rigid = rigid

    def _bump(self, x, y):
        cx, cy = self.centre
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        return np.exp(-r2 / (2.0 * self.sigma ** 2))

    def displacement(self, x, y):
        """Displacement (ux, uy) in pixels at reference coordinates (x, y)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        g = self._bump(x, y)
        ux = self.rigid[0] + self.e0 * x + self.amp * g
        uy = self.rigid[1] - self.nu * self.e0 * y
        return ux, uy

    def strain(self, x, y):
        """Analytic small-strain tensor components (exx, eyy, exy)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cx, cy = self.centre
        g = self._bump(x, y)
        s2 = self.sigma ** 2

        dux_dx = self.e0 - self.amp * (x - cx) / s2 * g
        dux_dy = -self.amp * (y - cy) / s2 * g
        duy_dx = np.zeros_like(x)
        duy_dy = np.full_like(y, -self.nu * self.e0)

        exx = dux_dx
        eyy = duy_dy
        exy = 0.5 * (dux_dy + duy_dx)
        return exx, eyy, exy

    def peak_exx(self, grid):
        """Largest exx anywhere on a given coordinate grid."""
        xx, yy = np.meshgrid(grid, grid, indexing="xy")
        exx, _, _ = self.strain(xx, yy)
        return float(exx.max())
