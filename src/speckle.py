"""Synthetic speckle images and the warp that produces the deformed frame.

The reference image is a sum of Gaussian blobs at random positions, which is
the usual stand-in for a sprayed speckle pattern. The deformed image is made
by sampling the reference at back-mapped coordinates, so the two frames carry
exactly the displacement field in field.py and nothing else except noise.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


def speckle_image(size=301, n_blobs=5500, radius=2.2, seed=0):
    """Reference image: Gaussian blobs on a mid-grey background, range [0, 1]."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=float)

    half = int(np.ceil(3 * radius))
    off = np.arange(-half, half + 1)
    dy, dx = np.meshgrid(off, off, indexing="ij")

    cx = rng.uniform(0, size, n_blobs)
    cy = rng.uniform(0, size, n_blobs)
    amp = rng.uniform(0.5, 1.0, n_blobs)

    ix = np.round(cx).astype(int)
    iy = np.round(cy).astype(int)

    for k in range(n_blobs):
        ys = iy[k] + dy
        xs = ix[k] + dx
        ok = (ys >= 0) & (ys < size) & (xs >= 0) & (xs < size)
        r2 = (xs - cx[k]) ** 2 + (ys - cy[k]) ** 2
        stamp = amp[k] * np.exp(-r2 / (2.0 * radius ** 2))
        np.add.at(img, (ys[ok], xs[ok]), stamp[ok])

    img = img / img.max()
    return 0.08 + 0.84 * img


def make_interpolator(img):
    """Bicubic interpolator over an image, indexed as f(y, x)."""
    n = img.shape[0]
    ax = np.arange(n, dtype=float)
    return RectBivariateSpline(ax, ax, img, kx=3, ky=3, s=0)


def warp(img, fieldmodel, n_iter=4):
    """Deformed image.

    The field gives the displacement of a point at its reference position.
    To fill a pixel of the deformed image at x we need the reference position
    X that landed there, which satisfies X + u(X) = x. That is solved by a few
    fixed-point passes rather than assumed to be x - u(x), because at these
    strains the difference is not quite negligible.
    """
    n = img.shape[0]
    ax = np.arange(n, dtype=float)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")

    Xx, Xy = xx.copy(), yy.copy()
    for _ in range(n_iter):
        ux, uy = fieldmodel.displacement(Xx, Xy)
        Xx = xx - ux
        Xy = yy - uy

    spl = make_interpolator(img)
    out = spl.ev(np.clip(Xy, 0, n - 1), np.clip(Xx, 0, n - 1))
    return out


def add_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return img + rng.normal(0.0, sigma, img.shape)
