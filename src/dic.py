"""Subset-based digital image correlation.

Each subset is matched independently. The match is first found to the nearest
pixel by normalised cross-correlation, then refined to sub-pixel by Gauss-Newton
on a zero-normalised sum of squared differences, taking the gradient from the
reference subset so that the Hessian can be built once per subset instead of
once per iteration.

The subset is allowed to translate but not to stretch or rotate. That is the
simplest shape function there is, and it is deliberate here: it is what makes
the subset size matter in the way this project is trying to measure.
"""

import numpy as np

from .speckle import make_interpolator


def _zero_norm(a):
    """Subtract the mean and divide by the L2 norm of what is left."""
    a = a - a.mean()
    n = np.sqrt((a * a).sum())
    return (a / n, n) if n > 0 else (a, 0.0)


def integer_search(ref, defm, x0, y0, hw, search=3):
    """Nearest-pixel displacement of one subset by normalised cross-correlation."""
    patch, _ = _zero_norm(ref[y0 - hw:y0 + hw + 1, x0 - hw:x0 + hw + 1])
    best, best_c = (0, 0), -np.inf
    n = ref.shape[0]
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            ya, yb = y0 + dy - hw, y0 + dy + hw + 1
            xa, xb = x0 + dx - hw, x0 + dx + hw + 1
            if ya < 0 or xa < 0 or yb > n or xb > n:
                continue
            cand, _ = _zero_norm(defm[ya:yb, xa:xb])
            c = float((patch * cand).sum())
            if c > best_c:
                best_c, best = c, (dx, dy)
    return float(best[0]), float(best[1]), best_c


def refine_subset(fhat, gradx, grady, dnorm, spl, gx, gy, guess,
                  tol=1e-5, max_iter=25):
    """Sub-pixel refinement of one subset by Gauss-Newton.

    fhat is the zero-normalised reference subset, gradx and grady its
    intensity gradients, dnorm the norm that normalised it, and gx, gy the
    subset's pixel coordinates in the image.
    """
    jx = gradx.ravel() / dnorm
    jy = grady.ravel() / dnorm
    H = np.array([[float((jx * jx).sum()), float((jx * jy).sum())],
                  [float((jx * jy).sum()), float((jy * jy).sum())]])
    if abs(np.linalg.det(H)) < 1e-14:
        return float(guess[0]), float(guess[1]), False
    Hinv = np.linalg.inv(H)

    p = np.asarray(guess, dtype=float)
    f = fhat.ravel()
    for _ in range(max_iter):
        vals = spl.ev(gy + p[1], gx + p[0])
        ghat, gn = _zero_norm(vals)
        if gn == 0:
            return float(p[0]), float(p[1]), False
        r = ghat.ravel() - f
        dp = -Hinv @ np.array([float((jx * r).sum()), float((jy * r).sum())])
        p = p + dp
        if np.hypot(dp[0], dp[1]) < tol:
            return float(p[0]), float(p[1]), True
    return float(p[0]), float(p[1]), False


def correlate(ref, defm, hw, spacing=12, margin=None, search=3):
    """Run DIC over a grid of subsets.

    Returns the grid coordinates, the measured displacement fields shaped
    (ny, nx), and the fraction of subsets whose refinement converged.
    """
    n = ref.shape[0]
    if margin is None:
        margin = hw + search + 2
    coords = np.arange(margin, n - margin + 1, spacing)
    if coords.size < 4:
        raise ValueError("subset too large for this image and spacing")

    spl = make_interpolator(defm)
    gy_img, gx_img = np.gradient(ref)

    off = np.arange(-hw, hw + 1, dtype=float)
    ox, oy = np.meshgrid(off, off, indexing="xy")

    ux = np.zeros((coords.size, coords.size))
    uy = np.zeros((coords.size, coords.size))
    converged = 0

    for j, y0 in enumerate(coords):
        y0 = int(y0)
        for i, x0 in enumerate(coords):
            x0 = int(x0)
            sl = (slice(y0 - hw, y0 + hw + 1), slice(x0 - hw, x0 + hw + 1))
            fhat, dnorm = _zero_norm(ref[sl])
            dx, dy, _ = integer_search(ref, defm, x0, y0, hw, search)
            rx, ry, ok = refine_subset(
                fhat, gx_img[sl], gy_img[sl], dnorm, spl,
                x0 + ox, y0 + oy, (dx, dy))
            ux[j, i] = rx
            uy[j, i] = ry
            converged += int(ok)

    return coords.astype(float), ux, uy, converged / ux.size
