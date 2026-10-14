"""Runs the two sweeps and writes the figures and the results table.

Sweep one varies the correlation subset size and asks what it costs the
displacement field and what it costs the strain field, which are not the
same question. Sweep two holds the subset fixed and varies the smoothing
window used to differentiate the displacements, and tracks both the RMS
strain error and the height of the strain concentration.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .field import StretchWithBulge
from .speckle import speckle_image, warp, add_noise
from .dic import correlate
from .strain import pointwise_least_squares, rms

HALF_WIDTHS = [7, 10, 14, 18, 23, 28]
STRAIN_WINDOWS = [3, 5, 7, 9, 11]
NOISE = 0.01
REPLICATES = 3
SPACING = 12
FIXED_HW = 14
FIXED_WINDOW = 5
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")


def truth_on(coords, fieldmodel):
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    tx, ty = fieldmodel.displacement(xx, yy)
    exx, eyy, exy = fieldmodel.strain(xx, yy)
    return tx, ty, exx, eyy, exy


def valid_peak(measured, truth):
    """Peak of the measured field and of the truth at the same valid points."""
    ok = np.isfinite(measured)
    return float(measured[ok].max()), float(truth[ok].max())


def sweep_subset_size(fieldmodel, ref, defm):
    rows = []
    for hw in HALF_WIDTHS:
        dx_errs, ex_errs = [], []
        for k in range(REPLICATES):
            r = add_noise(ref, NOISE, seed=100 + k)
            d = add_noise(defm, NOISE, seed=200 + k)
            coords, ux, uy, conv = correlate(r, d, hw=hw, spacing=SPACING)
            tx, ty, Txx, _, _ = truth_on(coords, fieldmodel)
            exx, _, _ = pointwise_least_squares(coords, ux, uy, FIXED_WINDOW)
            dx_errs.append(rms(ux, tx))
            ex_errs.append(rms(exx, Txx))

        coords, ux, uy, _ = correlate(ref, defm, hw=hw, spacing=SPACING)
        tx, ty, Txx, _, _ = truth_on(coords, fieldmodel)
        exx, _, _ = pointwise_least_squares(coords, ux, uy, FIXED_WINDOW)

        rows.append({
            "half_width": hw,
            "subset_px": 2 * hw + 1,
            "disp_rms_px": float(np.mean(dx_errs)),
            "disp_rms_sd": float(np.std(dx_errs)),
            "strain_rms": float(np.mean(ex_errs)),
            "strain_rms_sd": float(np.std(ex_errs)),
            "disp_rms_noiseless_px": rms(ux, tx),
            "strain_rms_noiseless": rms(exx, Txx),
        })
        print("  subset %3d px   disp %.4f px   strain %.6f   (noiseless strain %.6f)"
              % (rows[-1]["subset_px"], rows[-1]["disp_rms_px"],
                 rows[-1]["strain_rms"], rows[-1]["strain_rms_noiseless"]))
    return rows


def sweep_strain_window(fieldmodel, ref, defm):
    rows = []
    per_rep = {w: {"rms": [], "peak": []} for w in STRAIN_WINDOWS}
    for k in range(REPLICATES):
        r = add_noise(ref, NOISE, seed=100 + k)
        d = add_noise(defm, NOISE, seed=200 + k)
        coords, ux, uy, _ = correlate(r, d, hw=FIXED_HW, spacing=SPACING)
        _, _, Txx, _, _ = truth_on(coords, fieldmodel)
        for w in STRAIN_WINDOWS:
            exx, _, _ = pointwise_least_squares(coords, ux, uy, w)
            pk, pk_true = valid_peak(exx, Txx)
            per_rep[w]["rms"].append(rms(exx, Txx))
            per_rep[w]["peak"].append(pk / pk_true)

    for w in STRAIN_WINDOWS:
        rows.append({
            "window": w,
            "span_px": (w - 1) * SPACING,
            "strain_rms": float(np.mean(per_rep[w]["rms"])),
            "strain_rms_sd": float(np.std(per_rep[w]["rms"])),
            "peak_ratio": float(np.mean(per_rep[w]["peak"])),
            "peak_ratio_sd": float(np.std(per_rep[w]["peak"])),
        })
        print("  window %2d (%3d px)   strain rms %.6f   peak recovered %.3f"
              % (w, rows[-1]["span_px"], rows[-1]["strain_rms"], rows[-1]["peak_ratio"]))
    return rows


def figure_fields(fieldmodel, ref, defm):
    r = add_noise(ref, NOISE, seed=100)
    d = add_noise(defm, NOISE, seed=200)
    coords, ux, uy, _ = correlate(r, d, hw=FIXED_HW, spacing=SPACING)
    _, _, Txx, _, _ = truth_on(coords, fieldmodel)
    exx, _, _ = pointwise_least_squares(coords, ux, uy, FIXED_WINDOW)

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    ax[0].imshow(ref, cmap="gray")
    ax[0].set_title("reference speckle")
    ax[1].imshow(ux, cmap="viridis", origin="lower")
    ax[1].set_title("measured $u_x$ (px)")
    lim = np.nanmax(np.abs(Txx))
    im2 = ax[2].imshow(Txx, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    ax[2].set_title("true $\\epsilon_{xx}$")
    im3 = ax[3].imshow(exx, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    ax[3].set_title("measured $\\epsilon_{xx}$")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im2, ax=ax[2], fraction=0.046)
    fig.colorbar(im3, ax=ax[3], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fields.png"), dpi=130)
    plt.close(fig)


def figure_sweeps(size_rows, window_rows):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    s = [r["subset_px"] for r in size_rows]
    ax[0].errorbar(s, [r["disp_rms_px"] for r in size_rows],
                   yerr=[r["disp_rms_sd"] for r in size_rows], marker="o")
    ax[0].set_xlabel("subset size (px)")
    ax[0].set_ylabel("RMS displacement error (px)")
    ax[0].set_title("displacement")
    ax[0].grid(alpha=0.3)

    ax[1].errorbar(s, [r["strain_rms"] for r in size_rows],
                   yerr=[r["strain_rms_sd"] for r in size_rows],
                   marker="o", label="with noise")
    ax[1].plot(s, [r["strain_rms_noiseless"] for r in size_rows],
               marker="s", ls="--", label="noiseless")
    ax[1].set_xlabel("subset size (px)")
    ax[1].set_ylabel("RMS strain error")
    ax[1].set_title("strain, same runs")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    w = [r["span_px"] for r in window_rows]
    ax[2].errorbar(w, [r["strain_rms"] for r in window_rows],
                   yerr=[r["strain_rms_sd"] for r in window_rows],
                   marker="o", color="tab:blue")
    ax[2].set_xlabel("strain window span (px)")
    ax[2].set_ylabel("RMS strain error", color="tab:blue")
    ax[2].grid(alpha=0.3)
    twin = ax[2].twinx()
    twin.errorbar(w, [r["peak_ratio"] for r in window_rows],
                  yerr=[r["peak_ratio_sd"] for r in window_rows],
                  marker="s", color="tab:red")
    twin.axhline(1.0, color="tab:red", ls=":", lw=1)
    twin.set_ylabel("fraction of true peak recovered", color="tab:red")
    ax[2].set_title("smoothing")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sweeps.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fieldmodel = StretchWithBulge()
    ref = speckle_image(seed=1)
    defm = warp(ref, fieldmodel)

    coords_all = np.arange(FIXED_HW + 5, 301 - FIXED_HW - 5, SPACING, dtype=float)
    print("true exx on the measurement grid: %.5f to %.5f"
          % tuple(np.percentile(fieldmodel.strain(
              *np.meshgrid(coords_all, coords_all, indexing="xy"))[0], [0, 100])))

    print("\nsubset size sweep (strain window fixed at %d points)" % FIXED_WINDOW)
    size_rows = sweep_subset_size(fieldmodel, ref, defm)

    print("\nstrain window sweep (subset fixed at %d px)" % (2 * FIXED_HW + 1))
    window_rows = sweep_strain_window(fieldmodel, ref, defm)

    figure_fields(fieldmodel, ref, defm)
    figure_sweeps(size_rows, window_rows)

    out = {
        "settings": {
            "image_px": 301, "noise_sd": NOISE, "replicates": REPLICATES,
            "grid_spacing_px": SPACING, "fixed_subset_px": 2 * FIXED_HW + 1,
            "fixed_strain_window": FIXED_WINDOW,
        },
        "subset_size_sweep": size_rows,
        "strain_window_sweep": window_rows,
    }
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results.json and two figures")


if __name__ == "__main__":
    main()
