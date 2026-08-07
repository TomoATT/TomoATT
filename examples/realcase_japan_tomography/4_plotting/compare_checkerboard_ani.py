#!/usr/bin/env python3
"""
compare_checkerboard_ani.py

Anisotropy checkerboard recovery comparison (xi/eta + derived magnitude
epsilon_true = sqrt(xi^2+eta^2) and fast-axis angle phi = 0.5 atan2(eta, xi))
between the truth checkerboard model and a TomoATT anisotropy inversion
recovery model.

  * per-depth correlation and gain for xi, eta, epsilon
  * per-depth maps: truth epsilon vs recovered epsilon
  * CSV summary

Usage:
  python compare_checkerboard_ani.py \
      --truth ../2_data_processing/checkerboard_models/checker_ani_coarse.h5 \
      --recovered ../OUTPUT_FILES_cb_inversion_ani/final_model.h5 \
      --depths 20 60 100 140 180 --section-lat 35.6 --out figs_cb_ani
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEP_MIN, DEP_MAX = -10.0, 200.0
LAT_MIN, LAT_MAX = 24.0, 46.0
LON_MIN, LON_MAX = 122.0, 148.0


def load_axes(arr_shape):
    nd, nl, no = arr_shape
    return (np.linspace(DEP_MIN, DEP_MAX, nd),
            np.linspace(LAT_MIN, LAT_MAX, nl),
            np.linspace(LON_MIN, LON_MAX, no))


def load_ds(path, name):
    with h5py.File(path) as f:
        return f[name][:]


def interp_dep(arr, dep_src, dep_dst):
    out = np.empty((len(dep_dst),) + arr.shape[1:], dtype=float)
    for k, d in enumerate(dep_dst):
        if d <= dep_src[0]:
            out[k] = arr[0]; continue
        if d >= dep_src[-1]:
            out[k] = arr[-1]; continue
        j = int(np.searchsorted(dep_src, d) - 1)
        w = (d - dep_src[j]) / (dep_src[j + 1] - dep_src[j])
        out[k] = (1 - w) * arr[j] + w * arr[j + 1]
    return out


def metrics(true, rec, mask):
    t, r = true[mask], rec[mask]
    m = np.isfinite(t) & np.isfinite(r)
    t, r = t[m], r[m]
    if len(t) < 10 or np.std(t) == 0 or np.std(r) == 0:
        return np.nan, np.nan
    corr = np.corrcoef(t, r)[0, 1]
    gain = float((t * r).sum() / (t * t).sum())
    return corr, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--recovered", required=True)
    ap.add_argument("--depths", type=float, nargs="*",
                    default=[20, 60, 100, 140, 180])
    ap.add_argument("--section-lat", type=float, default=None)
    ap.add_argument("--out", default="figs_cb_ani")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    xi_t = load_ds(args.truth, "xi"); eta_t = load_ds(args.truth, "eta")
    xi_r = load_ds(args.recovered, "xi"); eta_r = load_ds(args.recovered, "eta")

    dep_t, lat_t, lon_t = load_axes(xi_t.shape)
    dep_r, _, _ = load_axes(xi_r.shape)
    xi_r = interp_dep(xi_r, dep_r, dep_t)
    eta_r = interp_dep(eta_r, dep_r, dep_t)

    eps_t = np.sqrt(xi_t**2 + eta_t**2)
    eps_r = np.sqrt(xi_r**2 + eta_r**2)

    import csv
    rows = []
    with open(out_dir / "cb_ani_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth_km", "corr_xi", "gain_xi", "corr_eta", "gain_eta",
                    "corr_eps", "gain_eps"])
        for d in args.depths:
            k = int(np.argmin(np.abs(dep_t - d)))
            mask = np.abs(xi_t[k]) > 0
            cx, gx = metrics(xi_t[k], xi_r[k], mask)
            ce, ge = metrics(eta_t[k], eta_r[k], mask)
            # epsilon recovery measured on non-zero truth cells
            ceps, geps = metrics(eps_t[k], eps_r[k],
                                 np.abs(eps_t[k]) > 0.15 * np.nanmax(eps_t[k]))
            rows.append((dep_t[k], cx, gx, ce, ge, ceps, geps))
            w.writerow([f"{dep_t[k]:.0f}", f"{cx:.4f}", f"{gx:.4f}",
                        f"{ce:.4f}", f"{ge:.4f}", f"{ceps:.4f}", f"{geps:.4f}"])

            # panels: truth xi, recovered xi, truth eta, recovered eta
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.4),
                                     constrained_layout=True)
            vmax = np.nanmax(np.abs(xi_t[k]))
            for ax, field, ttl in (
                    (axes[0, 0], xi_t[k], "truth xi"),
                    (axes[0, 1], xi_r[k], "recovered xi"),
                    (axes[1, 0], eta_t[k], "truth eta"),
                    (axes[1, 1], eta_r[k], "recovered eta")):
                pc = ax.pcolormesh(lon_t, lat_t, field, cmap="RdBu_r",
                                   vmin=-vmax, vmax=vmax, shading="auto")
                ax.set_title(f"{ttl} @ {dep_t[k]:.0f} km", fontsize=9)
                plt.colorbar(pc, ax=ax, shrink=0.8)
            fig.savefig(out_dir / f"cb_ani_fields_{int(dep_t[k]):03d}km.png",
                        dpi=160)
            plt.close(fig)

    for r in rows:
        print(f"dep {r[0]:5.0f}  corr_xi {r[1]:+.3f} gain {r[2]:+.3f} | "
              f"corr_eta {r[3]:+.3f} gain {r[4]:+.3f} | corr_eps {r[5]:+.3f}"
              f" gain {r[6]:+.3f}")

    if args.section_lat is not None:
        j = int(np.argmin(np.abs(lat_t - args.section_lat)))
        fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.6), sharex=True,
                                 constrained_layout=True)
        vmax = np.nanmax(np.abs(eps_t[:, j, :]))
        for ax, field, ttl in ((axes[0], eps_t[:, j, :] * 100,
                                "truth epsilon (% of eps norm)"),
                               (axes[1], eps_r[:, j, :] * 100, "recovered")):
            pc = ax.pcolormesh(lon_t, dep_t, field, cmap="viridis",
                               vmin=0, vmax=vmax * 100, shading="auto")
            ax.set_ylim(DEP_MAX, DEP_MIN)
            ax.set_title(f"{ttl} at {lat_t[j]:.2f} N", fontsize=9)
            plt.colorbar(pc, ax=ax, shrink=0.9)
        fig.savefig(out_dir / f"cb_ani_section_{lat_t[j]:.1f}N.png", dpi=170)
        plt.close(fig)
    print(f"wrote results under {out_dir}")


if __name__ == "__main__":
    main()
