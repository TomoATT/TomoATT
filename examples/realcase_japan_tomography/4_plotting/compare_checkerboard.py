#!/usr/bin/env python3
"""
compare_checkerboard.py

Compare a checkerboard recovery result with the truth model:

  * per-depth correlation coefficient and recovery amplitude ratio
    (least-squares gain  g = <true, rec> / <true, true>)
  * three-panel maps (true | recovered | pointwise difference)
  * W-E section at a chosen latitude
  * CSV summary table

Truth and initial models live on the (model) grid; the recovered model comes
from a TomoATT run out_data_grid.h5 on the solver grid (n_rtp) and is linearly
interpolated in depth onto the model grid before comparison.

Usage:
  python compare_checkerboard.py \
      --truth ../2_data_processing/checkerboard_models/checker_vel_coarse.h5 \
      --base  ../../eg_japan_initial_model/japan_model.h5 \
      --recovered ../OUTPUT_FILES_cb_inversion_vel/out_data_grid.h5 \
      --depths 20 60 100 140 180 --section-lat 35.6 --out figs_cb_compare
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


def load_vel(path):
    with h5py.File(path) as f:
        if "model" in f:
            grp = f["model"]
            cands = [k for k in grp.keys() if k.startswith("vel")]
            prefer = [k for k in cands if "final" in k]
            key = sorted(prefer or cands)[-1]
            print(f"  recovered dataset: model/{key} from {path}")
            return grp[key][:]
        return f["vel"][:]


def interp_dep(arr, dep_src, dep_dst):
    """Linear interpolation of arr (nd_src, nl, no) onto dep_dst."""
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


def anomaly(vel, base):
    return (vel - base) / np.where(base == 0, np.nan, base)


def metrics(true, rec, mask):
    t, r = true[mask], rec[mask]
    m = np.isfinite(t) & np.isfinite(r)
    t, r = t[m], r[m]
    if len(t) < 10 or np.std(t) == 0 or np.std(r) == 0:
        return np.nan, np.nan, len(t)
    corr = np.corrcoef(t, r)[0, 1]
    gain = float((t * r).sum() / (t * t).sum())
    return corr, gain, len(t)


def panel_maps(true, rec, diff, dep_act, lat, lon, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    v = np.nanmax(np.abs(true)) * 100
    v = 3.0 if not np.isfinite(v) else min(max(1.0, v), 5.0)
    for ax, field, ttl in zip(
            axes, (true * 100, rec * 100, diff * 100),
            ("truth dlnVp (%)", "recovered dlnVp (%)",
             "recovered - truth (%)")):
        pc = ax.pcolormesh(lon, lat, field, cmap="RdBu_r", vmin=-v, vmax=v,
                           shading="auto")
        ax.set_title(f"{ttl} @ {dep_act:.0f} km", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(pc, ax=ax, shrink=0.8)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--recovered", required=True)
    ap.add_argument("--depths", type=float, nargs="*",
                    default=[20, 60, 100, 140, 180])
    ap.add_argument("--section-lat", type=float, default=None)
    ap.add_argument("--out", default="figs_cb_compare")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    vel_true = load_vel(args.truth)
    vel_base = load_vel(args.base)
    vel_rec = load_vel(args.recovered)

    dep_t, lat_t, lon_t = load_axes(vel_true.shape)
    dep_r, _, _ = load_axes(vel_rec.shape)
    rec_on_true = interp_dep(vel_rec, dep_r, dep_t)

    abn_true = anomaly(vel_true, vel_base)
    abn_rec = anomaly(rec_on_true, vel_base)

    # cells that are non-zero anywhere in depth in the truth model (checker
    # tapers + sign structure) — metric domain
    mask3d = np.abs(abn_true) > 0.2 * np.nanmax(np.abs(abn_true))

    rows = []
    for d in args.depths:
        k = int(np.argmin(np.abs(dep_t - d)))
        dep_act = dep_t[k]
        corr, gain, n = metrics(abn_true[k], abn_rec[k],
                                np.abs(abn_true[k]) > 0)
        rows.append((dep_act, corr, gain, n))
        panel_maps(abn_true[k], abn_rec[k],
                   abn_rec[k] - abn_true[k], dep_act, lat_t, lon_t,
                   out_dir / f"cb_compare_{int(dep_act):03d}km.png")
        print(f"  {dep_act:6.1f} km   corr={corr:.3f}  gain={gain:.3f}  "
              f"cells={n}")

    if args.section_lat is not None:
        j = int(np.argmin(np.abs(lat_t - args.section_lat)))
        fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.0), sharex=True,
                                 constrained_layout=True)
        v = np.nanmax(np.abs(abn_true[:, j, :])) * 100
        v = 3.0 if not np.isfinite(v) else min(max(1.0, v), 5.0)
        for ax, field, ttl in zip(
                axes, (abn_true[:, j, :] * 100, abn_rec[:, j, :] * 100),
                ("truth", "recovered")):
            pc = ax.pcolormesh(lon_t, dep_t, field, cmap="RdBu_r",
                               vmin=-v, vmax=v, shading="auto")
            ax.set_ylim(DEP_MAX, DEP_MIN)
            ax.set_title(f"{ttl} dlnVp (%) at {lat_t[j]:.2f} N", fontsize=10)
            plt.colorbar(pc, ax=ax, shrink=0.9)
        fig.savefig(out_dir / f"cb_section_{lat_t[j]:.1f}N.png", dpi=170)
        plt.close(fig)

    import csv
    with open(out_dir / "cb_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth_km", "corr", "gain", "n_cells"])
        for r in rows:
            w.writerow([f"{r[0]:.1f}", f"{r[1]:.4f}", f"{r[2]:.4f}", r[3]])
    print(f"wrote {out_dir}/cb_metrics.csv")


if __name__ == "__main__":
    main()
