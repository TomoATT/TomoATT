#!/usr/bin/env python3
"""
plot_slices_overlays.py

Japan tomography map/section figures with geological overlays
(coastline, PB2002 plate boundaries, GEM active faults, GVP Holocene
volcanoes, USGS Slab1.0 slab-top contours).

Model input (any of):
  * plain model .h5 with datasets vel[/xi/eta/zeta]  (e.g. japan_model.h5,
    checkerboard models) — coordinates inferred from the domain extents and
    the dataset shape.
  * TomoATT run out_data_grid.h5 with group 'model' — the final inversion
    model is selected automatically (vel_final / last vel_inv_XXXX key).

Examples
  # slices of the checkerboard truth model (validates overlays)
  python plot_slices_overlays.py --model <cb>/checker_vel_coarse.h5 \
      --mode slices --depths 20 60 100 140 180 --type dlnv \
      --vel-mean ../eg_japan_initial_model/japan_model.h5

  # sections through a TomoATT output grid file
  python plot_slices_overlays.py --model OUTPUT/out_data_grid.h5 \
      --mode section --lats 35.6 39.7
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

from overlay_common import (REGION, draw_overlays, load_slab_grid,
                            SLAB_PLOTS, load_volcanoes, load_plate_boundaries)

# production domain
DEP_MIN, DEP_MAX = -10.0, 200.0
LAT_MIN, LAT_MAX = 24.0, 46.0
LON_MIN, LON_MAX = 122.0, 148.0


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_model(path):
    """Return (dep, lat, lon, vel) 1-D axes + 3-D array."""
    with h5py.File(path) as f:
        if "model" in f:
            grp = f["model"]
            cands = [k for k in grp.keys() if k.startswith("vel")]
            if not cands:
                raise KeyError(f"no vel* dataset under 'model' in {path}")
            prefer = [k for k in cands if "final" in k]
            key = sorted(prefer or cands)[-1]
            vel = grp[key][:]
            print(f"  using dataset model/{key}")
        else:
            vel = f["vel"][:]
    nd, nl, no = vel.shape
    dep = np.linspace(DEP_MIN, DEP_MAX, nd)
    lat = np.linspace(LAT_MIN, LAT_MAX, nl)
    lon = np.linspace(LON_MIN, LON_MAX, no)
    return dep, lat, lon, np.asarray(vel, dtype=float)


def slice_at(arr3, axis_vals, value, axis):
    i = int(np.argmin(np.abs(axis_vals - value)))
    return (arr3[i, :, :] if axis == 0 else
            arr3[:, i, :] if axis == 1 else arr3[:, :, i]), axis_vals[i]


def dlnv_of(vel, reference=None):
    """Velocity perturbation (%%) relative to depth-mean of the model itself
    (or of `reference` model if given)."""
    nd = vel.shape[0]
    base = reference if reference is not None else vel
    lay = np.nanmean(base, axis=(1, 2))          # (nd,)
    lay[lay == 0] = np.nan
    out = np.empty_like(vel)
    for k in range(nd):
        out[k] = 100.0 * (vel[k] - lay[k]) / lay[k]
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_slice(field, lat, lon, dep_km, title, cmap, vlim, out_png,
              label_slab=True, cb_label=""):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj),
                           figsize=(7.6, 8.4))
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    pc = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                       shading="auto", zorder=2, transform=proj)
    draw_overlays(ax, dep_km=dep_km, label_slab=label_slab)
    gl = ax.gridlines(draw_labels=True, lw=0.2, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    cb = plt.colorbar(pc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(cb_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"  wrote {out_png}")


def fig_section(field_t, lon, lat_actual, lon_axis, dep_axis, kind,
                out_png, cmap, vlim, cb_label):
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    pc = ax.pcolormesh(lon_axis, dep_axis, field_t, cmap=cmap,
                       vmin=vlim[0], vmax=vlim[1], shading="auto")
    ax.invert_yaxis()

    # slab top profiles along this latitude band
    for tag, fname, col in SLAB_PLOTS:
        try:
            slon, slat, sdep = load_slab_grid(fname)
        except Exception:
            continue
        band = np.abs(slat - lat_actual) <= 0.25
        if not band.any():
            continue
        zband = np.nanmedian(sdep[band, :], axis=0)
        m = np.isfinite(zband) & (slon >= LON_MIN) & (slon <= LON_MAX)
        if m.sum() > 5:
            ax.plot(slon[m], zband[m], color=col, lw=1.6,
                    label=tag.replace(" (kur)", ""))
    # volcanoes within +-0.5 deg of the section, projected on the surface line
    volc = load_volcanoes()
    vsel = volc[np.abs(volc.latitude - lat_actual) <= 0.5]
    ax.scatter(vsel.longitude, np.zeros(len(vsel)) - 3, marker="^", s=40,
               facecolor="red", edgecolor="k", lw=0.4, zorder=5,
               label="volcano projection")

    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(DEP_MAX, DEP_MIN)
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Depth (km)")
    ax.set_title(f"W-E section at {lat_actual:.2f} N ({kind})")
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    cb = plt.colorbar(pc, ax=ax, pad=0.01, shrink=0.9)
    cb.set_label(cb_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vel-mean", default=None,
                    help="optional background model h5 for dlnv computation")
    ap.add_argument("--mode", choices=["slices", "section"], required=True)
    ap.add_argument("--depths", type=float, nargs="*",
                    default=[7, 15, 25, 30, 50, 80, 100, 130, 160, 180])
    ap.add_argument("--lats", type=float, nargs="*", default=[35.6, 39.7])
    ap.add_argument("--type", choices=["vel", "dlnv", "both"],
                    default="both")
    ap.add_argument("--out", default="figs_overlays")
    ap.add_argument("--tag", default="model")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model  : {args.model}")
    dep, lat, lon, vel = load_model(args.model)
    if args.vel_mean:
        _, _, _, ref = load_model(args.vel_mean)
        print(f"ref    : {args.vel_mean}")
    else:
        ref = None
    dv = dlnv_of(vel, ref)

    if args.mode == "slices":
        for d in args.depths:
            svel, d_act = slice_at(vel, dep, d, 0)
            sdv, _ = slice_at(dv, dep, d, 0)
            if args.type in ("vel", "both"):
                fig_slice(svel, lat, lon, d_act,
                          f"P velocity at {d_act:.0f} km ({args.tag})",
                          "jet", (3.5, 9.0),
                          out_dir / f"map_vel_{args.tag}_{int(d_act):03d}km.png",
                          cb_label="Vp (km/s)")
            if args.type in ("dlnv", "both"):
                v = np.nanmax(np.abs(sdv[svel > 0]))
                vlim = 6.0 if not np.isfinite(v) else min(max(2.0, v), 8.0)
                fig_slice(sdv, lat, lon, d_act,
                          f"dlnVp at {d_act:.0f} km ({args.tag})",
                          "RdBu_r", (-vlim, vlim),
                          out_dir / f"map_abn_{args.tag}_{int(d_act):03d}km.png",
                          cb_label="dlnVp (%)")
    else:
        for la in args.lats:
            iv = int(np.argmin(np.abs(lat - la)))
            la_act = lat[iv]
            sec_vel = vel[:, iv, :]        # (dep, lon) rows=Y, cols=X
            sec_dv = dv[:, iv, :]
            if args.type in ("vel", "both"):
                fig_section(sec_vel, lon, la_act, lon, dep, "Vp",
                            out_dir / f"sec_vel_{args.tag}_{la_act:.1f}N.png",
                            "jet", (3.5, 9.0), "Vp (km/s)")
            if args.type in ("dlnv", "both"):
                v = np.nanmax(np.abs(sec_dv[sec_vel > 0]))
                vlim = 6.0 if not np.isfinite(v) else min(max(2.0, v), 8.0)
                fig_section(sec_dv, lon, la_act, lon, dep, "dlnVp",
                            out_dir / f"sec_abn_{args.tag}_{la_act:.1f}N.png",
                            "RdBu_r", (-vlim, vlim), "dlnVp (%)")


if __name__ == "__main__":
    main()
