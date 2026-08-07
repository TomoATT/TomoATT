#!/usr/bin/env python3
"""
plot_joint_v2_specials.py

Japan v2 (joint P + azimuthal anisotropy + teleseismic source augmentation)
special figure set:
  1. anisotropy amplitude maps with fast-axis bars at selected depths
     (slab contours + coast + volcanoes overlays)
  2. anisotropy W-E sections at 35.6 / 39.7 N
  3. convergence curves: objective + residual mean/std (v1 vs v2)
  4. ray-coverage map: local events, teleseismic sources, stations

Reads from ../OUTPUT_FILES_joint_tele_ani/final_model.h5,
            ../OUTPUT_FILES_joint_tele_ani/objective_history.txt,
            ../2_data_processing/src_rec_file_japan_tele.dat,
            ../../realcase_japan_tomography/OUTPUT_FILES_run_gpu
"""

import csv
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

from overlay_common import REGION, draw_overlays, load_slab_grid, SLAB_PLOTS

CASE = Path(__file__).parent.resolve().parent
FINAL = CASE / "OUTPUT_FILES_joint_tele_ani" / "final_model.h5"
HIST = CASE / "OUTPUT_FILES_joint_tele_ani" / "objective_history.txt"
SRCREC = CASE / "2_data_processing" / "src_rec_file_japan_tele.dat"
OUT = Path("figs_joint_v2")
OUT.mkdir(exist_ok=True)

DEP_MIN, DEP_MAX = -10.0, 200.0
LAT_MIN, LAT_MAX = 24.0, 46.0
LON_MIN, LON_MAX = 122.0, 148.0

f = h5py.File(FINAL)
VEL, XI, ETA = f["vel"][:], f["xi"][:], f["eta"][:]
f.close()
nd, nl, no = VEL.shape
dep = np.linspace(DEP_MIN, DEP_MAX, nd)
lat = np.linspace(LAT_MIN, LAT_MAX, nl)
lon = np.linspace(LON_MIN, LON_MAX, no)
amp = np.sqrt(XI**2 + ETA**2)
phi = 0.5 * np.degrees(np.arctan2(ETA, XI))   # fast-axis azimuth (deg)


def aniso_map(k, vmax):
    d = dep[k]
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(7.6, 8.2))
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    pc = ax.pcolormesh(lon, lat, amp[k] * 100, cmap="inferno", vmin=0,
                       vmax=vmax, shading="auto", zorder=2, transform=proj)
    # fast-axis bars (subsample)
    step_i = 3
    xs = lon[::step_i]; ys = lat[::step_i]
    X, Y = np.meshgrid(xs, ys)
    A = amp[k][::step_i, ::step_i]
    P = np.radians(phi[k][::step_i, ::step_i])
    u = np.cos(np.radians(90) - P)
    v = np.sin(np.radians(90) - P)
    L = np.clip(A * 100 / vmax, 0, 1)
    ax.quiver(X, Y, u * L, v * L, angles="xy", scale_units="xy", scale=260,
              color="cyan", width=0.0022, headlength=0, headwidth=0,
              zorder=3, transform=proj)
    draw_overlays(ax, dep_km=d, slab_every=100)
    gl = ax.gridlines(draw_labels=True, lw=0.2, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    cb = plt.colorbar(pc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("azimuthal anisotropy amplitude (%)")
    ax.set_title(f"dlnVp anisotropy at {d:.0f} km — bars = fast axis")
    fig.tight_layout()
    fn = OUT / f"aniso_map_{int(d):03d}km.png"
    fig.savefig(fn, dpi=180); plt.close(fig)
    print("wrote", fn)


def aniso_section(j, vmax):
    la = lat[j]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    pc = ax.pcolormesh(lon, dep, amp[:, j, :] * 100, cmap="inferno",
                       vmin=0, vmax=vmax, shading="auto")
    ax.set_ylim(DEP_MAX, DEP_MIN)
    ax.set_xlim(LON_MIN, LON_MAX)
    # slab top profiles
    for tag, fname, col in SLAB_PLOTS:
        try:
            slon, slat, sdep = load_slab_grid(fname)
        except Exception:
            continue
        band = np.abs(slat - la) <= 0.25
        if not band.any():
            continue
        zb = np.nanmedian(sdep[band, :], axis=0)
        m = np.isfinite(zb) & (slon >= LON_MIN) & (slon <= LON_MAX)
        if m.sum() > 5:
            ax.plot(slon[m], zb[m], color=col, lw=1.6, label=tag)
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Depth (km)")
    ax.set_title(f"Anisotropy amplitude, W-E section {la:.2f} N")
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    cb = plt.colorbar(pc, ax=ax, pad=0.01, shrink=0.9)
    cb.set_label("anisotropy amplitude (%)")
    fig.tight_layout()
    fn = OUT / f"aniso_sec_{la:.1f}N.png"
    fig.savefig(fn, dpi=180); plt.close(fig)
    print("wrote", fn)


for k in [22, 29, 34, 38, 42]:              # ≈ 100, 130, 150, 170, 190 km
    aniso_map(k, 6.0)
for j in [56, 76]:                          # ≈ 35.6 / 39.7 N
    aniso_section(j, 6.0)

# ---------------------------------------------------------------------------
# convergence curve
# ---------------------------------------------------------------------------
hist = []
with open(HIST) as fs:
    rd = csv.reader(fs)
    for row in rd:
        if not row or row[0].strip().startswith("#"):
            continue
        parts = [p.strip() for p in row]
        def num(s):
            try:
                return float(s)
            except Exception:
                return np.nan
        obj = num(parts[2]); res_m = num(parts[7].split("/")[0])
        res_s = num(parts[7].split("/")[1])
        hist.append((obj, res_m, res_s))
hist = np.array(hist, dtype=float)
xstep = np.arange(len(hist))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(xstep, hist[:, 0] / 355827 * 100, "o-", color="navy", ms=3)
axes[0].set_yscale("log")
axes[0].set_xlabel("cumulative inversion iterations")
axes[0].set_ylabel("objective (% of initial, log)")
axes[0].set_title("final: 41.9 % of initial (149,219 vs 355,827)")
axes[0].grid(alpha=0.3)
axes[1].plot(xstep, hist[:, 1], "o-", color="darkred", ms=3, label="res mean (s)")
axes[1].plot(xstep, hist[:, 2], "s-", color="darkorange", ms=3,
             label="res std (s)")
axes[1].axhline(0, color="gray", lw=0.5)
axes[1].set_xlabel("cumulative inversion iterations"); axes[1].legend()
axes[1].set_title("travel-time residuals")
axes[1].grid(alpha=0.3)
fig.suptitle("Japan v2 joint inversion convergence (v1: −83.1 %)")
fig.tight_layout()
fig.savefig(OUT / "convergence_v2.png", dpi=170); plt.close(fig)
print("wrote", OUT / "convergence_v2.png")

# ---------------------------------------------------------------------------
# coverage map
# ---------------------------------------------------------------------------
ev_l, ev_lo, ev_d = [], [], []
st_l, st_lo = [], []
tel_l, tel_lo = [], []
with open(SRCREC) as fs:
    lines = fs.readlines()
hdr_idx = [k for k, ln in enumerate(lines)
           if len(ln.split()) == 14
           and (ln.split()[12].startswith("ev_") or ln.split()[12].startswith("tel_"))]
for n, h in enumerate(hdr_idx):
    t = lines[h].split()
    if t[12].startswith("tel_"):
        tel_l.append(float(t[7])); tel_lo.append(float(t[8]))
    else:
        ev_l.append(float(t[7])); ev_lo.append(float(t[8])); ev_d.append(float(t[9]))
    nxt = hdr_idx[n + 1] if n + 1 < len(hdr_idx) else len(lines)
    for ln in lines[h + 1:nxt]:
        if len(ln.split()) == 9:
            rt = ln.split()
            st_l.append(float(rt[3])); st_lo.append(float(rt[4]))

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(8.6, 8.8))
ax.set_extent([118, 175, 18, 60], crs=proj)
ax.scatter(st_lo, st_l, s=5, c="navy", alpha=0.4, transform=proj,
           label=f"stations ({len(st_l):,})")
sc = ax.scatter(tel_lo, tel_l, s=14, c="crimson", alpha=0.6, marker="^",
                transform=proj, label=f"teleseismic srcs ({len(tel_l)})")
sc2 = ax.scatter(ev_lo, ev_l, s=8, c=ev_d, cmap="viridis", alpha=0.55,
                 transform=proj, label=f"local events ({len(ev_l)})")
plt.colorbar(sc2, ax=ax, pad=0.02, shrink=0.7, label="local source depth (km)")
ax.coastlines(resolution="10m", lw=0.6, zorder=3)
for lonv, latv in ():
    pass
draw_overlays(ax, dep_km=100, slab_every=100)
gl = ax.gridlines(draw_labels=True, lw=0.2, alpha=0.5)
gl.top_labels = gl.right_labels = False
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Japan v2 data: 5 575 local + 473 teleseismic sources, 57 k+ arrivals")
fig.tight_layout()
fig.savefig(OUT / "coverage_v2.png", dpi=180); plt.close(fig)
print("wrote", OUT / "coverage_v2.png")
