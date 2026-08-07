#!/usr/bin/env python3
"""
analytic_resolution.py

Analytic maximum-resolution survey for the Japan v2 traveltime dataset,
without running an inversion:

  1. Station Nyquist: median nearest-neighbour station spacing -> 2x spacing
     scales (in deg and km) for the station set actually used.
  2. Local / global Fresnel-zone bound: sqrt(lambda * L) with a dominant-P
     wavelength proxy, at representative mantle path lengths.
  3. Ray-illumination measure per column cell (2 deg x 2 deg x depth band):
     hit count, number of occupied source-azimuth bins and dip classes.
     A cell is "fully resolved" (highest confidence) if hits >= 100,
     azimuth bins >= 6 of 12, and >= 2 dip classes are present.
  4. Figures for each depth band: hit count, azimuth diversity, and the
     resolved map. W-E coverage profile at 150-200 km (teleseismic benefit).

Straight great-circle-ray approximation (spherical Earth, depth-linear
segment stations->source). This is optimistic on hits but unbiased on
cell-by-cell RELATIVE illumination, which is the output we trust.

Usage: python analytic_resolution.py (from 4_plotting dir)
"""

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

from overlay_common import draw_overlays

CASE = Path(__file__).parent.resolve().parent
SRCREC = CASE / "2_data_processing" / "src_rec_file_japan_tele.dat"
OUT = Path("figs_resolution")
OUT.mkdir(exist_ok=True)

R_E = 6371.0
DEG = math.pi / 180.0

LAT_GRID = np.arange(24.0, 46.0 + 1e-6, 2.0)
LON_GRID = np.arange(122.0, 148.0 + 1e-6, 2.0)
DEP_BANDS = [(0, 15), (15, 30), (30, 50), (50, 70), (70, 90),
             (110, 130), (130, 160), (160, 200), (90, 110)]

AZ_BINS = 12          # 30 deg each
DIP_CLASSES = 3       # shallow (<20 deg), medium (20-45), steep (>=45)
HIT_MIN = 100
AZMIN = 6
DIPMIN = 2


def cell_centers(grid):
    return (grid[:-1] + grid[1:]) * 0.5


def gc_azimuth(lat1, lon1, lat2, lon2):
    """Forward azimuth from point 1 to 2 (great circle, degrees)."""
    la1, lo1, la2, lo2 = map(lambda x: x * DEG, (lat1, lon1, lat2, lon2))
    dlon = lo2 - lo1
    y = np.sin(dlon) * np.cos(la2)
    x = (np.cos(la1) * np.sin(la2)
         - np.sin(la1) * np.cos(la2) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def gc_delta(lat1, lon1, lat2, lon2):
    """Angular distance, degrees."""
    la1, lo1, la2, lo2 = map(lambda x: x * DEG, (lat1, lon1, lat2, lon2))
    arg = (np.sin(la1) * np.sin(la2)
           + np.cos(la1) * np.cos(la2) * np.cos(lo2 - lo1))
    return np.degrees(np.arccos(np.clip(arg, -1.0, 1.0)))


def load_catalog():
    lines = SRCREC.read_text().splitlines()
    hdr = [k for k, ln in enumerate(lines)
           if len(ln.split()) == 14
           and (ln.split()[12].startswith("ev_") or ln.split()[12].startswith("tel_"))]
    events = []
    for n, h in enumerate(hdr):
        t = lines[h].split()
        nxt = hdr[n + 1] if n + 1 < len(hdr) else len(lines)
        recs = []
        for ln in lines[h + 1:nxt]:
            rt = ln.split()
            if len(rt) == 9:                 # absolute line
                recs.append((float(rt[3]), float(rt[4])))
            elif len(rt) >= 11:              # pair line: use rec1 coords once
                recs.append((float(rt[3]), float(rt[4])))
        events.append(dict(name=t[12], lat=float(t[7]), lon=float(t[8]),
                           dep=float(t[9]), recs=recs))
    return events


def main():
    # ------------------------------------------------------------------
    # 1 + 2: Nyquist + Fresnel
    # ------------------------------------------------------------------
    events = load_catalog()
    sta = {}
    for e in events:
        for la, lo in e["recs"]:
            sta[f"{la:.3f}_{lo:.3f}"] = (la, lo)
    pts = np.array(list(sta.values()))
    print(f"unique stations used: {len(pts)}")

    d = gc_delta(pts[:, None, 0], pts[:, None, 1],
                 pts[None, :, 0], pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=0) * 111.19
    med = np.median(nn)
    p25, p75 = np.percentile(nn, [25, 75])
    print(f"station NN spacing: p25 {p25:.1f} km, median {med:.1f} km, "
          f"p75 {p75:.1f} km")
    nyq_km = 2.0 * med
    nyq_deg = nyq_km / 111.19 / np.cos(35 * DEG)
    print(f"station Nyquist (2x median NN): {nyq_km:.0f} km "
          f"(= {nyq_km/111.19:.2f} deg N-S, {nyq_deg:.2f} deg E-W at 35N)")

    for lam_km, L_km in [(15, 200), (15, 400), (15, 700), (25, 400), (36, 700)]:
        f = math.sqrt(lam_km * L_km)
        print(f"Fresnel width sqrt(lambda*L): lambda={lam_km} km, "
              f"L={L_km} km -> {f:.0f} km ({f/111.19:.2f} deg)")

    # ------------------------------------------------------------------
    # 3: ray illumination per column
    # ------------------------------------------------------------------
    la_c = cell_centers(LAT_GRID)
    lo_c = cell_centers(LON_GRID)
    nb = len(DEP_BANDS)
    hits = np.zeros((nb, len(la_c), len(lo_c)), dtype=np.int32)
    azoc = np.zeros((nb, len(la_c), len(lo_c), AZ_BINS), dtype=bool)
    dipoc = np.zeros((nb, len(la_c), len(lo_c), DIP_CLASSES), dtype=bool)

    iaz = None
    for ev_i, e in enumerate(events):
        for la1, lo1 in e["recs"]:
            delta_km = gc_delta(la1, lo1, e["lat"], e["lon"]) * 111.19
            nstep = max(10, min(200, int(delta_km / 15)))
            tt = np.linspace(0.0, 1.0, nstep)
            la_path = la1 + (e["lat"] - la1) * tt
            lo_path = lo1 + (e["lon"] - lo1) * tt
            dp_path = e["dep"] * tt
            az_b = int(gc_azimuth(la1, lo1, e["lat"], e["lon"]) // 30) % AZ_BINS
            # dip from endpoints: angle below horizontal of the
            # (depth,horizontal) first half step
            mid = max(1, nstep // 2)
            hd = gc_delta(la_path[0], lo_path[0], la_path[mid],
                          lo_path[mid]) * 111.19
            vd = abs(dp_path[mid] - dp_path[0])
            dip_deg = math.degrees(math.atan2(vd, max(hd, 1e-3)))
            dcls = 0 if dip_deg < 20 else (1 if dip_deg < 45 else 2)

            ilat = np.searchsorted(LAT_GRID, la_path) - 1
            ilon = np.searchsorted(LON_GRID, lo_path) - 1
            ok = (ilat >= 0) & (ilat < len(la_c)) & (ilon >= 0) & (ilon < len(lo_c))
            for b, (d0, d1) in enumerate(DEP_BANDS):
                m = ok & (dp_path >= d0) & (dp_path < d1)
                if not m.any():
                    continue
                i_i = ilat[m]; i_j = ilon[m]
                # accumulate through 2D for speed: unique per-path cells
                uniq = np.unique(zip_idx := i_i * len(lo_c) + i_j)
                zi = uniq // len(lo_c)
                zj = uniq % len(lo_c)
                hits[b, zi, zj] += 1
                azoc[b, zi, zj, az_b] = True
                dipoc[b, zi, zj, dcls] = True

    az_div = azoc.sum(axis=3)
    dip_div = dipoc.sum(axis=3)
    resolved = (hits >= HIT_MIN) & (az_div >= AZMIN) & (dip_div >= DIPMIN)

    np.savez(OUT / "resolution_arrays.npz", hits=hits, az_div=az_div,
             dip_div=dip_div, resolved=resolved, lat=la_c, lon=lo_c,
             bands=np.array(DEP_BANDS))

    # summary numbers per band
    print("\nband depth      cells  hits>=100  az>=6  dip>=2  resolved")
    for b, (d0, d1) in enumerate(DEP_BANDS):
        tot = hits[b].size
        print(f"{d0:3d}-{d1:3d}  {tot:5d}  {int((hits[b]>=HIT_MIN).sum()):8d}  "
              f"{int((az_div[b]>=AZMIN).sum()):6d}  {int((dip_div[b]>=DIPMIN).sum()):6d}  "
              f"{int(resolved[b].sum()):5d} ({100*resolved[b].sum()/tot:.0f} %)")

    # ------------------------------------------------------------------
    # figures per band
    # ------------------------------------------------------------------
    proj = ccrs.PlateCarree()
    for b, (d0, d1) in enumerate(DEP_BANDS):
        for field, ttl, cmap, vmax, name in (
                (np.log10(np.maximum(hits[b], 1)), "hit count (log10)",
                 "viridis", math.log10(hits[b].max() + 1), "hits"),
                (az_div[b], "source-azimuth diversity (of 12)",
                 "plasma", 12, "az"),
                (dip_div[b], "dip-class diversity (of 3)",
                 "cividis", 3, "dip"),
                (resolved[b].astype(float), "fully resolved (hits>=100, az>=6, dip>=2)",
                 "RdYlGn", 1, "resolved")):
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj),
                                   figsize=(7.3, 7.8))
            ax.set_extent([122, 148, 24, 46], crs=proj)
            pc = ax.pcolormesh(LON_GRID, LAT_GRID, field, cmap=cmap,
                               vmin=0, vmax=vmax, shading="auto",
                               transform=proj, zorder=2)
            draw_overlays(ax, dep_km=None, slab_every=100)
            ax.set_title(f"{ttl}  [{d0}-{d1} km]")
            plt.colorbar(pc, ax=ax, shrink=0.75)
            gl = ax.gridlines(draw_labels=True, lw=0.2, alpha=0.5)
            gl.top_labels = gl.right_labels = False
            fig.tight_layout()
            fig.savefig(OUT / f"{name}_{d0:03d}_{d1:03d}km.png", dpi=160)
            plt.close(fig)

    # W-E coverage profile per longitude column
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for b in (0, 2, 5, 7):
        d0, d1 = DEP_BANDS[b]
        prof = hits[b].sum(axis=0)
        ax.plot(lo_c, prof, lw=1.4, label=f"{d0}-{d1} km")
    ax.set_yscale("log")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("hit counts per 2-deg column (log)")
    ax.set_title("E-W illumination profile (sum over latitude)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "we_profile.png", dpi=170)
    plt.close(fig)

    print("\nfigures under", OUT)


# mark bands of the profile explicitly
regression_index = None

if __name__ == "__main__":
    main()
