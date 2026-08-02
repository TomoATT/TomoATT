#!/usr/bin/env python3
"""
overlay_common.py

Shared overlay loading/drawing utilities for Japan tomography figures:
plate boundaries (PB2002), GEM active faults, Smithsonian GVP Holocene
volcanoes and USGS Slab1.0 slab-top grids for the four Japan subduction
zones (kur, izu, phi, ryu).

All datasets are fetched by fetch_overlay_data.py into ./overlay_data/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.resolve() / "overlay_data"

# study region
REGION = dict(lon_min=122.0, lon_max=148.0, lat_min=24.0, lat_max=46.0)

SLAB_PLOTS = [   # (display name, grd file, line color)
    # Note: the Philippine Sea slab subducting below SW Japan is part of the
    # Ryukyu (ryu) zone in Slab1.0; the standalone phi grid only covers the
    # Luzon-Taiwan segment and is not used.
    ("Pacific slab (kur)",      "kur_slab1.0_clip.grd", "magenta"),
    ("Izu-Bonin slab",          "izu_slab1.0_clip.grd", "darkviolet"),
    ("Philippine Sea slab (ryu)","ryu_slab1.0_clip.grd", "navy"),
]


# ---------------------------------------------------------------------------
# GeoJSON polylines
# ---------------------------------------------------------------------------
def _geojson_lines(path):
    with open(path) as f:
        gj = json.load(f)
    segs = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        t = geom.get("type")
        coords = geom.get("coordinates", [])
        if t == "LineString":
            segs.append(coords)
        elif t == "MultiLineString":
            segs.extend(coords)
        elif t == "MultiPolygon":
            for poly in coords:
                segs.extend(poly)
    return segs


def load_plate_boundaries():
    out = []
    box = REGION
    for seg in _geojson_lines(DATA / "PB2002_boundaries.json"):
        a = np.asarray(seg, dtype=float)
        lon, lat = a[:, 0], a[:, 1]
        if (lon.max() >= box["lon_min"] and lon.min() <= box["lon_max"]
                and lat.max() >= box["lat_min"] and lat.min() <= box["lat_max"]):
            out.append((lon, lat))
    return out


def load_faults():
    out = []
    box = REGION
    for seg in _geojson_lines(DATA / "gem_active_faults.geojson"):
        a = np.asarray(seg, dtype=float)
        if a.size == 0:
            continue
        lon, lat = a[:, 0], a[:, 1]
        if (lon.max() >= box["lon_min"] and lon.min() <= box["lon_max"]
                and lat.max() >= box["lat_min"] and lat.min() <= box["lat_max"]):
            out.append((lon, lat))
    return out


def load_volcanoes():
    df = pd.read_csv(DATA / "gvp_volcanoes.csv")
    m = ((df.latitude >= REGION["lat_min"]) & (df.latitude <= REGION["lat_max"])
         & (df.longitude >= REGION["lon_min"])
         & (df.longitude <= REGION["lon_max"]))
    return df[m]


# ---------------------------------------------------------------------------
# Slab1.0 GMT grids
# ---------------------------------------------------------------------------
def load_slab_grid(name):
    """Return (lon, lat, dep_km positive down) 1-D axes + 2-D depth array."""
    from netCDF4 import Dataset
    p = DATA / name
    ds = Dataset(p)
    x = ds.variables["x"][:].copy()
    y = ds.variables["y"][:].copy()
    z = ds.variables["z"][:].copy()
    ds.close()
    lon = np.where(x > 180.0, x - 360.0, x)
    order = np.argsort(lon)
    lon = lon[order]
    z = z[:, order] if z.shape[-1] == order.shape[0] else z[order, :]
    dep = np.abs(z.astype(float))
    dep[dep > 1e10] = np.nan
    return lon, np.asarray(y, dtype=float), dep


def draw_overlays(ax, dep_km=None, slab_band=20.0, slab_every=100,
                  coast_res="10m", label_slab=False):
    """Draw coastline + PB2002 plate boundaries + GEM faults + GVP volcanoes
    + Slab1.0 iso-depth contours on a cartopy GeoAxes.

    dep_km: slice depth; slab contours drawn at this depth (if given) plus a
    lighter reference grid every `slab_every` km.
    """
    import cartopy.feature as cfeature
    ax.coastlines(resolution=coast_res, lw=0.6, color="k", zorder=3)

    for lon, lat in load_plate_boundaries():
        ax.plot(lon, lat, color="k", lw=1.3, zorder=4,
                transform=_pc())
    for lon, lat in load_faults():
        ax.plot(lon, lat, color="gray", lw=0.4, alpha=0.7, zorder=3,
                transform=_pc())

    volc = load_volcanoes()
    ax.scatter(volc.longitude, volc.latitude, marker="^", s=30,
               facecolor="red", edgecolor="k", lw=0.4, zorder=6,
               transform=_pc(), label="volcano")

    levels = [dep_km] if dep_km is not None else []
    if slab_every:
        levels += [l for l in range(100, 610, slab_every)
                   if dep_km is None or abs(l - dep_km) > 0.1]
    levels = sorted(set(float(l) for l in levels)) or [100.0]
    for tag, fname, col in SLAB_PLOTS:
        try:
            slon, slat, sdep = load_slab_grid(fname)
        except Exception as e:
            print(f"  slab grid {fname} unreadable: {e}")
            continue
        m = ((slon >= REGION["lon_min"] - 1) & (slon <= REGION["lon_max"] + 1))
        n = ((slat >= REGION["lat_min"] - 1) & (slat <= REGION["lat_max"] + 1))
        if not m.any() or not n.any():
            continue
        sub_lon = slon[m]
        sub_lat = slat[n]
        sub_dep = sdep[np.ix_(n, m)]
        with np.errstate(all="ignore"):
            cs = ax.contour(sub_lon, sub_lat, sub_dep, levels=levels,
                            colors=[col], linewidths=0.9, linestyles="--",
                            transform=_pc(), zorder=5)
        if label_slab and dep_km is not None and len(cs.allsegs) > 0:
            ax.clabel(cs, fmt="%d km", fontsize=6)
    return ax


def _pc():
    import cartopy.crs as ccrs
    return ccrs.PlateCarree()
