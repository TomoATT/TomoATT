#!/usr/bin/env python3
"""
Visualise the Japan initial velocity model.

Plots:
  1. Vertical cross-section at 138°E (central Honshu) – Vp
  2. Horizontal slice at 35 km depth – Vp
  3. Topography map (elevation in km)

Usage:
    python plot_model.py [--model japan_model.h5] [--topo japan_topo.h5]
"""

import argparse
import os

import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt


def load_grid(param_file):
    """Load grid from YAML parameter file."""
    with open(param_file, "r") as f:
        cfg = yaml.safe_load(f)
    dom = cfg["domain"]
    n_rtp = list(dom["n_rtp"])
    depths = np.linspace(dom["min_max_dep"][0], dom["min_max_dep"][1], n_rtp[0])
    lats = np.linspace(dom["min_max_lat"][0], dom["min_max_lat"][1], n_rtp[1])
    lons = np.linspace(dom["min_max_lon"][0], dom["min_max_lon"][1], n_rtp[2])
    return depths, lats, lons, n_rtp


def main():
    parser = argparse.ArgumentParser(description="Plot Japan initial model")
    parser.add_argument("--model", default="japan_model.h5")
    parser.add_argument("--topo", default="japan_topo.h5")
    parser.add_argument("--param_file", default="input_params.yaml")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model) if not os.path.isabs(args.model) else args.model
    topo_path = os.path.join(script_dir, args.topo) if not os.path.isabs(args.topo) else args.topo
    param_path = os.path.join(script_dir, args.param_file) if not os.path.isabs(args.param_file) else args.param_file

    depths, lats, lons, _ = load_grid(param_path)

    with h5py.File(model_path, "r") as f:
        vel_p = f["vel"][:]

    with h5py.File(topo_path, "r") as f:
        topo_z = f["z"][:]  # metres

    os.makedirs(os.path.join(script_dir, "figs"), exist_ok=True)

    # --- Plot 1: Vertical cross-section at 138°E ---
    target_lon = 138.0
    ilon = np.argmin(np.abs(lons - target_lon))
    actual_lon = lons[ilon]

    fig, ax = plt.subplots(figsize=(10, 8))
    LAT, DEP = np.meshgrid(lats, depths)
    vmin, vmax = 1.0, 9.0
    im = ax.pcolormesh(LAT, DEP, vel_p[:, :, ilon], cmap="seismic_r",
                       vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("Latitude (°)", fontsize=14)
    ax.set_ylabel("Depth (km)", fontsize=14)
    ax.set_title(f"Vp at lon={actual_lon:.1f}°E", fontsize=16)
    ax.invert_yaxis()
    ax.set_ylim([300, depths[0]])
    fig.colorbar(im, ax=ax, label="Vp (km/s)")
    fig.savefig(os.path.join(script_dir, "figs", "vp_xsection_138E.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved figs/vp_xsection_138E.png")

    # --- Plot 2: Horizontal slice at 35 km depth ---
    target_dep = 35.0
    idep = np.argmin(np.abs(depths - target_dep))
    actual_dep = depths[idep]

    fig, ax = plt.subplots(figsize=(10, 8))
    LON, LAT2 = np.meshgrid(lons, lats)
    im = ax.pcolormesh(LON, LAT2, vel_p[idep, :, :], cmap="seismic_r",
                       vmin=5.0, vmax=9.0, shading="auto")
    ax.set_xlabel("Longitude (°)", fontsize=14)
    ax.set_ylabel("Latitude (°)", fontsize=14)
    ax.set_title(f"Vp at depth={actual_dep:.1f} km", fontsize=16)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="Vp (km/s)")
    fig.savefig(os.path.join(script_dir, "figs", "vp_horizontal_35km.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved figs/vp_horizontal_35km.png")

    # --- Plot 3: Topography map ---
    fig, ax = plt.subplots(figsize=(10, 8))
    topo_km = topo_z / 1000.0
    LON, LAT3 = np.meshgrid(lons, lats)
    im = ax.pcolormesh(LON, LAT3, topo_km, cmap="terrain",
                       vmin=-8, vmax=4, shading="auto")
    ax.set_xlabel("Longitude (°)", fontsize=14)
    ax.set_ylabel("Latitude (°)", fontsize=14)
    ax.set_title("ETOPO Topography", fontsize=16)
    ax.set_aspect("equal")
    # Add 0 km contour (coastline)
    ax.contour(LON, LAT3, topo_km, levels=[0], colors="k", linewidths=0.8)
    fig.colorbar(im, ax=ax, label="Elevation (km)")
    fig.savefig(os.path.join(script_dir, "figs", "topography.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved figs/topography.png")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
