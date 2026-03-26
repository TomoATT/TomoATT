#!/usr/bin/env python3
"""
Generate a TomoATT initial velocity model from ETOPO topography + CRUST1.0.

Outputs:
  - {output}_model.h5  : HDF5 with 'vel', 'vel_s', 'xi', 'eta' [nr, nt, np]
  - {output}_topo.h5   : HDF5 with '/lon', '/lat', '/z' (meters, SurfATT-compatible)

Usage:
  python make_model_etopo_crust1.py --param_file input_params.yaml --output mymodel
  python make_model_etopo_crust1.py --param_file input_params.yaml --output mymodel \
         --ak135 path/to/ak135.h5 --smooth

Requirements:
  pip install numpy scipy h5py pyyaml pytomoatt
  For ETOPO: pip install pygmt   (or xarray + netCDF4 as fallback)
"""

import argparse

import numpy as np
import h5py
import yaml
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml_grid(param_file):
    """Parse domain section of a TomoATT YAML parameter file.

    Returns
    -------
    depths : ndarray   – 1-D array of depth values (km), surface→deep
    lats   : ndarray   – 1-D array of latitudes (deg)
    lons   : ndarray   – 1-D array of longitudes (deg)
    n_rtp  : list[int] – [nr, nt, np]
    """
    with open(param_file, "r") as f:
        cfg = yaml.safe_load(f)

    dom = cfg["domain"]
    dep_min, dep_max = dom["min_max_dep"]
    lat_min, lat_max = dom["min_max_lat"]
    lon_min, lon_max = dom["min_max_lon"]
    n_rtp = list(dom["n_rtp"])

    depths = np.linspace(dep_min, dep_max, n_rtp[0])
    lats = np.linspace(lat_min, lat_max, n_rtp[1])
    lons = np.linspace(lon_min, lon_max, n_rtp[2])

    return depths, lats, lons, n_rtp


# ---------------------------------------------------------------------------
# ETOPO download
# ---------------------------------------------------------------------------

def download_etopo(lat_min, lat_max, lon_min, lon_max, cache_dir=None):
    """Download ETOPO elevation using pygmt (preferred) or xarray/OPeNDAP fallback.

    Returns
    -------
    etopo_lats : ndarray (1-D)
    etopo_lons : ndarray (1-D)
    etopo_elev : ndarray (2-D, [nlat, nlon]) – elevation in meters
    """
    # Add small margin so interpolation at edges works
    margin = 0.5
    region = [
        lon_min - margin,
        lon_max + margin,
        lat_min - margin,
        lat_max + margin,
    ]

    # --- Try pygmt ---
    try:
        import pygmt

        # Select finest available resolution that keeps the grid manageable
        span_deg = max(lat_max - lat_min, lon_max - lon_min)
        if span_deg < 5:
            res = "15s"
        elif span_deg < 30:
            res = "01m"
        else:
            res = "05m"

        print(f"  Downloading ETOPO via pygmt (resolution={res}) ...")
        grid = pygmt.datasets.load_earth_relief(
            resolution=res, region=region, registration="gridline"
        )
        etopo_lats = grid.lat.values.astype(np.float64)
        etopo_lons = grid.lon.values.astype(np.float64)
        etopo_elev = grid.values.astype(np.float64)  # metres
        print(f"  ETOPO grid: {etopo_elev.shape}")
        return etopo_lats, etopo_lons, etopo_elev

    except ImportError:
        pass

    # --- Fallback: NOAA THREDDS OPeNDAP via xarray ---
    try:
        import xarray as xr

        print("  pygmt not available – trying NOAA THREDDS OPeNDAP ...")
        url = (
            "https://www.ngdc.noaa.gov/thredds/dodsC/"
            "global/ETOPO2022/60s/60s_surface_elev_netcdf/"
            "ETOPO_2022_v1_60s_N90W180_surface.nc"
        )
        ds = xr.open_dataset(url)
        ds_sub = ds.sel(lat=slice(region[2], region[3]),
                        lon=slice(region[0], region[1]))
        etopo_lats = ds_sub.lat.values.astype(np.float64)
        etopo_lons = ds_sub.lon.values.astype(np.float64)
        etopo_elev = ds_sub["z"].values.astype(np.float64)
        ds.close()
        print(f"  ETOPO grid (OPeNDAP): {etopo_elev.shape}")
        return etopo_lats, etopo_lons, etopo_elev

    except Exception as exc:
        print(f"  OPeNDAP fallback failed: {exc}")

    raise RuntimeError(
        "Cannot download ETOPO.  Install pygmt  (pip install pygmt)  or  "
        "xarray + netCDF4  (pip install xarray netCDF4)."
    )


# ---------------------------------------------------------------------------
# CRUST1.0 via pytomoatt
# ---------------------------------------------------------------------------

def load_crust1_profiles(lats, lons):
    """Load CRUST1.0 velocity profiles using pytomoatt's bundled data.

    For each (lat, lon) point on the model grid, returns a 1-D profile
    of (depth, Vp, Vs) with bilinear interpolation between the four
    nearest CRUST1.0 grid nodes.

    Parameters
    ----------
    lats : 1-D array of latitudes (deg)
    lons : 1-D array of longitudes (deg)

    Returns
    -------
    profiles : list of ndarray, shape (n_layers, 3)
        Each entry: [[depth_km, vp, vs], ...] for one (lat, lon) column.
        profiles[j * len(lons) + i] corresponds to (lats[j], lons[i]).
    """
    try:
        from pytomoatt.io.crustmodel import CrustModel, degree_to_idx_and_ratio
    except ImportError:
        raise ImportError(
            "pytomoatt is required for CRUST1.0 data.\n"
            "Install it with:  pip install pytomoatt"
        )

    cm = CrustModel()
    print(f"  Loaded pytomoatt CRUST1.0 data ({len(cm.points_dict)} profiles)")

    profiles = []
    for lat in lats:
        for lon in lons:
            # Bilinear interpolation indices
            idx_lon_left, ratio_lon = degree_to_idx_and_ratio(lon)
            idx_lat_left, ratio_lat = degree_to_idx_and_ratio(lat)
            idx_lon_right = idx_lon_left + 1
            idx_lat_right = idx_lat_left + 1

            # Clamp edges
            if idx_lon_left < 0:
                idx_lon_left = 0
                idx_lon_right = 1
                ratio_lon = 0.0
            if idx_lon_right > 359:
                idx_lon_right = 359
                ratio_lon = 0.0
            if idx_lat_left < 90:
                idx_lat_left = 90
                idx_lat_right = 91
                ratio_lat = 0.0
            if idx_lat_right > 269:
                idx_lat_right = 269
                ratio_lat = 0.0

            # Get four corner profiles (may have different numbers of layers)
            p_ll = cm.points_dict[(idx_lon_left, idx_lat_left)]
            p_lr = cm.points_dict[(idx_lon_right, idx_lat_left)]
            p_ul = cm.points_dict[(idx_lon_left, idx_lat_right)]
            p_ur = cm.points_dict[(idx_lon_right, idx_lat_right)]

            # Common depth axis from the union of all corner depths
            all_depths = np.unique(np.concatenate([
                p_ll[:, 0], p_lr[:, 0], p_ul[:, 0], p_ur[:, 0]
            ]))

            # Interpolate each corner onto the common depth axis,
            # then bilinearly average (handles variable layer counts)
            n_common = len(all_depths)
            interp_profile = np.zeros((n_common, 3))
            interp_profile[:, 0] = all_depths
            w_ll = (1 - ratio_lon) * (1 - ratio_lat)
            w_lr = ratio_lon * (1 - ratio_lat)
            w_ul = (1 - ratio_lon) * ratio_lat
            w_ur = ratio_lon * ratio_lat
            for col_out, col_in in enumerate([3, 4]):  # vp, vs
                v_ll = np.interp(all_depths, p_ll[:, 0], p_ll[:, col_in],
                                 left=p_ll[0, col_in], right=p_ll[-1, col_in])
                v_lr = np.interp(all_depths, p_lr[:, 0], p_lr[:, col_in],
                                 left=p_lr[0, col_in], right=p_lr[-1, col_in])
                v_ul = np.interp(all_depths, p_ul[:, 0], p_ul[:, col_in],
                                 left=p_ul[0, col_in], right=p_ul[-1, col_in])
                v_ur = np.interp(all_depths, p_ur[:, 0], p_ur[:, col_in],
                                 left=p_ur[0, col_in], right=p_ur[-1, col_in])
                interp_profile[:, col_out + 1] = (
                    v_ll * w_ll + v_lr * w_lr + v_ul * w_ul + v_ur * w_ur
                )

            profiles.append(interp_profile)

    return profiles


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def fill_model(depths, lats, lons, n_rtp,
               etopo_lats, etopo_lons, etopo_elev,
               crust1_profiles,
               ak135_file=None, blend_thickness=50.0):
    """Build 3-D Vp and Vs arrays by combining ETOPO + CRUST1.0 (+ AK135).

    Parameters
    ----------
    depths : 1-D array of depth values (km)
    lats, lons : 1-D arrays of lat/lon (deg)
    n_rtp : [nr, nt, np]
    etopo_* : ETOPO data from download_etopo()
    crust1_profiles : list of ndarray from load_crust1_profiles()
    ak135_file : optional path to ak135.h5
    blend_thickness : km over which to blend from CRUST1.0 to AK135

    Returns
    -------
    vel_p  : ndarray [nr, nt, np]
    vel_s  : ndarray [nr, nt, np]
    topo_z : ndarray [nt, np] – surface elevation in metres
    """
    nr, nt, np_ = n_rtp

    # 1) Interpolate ETOPO elevation onto model lat/lon grid (metres)
    if etopo_lats[0] > etopo_lats[-1]:
        etopo_lats = etopo_lats[::-1]
        etopo_elev = etopo_elev[::-1, :]

    etopo_interp = RegularGridInterpolator(
        (etopo_lats, etopo_lons), etopo_elev,
        method="linear", bounds_error=False, fill_value=None,
    )
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    topo_z = etopo_interp((grid_lat, grid_lon))  # [nt, np_], metres
    topo_km = topo_z / 1000.0  # km, positive=above sea level, negative=below

    # 2) AK135 1-D model (optional)
    ak135_vp_1d = None
    ak135_vs_1d = None
    if ak135_file is not None:
        with h5py.File(ak135_file, "r") as f:
            pts = f["model"][:]
        ak135_vp_1d = np.interp(depths, pts[:, 0], pts[:, 1],
                                left=pts[0, 1], right=pts[-1, 1])
        ak135_vs_1d = np.interp(depths, pts[:, 0], pts[:, 2],
                                left=pts[0, 2], right=pts[-1, 2])

    # 3) Fill model column by column using CRUST1.0 profiles
    vel_p = np.zeros((nr, nt, np_), dtype=np.float64)
    vel_s = np.zeros((nr, nt, np_), dtype=np.float64)

    max_moho_depth = 0.0

    for jt in range(nt):
        for jp in range(np_):
            z_surf = topo_km[jt, jp]  # km, positive = land, negative = ocean

            # CRUST1.0 profile: [[depth, vp, vs], ...] sorted by depth
            prof = crust1_profiles[jt * np_ + jp]
            prof_dep = prof[:, 0]  # depth (km)
            prof_vp = prof[:, 1]
            prof_vs = prof[:, 2]
            moho_depth = prof_dep[-1]  # deepest layer boundary = Moho
            max_moho_depth = max(max_moho_depth, moho_depth)

            for ir in range(nr):
                dep = depths[ir]

                # --- Air nodes (above terrain on land) ---
                if z_surf > 0 and dep < -z_surf:
                    vel_p[ir, jt, jp] = prof_vp[0]
                    vel_s[ir, jt, jp] = prof_vs[0]
                    continue

                # --- Ocean water ---
                if z_surf < 0 and dep >= 0 and dep < -z_surf:
                    vel_p[ir, jt, jp] = 1.5
                    vel_s[ir, jt, jp] = 0.0
                    continue

                # --- Sub-surface: interpolate from CRUST1.0 profile ---
                vp_val = np.interp(dep, prof_dep, prof_vp,
                                   left=prof_vp[0], right=prof_vp[-1])
                vs_val = np.interp(dep, prof_dep, prof_vs,
                                   left=prof_vs[0], right=prof_vs[-1])
                vel_p[ir, jt, jp] = vp_val
                vel_s[ir, jt, jp] = vs_val

    # 4) AK135 blend below Moho (if requested)
    if ak135_vp_1d is not None:
        print("  Blending with AK135 below Moho ...")
        blend_start = max_moho_depth + 10.0
        blend_end = blend_start + blend_thickness

        for ir in range(nr):
            dep = depths[ir]
            if dep <= blend_start:
                continue
            if dep >= blend_end:
                vel_p[ir, :, :] = ak135_vp_1d[ir]
                vel_s[ir, :, :] = ak135_vs_1d[ir]
            else:
                ratio = (dep - blend_start) / blend_thickness
                vel_p[ir, :, :] = (1 - ratio) * vel_p[ir, :, :] + ratio * ak135_vp_1d[ir]
                vel_s[ir, :, :] = (1 - ratio) * vel_s[ir, :, :] + ratio * ak135_vs_1d[ir]

    return vel_p, vel_s, topo_z


def _shallowest_rock_vel(col_vp, col_vs):
    """Return the shallowest non-water, non-ice velocity from a CRUST1.0 column."""
    for vp, vs in zip(col_vp, col_vs):
        if vp > 0 and vp != 1.5:
            return vp, vs
    return col_vp[-1], col_vs[-1]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_model(fname, vel_p, vel_s):
    """Write TomoATT model HDF5 file."""
    with h5py.File(fname, "w") as f:
        f.create_dataset("vel", data=vel_p)
        f.create_dataset("vel_s", data=vel_s)
        f.create_dataset("xi", data=np.zeros_like(vel_p))
        f.create_dataset("eta", data=np.zeros_like(vel_p))
    print(f"  Wrote model: {fname}  shape={vel_p.shape}")


def write_topo(fname, lats, lons, topo_z):
    """Write SurfATT-compatible topo HDF5 file.

    Format: /lon (1-D), /lat (1-D), /z (2-D [nlat, nlon], metres).
    """
    with h5py.File(fname, "w") as f:
        f.create_dataset("lat", data=lats)
        f.create_dataset("lon", data=lons)
        f.create_dataset("z", data=topo_z)
    print(f"  Wrote topo:  {fname}  shape={topo_z.shape}")


# ---------------------------------------------------------------------------
# Optional smoothing
# ---------------------------------------------------------------------------

def smooth_model(vel, sigma_nodes=(1.0, 1.0, 1.0)):
    """Apply Gaussian smoothing to a 3-D velocity array."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(vel, sigma=sigma_nodes, mode="nearest")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate TomoATT initial model from ETOPO + CRUST1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--param_file", required=True,
                        help="TomoATT YAML parameter file (domain section used)")
    parser.add_argument("--output", default="initial",
                        help="Output filename prefix (default: initial)")
    parser.add_argument("--ak135", default=None, dest="ak135_file",
                        help="Path to ak135.h5 for deep-mantle blending")
    parser.add_argument("--smooth", action="store_true",
                        help="Apply Gaussian smoothing (1 grid-node sigma)")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory for caching downloaded data")

    args = parser.parse_args()

    print(f"Parameter file: {args.param_file}")
    depths, lats, lons, n_rtp = load_yaml_grid(args.param_file)
    print(f"  Grid: {n_rtp}  depth=[{depths[0]:.1f}, {depths[-1]:.1f}] km  "
          f"lat=[{lats[0]:.2f}, {lats[-1]:.2f}]  lon=[{lons[0]:.2f}, {lons[-1]:.2f}]")

    # 1. Download ETOPO
    print("\n[1/4] Downloading ETOPO ...")
    etopo_lats, etopo_lons, etopo_elev = download_etopo(
        lats[0], lats[-1], lons[0], lons[-1], cache_dir=args.cache_dir
    )

    # 2. Load CRUST1.0
    print("\n[2/4] Loading CRUST1.0 ...")
    crust1_profiles = load_crust1_profiles(lats, lons)

    # 3. Fill model
    print("\n[3/4] Building velocity model ...")
    vel_p, vel_s, topo_z = fill_model(
        depths, lats, lons, n_rtp,
        etopo_lats, etopo_lons, etopo_elev,
        crust1_profiles,
        ak135_file=args.ak135_file,
    )

    if args.smooth:
        print("  Applying Gaussian smoothing ...")
        vel_p = smooth_model(vel_p)
        vel_s = smooth_model(vel_s)

    # 4. Write outputs
    print("\n[4/4] Writing outputs ...")
    model_fname = f"{args.output}_model.h5"
    topo_fname = f"{args.output}_topo.h5"
    write_model(model_fname, vel_p, vel_s)
    write_topo(topo_fname, lats, lons, topo_z)

    # Print recommended YAML snippet
    mean_topo = np.mean(topo_z) / 1000.0
    max_elev = np.max(topo_z) / 1000.0
    min_elev = np.min(topo_z) / 1000.0
    print(f"\n  Topo stats: mean={mean_topo:.2f} km, "
          f"max={max_elev:.2f} km (land), min={min_elev:.2f} km (ocean)")

    print("\n  Suggested YAML model section:")
    print(f"    model:")
    print(f"      init_model_path: {model_fname}")
    print(f"      topo_file: {topo_fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
