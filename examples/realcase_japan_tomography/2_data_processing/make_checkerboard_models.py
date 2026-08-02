#!/usr/bin/env python3
"""
make_checkerboard_models.py

Generate checkerboard test models for the Japan production setup, based on
PyTomoATT's checkerboard.Checker with the real japan_model.h5 as background.

Pipeline role:
    make_checkerboard_models.py   (this script)  -> checker_*.h5
    -> run TomoATT FORWARD with checker model as init_model and the merged
       src_rec_file_japan_tele.dat  -> synthetic src_rec_file_forward.dat
    -> run TomoATT INVERSION from the unperturbed japan_model.h5
    -> compare_checkerboard.py (recovery maps / correlation per depth)

The japan initial model has no `zeta` dataset but PyTomoATT.Checker expects
one, so a zero-zeta copy is staged first.

Default cell size: ~2 deg x 2 deg x ~42 km (13 x 11 x 5 sign cells over the
production domain), perturbation +/-3 % in P velocity; an independent
anisotropy-only checkerboard (+/-4 % anisotropy amplitude with +/-45 deg
fast directions) is produced for the azimuthal-anisotropy recovery test.
"""

import argparse
import shutil
from pathlib import Path

import h5py
import yaml

from pytomoatt.checkerboard import Checker

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_MODEL = SCRIPT_DIR / ".." / ".." / "eg_japan_initial_model" / "japan_model.h5"
PARA_FILE = SCRIPT_DIR / "checkerboard_model_params.yml"   # written by this script
ZERO_ZETA_MODEL = SCRIPT_DIR / "japan_model_zerzeta.h5"

# production domain (must match the TomoATT run YAMLs)
MIN_MAX_DEP = [-10, 200]
MIN_MAX_LAT = [24, 46]
MIN_MAX_LON = [122, 148]


def stage_zero_zeta_model():
    """Copy japan_model.h5 adding a zero zeta dataset (Checker requires it)."""
    with h5py.File(BASE_MODEL) as fin, h5py.File(ZERO_ZETA_MODEL, "w") as fout:
        for key in fin.keys():
            fout.create_dataset(key, data=fin[key][:])
        fout.create_dataset("zeta", data=0.0 * fin["xi"][:])
    shape = h5py.File(ZERO_ZETA_MODEL)["vel"].shape
    return shape  # (n_dep, n_lat, n_lon) of the model grid


def write_para_file(shape):
    n_rtp = list(shape)
    para = {
        "domain": {
            "min_max_dep": MIN_MAX_DEP,
            "min_max_lat": MIN_MAX_LAT,
            "min_max_lon": MIN_MAX_LON,
            "n_rtp": n_rtp,
        }
    }
    with open(PARA_FILE, "w") as f:
        yaml.dump(para, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, nargs=3, default=[13, 11, 5],
                    metavar=("NLON", "NLAT", "NDEP"),
                    help="checkerboard sign-cell counts (lon, lat, dep); "
                         "coarse standard: 13 11 5 (~2x2deg x 42km), "
                         "fine: 26 22 10 (~1x1deg x 21km)")
    ap.add_argument("--pert-vel", type=float, default=0.03)
    ap.add_argument("--pert-ani", type=float, default=0.04)
    ap.add_argument("--tag", type=str, default="coarse")
    ap.add_argument("--lim-lon", type=float, nargs=2, default=[126.0, 144.0])
    ap.add_argument("--lim-lat", type=float, nargs=2, default=[28.0, 42.0])
    args = ap.parse_args()

    shape = stage_zero_zeta_model()
    write_para_file(shape)
    print(f"model grid  : {shape} (dep, lat, lon)")
    print(f"cells       : {args.n_cells[0]} x {args.n_cells[1]} x "
          f"{args.n_cells[2]} over "
          f"{MIN_MAX_LON[1]-MIN_MAX_LON[0]}deg x "
          f"{MIN_MAX_LAT[1]-MIN_MAX_LAT[0]}deg x "
          f"{MIN_MAX_DEP[1]-MIN_MAX_DEP[0]}km")

    out_dir = SCRIPT_DIR / "checkerboard_models"
    out_dir.mkdir(exist_ok=True)

    # --- velocity-only checkerboard ---------------------------------------
    ck = Checker(str(ZERO_ZETA_MODEL), str(PARA_FILE))
    ck.checkerboard(n_pert_x=args.n_cells[0], n_pert_y=args.n_cells[1],
                    n_pert_z=args.n_cells[2],
                    pert_vel=args.pert_vel, pert_ani=0.0,
                    lim_x=args.lim_lon, lim_y=args.lim_lat)
    f_vel = out_dir / f"checker_vel_{args.tag}.h5"
    ck.write(str(f_vel))
    print(f"wrote {f_vel}")

    # --- anisotropy-only checkerboard --------------------------------------
    ck = Checker(str(ZERO_ZETA_MODEL), str(PARA_FILE))
    ck.checkerboard(n_pert_x=args.n_cells[0], n_pert_y=args.n_cells[1],
                    n_pert_z=args.n_cells[2],
                    pert_vel=0.0, pert_ani=args.pert_ani, ani_dir=45,
                    lim_x=args.lim_lon, lim_y=args.lim_lat)
    f_ani = out_dir / f"checker_ani_{args.tag}.h5"
    ck.write(str(f_ani))
    print(f"wrote {f_ani}")

    print("\nnext steps:")
    print("  1) forward run with each checker model (input_params_cb_forward_*)")
    print("  2) inversion run from japan_model.h5 with the synthesized src_rec")
    print("  3) compare_checkerboard.py for recovery maps")


if __name__ == "__main__":
    main()
