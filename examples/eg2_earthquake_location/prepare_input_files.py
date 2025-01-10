import numpy as np
import os
import sys
try:
    from pytomoatt.checkerboard import Checker
    from pytomoatt.src_rec import SrcRec
except:
    print("ERROR: ATTModel not found. Please install pytomoatt first."
          "See https://tomoatt.github.io/PyTomoATT/installation.html for details.")
    sys.exit(1)


def build_ckb_model(output_dir="2_models"):
    cbk = Checker(f'{output_dir}/model_init_N61_61_61.h5', para_fname="./3_input_params/input_params_signal.yaml")
    cbk.checkerboard(
        n_pert_x=2, n_pert_y=2, n_pert_z=2,
        pert_vel=0.2, pert_ani=0.1, ani_dir=60.0,
        lim_x=[0.5, 1.5], lim_y=[0.5, 1.5], lim_z=[0, 40]
    )
    cbk.write(f'{output_dir}/model_ckb_N61_61_61.h5')


if __name__ == "__main__":
    # download src_rec_file
    url = 'https://zenodo.org/records/14053821/files/src_rec_config.dat'
    path = "1_src_rec_files/src_rec_config.dat"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        sr = SrcRec.read(url)
        sr.write(path)

    # build initial model
    output_dir = "2_models"
    os.makedirs(output_dir, exist_ok=True)
    build_ckb_model(output_dir)


