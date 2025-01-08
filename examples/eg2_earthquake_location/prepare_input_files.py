import numpy as np
import os
import sys
try:
    from pytomoatt.checkerboard import Checker
    from pytomoatt.src_rec import SrcRec
    from pytomoatt.model import ATTModel
except:
    print("ERROR: ATTModel not found. Please install pytomoatt first."
          "See https://tomoatt.github.io/PyTomoATT/installation.html for details.")
    sys.exit(1)


class BuildInitialModel():
    def __init__(self, par_file="./3_input_params/input_params_signal.yaml", output_dir="2_models"):
        """
        Build initial model for tomography inversion
        """
        self.am = ATTModel(par_file)
        self.output_dir = output_dir

    def build_initial_model(self, vel_min=5.0, vel_max=8.0):
        """
        Build initial model for tomography inversion
        """
        self.am.vel[self.am.depths < 0, :, :] = vel_min
        idx = np.where((0 <= self.am.depths) & (self.am.depths < 40.0))[0]
        self.am.vel[idx, :, :] = np.linspace(vel_min, vel_max, idx.size)[::-1][:, np.newaxis, np.newaxis]
        self.am.vel[self.am.depths >= 40.0, :, :] = vel_max


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

    bim = BuildInitialModel(output_dir=output_dir)
    bim.build_initial_model()
    bim.am.write('{}/model_init_N{:d}_{:d}_{:d}.h5'.format(bim.output_dir, *bim.am.n_rtp))

    build_ckb_model(output_dir)


