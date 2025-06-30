# %%
# download src_ref_files from Zenodo
import os
import numpy as np
import sys
try:
    from pytomoatt.model import ATTModel
    from pytomoatt.checkerboard import Checker
    from pytomoatt.src_rec import SrcRec
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

    def build_ckb_model(self):
        """
        Build checkerboard model for tomography inversion
        """
        nr = self.am.n_rtp[0]
        for ir in range(nr):
            dep = self.am.depths[ir] 
            self.am.vel[ir, :, :] = (1 + 0.05 * np.sin(np.pi * dep / 10.0)) * self.am.vel[ir, :, :]



if __name__ == "__main__":
    # download src_rec_config.dat
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

    bim.build_ckb_model()
    bim.am.write('{}/model_ckb_N{:d}_{:d}_{:d}.h5'.format(bim.output_dir, *bim.am.n_rtp))



