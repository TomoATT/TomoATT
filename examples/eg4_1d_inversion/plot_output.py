import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    os.mkdir("img")
except:
    pass

dep = np.linspace(50,-10, 61)    

with h5py.File("OUTPUT_FILES/OUTPUT_FILES_1dinv_inv/final_model.h5", "r") as f:
    vel_final= np.array(f["vel"])
with h5py.File("2_models/model_init_N61_61_61.h5", "r") as f:
    vel_init = np.array(f["vel"])
with h5py.File("2_models/model_ckb_N61_61_61.h5", "r") as f:
    vel_ckb = np.array(f["vel"])

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.plot(vel_init[:,0,0] , dep, label="init")
ax.plot(vel_ckb[:,0,0], dep, label="ckb")
ax.plot(vel_final[:,0,0], dep, label="inv")
ax.grid()
ax.set_xlabel("Velocity (m/s)",fontsize=16)
ax.set_ylabel("Depth (km)",fontsize=16)
ax.get_xaxis().set_tick_params(labelsize=16)
ax.get_yaxis().set_tick_params(labelsize=16)
ax.set_xlim([4.5,8.5])
ax.set_ylim([0,50])

plt.gca().invert_yaxis()
plt.legend(fontsize=16)

plt.show()
fig.savefig("img/1d_model_inversion.png", dpi=300, bbox_inches="tight", edgecolor="w", facecolor="w")
