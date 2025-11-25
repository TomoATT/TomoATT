# %%
from pytomoatt.model import ATTModel
import os
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter

# %% [markdown]
# # Step 1. Generate the ATT model based on the crust1.0 model.

# %%
# generate the .h5 model for TomoATT based on the crust1.0 model. Nearest extropolation is used.

param_file = "./3_input_params/input_params_real.yaml"
am_crust1p0 = ATTModel(param_file)
am_crust1p0.grid_data_crust1(type="vp")

# %% [markdown]
# # Step 2. Generate the ATT model based on ak135 model.

# %%
# Step 2. Generate the ATT model based on ak135 model.

# Load the 1D ak135 model from the .h5 file.
with h5py.File('ak135.h5', 'r') as f:
    points_ak135 = f['model'][:]

am_ak135 = ATTModel(param_file)

# interpolate the 1D ak135 velocity model to the depths of the ATT model.
am_depths = am_ak135.depths
vel_1d = np.interp(am_depths, points_ak135[:,0], points_ak135[:,1], left=points_ak135[0,1], right=points_ak135[-1,1])

# Set the 3D velocity model by tiling the 1D velocity model along lat and lon directions.
am_ak135.vel = np.tile(vel_1d[:, None, None], (1, am_ak135.n_rtp[1], am_ak135.n_rtp[2]))


# %% [markdown]
# # Step 3. Combine ak135 model with crust1.0 model

# %%
# 1. set two depths 
# if depth < depth_1,               vel = crust1p0 
# if depth_1 <= depth <= depth_2,   vel = linear_interp between crust1p0 and ak135 
# if depth > depth_2,               vel = ak135 

am_combined = ATTModel(param_file)

depth_1 = 35.0  
depth_2 = 70.0  

am_depths = am_ak135.depths
ratio = (am_depths - depth_1) / (depth_2 - depth_1)
ratio = np.clip(ratio, 0.0, 1.0)
ratio_3d = np.tile(ratio[:, None, None], (1, am_ak135.n_rtp[1], am_ak135.n_rtp[2]))

# linear interpolation
am_combined.vel = am_crust1p0.vel * (1 - ratio_3d) + am_ak135.vel * ratio_3d

# %% [markdown]
# # Step 4. post processing (OPTIONAL)

# %%
am_processed = ATTModel(param_file)
am_processed.vel = am_combined.vel.copy()

# 1. (OPTIONAL) monotonic increase check (OPTIONAL)
# Ensure that the velocity model increases monotonically with depth.
am_processed.vel[::-1,:,:] = np.maximum.accumulate(am_processed.vel[::-1,:,:], axis=0) 

# 2. (OPTIONAL) Gaussian smoothing to the combined model to avoid sharp discontinuities.
sigma = [1, 1, 1]  # standard deviation for Gaussian kernel along each axis (ddep, dlat, dlon)
am_processed.vel = gaussian_filter(am_processed.vel, sigma=sigma, mode='nearest')


# %%
# output as .h5 file
n_rtp = am_processed.n_rtp
fname = "constant_velocity_N%d_%d_%d_PyTomoATT.h5"%(n_rtp[0], n_rtp[1], n_rtp[2])
am_processed.write(fname)

# %%
# visualization of the central lat-lon slice
import matplotlib.pyplot as plt
dep = am_processed.depths
vel = am_processed.vel[:, am_processed.n_rtp[1]//2, am_processed.n_rtp[2]//2]   
lat = am_processed.latitudes[am_processed.n_rtp[1]//2]
lon = am_processed.longitudes[am_processed.n_rtp[2]//2]
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.plot(vel, dep, label='Velocity', color='blue')
ax.invert_yaxis()
ax.set_xlabel('Vp (km/s)', fontsize=16)
ax.set_ylabel('Depth (km)', fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax.set_title(f'Velocity Profile at Lat: {lat:.2f}, Lon: {lon:.2f}', fontsize=16)
ax.grid()
ax.legend(fontsize=16)
plt.show()

os.makedirs("figs", exist_ok=True)
fig.savefig("figs/velocity_profile_lat%.2f_lon%.2f.png"%(lat, lon), facecolor='white', edgecolor='white', bbox_inches='tight')


