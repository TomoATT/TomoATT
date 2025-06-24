# %%
from pytomoatt.model import ATTModel
from pytomoatt.checkerboard import Checker
import os
import numpy as np
import h5py

# %% [markdown]
# # read YAML parameter file to obtain the grid parameters for the model

# %%
output_path = "models"
os.makedirs(output_path, exist_ok=True)

par_file = "input_params/input_params.yaml"

att_model = ATTModel(par_file)

n_rtp          = att_model.n_rtp       # grid node numbers in r (depth), t (longtitude), p (latitude) directions
am_depths      = att_model.depths      # depths in km
am_latitudes   = att_model.latitudes   # latitudes in degrees
am_longitudes  = att_model.longitudes  # longitudes in degrees

print("grid node numbers (N_r, N_t, N_p):", n_rtp)
print("depths (km):", am_depths)
print("latitudes (degree):", am_latitudes)
print("longitudes (degree):", am_longitudes)

# %% [markdown]
# # eg1. generate model with constant velocity

# %%
# case 1. ---------- generate a constant velocity model using PyTomoATT module -------------

# set the velocity model to a constant value
constant_v = 6.0 # constant velocity (km/s)
att_model.vel[:,:,:] = constant_v
att_model.xi[:,:,:] = 0.0
att_model.eta[:,:,:] = 0.0

# write the model to a file
fname = "%s/constant_velocity_N%d_%d_%d_PyTomoATT.h5"%(output_path, n_rtp[0], n_rtp[1], n_rtp[2])
att_model.write(fname)
print("generate model using PyTomoATT:", fname)


# case 2. ---------- generate a constant velocity model using plain loop (h5py module is required) -------------

# set the velocity model to a constant value
vel = np.zeros(n_rtp)
xi  = np.zeros(n_rtp)
eta = np.zeros(n_rtp)
for ir in range(n_rtp[0]):
    for it in range(n_rtp[1]):
        for ip in range(n_rtp[2]):
            vel[ir, it, ip] = constant_v

fname = "%s/constant_velocity_N%d_%d_%d_loop.h5"%(output_path, n_rtp[0], n_rtp[1], n_rtp[2])

with h5py.File(fname, 'w') as f:
    f.create_dataset('vel', data=vel)
    f.create_dataset('xi',  data=xi)
    f.create_dataset('eta', data=eta)
print("generate model using plain loop:", fname)

# %% [markdown]
# # eg2. generate a linear velocity model:
# vel = 5.0,               if        depth < 0 km
# 
# vel = 5.0 + 0.1 * depth, if 0 km <= depth <= 30 km
# 
# vel = 8.0,               if depth > 30 km

# %%
# case 1. ---------- generate a constant velocity model using PyTomoATT module -------------

# set the velocity model to a constant value
idx = np.where((am_depths >= 0.0) & (am_depths <= 30.0))
depth = am_depths[idx]
att_model.vel[idx,:,:] = 5.0 + 0.1 * depth[:, np.newaxis, np.newaxis]   # velocity increases linearly from 5.0 to 8.0 km/s
att_model.vel[np.where(am_depths > 30.0),:,:] = 8.0                     # velocity is constant at 8.0 km/s below 30.0 km depth
att_model.vel[np.where(am_depths <  0.0),:,:] = 5.0                     # velocity is constant at 5.0 km/s above 0.0 km depth

att_model.xi[:,:,:] = 0.0
att_model.eta[:,:,:] = 0.0

# write the model to a file
fname = "%s/linear_velocity_N%d_%d_%d_PyTomoATT.h5"%(output_path, n_rtp[0], n_rtp[1], n_rtp[2])
att_model.write(fname)
print("generate model using PyTomoATT:", fname)

# case 2. ---------- generate a linear velocity model using plain loop (h5py module is required) -------------

# set the velocity model to a linear value
vel = np.zeros(n_rtp)
xi  = np.zeros(n_rtp)
eta = np.zeros(n_rtp)

for ir in range(n_rtp[0]):
    for it in range(n_rtp[1]):
        for ip in range(n_rtp[2]):
            if am_depths[ir] < 0.0:
                vel[ir, it, ip] = 5.0
            elif am_depths[ir] <= 30.0:
                vel[ir, it, ip] = 5.0 + 0.1 * am_depths[ir]
            else:
                vel[ir, it, ip] = 8.0
fname = "%s/linear_velocity_N%d_%d_%d_loop.h5"%(output_path, n_rtp[0], n_rtp[1], n_rtp[2])

with h5py.File(fname, 'w') as f:
    f.create_dataset('vel', data=vel)
    f.create_dataset('xi',  data=xi)
    f.create_dataset('eta', data=eta)
print("generate model using plain loop:", fname)

# %% [markdown]
# # eg3. generate checkerboard model for velocity and anisotropy.
# 
# assign perturbation 

# %%
# case 1. ---------- generate a constant velocity model using PyTomoATT module -------------

# file name of the background model
bg_model_fname = "%s/linear_velocity_N%d_%d_%d_PyTomoATT.h5" % (output_path, n_rtp[0], n_rtp[1], n_rtp[2])

lim_x       = [0.5, 1.5]    # longitude limits of the checkerboard
lim_y       = [0.25, 0.75]  # latitude limits of the checkerboard
lim_z       = [0, 30]       # depth limits of the checkerboard
pert_vel    = 0.1           # amplitude of velocity perturbation (%)
pert_ani    = 0.05          # amplitude of anisotropy perturbation (fraction)
ani_dir     = 60.0          # fast velocity direction (anti-clockwise from x-axis, in degrees)
n_pert_x    = 4             # number of checkers in x (lon) direction
n_pert_y    = 2             # number of checkers in y (lat) direction
n_pert_z    = 3             # number of checkers in z (dep) direction

size_x      = (lim_x[1] - lim_x[0]) / n_pert_x  # size of each checker in x direction
size_y      = (lim_y[1] - lim_y[0]) / n_pert_y  # size of each checker in y direction
size_z      = (lim_z[1] - lim_z[0]) / n_pert_z  # size of each checker in z direction


ckb = Checker(bg_model_fname, para_fname=par_file)
# n_pert_x, n_pert_y, n_pert_z: number of checkers in x (lon), y (lat), z (dep) directions
# pert_vel: amplitude of velocity perturbation (km/s)
# pert_ani: amplitude of anisotropy perturbation (fraction)
# ani_dir: fast velicty direction (anti-cloclkwise from x-axis, in degrees)
# lim_x, lim_y, lim_z: limits of the checkerboard in x (lon), y (lat), z (dep) directions
ckb.checkerboard(
    n_pert_x=n_pert_x, n_pert_y=n_pert_y, n_pert_z=n_pert_z,         
    pert_vel=pert_vel, pert_ani=pert_ani, ani_dir=ani_dir,   
    lim_x=lim_x, lim_y=lim_y, lim_z=lim_z
)

fname = "%s/linear_velocity_ckb_N%d_%d_%d_PyTomoATT.h5" % (output_path, n_rtp[0], n_rtp[1], n_rtp[2])
ckb.write(fname)
print("generate checkerboard model based on the linear velocity model using PyTomoATT:", fname)

# case 2. ---------- generate a checkerboard model using plain loop (h5py module is required) -------------

# read the background model
bg_model = np.zeros(n_rtp)
with h5py.File(bg_model_fname, 'r') as f:
    bg_model = f['vel'][:]

# set the checkerboard model
vel = np.zeros(n_rtp)
xi  = np.zeros(n_rtp)
eta = np.zeros(n_rtp)

for ir in range(n_rtp[0]):
    for it in range(n_rtp[1]):
        for ip in range(n_rtp[2]):
            depth = am_depths[ir]
            lat = am_latitudes[it]
            lon = am_longitudes[ip]

            # check if the current grid node is within the checkerboard limits
            if (lim_x[0] <= lon <= lim_x[1]) and (lim_y[0] <= lat <= lim_y[1]) and (lim_z[0] <= depth <= lim_z[1]):
                
                sigma_vel = np.sin(np.pi * (lon - lim_x[0])/size_x) * np.sin(np.pi * (lat - lim_y[0])/size_y) * np.sin(np.pi * (depth - lim_z[0])/size_z)
                sigma_ani = np.sin(np.pi * (lon - lim_x[0])/size_x) * np.sin(np.pi * (lat - lim_y[0])/size_y) * np.sin(np.pi * (depth - lim_z[0])/size_z)

                if (sigma_ani > 0):
                    psi = ani_dir / 180.0 * np.pi  # convert degrees to radians
                elif (sigma_ani < 0):
                    psi = (ani_dir + 90.0) / 180.0 * np.pi
                else:
                    psi = 0.0

            else:
                sigma_vel = 0.0
                sigma_ani = 0.0
                psi = 0.0

            # set the velocity and anisotropy
            vel[ir, it, ip] = bg_model[ir, it, ip] * (1.0 + pert_vel * sigma_vel)
            xi[ir, it, ip]  = pert_ani * abs(sigma_ani) * np.cos(2*psi)
            eta[ir, it, ip] = pert_ani * abs(sigma_ani) * np.sin(2*psi)

# write the model to a file
fname = "%s/linear_velocity_ckb_N%d_%d_%d_loop.h5" % (output_path, n_rtp[0], n_rtp[1], n_rtp[2])
with h5py.File(fname, 'w') as f:
    f.create_dataset('vel', data=vel)
    f.create_dataset('xi',  data=xi)
    f.create_dataset('eta', data=eta)
print("generate checkerboard model based on the linear velocity model using plain loop:", fname)


# %% [markdown]
# # eg4. generate flexible checkerboard model
# 
# the checker size increases with depth;
# 
# the checker size is large for anisotropy;

# %%
# only 

# file name of the background model
bg_model_fname = "%s/linear_velocity_N%d_%d_%d_PyTomoATT.h5" % (output_path, n_rtp[0], n_rtp[1], n_rtp[2])

# read the background model
bg_model = np.zeros(n_rtp)
with h5py.File(bg_model_fname, 'r') as f:
    bg_model = f['vel'][:]

# set the checkerboard model
vel = np.zeros(n_rtp)
xi  = np.zeros(n_rtp)
eta = np.zeros(n_rtp)

for ir in range(n_rtp[0]):
    for it in range(n_rtp[1]):
        for ip in range(n_rtp[2]):
            depth = am_depths[ir]
            lat = am_latitudes[it]
            lon = am_longitudes[ip]

            if ((depth >= 0.0) and (depth <= 8.0)):
                size_vel = 0.2
                size_ani = 0.3

                sigma_vel = np.sin(np.pi * lon/size_vel) * np.sin(np.pi * lat/size_vel) * np.sin(np.pi * depth/8.0)
                sigma_ani = np.sin(np.pi * lon/size_ani) * np.sin(np.pi * lat/size_ani) * np.sin(np.pi * depth/8.0)

            elif ((depth > 8.0) and (depth <= 20.0)):

                size_vel = 0.3
                size_ani = 0.4

                sigma_vel = np.sin(np.pi * lon/size_vel) * np.sin(np.pi * lat/size_vel) * np.sin(np.pi * (depth - 8.0)/12.0 + np.pi)
                sigma_ani = np.sin(np.pi * lon/size_ani) * np.sin(np.pi * lat/size_ani) * np.sin(np.pi * (depth - 8.0)/12.0 + np.pi)

            elif ((depth > 20.0) and (depth <= 36.0)):

                size_vel = 0.4
                size_ani = 0.5

                sigma_vel = np.sin(np.pi * lon/size_vel) * np.sin(np.pi * lat/size_vel) * np.sin(np.pi * (depth - 20.0)/16.0 + 2*np.pi)
                sigma_ani = np.sin(np.pi * lon/size_ani) * np.sin(np.pi * lat/size_ani) * np.sin(np.pi * (depth - 20.0)/16.0 + 2*np.pi)

            else:
                sigma_vel = 0.0
                sigma_ani = 0.0

            if (sigma_ani > 0):
                psi = ani_dir / 180.0 * np.pi  # convert degrees to radians
            elif (sigma_ani < 0):
                psi = (ani_dir + 90.0) / 180.0 * np.pi
            else:
                psi = 0.0

            # set the velocity and anisotropy
            vel[ir, it, ip] = bg_model[ir, it, ip] * (1.0 + pert_vel * sigma_vel)
            xi[ir, it, ip]  = pert_ani * abs(sigma_ani) * np.cos(2*psi)
            eta[ir, it, ip] = pert_ani * abs(sigma_ani) * np.sin(2*psi)

# write the model to a file
fname = "%s/linear_velocity_ckb_flex_N%d_%d_%d.h5" % (output_path, n_rtp[0], n_rtp[1], n_rtp[2])
with h5py.File(fname, 'w') as f:
    f.create_dataset('vel', data=vel)
    f.create_dataset('xi',  data=xi)
    f.create_dataset('eta', data=eta)

print("generate flexible checkerboard model based on the linear velocity model using plain loop:", fname)


