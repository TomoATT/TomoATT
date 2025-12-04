# %%
import pygmt
pygmt.config(FONT="16p", IO_SEGMENT_MARKER="<<<")

from pytomoatt.model import ATTModel
from pytomoatt.data import ATTData
import numpy as np

# %%
# (Option 1) plot the final model after inversion

# Need to set "output_final_model" as "True" in the input_params.yaml file

# ---------------- read model files ----------------
# file names
init_model_file  = 'input_files/model_init_N61_61_61.h5'        # initial model file
inv_model_file  = 'OUTPUT_FILES/final_model.h5'               # final model file
par_file    = 'input_files/input_params.yaml'                   # parameter file

# read initial and final model file
model = ATTModel.read(init_model_file, par_file)
init_model = model.to_xarray()

model = ATTModel.read(inv_model_file, par_file)
inv_model = model.to_xarray()


# %%
# # (Option 2) plot the model at the XX iteration

# # Need to set "output_middle_model" as "True" in the input_params.yaml file

# # ---------------- read model files ----------------
# # file names
# init_model_file  = 'input_files/model_init_N61_61_61.h5'        # initial model file
# inv_model_file  = 'OUTPUT_FILES/middle_model_step_0007.h5'               # final model file
# par_file    = 'input_files/input_params.yaml'                   # parameter file

# # read initial and final model file
# model = ATTModel.read(init_model_file, par_file)
# init_model = model.to_xarray()

# model = ATTModel.read(inv_model_file, par_file)
# inv_model = model.to_xarray()

# %%
import os
try: 
    os.mkdir('img')
except:
    pass

# %%
# ---------------- access 3D model parameters ----------------

# we can access 3D dataset with keys:
# 1. "vel" for velocity
# 2. "phi" fast velocity direction, anti-clock angle w.r.t the east direction. (only available for anisotropic model) 
# 3. "epsilon" for anisotropic magnitude (only available for anisotropic model)
# 4. "xi" and "eta" for anisotropic parameters: xi = epsilon * cos(phi), eta = epsilon * sin(phi)
vel_3d_array = inv_model["vel"]
phi_3d_array = inv_model["phi"]
epsilon_3d_array = inv_model["epsilon"]
xi_3d_array = inv_model["xi"]
eta_3d_array = inv_model["eta"]

print("3D array of model parameters. \n vel: ", vel_3d_array.shape, " \n phi: ", phi_3d_array.shape, 
      " \n epsilon: ", epsilon_3d_array.shape, " \n xi: ", xi_3d_array.shape, " \n eta: ", eta_3d_array.shape)

# %%
# ---------------- 2D depth profile of velocity perturbation ----------------

# interp vel at depth = 20 km
depth = 20.0
vel_init    = init_model.interp_dep(depth, field='vel')    # vel_init[i,:] are (lon, lat, vel)
vel_inv   = inv_model.interp_dep(depth, field='vel')    

print("vel_depth at depth = ", depth, " km. vel_depth:", vel_init.shape, ", (lon, lat, vel)")

# plot
fig = pygmt.Figure()
fig.basemap(region=[0,2,0,2], frame=["xa1","ya1","+tVelocity perturbation"], projection="M10c")   # base map
pygmt.makecpt(cmap="../utils/svel13_chen.cpt", series=[-20, 20], background=True, reverse=False)    # colorbar

x = vel_init[:,0];  # longitude
y = vel_init[:,1];  # latitude
value = (vel_inv[:,2] - vel_init[:,2])/vel_init[:,2] * 100    # velocity perturbation relative to the initial model
grid = pygmt.surface(x=x, y=y, z=value, spacing=0.04,region=[0,2,0,2])

fig.grdimage(grid = grid)   # plot figure
fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-1.5)  
fig.colorbar(frame = ["a20","y+ldlnVp (%)"], position="+e+w4c/0.3c+h") 

fig.savefig("img/1_dep_vel.png")

# %%
# ---------------- 2D depth profile of azimuthal anisotropy ----------------

# interp magnitude of anisotropy at depth = 20 km
depth = 20.0
epsilon_inv   = inv_model.interp_dep(depth, field='epsilon')    # epsilon_inv[i,:] are (lon, lat, epsilon)

print("epsilon_inv at depth = ", depth, " km. epsilon_inv:", epsilon_inv.shape, ", (lon, lat, epsilon)")

# generate fast velocity direction (anisotropic arrow)
samp_interval = 3
length = 10
width = 0.1
ani_thd = 0.02
ani_phi = inv_model.interp_dep(depth, field='phi', samp_interval=samp_interval)
ani_epsilon = inv_model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)
ani_arrow = np.hstack([ani_phi, ani_epsilon[:,2].reshape(-1, 1)*length, np.ones((ani_epsilon.shape[0],1))*width])   # lon, lat, angle, length, width
idx = np.where(ani_epsilon[:,2] > ani_thd)
ani_arrow = ani_arrow[idx[0],:]

print("ani_arrow at depth = ", depth, " km. ani_arrow:", ani_arrow.shape, ", (lon, lat, angle, length, width)")


# plot
fig = pygmt.Figure()
fig.basemap(region=[0,2,0,2], frame=["xa1","ya1","+tAzimuthal Anisotropy"], projection="M10c")   # base map
pygmt.makecpt(cmap="cool", series=[0, 0.1], background=True, reverse=False)    # colorbar

x = epsilon_inv[:,0];  # longitude
y = epsilon_inv[:,1];  # latitude
value = epsilon_inv[:,2] # magnitude of anisotropy
grid = pygmt.surface(x=x, y=y, z=value, spacing=0.04,region=[0,2,0,2])

fig.grdimage(grid = grid)   # plot magnitude of anisotropy
fig.plot(ani_arrow, style='j', fill='yellow1', pen='0.5p,black')    # plot fast velocity direction

fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-1.5)  
fig.colorbar(frame = ["a0.1","y+lAnisotropy"], position="+e+w4c/0.3c+h") 

fig.savefig("img/1_dep_ani.png")

# %%
# ---------------- 2D vertical profile of velocity perturbation ----------------

# interp vel from [0,0.75] in lon-lat to [2,0.75] in lon-lat, gap = 1 km
start = [0,0.75]; end = [2,0.75]; gap = 1
vel_init_sec    = init_model.interp_sec(start, end, field='vel', val = gap)      # vel_init_sec[i,:] are (lon, lat, dis, dep, vel)
vel_inv_sec   = inv_model.interp_sec(start, end, field='vel', val = gap)       

print("vel_init_sec:", vel_init_sec.shape, ", (lon, lat, distance, depth, vel)")

# plot
fig = pygmt.Figure()
fig.basemap(region=[0,2,0,40], frame=["xa1+lLongitude","ya20+lDepth (km)","+tVelocity perturbation"], projection="X10c/-4c")   # base map
pygmt.makecpt(cmap="../utils/svel13_chen.cpt", series=[-20, 20], background=True, reverse=False)    # colorbar

x = vel_init_sec[:,0];  # longitude
y = vel_init_sec[:,3];  # depth
value = (vel_inv_sec[:,4] - vel_init_sec[:,4])/vel_init_sec[:,4] * 100    # velocity perturbation relative to the initial model
grid = pygmt.surface(x=x, y=y, z=value, spacing="0.04/1",region=[0,2,0,40])

fig.grdimage(grid = grid)   # plot figure
fig.text(text="A", x = 0.1 , y = 5, font = "14p,Helvetica-Bold,black", fill = "white") 
fig.text(text="A@+'@+", x = 1.9 , y = 5, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-2)  
fig.colorbar(frame = ["a20","y+ldlnVp (%)"], position="+e+w4c/0.3c+h") 

fig.savefig("img/1_sec_vel.png")


