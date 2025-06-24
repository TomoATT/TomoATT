# %%
import pygmt
pygmt.config(FONT="16p", IO_SEGMENT_MARKER="<<<")

from pytomoatt.model import ATTModel
from pytomoatt.data import ATTData
import numpy as np


# %%
import os
output_path = "figs"
os.makedirs(output_path, exist_ok=True)


# %% [markdown]
# # eg1. plot constant velocity model

# %%
# ---------------- read model files ----------------
# file names
# init_model_file  = 'models/constant_velocity_N61_51_101_PyTomoATT.h5'        # initial model file
init_model_file  = 'models/constant_velocity_N61_51_101_loop.h5'        # initial model file
par_file    = 'input_params/input_params.yaml'                   # parameter file

# read initial and final model file
att_model = ATTModel.read(init_model_file, par_file)
init_model = att_model.to_xarray()

# interp vel at depth = 20 km
depth = 20.0
vel_init    = init_model.interp_dep(depth, field='vel')    # vel_init[i,:] are (lon, lat, vel)

# ----------------- pygmt plot ------------------

fig = pygmt.Figure()
pygmt.makecpt(cmap="seis", series=[5, 7], background=True, reverse=False)    # colorbar


# ------------ plot horizontal profile of velocity ------------
region = [0, 2, 0, 1]    # region of interest
fig.basemap(region=region, frame=["xa1","ya1","+tVelocity (km/s)"], projection="M10c")   # base map

depth = 20.0
prof_init    = init_model.interp_dep(depth, field='vel')    # prof_init[i,:] are (lon, lat, vel)
lon = prof_init[:,0]    # longitude
lat = prof_init[:,1]    # latitude
vel = prof_init[:,2]    # velocity

grid = pygmt.surface(x=lon, y=lat, z=vel, spacing=0.04,region=region)

fig.grdimage(grid = grid)   # plot figure
fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-1.5)  
fig.colorbar(frame = ["a1","y+lVp (km/s)"], position="+e+w4c/0.3c+h") 
fig.shift_origin(xshift=0, yshift= 1.5)  

# ------------ plot horivertical profile of velocity ------------
fig.shift_origin(xshift=11, yshift= 0)  

region = [0, 40, 0, 1]    # region of interest
fig.basemap(region=region, frame=["xa20+lDepth (km)","ya1+lLatitude","nSwE"], projection="X3c/5c")   # base map

start = [1,0]; end = [1,1]; gap = 1
prof_init    = init_model.interp_sec(start, end, field='vel', val = gap)      # prof_init[i,:] are (lon, lat, dis, dep, vel)
lat = prof_init[:,1]    # lat
dep = prof_init[:,3]    # depth
vel = prof_init[:,4]    # velocity

grid = pygmt.surface(x=dep, y=lat, z=vel, spacing="1/0.04",region=region)

fig.grdimage(grid = grid)   # plot figure

fig.savefig("figs/constant_velocity.png")  # save figure
fig.show()



# %% [markdown]
# # eg2. plot linear velocity model

# %%
# ---------------- read model files ----------------
# file names
# init_model_file  = 'models/linear_velocity_N61_51_101_PyTomoATT.h5'        # initial model file
init_model_file  = 'models/linear_velocity_N61_51_101_loop.h5'        # initial model file
par_file    = 'input_params/input_params.yaml'                   # parameter file

# read initial and final model file
att_model = ATTModel.read(init_model_file, par_file)
init_model = att_model.to_xarray()

# interp vel at depth = 20 km
depth = 20.0
vel_init    = init_model.interp_dep(depth, field='vel')    # vel_init[i,:] are (lon, lat, vel)

# ----------------- pygmt plot ------------------

fig = pygmt.Figure()
pygmt.makecpt(cmap="seis", series=[5, 8], background=True, reverse=False)    # colorbar


# ------------ plot horizontal profile of velocity ------------
region = [0, 2, 0, 1]    # region of interest
fig.basemap(region=region, frame=["xa1","ya1","+tVelocity (km/s)"], projection="M10c")   # base map

depth = 20.0
prof_init    = init_model.interp_dep(depth, field='vel')    # prof_init[i,:] are (lon, lat, vel)
lon = prof_init[:,0]    # longitude
lat = prof_init[:,1]    # latitude
vel = prof_init[:,2]    # velocity

grid = pygmt.surface(x=lon, y=lat, z=vel, spacing=0.04,region=region)

fig.grdimage(grid = grid)   # plot figure
fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-1.5)  
fig.colorbar(frame = ["a1","y+lVp (km/s)"], position="+e+w4c/0.3c+h") 
fig.shift_origin(xshift=0, yshift= 1.5)  

# ------------ plot horivertical profile of velocity ------------
fig.shift_origin(xshift=11, yshift= 0)  

region = [0, 40, 0, 1]    # region of interest
fig.basemap(region=region, frame=["xa20+lDepth (km)","ya1+lLatitude","nSwE"], projection="X3c/5c")   # base map

start = [1,0]; end = [1,1]; gap = 1
prof_init    = init_model.interp_sec(start, end, field='vel', val = gap)      # prof_init[i,:] are (lon, lat, dis, dep, vel)
lat = prof_init[:,1]    # lat
dep = prof_init[:,3]    # depth
vel = prof_init[:,4]    # velocity

grid = pygmt.surface(x=dep, y=lat, z=vel, spacing="1/0.04",region=region)

fig.grdimage(grid = grid)   # plot figure

fig.savefig("figs/linear_velocity.png")  # save figure
fig.show()



# %% [markdown]
# # eg3. plot checkerboard model

# %%
# ---------------- read model files ----------------
# file names
init_model_file  = 'models/linear_velocity_N61_51_101_PyTomoATT.h5'        # initial model file
ckb_model_file   = 'models/linear_velocity_ckb_N61_51_101_PyTomoATT.h5'     # checkerboard model file
# ckb_model_file   = 'models/linear_velocity_ckb_N61_51_101_loop.h5'          # checkerboard model file
par_file    = 'input_params/input_params.yaml'                   # parameter file

# read initial and final model file
att_model = ATTModel.read(init_model_file, par_file)
init_model = att_model.to_xarray()

att_model = ATTModel.read(ckb_model_file, par_file)
ckb_model = att_model.to_xarray()

# interp vel at depth = 20 km
depth = 20.0
vel_init    = init_model.interp_dep(depth, field='vel')    # vel_init[i,:] are (lon, lat, vel)
vel_ckb     = ckb_model.interp_dep(depth, field='vel')    # vel_ckb[i,:] are (lon, lat, vel)

# ----------------- pygmt plot ------------------

fig = pygmt.Figure()
pygmt.makecpt(cmap="../utils/svel13_chen.cpt", series=[-10,10], background=True, reverse=False)    # colorbar


# ------------ plot horizontal profile of velocity ------------
region = [0, 2, 0, 1]    # region of interest
fig.basemap(region=region, frame=["xa1","ya1","+tVelocity perturbation (%)"], projection="M10c")   # base map

# velocity perturbation at depth = 15 km
depth = 15.0
prof_init    = init_model.interp_dep(depth, field='vel')    # prof_init[i,:] are (lon, lat, vel)
prof_ckb     = ckb_model.interp_dep(depth, field='vel')    # prof_ckb[i,:] are (lon, lat, vel)
lon = prof_init[:,0]    # longitude
lat = prof_init[:,1]    # latitude
vel_pert = (prof_ckb[:,2] - prof_init[:,2])/prof_init[:,2] * 100    # velocity perturbation related to initial model

grid = pygmt.surface(x=lon, y=lat, z=vel_pert, spacing=0.01,region=region)

fig.grdimage(grid = grid)   # plot figure
fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

# fast velocity directions (FVDs)
samp_interval = 3   # control the density of anisotropic arrows
width = 0.06        # width of the anisotropic arrow
ani_per_1 = 0.01; ani_per_2 = 0.05; scale = 0.5; basic = 0.1    # control the length of anisotropic arrows related to the amplitude of anisotropy. length = 0.1 + (amplitude - ani_per_1) / (ani_per_2 - ani_per_1) * scale
ani_thd = ani_per_1 # if the amplitude of anisotropy is smaller than ani_thd, no anisotropic arrow will be plotted

phi         = ckb_model.interp_dep(depth, field='phi', samp_interval=samp_interval)          # phi_inv[i,:] are (lon, lat, phi)
epsilon     = ckb_model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)      # epsilon_inv[i,:] are (lon, lat, epsilon)
ani_lon     = phi[:,0].reshape(-1,1)
ani_lat     = phi[:,1].reshape(-1,1)
ani_phi     = phi[:,2].reshape(-1,1)
length      = ((epsilon[:,2] - ani_per_1) / (ani_per_2 - ani_per_1) * scale + basic).reshape(-1,1)
ani_arrow      = np.hstack([ani_lon, ani_lat, ani_phi, length, np.ones((ani_lon.size,1))*width])   # lon, lat, color, angle[-90,90], length, width

# remove arrows with small amplitude of anisotropy
idx = np.where(epsilon[:,2] > ani_thd)[0]    # indices of arrows with large enough amplitude of anisotropy
ani_arrow = ani_arrow[idx,:]    # remove arrows with small amplitude of anisotropy

# plot anisotropic arrows
fig.plot(ani_arrow, style='j', fill='yellow1', pen='0.5p,black')    # plot fast velocity direction


# colorbar
fig.shift_origin(xshift=0, yshift=-1.5)  
fig.colorbar(frame = ["a10","y+ldlnVp (%)"], position="+e+w4c/0.3c+h") 
fig.shift_origin(xshift=0, yshift= 1.5)  

# ------------ plot horivertical profile of velocity ------------
fig.shift_origin(xshift=11, yshift= 0)  

region = [0, 40, 0, 1]    # region of interest
fig.basemap(region=region, frame=["xa20+lDepth (km)","ya1+lLatitude","nSwE"], projection="X3c/5c")   # base map

start = [0.875,0]; end = [0.875,1]; gap = 1
prof_init    = init_model.interp_sec(start, end, field='vel', val = gap)      # prof_init[i,:] are (lon, lat, dis, dep, vel)
prof_ckb     = ckb_model.interp_sec(start, end, field='vel', val = gap)      # prof_ckb[i,:] are (lon, lat, dis, dep, vel)
lat = prof_init[:,1]    # lat
dep = prof_init[:,3]    # depth
vel = (prof_ckb[:,4] - prof_init[:,4])/prof_init[:,4] * 100    # velocity perturbation related to initial model    

grid = pygmt.surface(x=dep, y=lat, z=vel, spacing="1/0.01",region=region)

fig.grdimage(grid = grid)   # plot figure

fig.savefig("figs/checkerboard_velocity.png")  # save figure
fig.show()


# %% [markdown]
# # eg4. plot flexible checkerboard model

# %%
# ---------------- read model files ----------------
# file names
init_model_file  = 'models/linear_velocity_N61_51_101_PyTomoATT.h5'        # initial model file
ckb_model_file   = 'models/linear_velocity_ckb_flex_N61_51_101.h5'     # checkerboard model file
par_file    = 'input_params/input_params.yaml'                   # parameter file

# read initial and final model file
att_model = ATTModel.read(init_model_file, par_file)
init_model = att_model.to_xarray()

att_model = ATTModel.read(ckb_model_file, par_file)
ckb_model = att_model.to_xarray()

# interp vel at depth = 20 km
depth = 20.0
vel_init    = init_model.interp_dep(depth, field='vel')    # vel_init[i,:] are (lon, lat, vel)
vel_ckb     = ckb_model.interp_dep(depth, field='vel')    # vel_ckb[i,:] are (lon, lat, vel)

# ----------------- pygmt plot ------------------

fig = pygmt.Figure()
pygmt.makecpt(cmap="../utils/svel13_chen.cpt", series=[-10,10], background=True, reverse=False)    # colorbar


for depth in [4,14,28]:
    # ------------ plot horizontal profile of velocity ------------
    region = [0, 2, 0, 1]    # region of interest
    fig.basemap(region=region, frame=["xa1","ya1","NsEw"], projection="M10c")   # base map

    # velocity perturbation at depth = 15 km
    prof_init    = init_model.interp_dep(depth, field='vel')    # prof_init[i,:] are (lon, lat, vel)
    prof_ckb     = ckb_model.interp_dep(depth, field='vel')    # prof_ckb[i,:] are (lon, lat, vel)
    lon = prof_init[:,0]    # longitude
    lat = prof_init[:,1]    # latitude
    vel_pert = (prof_ckb[:,2] - prof_init[:,2])/prof_init[:,2] * 100    # velocity perturbation related to initial model

    grid = pygmt.surface(x=lon, y=lat, z=vel_pert, spacing=0.01,region=region)

    fig.grdimage(grid = grid)   # plot figure
    
    # fast velocity directions (FVDs)
    samp_interval = 3   # control the density of anisotropic arrows
    width = 0.06        # width of the anisotropic arrow
    ani_per_1 = 0.01; ani_per_2 = 0.05; scale = 0.5; basic = 0.1    # control the length of anisotropic arrows related to the amplitude of anisotropy. length = 0.1 + (amplitude - ani_per_1) / (ani_per_2 - ani_per_1) * scale
    ani_thd = ani_per_1 # if the amplitude of anisotropy is smaller than ani_thd, no anisotropic arrow will be plotted

    phi         = ckb_model.interp_dep(depth, field='phi', samp_interval=samp_interval)          # phi_inv[i,:] are (lon, lat, phi)
    epsilon     = ckb_model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)      # epsilon_inv[i,:] are (lon, lat, epsilon)
    ani_lon     = phi[:,0].reshape(-1,1)
    ani_lat     = phi[:,1].reshape(-1,1)
    ani_phi     = phi[:,2].reshape(-1,1)
    length      = ((epsilon[:,2] - ani_per_1) / (ani_per_2 - ani_per_1) * scale + basic).reshape(-1,1)
    ani_arrow      = np.hstack([ani_lon, ani_lat, ani_phi, length, np.ones((ani_lon.size,1))*width])   # lon, lat, color, angle[-90,90], length, width

    # remove arrows with small amplitude of anisotropy
    idx = np.where(epsilon[:,2] > ani_thd)[0]    # indices of arrows with large enough amplitude of anisotropy
    ani_arrow = ani_arrow[idx,:]    # remove arrows with small amplitude of anisotropy

    # plot anisotropic arrows
    fig.plot(ani_arrow, style='j', fill='yellow1', pen='0.5p,black')    # plot fast velocity direction

    fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

    # plot vertical profile
    fig.plot(x=[0.9, 0.9], y=[0, 1], pen="2p,black,-")    # vertical line

    fig.shift_origin(xshift=0, yshift=-6)  

# colorbar
fig.shift_origin(xshift=0, yshift= 4.5)  
fig.colorbar(frame = ["a10","y+ldlnVp (%)"], position="+e+w4c/0.3c+h") 
fig.shift_origin(xshift=0, yshift= 1.5)  

# ------------ plot horivertical profile of velocity ------------
fig.shift_origin(xshift=11, yshift= 0)  

region = [0, 40, 0, 1]    # region of interest
fig.basemap(region=region, frame=["xa20+lDepth (km)","ya1+lLatitude","nSwE"], projection="X3c/5c")   # base map

start = [0.9,0]; end = [0.9,1]; gap = 1
prof_init    = init_model.interp_sec(start, end, field='vel', val = gap)      # prof_init[i,:] are (lon, lat, dis, dep, vel)
prof_ckb     = ckb_model.interp_sec(start, end, field='vel', val = gap)      # prof_ckb[i,:] are (lon, lat, dis, dep, vel)
lat = prof_init[:,1]    # lat
dep = prof_init[:,3]    # depth
vel = (prof_ckb[:,4] - prof_init[:,4])/prof_init[:,4] * 100    # velocity perturbation related to initial model    

grid = pygmt.surface(x=dep, y=lat, z=vel, spacing="1/0.01",region=region)

fig.grdimage(grid = grid)   # plot figure

fig.savefig("figs/flexible_checkerboard_velocity.png")  # save figure
fig.show()



