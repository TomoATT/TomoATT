# %%
import pygmt
pygmt.config(FONT="16p", IO_SEGMENT_MARKER="<<<")

from pytomoatt.model import ATTModel
from pytomoatt.data import ATTData
import numpy as np

# %%
# plot sensitivity kernel at the XX iteration
# 1. set "output_kernel" to be "True" in the input_params.yaml file

# ---------------- read files ----------------
Nstep = "0007"
kernel_list = {}    # dictionary to store all the kernels

# file names
data_file   = 'OUTPUT_FILES/out_data_sim_group_0.h5'        # data file
par_file    = 'input_files/input_params.yaml'               # parameter file
grid_file   = 'OUTPUT_FILES/out_data_grid.h5'               # grid file
group       = 'model'               

# (Option 1) read original sensitivity kernel 
# Ks: kernel w.r.t. slowness at the 7-th iteration
dataset     = 'Ks_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# Kxi: kernel w.r.t. xi (anisotropic parameter) at the 7-th iteration
dataset     = 'Kxi_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# Keta: kernel w.r.t. eta (anisotropic parameter) at the 7-th iteration
dataset     = 'Keta_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# %%
# (Option 2) read kernel_density
# Ks_den: kernel density w.r.t. slowness at the 7-th iteration
dataset     = 'Ks_density_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# Kxi_den: kernel density w.r.t. xi (anisotropic parameter) at the 7-th iteration
dataset     = 'Kxi_density_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# Keta_den: kernel density w.r.t. eta (anisotropic parameter) at the 7-th iteration
dataset     = 'Keta_density_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# %%
# kernel density normalization is performed after smoothing. So tag 'Ks_over_Kden_inv_%s'%(Nstep) is not available for version > 1.0.3
# # (Option 3) read normalized kernel, K/(k_den)^\zeta
# dataset     = 'Ks_over_Kden_inv_%s'%(Nstep)                      
# data = ATTData.read(data_file, par_file, grid_file, group, dataset)
# kernel_list[dataset] = data.to_xarray()

# # Kxi_norm: normalized kernel w.r.t. xi (anisotropic parameter) at the 7-th iteration
# dataset     = 'Kxi_over_Kden_inv_%s'%(Nstep)                      
# data = ATTData.read(data_file, par_file, grid_file, group, dataset)
# kernel_list[dataset] = data.to_xarray()

# # Keta_norm: normalized kernel w.r.t. eta (anisotropic parameter) at the 7-th iteration
# dataset     = 'Keta_over_Kden_inv_%s'%(Nstep)                      
# data = ATTData.read(data_file, par_file, grid_file, group, dataset)
# kernel_list[dataset] = data.to_xarray()

# %%
# this part works for version > 1.0.3
# # (Option 3) read kernel density smoothed by multigrid parameterization, 
# dataset     = 'Ks_density_update_inv_%s'%(Nstep)                      
# data = ATTData.read(data_file, par_file, grid_file, group, dataset)
# kernel_list[dataset] = data.to_xarray()

# # Kxi_norm: normalized kernel w.r.t. xi (anisotropic parameter) at the 7-th iteration
# dataset     = 'Kxi_density_update_inv_%s'%(Nstep)                      
# data = ATTData.read(data_file, par_file, grid_file, group, dataset)
# kernel_list[dataset] = data.to_xarray()

# # Keta_norm: normalized kernel w.r.t. eta (anisotropic parameter) at the 7-th iteration
# dataset     = 'Keta_density_update_inv_%s'%(Nstep)                      
# data = ATTData.read(data_file, par_file, grid_file, group, dataset)
# kernel_list[dataset] = data.to_xarray()

# %%
# (Option 4) read normalized kernel smoothed by multigrid parameterization
# Ks_update: smoothed normalized kernel w.r.t. slowness at the 7-th iteration
dataset     = 'Ks_update_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# Kxi_update: smoothed normalized kernel w.r.t. xi (anisotropic parameter) at the 7-th iteration
dataset     = 'Kxi_update_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# Keta_update: smoothed normalized kernel w.r.t. eta (anisotropic parameter) at the 7-th iteration
dataset     = 'Keta_update_inv_%s'%(Nstep)                      
data = ATTData.read(data_file, par_file, grid_file, group, dataset)
kernel_list[dataset] = data.to_xarray()

# %%
import os
try: 
    os.mkdir('img')
except:
    pass

# %%
# ---------------- access 3D array ----------------
# we can access 3D dataset:

dep_1d_array = kernel_list['Ks_inv_0007']["dep"]
lat_1d_array = kernel_list['Ks_inv_0007']["lat"]
lon_1d_array = kernel_list['Ks_inv_0007']["lon"]

print("3D array of coordinates. \n dep: ", dep_1d_array.shape, " \n lat: ", lat_1d_array.shape, " \n lon: ", lon_1d_array.shape)

array = kernel_list['Ks_inv_0007']["Ks_inv_0007"]

print("3D array of kernel. \n Ks: ", array.shape)

# %%
# ---------------- 2D depth profile of kernels ----------------

for dataset in kernel_list:

    # interp vel at depth = 20 km
    depth = 20.0
    kernel = kernel_list[dataset].interp_dep(depth, field=dataset)    # kernel[i,:] are (lon, lat, kernel)

    print("kernel at depth = ", depth, " km. kernel:", kernel.shape, ", (lon, lat, kernel)")

    # plot
    fig = pygmt.Figure()
    fig.basemap(region=[0,2,0,2], frame=["xa1","ya1","+t%s"%(dataset)], projection="M10c")   # base map
    pygmt.makecpt(cmap="jet", series=[-0.5, 0.5], background=True, reverse=True)    # colorbar

    x = kernel[:,0];      # longitude
    y = kernel[:,1];      # latitude
    value = kernel[:,2]/np.nanmax(np.abs(kernel[:,2]))   # traveltime
    grid = pygmt.surface(x=x, y=y, z=value, spacing=0.04,region=[0,2,0,2])

    fig.grdimage(grid = grid)   # plot figure
    fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

    # colorbar
    fig.shift_origin(xshift=0, yshift=-1.5)  
    fig.colorbar(frame = ["a0.5","y+l%s"%(dataset)], position="+e+w4c/0.3c+h") 

    fig.savefig("img/3_dep_%s.png"%(dataset))

# %%
# ---------------- 2D vertical profile of kernels ----------------

for dataset in kernel_list:

    # interp from [0,0.6] in lon-lat to [2,0.6] in lon-lat, gap = 1 km
    start = [0,0.6]; end = [2,0.6]; gap = 1
    kernel_sec    = kernel_list[dataset].interp_sec(start, end, field=dataset, val = gap)      # kernel_sec[i,:] are (lon, lat, dis, dep, kernel)

    print("kernel_sec:", kernel_sec.shape, ", (lon, lat, distance, depth, kernel)")

    # plot
    fig = pygmt.Figure()
    fig.basemap(region=[0,2,0,40], frame=["xa1+lLongitude","ya20+lDepth (km)","+t%s"%(dataset)], projection="X10c/-2c")   # base map
    pygmt.makecpt(cmap="jet", series=[-0.5, 0.5], background=True, reverse=True)    # colorbar

    x = kernel_sec[:,0];  # longitude
    y = kernel_sec[:,3];  # depth
    value = kernel_sec[:,4]/np.nanmax(np.abs(kernel_sec[:,4])) # traveltime
    grid = pygmt.surface(x=x, y=y, z=value, spacing="0.04/1",region=[0,2,0,40])

    fig.grdimage(grid = grid)   # plot figure
    fig.text(text="A", x = 0.1 , y = 5, font = "14p,Helvetica-Bold,black", fill = "white") 
    fig.text(text="A@+'@+", x = 1.9 , y = 5, font = "14p,Helvetica-Bold,black", fill = "white") 

    # colorbar
    fig.shift_origin(xshift=0, yshift=-2)  
    fig.colorbar(frame = ["a0.5","y+l%s"%(dataset)], position="+e+w4c/0.3c+h") 

    fig.savefig("img/3_sec_%s.png"%(dataset))


