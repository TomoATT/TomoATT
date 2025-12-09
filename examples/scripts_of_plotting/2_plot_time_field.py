# %%
import pygmt
pygmt.config(FONT="16p", IO_SEGMENT_MARKER="<<<")

from pytomoatt.model import ATTModel
from pytomoatt.data import ATTData
import numpy as np

# %%
# plot the traveltime field and adjoint field of the source at the XX iteration
# 1. set "output_source_field" to be "True" in the input_params.yaml file

# Because source parallelizaion is used, the source field is only stored in one out_data_sim_group_XX.h5 file.
# For example,  if we use 8 processors for source parallelization, we have
#               src_JC00 is stored in out_data_sim_group_0.h5 file.
#               src_JC05 is stored in out_data_sim_group_5.h5 file.
#               src_JC07 is stored in out_data_sim_group_7.h5 file.
#               src_JC08 is stored in out_data_sim_group_0.h5 file.
#               src_JC09 is stored in out_data_sim_group_1.h5 file.
#               src_JC10 is stored in out_data_sim_group_2.h5 file.
#               ...
#               src_JC24 is stored in out_data_sim_group_0.h5 file.

# ---------------- read files ----------------
src_name = 'JC05'
Nstep = "0007"

# file names
data_file   = 'OUTPUT_FILES/out_data_sim_group_5.h5'        # data file
par_file    = 'input_files/input_params.yaml'               # parameter file
grid_file   = 'OUTPUT_FILES/out_data_grid.h5'               # grid file
group       = 'src_%s'%(src_name)                           # src_${src_name}     

# read traveltime field
dataset_time     = 'time_field_inv_%s'%(Nstep)                   # time_field_inv_${Nstep}
data = ATTData.read(data_file, par_file, grid_file, group, dataset_time)
time_field = data.to_xarray()

# read adjoint field
dataset_adjoint  = 'adjoint_field_inv_%s'%(Nstep)                # adjoint_field_inv_${Nstep}
data = ATTData.read(data_file, par_file, grid_file, group, dataset_adjoint)
adjoint_field = data.to_xarray()


# %%
import os
try: 
    os.mkdir('img')
except:
    pass

# %%
# ---------------- access 3D time field and adjoint field ----------------
# we can access 3D dataset:

dep_1d_array = time_field["dep"]
lat_1d_array = time_field["lat"]
lon_1d_array = time_field["lon"]

print("3D array of coordinates. \n dep: ", dep_1d_array.shape, " \n lat: ", lat_1d_array.shape, " \n lon: ", lon_1d_array.shape)

time_3d_array = time_field[dataset_time]
adjoint_3d_array = adjoint_field[dataset_adjoint]

print("3D array of fields. \n time: ", time_3d_array.shape, " \n adjoint: ", adjoint_3d_array.shape)

# %%
# ---------------- 2D depth profile of time field ----------------

# interp vel at depth = 20 km
depth = 20.0
time    = time_field.interp_dep(depth, field=dataset_time)    # time[i,:] are (lon, lat, vel)

print("time at depth = ", depth, " km. time:", time.shape, ", (lon, lat, time)")

# plot
fig = pygmt.Figure()
fig.basemap(region=[0,2,0,2], frame=["xa1","ya1","+tTraveltime"], projection="M10c")   # base map
pygmt.makecpt(cmap="jet", series=[0, 30], background=True, reverse=True)    # colorbar

x = time[:,0];      # longitude
y = time[:,1];      # latitude
value = time[:,2]   # traveltime
grid = pygmt.surface(x=x, y=y, z=value, spacing=0.04,region=[0,2,0,2])

fig.grdimage(grid = grid)   # plot figure
fig.contour(x=x, y=y, z=value, levels=5, pen="1.5p,white")   # contour
fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-1.5)  
fig.colorbar(frame = ["a20","y+lTraveltime (s)"], position="+e+w4c/0.3c+h") 

fig.savefig("img/2_dep_time.png")

# %%
# ---------------- 2D depth profile of adjoint field ----------------

# interp vel at depth = 20 km
depth = 20.0
adjoint = adjoint_field.interp_dep(depth, field=dataset_adjoint)    

print("time at depth = ", depth, " km. time:", time.shape, ", (lon, lat, time)")

# plot
fig = pygmt.Figure()
fig.basemap(region=[0,2,0,2], frame=["xa1","ya1","+tAdjoint field"], projection="M10c")   # base map
pygmt.makecpt(cmap="jet", series=[-0.5, 0.5], background=True, reverse=False)    # colorbar

x = time[:,0];      # longitude
y = time[:,1];      # latitude
value = adjoint[:,2]   # traveltime
value = value/np.nanmax(np.abs(value))
grid = pygmt.surface(x=x, y=y, z=value, spacing=0.04,region=[0,2,0,2])

fig.grdimage(grid = grid)   # plot figure
fig.contour(x=x, y=y, z=time[:,2], levels=5, pen="1.5p,white")   # contour
fig.text(text="%d km"%(depth), x = 0.2 , y = 0.1, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-1.5)  
fig.colorbar(frame = ["a0.5","y+lAdjoint field"], position="+e+w4c/0.3c+h") 

fig.savefig("img/2_dep_adjoint.png")

# %%
# ---------------- 2D vertical profile of traveltime field ----------------

# interp from [0,0.6] in lon-lat to [2,0.6] in lon-lat, gap = 1 km
start = [0,0.6]; end = [2,0.6]; gap = 1
time_sec    = time_field.interp_sec(start, end, field=dataset_time, val = gap)      # time_sec[i,:] are (lon, lat, dis, dep, time)

print("time_sec:", time_sec.shape, ", (lon, lat, distance, depth, time)")

# plot
fig = pygmt.Figure()
fig.basemap(region=[0,2,0,40], frame=["xa1+lLongitude","ya20+lDepth (km)","+tTraveltime"], projection="X10c/-2c")   # base map
pygmt.makecpt(cmap="jet", series=[0, 30], background=True, reverse=True)    # colorbar

x = time_sec[:,0];  # longitude
y = time_sec[:,3];  # depth
value = time_sec[:,4] # traveltime
grid = pygmt.surface(x=x, y=y, z=value, spacing="0.04/1",region=[0,2,0,40])

fig.grdimage(grid = grid)   # plot figure
fig.contour(x=x, y=y, z=value, levels=5, pen="1.5p,white")   # contour
fig.text(text="A", x = 0.1 , y = 5, font = "14p,Helvetica-Bold,black", fill = "white") 
fig.text(text="A@+'@+", x = 1.9 , y = 5, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-2)  
fig.colorbar(frame = ["a20","y+lTraveltime (s)"], position="+e+w4c/0.3c+h") 

fig.savefig("img/2_sec_time.png")

# %%
# ---------------- 2D vertical profile of adjoint field ----------------

# interp from [0,0.6] in lon-lat to [2,0.6] in lon-lat, gap = 1 km
start = [0,0.6]; end = [2,0.6]; gap = 1
adjoint_sec   = adjoint_field.interp_sec(start, end, field=dataset_adjoint, val = gap)       

print("adjoint_sec:", time_sec.shape, ", (lon, lat, distance, depth, adjoint)")

# plot
fig = pygmt.Figure()
fig.basemap(region=[0,2,0,40], frame=["xa1+lLongitude","ya20+lDepth (km)","+tAdjoint field"], projection="X10c/-2c")   # base map
pygmt.makecpt(cmap="jet", series=[-0.5, 0.5], background=True, reverse=False)    # colorbar

x = adjoint_sec[:,0];  # longitude
y = adjoint_sec[:,3];  # depth
value = adjoint_sec[:,4] # traveltime
value = value/np.nanmax(np.abs(value))
grid = pygmt.surface(x=x, y=y, z=value, spacing="0.04/1",region=[0,2,0,40])

fig.grdimage(grid = grid)   # plot figure
fig.contour(x=x, y=y, z=time_sec[:,4], levels=5, pen="1.5p,white")   # contour
fig.text(text="A", x = 0.1 , y = 5, font = "14p,Helvetica-Bold,black", fill = "white") 
fig.text(text="A@+'@+", x = 1.9 , y = 5, font = "14p,Helvetica-Bold,black", fill = "white") 

# colorbar
fig.shift_origin(xshift=0, yshift=-2)  
fig.colorbar(frame = ["a0.5","y+lAdjoint"], position="+e+w4c/0.3c+h") 

fig.savefig("img/2_sec_adjoint.png")


