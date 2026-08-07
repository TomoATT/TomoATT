# %% [markdown]
# This script aims at retaining the earthquakes and stations within the study region of a oblique box

# %%
# load functions for data processing
import sys
sys.path.append('../../utils')
import functions_for_data as ffd

import os

# %%
# read .dat file
fname = "input_data/src_rec_Turkey.dat"
[ev_info_obs, st_info_obs] = ffd.read_src_rec_file(fname)

# plot data distribution
# ffd.fig_ev_st_distribution_dep(ev_info_obs, st_info_obs)


# %%
# we only retain the earthquakes and stations within the target study region: 
# center of the region: 37N,37E  x-axis: [-1.3,1.3] (degree), y-axis [-2,3] degree, z-axis: 0.5-40 km.

central_lat = 37.0
central_lon = 37.0
rotation_angle = 45.0

# step 1: rotate the source and receiver locations to a local coordinate system. The center of the region is (0,0)
[ev_info_obs, st_info_obs] = ffd.rotate_src_rec(ev_info_obs,st_info_obs,central_lat,central_lon,rotation_angle)

# plot data distribution
# ffd.fig_ev_st_distribution_dep(ev_info_obs, st_info_obs)

# %%
# step 2, we only retain the earthquakes and stations within the study region:  x-axis: [-1.3,1.3] (degree), y-axis [-2,3] degree, z-axis: 0.5-40 km.

lat1 = -2.0;    lat2 = 3.0; 
lon1 = -1.3;    lon2 = 1.3; 
dep1 =  0.5;    dep2 = 40.0; 

# limit earthquake region
ev_info_obs = ffd.limit_ev_region(ev_info_obs,lat1,lat2,lon1,lon2,dep1,dep2)

# limit station region
ev_info_obs = ffd.limit_st_region(ev_info_obs,st_info_obs,lat1,lat2,lon1,lon2)

# plot data distribution
# ffd.fig_ev_st_distribution_dep(ev_info_obs, st_info_obs)

# %%
# output data
out_path = "output_data"
os.makedirs(out_path,exist_ok=True)

# save src_rec_file for TomoATT
out_fname = "%s/step1_src_rec.dat"%(out_path)
ffd.write_src_rec_file(out_fname,ev_info_obs,st_info_obs)


# %%
# We can rotate the source and receiver locations back to the original coordinate system for plotting or other purposes.
[ev_info_obs, st_info_obs] = ffd.rotate_src_rec_reverse(ev_info_obs, st_info_obs,central_lat,central_lon,rotation_angle)

# plot data distribution
# ffd.fig_ev_st_distribution_dep(ev_info_obs, st_info_obs)

# save earthquake list for plotting
out_fname_ev = "%s/step1_ev_list.dat"%(out_path)
ffd.write_src_list_file(out_fname_ev,ev_info_obs)

# save station list for plotting
out_fname_st = "%s/step1_st_list.dat"%(out_path)
ffd.write_rec_list_file(out_fname_st,ev_info_obs,st_info_obs)


