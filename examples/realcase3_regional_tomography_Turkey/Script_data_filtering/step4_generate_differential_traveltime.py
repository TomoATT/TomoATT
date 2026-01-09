# %% [markdown]
# This script aims at generating common source differential traveltime and common receiver differential traveltime based on absolute traveltime 

# %%
# load functions for data processing
import sys
sys.path.append('../../utils')
import functions_for_data as ffd

import os

# %%
# read .dat file
fname = "output_data/step3_src_rec.dat"
[ev_info_obs, st_info_obs] = ffd.read_src_rec_file(fname)

# plot data distribution
# ffd.fig_ev_st_distribution_dep(ev_info_obs, st_info_obs)

# %%
# generate common source differential traveltime

dis_thd = 100 # distance between two stations should be less than 100 km
azi_thd = 30  # the angle bwteen two great circle paths from the common source to two separated receivers should be less than 30

ev_info = ffd.generate_cs_dif(ev_info_obs,st_info_obs,dis_thd,azi_thd)

# %%
# output data 
out_path = "output_data"

os.makedirs(out_path,exist_ok=True)

# save data for TomoATT
out_fname = "%s/step4_src_rec_Abs_Cs.dat"%(out_path)
ffd.write_src_rec_file(out_fname,ev_info_obs,st_info_obs)


# %%
# output as the data file for inversion
os.makedirs("../1_src_rec_files",exist_ok=True)
ffd.write_src_rec_file("../1_src_rec_files/src_rec_file.dat",ev_info_obs,st_info_obs)

# %% [markdown]
# # generate common receiver differential traveltime (OPTIONAL)

# %%

# dis_thd = 3  # distance between two earthquakes should be less than 3 km
# azi_thd = 5  # the angle bwteen two great circle paths from the common receiver to two separated sources should be less than 5 degree

# ev_info = ffd.generate_cr_dif(ev_info_obs,st_info_obs,dis_thd,azi_thd)

# %%
# # output data 
# out_path = "output_data"

# os.makedirs(out_path,exist_ok=True)

# # save data for TomoATT
# out_fname = "%s/step4_src_rec_Abs_Cs_Cr.dat"%(out_path)
# ffd.write_src_rec_file(out_fname,ev_info_obs,st_info_obs)


