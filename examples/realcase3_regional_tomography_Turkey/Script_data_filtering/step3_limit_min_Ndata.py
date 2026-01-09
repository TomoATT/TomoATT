# %% [markdown]
# This script is used to delete earthquakes with the number of arrival times less than a specified threshold

# %%
# load functions for data processing
import sys
sys.path.append('../../utils')
import functions_for_data as ffd

import os

# %%
# read .dat file
fname = "output_data/step2_src_rec.dat"
[ev_info_obs, st_info_obs] = ffd.read_src_rec_file(fname)

# plot data distribution
# ffd.fig_ev_st_distribution_dep(ev_info_obs, st_info_obs)

# %%
# remove the earthquakes with less than 5 arrival times.
min_Nt_thd = 5
ev_info_obs = ffd.limit_min_Nt(min_Nt_thd, ev_info_obs)

# plot data distribution
# ffd.fig_ev_st_distribution_dep(ev_info_obs, st_info_obs)

# %%
# output data
out_path = "output_data"

os.makedirs(out_path,exist_ok=True)

# save data for TomoATT
out_fname = "%s/step3_src_rec.dat"%(out_path)
ffd.write_src_rec_file(out_fname,ev_info_obs,st_info_obs)

# # save earthquake list for plotting
# out_fname_ev = "%s/step3_ev_list.dat"%(out_path)
# ffd.write_src_list_file(out_fname_ev,ev_info_obs)

# # save station list for plotting
# out_fname_st = "%s/step3_st_list.dat"%(out_path)
# ffd.write_rec_list_file(out_fname_st,ev_info_obs,st_info_obs)


