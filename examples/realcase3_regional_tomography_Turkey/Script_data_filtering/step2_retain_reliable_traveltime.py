# %% [markdown]
# This script uses the linear regression method to select reliable traveltime data

# %%
# load functions for data processing
import sys
sys.path.append('../../utils')
import functions_for_data as ffd

import os

# %%
# read .dat file
fname = "output_data/step1_src_rec.dat"
[ev_info_obs, st_info_obs] = ffd.read_src_rec_file(fname)

# plot data distribution
# ffd.fig_ev_st_distribution_dep(ev_info_obs, st_info_obs)


# %%
# traveltime visualization (traveltime versus hypocentral distance) and discard outliers

# selection parameters
# retain data satisfying:     slope * dis + intercept + down < time < slope * dis + intercept + up
slope = 0.16
intercept = 0
up = 10
down = -10

# range of distance from source to receiver
dis_min = 0 
dis_max = 500

ev_info_obs = ffd.fig_data_plot_remove_outliers(ev_info_obs,st_info_obs,slope,intercept,up,down,dis_min,dis_max)

# %%
# plot distance-time scatter of given phases and do linear regression

# given phases and colors for plotting
phase_list = ["Pn","P","Pg","Pb"]
color_list = ["k","b","r","g"]


# range of distance from source to receiver
dis_min = 0 
dis_max = 500

# plot
ffd.fig_data_plot_phase(ev_info_obs,st_info_obs,phase_list,color_list,dis_min,dis_max)

# %%
# discard data with distance > 100 km

epi_dis1 = 100      # threshold of epicentral distance = 100 km
epi_dis2 = 1000000
ev_info_obs = ffd.limit_epi_dis(ev_info_obs, st_info_obs, epi_dis1, epi_dis2)


# given phases and colors for plotting
phase_list = ["Pn","P","Pg","Pb"]
color_list = ["k","b","r","g"]

# range of distance from source to receiver
dis_min = 0 
dis_max = 110

# plot
ffd.fig_data_plot_phase(ev_info_obs,st_info_obs,phase_list,color_list,dis_min,dis_max)

# %%
# retain given phases (P, Pg, Pb)

phase_list = ["P","Pg","Pb"]
ev_info_obs = ffd.limit_data_phase(ev_info_obs,phase_list)

# given phases and colors for plotting
color_list = ["b","r","g"]

# range of distance from source to receiver
dis_min = 0 
dis_max = 110

# plot
ffd.fig_data_plot_phase(ev_info_obs,st_info_obs,phase_list,color_list,dis_min,dis_max)

# %%
# Do linear regression of all data and retain data with residual < 3*SEE

[dis_obs,time_obs] = ffd.data_dis_time(ev_info_obs,st_info_obs)
(slope,intercept,SEE) = ffd.linear_regression(dis_obs,time_obs)
up      =  3*SEE
down    = -3*SEE

# range of distance from source to receiver
dis_min = 0 
dis_max = 110

ev_info_obs = ffd.fig_data_plot_remove_outliers(ev_info_obs,st_info_obs,slope,intercept,up,down,dis_min,dis_max)
[dis_obs,time_obs] = ffd.data_dis_time(ev_info_obs,st_info_obs)
(slope_2,intercept_2,SEE_2) = ffd.linear_regression(dis_obs,time_obs)

print("The (slope,intercept,SEE) of original data is (%6.3f,%6.3f,%6.3f)"%(slope,intercept,SEE))
print("The (slope,intercept,SEE) of filtered data is (%6.3f,%6.3f,%6.3f)"%(slope_2,intercept_2,SEE_2))

# %%
# output data
out_path = "output_data"
os.makedirs(out_path,exist_ok=True)

# save data for TomoATT
out_fname = "%s/step2_src_rec.dat"%(out_path)
ffd.write_src_rec_file(out_fname,ev_info_obs,st_info_obs)

# # save earthquake list for plotting
# out_fname_ev = "%s/step2_ev_list.dat"%(out_path)
# ffd.write_src_list_file(out_fname_ev,ev_info_obs)

# # save station list for plotting
# out_fname_st = "%s/step2_st_list.dat"%(out_path)
# ffd.write_rec_list_file(out_fname_st,ev_info_obs,st_info_obs)


