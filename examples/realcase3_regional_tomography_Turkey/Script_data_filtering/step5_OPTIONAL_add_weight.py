# %% [markdown]
# This script provides several methods to add weights to earthquake, station and traveltime data

# %%
# load functions for data processing
import sys
sys.path.append('../../utils')
import functions_for_data as ffd

import os

# %%
# read .dat file
fname = "output_data/step4_src_rec_Abs_Cs.dat"
[ev_info_obs, st_info_obs] = ffd.read_src_rec_file(fname)

# plot weight distribution
ffd.fig_ev_st_distribution_wt(ev_info_obs, st_info_obs)


# %%
# Option 1 (recommended) , assign box weighting to the earthquake. Divide the region into several boxes, and the station weights in each box are the same, all set to 1/sqrt(N), where N is the total number of stations in the box.

# box size is dlon*dlat*ddep (degree, degree, km)
dlon = 0.4; dlat = 0.4; ddep = 5
ev_info_obs = ffd.box_weighting_ev(ev_info_obs,dlon,dlat,ddep)

# plot weight distribution
ffd.fig_ev_st_distribution_wt(ev_info_obs, st_info_obs)


# %%
# # Option 2, assign geographical weighting to the earthquake roughly. inspired by [Youyi Ruan et al., 2019, GJI]

# # box size is dlon*dlat*ddep (degree, degree, km)
# dlon = 0.2; dlat = 0.2; ddep = 5
# ev_info_obs = geographical_weighting_ev_rough(ev_info_obs,dlon,dlat,ddep)

# # 权重分布画图 plot weight distribution
# fig_ev_st_distribution_wt(ev_info_obs, st_info_obs)


# %%
# # option 3, declustering. Divide the region into several subdomains, retain the top N earthquakes in terms of the number of arrival times in each subdomain.

# dlon = 0.02; dlat = 0.02; ddep = 1;  # subdomain size (degree, degree, km)
# Top_N = 1 # retain the top N earthquakes in each subdomain
# ev_info_obs = limit_earthquake_decluster_Nt(ev_info_obs, dlon, dlat, ddep,Top_N)

# # plot weight distribution
# fig_ev_st_distribution_wt(ev_info_obs, st_info_obs)

# %%
# # Option 5 (recommended), add box weighting to the stations

# # box size is dlon*dlat*ddep (degree, degree, km)
# dlon = 0.2; dlat = 0.2; 
# (ev_info_obs,st_info_obs) = box_weighting_st(ev_info_obs,st_info_obs,dlon,dlat)

# # plot weight distribution
# fig_ev_st_distribution_wt(ev_info_obs, st_info_obs)

# %%
# Option 5 (recommended), add geographical weighting to the stations. [Youyi Ruan et al., 2019, GJI]

(ev_info_obs,st_info_obs) = ffd.geographical_weighting_st(ev_info_obs,st_info_obs)

# plot weight distribution
ffd.fig_ev_st_distribution_wt(ev_info_obs, st_info_obs)

# %%
# # output data 
# out_path = "output_data"

# # save data for TomoATT
# out_fname = "%s/alg4_src_rec.dat"%(out_path)
# ffd.write_src_rec_file(out_fname,ev_info_obs,st_info_obs)

# # save earthquake list for plotting
# out_fname_ev = "%s/alg4_ev_list.dat"%(out_path)
# ffd.write_src_list_file(out_fname_ev,ev_info_obs)

# # save station list for plotting
# out_fname_st = "%s/alg4_st_list.dat"%(out_path)
# ffd.write_rec_list_file(out_fname_st,ev_info_obs,st_info_obs)


