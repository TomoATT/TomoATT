# %%
import pygmt
pygmt.config(FONT="16p", IO_SEGMENT_MARKER="<<<")

import os

# %%
from pytomoatt.model import ATTModel
from pytomoatt.data import ATTData
import numpy as np

# %%
# read models

Ngrid = [61,61,61]
data_file = '2_models/model_init_N%d_%d_%d.h5'%(Ngrid[0],Ngrid[1],Ngrid[2])
par_file = '3_input_params/input_params_signal.yaml'  
model = ATTModel.read(data_file, par_file)
initial_model = model.to_xarray()

data_file = '2_models/model_ckb_N%d_%d_%d.h5'%(Ngrid[0],Ngrid[1],Ngrid[2])
model = ATTModel.read(data_file, par_file)
ckb_model = model.to_xarray()

# initial model
depth = 10.0
vel_init = initial_model.interp_dep(depth, field='vel')
start = [1.25,0]; end = [1.25,2]
vel_init_sec = initial_model.interp_sec(start, end, field='vel', val = 1)

# checkerboard model
vel_ckb = ckb_model.interp_dep(depth, field='vel') # lon = [:,0], lat = [:,1], vel = [:,2]  
vel_ckb_sec = ckb_model.interp_sec(start, end, field='vel', val = 1)

# anisotropic arrow
samp_interval = 3
length = 7
width = 0.1
ani_thd = 0.02

ani_ckb_phi = ckb_model.interp_dep(depth, field='phi', samp_interval=samp_interval)
ani_ckb_epsilon = ckb_model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)
ani_ckb = np.hstack([ani_ckb_phi, ani_ckb_epsilon[:,2].reshape(-1, 1)*length, np.ones((ani_ckb_epsilon.shape[0],1))*width])   # lon, lat, angle, length, width
idx = np.where(ani_ckb_epsilon[:,2] > ani_thd)
ani_ckb = ani_ckb[idx[0],:]

try:
    os.mkdir('img')
except:
    pass

# %%
# read src_rec_file for data
from pytomoatt.src_rec import SrcRec

sr = SrcRec.read('1_src_rec_files/src_rec_config.dat')
station = sr.receivers[['stlo','stla','stel']].values.T
true_loc = sr.sources[['evlo','evla','evdp']].values.T
earthquake = true_loc

sr = SrcRec.read('OUTPUT_FILES/OUTPUT_FILES_signal/src_rec_file_forward_errloc.dat')
init_loc = sr.sources[['evlo','evla','evdp']].values.T

# %%
# categorize earthquakes
ev_idx1 = []
ev_idx2 = []
ev_idx3 = []
for i in range(earthquake.shape[1]):
    dep = earthquake[2,i]
    if dep < 15:
        ev_idx1.append(i)
    elif dep < 25:
        ev_idx2.append(i)
    elif dep < 35:
        ev_idx3.append(i)

# %%
# plot the model setting
fig = pygmt.Figure()

region = [0,2,0,2]
frame  = ["xa1","ya1"]
projection = "M10c"
spacing = 0.04

vel_range = 20

# -------------- initial model and earthquake location --------------
fig.basemap(region=region, frame=["xa1","ya1","+tInitial model and locations"], projection=projection)
# velocity perturbation
pygmt.makecpt(cmap="../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
x = vel_init[:,0]; y = vel_init[:,1]; value = (vel_init[:,2] - vel_init[:,2])/vel_init[:,2] * 100
grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
fig.grdimage(grid = grid)
# earthquakes
fig.plot(x = init_loc[0,ev_idx1], y = init_loc[1,ev_idx1], style = "c0.1c", fill = "red")
fig.plot(x = init_loc[0,ev_idx2], y = init_loc[1,ev_idx2], style = "c0.1c", fill = "green")
fig.plot(x = init_loc[0,ev_idx3], y = init_loc[1,ev_idx3], style = "c0.1c", fill = "black")

# stations
fig.plot(x = station[0,:], y = station[1,:], style = "t0.4c", fill = "blue", pen = "black", label = "Station")

# # anisotropic arrow
# fig.plot(ani_ckb, style='j', fill='yellow1', pen='0.5p,black')

fig.shift_origin(xshift=11)

fig.basemap(region=[0,40,0,2], frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
x = vel_init_sec[:,3]; y = vel_init_sec[:,1]; value = (vel_init_sec[:,4] - vel_init_sec[:,4])/vel_init_sec[:,4] * 100
grid = pygmt.surface(x=x, y=y, z=value, spacing="1/0.04",region=[0,40,0,2])
fig.grdimage(grid = grid)

# earthquakes
fig.plot(x = init_loc[2,ev_idx1], y = init_loc[1,ev_idx1], style = "c0.1c", fill = "red")
fig.plot(x = init_loc[2,ev_idx2], y = init_loc[1,ev_idx2], style = "c0.1c", fill = "green")
fig.plot(x = init_loc[2,ev_idx3], y = init_loc[1,ev_idx3], style = "c0.1c", fill = "black")


fig.shift_origin(xshift=4)


# -------------- true model and earthquake location --------------
fig.basemap(region=region, frame=["xa1","ya1","+tTrue model and locations"], projection=projection)
# velocity perturbation
pygmt.makecpt(cmap="../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
x = vel_ckb[:,0]; y = vel_ckb[:,1]; value = (vel_ckb[:,2] - vel_init[:,2])/vel_init[:,2] * 100
grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
fig.grdimage(grid = grid)
# earthquakes
fig.plot(x = true_loc[0,ev_idx1], y = true_loc[1,ev_idx1], style = "c0.1c", fill = "red")
fig.plot(x = true_loc[0,ev_idx2], y = true_loc[1,ev_idx2], style = "c0.1c", fill = "green")
fig.plot(x = true_loc[0,ev_idx3], y = true_loc[1,ev_idx3], style = "c0.1c", fill = "black")

# stations
# fig.plot(x = loc_st[0,:], y = loc_st[1,:], style = "t0.4c", fill = "blue", pen = "black", label = "Station")

# anisotropic arrow
fig.plot(ani_ckb, style='j', fill='yellow1', pen='0.5p,black')

fig.shift_origin(xshift=11)

fig.basemap(region=[0,40,0,2], frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
x = vel_ckb_sec[:,3]; y = vel_ckb_sec[:,1]; value = (vel_ckb_sec[:,4] - vel_init_sec[:,4])/vel_init_sec[:,4] * 100
grid = pygmt.surface(x=x, y=y, z=value, spacing="1/0.04",region=[0,40,0,2])
fig.grdimage(grid = grid)

# earthquakes
fig.plot(x = true_loc[2,ev_idx1], y = true_loc[1,ev_idx1], style = "c0.1c", fill = "red")
fig.plot(x = true_loc[2,ev_idx2], y = true_loc[1,ev_idx2], style = "c0.1c", fill = "green")
fig.plot(x = true_loc[2,ev_idx3], y = true_loc[1,ev_idx3], style = "c0.1c", fill = "black")


# ------------------- colorbar -------------------
fig.shift_origin(xshift=-11, yshift=-1.5)
fig.colorbar(frame = ["a%f"%(vel_range),"x+ldlnVp (%)"], position="+e+w4c/0.3c+h")

fig.shift_origin(xshift=6, yshift=-1)
fig.basemap(region=[0,1,0,1], frame=["wesn"], projection="X6c/1.5c")
ani = [
    [0.2, 0.6, 45, 0.02*length, width],    # lon, lat, phi, epsilon, size
    [0.5, 0.6, 45, 0.05*length, width],
    [0.8, 0.6, 45, 0.10*length, width],
    ]
fig.plot(ani, style='j', fill='yellow1', pen='0.5p,black')
fig.text(text=["0.02", "0.05", "0.10"], x=[0.2,0.5,0.8], y=[0.2]*3, font="16p,Helvetica", justify="CM")
fig.shift_origin(xshift= 11, yshift=2.5)

fig.show()
fig.savefig('img/model_setting.png', dpi=300)

# %%
# plot the joint inversion results

fig = pygmt.Figure()

region = [0,2,0,2]
projection = "M10c"
spacing = 0.04

vel_range = 20

tag_list = ["joint_step1", "joint_step2", "joint_step3"]

for itag, tag in enumerate(tag_list):

    if (tag == "joint_step1"):
        # model
        x = vel_init[:,0]; y = vel_init[:,1]; value = (vel_init[:,2] - vel_init[:,2])/vel_init[:,2] * 100
        x_sec = vel_init_sec[:,3]; y_sec = vel_init_sec[:,1]; value_sec = (vel_init_sec[:,4] - vel_init_sec[:,4])/vel_init_sec[:,4] * 100

        # location
        sr = SrcRec.read('OUTPUT_FILES/OUTPUT_FILES_%s/src_rec_file_reloc_0050.dat'%(tag))
        re_loc = sr.sources[['evlo','evla','evdp']].values.T

        frame  = ["xa1","ya1","+tStep %d, preliminary location"%(itag+1)]
    elif (tag == "joint_step2"):
        # model
        data_file = "OUTPUT_FILES/OUTPUT_FILES_%s/final_model.h5"%(tag)
        model = ATTModel.read(data_file, par_file)
        inv_model = model.to_xarray()
        vel_inv = inv_model.interp_dep(depth, field='vel') # lon = [:,0], lat = [:,1], vel = [:,2]  
        x = vel_inv[:,0]; y = vel_inv[:,1]; value = (vel_inv[:,2] - vel_init[:,2])/vel_init[:,2] * 100
        vel_inv_sec = inv_model.interp_sec(start, end, field='vel', val = 1)
        x_sec = vel_inv_sec[:,3]; y_sec = vel_inv_sec[:,1]; value_sec = (vel_inv_sec[:,4] - vel_init_sec[:,4])/vel_init_sec[:,4] * 100

        ani_inv_phi = inv_model.interp_dep(depth, field='phi', samp_interval=samp_interval)
        ani_inv_epsilon = inv_model.interp_dep(depth, field='epsilon', samp_interval=samp_interval)
        ani_inv = np.hstack([ani_inv_phi, ani_inv_epsilon[:,2].reshape(-1, 1)*length, np.ones((ani_inv_epsilon.shape[0],1))*width])   # lon, lat, angle, length, width
        idx = np.where(ani_inv_epsilon[:,2] > ani_thd)
        ani = ani_inv[idx[0],:]

        # location
        sr = SrcRec.read('OUTPUT_FILES/OUTPUT_FILES_%s/src_rec_file_inv_0039_reloc_0039.dat'%(tag))
        re_loc = sr.sources[['evlo','evla','evdp']].values.T

        frame  = ["xa1","ya1","+tStep %d, joint inversion"%(itag+1)]
    elif (tag == "joint_step3"):

        # model
        x = vel_inv[:,0]; y = vel_inv[:,1]; value = (vel_inv[:,2] - vel_init[:,2])/vel_init[:,2] * 100
        x_sec = vel_inv_sec[:,3]; y_sec = vel_inv_sec[:,1]; value_sec = (vel_inv_sec[:,4] - vel_init_sec[:,4])/vel_init_sec[:,4] * 100
        ani = ani_inv[idx[0],:]

        # location
        sr = SrcRec.read('OUTPUT_FILES/OUTPUT_FILES_%s/src_rec_file_reloc_0100.dat'%(tag))
        re_loc = sr.sources[['evlo','evla','evdp']].values.T

        frame  = ["xa1","ya1","+tStep %d, relocation"%(itag+1)]

    # plot the inversion result


    # -------------- inversion model --------------
    fig.basemap(region=region, frame=frame, projection=projection)
    # velocity perturbation
    pygmt.makecpt(cmap="../utils/svel13_chen.cpt", series=[-vel_range, vel_range], background=True, reverse=False)
    grid = pygmt.surface(x=x, y=y, z=value, spacing=spacing,region=region)
    fig.grdimage(grid = grid)
    # earthquakes
    fig.plot(x = re_loc[0,ev_idx1], y = re_loc[1,ev_idx1], style = "c0.1c", fill = "red")
    fig.plot(x = re_loc[0,ev_idx2], y = re_loc[1,ev_idx2], style = "c0.1c", fill = "green")
    fig.plot(x = re_loc[0,ev_idx3], y = re_loc[1,ev_idx3], style = "c0.1c", fill = "black")

    # stations
    # fig.plot(x = loc_st[0,:], y = loc_st[1,:], style = "t0.4c", fill = "blue", pen = "black", label = "Station")

    # anisotropic arrow
    if (tag == "joint_step2" or tag == "joint_step3"):
        fig.plot(ani, style='j', fill='yellow1', pen='0.5p,black')

    fig.shift_origin(xshift=11)

    fig.basemap(region=[0,40,0,2], frame=["xa20+lDepth (km)","ya1","Nswe"], projection="X2c/10c")
    grid = pygmt.surface(x=x_sec, y=y_sec, z=value_sec, spacing="1/0.04",region=[0,40,0,2])
    fig.grdimage(grid = grid)

    # earthquakes
    fig.plot(x = re_loc[2,ev_idx1], y = re_loc[1,ev_idx1], style = "c0.1c", fill = "red")
    fig.plot(x = re_loc[2,ev_idx2], y = re_loc[1,ev_idx2], style = "c0.1c", fill = "green")
    fig.plot(x = re_loc[2,ev_idx3], y = re_loc[1,ev_idx3], style = "c0.1c", fill = "black")

    fig.shift_origin(xshift=4)

# ------------------- colorbar -------------------
fig.shift_origin(xshift=-4)

fig.shift_origin(xshift=-11, yshift=-1.5)
fig.colorbar(frame = ["a%f"%(vel_range),"x+ldlnVp (%)"], position="+e+w4c/0.3c+h") 

fig.shift_origin(xshift=6, yshift=-1)
fig.basemap(region=[0,1,0,1], frame=["wesn"], projection="X6c/1.5c")
ani = [
    [0.2, 0.6, 45, 0.02*length, width],    # lon, lat, phi, epsilon, size
    [0.5, 0.6, 45, 0.05*length, width],
    [0.8, 0.6, 45, 0.10*length, width],
    ]
fig.plot(ani, style='j', fill='yellow1', pen='0.5p,black')
fig.text(text=["0.02", "0.05", "0.10"], x=[0.2,0.5,0.8], y=[0.2]*3, font="16p,Helvetica", justify="CM")
fig.shift_origin(xshift= 11, yshift=2.5)


fig.show()
fig.savefig('img/model_joint.png', dpi=300)


