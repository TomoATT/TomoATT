# %%
import sys
sys.path.append('../utils')
import functions_for_data as ffd

# %%
from pytomoatt.model import ATTModel

import pygmt
import os
import numpy as np
from pygmt.clib import Session

with pygmt.clib.Session() as session:
    session.call_module('gmtset', 'FONT 20p')
fig = pygmt.Figure()
pygmt.config(IO_SEGMENT_MARKER="<<<")

import os

# %%
theta0_rotate  =   37.0    # degree
phi0_rotate    =   37.0    # degree
psi_rotate     =   45      # degree

# projectionmust be "psi_rotate+90", to make angle of anisotropy consistent in physical coordinate consistent with that in physical coordinate
# otherwise, please make necessary adjustment when plotting anisotropy arrows
projection = "OA%f/%f/%f/5c"%(theta0_rotate,phi0_rotate,psi_rotate+90)  
perspective = "135/90"
# plot specified depths
all_dep = [5,10,15]

region = "-1.5/1.5/-2.5/3.5",
region_large = [33,42,34,42]
xshift = [ 3,  8.5, 8.5, 8.5, ]
yshift = [ 80, 0,  0,  0, 0]
xshift_ani = [ -17,  8.5, 8.5, 8.5, ]
yshift_ani = [ -10, 0,  0,  0, ]

frame = [
    ["xa2g2","ya2g2","NsWe"],["xa2g2","ya2g2","Nswe"],
    ["xa2g2","ya2g2","NswE"],
]
frame_ani = [
    ["xa2g2","ya2g2","NSWe"],["xa2g2","ya2g2","NSwe"],
    ["xa2g2","ya2g2","NSwE"],
]

region_large = [33,42,34,42]
vel_per     = 5     # velocity perturbation colorbar range
ani_per     = 0.05  # anisotropy colorbar length
ani_per_1 = 0.015; ani_per_2 = 0.05 # anisotropy bar length [ani_per_1,ani_per_2] -> [0.1,0.6] (ani -> length)

subfigure = ['(a)', '(b)','(c)']
subfigure_ani = ['(d)', '(e)','(f)']


# %%
# read TomoATT model files

input_file = "./3_input_params/input_params_step3_inv_reloc.yaml"   # parameter file


fname_inv = "./OUTPUT_FILES/OUTPUT_FILES_step3_inv_reloc/final_model.h5"   # final model of inversion result
# fname_ini = "../OUTPUT_FILES/OUTPUT_FILES_step1_1D_inv/final_model.h5"      # initial model

model_inv = ATTModel.read(fname_inv,input_file)
# model_ini = ATTModel.read(fname_ini,input_file)

# get anisotropy model
model_inv.to_ani()

# bm_ini = model_ini.to_xarray()
bm_inv = model_inv.to_xarray()

n_rtp = bm_inv.vel.shape

# %%
# -------------------------- velocity model --------------------------------


for i in range(0,len(all_dep)):
# for i in range(0,1):
    print('plotting: dep %3d'%(all_dep[i]))

    x_cal         = bm_inv.interp_dep(all_dep[i],"vel")[:,0]
    y_cal         = bm_inv.interp_dep(all_dep[i],"vel")[:,1]
    (y_phy,x_phy)   = ffd.rtp_rotation_reverse(y_cal,x_cal,theta0_rotate,phi0_rotate,psi_rotate)    # rotate back to physical coordinate

    # average velocity as reference model
    ref_model = np.mean(bm_inv.interp_dep(all_dep[i],"vel")[:,2])

    pert        = bm_inv.interp_dep(all_dep[i],"vel")[:,2]
    pert        = (pert - ref_model)/ref_model * 100
    
    grid    = pygmt.surface(x=x_phy, y=y_phy, z=pert, spacing="01m", region=region_large)
   
    # ----------- basemap ----------------
    fig.shift_origin(xshift=xshift[i],yshift = yshift[i])
    fig.basemap(frame=frame[i], projection=projection, region=region, perspective = perspective)


    # ----------- colorbar ---------------- 
    pygmt.makecpt(cmap="./tectonics/svel13_chen.cpt", series=[-vel_per, vel_per], background = True)

    # ----------- velocity perturbation map ----------------
    fig.grdimage(frame=frame[i],grid = grid,projection=projection, region=region, perspective = perspective,transparency=30)
    
    # ----------- linement ----------------
    fig.plot("./tectonics/eshm20_new.txt", pen = '1p,gray38',perspective = perspective,)


    # ----------- coast ----------------
    fig.coast(shorelines="0.5p,black",perspective = perspective,)


    # rupture segments
    fig.plot("./tectonics/NPF_rupture_region.txt", pen = '5p,white',perspective=perspective,transparency = 0)
    fig.plot("./tectonics/EAF_rupture_region_Song1.txt", pen = '5p,white',perspective=perspective,transparency = 0)   # Seg.A 走滑 正 
    fig.plot("./tectonics/EAF_rupture_region_Song2.txt", pen = '5p,white',perspective=perspective,transparency = 0) # Seg P 走滑
    fig.plot("./tectonics/EAF_rupture_region_Song3.txt", pen = '5p,white',perspective=perspective,transparency = 0)   # Seg.E 走滑 逆冲
    
    fault_color = ["magenta","blue","purple"]
    fig.plot("./tectonics/NPF_rupture_region.txt", pen = '3p,%s'%(fault_color[1]),perspective=perspective,transparency = 0)
    fig.plot("./tectonics/EAF_rupture_region_Song1.txt", pen = '3p,%s'%(fault_color[0]),perspective=perspective,transparency = 0)   # Seg.A 走滑 正 
    fig.plot("./tectonics/EAF_rupture_region_Song2.txt", pen = '3p,%s'%(fault_color[1]),perspective=perspective,transparency = 0) # Seg P 走滑
    fig.plot("./tectonics/EAF_rupture_region_Song3.txt", pen = '3p,%s'%(fault_color[2]),perspective=perspective,transparency = 0)   # Seg.E 走滑 逆冲


    # ----------- major earthquakes ----------------
    fig.plot(x = [37.206], y = [38.016], style = "a", fill = "green3", size = [0.5],pen = '1p,black',perspective = perspective)
    fig.plot(x = [37.019], y = [37.220], style = "a", fill = "red", size = [0.8],pen = '1.5p,white',perspective = perspective)
    fig.plot(x = [39.061], y = [38.431], style = "a", fill = "blue", size = [0.6],pen = '1p,white',perspective = perspective)  #USGS


    # ----------- notations ----------------
    fig.shift_origin(xshift = 0.4,yshift = 0)
    fig.basemap( frame=["NSEW+gwhite"], projection="X5/2", region=[0,1,0,1],)
    fig.text(text="Vp  = %4.2f km/s"%(ref_model), x=0.5, y=0.75, font="18p,Helvetica-Bold,black", fill="white")
    fig.text(text="Depth = %3d km"%(all_dep[i]), x=0.5, y=0.25, font="18p,Helvetica-Bold,black", fill="white")
    fig.shift_origin(xshift =-0.4,yshift = 0)

# fig.show()

# %%
# -------------------------- azimuthal anisotropy --------------------------------
def ani_to_length(ani, ani_per_1, ani_per_2, scale, basic):
    return (ani - ani_per_1) / (ani_per_2 - ani_per_1) * scale + basic


for i in range(0,len(all_dep)):
    print('plotting: dep %3d'%(all_dep[i]))
    
    # anisotropy map
    x_cal         = bm_inv.interp_dep(all_dep[i],"vel")[:,0]
    y_cal         = bm_inv.interp_dep(all_dep[i],"vel")[:,1]
    (y_phy,x_phy)   = ffd.rtp_rotation_reverse(y_cal,x_cal,theta0_rotate,phi0_rotate,psi_rotate)
    epsilon     = bm_inv.interp_dep(all_dep[i],"epsilon")[:,2]

    grid = pygmt.surface(x=x_phy, y=y_phy, z=epsilon, spacing="01m", region=region_large)

    # fast velocity direction bars
    samp_interval = 5
    width = 0.1
    scale = 0.7
    basic = 0.3
    ani_thd = ani_per_1

    phi             = bm_inv.interp_dep(all_dep[i], field='phi')    # phi_inv[i,:] are (lon, lat, phi)
    ani_phi         = phi[:,2].reshape(n_rtp[1],n_rtp[2])    # phi
    ani_phi         = ani_phi[0:-1:samp_interval,0:-1:samp_interval].reshape(-1,1)    # phi

    epsilon         = bm_inv.interp_dep(all_dep[i], field='epsilon')    # magnitude of anisotropy
    ani_epsilon     = epsilon[:,2].reshape(n_rtp[1],n_rtp[2])    # magnitude of anisotropy
    ani_epsilon     = ani_epsilon[0:-1:samp_interval,0:-1:samp_interval].reshape(-1,1)    # magnitude of anisotropy
    
    ani_lon         = x_phy.reshape(n_rtp[1],n_rtp[2])    # longitude
    ani_lon         = ani_lon[0:-1:samp_interval,0:-1:samp_interval].reshape(-1,1)    # longitude

    ani_lat         = y_phy.reshape(n_rtp[1],n_rtp[2])    # latitude
    ani_lat         = ani_lat[0:-1:samp_interval,0:-1:samp_interval].reshape(-1,1)    # latitude

    length          = ani_to_length(ani_epsilon, ani_per_1, ani_per_2, scale, basic)

    ani_arrow_full  = np.hstack([ani_lon, ani_lat, ani_phi, length, np.ones((ani_lon.size,1))*width])   # lon, lat, angle[-90,90], length, width
    
    idx = np.where((ani_epsilon > ani_thd))
    ani_arrow       = ani_arrow_full[idx[0],:]


    # ----------- colorbar ---------------- 
    pygmt.makecpt(cmap="cool", series=[0, ani_per],background=True)


    # ----------- basemap ----------------
    fig.shift_origin(xshift=xshift_ani[i],yshift = yshift_ani[i])
    fig.basemap(frame=frame_ani[i], projection=projection, region=region, perspective = perspective)

    # ----------- anisotropy map ----------------
    fig.grdimage(frame=frame_ani[i],grid = grid,projection=projection,region=region,perspective = perspective,)

    # fast velocity direction bars
    fig.plot(ani_arrow, style = "j", pen = "0.5p,white",fill="black",perspective = perspective)

    # ----------- linement ----------------    
    fig.plot("./tectonics/eshm20_new.txt", pen = '1p,gray38',perspective = perspective,)

    # ----------- coastlines ----------------
    fig.coast(shorelines="0.5p,black",perspective = perspective,)
    
    # rupture segments
    fig.plot("./tectonics/NPF_rupture_region.txt", pen = '5p,white',perspective=perspective,transparency = 0)
    fig.plot("./tectonics/EAF_rupture_region_Song1.txt", pen = '5p,white',perspective=perspective,transparency = 0)   # Seg.A 
    fig.plot("./tectonics/EAF_rupture_region_Song2.txt", pen = '5p,white',perspective=perspective,transparency = 0) # Seg P
    fig.plot("./tectonics/EAF_rupture_region_Song3.txt", pen = '5p,white',perspective=perspective,transparency = 0)   # Seg.E 

    fault_color = ["magenta","blue","purple"]
    fig.plot("./tectonics/NPF_rupture_region.txt", pen = '3p,%s'%(fault_color[1]),perspective=perspective,transparency = 0)
    fig.plot("./tectonics/EAF_rupture_region_Song1.txt", pen = '3p,%s'%(fault_color[0]),perspective=perspective,transparency = 0)   # Seg.A
    fig.plot("./tectonics/EAF_rupture_region_Song2.txt", pen = '3p,%s'%(fault_color[1]),perspective=perspective,transparency = 0) # Seg P
    fig.plot("./tectonics/EAF_rupture_region_Song3.txt", pen = '3p,%s'%(fault_color[2]),perspective=perspective,transparency = 0)   # Seg.E 


    # ----------- major earthquakes ----------------
    fig.plot(x = [37.206], y = [38.016], style = "a", fill = "green3", size = [0.5],pen = '1p,black',perspective = perspective)
    fig.plot(x = [37.019], y = [37.220], style = "a", fill = "red", size = [0.8],pen = '1.5p,white',perspective = perspective)
    fig.plot(x = [39.061], y = [38.431], style = "a", fill = "blue", size = [0.6],pen = '1p,white',perspective = perspective)  #USGS
    

    # ----------- notations ----------------
    fig.shift_origin(xshift = 4.5,yshift = -1.5)
    fig.basemap( frame=["NSEW"], projection="X5/1", region=[0,1,0,1],)
    fig.text(text="Depth = %3d km"%(all_dep[i]), x=0.5, y=0.5, font="18p,Helvetica-Bold,black", fill="white")
    fig.shift_origin(xshift =-4.5,yshift = 1.5)

# ----------- colorbars ----------------
pygmt.makecpt(cmap="cool", series=[0, ani_per],background=True)
fig.shift_origin(xshift=2,yshift = -3)
fig.colorbar(frame = ["a0.02","x+lAnisotropy"], position="+ef+w7c/0.5c+h") 

pygmt.makecpt(cmap="./tectonics/svel13_chen.cpt", series=[-vel_per, vel_per], background = True)
fig.shift_origin(xshift = -10, yshift = 0)
fig.colorbar(frame = ["a4f2","x+lVelocity pertubation (%)"], position="+e+w7c/0.5c+h", ) 

fig.shift_origin(xshift =  -8,yshift = -1)
fig.basemap(frame=["NSEW"],projection="X6/2", region=[0,1,0,1])

length = ani_to_length(0.02, ani_per_1, ani_per_2, 0.7, 0.3)
fig.plot(x = 0.2, y = 0.6, style = "j45/%f/0.1"%(length), pen = "0.5p,white",fill="black")
fig.text(text="0.02", x=0.2, y=0.2, font="18p,Helvetica-Bold,black")

length = ani_to_length(0.03, ani_per_1, ani_per_2, 0.7, 0.3)
fig.plot(x = 0.5, y = 0.6, style = "j45/%f/0.1"%(length), pen = "0.5p,white",fill="black")
fig.text(text="0.03", x=0.5, y=0.2, font="18p,Helvetica-Bold,black")

length = ani_to_length(0.05, ani_per_1, ani_per_2, 0.7, 0.3)
fig.plot(x = 0.8, y = 0.6, style = "j45/%f/0.1"%(length), pen = "0.5p,white",fill="black")
fig.text(text="0.05", x=0.8, y=0.2, font="18p,Helvetica-Bold,black")

os.makedirs('./img', exist_ok=True)
fig.savefig('./img/imaging_result.jpg')


# fig.show()



