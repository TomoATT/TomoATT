# %%
# Initialization, class definition, and declaration.

import os
import math
from obspy import UTCDateTime
import numpy as np
import copy

class Event():      # class of earthquake
    def __init__(self):
        self.name = "nan"       # evname1 Earthquake name, recommended as "earthquake".
        self.id = -1
        self.lat = 0.0
        self.lon = 0.0
        self.dep = 0.0
        self.mag = 0.0
        self.ortime = UTCDateTime(1999,1,1,0,0,0)
        self.Nt = 0             # Number of the absolute traveltime of earthquake
        self.Ncs_dt = 0         # Number of the commmon source differential traveltime of earthquake
        self.Ncr_dt = 0         # Number of the commmon receiver differential traveltime of earthquake
        self.t = {}             # stname1+phase -> (stname1, phase, time, data_weight)
        self.cs_dt = {}         # stname1 + stname2 + phase -> (stname1, stname2, phase, dif_time, data_weight)
        self.cr_dt = {}         # stname1 + evname2 + phase -> (stname1, evname2, phase, dif_time, data_weight)
        self.azi_gap = 360.0    # the max azimuthal gap of each earthquake
        self.misfit = {}        # traveltime residual of the data, the difference between real data and synthetic data, used for evaluation. stname or stname1+stname2 or stname1+evname2 -> residual
        self.tag    = {}        # additional tags for the earthquake, e.g., azi_gap, weight. (azimuthal gap, weight of the earthquake)

class Station():
    def __init__(self):
        self.name = "nan"       # stname1, recommend: network.stname
        self.id = -1
        self.lat = 0.0
        self.lon = 0.0
        self.ele = 0.0
        self.tag = {}           # additional tags for the station, e.g., wright




# %% [markdown]
# Functions: some basic auxiliary functions for processing data

# %%
# function： cal_dis(lat1, lon1,lat2, lon2) (in kilometers)， cal_azimuth(lat1, lon1, lat2, lon2) (degree) calculate epicentral distance (km) and azimuth (degree)

def cal_dis(lat1, lon1,lat2, lon2, R = 6371):
    latitude1 = (math.pi/180)*lat1
    latitude2 = (math.pi/180)*lat2
    longitude1 = (math.pi/180)*lon1
    longitude2= (math.pi/180)*lon2
    # Therefore, the spherical distance between points A and B is:{arccos[sinb*siny+cosb*cosy*cos(a-x)]}*R
    # Radius of the earth
    if((lat1-lat2)**2+(lon1-lon2)**2<0.000001):
        return 0

    d = math.acos(math.sin(latitude1)*math.sin(latitude2)+ math.cos(latitude1)*math.cos(latitude2)*math.cos(longitude2-longitude1))/math.pi*180
    return d * 2 * math.pi * R / 360

def cal_azimuth(lat1, lon1, lat2, lon2):
    lat1_rad = lat1 * math.pi / 180
    lon1_rad = lon1 * math.pi / 180
    lat2_rad = lat2 * math.pi / 180
    lon2_rad = lon2 * math.pi / 180

    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
    brng = math.atan2(y, x) * 180 / math.pi
    if((lat1-lat2)**2+(lon1-lon2)**2<0.0001):
        return 0
    return float((brng + 360.0) % 360.0)


# %%
# Function: Coordinate rotation rotate_src_rec(ev_info, st_info, theta0, phi0, psi): rotate to the new coordinate system, satisfying the center point transformation r0, t0, p0 -> r0, 0, 0 and an anticlockwise rotation angle psi.
# Satisfying the center point transformation r0, t0, p0 -> r0, 0, 0 and an anticlockwise rotation angle psi.

import numpy as np

RAD2DEG = 180/np.pi
DEG2RAD = np.pi/180
R_earth = 6371.0

# Spherical coordinates to Cartesian coordinate
def rtp2xyz(r,theta,phi):
    x = r * np.cos(theta*DEG2RAD) * np.cos(phi*DEG2RAD)
    y = r * np.cos(theta*DEG2RAD) * np.sin(phi*DEG2RAD)
    z = r * np.sin(theta*DEG2RAD)
    return (x,y,z)

# Cartesian coordinates to Spherical coordinate
def xyz2rtp(x,y,z):
    # theta: -90~90;  phi: -180~180
    r       = np.sqrt(x**2+y**2+z**2)
    theta   = np.arcsin(z/r)
    phi     = np.arcsin(y/r/np.cos(theta))


    idx = np.where((phi > 0) & (x*y < 0))
    phi[idx] = np.pi - phi[idx]
    idx = np.where((phi < 0) & (x*y > 0))
    phi[idx] = -np.pi - phi[idx]


    # for i in range(phi.size):
    #     if(phi[i] > 0 and x[i]*y[i] < 0):
    #         phi[i] = np.pi - phi[i]
    #     if(phi[i] < 0 and x[i]*y[i] > 0):
    #         phi[i] = -np.pi - phi[i]

    return (r,theta*RAD2DEG,phi*RAD2DEG)

# anti-clockwise rotation along x-axis
def rotate_x(x,y,z,theta):
    new_x = x
    new_y = y *  np.cos(theta*DEG2RAD) + z * -np.sin(theta*DEG2RAD)
    new_z = y *  np.sin(theta*DEG2RAD) + z *  np.cos(theta*DEG2RAD)
    return (new_x,new_y,new_z)

# anti-clockwise rotation along y-axis
def rotate_y(x,y,z,theta):
    new_x = x *  np.cos(theta*DEG2RAD) + z *  np.sin(theta*DEG2RAD)
    new_y = y
    new_z = x * -np.sin(theta*DEG2RAD) + z *  np.cos(theta*DEG2RAD)
    return (new_x,new_y,new_z)

# anti-clockwise rotation along z-axis
def rotate_z(x,y,z,theta):
    new_x = x *  np.cos(theta*DEG2RAD) + y * -np.sin(theta*DEG2RAD)
    new_y = x *  np.sin(theta*DEG2RAD) + y *  np.cos(theta*DEG2RAD)
    new_z = z
    return (new_x,new_y,new_z)

# spherical Rotation

# rotate to the new coordinate, satisfying the center r0,t0,p0 -> r0,0,0 and a anticlockwise angle psi
def rtp_rotation(t,p,theta0,phi0,psi):
    # step 1: r,t,p -> x,y,z
    (x,y,z) = rtp2xyz(1.0,t,p)

    # step 2: anti-clockwise rotation with -phi0 along z-axis:   r0,t0,p0 -> r0,t0,0
    (x,y,z) = rotate_z(x,y,z,-phi0)

    # step 3: anti-clockwise rotation with theta0 along y-axis:  r0,t0,0 -> r0,0,0
    (x,y,z) = rotate_y(x,y,z,theta0)

    # # step 4: anti-clockwise rotation with psi along x-axis
    (x,y,z) = rotate_x(x,y,z,psi)

    # step 5: x,y,z -> r,t,p
    (new_r,new_t,new_p) = xyz2rtp(x,y,z)

    return (new_t,new_p)


def rtp_rotation_reverse(new_t,new_p,theta0,phi0,psi):
    # step 1: r,t,p -> x,y,z
    (x,y,z) = rtp2xyz(1.0,new_t,new_p)

    # step 2: anti-clockwise rotation with -psi along x-axis
    (x,y,z) = rotate_x(x,y,z,-psi)

    # step 3: anti-clockwise rotation with -theta0 along y-axis:  r0,0,0 -> r0,t0,0
    (x,y,z) = rotate_y(x,y,z,-theta0)

    # step 4: anti-clockwise rotation with phi0 along z-axis:   r0,t0,0 -> r0,t0,p0
    (x,y,z) = rotate_z(x,y,z,phi0)

    # step 5: x,y,z -> r,t,p
    (r,t,p) = xyz2rtp(x,y,z)

    return (t,p)

def rotate_src_rec(ev_info,st_info,theta0,phi0,psi):
    ev_info_rotate = {}
    st_info_rotate = {}

    # rotate earthquakes
    for key_ev in ev_info:
        ev = ev_info[key_ev]
        ev_lat = np.array([ev.lat]); ev_lon = np.array([ev.lon])
        (ev_lat,ev_lon,) = rtp_rotation(ev_lat,ev_lon,theta0,phi0,psi)
        ev.lat = ev_lat[0]; ev.lon = ev_lon[0]
        ev_info_rotate[key_ev] = ev

    # rotate stations
    for key_st in st_info:
        st = st_info[key_st]
        st_lat = np.array([st.lat]); st_lon = np.array([st.lon])
        (st_lat,st_lon) = rtp_rotation(st_lat,st_lon,theta0,phi0,psi)
        st.lat = st_lat[0]; st.lon = st_lon[0]
        st_info_rotate[key_st] = st

    return (ev_info_rotate,st_info_rotate)

def rotate_src_rec_reverse(ev_info_rotate,st_info_rotate,theta0,phi0,psi):
    ev_info = {}
    st_info = {}

    # rotate earthquakes
    for key_ev in ev_info_rotate:
        ev = ev_info_rotate[key_ev]
        ev_lat = np.array([ev.lat]); ev_lon = np.array([ev.lon])
        (ev_lat,ev_lon,) = rtp_rotation_reverse(ev_lat,ev_lon,theta0,phi0,psi)
        ev.lat = ev_lat[0]; ev.lon = ev_lon[0]
        ev_info[key_ev] = ev

    # rotate stations
    for key_st in st_info_rotate:
        st = st_info_rotate[key_st]
        st_lat = np.array([st.lat]); st_lon = np.array([st.lon])
        (st_lat,st_lon) = rtp_rotation_reverse(st_lat,st_lon,theta0,phi0,psi)
        st.lat = st_lat[0]; st.lon = st_lon[0]
        st_info[key_st] = st

    return (ev_info,st_info)

# %%
# # Function: Coordinate rotation rotate_src_rec(ev_info, st_info, theta0, phi0, psi): rotate to the new coordinate system, satisfying the center point transformation r0, t0, p0 -> r0, 0, 0 and an anticlockwise rotation angle psi.
# # Satisfying the center point transformation r0, t0, p0 -> r0, 0, 0 and an anticlockwise rotation angle psi.

# import numpy as np

# RAD2DEG = 180/np.pi
# DEG2RAD = np.pi/180
# R_earth = 6371.0

# # Spherical coordinates to Cartesian coordinate
# def rtp2xyz(r,theta,phi):
#     x = r * np.cos(theta*DEG2RAD) * np.cos(phi*DEG2RAD)
#     y = r * np.cos(theta*DEG2RAD) * np.sin(phi*DEG2RAD)
#     z = r * np.sin(theta*DEG2RAD)
#     return (x,y,z)

# # Cartesian coordinates to Spherical coordinate
# def xyz2rtp(x,y,z):
#     # theta: -90~90;  phi: -180~180
#     r       = np.sqrt(x**2+y**2+z**2)
#     theta   = np.arcsin(z/r)
#     phi     = np.arcsin(y/r/np.cos(theta))


#     if(phi > 0 and x*y < 0):
#         phi = np.pi - phi
#     if(phi < 0 and x*y > 0):
#         phi = -np.pi - phi

#     return (r,theta*RAD2DEG,phi*RAD2DEG)

# # anti-clockwise rotation along x-axis
# def rotate_x(x,y,z,theta):
#     new_x = x
#     new_y = y *  np.cos(theta*DEG2RAD) + z * -np.sin(theta*DEG2RAD)
#     new_z = y *  np.sin(theta*DEG2RAD) + z *  np.cos(theta*DEG2RAD)
#     return (new_x,new_y,new_z)

# # anti-clockwise rotation along y-axis
# def rotate_y(x,y,z,theta):
#     new_x = x *  np.cos(theta*DEG2RAD) + z *  np.sin(theta*DEG2RAD)
#     new_y = y
#     new_z = x * -np.sin(theta*DEG2RAD) + z *  np.cos(theta*DEG2RAD)
#     return (new_x,new_y,new_z)

# # anti-clockwise rotation along z-axis
# def rotate_z(x,y,z,theta):
#     new_x = x *  np.cos(theta*DEG2RAD) + y * -np.sin(theta*DEG2RAD)
#     new_y = x *  np.sin(theta*DEG2RAD) + y *  np.cos(theta*DEG2RAD)
#     new_z = z
#     return (new_x,new_y,new_z)

# # spherical Rotation

# # rotate to the new coordinate, satisfying the center r0,t0,p0 -> r0,0,0 and a anticlockwise angle psi
# def rtp_rotation(t,p,theta0,phi0,psi):
#     # step 1: r,t,p -> x,y,z
#     (x,y,z) = rtp2xyz(1.0,t,p)

#     # step 2: anti-clockwise rotation with -phi0 along z-axis:   r0,t0,p0 -> r0,t0,0
#     (x,y,z) = rotate_z(x,y,z,-phi0)

#     # step 3: anti-clockwise rotation with theta0 along y-axis:  r0,t0,0 -> r0,0,0
#     (x,y,z) = rotate_y(x,y,z,theta0)

#     # # step 4: anti-clockwise rotation with psi along x-axis
#     (x,y,z) = rotate_x(x,y,z,psi)

#     # step 5: x,y,z -> r,t,p
#     (new_r,new_t,new_p) = xyz2rtp(x,y,z)

#     return (new_t,new_p)


# def rtp_rotation_reverse(new_t,new_p,theta0,phi0,psi):
#     # step 1: r,t,p -> x,y,z
#     (x,y,z) = rtp2xyz(1.0,new_t,new_p)

#     # step 2: anti-clockwise rotation with -psi along x-axis
#     (x,y,z) = rotate_x(x,y,z,-psi)

#     # step 3: anti-clockwise rotation with -theta0 along y-axis:  r0,0,0 -> r0,t0,0
#     (x,y,z) = rotate_y(x,y,z,-theta0)

#     # step 4: anti-clockwise rotation with phi0 along z-axis:   r0,t0,0 -> r0,t0,p0
#     (x,y,z) = rotate_z(x,y,z,phi0)

#     # step 5: x,y,z -> r,t,p
#     (r,t,p) = xyz2rtp(x,y,z)

#     return (t,p)

# def rotate_src_rec(ev_info,st_info,theta0,phi0,psi):
#     ev_info_rotate = {}
#     st_info_rotate = {}

#     # rotate earthquakes
#     for key_ev in ev_info:
#         ev = ev_info[key_ev]
#         (ev.lat,ev.lon,) = rtp_rotation(ev.lat,ev.lon,theta0,phi0,psi)
#         ev_info_rotate[key_ev] = ev

#     # rotate stations
#     for key_st in st_info:
#         st = st_info[key_st]
#         (st.lat,st.lon) = rtp_rotation(st.lat,st.lon,theta0,phi0,psi)
#         st_info_rotate[key_st] = st

#     return (ev_info_rotate,st_info_rotate)

# def rotate_src_rec_reverse(ev_info_rotate,st_info_rotate,theta0,phi0,psi):
#     ev_info = {}
#     st_info = {}

#     # rotate earthquakes
#     for key_ev in ev_info_rotate:
#         ev = ev_info_rotate[key_ev]
#         (ev.lat,ev.lon) = rtp_rotation_reverse(ev.lat,ev.lon,theta0,phi0,psi)
#         ev_info[key_ev] = ev

#     # rotate stations
#     for key_st in st_info_rotate:
#         st = st_info_rotate[key_st]
#         (st.lat,st.lon) = rtp_rotation_reverse(st.lat,st.lon,theta0,phi0,psi)
#         st_info[key_st] = st

#     return (ev_info,st_info)

# %%
# linear_regression(X,Y)
def linear_regression(X,Y):
    slope,intercept = np.polyfit(X,Y,deg=1)
    fitted_values = slope * X + intercept
    residual = Y - fitted_values
    SEE = np.std(residual)
    return (slope,intercept,SEE)

# %% [markdown]
# 
# Functions: obtain target information from ev_info and st_info

# %%
# function: output the [lon,lat,dep,weight] of the earthquake
def data_lon_lat_dep_wt_ev(ev_info):
    lat = []
    lon = []
    dep = []
    weight = []
    for key in ev_info:
        lat.append(ev_info[key].lat)
        lon.append(ev_info[key].lon)
        dep.append(ev_info[key].dep)
        try:
            weight.append(ev_info[key].tag["weight"])
        except:
            weight.append(1.0)
    return [np.array(lon),np.array(lat),np.array(dep),np.array(weight)]

# %%
# function: output the [lon, lat, dep, ortime] of the earthquake
def data_ev_loc(ev_info):
    lat = []
    lon = []
    dep = []
    ortime = []
    for key in ev_info:
        lat.append(ev_info[key].lat)
        lon.append(ev_info[key].lon)
        dep.append(ev_info[key].dep)
        ortime.append(ev_info[key].ortime.timestamp)
    return [np.array(lon),np.array(lat),np.array(dep),np.array(ortime)]

# %%
# function: output the [lon,lat,dep,weight] of the station
def data_lon_lat_ele_wt_st(ev_info,st_info):
    names = {}
    lat = []
    lon = []
    ele = []
    weight  = []
    for key_ev in ev_info:
        for key_t in ev_info[key_ev].t:     # absolute traveltime data
            name_st = ev_info[key_ev].t[key_t][0]
            names[name_st] = name_st

        for key_t in ev_info[key_ev].cs_dt: # common source differential traveltime data
            name_st = ev_info[key_ev].cs_dt[key_t][0]
            names[name_st] = name_st
            name_st = ev_info[key_ev].cs_dt[key_t][1]
            names[name_st] = name_st

        for key_t in ev_info[key_ev].cr_dt: # common receiver differential traveltime data
            name_st = ev_info[key_ev].cr_dt[key_t][0]
            names[name_st] = name_st

    for name in names:  # only output the station which has data
        lat.append(st_info[name].lat)
        lon.append(st_info[name].lon)
        ele.append(st_info[name].ele)
        try:
            weight.append(st_info[name].tag["weight"])
        except:
            weight.append(1.0)
    return [np.array(lon),np.array(lat),np.array(ele),np.array(weight)]

# %%
# function: output the [dis,time] of all data
def data_dis_time(ev_info,st_info):
    all_dis = []
    all_time = []
    for key_ev in ev_info:
        lat_ev = ev_info[key_ev].lat
        lon_ev = ev_info[key_ev].lon
        dep_ev = ev_info[key_ev].dep
        for key_t in ev_info[key_ev].t:
            all_time.append(ev_info[key_ev].t[key_t][2])
            lat_st = st_info[ev_info[key_ev].t[key_t][0]].lat
            lon_st = st_info[ev_info[key_ev].t[key_t][0]].lon
            ele_st = st_info[ev_info[key_ev].t[key_t][0]].ele
            dis = math.sqrt(cal_dis(lat_ev,lon_ev,lat_st,lon_st)**2 + (dep_ev+ele_st/1000)**2)
            all_dis.append(dis)

    return [np.array(all_dis),np.array(all_time)]

# %%
# function: output the [epidis,time] of all data
def data_epidis_time(ev_info,st_info):
    all_dis = []
    all_time = []
    for key_ev in ev_info:
        lat_ev = ev_info[key_ev].lat
        lon_ev = ev_info[key_ev].lon
        for key_t in ev_info[key_ev].t:
            all_time.append(ev_info[key_ev].t[key_t][2])
            lat_st = st_info[ev_info[key_ev].t[key_t][0]].lat
            lon_st = st_info[ev_info[key_ev].t[key_t][0]].lon
            dis = cal_dis(lat_ev,lon_ev,lat_st,lon_st)**2
            all_dis.append(dis)

    return [np.array(all_dis),np.array(all_time)]

# %%
# function: output the [cs_dt] of all data
def data_cs_dt(ev_info):
    all_time = []
    for key_ev in ev_info:
        for key_dt in ev_info[key_ev].cs_dt:
            all_time.append(ev_info[key_ev].cs_dt[key_dt][3])

    return np.array(all_time)

# %%
# Function: data_dis_time_phase(ev_info, st_info, phase_list) Given a list of seismic phases, output the [epicentral distance, arrival time] for each phase.
def data_dis_time_phase(ev_info,st_info,phase_list):
    all_dis  = {}
    all_time = {}
    for phase in phase_list:
        all_dis[phase] = []
        all_time[phase] = []

    for key_ev in ev_info:
        lat_ev = ev_info[key_ev].lat
        lon_ev = ev_info[key_ev].lon
        dep_ev = ev_info[key_ev].dep
        for key_t in ev_info[key_ev].t:
            phase = key_t.split("+")[1]
            if (not phase in phase_list):
                continue

            all_time[phase].append(ev_info[key_ev].t[key_t][2])
            lat_st = st_info[ev_info[key_ev].t[key_t][0]].lat
            lon_st = st_info[ev_info[key_ev].t[key_t][0]].lon
            ele_st = st_info[ev_info[key_ev].t[key_t][0]].ele

            dis = math.sqrt(cal_dis(lat_ev,lon_ev,lat_st,lon_st)**2 + (dep_ev+ele_st/1000)**2)
            all_dis[phase].append(dis)

    for phase in phase_list:
        all_dis[phase] = np.array(all_dis[phase])
        all_time[phase] = np.array(all_time[phase])

    return [all_dis,all_time]

# %%
# Function: data_lon_lat_dep_wt_ev(ev_info) Outputs the lines connecting station and earthquake for traveltime data as [line_x, line_y].

def data_line(ev_info,st_info):
    line_x = []
    line_y = []

    for key_ev in ev_info:
        lat_ev = ev_info[key_ev].lat
        lon_ev = ev_info[key_ev].lon
        for key_t in ev_info[key_ev].t:
            lat_st = st_info[ev_info[key_ev].t[key_t][0]].lat
            lon_st = st_info[ev_info[key_ev].t[key_t][0]].lon

            line_x.append([lon_ev,lon_st])
            line_y.append([lat_ev,lat_st])

    return [line_x,line_y]

# %% [markdown]
# Functions: discard some data in ev_info and st_info based on selection criteria

# %%
# Function: limit_ev_region(ev_info, lat1, lat2, lon1, lon2, dep1, dep2) Delete the earthquakes that are out of the specified region.

def limit_ev_region(ev_info,lat_min,lat_max,lon_min,lon_max,dep_min,dep_max):
    count_delete = 0

    del_key_ev = []
    for key_ev in ev_info:
        ev = ev_info[key_ev]
        lat = ev.lat
        lon = ev.lon
        dep = ev.dep

        if (lat < min(lat_min,lat_max) or lat > max(lat_min,lat_max)    \
            or lon < min(lon_min,lon_max) or lon > max(lon_min,lon_max)  \
            or dep < min(dep_min,dep_max) or dep > max(dep_min,dep_max)):
            del_key_ev.append(key_ev)
            count_delete += 1

        del_key_t = []
        for key_t in ev_info[key_ev].cr_dt:
            name_ev2 = ev_info[key_ev].cr_dt[key_t][1]
            lat2 = ev_info[name_ev2].lat
            lon2 = ev_info[name_ev2].lon
            dep2 = ev_info[name_ev2].dep
            if (lat2 < min(lat_min,lat_max) or lat2 > max(lat_min,lat_max)    \
                or lon2 < min(lon_min,lon_max) or lon2 > max(lon_min,lon_max) \
                or dep2 < min(dep_min,dep_max) or dep2 > max(dep_min,dep_max)):

                del_key_t.append(key_t)

        for key_t in del_key_t:
            del ev_info[key_ev].cr_dt[key_t]

        ev_info[key_ev].Ncr_dt = len(ev_info[key_ev].cr_dt)

    for key_ev in del_key_ev:
        del ev_info[key_ev]

    print("delete %d events out of the region, now %d earthquakes are retained within the study region"%(count_delete,len(ev_info)))
    return ev_info

# %%
# Function: limit_st_region(ev_info, st_info, lat1, lat2, lon1, lon2) Delete the stations that are out of the specified region.

def limit_st_region(ev_info,st_info,lat1,lat2,lon1,lon2):

    for key_ev in ev_info:
        # delete the station out of the region in the absolute traveltime data
        del_key_t = []
        for key_t in ev_info[key_ev].t:
            name_st = ev_info[key_ev].t[key_t][0]
            lat_st = st_info[name_st].lat
            lon_st = st_info[name_st].lon
            if(lat_st < min(lat1,lat2) or lat_st > max(lat1,lat2) or lon_st < min(lon1,lon2) or lon_st > max(lon1,lon2)):
                del_key_t.append(key_t)

        for key_t in del_key_t:
            del ev_info[key_ev].t[key_t]
        ev_info[key_ev].Nt = len(ev_info[key_ev].t)

        # delete the station out of the region in the common source differential traveltime data
        del_key_t = []
        for key_t in ev_info[key_ev].cs_dt:
            name_st1 = ev_info[key_ev].cs_dt[key_t][0]
            lat_st1 = st_info[name_st1].lat
            lon_st1 = st_info[name_st1].lon

            name_st2 = ev_info[key_ev].cs_dt[key_t][1]
            lat_st2 = st_info[name_st2].lat
            lon_st2 = st_info[name_st2].lon
            if(lat_st1 < min(lat1,lat2) or lat_st1 > max(lat1,lat2) or lon_st1 < min(lon1,lon2) or lon_st1 > max(lon1,lon2) \
                or lat_st2 < min(lat1,lat2) or lat_st2 > max(lat1,lat2) or lon_st2 < min(lon1,lon2) or lon_st2 > max(lon1,lon2)):
                del_key_t.append(key_t)

        for key_t in del_key_t:
            del ev_info[key_ev].cs_dt[key_t]
        ev_info[key_ev].Ncs_dt = len(ev_info[key_ev].cs_dt)

        # delete the station out of the region in the common receiver differential traveltime data
        del_key_st = []
        for key_t in ev_info[key_ev].cr_dt:
            name_st = ev_info[key_ev].cr_dt[key_t][0]
            lat_st = st_info[name_st].lat
            lon_st = st_info[name_st].lon
            if(lat_st < min(lat1,lat2) or lat_st > max(lat1,lat2) or lon_st < min(lon1,lon2) or lon_st > max(lon1,lon2)):
                del_key_st.append(key_t)

        for key_t in del_key_st:
            del ev_info[key_ev].cr_dt[key_t]
        ev_info[key_ev].Ncr_dt = len(ev_info[key_ev].cr_dt)

    return ev_info



# %%
# Function: limit_epi_dis(ev_info, st_info, epi_dis1, epi_dis2) Delete the stations with epicentral distance in the range from epi_dis1 to epi_dis2.

def limit_epi_dis(ev_info,st_info,epi_dis1,epi_dis2):

    for key_ev in ev_info:
        ev = ev_info[key_ev]

        lat_ev = ev.lat
        lon_ev = ev.lon

        # delete the absolute traveltime data
        del_key_t = []
        for key_t in ev.t:
            stname = ev.t[key_t][0]
            lat_st = st_info[stname].lat
            lon_st = st_info[stname].lon
            dis = cal_dis(lat_ev, lon_ev, lat_st, lon_st)
            if (dis > epi_dis1 and dis < epi_dis2):
                del_key_t.append(key_t)
        for key_t in del_key_t:
            del ev.t[key_t]
        ev.Nt = len(ev.t)

        # delete the common source differential traveltime data
        del_key_t = []
        for key_t in ev.cs_dt:
            for i in range(2):
                stname = ev.t[key_t][i]
                lat_st = st_info[stname].lat
                lon_st = st_info[stname].lon
                dis = cal_dis(lat_ev, lon_ev, lat_st, lon_st)

                if (dis > epi_dis1 and dis < epi_dis2):
                    del_key_t.append(key_t)
                    break
        for key_t in del_key_t:
            del ev.cs_dt[key_t]
        ev.Ncs_dt = len(ev.cs_dt)

        # delete the common receiver differential traveltime data
        del_key_t = []
        for key_t in ev.cr_dt:
            stname = ev.cr_dt[key_t][0]
            lat_st = st_info[stname].lat
            lon_st = st_info[stname].lon
            dis = cal_dis(lat_ev, lon_ev, lat_st, lon_st)
            if (dis > epi_dis1 and dis < epi_dis2):
                del_key_t.append(key_t)

            lat_ev2 = ev_info[ev.cr_dt[key_t][1]].lat
            lon_ev2 = ev_info[ev.cr_dt[key_t][1]].lon
            dis = cal_dis(lat_ev2, lon_ev2, lat_st, lon_st)
            if (dis > epi_dis1 and dis < epi_dis2):
                del_key_t.append(key_t)

        for key_t in del_key_t:
            del ev.cr_dt[key_t]
        ev.Ncr_dt = len(ev.cr_dt)


        ev_info[key_ev] = ev

    return ev_info

# %%
# Function: limit_data_residual(ev_info, st_info, slope, intercept, up, down) Limit the data within the range defined by the line time = dis * slope + intercept and the bounds up and down.

# remove outliers, only retain data satisfying:     slope * dis + intercept + down < time < slope * dis + intercept + up
def limit_data_residual(ev_info,st_info,slope,intercept,up,down):
    for key_ev in ev_info:
        lat_ev = ev_info[key_ev].lat
        lon_ev = ev_info[key_ev].lon
        dep_ev = ev_info[key_ev].dep
        del_key_t = []
        for key_t in ev_info[key_ev].t:
            name_st = ev_info[key_ev].t[key_t][0]
            lat_st = st_info[name_st].lat
            lon_st = st_info[name_st].lon
            ele_st = st_info[name_st].ele
            dis = math.sqrt(cal_dis(lat_ev,lon_ev,lat_st,lon_st)**2 + (dep_ev+ele_st/1000)**2)
            residual = ev_info[key_ev].t[key_t][2] - (slope*dis+intercept)

            if (residual < down or residual > up):
                del_key_t.append(key_t)

        for key_t in del_key_t:
            del ev_info[key_ev].t[key_t]

    for key_ev in ev_info:
        ev_info[key_ev].Nt = len(ev_info[key_ev].t)

    return ev_info



# %%
# Function: limit_data_phase(ev_info, phase_list) Retain only the specified seismic phases.

def limit_data_phase(ev_info,phase_list):
    for key_ev in ev_info:
        # process the absolute traveltime data
        new_t = {}
        for key_t in ev_info[key_ev].t:
            phase = ev_info[key_ev].t[key_t][1]
            if phase in phase_list:
                new_t[key_t] = ev_info[key_ev].t[key_t]

        ev_info[key_ev].t = new_t
        ev_info[key_ev].Nt = len(ev_info[key_ev].t)

        # process the common source differential traveltime data
        new_t = {}
        for key_t in ev_info[key_ev].cs_dt:
            phase = ev_info[key_ev].cs_dt[key_t][2]
            phase = phase.split(",")[0]
            if phase in phase_list:
                new_t[key_t] = ev_info[key_ev].cs_dt[key_t]

        ev_info[key_ev].cs_dt = new_t
        ev_info[key_ev].Ncs_dt = len(ev_info[key_ev].cs_dt)

        # process the common receiver differential traveltime data
        new_t = {}
        for key_t in ev_info[key_ev].cr_dt:
            phase = ev_info[key_ev].cr_dt[key_t][2]
            phase = phase.split(",")[0]
            if phase in phase_list:
                new_t[key_t] = ev_info[key_ev].cr_dt[key_t]

        ev_info[key_ev].cr_dt = new_t
        ev_info[key_ev].Ncr_dt = len(ev_info[key_ev].cr_dt)

    return ev_info

# %%
# Function: limit_min_Nt(min_Nt_thd, ev_info) Delete the earthquakes with the number of data less than min_Nt_thd.

def limit_min_Nt(min_Nt_thd, ev_info):
    Nev = len(ev_info)

    del_key_ev = []
    for key_ev in ev_info:
        if(ev_info[key_ev].Nt < min_Nt_thd):
            del_key_ev.append(key_ev)

    for key_ev in del_key_ev:
        del ev_info[key_ev]

    print("Original data set has %d earthquakes, %d earthquakes are deleted, %d earthquakes are retained"%(Nev,len(del_key_ev),len(ev_info)))

    return ev_info

# %%
# Function: limit_azi_gap(gap_thd) Calculate the azimuthal gap for all events and delete events with a gap greater than gap_thd.
def limit_azi_gap(gap_thd,ev_info,st_info):
    Nev = len(ev_info)

    del_key_ev = []
    for key_ev in ev_info:
        ev = ev_info[key_ev]
        gap = cal_azi_gap(ev,st_info)
        if (gap > gap_thd):
            del_key_ev.append(key_ev)
        else:
            ev_info[key_ev].tag["azi_gap"] = gap
    for key_ev in del_key_ev:
        del ev_info[key_ev]

    print("Original data set has %d earthquakes, %d earthquakes are deleted, %d earthquakes are retained"%(Nev,len(del_key_ev),len(ev_info)))

    return ev_info

# Function: cal_azi_gap(ev, st_info) Calculate the azimuthal gap of a single earthquake.
def cal_azi_gap(ev,st_info):
    azi_all = []
    lat_ev = ev.lat
    lon_ev = ev.lon
    stlist = {}
    for key in ev.t:
        stname = ev.t[key][0]
        if (not stname in stlist):
            lat_st = st_info[stname].lat
            lon_st = st_info[stname].lon
            azi = cal_azimuth(lat_ev, lon_ev, lat_st, lon_st)
            azi_all.append(azi)
            stlist[stname] = 1

    azi_all.sort()
    if(len(azi_all) < 2):
        return 360.0
    else:
        gap = 0.0
        for i in range(len(azi_all)-1):
            gap = max(gap,azi_all[i+1] - azi_all[i])
        gap = max(gap,azi_all[0] + 360 - azi_all[-1])
        return gap



# %%
# Function: limit_earthquake_decluster_Nt(ev_info, dlat, dlon, ddep, Top_N) Divide the region into several subdomains, sort by the number of arrival times, and retain only the top Top_N earthquakes with the most arrival times in each box.
# option 3, declustering. Divide the region into several subdomains, retain the Top N earthquakes in terms of the number of arrival times in each subdomain.
def limit_earthquake_decluster_Nt(ev_info,dlat,dlon,ddep,Top_N):
    # subdivide earthquakes into different subdomains
    [ev_info,tag2name] = tag_event_cluster(dlat,dlon,ddep,ev_info)

    # sort earthquakes in the same subdomain
    # Sort the quality of earthquakes within each tag based on the number of arrivals.
    tag2name = sort_cluster_Nt(ev_info, tag2name)

    # only retain Top_N earthquakes in each subdomain
    # Within each tag, prioritize selecting the top Top_N earthquakes.
    [ev_info,tag2name] = limit_decluster(ev_info, tag2name,Top_N)

    return ev_info



# Function: tag_event_cluster(size_lat, size_lon, size_dep, ev_info) Subdivide the study area, assign each earthquake to a subregion, and place it in a tag.
def tag_event_cluster(size_lat,size_lon,size_dep,ev_info):
    tag2name = {}
    for key_ev in ev_info:
        name = ev_info[key_ev].name
        lat = ev_info[key_ev].lat
        lon = ev_info[key_ev].lon
        dep = ev_info[key_ev].dep
        tag = "%d_%d_%d"%(math.floor(lon/size_lon),math.floor(lat/size_lat),math.floor(dep/size_dep))
        ev_info[key_ev].tag["cluster"] = tag

        if (tag in tag2name):
            tag2name[tag].append(name)
        else:
            tag2name[tag] = []
            tag2name[tag].append(name)

    return [ev_info,tag2name]

# Function: sort_cluster_Nt(ev_info, tag2name) Sort the quality of earthquakes within each tag based on the number of arrivals.
def sort_cluster_Nt(ev_info, tag2name):
    for key_tag in tag2name:
        names_ev = tag2name[key_tag]
        Nt = []
        for key_ev in names_ev:
            Nt.append(len(ev_info[key_ev].t))

        # Sort the earthquakes within each tag based on the number of arrivals.
        sorted_Nt = sorted(enumerate(Nt), key=lambda x: x[1], reverse=True)
        tag2name[key_tag] = []
        for index, Nt in sorted_Nt:
            tag2name[key_tag].append(names_ev[index])

    return tag2name

# Function: limit_cluster(ev_info, tag2name, Max) Prioritize selecting the top Max earthquakes within each tag.
def limit_decluster(ev_info, tag2name, Max):
    del_key_ev = []
    for key_tag in tag2name:
        names_ev = tag2name[key_tag]

        if(len(names_ev) > Max):
            tag2name[key_tag] = names_ev[0:Max]
            for i in range(Max,len(names_ev)):    # Delete earthquakes that exceed the threshold in the sorted list.
                del_key_ev.append(names_ev[i])

    for key_ev in del_key_ev:
        del ev_info[key_ev]

    return [ev_info,tag2name]



# %% [markdown]
# Functions: assign weights to earthquakes, stations, and data

# %%
# Function: box_weighting_ev(ev_info, dlat, dlon, ddep) Assign box-weight to the earthquakes.
def box_weighting_ev(ev_info,dlon,dlat,ddep):

    # categorization
    distribute = {}
    all_tag_wt = {}

    for key_ev in ev_info:
        lat_id = math.floor((ev_info[key_ev].lat) / dlat)
        lon_id = math.floor((ev_info[key_ev].lon) / dlon)
        dep_id = math.floor((ev_info[key_ev].dep) / ddep)

        tag = '%d_%d_%d'%(lat_id,lon_id,dep_id)
        if (tag in distribute):
            distribute[tag] += 1
        else:
            distribute[tag] = 1

    max_weight = 0
    for tag in distribute:
        all_tag_wt[tag] = 1.0/math.sqrt(distribute[tag])
        max_weight = max(max_weight,all_tag_wt[tag])

    for key_ev in ev_info:
        lat_id = math.floor((ev_info[key_ev].lat) / dlat)
        lon_id = math.floor((ev_info[key_ev].lon) / dlon)
        dep_id = math.floor((ev_info[key_ev].dep) / ddep)

        tag = '%d_%d_%d'%(lat_id,lon_id,dep_id)

        ev_info[key_ev].tag["weight"] = all_tag_wt[tag]/max_weight

    return ev_info

# %%
# Function: geographical_weighting_ev_rough(ev_info, dlat, dlon, ddep) Assign geographical weighting to the earthquakes roughly.
def geographical_weighting_ev_rough(ev_info,dlat,dlon,ddep,coefficient = 0.5):

    # categorization
    distribute = {}
    all_tag_wt = {}

    for key_ev in ev_info:
        lat_id = int(ev_info[key_ev].lat/dlat)
        lon_id = int(ev_info[key_ev].lon/dlat)
        dep_id = int(ev_info[key_ev].dep/ddep)


        tag = '%d_%d_%d'%(lat_id,lon_id,dep_id)
        if (tag in distribute):
            distribute[tag] += 1
        else:
            distribute[tag] = 1

    # Calculate the weight of each category.
    delta0 = 0
    for tag1 in distribute:
        tmp1 = tag1.split('_')
        # evlat1 = float(tmp1[0])*dlat; evlon1 = float(tmp1[1])*dlon; evdep1 = float(tmp1[2])*ddep

        for tag2 in distribute:
            tmp2 = tag2.split('_')
            # evlat2 = float(tmp2[0])*dlat; evlon2 = float(tmp2[1])*dlon; evdep2 = float(tmp2[2])*ddep

            # distance of id
            delta_tp = math.sqrt((int(tmp1[0]) - int(tmp2[0]))**2 + (int(tmp1[1]) - int(tmp2[1]))**2 + (int(tmp1[2]) - int(tmp2[2]))**2)
            delta0 = delta0 + distribute[tag1] * distribute[tag2] * delta_tp

    delta0 = delta0/(len(ev_info)**2) * coefficient

    max_weight = 0.0
    for tag1 in distribute:
        tmp1 = tag1.split('_')
        # evlat1 = float(tmp1[0])*dlat; evlon1 = float(tmp1[1])*dlon; evdep1 = float(tmp1[2])*ddep

        weight = 0
        for tag2 in distribute:
            tmp2 = tag2.split('_')
            # evlat2 = float(tmp2[0])*dlat; evlon2 = float(tmp2[1])*dlon; evdep2 = float(tmp2[2])*ddep

            delta_tp = math.sqrt((int(tmp1[0]) - int(tmp2[0]))**2 + (int(tmp1[1]) - int(tmp2[1]))**2 + (int(tmp1[2]) - int(tmp2[2]))**2)
            weight = weight + math.exp(-(delta_tp/delta0)**2) * distribute[tag2]

        all_tag_wt[tag1] = (1.0/weight)
        max_weight = max(max_weight,1.0/weight)

    # Assign weights to each earthquake based on its tag.
    for key_ev in ev_info:
        lat_id = int(ev_info[key_ev].lat/dlat)
        lon_id = int(ev_info[key_ev].lon/dlon)
        dep_id = int(ev_info[key_ev].dep/ddep)

        tag = '%d_%d_%d'%(lat_id,lon_id,dep_id)

        ev_info[key_ev].tag["weight"] = all_tag_wt[tag]/max_weight
    return ev_info


# %%
# Function: box_weighting_st(ev_info, st_info, dlat, dlon) Assign geographical weighting to the stations roughly.
def box_weighting_st(ev_info,st_info,dlon,dlat):

    [lon_ev,lat_ev,dep_ev,wt_ev] = data_lon_lat_dep_wt_ev(ev_info)

    # Integrate all involved stations.
    wt_st = {}
    name_st = {}
    for key_ev in ev_info:
        for key_t in ev_info[key_ev].t:
            name_rec = ev_info[key_ev].t[key_t][0]
            wt_st[name_rec] = -1.0
            name_st[name_rec] = 1

    # categorization
    distribute = {}
    all_tag_wt = {}

    # Count the number of stations in each subdomain.
    for key_st in name_st:
        lat_id = math.floor((st_info[key_st].lat) / dlat)
        lon_id = math.floor((st_info[key_st].lon) / dlon)

        tag = '%d_%d'%(lat_id,lon_id)
        if (tag in distribute):
            distribute[tag] += 1
        else:
            distribute[tag] = 1

    max_weight = 0
    for tag in distribute:
        all_tag_wt[tag] = 1.0/math.sqrt(distribute[tag])
        max_weight = max(max_weight,all_tag_wt[tag])

    # Assign weights to each station based on its tag.
    for key_st in name_st:
        lat_id = math.floor((st_info[key_st].lat) / dlat)
        lon_id = math.floor((st_info[key_st].lon) / dlon)
        tag = '%d_%d'%(lat_id,lon_id)
        wt_st[key_st] = all_tag_wt[tag]/max_weight

    # modify weight tag in st_info
    for key_t in wt_st:
        st_info[key_t].tag["weight"] = wt_st[key_t]

    # modify weight of abs data ev_info
    for key_ev in ev_info:
        for key_t in ev_info[key_ev].t:
            name_rec = ev_info[key_ev].t[key_t][0]
            ev_info[key_ev].t[key_t][3] = wt_st[name_rec]

    return [ev_info,st_info]


# %%
# Function: geographical_weighting_st(ev_info,st_info) Assign geographical weighting to the stations roughly.
def geographical_weighting_st(ev_info,st_info,coefficient = 0.5):

    # Integrate all involved stations.
    wt_st = {}
    name_st = {}
    for key_ev in ev_info:
        for key_t in ev_info[key_ev].t:
            name_rec = ev_info[key_ev].t[key_t][0]
            wt_st[name_rec] = -1.0
            name_st[name_rec] = 1

    # Calculate the weight of each station.
    delta0 = 0
    for key_st1 in name_st:
        stlat1 = st_info[key_st1].lat
        stlon1 = st_info[key_st1].lon

        for key_st2 in name_st:
            stlat2 = st_info[key_st2].lat
            stlon2 = st_info[key_st2].lon

            delta_tp = cal_dis(stlat1,stlon1,stlat2,stlon2)
            delta0 = delta0 + delta_tp

    delta0 = delta0/(len(wt_st)**2)*coefficient

    max_weight = 0.0
    for key_st1 in name_st:
        stlat1 = st_info[key_st1].lat
        stlon1 = st_info[key_st1].lon

        weight = 0
        for key_st2 in name_st:
            stlat2 = st_info[key_st2].lat
            stlon2 = st_info[key_st2].lon

            delta_tp = cal_dis(stlat1,stlon1,stlat2,stlon2)
            weight = weight + math.exp(-(delta_tp/delta0)**2)

        wt_st[key_st1] = (1.0/weight)
        max_weight = max(max_weight,1.0/weight)

    for key_st1 in wt_st:
        wt_st[key_st1] = wt_st[key_st1]/max_weight

    # Add weight to each data point in the earthquakes.
    for key_ev in ev_info:
        for key_t in ev_info[key_ev].t:
            name_rec = ev_info[key_ev].t[key_t][0]
            if (not name_rec in wt_st):
                ValueError("The station of the data is not in the calculation list")

            if (len(ev_info[key_ev].t[key_t])==3):
                ev_info[key_ev].t[key_t].append(wt_st[name_rec])
            elif (len(ev_info[key_ev].t[key_t])==4):
                ev_info[key_ev].t[key_t][3] = wt_st[name_rec]
            else:
                ValueError("Error in the weight information of the absolute traveltime data")

    # modify weight tag in st_info
    for key_t in wt_st:
        st_info[key_t].tag["weight"] = wt_st[key_t]

    # modify weight of abs data ev_info
    for key_ev in ev_info:
        for key_t in ev_info[key_ev].t:
            name_rec = ev_info[key_ev].t[key_t][0]
            ev_info[key_ev].t[key_t][3] = wt_st[name_rec]

    return [ev_info,st_info]


# %% [markdown]
# Function: add noise into data

# %%
# Function：assign_gaussian_noise():
def assign_gaussian_noise(ev_info,sigma):

    # Record which seismic phases correspond to each station.
    st2phase = {}   # Station name -> [Keys of absolute arrival time data related to this station]


    for key_ev in ev_info:
        # Absolute arrival time noise
        for key_t in ev_info[key_ev].t:
            stname = ev_info[key_ev].t[key_t][0]
            ev_info[key_ev].t[key_t][2] = ev_info[key_ev].t[key_t][2] + np.random.normal(0,sigma)
            if(stname in st2phase):
                st2phase[stname].append(key_t)
            else:
                st2phase[stname] = [key_t]

    for key_ev in ev_info:
        # Double-difference arrival time noise
        for key_dt in ev_info[key_ev].cs_dt:
            stname1 = ev_info[key_ev].cs_dt[key_dt][0]
            stname2 = ev_info[key_ev].cs_dt[key_dt][1]
            t1 = -999
            t2 = -999
            # Search for the arrival time of the data.
            if (stname1 in st2phase):
                for key_t in st2phase[stname1]:
                    if (key_t in ev_info[key_ev].t):
                        t1 = ev_info[key_ev].t[key_t][2]
                        break
            if (stname2 in st2phase):
                for key_t in st2phase[stname2]:
                    if (key_t in ev_info[key_ev].t):
                        t2 = ev_info[key_ev].t[key_t][2]
                        break

            if (t1 == -999 or t2 == -999):
                # If there is no absolute arrival time data, the double-difference data residuals increase by a factor of sqrt(2) in noise.
                ev_info[key_ev].cs_dt[key_dt][3] = ev_info[key_ev].cs_dt[key_dt][3] + np.random.normal(0,sigma*np.sqrt(2))
                print('no data: ', key_ev, key_dt)
            else:
                # If there is absolute arrival time data, the double-difference data is obtained by subtraction.
                ev_info[key_ev].cs_dt[key_dt][3] = t1 - t2

        # Common station double-difference arrival time
        for key_dt in ev_info[key_ev].cr_dt:
            stname  = ev_info[key_ev].cr_dt[key_dt][0]
            key_ev2 = ev_info[key_ev].cr_dt[key_dt][1]

            t1 = -999
            t2 = -999
            # Search for the arrival time of the data.
            if (stname in st2phase):
                for key_t in st2phase[stname]:
                    if (key_t in ev_info[key_ev].t):
                        t1 = ev_info[key_ev].t[key_t][2]
                        break
                    else:
                        print('not found 1: ', key_ev, key_t)

                for key_t in st2phase[stname]:
                    if (key_t in ev_info[key_ev2].t):
                        t2 = ev_info[key_ev2].t[key_t][2]
                        break
                    else:
                        print('not found 2: ', key_ev, key_t)

            if (t1 == -999 or t2 == -999):
                # If there is no absolute arrival time data, the double-difference data residuals increase by a factor of sqrt(2) in noise.
                ev_info[key_ev].cr_dt[key_dt][3] = ev_info[key_ev].cr_dt[key_dt][3] + np.random.normal(0,sigma*np.sqrt(2))
                print('no data: ', key_ev, key_dt)
            else:
                # If there is absolute arrival time data, the double-difference data is obtained by subtraction.
                ev_info[key_ev].cr_dt[key_dt][3] = t1 - t2

    return ev_info

# %%
# Function：assign_uniform_noise_to_ev():
def assign_uniform_noise_to_ev(ev_info, range_lat, range_lon, range_dep, range_time):

    # Loop through all earthquakes and assign noise to them.
    ev_noise = {}   # Name of the earthquake -> noise of [lat,lon,dep,ortime]
    # loop list of earthquakes
    for key_ev in ev_info:
        evname = key_ev
        if (evname in ev_noise):
            print("error: repeated earthquake name")
            exit()
        else:
            # generate noise
            ev_noise[evname] = np.random.uniform(-1,1,4) * np.array([range_lat,range_lon,range_dep,range_time])

    # Add noise to each data point.
    for key_ev in ev_info:

        # Absolute arrival time noise
        for key_t in ev_info[key_ev].t:

            ev_info[key_ev].t[key_t][2] = ev_info[key_ev].t[key_t][2] - ev_noise[key_ev][3]


        # Double-difference arrival time noise (double-difference arrival time remains unchanged)

        # Common station double-difference arrival time
        for key_dt in ev_info[key_ev].cr_dt:
            key_ev2 = ev_info[key_ev].cr_dt[key_dt][1]

            if (key_ev2 in ev_noise):
                ev_info[key_ev].cr_dt[key_dt][3] = ev_info[key_ev].cr_dt[key_dt][3] - ev_noise[key_ev][3] + ev_noise[key_ev2][3]
            else:
                print("earthquake %s is not included in ev_list"%(key_ev2))
                ev_noise[key_ev2] = np.random.uniform(-1,1,4) * np.array([range_lat,range_lon,range_dep,range_time])
                ev_info[key_ev].cr_dt[key_dt][3] = ev_info[key_ev].cr_dt[key_dt][3] - ev_noise[key_ev][3] + ev_noise[key_ev2][3]


    # Add noise to each earthquake.
    for key_ev in ev_noise:
        ev_info[key_ev].lat = ev_info[key_ev].lat + ev_noise[key_ev][0]
        ev_info[key_ev].lon = ev_info[key_ev].lon + ev_noise[key_ev][1]
        ev_info[key_ev].dep = abs(ev_info[key_ev].dep + ev_noise[key_ev][2])
        ev_info[key_ev].ortime = ev_info[key_ev].ortime + ev_noise[key_ev][3]


    return ev_info

# %% [markdown]
# Functions: generate differential traveltime

# %%
# Function: generate_cs_dif(ev_info, st_info, dis_thd, azi_thd) Generate double-difference arrival times from absolute arrival times, with inter-station distance less than dis_thd and azimuthal difference less than azi_thd.
# function: generate common source differential traveltime data from absolute traveltime data, the stations separation is less than dis_thd, the azimuth difference is less than azi_thd
def generate_cs_dif(ev_info,st_info,dis_thd,azi_thd):
    count_t = 0
    count_cs_dt = 0

    for key_ev in ev_info:
        ev = ev_info[key_ev]

        lat_ev = ev.lat
        lon_ev = ev.lon

        # traverse all arrival times
        name_st_list = []   # names of stations
        t_list = []         # traveltime
        wt_list = []        # weight
        for key_t in ev.t:
            name_st_list.append(ev.t[key_t][0])
            t_list.append(ev.t[key_t][2])
            wt_list.append(ev.t[key_t][3])
            count_t += 1

        # search for possible double-difference arrival times
        for id_st1 in range(len(name_st_list)-1):
            name_st1 = name_st_list[id_st1]
            lat_st1  = st_info[name_st1].lat
            lon_st1  = st_info[name_st1].lon
            t_st1    = t_list[id_st1]
            wt_st1   = wt_list[id_st1]

            for id_st2 in range(id_st1+1,len(name_st_list)):
                name_st2 = name_st_list[id_st2]
                lat_st2  = st_info[name_st2].lat
                lon_st2  = st_info[name_st2].lon
                t_st2    = t_list[id_st2]
                wt_st2   = wt_list[id_st2]

                dis = cal_dis(lat_st1,lon_st1,lat_st2,lon_st2)
                azi_st1 = cal_azimuth(lat_ev,lon_ev,lat_st1,lon_st1)
                azi_st2 = cal_azimuth(lat_ev,lon_ev,lat_st2,lon_st2)

                azi_dif = abs(azi_st1 - azi_st2)

                if(dis < dis_thd and (azi_dif < azi_thd or (360-azi_dif) < azi_thd )):
                    ev.cs_dt["%s+%s+%s"%(name_st1,name_st2,"P,cs")] = [name_st1,name_st2,"P,cs",t_st1-t_st2,(wt_st1+wt_st2)/2]
                    count_cs_dt += 1

        ev_info[key_ev].Ncs_dt = len(ev.cs_dt)

    print('we generate %d common source differential traveltimes from %s absolute traveltimes'%(count_cs_dt,count_t))
    return ev_info


# %%
# Function: generate_cr_dif(ev_info, st_info, dis_thd) Generate common station double-difference arrival times from absolute arrival times, with inter-event distance less than dis_thd.
# Function: generate common receiver differential traveltime data from absolute traveltime data, the earthquake separation is less than dis_thd
def generate_cr_dif(ev_info,st_info,dis_thd,azi_thd):

    # Construct mapping：rec2src[name_ev] -> {name_st: [name_ev, name_st, t, wt]; name_st: [name_ev, name_st, t, wt]; ...}
    rec2src = build_rec_src_map(ev_info,dis_thd)
    print("rec to src map generation finished")

    # Construct double-difference data association mapping：rec2src_pair[key_t]
    rec2src_pair = build_rec_src_pair_map(rec2src)
    print("rec to src_pair map generation finished")

    for key_t in rec2src_pair:
        name_st = key_t.split('+')[0]
        lat_st = st_info[name_st].lat
        lon_st = st_info[name_st].lon

        for ev_tag in rec2src_pair[key_t]:
            name_ev1 = rec2src_pair[key_t][ev_tag][0]
            lat_ev1  = ev_info[name_ev1].lat
            lon_ev1  = ev_info[name_ev1].lon
            dep_ev1  = ev_info[name_ev1].dep

            name_ev2 = rec2src_pair[key_t][ev_tag][1]
            lat_ev2  = ev_info[name_ev2].lat
            lon_ev2  = ev_info[name_ev2].lon
            dep_ev2  = ev_info[name_ev2].dep

            dis_xy = cal_dis(lat_ev1,lon_ev1,lat_ev2,lon_ev2)
            dis_z  = abs(dep_ev1 - dep_ev2)
            dis = math.sqrt(dis_xy**2 + dis_z**2)
            if(dis > dis_thd):      # limit of the distance between two earthquakes
                continue

            azi1 = cal_azimuth(lat_ev1,lon_ev1,lat_st,lon_st)
            azi2 = cal_azimuth(lat_ev2,lon_ev2,lat_st,lon_st)
            azi_dif = abs(azi1 - azi2)

            if(azi_dif > azi_thd and (360-azi_dif) > azi_thd):     # limit of the azimuth difference between two earthquakes
                continue

            t_ev1 = ev_info[name_ev1].t[key_t][2]
            t_ev2 = ev_info[name_ev2].t[key_t][2]
            wt_ev1 = ev_info[name_ev1].t[key_t][3] * ev_info[name_ev1].tag["weight"]
            wt_ev2 = ev_info[name_ev2].t[key_t][3] * ev_info[name_ev2].tag["weight"]
            # The actual data weight is wt_ev1 + wt_ev2, but in TomoATT calculations, we need to divide it by ev_info[name_ev1].tag["weight"].
            wt = (wt_ev1 + wt_ev2)/2/ev_info[name_ev1].tag["weight"]

            ev_info[name_ev1].cr_dt["%s+%s+%s"%(name_st,name_ev2,"P,cr")] = [name_st,name_ev2,"P,cr",t_ev1-t_ev2,wt]

    # Count the number of double-difference data points.
    count_cr_dt = 0
    count_t     = 0
    for key_ev in ev_info:
        ev_info[key_ev].Ncr_dt = len(ev_info[key_ev].cr_dt)
        count_cr_dt += ev_info[key_ev].Ncr_dt
        count_t     += ev_info[key_ev].Nt

    print('we generate %d common receiver differential traveltimes from %s absolute traveltimes'%(count_cr_dt,count_t))

    return ev_info

# Construct mapping: rec2src = {key_t: dict_tag; key_t: dict_tag; ...}
# dict_tag = {tag: list_name_ev; tag: list_name_ev; ...}
# list_name_ev = [name_ev1, name_ev2, ...]
# Assign earthquakes to different subregions based on their locations. The subregion size is dlat * dlon * ddep. When performing common station double-difference calculations, only earthquake pairs within the same subregion or adjacent subregions will be considered.
def build_rec_src_map(ev_info,dis_thd):
    rec2src = {}
    for key_ev in ev_info:
        name_ev = ev_info[key_ev].name
        lat = ev_info[key_ev].lat
        lon = ev_info[key_ev].lon
        dep = ev_info[key_ev].dep
        tag_dep = math.floor(dep/dis_thd)
        tag_lat = math.floor(lat/180*math.pi*R_earth/dis_thd)
        tag_lon = math.floor(lon/180*math.pi*R_earth*math.cos(lat)/dis_thd)
        tag = "%d_%d_%d"%(tag_lon,tag_lat,tag_dep)


        for key_t in ev_info[key_ev].t:

            # create dictionary
            if (not key_t in rec2src):
                rec2src[key_t] = {tag:[]}
            elif (not tag in rec2src[key_t]):
                rec2src[key_t][tag] = []

            # Add data
            rec2src[key_t][tag].append(name_ev)

    return rec2src

# Function: generate_adjacent_tag(tag) Generate tags surrounding the given tag.
def generate_adjacent_tag(tag): # Excluding the tag itself.
    adjacent_tag_list = []
    tmp = tag.split('_')
    tag_lon = int(tmp[0])
    tag_lat = int(tmp[1])
    tag_dep = int(tmp[2])

    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if(i == 0 and j == 0 and k == 0):
                    continue
                adjacent_tag_list.append("%d_%d_%d"%(tag_lon+i,tag_lat+j,tag_dep+k))

    return adjacent_tag_list


# construct mapping：rec2src_pair
def build_rec_src_pair_map(rec2src):
    rec2src_pair = {}

    for key_t in rec2src:
        rec2src_pair[key_t] = {}

        for tag in rec2src[key_t]:
            name_ev_list1 = rec2src[key_t][tag]

            name_ev_list2 = rec2src[key_t][tag]
            adjacent_tag_list = generate_adjacent_tag(tag)
            for adjacent_tag in adjacent_tag_list:
                if (adjacent_tag in rec2src[key_t]):      # If the surrounding tag's region has earthquakes, add them to the earthquake list.
                    name_ev_list2 = name_ev_list2 + rec2src[key_t][adjacent_tag]

            # Find possible earthquake pairs.
            for id_ev1 in range(len(name_ev_list1)-1):
                name_ev1 = name_ev_list1[id_ev1]

                for id_ev2 in range(id_ev1+1,len(name_ev_list2)):   # Starting from id_ev1 + 1 already excludes duplicate earthquakes within the tag.
                    name_ev2 = name_ev_list2[id_ev2]

                    ev_tag1 = "%s+%s"%(name_ev1,name_ev2)
                    ev_tag2 = "%s+%s"%(name_ev2,name_ev1)

                    if(ev_tag1 in rec2src_pair[key_t] or ev_tag2 in rec2src_pair[key_t]):
                        continue

                    rec2src_pair[key_t][ev_tag1] = [name_ev1,name_ev2]


    return rec2src_pair

# %% [markdown]
# Functions: read and write src_rec.dat file

# %%
# Function: reorder_src(ev) Reorder the earthquake IDs. If the earthquake has no data, the ID is -999.
def reorder_src(ev_info):

    ev_id = 0
    for key_ev in ev_info:
        ev      = ev_info[key_ev]

        if(ev.Nt + ev.Ncs_dt + ev.Ncr_dt  == 0):
            ev.id = -999
        else:
            ev_info[key_ev].id = ev_id
            ev_id += 1

    return ev_info


# %%
# Function: read_src_rec_file(fname) Read the src_rec.dat file.
#
def read_src_rec_file(fname):
    ev_info = {}
    st_info = {}

    tmp_ev_info = {}

    doc = open(fname,'r')
    doc_input = doc.readlines()
    doc.close()

    cc = 0
    for info in doc_input:
        tmp=info.split()
        if (cc == 0):   # event line
            ev = Event()
            #  1 2000  1  2 20 28  37.270   38.2843     39.0241  11.00  3.60    8   1725385
            # id_ev   = int(tmp[0])
            ev.id   = int(tmp[0])
            year    = int(tmp[1])
            month   = int(tmp[2])
            day     = int(tmp[3])
            hour    = int(tmp[4])
            minute  = int(tmp[5])
            second  = float(tmp[6])
            ev.ortime = UTCDateTime(year,month,day,hour,minute,0) + second
            ev.lat  = float(tmp[7])
            ev.lon  = float(tmp[8])
            ev.dep  = float(tmp[9])
            ev.mag  = float(tmp[10])
            ev.Nt   = 0
            ev.Ncs_dt  = 0
            ev.Ncr_dt  = 0
            ev.t    = {}
            ev.cs_dt   = {}
            ev.cr_dt   = {}
            ndata   = int(tmp[11])
            name_ev = tmp[12]
            ev.name = name_ev
            cc += 1
            try:
                ev.tag["weight"] = float(tmp[13])
            except:
                pass

            if (ndata == 0):
                cc = 0
                ev_info[name_ev] = ev

        else:   # data line
            #  1      1      MYA       38.3261     38.4253   1050.0000  P   52.46   6.630 weight
            if(len(tmp) < 10):  # absolue traveltime data
                name_st = tmp[2]
                phase   = tmp[6]
                if (phase == "PG"):
                    phase = "Pg"
                if (phase == "PB"):
                    phase = "Pb"
                if (phase == "PN"):
                    phase = "Pn"

                if (not name_st in st_info):
                    st = Station()
                    st.name = name_st
                    st.id   = float(tmp[1])
                    st.lat  = float(tmp[3])
                    st.lon  = float(tmp[4])
                    st.ele  = float(tmp[5])
                    st_info[name_st] = st

                time    = float(tmp[7])
                if(len(tmp) == 9):
                    weight = float(tmp[8])
                else:
                    weight = 1.0
                ev.t["%s+%s"%(name_st,phase)] = [name_st,phase,time,weight]
                ev.Nt += 1

            else:   # differential traveltime data
                phase = tmp[11]
                if (phase.__contains__("cr")):  # common receiver differential traveltime
                    #  evid     stid1      stname1      lat1 lon1 eve1 evid2 evname2 lat2 lon2 dep2 phase,cr diftime weight

                    name_st1 = tmp[2]
                    if (not name_st1 in st_info):   # add station to the station list
                        st = Station()
                        st.name = name_st1
                        st.id   = float(tmp[1])
                        st.lat  = float(tmp[3])
                        st.lon  = float(tmp[4])
                        st.ele  = float(tmp[5])
                        st_info[name_st1] = st

                    name_ev2 = tmp[7]
                                                    # add earthquake to the temp earthquake list
                    ev2 = Event()
                    ev2.name = name_ev2
                    ev2.id   = float(tmp[6])
                    ev2.lat  = float(tmp[8])
                    ev2.lon  = float(tmp[9])
                    ev2.dep  = float(tmp[10])
                    tmp_ev_info[name_ev2] = ev2


                    dif_time = float(tmp[12])
                    if(len(tmp) == 14):
                        weight = float(tmp[13])
                    else:
                        weight = 1.0
                    ev.cr_dt["%s+%s+%s"%(name_st1,name_ev2,phase)] = [name_st1,name_ev2,phase,dif_time,weight]
                    ev.Ncr_dt += 1

                else:                           # common source differential traveltime
                    #  evid     stid1      stname1      lat1 lon1 eve1 stid2 stname2 lat2 lon2 ele2 phase,cs diftime weight

                    name_st1 = tmp[2]
                    if (not name_st1 in st_info):
                        st = Station()
                        st.name = name_st1
                        st.id   = len(st_info)
                        st.lat  = float(tmp[3])
                        st.lon  = float(tmp[4])
                        st.ele  = float(tmp[5])
                        st_info[name_st1] = st

                    name_st2 = tmp[7]
                    if (not name_st2 in st_info):
                        st = Station()
                        st.name = name_st2
                        st.id   = float(tmp[6])
                        st.lat  = float(tmp[8])
                        st.lon  = float(tmp[9])
                        st.ele  = float(tmp[10])
                        st_info[name_st2] = st

                    dif_time = float(tmp[12])
                    if(len(tmp) == 14):
                        weight = float(tmp[13])
                    else:
                        weight = 1.0
                    ev.cs_dt["%s+%s+%s"%(name_st1,name_st2,phase)] = [name_st1,name_st2,phase,dif_time,weight]
                    ev.Ncs_dt += 1

            if (cc == ndata):   # end of the event data
                cc = 0
                ev_info[name_ev] = ev
            else:
                cc += 1

    # Add earthquakes from the temporary earthquake list to the main earthquake list.
    for key_ev in tmp_ev_info:
        if (not key_ev in ev_info):
            ev_info[key_ev] = tmp_ev_info[key_ev]

    return [ev_info,st_info]

# %%
# Function: write_src_rec_file(fname, ev_info, st_info) Output the src_rec.dat file.
def write_src_rec_file(fname,ev_info,st_info):
    ev_info = reorder_src(ev_info)
    doc_src_rec = open(fname,'w')

    min_lat =  9999
    max_lat = -9999
    min_lon =  9999
    max_lon = -9999
    min_dep =  9999
    max_dep = -9999

    record_ev = {}
    record_st = {}
    Nt_total  = 0
    Ncs_dt_total = 0
    Ncr_dt_total = 0

    for key_ev in ev_info:
        ev      = ev_info[key_ev]
        evid    = ev.id
        year    = ev.ortime.year
        month   = ev.ortime.month
        day     = ev.ortime.day
        hour    = ev.ortime.hour
        minute  = ev.ortime.minute
        second  = ev.ortime.second
        msec    = ev.ortime.microsecond
        lat_ev  = ev.lat
        lon_ev  = ev.lon
        dep_ev  = ev.dep
        mag     = ev.mag
        ndata   = ev.Nt + ev.Ncs_dt + ev.Ncr_dt
        name_ev = ev.name
        try:
            weight_ev = ev.tag["weight"]
        except:
            weight_ev = 1.0

        if(ndata == 0):     # if the earthquake has no data, do not output it
            continue

        doc_src_rec.write('%7d %6d %2d %2d %2d %2d %5.2f %9.4f %9.4f %9.4f %5.2f %7d %s %7.3f\n'%(\
            evid,year,month,day,hour,minute,second+msec/1000000,lat_ev,lon_ev,dep_ev,mag,ndata,name_ev,weight_ev))

        min_lat =  min(min_lat, lat_ev)
        max_lat =  max(max_lat, lat_ev)
        min_lon =  min(min_lon, lon_ev)
        max_lon =  max(max_lon, lon_ev)
        min_dep =  min(min_dep, dep_ev)
        max_dep =  max(max_dep, dep_ev)

        record_ev[name_ev] = 1  # record this earthquake
        Nt_total += ev.Nt
        Ncs_dt_total += ev.Ncs_dt
        Ncr_dt_total += ev.Ncr_dt

        for key_t in ev.t:
            data    = ev.t[key_t]
            st      = st_info[data[0]]
            stid    = st.id
            name_st = st.name
            lat_st  = st.lat
            lon_st  = st.lon
            ele_st  = st.ele
            phase   = data[1]
            time    = data[2]
            try:
                weight_data = data[3]
            except:
                weight_data = 1.0
            doc_src_rec.write('%7d %7d %6s %9.4f %9.4f %9.4f %s %8.4f %7.3f \n'%(evid,stid,name_st,lat_st,lon_st,ele_st,phase,time,weight_data))

            min_lat =  min(min_lat, lat_st)
            max_lat =  max(max_lat, lat_st)
            min_lon =  min(min_lon, lon_st)
            max_lon =  max(max_lon, lon_st)
            min_dep =  min(min_dep, -ele_st/1000)
            max_dep =  max(max_dep, -ele_st/1000)

            record_st[name_st] = 1  # record this station

        for key_t in ev.cs_dt:
            data    = ev.cs_dt[key_t]
            st1     = st_info[data[0]]
            stid1   = st1.id
            name_st1= st1.name
            lat_st1 = st1.lat
            lon_st1 = st1.lon
            ele_st1 = st1.ele
            st2     = st_info[data[1]]
            stid2   = st2.id
            name_st2= st2.name
            lat_st2 = st2.lat
            lon_st2 = st2.lon
            ele_st2 = st2.ele
            phase   = data[2]
            time    = data[3]
            try:
                weight_data = data[4]
            except:
                weight_data = 1.0
            doc_src_rec.write('%7d %7d %6s %9.4f %9.4f %9.4f %7d %6s %9.4f %9.4f %9.4f %s %8.4f %7.3f \n'%(\
                evid,stid1,name_st1,lat_st1,lon_st1,ele_st1,stid2,name_st2,lat_st2,lon_st2,ele_st2,phase,time,weight_data))

            min_lat =  min(min_lat, lat_st1)
            max_lat =  max(max_lat, lat_st1)
            min_lon =  min(min_lon, lon_st1)
            max_lon =  max(max_lon, lon_st1)
            min_dep =  min(min_dep, -ele_st1/1000)
            max_dep =  max(max_dep, -ele_st1/1000)

            min_lat =  min(min_lat, lat_st2)
            max_lat =  max(max_lat, lat_st2)
            min_lon =  min(min_lon, lon_st2)
            max_lon =  max(max_lon, lon_st2)
            min_dep =  min(min_dep, -ele_st2/1000)
            max_dep =  max(max_dep, -ele_st2/1000)

            record_st[name_st1] = 1  # record this station
            record_st[name_st2] = 1  # record this station

        for key_t in ev.cr_dt:
            data    = ev.cr_dt[key_t]
            st      = st_info[data[0]]
            stid    = st.id
            name_st = st.name
            lat_st  = st.lat
            lon_st  = st.lon
            ele_st  = st.ele
            ev2     = ev_info[data[1]]
            evid2   = ev2.id
            name_ev2= ev2.name
            lat_ev2 = ev2.lat
            lon_ev2 = ev2.lon
            dep_ev2 = ev2.dep
            phase   = data[2]
            time    = data[3]
            try:
                weight_data = data[4]
            except:
                weight_data = 1.0
            doc_src_rec.write('%7d %7d %6s %9.4f %9.4f %9.4f %7d %6s %9.4f %9.4f %9.4f %s %8.4f %7.3f \n'%(\
                evid,stid,name_st,lat_st,lon_st,ele_st,evid2,name_ev2,lat_ev2,lon_ev2,dep_ev2,phase,time,weight_data))

            min_lat =  min(min_lat, lat_st)
            max_lat =  max(max_lat, lat_st)
            min_lon =  min(min_lon, lon_st)
            max_lon =  max(max_lon, lon_st)
            min_dep =  min(min_dep, -ele_st/1000)
            max_dep =  max(max_dep, -ele_st/1000)

            min_lat =  min(min_lat, lat_ev2)
            max_lat =  max(max_lat, lat_ev2)
            min_lon =  min(min_lon, lon_ev2)
            max_lon =  max(max_lon, lon_ev2)
            min_dep =  min(min_dep, dep_ev2)
            max_dep =  max(max_dep, dep_ev2)

            record_ev[name_ev2] = 1  # record this station
            record_st[name_st] = 1   # record this station

    doc_src_rec.close()

    print("src_rec.dat has been outputed: %d events, %d stations, %d abs traveltime, %d cs_dif traveltime, %d cr_dif traveltime. " \
          %(len(record_ev),len(record_st),Nt_total,Ncs_dt_total,Ncr_dt_total))
    print("earthquake and station region, lat: %6.1f - %6.1f, lon: %6.1f - %6.1f, dep: %6.1f - %6.1f"%(min_lat,max_lat,min_lon,max_lon,min_dep,max_dep)  )


# %%
# Function: write_src_list_file(fname, ev_info) Output the event list file.
def write_src_list_file(fname,ev_info):
    doc_ev_list = open(fname,'w')

    for key_ev in ev_info:
        ev      = ev_info[key_ev]
        evid    = ev.id
        lat_ev  = ev.lat
        lon_ev  = ev.lon
        dep_ev  = ev.dep
        mag     = ev.mag
        name_ev = ev.name
        if (ev.id == -999):     # if the earthquake has no data, do not output it
            continue
        doc_ev_list.write("%7d %s %s %9.4f %9.4f %9.4f %5.2f \n"%(evid,name_ev,ev.ortime,lat_ev,lon_ev,dep_ev,mag))
    doc_ev_list.close()

# %%
# Function: write_rec_list_file(fname, ev_info, st_info) Output the station list file.
def write_rec_list_file(fname,ev_info,st_info):
    doc_st_list = open(fname,'w')

    st_list = {}
    for key_ev in ev_info:
        ev      = ev_info[key_ev]

        for key_t in ev.t:
            data    = ev.t[key_t]
            st      = st_info[data[0]]
            name_st = st.name
            lat_st  = st.lat
            lon_st  = st.lon
            ele_st  = st.ele
            if(not name_st in st_list):
                doc_st_list.write("%6s %9.4f %9.4f %10.4f \n"%(name_st,lat_st,lon_st,ele_st))
                st_list[name_st] = 1

        for key_t in ev.cs_dt:
            data    = ev.cs_dt[key_t]
            st1     = st_info[data[0]]
            name_st1= st1.name
            lat_st1 = st1.lat
            lon_st1 = st1.lon
            ele_st1 = st1.ele
            st2     = st_info[data[0]]
            name_st2= st2.name
            lat_st2 = st2.lat
            lon_st2 = st2.lon
            ele_st2 = st2.ele
            if(not name_st1 in st_list):
                doc_st_list.write("%6s %9.4f %9.4f %10.4f \n"%(name_st1,lat_st1,lon_st1,ele_st1))
                st_list[name_st1] = 1
            if(not name_st2 in st_list):
                doc_st_list.write("%6s %9.4f %9.4f %10.4f \n"%(name_st2,lat_st2,lon_st2,ele_st2))
                st_list[name_st2] = 1

        for key_t in ev.cr_dt:
            data    = ev.cr_dt[key_t]
            st      = st_info[data[0]]
            name_st = st.name
            lat_st  = st.lat
            lon_st  = st.lon
            ele_st  = st.ele
            if(not name_st in st_list):
                doc_st_list.write("%6s %9.4f %9.4f %10.4f \n"%(name_st,lat_st,lon_st,ele_st))
                st_list[name_st] = 1

    doc_st_list.close()

# %% [markdown]
# Functions: read objective function file

# %%
# Function: read_objective_function_file(path)
def read_objective_function_file(path):

    full_curve = []
    location_curve = []
    model_curve = []

    with open('%s/objective_function.txt'%(path)) as f:
        for i,line in enumerate(f):
            tmp = line.split(',')
            if (tmp[0].__contains__("#")):
                continue   # skip the comment line
        
            iter        = int(tmp[0])
            tag         = tmp[1]
            obj         = float(tmp[2])
            obj_abs     = float(tmp[3])
            obj_cs      = float(tmp[4])
            obj_cr      = float(tmp[5])
            obj_tele    = float(tmp[6])
            tmp2        = tmp[7].split('/')
            mean        = float(tmp2[0])
            std         = float(tmp2[1])
            tmp2        = tmp[8].split('/')
            mean_abs    = float(tmp2[0])
            std_abs     = float(tmp2[1])
            tmp2        = tmp[9].split('/')
            mean_cs     = float(tmp2[0])
            std_cs      = float(tmp2[1])
            tmp2        = tmp[10].split('/')
            mean_cr     = float(tmp2[0])
            std_cr      = float(tmp2[1])
            tmp2        = tmp[11].split('/')
            mean_tele   = float(tmp2[0])
            std_tele    = float(tmp2[1])
            try:
                step_length = float(tmp[12])
            except:
                step_length = np.nan

            full_curve.append([obj,obj_abs,obj_cs,obj_cr,obj_tele,mean,std,mean_abs,std_abs,mean_cs,std_cs,mean_cr,std_cr,mean_tele,std_tele,step_length])
            if tag.__contains__("relocation"):
                location_curve.append([obj,obj_abs,obj_cs,obj_cr,obj_tele,mean,std,mean_abs,std_abs,mean_cs,std_cs,mean_cr,std_cr,mean_tele,std_tele,step_length])
            if tag.__contains__("model"):
                model_curve.append([obj,obj_abs,obj_cs,obj_cr,obj_tele,mean,std,mean_abs,std_abs,mean_cs,std_cs,mean_cr,std_cr,mean_tele,std_tele,step_length])
    return np.array(full_curve),np.array(location_curve),np.array(model_curve)

# %% [markdown]
# Functions: read inversion grid file

# %%
# Function: read the inversion grid file
def read_inversion_grid_file(path):

    inv_grid_vel = []
    inv_grid_ani = []

    switch = False
    igrid = -1
    with open('%s/inversion_grid.txt'%(path)) as f:
        tmp_inv_grid = []
        for i,line in enumerate(f):

            # read the number of inversion grid in dep, lat, lon directions
            if(i==0):
                tmp = line.split()
                ndep = int(tmp[1])
                nlines = 3*ndep+1   # The number of rows for each inversion grid is 3*ndep+1

            iline = i % nlines

            if(iline == 0):    # info: number of inversion grid
                tmp = line.split()
                if (int(tmp[0]) > igrid):
                    igrid = int(tmp[0])
                else:   # change from vel to ani
                    switch = True
                    igrid = int(tmp[0])

            else:               # info location of inversion grid
                iline_sub = (iline-1) % 3
                if(iline_sub == 0): # dep
                    tmp = line.split()
                    dep = float(tmp[0])
                if(iline_sub == 1): # list of lat
                    lat_list = line.split()
                if(iline_sub == 2): # list of lon
                    lon_list = line.split()

                    # add inversion grid
                    for lat in lat_list:
                        for lon in lon_list:
                            tmp_inv_grid.append([float(lon), float(lat), dep])

                if(iline == nlines-1): # the last line of inversion grid
                    if(switch):
                        inv_grid_ani.append(tmp_inv_grid)
                    else:
                        inv_grid_vel.append(tmp_inv_grid)
                    tmp_inv_grid = []

    return [np.array(inv_grid_vel),np.array(inv_grid_ani)]

# %% [markdown]
# Functions: for plotting

# %%
# Function: fig_ev_st_distribution_dep(ev_info, st_info) Plot the distribution of the earthquakes and stations, color-coded by earthquake depth.
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def fig_ev_st_distribution_dep(ev_info,st_info):

    [lon_ev,lat_ev,dep_ev,wt_ev] = data_lon_lat_dep_wt_ev(ev_info)
    [lon_st,lat_st,ele_st,wt_st] = data_lon_lat_ele_wt_st(ev_info,st_info)

    min_lon = min(min(lon_ev),min(lon_st))
    max_lon = max(max(lon_ev),max(lon_st))

    min_lat = min(min(lat_ev),min(lat_st))
    max_lat = max(max(lat_ev),max(lat_st))

    max_dep = max(dep_ev)

    # Insert a value that does not affect the plot to make the colorbar range look better.
    lon_ev = np.insert(lon_ev, 0, 9999); lat_ev = np.insert(lat_ev, 0, 9999); dep_ev = np.insert(dep_ev, 0, 0);

    fig = plt.figure(figsize=(12,12))
    gridspace = GridSpec(12,12,figure = fig)

    xrange = max_lon - min_lon + 1.0
    yrange = max_lat - min_lat + 1.0

    if (xrange > yrange):
        fig_x_size = 6
        fig_y_size = round(6*yrange/xrange)
    else:
        fig_x_size = round(6*xrange/yrange)
        fig_y_size = 6


    ax1 = fig.add_subplot(gridspace[0:fig_y_size,0:fig_x_size])

    bar_ev = ax1.scatter(lon_ev,lat_ev,c=dep_ev,cmap="jet",label = "src",s = 3)
    # ax1.plot(lon_st,lat_st,'rv',label = "rec",markersize = 6)
    bar_st = ax1.scatter(lon_st,lat_st,c="red",label = "rec",s = 100,marker='v',edgecolors='white')

    ax1.legend(fontsize = 14)
    ax1.tick_params(axis='x',labelsize=18)
    ax1.tick_params(axis='y',labelsize=18)
    ax1.set_xlabel('Lon',fontsize=18)
    ax1.set_ylabel('Lat',fontsize=18)
    ax1.set_xlim((min_lon - (max_lon - min_lon)*0.1,max_lon + (max_lon - min_lon)*0.1))
    ax1.set_ylim((min_lat - (max_lat - min_lat)*0.1,max_lat + (max_lat - min_lat)*0.1))


    ax2 = fig.add_subplot(gridspace[0:fig_y_size, fig_x_size+1 : fig_x_size+3])

    ax2.scatter(dep_ev,lat_ev,c=dep_ev,cmap="jet",label = "src",s = 3)

    ax2.tick_params(axis='x',labelsize=18)
    ax2.tick_params(axis='y',labelsize=18)
    ax2.set_xlabel('Dep',fontsize=18)
    ax2.set_ylabel('Lat',fontsize=18)
    ax2.set_xlim((-max_dep*0.05,max_dep*1.1))
    ax2.set_ylim((min_lat - (max_lat - min_lat)*0.1,max_lat + (max_lat - min_lat)*0.1))


    ax3 = fig.add_subplot(gridspace[fig_y_size+1:fig_y_size+3,0:fig_x_size])

    ax3.scatter(lon_ev,dep_ev,c=dep_ev,cmap="jet",label = "src",s = 3)

    ax3.tick_params(axis='x',labelsize=18)
    ax3.tick_params(axis='y',labelsize=18)
    ax3.set_xlabel('Lon',fontsize=18)
    ax3.set_ylabel('Dep',fontsize=18)
    ax3.set_xlim((min_lon - (max_lon - min_lon)*0.1,max_lon + (max_lon - min_lon)*0.1))
    ax3.set_ylim((-max_dep*0.05,max_dep*1.1))
    ax3.invert_yaxis()

    # Place the colorbar on a new axis.
    ax4 = fig.add_subplot(gridspace[fig_y_size+2:fig_y_size+3,fig_x_size+1:fig_x_size+3])
    cbar1 = plt.colorbar(bar_ev, ax=ax4,orientation='horizontal')
    cbar1.set_label('Depth of earthquakes',fontsize=16)
    cbar1.ax.tick_params(axis='x', labelsize=16)    # Colorbar font size.

    # Hide the borders of the axes.
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)

    # Hide the tick values of the axes.
    ax4.set_xticks([])
    ax4.set_yticks([])


# %%
# Function: fig_ev_st_distribution_wt(ev_info, st_info) Plot the distribution of the earthquakes and stations, color-coded by weight.
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

def fig_ev_st_distribution_wt(ev_info,st_info):


    [lon_ev,lat_ev,dep_ev,wt_ev] = data_lon_lat_dep_wt_ev(ev_info)
    [lon_st,lat_st,ele_st,wt_st] = data_lon_lat_ele_wt_st(ev_info,st_info)

    # Insert a value that does not affect the plot to make the colorbar range look better.
    lon_ev = np.insert(lon_ev, 0, lon_ev[0]); lat_ev = np.insert(lat_ev, 0, lat_ev[0]); dep_ev = np.insert(dep_ev, 0, dep_ev[0]); wt_ev = np.insert(wt_ev, 0, 0.0)
    lon_ev = np.insert(lon_ev, 0, lon_ev[0]); lat_ev = np.insert(lat_ev, 0, lat_ev[0]); dep_ev = np.insert(dep_ev, 0, dep_ev[0]); wt_ev = np.insert(wt_ev, 0, 1.0)
    lon_st = np.insert(lon_st, 0, lon_st[0]); lat_st = np.insert(lat_st, 0, lat_st[0]); ele_st = np.insert(ele_st, 0, ele_st[0]); wt_st = np.insert(wt_st, 0, 0.0)
    lon_st = np.insert(lon_st, 0, lon_st[0]); lat_st = np.insert(lat_st, 0, lat_st[0]); ele_st = np.insert(ele_st, 0, ele_st[0]); wt_st = np.insert(wt_st, 0, 1.0)

    min_lon = min(min(lon_ev),min(lon_st))
    max_lon = max(max(lon_ev),max(lon_st))

    min_lat = min(min(lat_ev),min(lat_st))
    max_lat = max(max(lat_ev),max(lat_st))

    max_dep = max(dep_ev)

    fig = plt.figure(figsize=(12,12))
    gridspace = GridSpec(12,12,figure = fig)

    xrange = max_lon - min_lon + 1.0
    yrange = max_lat - min_lat + 1.0

    if (xrange > yrange):
        fig_x_size = 6
        fig_y_size = round(6*yrange/xrange)
    else:
        fig_x_size = round(6*xrange/yrange)
        fig_y_size = 6


    ax1 = fig.add_subplot(gridspace[0:fig_y_size,0:fig_x_size])

    bar_ev = ax1.scatter(lon_ev,lat_ev,c=wt_ev,cmap="jet",label = "src",s = 3)
    bar_st = ax1.scatter(lon_st,lat_st,c=wt_st,cmap="jet",label = "rec",s = 100,marker='^',edgecolors='white')

    ax1.legend(fontsize = 14)
    ax1.tick_params(axis='x',labelsize=18)
    ax1.tick_params(axis='y',labelsize=18)
    ax1.set_xlabel('Lon',fontsize=18)
    ax1.set_ylabel('Lat',fontsize=18)
    ax1.set_xlim((min_lon - (max_lon - min_lon)*0.1,max_lon + (max_lon - min_lon)*0.1))
    ax1.set_ylim((min_lat - (max_lat - min_lat)*0.1,max_lat + (max_lat - min_lat)*0.1))


    ax2 = fig.add_subplot(gridspace[0:fig_y_size, fig_x_size+1 : fig_x_size+3])

    ax2.scatter(dep_ev,lat_ev,c=wt_ev,cmap="jet",label = "src",s = 3)

    ax2.tick_params(axis='x',labelsize=18)
    ax2.tick_params(axis='y',labelsize=18)
    ax2.set_xlabel('Dep',fontsize=18)
    ax2.set_ylabel('Lat',fontsize=18)
    ax2.set_xlim((-max_dep*0.05,max_dep*1.1))
    ax2.set_ylim((min_lat - (max_lat - min_lat)*0.1,max_lat + (max_lat - min_lat)*0.1))


    ax3 = fig.add_subplot(gridspace[fig_y_size+1:fig_y_size+3,0:fig_x_size])

    ax3.scatter(lon_ev,dep_ev,c=wt_ev,cmap="jet",label = "src",s = 3)

    ax3.tick_params(axis='x',labelsize=18)
    ax3.tick_params(axis='y',labelsize=18)
    ax3.set_xlabel('Lon',fontsize=18)
    ax3.set_ylabel('Dep',fontsize=18)
    ax3.set_xlim((min_lon - (max_lon - min_lon)*0.1,max_lon + (max_lon - min_lon)*0.1))
    ax3.set_ylim((-max_dep*0.05,max_dep*1.1))
    ax3.invert_yaxis()

    # Place the colorbar on a new axis.
    ax4 = fig.add_subplot(gridspace[fig_y_size+2:fig_y_size+3,fig_x_size+1:fig_x_size+3])
    cbar1 = plt.colorbar(bar_st, ax=ax4,orientation='horizontal')
    cbar1.set_label('Weight of stations',fontsize=16)
    cbar1.ax.tick_params(axis='x', labelsize=16)    # colorbar font size.
    # Hide the borders of the axes.
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)

    # Hide the tick values of the axes.
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Place the colorbar on a new axis.
    ax5 = fig.add_subplot(gridspace[fig_y_size+1:fig_y_size+2,fig_x_size+1:fig_x_size+3])
    cbar1 = plt.colorbar(bar_ev, ax=ax5,orientation='horizontal')
    cbar1.set_label('Weight of earthquakes',fontsize=16)
    cbar1.ax.tick_params(axis='x', labelsize=16)    # colorbar font size.
    # Hide the borders of the axes.
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(False)

    # Hide the tick values of the axes.
    ax5.set_xticks([])
    ax5.set_yticks([])


# %%
# Plot and function: plot the distance-time scatter plot, remove the outliers.
# Limit the data within the range defined by the line time = dis * slope + intercept and the bounds up and down.
# Remove outliers, only retain data satisfying: slope * dis + intercept + down < time < slope * dis + intercept + up.

def fig_data_plot_remove_outliers(ev_info,st_info,slope,intercept,up,down,dis_min,dis_max):

    fig = plt.figure(figsize=(10,10))
    gridspace = GridSpec(6,6,figure = fig)
    ax2 = fig.add_subplot(gridspace[0:6, 0:6])

    # plot original data
    [dis_obs,time_obs] = data_dis_time(ev_info,st_info)
    ax2.plot(dis_obs,time_obs,'r.',markersize=1.5,label = "discarded")

    # remove outliers, only retain data satisfying:     slope * dis + intercept + down < time < slope * dis + intercept + up
    ev_info = limit_data_residual(ev_info,st_info,slope,intercept,up,down)

    [dis_obs,time_obs] = data_dis_time(ev_info,st_info)
    ax2.plot(dis_obs,time_obs,'b.',markersize=1.5,label = "retained")

    ax2.plot([dis_min,dis_max],[slope*dis_min+intercept+up,slope*dis_max+intercept+up],'b-',linewidth=2)
    ax2.plot([dis_min,dis_max],[slope*dis_min+intercept+down,slope*dis_max+intercept+down],'b-',linewidth=2)
    ax2.plot([dis_min,dis_max],[slope*dis_min+intercept,slope*dis_max+intercept],'k-',linewidth=2)

    ax2.legend(fontsize = 14)
    ax2.tick_params(axis='x',labelsize=18)
    ax2.tick_params(axis='y',labelsize=18)
    ax2.set_xlabel('Distance (km)',fontsize=18)
    ax2.set_ylabel('Traveltime',fontsize=18)
    ax2.set_xlim((dis_min,dis_max))
    ax2.set_ylim((intercept+down-5,slope*dis_max+intercept+up+5))

    return ev_info

# %%
# Plot: distance-time scatter plot of given phases.

def fig_data_plot_phase(ev_info,st_info,phase_list,color_list,dis_min,dis_max):

    [dis_obs_phase,time_obs_phase] = data_dis_time_phase(ev_info,st_info,phase_list)

    regression = {}
    # Calculate the least squares y = ax+b
    for key_phase in phase_list:
        X = dis_obs_phase[key_phase]
        Y = time_obs_phase[key_phase]

        if(len(X)>20):
            regression[key_phase] = linear_regression(X,Y)
        else:
            print("No enough data: %d, for %s"%(len(X),key_phase))
            regression[key_phase] = [0,0,0]


    # draw
    fig = plt.figure(figsize=(10,10))
    gridspace = GridSpec(6,6,figure = fig)
    ax2 = fig.add_subplot(gridspace[0:6, 0:6])
    y1 = 99999; y2 = -99999

    # scatter plot
    for iphase in range(len(phase_list)):
        phase = phase_list[iphase]
        color = color_list[iphase]
        ax2.plot(dis_obs_phase[phase],time_obs_phase[phase],'%s.'%(color),markersize=1)

    # linear regression plot
    for iphase in range(len(phase_list)):
        phase = phase_list[iphase]
        color = color_list[iphase]
        (slope,intercept,SEE)= regression[phase]
        ax2.plot([dis_min,dis_max],[dis_min*slope+intercept,dis_max*slope+intercept],'%s-'%(color),linewidth=2,label = "%s: a,b,SEE=%5.2f,%5.2f,%5.2f"%(phase,slope,intercept,SEE))
        y1 = min(y1,intercept-5)
        y2 = max(y2,dis_max*slope+intercept+5)

    ax2.legend(fontsize = 14)
    ax2.tick_params(axis='x',labelsize=18)
    ax2.tick_params(axis='y',labelsize=18)
    ax2.set_xlabel('Distance (km)',fontsize=18)
    ax2.set_ylabel('Traveltime (s)',fontsize=18)
    ax2.set_xlim((dis_min,dis_max))


    for iphase in range(len(phase_list)):
        try:
            y1 = min(y1,min(time_obs_phase[phase]))
            y2 = max(y2,max(time_obs_phase[phase]))
        except:
            pass
    ax2.set_ylim((y1,y2))

    title = ""
    for phase in dis_obs_phase:
        title = title + "%s(%d) "%(phase,len(dis_obs_phase[phase]))

    ax2.set_title(title)

    print("a is slope, b is intercept, SEE is standard error of estimate")

# %% [markdown]
# Functions: results analysis and evaluation

# %%
# plot: plot the residual histogram of the initial and final model
def fig_residual_histogram(fn_syn_init,fn_syn_final,fn_obs,range_l,range_r,Nbar,tag1 = "initial",tag2 = "final"):

    # read synthetic traveltime data in the initial model
    [ev_info_syn_init, st_info_syn_init] = read_src_rec_file(fn_syn_init)
    time_syn_init = data_dis_time(ev_info_syn_init,st_info_syn_init)[1]

    # read synthetic traveltime data in the final model
    [ev_info_syn_final, st_info_syn_final] = read_src_rec_file(fn_syn_final)
    time_syn_final = data_dis_time(ev_info_syn_final,st_info_syn_final)[1]

    # read observed traveltime data
    [ev_info_obs, st_info_obs] = read_src_rec_file(fn_obs)
    time_obs = data_dis_time(ev_info_obs,st_info_obs)[1]

    fig = plt.figure(figsize=(6,6))
    gridspace = GridSpec(6,6,figure = fig)

    ax2 = fig.add_subplot(gridspace[0:6, 0:6])

    bins=np.linspace(range_l,range_r,Nbar)
    error_init = time_syn_init - time_obs
    error_final = time_syn_final - time_obs

    hist_init,  _, _ = ax2.hist(error_init,bins=bins,histtype='step', edgecolor = "red", linewidth = 2,
            label = "%s: std = %5.3f s, mean = %5.3f s"%(tag1,np.std(error_init),np.mean(error_init)))

    hist_final, _, _ = ax2.hist(error_final,bins=bins,alpha = 0.5, color = "blue",
            label = "%s: std = %5.3f s, mean = %5.3f s"%(tag2,np.std(error_final),np.mean(error_final)))

    print("residual for ",tag1," model is: ","mean: ",np.mean(error_init),"sd: ",np.std(error_init))
    print("residual for ",tag2," model is: ","mean: ",np.mean(error_final),"sd: ",np.std(error_final))
    ax2.legend(fontsize=14)

    ax2.set_xlim(range_l - abs(range_l)*0.1,range_r + abs(range_r)*0.1)
    ax2.set_ylim(0,1.3*max(max(hist_init),max(hist_final)))

    ax2.tick_params(axis='x',labelsize=18)
    ax2.tick_params(axis='y',labelsize=18)
    ax2.set_ylabel('Number of data',fontsize=18)
    ax2.set_xlabel('Traveltime residuals (s)',fontsize=18)
    ax2.set_title("$t_{syn} - t_{obs}$",fontsize=18)
    ax2.grid()


# %%
# plot: plot the residual histogram of the initial and final model
def fig_data_difference_histogram(fn_A,fn_B,range_l,range_r,Nbar):

    # read data A
    [ev_info_A, st_info_A] = read_src_rec_file(fn_A)

    # read data B
    [ev_info_B, st_info_B] = read_src_rec_file(fn_B)

    # absolute traveltime residual
    error_t = []
    for key_ev in ev_info_A:
        for key_t in ev_info_A[key_ev].t:
            data_A = ev_info_A[key_ev].t[key_t]
            if (key_ev in ev_info_B and key_t in ev_info_B[key_ev].t):
                data_B = ev_info_B[key_ev].t[key_t]
                error_t.append(data_A[2] - data_B[2])

    # common-source differential traveltime residual
    error_cs_dt = []
    for key_ev in ev_info_A:
        for key_dt in ev_info_A[key_ev].cs_dt:
            data_A = ev_info_A[key_ev].cs_dt[key_dt]
            if (key_ev in ev_info_B and key_dt in ev_info_B[key_ev].cs_dt):
                data_B = ev_info_B[key_ev].cs_dt[key_dt]
                error_cs_dt.append(data_A[3] - data_B[3])
            else:
                print(key_ev,key_dt)

    # common-receiver differential traveltime residual
    error_cr_dt = []
    for key_ev in ev_info_A:
        for key_dt in ev_info_A[key_ev].cr_dt:
            data_A = ev_info_A[key_ev].cr_dt[key_dt]
            if (key_ev in ev_info_B and key_dt in ev_info_B[key_ev].cr_dt):
                data_B = ev_info_B[key_ev].cr_dt[key_dt]
                error_cr_dt.append(data_A[3] - data_B[3])

    # plot
    fig = plt.figure(figsize=(14,6))
    gridspace = GridSpec(6,14,figure = fig)


    ax2 = fig.add_subplot(gridspace[0:6, 0:6])
    bins=np.linspace(range_l,range_r,Nbar)
    # hist_t,  _, _ = ax2.hist(error_t,bins=bins,histtype='step', edgecolor = "red", linewidth = 2,
    #         label = "noise: std = %5.3f s, mean = %5.3f s"%(np.std(error_t),np.mean(error_t)))
    hist_t, _, _ = ax2.hist(error_t,bins=bins,alpha = 0.5, color = "blue",
            label = "noise: std = %5.3f s, mean = %5.3f s"%(np.std(error_t),np.mean(error_t)))

    ax2.legend(fontsize=14)
    ax2.set_xlim(range_l - abs(range_l)*0.1,range_r + abs(range_r)*0.1)
    try:
        ax2.set_ylim(0,1.3*max(hist_t))
    except:
        ax2.set_ylim(0,1.0)
    ax2.tick_params(axis='x',labelsize=18)
    ax2.tick_params(axis='y',labelsize=18)
    ax2.set_ylabel('Number of data',fontsize=18)
    ax2.set_xlabel('Noise (s)',fontsize=18)
    ax2.set_title("Noise of traveltime",fontsize=18)


    ax3 = fig.add_subplot(gridspace[0:6,8:14])
    bins=np.linspace(range_l,range_r,Nbar)
    # hist_t,  _, _ = ax3.hist(error_t,bins=bins,histtype='step', edgecolor = "red", linewidth = 2,
    #         label = "noise: std = %5.3f s, mean = %5.3f s"%(np.std(error_t),np.mean(error_t)))
    hist_cs_dt, _, _ = ax3.hist(error_cs_dt,bins=bins,alpha = 0.5, color = "blue",
            label = "noise: std = %5.3f s, mean = %5.3f s"%(np.std(error_cs_dt),np.mean(error_cs_dt)))

    ax3.legend(fontsize=14)
    ax3.set_xlim(range_l - abs(range_l)*0.1,range_r + abs(range_r)*0.1)
    try:
        ax3.set_ylim(0,1.3*max(hist_cs_dt))
    except:
        ax3.set_ylim(0,1.0)
    ax3.tick_params(axis='x',labelsize=18)
    ax3.tick_params(axis='y',labelsize=18)
    ax3.set_ylabel('Number of data',fontsize=18)
    ax3.set_xlabel('Noise (s)',fontsize=18)
    ax3.set_title("Noise of differential traveltime",fontsize=18)


