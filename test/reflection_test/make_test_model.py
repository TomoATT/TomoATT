"""
Create a two-layer velocity model for testing reflected-wave computation.

Model:
  - Layer 1 (0-35 km depth): Vp=6.0 km/s, Vs=3.5 km/s
  - Layer 2 (35-70 km depth): Vp=8.0 km/s, Vs=4.5 km/s
  - Interface ("Moho") at 35 km depth

Grid: 71 nodes in depth (0-70 km), 41x41 in lat/lon (0-2 deg)
"""

import numpy as np
import math
import h5py

R_earth = 6371.0

# Domain: 0-70 km depth, 0-2 deg lat/lon
dep_min = -5.0   # slight extension above surface
dep_max = 75.0

lat_min = (0.0 - 0.3) / 180.0 * math.pi
lat_max = (2.0 + 0.3) / 180.0 * math.pi
lon_min = (0.0 - 0.3) / 180.0 * math.pi
lon_max = (2.0 + 0.3) / 180.0 * math.pi

r_min = R_earth - dep_max
r_max = R_earth - dep_min

n_rtp = [41, 41, 41]  # [nr, nt, np] = [depth, lat, lon]

dr = (r_max - r_min) / (n_rtp[0] - 1)
dt = (lat_max - lat_min) / (n_rtp[1] - 1)
dp = (lon_max - lon_min) / (n_rtp[2] - 1)

rr = np.array([r_min + k * dr for k in range(n_rtp[0])])
tt = np.array([lat_min + j * dt for j in range(n_rtp[1])])
pp = np.array([lon_min + i * dp for i in range(n_rtp[2])])

# Moho depth
moho_depth_km = 35.0
moho_r = R_earth - moho_depth_km

# Velocities
Vp_layer1 = 6.0   # km/s
Vp_layer2 = 8.0   # km/s
Vs_layer1 = 3.5   # km/s
Vs_layer2 = 4.5   # km/s

# Create model arrays
vel_p = np.zeros(n_rtp)   # P-wave velocity
vel_s = np.zeros(n_rtp)   # S-wave velocity
xi    = np.zeros(n_rtp)   # anisotropy (isotropic)
eta   = np.zeros(n_rtp)   # anisotropy (isotropic)

for ir in range(n_rtp[0]):
    for it in range(n_rtp[1]):
        for ip in range(n_rtp[2]):
            if rr[ir] >= moho_r:
                # Above Moho (shallower)
                vel_p[ir, it, ip] = Vp_layer1
                vel_s[ir, it, ip] = Vs_layer1
            else:
                # Below Moho (deeper)
                vel_p[ir, it, ip] = Vp_layer2
                vel_s[ir, it, ip] = Vs_layer2

print(f"Grid: {n_rtp[0]}x{n_rtp[1]}x{n_rtp[2]}")
print(f"Depth range: {dep_min} to {dep_max} km")
print(f"Moho at depth={moho_depth_km} km (r={moho_r} km)")
print(f"dr={dr:.2f} km, dt={dt*180/math.pi:.4f} deg, dp={dp*180/math.pi:.4f} deg")

# Write HDF5 model
with h5py.File('two_layer_model.h5', 'w') as f:
    f.create_dataset('vel',   data=vel_p)
    f.create_dataset('vel_s', data=vel_s)
    f.create_dataset('xi',    data=xi)
    f.create_dataset('eta',   data=eta)

print("Wrote two_layer_model.h5")

# -------------------------------------------------------
# Create source-receiver file
# -------------------------------------------------------
# Source at surface (lat=1.0, lon=1.0, depth=0 km)
# Receivers at surface in a line from (1.0, 0.1) to (1.0, 1.9)

src_lat = 1.0
src_lon = 1.0
src_dep = 5.0  # 5 km depth

n_rec = 20
rec_lats = np.full(n_rec, 1.0)
rec_lons = np.linspace(0.2, 1.8, n_rec)
rec_dep  = 0.0  # surface receivers

# Expected PmP traveltime (analytical image-source for asymmetric geometry):
# T_PmP = sqrt((2*z_m - z_s)^2 + x^2) / Vp
# Image source is at depth (2*z_m - z_s) below the surface receiver.
h_img = 2.0 * moho_depth_km - src_dep  # image depth for surface receiver

print("\nExpected PmP traveltimes (approximate):")
for i in range(n_rec):
    # horizontal distance (approx, in km)
    delta_lon = rec_lons[i] - src_lon
    x_km = delta_lon * math.pi / 180.0 * R_earth * math.cos(src_lat * math.pi / 180.0)
    t_PmP = math.sqrt(h_img**2 + x_km**2) / Vp_layer1
    print(f"  Receiver {i+1}: offset={x_km:.1f} km, T_PmP={t_PmP:.3f} s")

# Write source-receiver file
with open('src_rec_config.dat', 'w') as f:
    # Source line
    f.write(f"         1 1998  1  1  0  0   0.000    {src_lat:.4f}     {src_lon:.4f}   {src_dep:.2f}  3.00    {n_rec}    100001\n")

    for i in range(n_rec):
        rec_name = f"REC{i+1:04d}"
        delta_lon = rec_lons[i] - src_lon
        x_km = delta_lon * math.pi / 180.0 * R_earth * math.cos(src_lat * math.pi / 180.0)
        # PmP traveltime (image-source for asymmetric geometry)
        t_PmP = math.sqrt(h_img**2 + x_km**2) / Vp_layer1
        f.write(f"         1      {i+1}      {rec_name}       {rec_lats[i]:.4f}     {rec_lons[i]:.4f}      0.0000  PmP   {abs(x_km):.2f}  {t_PmP:.3f}      1.000\n")

print(f"\nWrote src_rec_config.dat with {n_rec} receivers (phase=PmP)")
