"""
Create source-receiver file for free-surface reflection test (pP, sP).

Uses the same two-layer model as the main test, but with a deeper source
so that pP and sP phases have meaningful traveltimes.

Source at depth=30 km so the upgoing leg is significant.
Receivers at the surface.
"""

import numpy as np
import math

R_earth = 6371.0

src_lat = 1.0
src_lon = 1.0
src_dep = 30.0  # 30 km depth — deep source for clear pP/sP

Vp1 = 6.0   # km/s — layer 1 (above Moho)
Vs1 = 3.5   # km/s

n_rec = 20
rec_lats = np.full(n_rec, 1.0)
rec_lons = np.linspace(0.2, 1.8, n_rec)

# pP traveltime (flat-earth, image-source for surface receiver):
# Image source at depth -src_dep, receiver at surface.
# T_pP = sqrt(src_dep^2 + x^2) / Vp
# (equals the direct P time for surface receivers)

# sP traveltime (flat-earth, surface receiver):
# S leg from source to surface, then P along surface to receiver.
# T_sP = min over x_r: sqrt(dep^2 + x_r^2)/Vs + |x - x_r|/Vp
# For |x| >= x_crit: T = sqrt(dep^2 + x_crit^2)/Vs + (|x| - x_crit)/Vp
# For |x| < x_crit:  T = sqrt(dep^2 + x^2)/Vs
# where x_crit = dep * Vs / sqrt(Vp^2 - Vs^2)
x_crit = src_dep * Vs1 / math.sqrt(Vp1**2 - Vs1**2)

print("Expected pP and sP traveltimes (approximate flat-earth):")
with open('src_rec_surface.dat', 'w') as f:
    f.write(f"         1 1998  1  1  0  0   0.000    {src_lat:.4f}     {src_lon:.4f}   {src_dep:.2f}  3.00    {n_rec * 2}    100002\n")

    rec_id = 0
    # pP receivers
    for i in range(n_rec):
        rec_id += 1
        rec_name = f"pP{i+1:05d}"
        delta_lon = rec_lons[i] - src_lon
        x_km = delta_lon * math.pi / 180.0 * R_earth * math.cos(src_lat * math.pi / 180.0)
        # pP for surface receiver = sqrt(src_dep^2 + x^2) / Vp (= direct P time)
        t_pP = math.sqrt(src_dep**2 + x_km**2) / Vp1
        f.write(f"         1      {rec_id}      {rec_name}       {rec_lats[i]:.4f}     {rec_lons[i]:.4f}      0.0000  pP    {abs(x_km):.2f}  {t_pP:.3f}      1.000\n")
        print(f"  pP Receiver {i+1}: offset={x_km:.1f} km, T_pP={t_pP:.3f} s")

    # sP receivers
    for i in range(n_rec):
        rec_id += 1
        rec_name = f"sP{i+1:05d}"
        delta_lon = rec_lons[i] - src_lon
        x_km = delta_lon * math.pi / 180.0 * R_earth * math.cos(src_lat * math.pi / 180.0)
        # sP for surface receiver: S leg to surface, P along surface
        x_abs = abs(x_km)
        if x_abs <= x_crit:
            t_sP = math.sqrt(src_dep**2 + x_km**2) / Vs1
        else:
            T_crit = math.sqrt(src_dep**2 + x_crit**2) / Vs1
            t_sP = T_crit + (x_abs - x_crit) / Vp1
        f.write(f"         1      {rec_id}      {rec_name}       {rec_lats[i]:.4f}     {rec_lons[i]:.4f}      0.0000  sP    {abs(x_km):.2f}  {t_sP:.3f}      1.000\n")
        print(f"  sP Receiver {i+1}: offset={x_km:.1f} km, T_sP={t_sP:.3f} s")

print(f"\nWrote src_rec_surface.dat with {rec_id} receivers (phases: pP, sP)")
