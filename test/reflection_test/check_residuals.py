import math, numpy as np

R_earth = 6371.0
src_lat, src_lon, src_dep = 1.0, 1.0, 5.0
moho_depth = 35.0
Vp1 = 6.0
h = moho_depth - src_dep

rec_lons, T_syn, T_obs = [], [], []
with open('OUTPUT_FILES/src_rec_file_forward.dat', 'r') as f:
    for i, line in enumerate(f):
        if i == 0: continue
        parts = line.split()
        rec_lons.append(float(parts[4]))
        T_syn.append(float(parts[7]))
        T_obs.append(float(parts[8]))

rec_lons = np.array(rec_lons)
T_syn = np.array(T_syn)
T_obs = np.array(T_obs)

offset_rec = (rec_lons - src_lon) * math.pi / 180.0 * R_earth * math.cos(math.radians(src_lat))
T_anal = 2.0 * np.sqrt(h**2 + (offset_rec / 2.0)**2) / Vp1

print("Rec   Lon     Offset(km)  T_anal   T_comp   Residual   %err")
print("-" * 65)
for i in range(len(rec_lons)):
    res = T_syn[i] - T_anal[i]
    pct = 100 * res / T_anal[i]
    print("{:3d}  {:6.3f}  {:8.2f}  {:7.3f}  {:7.3f}  {:+8.3f}  {:+6.1f}%".format(
        i+1, rec_lons[i], offset_rec[i], T_anal[i], T_syn[i], res, pct))

print()
print("Max |residual| = {:.3f} s".format(np.max(np.abs(T_syn - T_anal))))
print("Mean |residual| = {:.3f} s".format(np.mean(np.abs(T_syn - T_anal))))
print("Capped at 20.0 count: {}".format(np.sum(T_syn >= 19.99)))
