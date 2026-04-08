"""
Create a larger two-layer velocity model and source-receiver file for the SIMD
efficiency benchmark. The physics match the reflected-wave test fixture, but the
grid is denser to make sweep performance differences easier to observe.
"""

import math

import h5py
import numpy as np


R_EARTH = 6371.0

# Domain: 0-70 km depth, 0-2 deg lat/lon
DEP_MIN = -5.0
DEP_MAX = 75.0
LAT_MIN = 0.0
LAT_MAX = 2.0
LON_MIN = 0.0
LON_MAX = 2.0

# Benchmark grid
N_RTP = [61, 61, 61]

R_MIN = R_EARTH - DEP_MAX
R_MAX = R_EARTH - DEP_MIN

DR = (R_MAX - R_MIN) / (N_RTP[0] - 1)

RR = np.array([R_MIN + index * DR for index in range(N_RTP[0])])

MOHO_DEPTH_KM = 35.0
MOHO_R = R_EARTH - MOHO_DEPTH_KM

VP_LAYER1 = 6.0
VP_LAYER2 = 8.0
VS_LAYER1 = 3.5
VS_LAYER2 = 4.5

VEL_P = np.zeros(N_RTP)
VEL_S = np.zeros(N_RTP)
XI = np.zeros(N_RTP)
ETA = np.zeros(N_RTP)

for ir in range(N_RTP[0]):
    layer_is_shallow = RR[ir] >= MOHO_R
    vp = VP_LAYER1 if layer_is_shallow else VP_LAYER2
    vs = VS_LAYER1 if layer_is_shallow else VS_LAYER2
    VEL_P[ir, :, :] = vp
    VEL_S[ir, :, :] = vs

with h5py.File("two_layer_model_benchmark.h5", "w") as h5file:
    h5file.create_dataset("vel", data=VEL_P)
    h5file.create_dataset("vel_s", data=VEL_S)
    h5file.create_dataset("xi", data=XI)
    h5file.create_dataset("eta", data=ETA)


SRC_LAT = 1.0
SRC_LON = 1.0
SRC_DEP = 5.0
N_REC = 20
REC_LATS = np.full(N_REC, 1.0)
REC_LONS = np.linspace(0.2, 1.8, N_REC)
H_IMG = 2.0 * MOHO_DEPTH_KM - SRC_DEP

with open("src_rec_benchmark.dat", "w", encoding="utf-8") as src_rec_file:
    src_rec_file.write(
        f"         1 1998  1  1  0  0   0.000    {SRC_LAT:.4f}     {SRC_LON:.4f}   {SRC_DEP:.2f}  3.00    {N_REC}    100001\n"
    )

    for receiver_index, receiver_lon in enumerate(REC_LONS, start=1):
        receiver_name = f"REC{receiver_index:04d}"
        delta_lon = receiver_lon - SRC_LON
        offset_km = delta_lon * math.pi / 180.0 * R_EARTH * math.cos(SRC_LAT * math.pi / 180.0)
        traveltime_pmp = math.sqrt(H_IMG**2 + offset_km**2) / VP_LAYER1
        src_rec_file.write(
            f"         1      {receiver_index}      {receiver_name}       {REC_LATS[receiver_index - 1]:.4f}     {receiver_lon:.4f}      0.0000  PmP   {abs(offset_km):.2f}  {traveltime_pmp:.3f}      1.000\n"
        )

print(f"Wrote benchmark model with grid {N_RTP[0]}x{N_RTP[1]}x{N_RTP[2]}")
print("Wrote two_layer_model_benchmark.h5")
print("Wrote src_rec_benchmark.dat")