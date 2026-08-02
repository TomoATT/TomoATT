#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH -J jap_cbv
#SBATCH -o %x_%j.out
# Checkerboard (velocity) suite: forward synthesis through checker_vel_coarse.h5
# then 10-iteration recovery inversion, on 1 spark node (GB10), GPU, UPWIND.
set -e

export PATH=$HOME/tomoatt_deps/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
CASE=$HOME/TomoATT/examples/realcase_japan_tomography
cd $CASE/3_input_params

# force GPU + single-rank GPU settings; keep swap_src_rec false (tele data)
for VAR in input_params_cb_forward_vel.yml input_params_cb_inversion_vel.yml; do
python3 - << EOF
import re
VAR = "$VAR"
t = open(VAR).read()
def setv(t, key, val):
    if re.search(r'^\s*'+key+r'\s*:', t, flags=re.M):
        return re.sub(r'^(\s*)'+key+r'\s*:.*$', r'\g<1>'+key+': '+val, t, flags=re.M)
    return t
t = setv(t, 'use_gpu', 'true')
t = setv(t, 'nproc_sub', '1')
t = setv(t, 'ndiv_rtp', '[1, 1, 1]')
open(VAR, "w").write(t)
EOF
done

echo '=== [1/2] checkerboard FORWARD (velocity) ==='
time $BIN -i input_params_cb_forward_vel.yml
ls -lh ../OUTPUT_FILES_cb_forward_vel/src_rec_file_forward.dat

echo '=== [2/2] checkerboard INVERSION (velocity) ==='
time $BIN -i input_params_cb_inversion_vel.yml
ls -lh ../OUTPUT_FILES_cb_inversion_vel/ | head
echo "=== JOB DONE rc=$? ==="
