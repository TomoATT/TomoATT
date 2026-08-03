#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --gres=gpu:GB10:1
#SBATCH --time=48:00:00
#SBATCH -J jap_cbvr
#SBATCH -o %x_%j.out
# Checkerboard (velocity) RESTART: skip forward (src_rec_file_forward.dat
# already synthesized), run inversion only from warm model (9 iterations
# left after iter-0 model update), 1 spark node (GB10), GPU, UPWIND.
# NOTE: one TOMOATT per GPU node — co-location OOMs the 122 GB GB10.
set -e

export PATH=$HOME/tomoatt_deps/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
CASE=$HOME/TomoATT/examples/realcase_japan_tomography
cd $CASE/3_input_params

VAR=input_params_cb_inversion_vel_restart.yml
python3 - << PYEOF
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
PYEOF

grep -nE "init_model_path|use_gpu|max_iterations:" $VAR | head -4
time $BIN -i $VAR
echo "=== JOB DONE rc=$? ==="
ls -lh $CASE/OUTPUT_FILES_cb_inversion_vel/ | head
