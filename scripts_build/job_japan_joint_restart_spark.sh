#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH -J jap_v2r
#SBATCH -o %x_%j.out
# Japan production v2 RESTART: continues from restart_model_joint.h5
# (model after iteration 0 of the killed joint run), 19 iterations left,
# regional + teleseismic, joint P + azimuthal anisotropy, GPU.
set -e

export PATH=$HOME/tomoatt_deps/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
CASE=$HOME/TomoATT/examples/realcase_japan_tomography
cd $CASE/3_input_params

VAR=input_params_joint_restart.yml
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
nvidia-smi -L || true
time $BIN -i $VAR
echo "=== JOB DONE rc=$? ==="
ls -lh $CASE/OUTPUT_FILES_joint_tele_ani/ | head
