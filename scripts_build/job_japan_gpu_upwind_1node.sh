#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J japan_gpu
#SBATCH -o %x_%j.out
# Japan tomography: 1 spark node (GB10), GPU + UPWIND 1st order, single rank.
set -e

export PATH=$HOME/tomoatt_deps/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nomps_pipe_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/nomps_log_$$

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
CASE=$HOME/TomoATT/examples/realcase_japan_tomography
RUNDIR=$CASE/OUTPUT_FILES_run_gpu
mkdir -p $RUNDIR

cd $CASE/3_input_params
VAR=input_params_gpu.yml
cp input_params_inversion.yml $VAR
python3 - << EOF
import re
t = open("$VAR").read()
def setv(t, key, val):
    if re.search(r'^\s*'+key+r'\s*:', t, flags=re.M):
        return re.sub(r'^(\s*)'+key+r'\s*:.*$', r'\g<1>'+key+': '+val, t, flags=re.M)
    return t
t = setv(t, 'use_gpu', 'true')
t = setv(t, 'nproc_sub', '1')
t = setv(t, 'ndiv_rtp', '[1, 1, 1]')
t = setv(t, 'swap_src_rec', 'true')
t = setv(t, 'stencil_order', '1')
t = setv(t, 'stencil_type', '1')
t = re.sub(r'^\s*output_dir\s*:.*$', '  output_dir: '+r'$RUNDIR', t, flags=re.M)
open("$VAR", "w").write(t)
EOF

grep -nE "use_gpu|ndiv_rtp|swap_src_rec|stencil_order|stencil_type|output_dir" $VAR
nvidia-smi -L || true
time $BIN -i $VAR
echo "=== JOB DONE rc=$? ==="
ls -lh $RUNDIR | head
