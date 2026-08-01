#!/bin/bash
# Detached multi-node Japan tomography run (GPU + UPWIND, sweep-parallel).
# Launched via:  ssh node "nohup setsid /path/to/this > log 2>&1 < /dev/null &"
# NOT an sbatch script: batch-mode multi-node jobs are torn down on this cluster
# (see run notes). The 4 ranks come from ONE srun -N4 -n4 (proven interactive).

set -e

# NRANK=1: single spark node (spark partition allows MaxNodes=1).
# NRANK=2: spark-double multi-node GPU. NOTE (2026-08-01): multi-node GPU jobs are
# killed by the cluster ~60-90 s after the TOMOATT step starts irrespective of
# launcher (batch/interactive, CPU or GPU); single node works indefinitely.
NRANK=${1:-1}
CASE=$HOME/TomoATT/examples/realcase_japan_tomography
BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
RUNDIR=$CASE/OUTPUT_FILES_run_gpu
mkdir -p $RUNDIR

# --- environment (PMIx stack for multi-node) ---
export PATH=$HOME/mpi_stack/ompi/bin:$HOME/tomoatt_deps/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mpi_stack/ompi/lib:$HOME/mpi_stack/pmix/lib:$HOME/mpi_stack/hwloc/lib:$HOME/mpi_stack/libevent/lib:$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_oob_tcp_if_include="enP7s7"
export OMPI_MCA_btl_tcp_if_include="enP7s7"
export OMPI_MCA_opal_warn_on_missing_libevent=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nomps_pipe_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/nomps_log_$$

# --- config variant (original yml untouched) ---
cd $CASE
YML=3_input_params/input_params_inversion.yml
VAR=$RUNDIR/input_params_gpu.yml
cp $YML $VAR
python3 - << EOF
import re
t = open("$VAR").read()
def setv(t, key, val):
    if re.search(r'^\s*'+key+r'\s*:', t, flags=re.M):
        return re.sub(r'^(\s*)'+key+r'\s*:.*$', r'\g<1>'+key+': '+val, t, flags=re.M)
    return t
t = setv(t, 'use_gpu', 'true')
# GPU mode requires nproc_sub=1 (sweep-parallel unsupported on GPU);
# use domain decomposition 'ndiv_rtp' across the ranks instead.
t = setv(t, 'nproc_sub', '1')
NDIV = '[1, 1, 1]' if ${NRANK} == 1 else '[1, 1, ' + str(${NRANK}) + ']'
t = setv(t, 'ndiv_rtp', NDIV)
t = setv(t, 'swap_src_rec', 'true')
t = setv(t, 'stencil_order', '1')
t = setv(t, 'stencil_type', '1')
# outputs go to the run dir instead of the case root
t = re.sub(r'^\s*output_dir\s*:.*$', '  output_dir: '+r'$RUNDIR', t, flags=re.M)
open("$VAR", "w").write(t)
EOF

echo "=== environment check ==="
nvidia-smi -L || true
grep -nE "use_gpu|nproc_sub|ndiv_rtp|swap_src_rec|stencil_order|stencil_type|output_dir" $VAR

echo "=== launch ${NRANK} ranks on spark-double nodes ==="
# run from 3_input_params so relative data paths (../2_data_processing) resolve
cd $CASE/3_input_params
if [ $NRANK -eq 1 ]; then
  # still needs a compute node (login node is x86_64; binary is aarch64)
  time srun -p spark -N 1 --export=ALL $BIN -i $VAR
else
  time srun -p spark-double -N $NRANK -n $NRANK --mpi=pmix_v5 --export=ALL $BIN -i $VAR
fi
echo "=== DONE rc=$? ==="
ls -lh $RUNDIR | head -20
