#!/bin/bash
#SBATCH -p spark-double
#SBATCH -N 2
#SBATCH -J tatt_2node
#SBATCH -t 03:00:00
#SBATCH -o %x_%j.out
# Multi-node GPU test: MPI ranks across 2 spark nodes (1 GB10 each) via PMIx.
# 1) hello-MPI across nodes. 2) TOMOATT 2-rank (ndiv_rtp = [1,1,2]) GPU inversion,
#    compared against the single-rank GPU reference from the verified run.
set -e

S=$HOME/mpi_stack
export PATH=$S/ompi/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$S/ompi/lib:$S/pmix/lib:$S/hwloc/lib:$S/libevent/lib:$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
# inter-node links: RoCE openib init fails on this cluster; go plain TCP over enP7s7.
# NOTE: these must be exported BEFORE srun; passing VAR=a,b in --export=ALL,VAR=...
# breaks parsing at the comma and the MCA settings never reach the ranks.
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_oob_tcp_if_include="enP7s7"
export OMPI_MCA_btl_tcp_if_include="enP7s7"

echo "### nodes: $(hostname)"
nvidia-smi -L || true

# ---- 1) hello MPI across 2 nodes ----
if [ ! -x $S/bin/hello_mpi ]; then
  mkdir -p $S/bin
  cat > /tmp/hello.c << 'EOC'
#include <mpi.h>
#include <stdio.h>
int main(int c, char** v){ MPI_Init(&c,&v); int r,n; char h[256]; int l=256;
  MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&n);
  MPI_Get_processor_name(h,&l); printf("rank %d/%d on %s\n",r,n,h);
  MPI_Finalize(); return 0; }
EOC
  $S/ompi/bin/mpicc /tmp/hello.c -o $S/bin/hello_mpi
fi
echo "=== srun -N2 -n2 hello ==="
srun -N 2 -n 2 --mpi=pmix_v5 --export=ALL $S/bin/hello_mpi | sort
echo "=== srun -N2 -n4 hello ==="
srun -N 2 -n 4 --mpi=pmix_v5 --export=ALL $S/bin/hello_mpi | sort
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd, -)
echo "nodes for mpirun: $NODES"

# ---- 2) TOMOATT 2-rank multi-node GPU run ----
BIN=$HOME/TomoATT/build_spark_mpi/bin/TOMOATT
TEST=$HOME/TomoATT/test/inversion_small
cd $TEST

restore_baseline () {
  for YML in input_params.yml input_params_pre.yml; do
    grep -qE "^\s*use_gpu\s*:" $YML       && sed -i -E "s/^(\s*)use_gpu\s*:.*/\1use_gpu: false # restored/" $YML
    grep -qE "^\s*ndiv_rtp\s*:" $YML      && sed -i -E "s/^(\s*)ndiv_rtp\s*:.*/\1ndiv_rtp: [1, 1, 1] # restored/" $YML
  done
}
trap restore_baseline EXIT

for YML in input_params.yml input_params_pre.yml; do
  # 2 ranks: split domain only in r (depth) direction
  if grep -qE "^\s*ndiv_rtp\s*:" $YML; then
    sed -i -E "s/^(\s*)ndiv_rtp\s*:.*/\1ndiv_rtp: [1, 1, 2] # multi-node test/" $YML
  fi
  if grep -qE "^\s*use_gpu\s*:" $YML; then
    sed -i -E "s/^(\s*)use_gpu\s*:.*/\1use_gpu: true # multi-node test/" $YML
  else
    sed -i -E "/^parallel:/a\\  use_gpu: true # multi-node test" $YML
  fi
done
grep -nE "^\s*(use_gpu|ndiv_rtp)\s*:" input_params_pre.yml input_params.yml

echo "=== 2-rank GPU pre run ==="
rm -f cuda_device_info.txt
# launch via mpirun (ORTE) rather than inner srun steps: batch+nested srun gets
# torn down after ~90s on this cluster (exit 15:0), while direct batches are fine.
$S/ompi/bin/mpirun -H $NODES -n 2 --map-by node --bind-to none $BIN -i input_params_pre.yml > mn_pre.log 2>&1
[ -f cuda_device_info.txt ] && head -1 cuda_device_info.txt

echo "=== 2-rank GPU main run ==="
$S/ompi/bin/mpirun -H $NODES -n 2 --map-by node --bind-to none $BIN -i input_params.yml > mn_main.log 2>&1
grep -E "converged at iteration" mn_main.log | sort | uniq -c | head -6
grep -E "converged at iteration" mn_pre.log  | sort | uniq -c | head -6

rm -rf OUTPUT_FILES_2node_gpu
cp -r OUTPUT_FILES OUTPUT_FILES_2node_gpu

# ---- compare vs 1-rank GPU reference (from verified job, if present) ----
if [ -d OUTPUT_FILES_gpu ]; then
  echo "=== DIFF 2-node GPU vs 1-rank GPU ==="
  ~/tomoatt_deps/bin/h5diff -d 1e-8 OUTPUT_FILES_gpu/final_model.h5 OUTPUT_FILES_2node_gpu/final_model.h5 >/tmp/h5mn.txt 2>&1 \
    && echo "final_model.h5: IDENTICAL <=1e-8" || { echo "final_model.h5: DIFFERENT:"; head -8 /tmp/h5mn.txt; }
  diff OUTPUT_FILES_gpu/objective_function.txt OUTPUT_FILES_2node_gpu/objective_function.txt >/dev/null \
    && echo "objective_function.txt: IDENTICAL" || echo "objective_function.txt: DIFFERENT"
else
  echo "1-rank reference (OUTPUT_FILES_gpu) not present; skip diff"
fi

echo "=== MULTI-NODE TEST DONE ==="
