#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --gres=gpu:GB10:1
#SBATCH --time=15:00:00
#SBATCH -o %x_%j.out
# Resolution-attack checkerboard: recovery run for one yml (export VAR).
# Single-process: v1-style swap config is leak-free (~1100 solver sources).
# usage: sbatch -p spark -N 1 -w <node> --export=VAR=input_params_cbv2_inv_R1bfgs.yml scripts_build/job_japan_cbv2_inv_spark.sh
set -e

export PATH=$HOME/tomoatt_deps/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
CASE=$HOME/TomoATT/examples/realcase_japan_tomography
cd $CASE/3_input_params

: "${VAR:?missing yml name}"
grep -nE "src_rec_file|swap_src_rec|output_dir|optim_method|Kdensity_coe|use_gpu|dep_inv" $VAR | head -8
# precondition: forward data must exist
test -s $CASE/OUTPUT_FILES_cbv2_forward_vel/src_rec_file_forward.dat
time $BIN -i $VAR
echo "=== RECOVERY DONE rc=$? ==="
ls -la $CASE/$(grep -oE "output_dir:.*$" $VAR | awk '{print $2}' | sed 's|\.\./||') | head
