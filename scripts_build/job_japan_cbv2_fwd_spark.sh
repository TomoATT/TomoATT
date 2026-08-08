#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --gres=gpu:GB10:1
#SBATCH --time=02:00:00
#SBATCH -J cbv2f
#SBATCH -o %x_%j.out
# Resolution-attack checkerboard: forward synthesis (v1 regime, ~30 min).
set -e

export PATH=$HOME/tomoatt_deps/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
CASE=$HOME/TomoATT/examples/realcase_japan_tomography
cd $CASE/3_input_params

VAR=input_params_cbv2_forward_vel.yml
grep -nE "src_rec_file|init_model_path|swap_src_rec|use_gpu" $VAR
time $BIN -i $VAR
echo "=== FWD DONE rc=$? ==="
ls -lh $CASE/OUTPUT_FILES_cbv2_forward_vel/src_rec_file_forward.dat
