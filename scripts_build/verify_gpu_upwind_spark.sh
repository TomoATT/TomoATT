#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J tomoatt_verify
#SBATCH -o %x_%j.out
# Verify GPU UPWIND solver vs CPU on the inversion_small test
set -e

export DEPS=$HOME/tomoatt_deps
export PATH=$DEPS/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$DEPS/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
TEST=$HOME/TomoATT/test/inversion_small
cd $TEST

echo "### node: $(hostname)"
nvidia-smi -L || true

set_gpu_flag () {
  local flag=$1
  for YML in input_params.yml input_params_pre.yml; do
    if grep -qE "^\s*use_gpu:" $YML; then
      # replace existing (indentation-preserving)
      sed -i -E "s/^(\s*)use_gpu: .*/\1use_gpu: ${flag} # toggled by verify script/" $YML
    else
      # key missing: insert right after the "parallel:" line
      sed -i -E "/^parallel:/a\\  use_gpu: ${flag} # toggled by verify script" $YML
    fi
  done
  echo "--- use_gpu now:"; grep -nE "^\s*use_gpu:" input_params.yml input_params_pre.yml
}

run_case () {
  local tag=$1   # cpu | gpu
  local gpuflag=$2
  set_gpu_flag $gpuflag
  rm -f cuda_device_info.txt
  # true traveltime (pre) run. Direct binary (MPI singleton): mpirun under this
  # cluster's slurm allocation spans spark-atom-0<->dgx-spark-N and ORTE dies.
  $BIN -i input_params_pre.yml > ${tag}_pre.log 2>&1
  # main inversion run
  $BIN -i input_params.yml > ${tag}_main.log 2>&1
  # proof that the GPU code path executed
  if [ "$gpuflag" = "true" ]; then
    if [ -f cuda_device_info.txt ]; then
      echo ">>> CUDA device info written (GPU engaged):"
      head -3 cuda_device_info.txt
    else
      echo ">>> FATAL: cuda_device_info.txt NOT written -- GPU path never ran!"
      exit 1
    fi
  else
    if [ -f cuda_device_info.txt ]; then
      echo ">>> UNEXPECTED: cpu run produced cuda_device_info.txt"; exit 1
    else
      echo ">>> CPU run: no cuda_device_info.txt (as expected)"
    fi
  fi
  # stash outputs
  rm -rf OUTPUT_FILES_${tag}
  cp -r OUTPUT_FILES OUTPUT_FILES_${tag}
}

# ---- CPU baseline ----
echo "=== CPU run ==="
run_case cpu false

# ---- GPU run ----
echo "=== GPU run ==="
nvidia-smi -L >/dev/null 2>&1 || { echo "NO GPU ON THIS NODE"; exit 1; }
run_case gpu true

# both cases must have actually flipped the flag
grep -qE "^\s*use_gpu: true" input_params.yml || { echo "yml flag not true"; exit 1; }

# ---- compare ----
echo "===== ITERATION COUNTS (per-source forward solves) ====="
echo "-- CPU pre:"; grep -E "converged at iteration" cpu_pre.log | sort | uniq -c || true
echo "-- GPU pre:"; grep -E "converged at iteration" gpu_pre.log | sort | uniq -c || true
echo "-- CPU main:"; grep -E "converged at iteration" cpu_main.log | sort | uniq -c || true
echo "-- GPU main:"; grep -E "converged at iteration" gpu_main.log | sort | uniq -c || true

echo "===== H5 DIFFS (CPU vs GPU) ====="
for f in final_model.h5 out_data_sim_group_0.h5; do
  if [ -f OUTPUT_FILES_cpu/$f ] && [ -f OUTPUT_FILES_gpu/$f ]; then
    $DEPS/bin/h5diff -d 1e-8 OUTPUT_FILES_cpu/$f OUTPUT_FILES_gpu/$f >/tmp/h5d.txt 2>&1 \
      && echo "--- $f: IDENTICAL within 1e-8" \
      || { echo "--- $f: DIFFERENT (>1e-8), first lines:"; head -10 /tmp/h5d.txt; }
  else
    echo "--- $f: missing in one of the outputs"
  fi
done

$DEPS/bin/h5diff -d 1e-6 OUTPUT_FILES_cpu/out_data_sim_group_0.h5 OUTPUT_FILES_gpu/out_data_sim_group_0.h5 >/dev/null 2>&1 \
  && echo "--- out_data_sim_group_0.h5: IDENTICAL within 1e-6" || echo "--- out_data_sim_group_0.h5: DIFFERENT (>1e-6)"

diff OUTPUT_FILES_cpu/objective_function.txt OUTPUT_FILES_gpu/objective_function.txt >/dev/null \
  && echo "--- objective_function.txt: IDENTICAL" || echo "--- objective_function.txt: DIFFERENT"

echo "=== VERIFY DONE ==="
