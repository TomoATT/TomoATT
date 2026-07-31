#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J tatt_v3rd
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out
# Verify GPU 3rd-order LF stencil vs CPU (inversion_small)
set -e

export DEPS=$HOME/tomoatt_deps
export PATH=$DEPS/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$DEPS/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
TEST=$HOME/TomoATT/test/inversion_small
cd $TEST

echo "### node: $(hostname)"
nvidia-smi -L || true

# always restore the baseline flags at exit, even on crash
restore_baseline () {
  for YML in input_params.yml input_params_pre.yml; do
    grep -qE "^\s*stencil_order\s*:" $YML && sed -i -E "s/^(\s*)stencil_order\s*:.*/\1stencil_order: 1 # order of stencil, 1 or 3/" $YML
    grep -qE "^\s*use_gpu\s*:" $YML      && sed -i -E "s/^(\s*)use_gpu\s*:.*/\1use_gpu: false # true if use gpu (EXPERIMENTAL)/" $YML
  done
}
trap restore_baseline EXIT
# normalize before starting (idempotent)
restore_baseline

set_flags () {
  local flag=$1
  # use_gpu toggle (insert if missing)
  for YML in input_params.yml input_params_pre.yml; do
    if grep -qE "^\s*use_gpu:" $YML; then
      sed -i -E "s/^(\s*)use_gpu: .*/\1use_gpu: ${flag} # toggled by verify script/" $YML
    else
      sed -i -E "/^parallel:/a\\  use_gpu: ${flag} # toggled by verify script" $YML
    fi
    # stencil_order -> 3, stencil_type -> 0 (LF) so we exercise the real LF-3rd solver
    if grep -qE "^\s*stencil_order\s*:" $YML; then
      sed -i -E "s/^(\s*)stencil_order\s*:.*/\1stencil_order: 3 # toggled by verify script/" $YML
    fi
    if grep -qE "^\s*stencil_type\s*:" $YML; then
      sed -i -E "s/^(\s*)stencil_type\s*:.*/\1stencil_type: 0 # toggled by verify script/" $YML
    fi
  done
  echo "--- flags now:"; grep -nE "^\s*(use_gpu|stencil_order|stencil_type)\s*:" input_params.yml input_params_pre.yml
}

run_case () {
  local tag=$1 gpuflag=$2
  set_flags $gpuflag
  rm -f cuda_device_info.txt
  $BIN -i input_params_pre.yml > ${tag}_3rd_pre.log 2>&1
  $BIN -i input_params.yml     > ${tag}_3rd_main.log 2>&1
  if [ "$gpuflag" = "true" ] && [ ! -f cuda_device_info.txt ]; then
    echo ">>> FATAL: GPU path never ran"; exit 1
  fi
  rm -rf OUTPUT_FILES_${tag}_3rd
  cp -r OUTPUT_FILES OUTPUT_FILES_${tag}_3rd
}

echo "=== CPU 3rd-order run ==="
run_case cpu false
echo "=== GPU 3rd-order run ==="
run_case gpu true

echo "===== ITERATION COUNTS (3rd order) ====="
echo "-- CPU pre:"; grep -E "converged at iteration" cpu_3rd_pre.log | sort | uniq -c || true
echo "-- GPU pre:"; grep -E "converged at iteration" gpu_3rd_pre.log | sort | uniq -c || true
echo "-- CPU main:"; grep -E "converged at iteration" cpu_3rd_main.log | sort | uniq -c || true
echo "-- GPU main:"; grep -E "converged at iteration" gpu_3rd_main.log | sort | uniq -c || true

echo "===== H5 DIFFS (3rd order, CPU vs GPU) ====="
for f in final_model.h5 out_data_sim_group_0.h5; do
  if [ -f OUTPUT_FILES_cpu_3rd/$f ] && [ -f OUTPUT_FILES_gpu_3rd/$f ]; then
    $DEPS/bin/h5diff -d 1e-8 OUTPUT_FILES_cpu_3rd/$f OUTPUT_FILES_gpu_3rd/$f >/tmp/h5d3.txt 2>&1 \
      && echo "--- $f: IDENTICAL within 1e-8" \
      || { echo "--- $f: DIFFERENT (>1e-8), first lines:"; head -8 /tmp/h5d3.txt; }
  else
    echo "--- $f: missing"
  fi
done
diff OUTPUT_FILES_cpu_3rd/objective_function.txt OUTPUT_FILES_gpu_3rd/objective_function.txt >/dev/null \
  && echo "--- objective_function.txt: IDENTICAL" || echo "--- objective_function.txt: DIFFERENT"

# flags restored by the EXIT trap

echo "=== 3RD ORDER VERIFY DONE ==="
