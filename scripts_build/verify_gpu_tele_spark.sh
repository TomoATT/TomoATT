#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J tatt_vtel
#SBATCH -t 02:00:00
#SBATCH --exclude=spark-edge-0
#SBATCH -o %x_%j.out
# Verify teleseismic UPWIND GPU solver vs CPU on a synthetic mini-tele case.
# Teleseismic sources are auto-detected when a source lies outside the domain,
# so we generate src_rec_test_tele.dat with the first 3 events moved far outside.
set -e

export DEPS=$HOME/tomoatt_deps
export PATH=$DEPS/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$DEPS/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
TEST=$HOME/TomoATT/test/inversion_small
cd $TEST

echo "### node: $(hostname)"
nvidia-smi -L || { echo "NO GPU on $(hostname), abort"; exit 1; }

# ---- make mini-tele src_rec: move first 3 source events far outside the domain ----
awk 'BEGIN{OFS=" "}
     /src_/ && count < 3 && $1 < 3 { $8=30.0; $9=35.0; $10=500.0; count++; print; next }
     { print }' src_rec_test.dat > src_rec_test_tele.dat
echo "--- tele-ized event headers:"; grep "src_" src_rec_test_tele.dat | head -4

restore_baseline () {
  for YML in input_params.yml input_params_pre.yml; do
    grep -qE "^\s*src_rec_file\s*:" $YML && sed -i -E "s|src_rec_file: [^ ]*|src_rec_file: src_rec_test.dat|" $YML
    grep -qE "^\s*use_gpu\s*:" $YML && sed -i -E "s/^(\s*)use_gpu\s*:.*/\1use_gpu: false # restored/" $YML
    grep -qE "^\s*have_tele_data\s*:" $YML && sed -i -E "/^\s*have_tele_data\s*:/d" $YML
  done
}
trap restore_baseline EXIT

set_flags () {
  local flag=$1
  for YML in input_params_pre.yml; do
    # point to the tele src_rec (pre run only; this is a forward-only test)
    grep -qE "^\s*src_rec_file\s*:" $YML && sed -i -E "s|src_rec_file: [^ ]*|src_rec_file: src_rec_test_tele.dat|" $YML
    # stencil: 1st order upwind (picks the tele upwind class for out-of-region sources)
    grep -qE "^\s*stencil_order\s*:" $YML && sed -i -E "s/^(\s*)stencil_order\s*:.*/\1stencil_order: 1/" $YML
    grep -qE "^\s*stencil_type\s*:"  $YML && sed -i -E "s/^(\s*)stencil_type\s*:.*/\1stencil_type: 1/" $YML
    # no swap for tele
    grep -qE "^\s*swap_src_rec\s*:" $YML && sed -i -E "s/^(\s*)swap_src_rec\s*:.*/\1swap_src_rec: false/" $YML
    # teleseismic data flag (required when sources lie outside the region)
    if grep -qE "^\s*have_tele_data\s*:" $YML; then
      sed -i -E "s/^(\s*)have_tele_data\s*:.*/\1have_tele_data: true/" $YML
    else
      # insert at top (top-level yaml key)
      sed -i "1i have_tele_data: true" $YML
    fi
    # gpu toggle
    if grep -qE "^\s*use_gpu:" $YML; then
      sed -i -E "s/^(\s*)use_gpu: .*/\1use_gpu: ${flag} # toggled/" $YML
    else
      sed -i -E "/^parallel:/a\\  use_gpu: ${flag} # toggled" $YML
    fi
  done
  grep -nE "^\s*(use_gpu|stencil_order|stencil_type|swap_src_rec)\s*:|src_rec_file:" input_params_pre.yml
}

run_case () {
  local tag=$1 gpuflag=$2
  set_flags $gpuflag
  rm -f cuda_device_info.txt
  rm -rf OUTPUT_FILES
  set +e
  timeout 1800 $BIN -i input_params_pre.yml > tele_${tag}.log 2>&1
  rc=$?
  set -e
  echo ">>> ${tag} run exit code: ${rc}"
  grep -iE "teleseismic|out.of.region|tele" tele_${tag}.log | head -5 || true
  grep -E "converged at iteration" tele_${tag}.log | sort | uniq -c | head -8 || true
  if [ "$gpuflag" = "true" ] && [ ! -f cuda_device_info.txt ]; then
    echo ">>> FATAL: GPU path never ran"; exit 1
  fi
  rm -rf OUTPUT_FILES_tele_${tag}
  cp -r OUTPUT_FILES OUTPUT_FILES_tele_${tag}
}

echo "=== CPU tele run ==="
run_case cpu false
echo "=== GPU tele run ==="
run_case gpu true

echo "===== H5 DIFFS (tele CPU vs GPU) ====="
for f in out_data_sim_group_0.h5 final_model.h5; do
  if [ -f OUTPUT_FILES_tele_cpu/$f ] && [ -f OUTPUT_FILES_tele_gpu/$f ]; then
    $DEPS/bin/h5diff -d 1e-8 OUTPUT_FILES_tele_cpu/$f OUTPUT_FILES_tele_gpu/$f >/tmp/h5dt.txt 2>&1 \
      && echo "--- $f: IDENTICAL within 1e-8" \
      || { echo "--- $f: DIFFERENT (>1e-8):"; head -10 /tmp/h5dt.txt; }
  else
    echo "--- $f: MISSING (run failed before output?)"
  fi
done

echo "=== TELE VERIFY DONE ==="
