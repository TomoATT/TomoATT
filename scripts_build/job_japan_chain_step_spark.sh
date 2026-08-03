#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --gres=gpu:GB10:1
#SBATCH --time=07:30:00
#SBATCH -o %x_%j.out
#
# Iteration-chaining restart driver for the host-memory-leak workaround:
# runs ONE inversion iteration (<65 GB peak instead of the 125 GB OOM point
# around iteration 2+), then resubmits itself for the next iteration with
# the freshly written final_model.h5 as the next init model.
#
# sbatch -p spark -N 1 -w <node> --export=YML=<base yml>,OUTDIR=<output dir>,STEPS=<total run steps to go>,STEP=<current step nr>,JNAME=<name> scripts_build/job_japan_chain_step_spark.sh
set -e

export PATH=$HOME/tomoatt_deps/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tomoatt_deps/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

BIN=$HOME/TomoATT/build_spark/bin/TOMOATT
CASE=$HOME/TomoATT/examples/realcase_japan_tomography
cd $CASE/3_input_params

: "${YML:?missing}"; : "${OUTDIR:?missing}"; : "${STEPS:?missing}"; : "${STEP:?missing}"
: "${JNAME:=chain}"

FINAL=$CASE/$OUTDIR/final_model.h5
if [ "$STEP" -gt 0 ]; then
  # previous chained step must have produced the model
  test -s "$FINAL"
  cp "$FINAL" $CASE/2_data_processing/chain_model_${JNAME}.h5
fi

VAR=input_chain_${JNAME}.yml
python3 - << PYEOF
import re
t = open("$YML").read()
def setv(t, key, val):
    if re.search(r'^\s*'+key+r'\s*:', t, flags=re.M):
        return re.sub(r'^(\s*)'+key+r'\s*:.*$', r'\g<1>'+key+': '+val, t, flags=re.M)
    return t
t = setv(t, 'use_gpu', 'true')
t = setv(t, 'nproc_sub', '1')
t = setv(t, 'ndiv_rtp', '[1, 1, 1]')
if $STEP > 0:
    t = setv(t, 'init_model_path', '../2_data_processing/chain_model_${JNAME}.h5')
# cap ALL max_iterations to 1 (tolerant of trailing comments), then restore
# the solver-side block in 'calculation:' back to 500 so only the inversion
# max_iterations (model_update) is capped to one iteration per chained job.
t = re.sub(r'^(\s*)max_iterations:\s*[0-9]+.*$',
           r'\g<1>max_iterations: 1', t, flags=re.M)
t = t.replace("calculation:\n  convergence_tolerance: 1.0e-4\n  max_iterations: 1",
              "calculation:\n  convergence_tolerance: 1.0e-4\n  max_iterations: 500")
open("$VAR", "w").write(t)
PYEOF

echo "=== chain step $STEP/$STEPS ($JNAME) ==="
grep -nE "init_model_path|use_gpu|max_iterations:" $VAR | head -4
time $BIN -i $VAR
echo "=== STEP DONE rc=$? ==="
test -s "$FINAL"

NEXT=$((STEP+1))
if [ "$NEXT" -le "$STEPS" ]; then
  cd ~/TomoATT
  sbatch -p spark -N 1 -w $SLURM_JOB_NODELIST \
    --gres=gpu:GB10:1 \
    --export=YML=$YML,OUTDIR=$OUTDIR,STEPS=$STEPS,STEP=$NEXT,JNAME=$JNAME \
    --time=07:30:00 -J ${JNAME}$NEXT \
    scripts_build/job_japan_chain_step_spark.sh
  echo "submitted next chain step $NEXT"
else
  echo "chain complete: $STEPS steps finished"
fi
