#!/bin/bash
# ===========================================================================
# benchmark_inversion_small.sh
#
# Benchmark CPU vs GPU v2 using the inversion_small test case.
# Varies grid resolution to test scaling:
#   - Small: n_rtp [10, 50, 50]  (default)
#   - Medium: n_rtp [20, 100, 100]
#   - Large: n_rtp [30, 150, 150]
#
# Measures:
#   1. Time to completion (wall-clock)
#   2. RAM consumption (peak RSS)
#   3. Traveltime field errors (CPU vs GPU)
#
# Usage:
#   ./benchmark_inversion_small.sh [nproc_sweep] [nproc_dd]
#
# ===========================================================================

set -e

# Configuration
NPROC_SWEEP=${1:-1}
NPROC_DD=${2:-1}
NPROC_TOTAL=$((NPROC_SWEEP*NPROC_DD*NPROC_DD*NPROC_DD))

# Grid configurations: "label" "nr" "nt" "np"
GRIDS=(
    "small 10 50 50"
    "medium 20 100 100"
    "large 30 150 150"
)

# Output directory
OUTPUT_DIR="benchmark_results"
mkdir -p ${OUTPUT_DIR}

# Build directory (relative to test/inversion_small)
BUILD_DIR="../../build"
TOMOATT_BIN="${BUILD_DIR}/bin/TOMOATT"

# Check if TOMOATT binary exists
if [ ! -f "${TOMOATT_BIN}" ]; then
    echo "Error: TOMOATT binary not found at ${TOMOATT_BIN}"
    echo "Please build TomoATT first: cd .. && mkdir build && cd build && cmake .. && make"
    exit 1
fi

echo "============================================================"
echo "  CPU vs GPU v2 Benchmark (inversion_small)"
echo "============================================================"
echo "NPROC_SWEEP: ${NPROC_SWEEP}"
echo "NPROC_DD: ${NPROC_DD}"
echo "NPROC_TOTAL: ${NPROC_TOTAL}"
echo "TOMOATT binary: ${TOMOATT_BIN}"
echo ""

# Step 1: Create test model
echo "Step 1: Creating test model..."
cd test/inversion_small
python make_test_model.py
echo "Test model created."
echo ""

# Results file
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results.csv"
echo "grid_label,nr,nt,np,n_nodes,cpu_time_s,cpu_rss_mb,gpu_time_s,gpu_rss_mb,l1_err,linf_err" > ${RESULTS_FILE}

# Step 2: Run benchmarks for each grid configuration
for grid_config in "${GRIDS[@]}"; do
    # Parse grid configuration
    read -r label nr nt np <<< "${grid_config}"

    echo "============================================================"
    echo "  Grid: ${label} (${nr}x${nt}x${np})"
    echo "============================================================"

    # Create temporary YAML files with modified grid size
    for yaml_file in input_params_pre.yml input_params.yml; do
        cp ${yaml_file} ${yaml_file}.bak
    done

    # Modify grid size in YAML files
    for yaml_file in input_params_pre.yml input_params.yml; do
        # Update n_rtp
        if grep -q "n_rtp:" ${yaml_file}; then
            sed -i "s/n_rtp: \[.*\]/n_rtp: [${nr}, ${nt}, ${np}]/" ${yaml_file}
        fi
        # Update parallel settings
        sed -i "s/nproc_sub: .*/nproc_sub: ${NPROC_SWEEP}/" ${yaml_file}
        sed -i "s/ndiv_rtp: \[.*\]/ndiv_rtp: [${NPROC_DD}, ${NPROC_DD}, ${NPROC_DD}]/" ${yaml_file}
    done

    # Calculate total nodes
    n_nodes=$((nr * nt * np))
    echo "Total nodes: ${n_nodes}"

    # --- CPU Run (forward simulation for true travel times) ---
    echo ""
    echo "Running CPU forward simulation (${label})..."
    rm -rf OUTPUT_FILES
    mkdir -p OUTPUT_FILES

    cpu_start=$(date +%s.%N)
    mpirun --oversubscribe -n ${NPROC_TOTAL} ${TOMOATT_BIN} -i input_params_pre.yml 2>&1 | tee ${OUTPUT_DIR}/cpu_${label}.log
    cpu_end=$(date +%s.%N)

    cpu_time=$(echo "${cpu_end} - ${cpu_start}" | bc)
    echo "CPU time: ${cpu_time} s"

    # Get CPU memory (from /proc/meminfo before and after, or use ulimit)
    cpu_rss=$(grep "Maximum resident set size" ${OUTPUT_DIR}/cpu_${label}.log 2>/dev/null | awk '{print $NF/1024}' || echo "0")
    if [ "${cpu_rss}" = "0" ] || [ -z "${cpu_rss}" ]; then
        cpu_rss=$(free -m | awk '/Mem:/ {print $3}')
    fi

    # Save CPU output for comparison
    cp -r OUTPUT_FILES ${OUTPUT_DIR}/cpu_output_${label}

    # --- GPU Run (forward simulation for true travel times) ---
    echo ""
    echo "Running GPU v2 forward simulation (${label})..."

    # Modify input_params_pre.yml to use GPU
    sed -i "s/use_gpu: false/use_gpu: true/" input_params_pre.yml

    rm -rf OUTPUT_FILES
    mkdir -p OUTPUT_FILES

    gpu_start=$(date +%s.%N)
    mpirun --oversubscribe -n ${NPROC_TOTAL} ${TOMOATT_BIN} -i input_params_pre.yml 2>&1 | tee ${OUTPUT_DIR}/gpu_${label}.log
    gpu_end=$(date +%s.%N)

    gpu_time=$(echo "${gpu_end} - ${gpu_start}" | bc)
    echo "GPU time: ${gpu_time} s"

    # Get GPU memory
    gpu_rss=$(grep "Maximum resident set size" ${OUTPUT_DIR}/gpu_${label}.log 2>/dev/null | awk '{print $NF/1024}' || echo "0")
    if [ "${gpu_rss}" = "0" ] || [ -z "${gpu_rss}" ]; then
        gpu_rss=$(free -m | awk '/Mem:/ {print $3}')
    fi

    # Save GPU output for comparison
    cp -r OUTPUT_FILES ${OUTPUT_DIR}/gpu_output_${label}

    # --- Compare CPU and GPU outputs ---
    echo ""
    echo "Comparing CPU and GPU outputs..."

    # Find the T field files
    cpu_t_file=$(find ${OUTPUT_DIR}/cpu_output_${label} -name "T_*.dat" -o -name "T_*.h5" | head -1)
    gpu_t_file=$(find ${OUTPUT_DIR}/gpu_output_${label} -name "T_*.dat" -o -name "T_*.h5" | head -1)

    if [ -n "${cpu_t_file}" ] && [ -n "${gpu_t_file}" ]; then
        echo "CPU T file: ${cpu_t_file}"
        echo "GPU T file: ${gpu_t_file}"

        # Compare using Python
        python3 -c "
import numpy as np
import h5py
import sys

# Read CPU T field
cpu_file = '${cpu_t_file}'
gpu_file = '${gpu_t_file}'

# Try HDF5 first
try:
    with h5py.File(cpu_file, 'r') as f:
        cpu_T = f['T'][:]
    with h5py.File(gpu_file, 'r') as f:
        gpu_T = f['T'][:]
except:
    # Try ASCII
    cpu_T = np.loadtxt(cpu_file)
    gpu_T = np.loadtxt(gpu_file)

# Compute errors
l1_err = np.mean(np.abs(cpu_T - gpu_T))
linf_err = np.max(np.abs(cpu_T - gpu_T))

print(f'L1 Error: {l1_err:.6e}')
print(f'Linf Error: {linf_err:.6e}')
print(f'L1 Error: {l1_err:.6e}' > '${OUTPUT_DIR}/errors_${label}.txt')
print(f'Linf Error: {linf_err:.6e}' >> '${OUTPUT_DIR}/errors_${label}.txt')
" 2>&1 | tee -a ${OUTPUT_DIR}/comparison_${label}.log

        l1_err=$(grep "L1 Error" ${OUTPUT_DIR}/errors_${label}.txt | awk '{print $3}')
        linf_err=$(grep "Linf Error" ${OUTPUT_DIR}/errors_${label}.txt | awk '{print $3}')
    else
        echo "Warning: T field files not found. Skipping comparison."
        l1_err="N/A"
        linf_err="N/A"
    fi

    # Write results to CSV
    echo "${label},${nr},${nt},${np},${n_nodes},${cpu_time},${cpu_rss},${gpu_time},${gpu_rss},${l1_err},${linf_err}" >> ${RESULTS_FILE}

    # Restore original YAML files
    for yaml_file in input_params_pre.yml input_params.yml; do
        mv ${yaml_file}.bak ${yaml_file}
    done

    echo ""
done

# Step 3: Generate comparison report
echo "============================================================"
echo "  Benchmark Complete"
echo "============================================================"
echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""
cat ${RESULTS_FILE}
