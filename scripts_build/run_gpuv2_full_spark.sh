#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J tomoatt_v2run
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
# Run ALL V2 (gpu/ backend) tests and benchmarks with a generous time limit
set -e

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
cd $HOME/TomoATT/build_spark_v2

echo "=== test_gpu_memory ==="
./bin/test_gpu_memory

echo "=== test_gpu_correctness ==="
./bin/test_gpu_correctness

echo "=== bench_gpu_sweep ==="
./bin/bench_gpu_sweep

echo "=== bench_cpu_vs_gpu (full sizes 32..256) ==="
./bin/bench_cpu_vs_gpu

echo "=== V2 FULL RUN DONE ==="
