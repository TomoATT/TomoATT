#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J tomoatt_v2
#SBATCH -o %x_%j.out
# Build and test the clean-slate GPU backend (gpu/, USE_CUDA_V2) on a spark node
set -e

export DEPS=$HOME/tomoatt_deps
export PATH=$DEPS/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$DEPS/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

cd $HOME/TomoATT
rm -rf build_spark_v2
mkdir -p build_spark_v2
cd build_spark_v2

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=True \
  -DUSE_CUDA_V2=True \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_C_COMPILER=$DEPS/bin/mpicc \
  -DCMAKE_CXX_COMPILER=$DEPS/bin/mpicxx \
  -DHDF5_PREFER_PARALLEL=TRUE \
  -DHDF5_ROOT=$DEPS \
  -DTOMOATT_CUDA_ARCH=121 \
  -DCMAKE_CUDA_ARCHITECTURES=121

make -j$(nproc) tomogpu test_gpu_correctness test_gpu_memory bench_gpu_sweep bench_cpu_vs_gpu 2>&1 | tail -25

echo "=== V2 BUILD DONE ==="
ls -la gpu/tests/ gpu/benchmarks/ 2>/dev/null | grep -E "^-rwx|total" || true

echo "=== RUN test_gpu_memory ==="
./gpu/tests/test_gpu_memory

echo "=== RUN test_gpu_correctness ==="
./gpu/tests/test_gpu_correctness

echo "=== RUN bench_gpu_sweep ==="
./gpu/benchmarks/bench_gpu_sweep 2>&1 | tail -30

echo "=== RUN bench_cpu_vs_gpu ==="
./gpu/benchmarks/bench_cpu_vs_gpu 2>&1 | tail -30

echo "=== V2 TESTS DONE ==="
