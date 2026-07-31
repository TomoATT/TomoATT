#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J tomoatt_build
#SBATCH -o %x_%j.out
# Build TOMOATT with CUDA (sm_121 for GB10) on a spark node
set -e

export DEPS=$HOME/tomoatt_deps
export PATH=$DEPS/bin:/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$DEPS/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

cd $HOME/TomoATT
rm -rf build_spark
mkdir -p build_spark
cd build_spark

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=True \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_C_COMPILER=$DEPS/bin/mpicc \
  -DCMAKE_CXX_COMPILER=$DEPS/bin/mpicxx \
  -DHDF5_PREFER_PARALLEL=TRUE \
  -DHDF5_ROOT=$DEPS \
  -DTOMOATT_CUDA_ARCH=121

make -j$(nproc) 2>&1 | tail -30

echo "=== BUILD DONE ==="
ls -la bin/ 2>/dev/null
