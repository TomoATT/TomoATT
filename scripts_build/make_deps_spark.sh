#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J tomoatt_deps
#SBATCH -o %x_%j.out
# Build OpenMPI + parallel HDF5 for aarch64 (GB10 spark nodes)
set -e

export PREFIX=$HOME/tomoatt_deps
export WORK=$HOME/deps_build_src
mkdir -p $PREFIX $WORK
cd $WORK

NPROC=$(nproc)
echo "node: $(hostname), arch: $(uname -m), nproc: $NPROC"

# ---------- OpenMPI 4.1.6 ----------
if [ ! -f $PREFIX/bin/mpicxx ]; then
  if [ ! -d openmpi-4.1.6 ]; then
    curl -LO https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
    tar xf openmpi-4.1.6.tar.gz
  fi
  cd openmpi-4.1.6
  ./configure --prefix=$PREFIX --without-cuda
  make -j$NPROC
  make install
  cd $WORK
fi

# ---------- HDF5 1.13.3 (parallel) ----------
if [ ! -f $PREFIX/bin/h5pcc ]; then
  if [ ! -d hdf5-1.13.3 ]; then
    curl -LO https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-1.13.3/src/hdf5-1.13.3.tar.gz
    tar xf hdf5-1.13.3.tar.gz
  fi
  cd hdf5-1.13.3
  export PATH=$PREFIX/bin:$PATH
  CC=$PREFIX/bin/mpicc CXX=$PREFIX/bin/mpicxx ./configure \
    --enable-parallel --enable-unsupported --enable-shared --enable-cxx \
    --prefix=$PREFIX
  make -j$NPROC
  make install
  cd $WORK
fi

echo "=== DEPS DONE ==="
ls $PREFIX/bin | grep -E "^(mpi|h5)" || true
