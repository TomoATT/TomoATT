#!/bin/bash
#SBATCH -p spark-double
#SBATCH -N 2
#SBATCH -J tatt_mpistack
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
# Build slurm-native MPI stack for multi-node GPU runs:
#   hwloc -> PMIx -> OpenMPI (--with-pmix --with-slurm) -> relink TOMOATT
# Then sanity-test `srun -N 2` across the two spark-double nodes.
set -e

S=$HOME/mpi_stack
SRC=$HOME/mpi_stack_src
mkdir -p $S $SRC
cd $SRC
NPROC=$(nproc)
echo "node: $(hostname)"

# ---------- libevent 2.2 (PMIx 5 needs event_getcode4name; Ubuntu has 2.1.12) ----------
if [ ! -f $S/libevent/lib/libevent.so ]; then
  if [ ! -d libevent-2.2.1-alpha-dev ]; then
    curl -LO https://github.com/libevent/libevent/releases/download/release-2.2.1-alpha-dev/libevent-2.2.1-alpha-dev.tar.gz
    tar xf libevent-2.2.1-alpha-dev.tar.gz
  fi
  cd libevent-2.2.1-alpha-dev
  ./configure --prefix=$S/libevent --disable-openssl --disable-samples >/dev/null
  make -j$NPROC >/dev/null && make install >/dev/null
  cd $SRC
  echo "--- libevent done"
fi

# ---------- hwloc 2.11.1 ----------
if [ ! -f $S/hwloc/lib/libhwloc.so ]; then
  if [ ! -d hwloc-2.11.1 ]; then
    curl -LO https://download.open-mpi.org/release/hwloc/v2.11/hwloc-2.11.1.tar.gz
    tar xf hwloc-2.11.1.tar.gz
  fi
  cd hwloc-2.11.1
  ./configure --prefix=$S/hwloc --disable-io --disable-cairo >/dev/null
  make -j$NPROC >/dev/null && make install >/dev/null
  cd $SRC
  echo "--- hwloc done"
fi

# ---------- PMIx 5.0.1 ----------
if [ ! -f $S/pmix/lib/libpmix.so ]; then
  if [ ! -d pmix-5.0.1 ]; then
    curl -LO https://github.com/openpmix/openpmix/releases/download/v5.0.1/pmix-5.0.1.tar.gz
    tar xf pmix-5.0.1.tar.gz
  fi
  cd pmix-5.0.1
  ./configure --prefix=$S/pmix \
    --with-libevent=$S/libevent \
    --with-hwloc=$S/hwloc \
    --disable-man-pages >/dev/null
  make -j$NPROC >/dev/null && make install >/dev/null
  cd $SRC
  echo "--- pmix done"
fi

# ---------- OpenMPI 4.1.6 with PMIx + slurm ----------
if [ ! -f $S/ompi/bin/mpicxx ]; then
  if [ ! -d openmpi-4.1.6 ]; then
    curl -LO https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
    tar xf openmpi-4.1.6.tar.gz
  fi
  cd openmpi-4.1.6
  ./configure --prefix=$S/ompi \
    --without-cuda \
    --with-slurm \
    --with-pmix=$S/pmix \
    --with-libevent=$S/libevent \
    --with-hwloc=$S/hwloc >/dev/null
  make -j$NPROC >/dev/null && make install >/dev/null
  cd $SRC
  echo "--- openmpi done"
fi

export PATH=$S/ompi/bin:$PATH
export LD_LIBRARY_PATH=$S/ompi/lib:$S/pmix/lib:$S/hwloc/lib:$S/libevent/lib:$LD_LIBRARY_PATH

# ---------- hello MPI over srun -N2 ----------
cat > /tmp/hello_mpi.c << 'EOC'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  int r,n; char h[256]; int hl=256;
  MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&n);
  MPI_Get_processor_name(h,&hl);
  printf("rank %d/%d on %s\n",r,n,h);
  MPI_Finalize(); return 0;
}
EOC
$S/ompi/bin/mpicc /tmp/hello_mpi.c -o /tmp/hello_mpi
echo "=== srun -N2 -n2 hello ==="
srun -N 2 -n 2 --mpi=pmix_v5 /tmp/hello_mpi | sort
echo "=== srun -N2 -n4 hello ==="
srun -N 2 -n 4 --mpi=pmix_v5 /tmp/hello_mpi | sort

# ---------- relink TOMOATT against the new MPI ----------
cd $HOME/TomoATT
rm -rf build_spark_mpi && mkdir -p build_spark_mpi && cd build_spark_mpi
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=True \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_C_COMPILER=$S/ompi/bin/mpicc \
  -DCMAKE_CXX_COMPILER=$S/ompi/bin/mpicxx \
  -DMPI_C_COMPILER=$S/ompi/bin/mpicc \
  -DMPI_CXX_COMPILER=$S/ompi/bin/mpicxx \
  -DHDF5_PREFER_PARALLEL=TRUE \
  -DHDF5_ROOT=$HOME/tomoatt_deps \
  -DTOMOATT_CUDA_ARCH=121 >/dev/null
make -j$NPROC 2>&1 | tail -4
echo "--- tomoatt relinked"
ls -la bin/TOMOATT

echo "=== MPI STACK DONE ==="
