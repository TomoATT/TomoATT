#!/bin/bash
#SBATCH -p spark
#SBATCH -N 1
#SBATCH -J tatt_pmix
#SBATCH -t 03:00:00
#SBATCH -o %x_%j.out
# Rebuild OpenMPI with slurm PMI support so that `srun -n N` can launch multi-node
# (multi-GPU) ranks directly, without the broken orted cross-node startup.
set -e

PREFIX=$HOME/tomoatt_deps_pmix
WORK=$HOME/deps_build_src_pmix
mkdir -p $PREFIX $WORK
cd $WORK
NPROC=$(nproc)

echo "=== probe slurm PMI libraries ==="
ls -la /usr/lib/aarch64-linux-gnu/libpmi* /usr/lib/x86_64-linux-gnu/libpmi* /opt/slurm*/lib*/libpmi* /usr/local/slurm*/lib*/libpmi* 2>/dev/null || true
ls -la /usr/lib/aarch64-linux-gnu/slurm*/ 2>/dev/null | head || true
find / -name "libpmi2.so*" -o -name "libpmi.so*" 2>/dev/null | grep -v proc | head || true
which pmi2_info srun 2>/dev/null || true
dpkg -l | grep -i -E "slurm|pmi" | head || true

OMPI_DIR=openmpi-4.1.6
if [ ! -d $OMPI_DIR ]; then
  curl -LO https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
  tar xf openmpi-4.1.6.tar.gz
fi
cd $OMPI_DIR

# Try PMIx first (slurm's bundled pmix), fall back to PMI1/2 (--with-pmi)
CONFIGURED=0
for PMIX_PATH in /usr /usr/local /opt/slurm /opt/pmix; do
  if [ -f $PMIX_PATH/include/pmix.h ]; then
    echo "=== trying PMIx at $PMIX_PATH ==="
    ./configure --prefix=$PREFIX --without-cuda --with-pmix=$PMIX_PATH --with-slurm && CONFIGURED=1 && break
  fi
done
if [ $CONFIGURED -eq 0 ]; then
  echo "=== no PMIx found, trying PMI1/2 from slurm libpmi ==="
  PMI_LIBDIR=$(dirname $(find / -name "libpmi2.so*" 2>/dev/null | grep -v proc | head -1) 2>/dev/null || echo "")
  if [ -n "$PMI_LIBDIR" ]; then
    ./configure --prefix=$PREFIX --without-cuda --with-slurm --with-pmi=$PMI_LIBDIR/.. && CONFIGURED=1
  fi
fi

if [ $CONFIGURED -eq 0 ]; then
  echo "ERROR: could not configure OpenMPI with any PMI/PMIx; falling back to TCP-less mpirun usage"
  exit 2
fi

make -j$NPROC
make install

echo "=== sanity: srun single + multi-node rank check ==="
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
srun -p spark -N 1 -n 2 $PREFIX/bin/mpirun --version | head -1
cat > /tmp/hello_mpi.c << 'EOC'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  int r, n; char h[256]; int hl=256;
  MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&n);
  MPI_Get_processor_name(h,&hl);
  printf("rank %d/%d on %s\n", r, n, h);
  MPI_Finalize(); return 0;
}
EOC
$PREFIX/bin/mpicc /tmp/hello_mpi.c -o /tmp/hello_mpi
echo "--- srun -N2 -n2 across spark-double:"
srun -p spark-double -N 2 -n 2 /tmp/hello_mpi | sort

echo "=== PMIX MPI BUILD DONE ==="
