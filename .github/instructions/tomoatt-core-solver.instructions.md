---
description: "Use when editing TomoATT core solver, grid, iterator, MPI, SIMD, CUDA, or inversion-kernel code. Covers architecture, shared-memory invariants, and performance-sensitive sweep paths."
applyTo:
  - include/**
  - src/**
  - cuda/**
---
# TomoATT Core Solver

- Treat TomoATT as a monolithic C++17 executable build: each app target links almost all shared source files, so declaration and definition drift usually breaks the whole project, not just one executable.
- The core ownership split is stable: `Grid` owns geometry, field arrays, boundary buffers, inversion arrays, and MPI shared-memory windows; `Iterator*` classes own sweep order and numerical updates; `InputParams` and `IO_utils` own runtime configuration and file I/O.
- Keep hot-path edits minimal in `src/iterator.cpp`, `src/iterator_level.cpp`, `include/vectorized_sweep.h`, and `cuda/`. Avoid extra allocations, container churn, logging, or repeated transcendental work inside sweep loops unless the performance tradeoff is deliberate.
- Any new `Grid` array or MPI shared-memory window must be updated symmetrically in `include/grid.h`, `Grid::init_mpi_wins`, `Grid::cleanup_mpi_wins`, `Grid::memory_allocation`, `Grid::memory_deallocation`, `Grid::shm_memory_allocation`, and `Grid::shm_memory_deallocation`.
- Preserve the flattened `I2V(i,j,k)` storage assumptions when refactoring loops. Many routines rely on contiguous one-dimensional storage for performance, MPI exchange, and SIMD/GPU preloading.
- `SWEEP_TYPE_LEVEL` is the only mode that supports `n_subprocs > 1` and the SIMD/GPU paths. Upwind sweeps and adjoint sweeps remain scalar, so do not assume feature parity across all iterator variants.
- Teleseismic acceleration support is partial in this codebase. If you touch shared logic used by regional and teleseismic paths, verify that the scalar teleseismic path still builds and behaves sensibly.
- When optimizing arithmetic in sweep kernels, keep scalar and accelerated paths numerically aligned enough for regression tests and avoid changing units or sign conventions implicitly.