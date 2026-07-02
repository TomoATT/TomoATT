# Migration Guide: Legacy GPU Backend → Clean-Slate v2

This guide walks through migrating from the legacy `cuda/` GPU backend to the new `gpu/` clean-slate implementation.

---

## 1. Why Migrate?

| Aspect | Legacy (`cuda/`) | New (`gpu/`) |
|--------|------------------|--------------|
| Memory usage | ~872 × N bytes | ~89 × N bytes (**9.8× reduction**) |
| Memory layout | Vectorized (8× duplication) | Native 3D (single copy) |
| Kernel launches | 8 × N_levels per sweep (high overhead) | CUDA Graphs (single replay) |
| Maintainability | 16 hardcoded kernel-arg branches | Template kernels, RAII |
| Target hardware | Older NVIDIA (sm_61) | Blackwell GB200 (sm_100) |
| CUDA features | Legacy `cudaMalloc`, manual streams | Stream-ordered alloc, CUDA Graphs |

---

## 2. Build System Changes

### 2.1 Enable the new GPU backend

In `CMakeLists.txt`, add:

```cmake
# Clean-slate GPU backend (gpu/)
if(USE_CUDA_V2)
    add_subdirectory(gpu)
endif()
```

Build with:
```bash
cmake -DUSE_CUDA=ON -DUSE_CUDA_V2=ON -DCMAKE_CUDA_ARCHITECTURES=100 ..
make
```

### 2.2 Architecture flags

The new backend targets Blackwell (sm_100) by default, with fallbacks for development:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 70 80 90 100)
```

| Architecture | GPU | Use case |
|-------------|-----|----------|
| sm_70 | Volta V100 | CI, legacy dev |
| sm_80 | A100 | Dev, CI |
| sm_90 | Hopper H100 | Dev |
| sm_100 | Blackwell B200 | **Production target** |

---

## 3. Code Changes

### 3.1 Replace `Grid_on_device` with `GpuGrid`

**Legacy** (`cuda/grid_wrapper.cuh`):
```cpp
Grid_on_device* gpu_grid = new Grid_on_device();
cuda_initialize_grid_1st(ijk, gpu_grid, loc_I, loc_J, loc_K, ...);
```

**New** (`gpu/include/gpu_grid.h`):
```cpp
#include "gpu_grid.h"
tomogpu::GpuGrid gpu_grid(dims, dr, dt, dp, order, stream);
```

### 3.2 Replace sweep orchestration

**Legacy** (`cuda/iterator_wrapper.cu`):
```cpp
cuda_copy_tau_to_device(gpu_grid, grid.tau_loc);
cuda_run_iteration_forward(gpu_grid, iswp);
cuda_copy_tau_to_host(gpu_grid, grid.tau_loc);
```

**New** (`gpu/include/gpu_backend.h`):
```cpp
#include "gpu_backend.h"
tomogpu::GpuSweepBackend backend(dims, dr, dt, dp, order, stream);
backend.initialize_from_host(...);
backend.run_sweep();  // All 8 directions, via CUDA Graph
backend.copy_tau_to_host(host_tau);
```

### 3.3 Replace stencil computation

**Legacy** (16 hardcoded branches in `run_kernel`):
```cpp
if (iswp == 0) {
    void* kernelArgs[] = {
        &(grid_dv->vv_i__j__k___0), &(grid_dv->vv_ip1j__k___0), ...
    };
    cudaLaunchKernel(...);
} else if (iswp == 1) { ... }
// ... 8 branches × 2 orders = 16 branches
```

**New** (single template kernel):
```cpp
sweep_level_kernel<1><<<grid, block, 0, stream>>>(
    tau, T0v, T0r, T0t, T0p, fac_a, fac_b, fac_c, fac_f, fun,
    is_changed, level_offsets, level_counts, level_indices,
    level, sign_r, sign_t, sign_p, nx, ny, nz, dr, dt, dp);
```

### 3.4 Replace memory allocation

**Legacy** (`cuda/cuda_utils.cuh`):
```cpp
cudaError_t allocate_memory_on_device_cv(void** d_ptr, size_t size);
// Manual alloc/free, no RAII
```

**New** (`gpu/include/gpu_memory.h`):
```cpp
tomogpu::RealDeviceBuffer buf(N, stream);  // RAII, stream-ordered
buf.copy_from_host(host_data, stream);
// Auto-freed on scope exit
```

---

## 4. Integration Points in TomoATT

### 4.1 `src/iterator_level.cpp`

The `do_sweep()` method in `Iterator_level_1st_order` and `Iterator_level_3rd_order` currently calls the legacy GPU path:

```cpp
// Legacy path in do_sweep():
if (use_gpu) {
    cuda_copy_tau_to_device(gpu_grid, grid.tau_loc);
    cuda_run_iteration_forward(gpu_grid, iswp);
    cuda_copy_tau_to_host(gpu_grid, grid.tau_loc);
}
```

**Migration**: Replace with the new backend:

```cpp
// New path:
if (use_gpu) {
    backend.run_sweep();  // Handles all 8 directions internally
}
```

### 4.2 `include/iterator.h`

The `Iterator` class currently holds a `Grid_on_device* gpu_grid`:

```cpp
#ifdef USE_CUDA
    Grid_on_device *gpu_grid;
#endif
```

**Migration**: Replace with `GpuSweepBackend`:

```cpp
#ifdef USE_CUDA
    std::unique_ptr<tomogpu::GpuSweepBackend> gpu_backend;
#endif
```

### 4.3 `src/iterator.cpp`

The `initialize_arrays()` method currently calls `cuda_initialize_grid_*`:

```cpp
if (use_gpu) {
    gpu_grid = new Grid_on_device();
    cuda_initialize_grid_1st(...);
}
```

**Migration**: Replace with `GpuSweepBackend` construction:

```cpp
if (use_gpu) {
    tomogpu::GridDims dims{loc_I, loc_J, loc_K};
    gpu_backend = std::make_unique<tomogpu::GpuSweepBackend>(
        dims, grid.dr, grid.dt, grid.dp,
        tomogpu::StencilOrder::First, stream);
    gpu_backend->initialize_from_host(...);
}
```

---

## 5. Testing the Migration

### 5.1 Unit tests

Run the standalone unit tests to verify the new backend:

```bash
cd build && make test_gpu_correctness test_gpu_memory
./test/test_gpu_correctness
./test/test_gpu_memory
```

### 5.2 Numerical validation

Compare CPU and GPU results on small, medium, and large grids:

```bash
# Run CPU solver
./TOMOATT input_cpu.yaml

# Run GPU solver
./TOMOATT input_gpu.yaml

# Compare outputs
python3 tests/compare_outputs.py cpu_output.h5 gpu_output.h5 --tol 1e-6
```

### 5.3 Performance benchmarks

Run the benchmark suite to measure performance gains:

```bash
make bench_gpu_sweep
./bench/bench_gpu_sweep --sizes 32,64,128 --iters 20
```

---

## 6. Rollback Plan

If the new backend causes issues, you can roll back to the legacy code:

1. **Build**: Remove `-DUSE_CUDA_V2=ON` from the cmake command.
2. **Code**: The legacy `cuda/` directory is preserved and unaffected.
3. **Runtime**: The `use_gpu` flag controls whether GPU acceleration is used.

The new backend is designed to **coexist** with the legacy code during the migration period. Both can be compiled simultaneously, and the active backend is selected at runtime or compile time.

---

## 7. Frequently Asked Questions

### Q: Why not just optimize the existing GPU code?

The existing code has fundamental architectural issues:
- 8× memory duplication is baked into the `Grid_on_device` structure
- 16 hardcoded kernel-arg branches make maintenance impossible
- The vectorized layout is incompatible with native 3D memory access
- The code targets sm_61 (Pascal), missing 4 generations of CUDA features

Incremental optimization would require touching every file while fighting the existing architecture. A clean-slate reimplementation is cleaner and faster to develop.

### Q: Does the new backend support all stencil orders?

Yes. The `sweep_level_kernel<kOrder>` template supports both 1st and 3rd order stencils. The 3rd order WENO stencil is implemented with the same memory layout as the 1st order, just with additional neighbor accesses (±2 in each direction).

### Q: How does the new backend handle MPI communication?

The new backend focuses on the per-GPU sweep computation. MPI communication (ghost cell synchronization) is handled by the existing `Grid::send_recev_boundary_data()` infrastructure, which is unchanged. The new backend copies `tau` to/from the device around MPI communication points.

### Q: What about the adjoint (backpropagation) sweep?

The adjoint sweep is not yet implemented in the new backend (it's a Phase 4+ deliverable). The legacy CPU path remains available for adjoint computation. The adjoint sweep has the same structure as the forward sweep, so extending the new backend to support it is straightforward.
