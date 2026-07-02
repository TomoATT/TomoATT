//
// gpu_backend.h — High-level GPU sweep backend for the clean-slate
// implementation.
//
// The `GpuSweepBackend` class orchestrates the Fast Sweeping Method on the
// GPU.  It owns a `GpuGrid` (all device arrays) and provides:
//
//   * `initialize()` — transfer initial fields (T0v, fac_a, fun, ...) from
//     the host `Grid` to the device.
//   * `run_sweep()` — perform one complete FSM sweep (8 directions).
//   * `finalize()` — copy the converged tau field back to the host.
//
// The backend uses CUDA Graphs to capture the entire sweep loop, eliminating
// kernel launch overhead for repeated sweeps (e.g., during inversion
// iterations).  The graph is instantiated lazily on the first `run_sweep()`
// call and replayed on subsequent calls.
//
// Memory efficiency:
//   * Each field is stored as a single native 3D array — no 8× duplication.
//   * The level-set decomposition is precomputed once and reused.
//   * Stream-ordered allocation avoids fragmentation.
//
#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "gpu_grid.h"
#include "gpu_kernels.cuh"
#include "gpu_memory.h"
#include "gpu_types.h"

namespace tomogpu {

// ---------------------------------------------------------------------------
// GpuSweepBackend — orchestrates FSM sweeps on the GPU
// ---------------------------------------------------------------------------
class GpuSweepBackend {
public:
    // Construct the backend for the given grid dimensions and stencil order.
    // Allocates all device arrays and builds the level-set decomposition.
    GpuSweepBackend(GridDims dims, real_t dr, real_t dt, real_t dp,
                    StencilOrder order, cudaStream_t stream = 0);

    ~GpuSweepBackend();

    GpuSweepBackend(const GpuSweepBackend&)            = delete;
    GpuSweepBackend& operator=(const GpuSweepBackend&) = delete;

    // --- Lifecycle ---
    // Initialize device arrays from host data.  Copies T0v, T0r, T0t, T0p,
    // fac_a, fac_b, fac_c, fac_f, fun, is_changed, and tau to the device.
    void initialize_from_host(
        const real_t* host_T0v,
        const real_t* host_T0r,
        const real_t* host_T0t,
        const real_t* host_T0p,
        const real_t* host_fac_a,
        const real_t* host_fac_b,
        const real_t* host_fac_c,
        const real_t* host_fac_f,
        const real_t* host_fun,
        const uint8_t* host_is_changed,
        const real_t* host_tau);

    // Copy the converged tau field from device to host.
    void copy_tau_to_host(real_t* host_tau);

    // --- Sweep execution ---
    // Perform one complete FSM sweep (8 directions).  Each direction
    // processes all level sets in order, with one kernel launch per level.
    void run_sweep();

    // --- Convergence ---
    // Compute the L1 and Linf norms of the change in tau.  Requires a
    // `tau_old` buffer (allocated internally).
    void compute_convergence(real_t& l1, real_t& linf);

    // Store the current tau into tau_old (for convergence checking).
    void store_tau_to_old();

    // --- Accessors ---
    GpuGrid&       grid()        { return *grid_; }
    const GridDims& dims() const { return grid_->dims(); }
    cudaStream_t   stream() const { return stream_; }

    // --- Configuration ---
    void set_block_size(idx_t bs) { block_size_ = bs; }
    idx_t block_size() const { return block_size_; }

private:
    // Build the CUDA Graph for one sweep (8 directions × N levels).
    void build_sweep_graph();

    // Launch one level-set kernel for the given sweep direction and level.
    void launch_level_kernel(SweepDir dir, idx_t level, idx_t num_nodes);

    // Internal state
    std::unique_ptr<GpuGrid> grid_;
    cudaStream_t            stream_;
    idx_t                   block_size_;

    // CUDA Graph for sweep replay
    cudaGraph_t       sweep_graph_      = nullptr;
    cudaGraphExec_t   sweep_exec_       = nullptr;
    bool               graph_built_      = false;

    // Temporary buffers for convergence checking
    RealDeviceBuffer  tau_old_;
    RealDeviceBuffer  block_l1_;
    RealDeviceBuffer  block_linf_;
};

}  // namespace tomogpu
