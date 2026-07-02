//
// gpu_backend.cu — Implementation of the GpuSweepBackend class.
//
// This file implements the sweep orchestration for the FSM on the GPU.
// The backend uses CUDA Graphs to capture the entire sweep loop,
// eliminating kernel launch overhead for repeated sweeps.
//
#include "gpu_backend.h"

#include <cmath>
#include <cstring>

namespace tomogpu {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
GpuSweepBackend::GpuSweepBackend(
    GridDims dims, real_t dr, real_t dt, real_t dp,
    StencilOrder order, cudaStream_t stream)
    : stream_(stream), block_size_(kDefaultBlockSize) {

    grid_ = std::make_unique<GpuGrid>(dims, dr, dt, dp, order, stream);

    // Allocate temporary buffers for convergence checking
    const idx_t N = dims.total();
    tau_old_.resize(N, stream);
    block_l1_.resize(1024, stream);    // enough for grids up to ~256M nodes
    block_linf_.resize(1024, stream);
}

GpuSweepBackend::~GpuSweepBackend() {
    if (sweep_exec_) cudaGraphExecDestroy(sweep_exec_);
    if (sweep_graph_) cudaGraphDestroy(sweep_graph_);
}

// ---------------------------------------------------------------------------
// initialize_from_host
// ---------------------------------------------------------------------------
void GpuSweepBackend::initialize_from_host(
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
    const real_t* host_tau)
{
    GpuGrid& g = *grid_;

    // Copy all fields from host to device (async on the grid's stream)
    g.T0v_buf().copy_from_host(host_T0v, stream_);
    g.T0r_buf().copy_from_host(host_T0r, stream_);
    g.T0t_buf().copy_from_host(host_T0t, stream_);
    g.T0p_buf().copy_from_host(host_T0p, stream_);
    g.fac_a_buf().copy_from_host(host_fac_a, stream_);
    g.fac_b_buf().copy_from_host(host_fac_b, stream_);
    g.fac_c_buf().copy_from_host(host_fac_c, stream_);
    g.fac_f_buf().copy_from_host(host_fac_f, stream_);
    g.fun_buf().copy_from_host(host_fun, stream_);
    g.is_changed_buf().copy_from_host(host_is_changed, stream_);
    g.tau_buf().copy_from_host(host_tau, stream_);
}

// ---------------------------------------------------------------------------
// copy_tau_to_host
// ---------------------------------------------------------------------------
void GpuSweepBackend::copy_tau_to_host(real_t* host_tau) {
    grid_->copy_tau_to_host(host_tau);
}

// ---------------------------------------------------------------------------
// launch_level_kernel
// ---------------------------------------------------------------------------
// Launches one level-set kernel for the given sweep direction and level.
// The kernel processes `num_nodes` nodes in parallel.
//
void GpuSweepBackend::launch_level_kernel(
    SweepDir dir, idx_t level, idx_t num_nodes) {

    GpuGrid& g = *grid_;
    const GridDims& dims = g.dims();

    // Compute grid dimensions for the kernel launch
    idx_t blocks = (num_nodes + block_size_ - 1) / block_size_;
    dim3 grid_dim(blocks);
    dim3 block_dim(block_size_);

    // Get sweep direction signs
    int sr = sign_r(dir);
    int st = sign_t(dir);
    int sp = sign_p(dir);

    // Launch the appropriate kernel (1st or 3rd order)
    const int order = static_cast<int>(g.order());
    if (order == 1) {
        sweep_level_kernel<1><<<grid_dim, block_dim, 0, stream_>>>(
            g.tau(), g.T0v(), g.T0r(), g.T0t(), g.T0p(),
            g.fac_a(), g.fac_b(), g.fac_c(), g.fac_f(), g.fun(),
            g.is_changed(),
            g.levels().offsets_device(),
            g.levels().counts_device(),
            g.levels().indices_device(),
            level, sr, st, sp,
            dims.nx, dims.ny, dims.nz,
            g.dr(), g.dt(), g.dp());
    } else {
        sweep_level_kernel<3><<<grid_dim, block_dim, 0, stream_>>>(
            g.tau(), g.T0v(), g.T0r(), g.T0t(), g.T0p(),
            g.fac_a(), g.fac_b(), g.fac_c(), g.fac_f(), g.fun(),
            g.is_changed(),
            g.levels().offsets_device(),
            g.levels().counts_device(),
            g.levels().indices_device(),
            level, sr, st, sp,
            dims.nx, dims.ny, dims.nz,
            g.dr(), g.dt(), g.dp());
    }
}

// ---------------------------------------------------------------------------
// build_sweep_graph
// ---------------------------------------------------------------------------
// Captures the entire sweep loop (8 directions × N levels) as a CUDA Graph.
// The graph can then be replayed on subsequent sweeps, eliminating kernel
// launch overhead.
//
void GpuSweepBackend::build_sweep_graph() {
    if (graph_built_) return;

    GpuGrid& g = *grid_;
    const LevelSetDecomposition& levels = g.levels();
    const idx_t num_levels = levels.num_levels();

    // Begin graph capture
    TOMOGPU_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed));

    // For each of the 8 sweep directions, process all levels in order
    for (idx_t dir_id = 0; dir_id < kNumSweepDirs; ++dir_id) {
        SweepDir dir = static_cast<SweepDir>(dir_id);

        // Process levels in increasing order (level 0, 1, ..., max_level)
        for (idx_t level = 0; level < num_levels; ++level) {
            idx_t num_nodes = levels.level_count(level);
            if (num_nodes == 0) continue;

            launch_level_kernel(dir, level, num_nodes);
        }
    }

    // End graph capture
    TOMOGPU_CUDA_CHECK(cudaStreamEndCapture(stream_, &sweep_graph_));
    TOMOGPU_CUDA_CHECK(cudaGraphInstantiate(&sweep_exec_, sweep_graph_, nullptr, nullptr, 0));

    graph_built_ = true;
}

// ---------------------------------------------------------------------------
// run_sweep
// ---------------------------------------------------------------------------
void GpuSweepBackend::run_sweep() {
    GpuGrid& g = *grid_;
    const LevelSetDecomposition& levels = g.levels();
    const idx_t num_levels = levels.num_levels();

    // For each of the 8 sweep directions, process all levels in order
    for (idx_t dir_id = 0; dir_id < kNumSweepDirs; ++dir_id) {
        SweepDir dir = static_cast<SweepDir>(dir_id);

        // Process levels in increasing order (level 0, 1, ..., max_level)
        for (idx_t level = 0; level < num_levels; ++level) {
            idx_t num_nodes = levels.level_count(level);
            if (num_nodes == 0) continue;

            launch_level_kernel(dir, level, num_nodes);
        }
    }
}

// ---------------------------------------------------------------------------
// store_tau_to_old
// ---------------------------------------------------------------------------
void GpuSweepBackend::store_tau_to_old() {
    const idx_t N = grid_->dims().total();
    copy_real_kernel<<<(N + block_size_ - 1) / block_size_, block_size_, 0, stream_>>>(
        tau_old_.get(), grid_->tau(), N);
}

// ---------------------------------------------------------------------------
// compute_convergence
// ---------------------------------------------------------------------------
void GpuSweepBackend::compute_convergence(real_t& l1, real_t& linf) {
    const idx_t N = grid_->dims().total();

    // Launch reduction kernel
    idx_t blocks = std::min<idx_t>(1024, (N + block_size_ - 1) / block_size_);
    dim3 grid_dim(blocks);
    dim3 block_dim(block_size_);
    idx_t shared_mem = block_size_ * sizeof(real_t);

    convergence_reduce_kernel<<<grid_dim, block_dim, shared_mem, stream_>>>(
        grid_->tau(), tau_old_.get(), N,
        block_l1_.get(), block_linf_.get());

    // Copy results to host and reduce
    std::vector<real_t> host_l1(blocks), host_linf(blocks);
    TOMOGPU_CUDA_CHECK(cudaMemcpyAsync(host_l1.data(), block_l1_.get(),
                                        blocks * sizeof(real_t),
                                        cudaMemcpyDeviceToHost, stream_));
    TOMOGPU_CUDA_CHECK(cudaMemcpyAsync(host_linf.data(), block_linf_.get(),
                                        blocks * sizeof(real_t),
                                        cudaMemcpyDeviceToHost, stream_));
    TOMOGPU_CUDA_CHECK(cudaStreamSynchronize(stream_));

    l1 = 0;
    linf = 0;
    for (idx_t i = 0; i < blocks; ++i) {
        l1 += host_l1[i];
        linf = std::max(linf, host_linf[i]);
    }
}

}  // namespace tomogpu
