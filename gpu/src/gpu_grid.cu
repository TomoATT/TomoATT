//
// gpu_grid.cu — Implementation of GpuGrid and LevelSetDecomposition.
//
// This file handles:
//   1. Building the level-set decomposition on the host and copying it to
//      the device.
//   2. Constructing the GpuGrid, which allocates all device arrays (native
//      3D layout, one copy per field — no 8× duplication).
//   3. Host ↔ Device transfers for the tau field.
//
// ---------------------------------------------------------------------------
// Level-set decomposition
// ---------------------------------------------------------------------------
// For the Fast Sweeping Method, nodes are grouped by "level" = i+j+k.
// Nodes at the same level are independent (they don't depend on each
// other's updated values) and can be computed in parallel.
//
// The decomposition maps each level L to a list of flat grid indices:
//   level_indices[offset[L] .. offset[L]+count[L]-1]
//
// For a given level L:
//   k ranges from max(0, L-(nx-1)-(ny-1)) to min(nz-1, L)
//   for each k:
//     j ranges from max(0, L-k-(nx-1)) to min(ny-1, L-k)
//     i = L - j - k
//
// Only INTERIOR nodes are included (1 <= i <= nx-2, etc.).  Boundary
// nodes are handled by a separate kernel.
//
#include "gpu_grid.h"

#include <algorithm>
#include <cstring>

namespace tomogpu {

// ---------------------------------------------------------------------------
// LevelSetDecomposition::build
// ---------------------------------------------------------------------------
void LevelSetDecomposition::build(const GridDims& dims, cudaStream_t stream) {
    const idx_t nx = dims.nx, ny = dims.ny, nz = dims.nz;

    // Total number of levels: 0 to (nx-2)+(ny-2)+(nz-2) = nx+ny+nz-6
    // But we include levels from 0 to nx+ny+nz-3 (matching CPU st_level/ed_level).
    // Interior nodes span levels 3 to nx+ny+nz-6 (since i,j,k >= 1 for interior).
    //
    // Actually, the CPU code uses st_level=0, ed_level=nx+ny+nz-3.
    // The level decomposition includes ALL nodes (interior + boundary).
    // The kernel skips boundary nodes.
    //
    // Let's match the CPU: levels from 0 to max_level (nx+ny+nz-3).
    num_levels_ = dims.max_level() + 1;  // levels 0..max_level
    if (num_levels_ <= 0) return;

    // Allocate host storage for offsets and counts
    offsets_host_.resize(num_levels_ + 1);  // +1 for exclusive scan
    counts_host_.resize(num_levels_);

    // First pass: count nodes per level (interior only: 1 <= i <= nx-2, etc.)
    // For the FSM, the level decomposition should include all nodes that
    // need to be updated.  We include interior nodes (skip the 1-node ghost
    // boundary).  The kernel will skip nodes that are on the boundary.
    //
    // Actually, to match the CPU code exactly, we include ALL nodes at each
    // level (both interior and boundary).  The kernel checks if the node is
    // interior before computing the stencil, and boundary nodes are handled
    // separately.
    //
    // Wait — re-reading the CPU `assign_processes_for_levels`: it includes
    // all nodes from i=0..nx-1, j=0..ny-1, k=0..nz-1 where i+j+k=level.
    // And the level ranges from 0 to nx+ny+nz-3.
    //
    // But the CPU `do_sweep` then processes these nodes, and for boundary
    // nodes it calls `calculate_boundary_nodes` separately.  So the level
    // decomposition includes ALL nodes, and the kernel handles interior vs
    // boundary.
    //
    // For the GPU, we precompute the level decomposition for ALL nodes
    // (interior + boundary).  The kernel skips boundary nodes (they're
    // handled by the boundary kernel).

    // Count nodes per level
    std::fill(counts_host_.begin(), counts_host_.end(), 0);
    for (idx_t k = 0; k < nz; ++k) {
        for (idx_t j = 0; j < ny; ++j) {
            for (idx_t i = 0; i < nx; ++i) {
                idx_t level = i + j + k;
                if (level >= 0 && level < num_levels_) {
                    counts_host_[level]++;
                }
            }
        }
    }

    // Compute offsets via exclusive scan
    offsets_host_[0] = 0;
    for (idx_t L = 0; L < num_levels_; ++L) {
        offsets_host_[L + 1] = offsets_host_[L] + counts_host_[L];
    }
    total_nodes_ = offsets_host_[num_levels_];

    // Build the index list
    std::vector<idx_t> indices_host(total_nodes_);
    std::vector<idx_t> level_cursor(num_levels_, 0);
    for (idx_t k = 0; k < nz; ++k) {
        for (idx_t j = 0; j < ny; ++j) {
            for (idx_t i = 0; i < nx; ++i) {
                idx_t level = i + j + k;
                if (level >= 0 && level < num_levels_) {
                    idx_t pos = offsets_host_[level] + level_cursor[level]++;
                    indices_host[pos] = flat_index(i, j, k, dims);
                }
            }
        }
    }

    // Find max level count (for kernel launch configuration)
    max_level_count_ = 0;
    for (idx_t L = 0; L < num_levels_; ++L) {
        max_level_count_ = std::max(max_level_count_, counts_host_[L]);
    }

    // Copy to device
    offsets_dev_.resize(num_levels_ + 1, stream);
    counts_dev_.resize(num_levels_, stream);
    indices_dev_.resize(total_nodes_, stream);

    offsets_dev_.copy_from_host(offsets_host_.data(), stream);
    counts_dev_.copy_from_host(counts_host_.data(), stream);
    indices_dev_.copy_from_host(indices_host.data(), stream);
}

// ---------------------------------------------------------------------------
// GpuGrid constructor
// ---------------------------------------------------------------------------
GpuGrid::GpuGrid(GridDims dims, real_t dr, real_t dt, real_t dp,
                  StencilOrder order, cudaStream_t stream)
    : dims_(dims), dr_(dr), dt_(dt), dp_(dp), order_(order), stream_(stream) {

    const idx_t N = dims.total();
    if (N == 0) return;

    // Allocate all device arrays (native 3D layout, one copy per field)
    tau_.        resize(N, stream);
    T0v_.        resize(N, stream);
    T0r_.        resize(N, stream);
    T0t_.        resize(N, stream);
    T0p_.        resize(N, stream);
    fac_a_.      resize(N, stream);
    fac_b_.      resize(N, stream);
    fac_c_.      resize(N, stream);
    fac_f_.      resize(N, stream);
    fun_.        resize(N, stream);
    is_changed_. resize(N, stream);

    // Build the level-set decomposition
    levels_.build(dims, stream);
}

// ---------------------------------------------------------------------------
// Host ↔ Device transfers
// ---------------------------------------------------------------------------
void GpuGrid::copy_tau_to_device(const real_t* host_tau) {
    if (host_tau == nullptr) return;
    tau_.copy_from_host(host_tau, stream_);
}

void GpuGrid::copy_tau_to_host(real_t* host_tau) {
    if (host_tau == nullptr) return;
    tau_.copy_to_host(host_tau, stream_);
}

void GpuGrid::synchronize() {
    TOMOGPU_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace tomogpu
