//
// gpu_grid.h — GPU grid data structure: native 3D layout, level-set
// decomposition, and host↔device transfers.
//
// The `GpuGrid` class owns all device arrays required for one MPI rank's
// local subdomain.  It replaces the legacy `Grid_on_device` structure that
// duplicated the entire vectorized 3D-array representation 8× (one per
// sweep direction).
//
// Key memory-efficiency decisions:
//   1. Native 3D layout: each field (tau, T0v, fac_a, ...) is stored as a
//      single flat array in (i,j,k) row-major order, matching the CPU I2V
//      macro.  No flattening/vectorizing into an 8×-larger representation.
//   2. Single copy per field: the 8 sweep directions share the SAME grid
//      data.  The sweep direction only affects the traversal order and the
//      index flip in the kernel.
//   3. Level-set decomposition is precomputed ONCE per grid and reused for
//      all 8 sweep directions.  The precomputed data is O(nx+ny+nz) +
//      O(total_nodes) integers — a single compact copy.
//
#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "gpu_memory.h"
#include "gpu_types.h"

namespace tomogpu {

// ---------------------------------------------------------------------------
// LevelSetDecomposition — precomputed level-set structure
// ---------------------------------------------------------------------------
// For the Fast Sweeping Method, nodes are grouped by "level" = i+j+k.
// Nodes at the same level are independent (can be computed in parallel).
//
// This structure stores, for each level L (0 to max_level):
//   * level_offsets_[L]: starting index into level_indices_ for level L
//   * level_counts_[L]:  number of nodes at level L
//   * level_indices_:    flat indices (k*nx*ny + j*nx + i) of all nodes,
//                        grouped by level
//
// The decomposition is computed ONCE for the natural grid (level = i+j+k)
// and reused for all 8 sweep directions.  For a given sweep direction,
// the kernel applies the index flip in O(1) per node.
//
// Memory cost: ~total_nodes * sizeof(idx_t) + max_level * 2 * sizeof(idx_t)
// For a 120×120×120 grid: ~17 MB of index data (one copy).
//
class LevelSetDecomposition {
public:
    LevelSetDecomposition() = default;

    // Build the decomposition for the given grid dimensions.
    // This runs on the host and populates the device arrays.
    void build(const GridDims& dims, cudaStream_t stream = 0);

    // Device accessors
    const idx_t* offsets_device() const { return offsets_dev_.get(); }
    const idx_t* counts_device()  const { return counts_dev_.get(); }
    const idx_t* indices_device() const { return indices_dev_.get(); }

    // Host-side accessors (for kernel launch configuration)
    idx_t num_levels()          const { return num_levels_; }
    idx_t level_count(idx_t L)   const { return counts_host_[L]; }
    idx_t level_offset(idx_t L)  const { return offsets_host_[L]; }
    idx_t total_nodes()          const { return total_nodes_; }
    idx_t max_level_count()      const { return max_level_count_; }

private:
    // Host copies (for launch configuration)
    std::vector<idx_t> offsets_host_;
    std::vector<idx_t> counts_host_;
    idx_t num_levels_      = 0;
    idx_t total_nodes_     = 0;
    idx_t max_level_count_ = 0;

    // Device copies
    IndexDeviceBuffer offsets_dev_;
    IndexDeviceBuffer counts_dev_;
    IndexDeviceBuffer indices_dev_;
};

// ---------------------------------------------------------------------------
// GpuGrid — owns all device arrays for one local subdomain
// ---------------------------------------------------------------------------
// Memory layout: each field is a flat array of size dims.total() stored in
// (i,j,k) row-major order (i varies fastest), matching the CPU I2V macro.
//
// The grid holds:
//   * Working array:      tau (normalized traveltime, updated in-place)
//   * Initial field:       T0v, T0r, T0t, T0p (analytical seed + derivatives)
//   * Anisotropy factors:  fac_a, fac_b, fac_c, fac_f (Thomsen parameters)
//   * Slowness field:      fun (1/velocity)
//   * Change flags:        is_changed (uint8 array, one byte per node)
//
// All arrays are allocated via stream-ordered `cudaMallocAsync` and freed
// automatically when the GpuGrid is destroyed (RAII).
//
class GpuGrid {
public:
    // Construct and allocate all device arrays for the given dimensions.
    GpuGrid(GridDims dims, real_t dr, real_t dt, real_t dp,
            StencilOrder order, cudaStream_t stream = 0);

    ~GpuGrid() = default;

    GpuGrid(const GpuGrid&)            = delete;
    GpuGrid& operator=(const GpuGrid&) = delete;
    GpuGrid(GpuGrid&&)                 = delete;
    GpuGrid& operator=(GpuGrid&&)     = delete;

    // --- Grid metadata ---
    const GridDims&   dims()     const { return dims_; }
    idx_t             total()    const { return dims_.total(); }
    real_t            dr()       const { return dr_; }
    real_t            dt()       const { return dt_; }
    real_t            dp()       const { return dp_; }
    StencilOrder      order()    const { return order_; }
    cudaStream_t      stream()   const { return stream_; }

    // --- DeviceBuffer accessors (for copy/transfer operations) ---
    RealDeviceBuffer&  tau_buf()       noexcept { return tau_; }
    RealDeviceBuffer&  T0v_buf()       noexcept { return T0v_; }
    RealDeviceBuffer&  T0r_buf()       noexcept { return T0r_; }
    RealDeviceBuffer&  T0t_buf()       noexcept { return T0t_; }
    RealDeviceBuffer&  T0p_buf()       noexcept { return T0p_; }
    RealDeviceBuffer&  fac_a_buf()     noexcept { return fac_a_; }
    RealDeviceBuffer&  fac_b_buf()     noexcept { return fac_b_; }
    RealDeviceBuffer&  fac_c_buf()     noexcept { return fac_c_; }
    RealDeviceBuffer&  fac_f_buf()     noexcept { return fac_f_; }
    RealDeviceBuffer&  fun_buf()       noexcept { return fun_; }
    BoolDeviceBuffer&   is_changed_buf() noexcept { return is_changed_; }

    // --- Raw pointer accessors (for kernel launches) ---
    real_t*       tau()        noexcept { return tau_.get(); }
    real_t*       T0v()        noexcept { return T0v_.get(); }
    real_t*       T0r()        noexcept { return T0r_.get(); }
    real_t*       T0t()        noexcept { return T0t_.get(); }
    real_t*       T0p()        noexcept { return T0p_.get(); }
    real_t*       fac_a()      noexcept { return fac_a_.get(); }
    real_t*       fac_b()      noexcept { return fac_b_.get(); }
    real_t*       fac_c()      noexcept { return fac_c_.get(); }
    real_t*       fac_f()      noexcept { return fac_f_.get(); }
    real_t*       fun()        noexcept { return fun_.get(); }
    uint8_t*      is_changed() noexcept { return is_changed_.get(); }

    // --- Level-set decomposition ---
    const LevelSetDecomposition& levels()      const { return levels_; }
    LevelSetDecomposition&       levels_mut()         { return levels_; }

    // --- Host ↔ Device transfers ---
    // Copy the tau field from host to device (async on the grid's stream).
    void copy_tau_to_device(const real_t* host_tau);
    // Copy the tau field from device to host.
    void copy_tau_to_host(real_t* host_tau);

    // Synchronize the grid's stream (wait for all async operations).
    void synchronize();

private:
    GridDims       dims_;
    real_t         dr_, dt_, dp_;
    StencilOrder   order_;
    cudaStream_t   stream_;

    // Device buffers for each field (native 3D layout, flat indexing)
    RealDeviceBuffer  tau_;          // working array
    RealDeviceBuffer  T0v_;          // initial analytical traveltime
    RealDeviceBuffer  T0r_;          // dT0/dr
    RealDeviceBuffer  T0t_;          // dT0/dt
    RealDeviceBuffer  T0p_;          // dT0/dp
    RealDeviceBuffer  fac_a_;        // anisotropy factor a
    RealDeviceBuffer  fac_b_;        // anisotropy factor b
    RealDeviceBuffer  fac_c_;        // anisotropy factor c
    RealDeviceBuffer  fac_f_;        // anisotropy factor f
    RealDeviceBuffer  fun_;         // slowness (1/velocity)
    BoolDeviceBuffer  is_changed_;   // change flags (uint8)

    // Level-set decomposition (precomputed once, reused for all sweeps)
    LevelSetDecomposition levels_;
};

}  // namespace tomogpu
