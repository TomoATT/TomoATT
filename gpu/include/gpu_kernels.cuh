//
// gpu_kernels.cuh — CUDA kernel declarations for the clean-slate GPU backend.
//
// Kernel responsibilities are kept minimal and focused:
//   * `sweep_level_kernel` — processes one level-set of the FSM sweep.
//     Each thread updates one interior node using the Lax-Friedrichs
//     Hamiltonian stencil.  The kernel is parameterized by stencil order
//     (1st or 3rd) and handles the 8 sweep directions via an index flip.
//
//   * `boundary_kernel` — updates the 6 boundary faces of the subdomain
//     using the CPU-matched extrapolation formula:
//       tau[boundary] = max(2*tau[adjacent] - tau[adjacent2], tau[adjacent2])
//
//   * `init_kernel` — initializes device arrays from host data.
//
// Design notes:
//   * Each field is accessed via direct (i,j,k) indexing into the native
//     3D layout — no gather/scatter through precomputed index arrays.
//   * The level-set traversal order is handled by the host-side sweep
//     orchestrator, which launches one kernel per level in the correct
//     order.  Within a level, all nodes are independent.
//   * `__restrict__` and `__launch_bounds__` are used to guide the
//     compiler toward efficient register allocation and occupancy.
//
#pragma once

#include <cuda_runtime.h>

#include "gpu_grid.h"
#include "gpu_types.h"

namespace tomogpu {

// ---------------------------------------------------------------------------
// Sweep level kernel
// ---------------------------------------------------------------------------
// Processes all interior nodes at FSM level `level` for the given sweep
// direction.  Each thread handles one node.
//
// The level-set decomposition maps thread IDs to natural (i,j,k)
// coordinates via the precomputed `level_indices` array.  For the given
// sweep direction, the kernel then applies the index flip to obtain the
// "sweep-coordinate" (i',j',k') at which the stencil is evaluated.
//
// However, since the LF Hamiltonian stencil uses centered differences
// (symmetric in neighbors), the stencil computation at a node (i,j,k) is
// identical regardless of sweep direction.  The sweep direction only
// affects the TRAVERSAL ORDER (which levels come first), which is handled
// by the host-side orchestrator.  Therefore, the kernel does NOT need to
// apply any index flip — it simply processes the nodes at the given level
// in their natural (i,j,k) order.
//
// Wait — this is only correct if the level-set decomposition is the same
// for all sweep directions.  In the CPU code, the level structure
// (`ijk_for_this_subproc`) IS the same for all sweep directions (based on
// level = i+j+k), and the sweep direction is applied as an index flip
// AFTER the level decomposition.  So the kernel can process nodes in
// natural order, and the host orchestrator handles the sweep direction
// by choosing the level traversal order.
//
// Actually, re-examining the CPU code: the sweep direction flips the
// indices to determine WHICH physical node to update.  For sweep
// direction (i+,j+,k+), the physical node is (i-1, j-1, k-1).  For
// (i-,j+,k+), the physical node is (nx-i, j-1, k-1).  This means the
// level structure is effectively different for each sweep direction.
//
// To handle this correctly and efficiently on the GPU, we precompute
// the level-set decomposition for the NATURAL grid (level = i+j+k) and
// reuse it for all 8 sweep directions by applying the index flip inside
// the kernel.  The flip is an O(1) operation per node.
//
// Template parameter `kOrder` selects 1st or 3rd order stencil.
//
template <int kOrder>
__global__ void sweep_level_kernel(
    real_t* __restrict__       tau,        // normalized traveltime (in-place)
    const real_t* __restrict__ T0v,        // initial analytical traveltime
    const real_t* __restrict__ T0r,        // dT0/dr
    const real_t* __restrict__ T0t,        // dT0/dt
    const real_t* __restrict__ T0p,        // dT0/dp
    const real_t* __restrict__ fac_a,      // anisotropy factor a
    const real_t* __restrict__ fac_b,      // anisotropy factor b
    const real_t* __restrict__ fac_c,      // anisotropy factor c
    const real_t* __restrict__ fac_f,      // anisotropy factor f
    const real_t* __restrict__ fun,        // slowness (1/velocity)
    const uint8_t* __restrict__ is_changed,// change flags
    // Level-set decomposition (precomputed on host)
    const idx_t* __restrict__ level_offsets,
    const idx_t* __restrict__ level_counts,
    const idx_t* __restrict__ level_indices,
    // Sweep parameters
    int level,                            // current FSM level
    int sign_r, int sign_t, int sign_p,   // sweep direction signs
    // Grid metadata
    int nx, int ny, int nz,
    real_t dr, real_t dt, real_t dp);

// ---------------------------------------------------------------------------
// Boundary update kernel
// ---------------------------------------------------------------------------
// Updates the 6 boundary faces using the CPU-matched extrapolation:
//   tau[boundary] = max(2*tau[adjacent] - tau[adjacent2], tau[adjacent2])
//
// One kernel launch handles all 6 faces.  Each thread handles one
// boundary node.
//
__global__ void boundary_kernel(
    real_t* __restrict__       tau,
    const real_t* __restrict__ T0v,
    const real_t* __restrict__ T0r,
    const real_t* __restrict__ T0t,
    const real_t* __restrict__ T0p,
    const real_t* __restrict__ fac_a,
    const real_t* __restrict__ fac_b,
    const real_t* __restrict__ fac_c,
    const real_t* __restrict__ fac_f,
    const real_t* __restrict__ fun,
    const uint8_t* __restrict__ is_changed,
    int nx, int ny, int nz,
    real_t dr, real_t dt, real_t dp,
    int stencil_order);

// ---------------------------------------------------------------------------
// Initialization kernel
// ---------------------------------------------------------------------------
// Initializes the `is_changed` flag array on the device.  Sets all
// interior nodes to `true` (needs update) and all boundary nodes to
// `false` (boundary handled separately).
//
__global__ void init_changed_kernel(
    uint8_t* __restrict__ is_changed,
    int nx, int ny, int nz);

// ---------------------------------------------------------------------------
// Convergence check kernel (L1 and Linf reduction)
// ---------------------------------------------------------------------------
// Computes the L1 and Linf norms of the change in tau between iterations.
// Uses a two-pass approach: first compute per-block partial sums, then
// reduce on the host (or with a second kernel for very large grids).
//
__global__ void convergence_reduce_kernel(
    const real_t* __restrict__ tau,
    const real_t* __restrict__ tau_old,
    int total_nodes,
    real_t* __restrict__ block_l1,    // per-block L1 partial sums
    real_t* __restrict__ block_linf); // per-block Linf partial max

// ---------------------------------------------------------------------------
// Utility kernels
// ---------------------------------------------------------------------------
// Set all elements of a real_t array to a constant value.
__global__ void fill_real_kernel(real_t* arr, real_t value, int n);

// Set all elements of a uint8_t array to a constant value.
__global__ void fill_bool_kernel(uint8_t* arr, uint8_t value, int n);

// Copy one real_t array to another (device-to-device).
__global__ void copy_real_kernel(real_t* dst, const real_t* src, int n);

}  // namespace tomogpu
