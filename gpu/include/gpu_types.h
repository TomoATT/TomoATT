//
// gpu_types.h — Fundamental types, constants, and enumerations for the
// clean-slate GPU backend.
//
// Design goals:
//   * Strong typing for grid indices, level indices, and sweep directions.
//   * constexpr for all compile-time constants (no macros).
//   * Single source of truth for numerical precision and grid topology.
//
// This header is included by both host (C++) and device (CUDA) translation
// units, so it must remain free of CUDA-specific extensions.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace tomogpu {

// ---------------------------------------------------------------------------
// Numerical precision
// ---------------------------------------------------------------------------
// The CPU reference solver uses `CUSTOMREAL` (double by default).  The GPU
// backend mirrors this choice via `real_t`.  Single-precision can be enabled
// at compile time (-DTOMOGPU_SINGLE_PRECISION) for memory-constrained
// regimes; the correctness framework quantifies the resulting error.
#ifdef TOMOGPU_SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

// Index type for flat 3D grid storage.  int32_t is sufficient for all
// problem sizes that fit in GB200 memory and keeps index arrays compact.
using idx_t = int32_t;

// ---------------------------------------------------------------------------
// Sweep directions
// ---------------------------------------------------------------------------
// The Fast Sweeping Method (FSM) traverses the grid in 8 quadrant
// directions.  Each direction is identified by the sign of the unit vector
// along the three axes (r, t, p) = (k, j, i):
//
//   dir_id = (p_sign > 0 ? 4 : 0) | (t_sign > 0 ? 2 : 0) | (r_sign > 0 ? 1 : 0)
//
// The convention matches the CPU `set_sweep_direction(iswp)` ordering.
//
//   0: (+r,+t,+p)   1: (+r,+t,-p)   2: (+r,-t,+p)   3: (+r,-t,-p)
//   4: (-r,+t,+p)   5: (-r,+t,-p)   6: (-r,-t,+p)   7: (-r,-t,-p)
//
enum class SweepDir : idx_t {
    PPP = 0, PPM = 1, PMP = 2, PMM = 3,
    MPP = 4, MPM = 5, MMP = 6, MMM = 7
};

constexpr idx_t kNumSweepDirs = 8;

// Sign helpers for a given sweep direction.
// Returns +1 or -1 for the r/t/p axis.
constexpr int sweep_sign_r(SweepDir d) {
    return (static_cast<idx_t>(d) & 1) ? +1 : -1;
    // Note: bit 0 encodes the r sign in the CPU convention where
    // iswp=0..3 have r_dirc=+1 and iswp=4..7 have r_dirc=-1.
}
// Correct bit layout:
//   bit 0: r sign (0 => -1, 1 => +1)  ... actually CPU uses:
//   iswp 0: r+ t+ p+   iswp 1: r+ t+ p-   iswp 2: r+ t- p+   iswp 3: r+ t- p-
//   iswp 4: r- t+ p+   iswp 5: r- t+ p-   iswp 6: r- t- p+   iswp 7: r- t- p-
// So: r_sign = (iswp < 4) ? +1 : -1
//     t_sign = (iswp % 4 < 2) ? +1 : -1
//     p_sign = (iswp % 2 == 0) ? +1 : -1

constexpr int sign_r(SweepDir d) {
    idx_t i = static_cast<idx_t>(d);
    return (i < 4) ? +1 : -1;
}
constexpr int sign_t(SweepDir d) {
    idx_t i = static_cast<idx_t>(d) % 4;
    return (i < 2) ? +1 : -1;
}
constexpr int sign_p(SweepDir d) {
    idx_t i = static_cast<idx_t>(d) % 2;
    return (i == 0) ? +1 : -1;
}

// ---------------------------------------------------------------------------
// Stencil configuration
// ---------------------------------------------------------------------------
enum class StencilOrder : idx_t { First = 1, Third = 3 };

// ---------------------------------------------------------------------------
// Grid topology
// ---------------------------------------------------------------------------
// Describes the local subdomain owned by one MPI rank / GPU.
//   nx = loc_I  (longitude / phi,   index i)
//   ny = loc_J  (latitude / theta, index j)
//   nz = loc_K  (radius  / r,      index k)
//
// Storage is row-major with i varying fastest, matching the CPU `I2V` macro:
//   flat_index(i,j,k) = k * nx*ny + j * nx + i
//
struct GridDims {
    idx_t nx = 0;
    idx_t ny = 0;
    idx_t nz = 0;

    constexpr idx_t total()    const { return nx * ny * nz; }
    constexpr idx_t interior()  const {
        // interior nodes (exclude 1 ghost layer each side)
        return (nx > 2 && ny > 2 && nz > 2)
            ? (nx - 2) * (ny - 2) * (nz - 2)
            : 0;
    }
    constexpr idx_t max_level() const { return nx + ny + nz - 3; }
};

// Flat index into the 3D grid.
constexpr idx_t flat_index(idx_t i, idx_t j, idx_t k, const GridDims& d) {
    return k * d.nx * d.ny + j * d.nx + i;
}

// ---------------------------------------------------------------------------
// Physical constants used by the stencil kernels
// ---------------------------------------------------------------------------
// Eikonal normalization: T = T0v * tau.  The LF Hamiltonian uses these
// small epsilon values for the WENO-style smoothness indicators.
constexpr real_t kEps       = static_cast<real_t>(1e-12);
constexpr real_t kHalf      = static_cast<real_t>(0.5);
constexpr real_t kOne       = static_cast<real_t>(1.0);
constexpr real_t kTwo       = static_cast<real_t>(2.0);
constexpr real_t kThree     = static_cast<real_t>(3.0);
constexpr real_t kFour      = static_cast<real_t>(4.0);
constexpr real_t kTauInit   = static_cast<real_t>(1.0);   // TAU_INITIAL_VAL
constexpr real_t kTauInf    = static_cast<real_t>(20.0);  // TAU_INF_VAL

// Sweeping coefficient (matches CPU SWEEPING_COEFF).
constexpr real_t kSweepCoeff = static_cast<real_t>(1.0);

// ---------------------------------------------------------------------------
// Execution configuration
// ---------------------------------------------------------------------------
// Block size chosen for high occupancy on Blackwell (148 SMs, 2048 threads/SM
// for 64-bit registers).  256 threads gives 8 warps per block and allows
// 8 blocks per SM at typical register pressure.
constexpr idx_t kDefaultBlockSize = 256;

// Maximum number of CUDA streams used for overlapping computation and
// communication.
constexpr idx_t kMaxStreams = 4;

}  // namespace tomogpu
