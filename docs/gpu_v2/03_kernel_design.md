# GPU Backend Kernel Design (Clean-Slate v2)

**Objective**: Design kernels with clear responsibilities, minimal branching, predictable execution, and reusable components.

---

## 1. Kernel Inventory

| Kernel | Responsibility | Template | Launch pattern |
|--------|---------------|----------|---------------|
| `sweep_level_kernel<kOrder>` | FSM stencil update for one level | 1st / 3rd order | 1D grid, one launch per level |
| `boundary_kernel` | Boundary face extrapolation | — | 1D grid, one launch per sweep |
| `init_changed_kernel` | Initialize `is_changed` flags | — | 1D grid, one launch at init |
| `convergence_reduce_kernel` | L1/Linf reduction for convergence | — | 1D grid with shared-mem reduction |
| `copy_real_kernel` | Device-to-device copy | — | 1D grid |
| `fill_real_kernel` | Fill array with constant | — | 1D grid |

---

## 2. `sweep_level_kernel` — Detailed Design

### 2.1 Signature

```cuda
template <int kOrder>
__global__ void sweep_level_kernel(
    real_t* __restrict__       tau,        // in-place update
    const real_t* __restrict__ T0v,        // initial traveltime
    const real_t* __restrict__ T0r,        // dT0/dr
    const real_t* __restrict__ T0t,        // dT0/dt
    const real_t* __restrict__ T0p,        // dT0/dp
    const real_t* __restrict__ fac_a,      // anisotropy a
    const real_t* __restrict__ fac_b,      // anisotropy b
    const real_t* __restrict__ fac_c,      // anisotropy c
    const real_t* __restrict__ fac_f,      // anisotropy f
    const real_t* __restrict__ fun,        // slowness
    const uint8_t* __restrict__ is_changed,// change flags
    const idx_t* __restrict__ level_offsets,
    const idx_t* __restrict__ level_counts,
    const idx_t* __restrict__ level_indices,
    int level,                            // current FSM level
    int sign_r, int sign_t, int sign_p,   // sweep direction signs
    int nx, int ny, int nz,               // grid dimensions
    real_t dr, real_t dt, real_t dp);     // grid spacings
```

### 2.2 Algorithm

```cuda
// 1. Get thread ID and check bounds
tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid >= level_counts[level]) return;

// 2. Look up flat grid index from precomputed level decomposition
flat_idx = level_indices[level_offsets[level] + tid];

// 3. Convert flat index to natural (i, j, k)
k = flat_idx / (nx * ny);
j = (flat_idx - k * nx * ny) / nx;
i = flat_idx - k * nx * ny - j * nx;

// 4. Apply sweep direction index flip to get physical (i, j, k)
phys_i = (sign_p > 0) ? (i - 1) : (nx - i);
phys_j = (sign_t > 0) ? (j - 1) : (ny - j);
phys_k = (sign_r > 0) ? (k - 1) : (nz - k);

// 5. Skip out-of-bounds and boundary nodes
if (phys_i < 1 || phys_i >= nx-1) return;
if (phys_j < 1 || phys_j >= ny-1) return;
if (phys_k < 1 || phys_k >= nz-1) return;

phys_flat = flat_index(phys_i, phys_j, phys_k, {nx, ny, nz});

// 6. Skip unchanged nodes
if (!is_changed[phys_flat]) return;

// 7. Compute stencil coefficients (sigma values)
sigr = kSweepCoeff * sqrt(fac_a[phys_flat]) * T0v[phys_flat];
sigt = kSweepCoeff * sqrt(fac_b[phys_flat]) * T0v[phys_flat];
sigp = kSweepCoeff * sqrt(fac_c[phys_flat]) * T0v[phys_flat];
coe  = 1 / (sigr/dr + sigt/dt + sigp/dp);

// 8. Compute finite-difference quotients (centered, 1st or 3rd order)
pp1 = (tau[phys_flat] - tau[phys_flat - 1])      / dp;
pp2 = (tau[phys_flat + 1] - tau[phys_flat])      / dp;
pt1 = (tau[phys_flat] - tau[phys_flat - nx])     / dt;
pt2 = (tau[phys_flat + nx] - tau[phys_flat])     / dt;
pr1 = (tau[phys_flat] - tau[phys_flat - nx*ny])  / dr;
pr2 = (tau[phys_flat + nx*ny] - tau[phys_flat])  / dr;

// 9. Compute LF Hamiltonian
H = sqrt(fac_a * (T0r*tau + T0v*(pr1+pr2)/2)²
       + fac_b * (T0t*tau + T0v*(pt1+pt2)/2)²
       + fac_c * (T0p*tau + T0v*(pp1+pp2)/2)²
       - 2*fac_f * (T0t*tau + ...) * (T0p*tau + ...));

// 10. Update tau in-place
correction = (sigr*(pr2-pr1) + sigt*(pt2-pt1) + sigp*(pp2-pp1)) / 2;
tau[phys_flat] += coe * (fun - H + correction);
```

### 2.3 Design decisions

**Why template `<int kOrder>`?**
- Enables compile-time dispatch between 1st and 3rd order stencils.
- Zero runtime overhead — the compiler eliminates dead code for the unused order.
- The kernel body is shared; only the stencil computation differs.

**Why `__restrict__` on all pointers?**
- Tells the compiler that `tau`, `T0v`, `fac_a`, etc. don't alias each other.
- Enables aggressive load/store reordering and vectorization.

**Why `__launch_bounds__(256)`?**
- Limits the compiler to 256 threads per block, matching our launch configuration.
- Guides register allocation to ensure high occupancy (8+ blocks per SM).

**Why skip boundary nodes in the kernel?**
- Boundary nodes require a different update formula (extrapolation).
- Handling them separately avoids branch divergence in the stencil kernel.
- The `boundary_kernel` is launched once per sweep direction.

**Why in-place update?**
- Nodes at the same FSM level are independent — they don't read each other's updated values.
- In-place update avoids double-buffering, halving the memory for `tau`.
- The level-set decomposition guarantees no read-after-write hazards within a level.

---

## 3. `boundary_kernel` — Boundary Handling

### 3.1 Algorithm

The boundary kernel updates the 6 boundary faces using the CPU-matched extrapolation:

```cuda
// For each boundary face:
//   tau[boundary] = max(2*tau[adjacent] - tau[adjacent2], tau[adjacent2])
//
// Faces:
//   k=0 (bottom):    tau[i,j,0]   = max(2*tau[i,j,1] - tau[i,j,2], tau[i,j,2])
//   k=nz-1 (top):    tau[i,j,nz-1] = max(2*tau[i,j,nz-2] - tau[i,j,nz-3], tau[i,j,nz-3])
//   j=0 (south):     tau[i,0,k]   = max(2*tau[i,1,k] - tau[i,2,k], tau[i,2,k])
//   j=ny-1 (north):  tau[i,ny-1,k] = max(2*tau[i,ny-2,k] - tau[i,ny-3,k], tau[i,ny-3,k])
//   i=0 (west):      tau[0,j,k]   = max(2*tau[1,j,k] - tau[2,j,k], tau[2,j,k])
//   i=nx-1 (east):   tau[nx-1,j,k] = max(2*tau[nx-2,j,k] - tau[nx-3,j,k], tau[nx-3,j,k])
```

### 3.2 Launch configuration

- One kernel launch per sweep direction (after the level-set sweep).
- Each thread handles one boundary node.
- Total threads = 2*(nx*ny + nx*nz + ny*nz) (6 faces, counting edges once).

---

## 4. `convergence_reduce_kernel` — Convergence Check

### 4.1 Algorithm

This kernel computes the L1 and Linf norms of the change in tau between iterations:

```cuda
// For each node i:
//   diff = |tau[i] - tau_old[i]|
//   L1  += diff
//   Linf = max(Linf, diff)
//
// Two-pass reduction:
//   1. Each block computes partial L1 sum and Linf max (using shared memory)
//   2. Host reduces the per-block results
```

### 4.2 Shared memory usage

```cuda
extern __shared__ real_t sdata[];
// sdata[0..blockDim.x-1] = per-thread L1 contributions
// After reduction: sdata[0] = block-level L1 sum
// Repeat for Linf
```

Shared memory per block: `blockDim.x * sizeof(real_t)` = 256 × 8 = 2 KiB.

---

## 5. Kernel Launch Overhead and CUDA Graphs

### 5.1 Problem

The FSM requires launching one kernel per level per sweep direction:
```
total_launches = 8 directions × (nx + ny + nz - 2) levels ≈ 8 × 3N^{1/3}
```

For N = 128³: ~3000 kernel launches per sweep. At ~5 μs launch overhead each, this is ~15 ms of pure overhead per sweep.

### 5.2 Solution: CUDA Graphs

The entire sweep loop is captured as a CUDA Graph:

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
for (dir = 0; dir < 8; dir++) {
    for (level = 0; level < num_levels; level++) {
        launch_level_kernel(dir, level, ...);
    }
}
cudaStreamEndCapture(stream, &sweep_graph);
cudaGraphInstantiate(&sweep_exec, sweep_graph, ...);
```

Subsequent sweeps replay the graph:
```cpp
cudaGraphLaunch(sweep_exec, stream);
```

**Benefits**:
- Eliminates kernel launch overhead — the graph is a single API call.
- The graph structure is optimized by the driver (kernel fusion, memory transfer coalescing).
- Replay is ~10× faster than individual launches.

---

## 6. Thread Block Shape Analysis

### 6.1 Why 1D blocks of 256 threads?

The level-set decomposition produces a **1D list of nodes** per level. Mapping threads to nodes is simplest with 1D blocks.

256 threads = 8 warps per block:
- **Occupancy**: On Blackwell (148 SMs, 2048 threads/SM), 256-thread blocks allow 8 blocks per SM, achieving 100% occupancy.
- **Register pressure**: The stencil kernel uses ~40 registers per thread. At 256 threads/block, this is ~10 KiB of registers per block, fitting comfortably in the 64 KiB register file.
- **Shared memory**: The kernel doesn't use shared memory (the level-set decomposition is read from global memory with good L2 cache reuse).

### 6.2 Why not 2D/3D blocks?

The stencil is 3D, so 3D blocks seem natural. However:
- The level-set decomposition is 1D (nodes at the same level), so 3D blocks would require complex index mapping.
- 1D blocks are simpler and equally efficient for this access pattern.
- The physical (i,j,k) coordinates are computed from the 1D thread ID, not from 3D block indices.

---

## 7. Memory Coalescing Analysis

### 7.1 `level_indices` access

```cuda
flat_idx = level_indices[level_offsets[level] + tid];
```

Consecutive threads (tid, tid+1, tid+2, ...) access consecutive elements of `level_indices` → **perfectly coalesced**.

### 7.2 Field array access

```cuda
tau[phys_flat]  // phys_flat varies per thread
```

The `phys_flat` index varies per thread (different physical nodes). However:
- Nodes at the same FSM level have similar (i,j,k) values (they differ by shifting between axes).
- So `phys_flat` for consecutive threads varies by small amounts → **spatial locality** is good.
- The L2 cache captures the neighbor accesses well.

**Trade-off**: The legacy code used precomputed index arrays to achieve gather/scatter access, but this required 8× memory duplication. The new code uses direct (i,j,k) indexing with O(1) neighbor computation, accepting slightly less coalescing in exchange for ~10× memory reduction.

---

## 8. Occupancy and Register Usage

### 8.1 Register budget

The `sweep_level_kernel` uses approximately:
- 20 registers for loop variables and intermediate values
- 10 registers for the 7-13 neighbor values
- 10 registers for the stencil computation
- **Total: ~40 registers per thread**

### 8.2 Occupancy calculation

For Blackwell (B200):
- 148 SMs per GPU
- 65,536 registers per SM (32-bit)
- 2048 threads per SM (max)

At 40 registers/thread:
- Max threads per SM = 65,536 / 40 = 1638 (rounded down to a multiple of 32)
- Occupancy = 1638 / 2048 = 80%

At 256 threads/block:
- Blocks per SM = 1638 / 256 = 6 (rounded down)
- Active threads per SM = 6 × 256 = 1536
- Occupancy = 1536 / 2048 = 75%

This is good occupancy for a memory-bound kernel.

---

## 9. Future Kernel Optimizations

### 9.1 Shared memory tiling

For very large grids, the stencil kernel could benefit from shared memory tiling:
- Load a 3D tile of `tau` into shared memory.
- Compute the stencil using shared memory accesses.
- Reduces global memory traffic by ~7× (7 neighbors per node).

**Trade-off**: Shared memory tiling adds complexity and may reduce occupancy. Should be evaluated with Nsight Compute profiling after the baseline is correct.

### 9.2 Warp-level primitives

The convergence reduction kernel could use `__shfl_down_sync` for warp-level reduction instead of shared memory:
```cuda
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
```

This reduces shared memory usage and improves reduction performance.

### 9.3 Persistent kernels with cooperative groups

For very large grids with many levels, a persistent kernel with cooperative groups could process all levels in a single kernel launch, using `cudaGridGroupSync()` for inter-level synchronization. This would eliminate all kernel launch overhead.

**Trade-off**: Cooperative launch requires all blocks to be resident simultaneously, limiting the grid size. Should be evaluated after the CUDA Graph approach is validated.
