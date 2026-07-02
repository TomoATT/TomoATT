# GPU Backend Memory Layout (Clean-Slate v2)

**Objective**: Minimize GPU memory usage while maintaining native multidimensional layout and coalesced access patterns.

---

## 1. Memory Layout Comparison

### 1.1 Legacy GPU backend (cuda/)

The legacy code uses a **vectorized 3D-array representation** that duplicates the entire grid data **8×** (one copy per sweep direction):

```
Grid_on_device {
    // 8 copies of index arrays (one per sweep direction iswp ∈ [0,7])
    int* vv_i__j__k___0, ..., vv_i__j__k___7;    // 8 × N × 4 bytes
    int* vv_ip1j__k___0, ..., vv_ip1j__k___7;    // 8 × N × 4 bytes
    int* vv_im1j__k___0, ..., vv_im1j__k___7;    // 8 × N × 4 bytes
    // ... (7 index arrays × 8 copies = 56 arrays)

    // 8 copies of coefficient arrays
    CUSTOMREAL* vv_fac_a_0, ..., vv_fac_a_7;     // 8 × N × 8 bytes
    CUSTOMREAL* vv_T0v_0, ..., vv_T0v_7;         // 8 × N × 8 bytes
    // ... (10 coefficient arrays × 8 copies = 80 arrays)

    CUSTOMREAL* tau;                              // 1 × N × 8 bytes
}
```

**Total memory** (per field group, for N = nx*ny*nz nodes):
- Index arrays: 8 copies × 7 arrays × N × 4 bytes = **224 × N bytes**
- Coefficient arrays: 8 copies × 10 arrays × N × 8 bytes = **640 × N bytes**
- tau: N × 8 bytes
- **Grand total: ~872 × N bytes**

For a 128³ grid (N ≈ 2.1M): **~1.8 GiB**

### 1.2 New GPU backend (gpu/)

The new code stores **one copy** of each field in native 3D layout:

```
GpuGrid {
    // Native 3D arrays (one copy each, flat indexing)
    real_t* tau;         // N × 8 bytes (working array)
    real_t* T0v;         // N × 8 bytes
    real_t* T0r;         // N × 8 bytes
    real_t* T0t;         // N × 8 bytes
    real_t* T0p;         // N × 8 bytes
    real_t* fac_a;       // N × 8 bytes
    real_t* fac_b;       // N × 8 bytes
    real_t* fac_c;       // N × 8 bytes
    real_t* fac_f;       // N × 8 bytes
    real_t* fun;         // N × 8 bytes
    uint8_t* is_changed; // N × 1 byte

    // Level-set decomposition (precomputed once, reused for all 8 sweeps)
    idx_t* level_indices;  // N × 4 bytes
    idx_t* level_offsets;  // (max_level+1) × 4 bytes ≈ small
    idx_t* level_counts;   // (max_level+1) × 4 bytes ≈ small
}
```

**Total memory**:
- Field arrays: 10 × N × 8 bytes + N × 1 byte = **81 × N bytes**
- tau: N × 8 bytes
- Level decomposition: N × 4 bytes + O(max_level) × 12 bytes
- **Grand total: ~89 × N bytes**

For a 128³ grid (N ≈ 2.1M): **~180 MiB**

### 1.3 Memory reduction

```
Reduction factor = 872 / 89 ≈ 9.8×
```

| Grid size | Legacy (GiB) | New (MiB) | Reduction |
|-----------|-------------|-----------|-----------|
| 32³ | 0.028 | 2.9 | 9.8× |
| 64³ | 0.224 | 23 | 9.8× |
| 128³ | 1.79 | 183 | 9.8× |
| 256³ | 14.3 | 1460 | 9.8× |

---

## 2. Native 3D Layout Details

### 2.1 Indexing convention

The grid is stored in **row-major order** with i (longitude) varying fastest:

```
flat_index(i, j, k) = k * (nx * ny) + j * nx + i
```

This matches the CPU `I2V` macro:
```cpp
#define I2V(i, j, k) (k * loc_I * loc_J + j * loc_I + i)
```

### 2.2 Neighbor access

For a node at flat index `idx` with coordinates (i, j, k):
```
idx(i, j, k)         = k * nx*ny + j * nx + i
idx(i+1, j, k)       = idx + 1           (if i < nx-1)
idx(i-1, j, k)       = idx - 1           (if i > 0)
idx(i, j+1, k)       = idx + nx          (if j < ny-1)
idx(i, j-1, k)       = idx - nx          (if j > 0)
idx(i, j, k+1)       = idx + nx*ny       (if k < nz-1)
idx(i, j, k-1)       = idx - nx*ny       (if k > 0)
```

This enables O(1) neighbor access without precomputed index arrays.

### 2.3 SoA vs AoS analysis

The solver uses **Structure of Arrays (SoA)** layout:
- Each field (tau, T0v, fac_a, ...) is a separate contiguous array.
- All nodes' `tau` values are contiguous, all `T0v` values are contiguous, etc.

**Why SoA?**
- The stencil kernel accesses the same field across multiple nodes → **coalesced** memory access within each field.
- AoS would interleave fields, causing uncoalesced access when the kernel reads `tau[i]` then `T0v[i]`.
- SoA enables the compiler to vectorize memory accesses more effectively.

---

## 3. Level-Set Decomposition Memory

### 3.1 Structure

The level-set decomposition precomputes the mapping from (level, thread_id) to flat grid index:

```
level_indices[offset[L] + tid] = flat_index(i, j, k)  where i+j+k = L
```

**Memory cost**:
- `level_indices`: N × 4 bytes (one int32 per node)
- `level_offsets`: (max_level + 2) × 4 bytes ≈ (nx + ny + nz) × 4 bytes
- `level_counts`: (max_level + 1) × 4 bytes

For a 128³ grid:
- `level_indices`: 2.1M × 4 bytes = 8.4 MiB
- `level_offsets`: ~384 × 4 bytes ≈ 1.5 KiB
- `level_counts`: ~384 × 4 bytes ≈ 1.5 KiB

**Total level decomposition memory**: ~8.4 MiB (for 128³)

### 3.2 Reuse across sweep directions

The level-set decomposition is precomputed **once** for the natural grid (level = i + j + k) and **reused for all 8 sweep directions**.

For each sweep direction, the kernel applies an O(1) index flip to map the natural (i, j, k) to the physical (i, j, k) where the stencil is computed:

```cuda
// Sweep direction signs: sign_r, sign_t, sign_p ∈ {+1, -1}
phys_i = (sign_p > 0) ? (i - 1) : (nx - i);
phys_j = (sign_t > 0) ? (j - 1) : (ny - j);
phys_k = (sign_r > 0) ? (k - 1) : (nz - k);
```

This eliminates the need for 8 separate copies of the index arrays.

---

## 4. Temporary Buffers

### 4.1 Convergence checking

- `tau_old`: N × 8 bytes — stores the previous iteration's tau for L1/Linf comparison.
- `block_l1`, `block_linf`: ~1024 × 8 bytes each — per-block partial sums for reduction.

### 4.2 CUDA Graph capture

During CUDA Graph capture, the stream is in capture mode. No additional memory is required beyond the graph itself (~a few KiB for the graph structure).

---

## 5. Stream-Ordered Memory Allocation

All device allocations use `cudaMallocAsync` / `cudaFreeAsync` (stream-ordered allocation):

```cpp
// In DeviceBuffer<T>::resize():
cudaMallocAsync(reinterpret_cast<void**>(&ptr), count * sizeof(T), stream);
// ...
// In DeviceBuffer<T>::reset():
cudaFreeAsync(ptr, stream);
```

**Benefits**:
- **Zero fragmentation**: Stream-ordered allocations come from a memory pool that reuses freed regions.
- **Fast allocation**: No driver overhead for repeated alloc/free cycles.
- **Thread-safe**: Multiple streams can allocate concurrently.

**Blackwell-specific**: The GB200's HBM3 has 8 TB/s bandwidth per GPU. Stream-ordered allocation ensures the memory pool is cache-friendly.

---

## 6. Memory Bandwidth Analysis

### 6.1 Per-sweep memory traffic

Each FSM sweep reads and writes the following arrays per node:

| Array | Access | Bytes (double) |
|-------|--------|----------------|
| tau (center) | read | 8 |
| tau (6 neighbors) | read | 48 |
| T0v, T0r, T0t, T0p | read | 32 |
| fac_a, fac_b, fac_c, fac_f | read | 32 |
| fun | read | 8 |
| is_changed | read | 1 |
| tau (center) | write | 8 |
| **Total** | | **137 bytes/node** |

### 6.2 Achievable bandwidth

For a 128³ grid (2.1M nodes):
- Memory traffic per sweep: 2.1M × 137 bytes ≈ 287 MiB
- At 8 TB/s (B200 HBM3): theoretical minimum = 287 MiB / 8 TB/s ≈ 36 μs
- Actual (with launch overhead): ~1-10 ms per sweep

The kernel is **memory-bandwidth bound**, so optimizing for coalesced access and L2 cache reuse is critical.

---

## 7. Summary

| Aspect | Legacy (cuda/) | New (gpu/) | Improvement |
|--------|----------------|------------|-------------|
| Field storage | 8 copies × 10 fields | 1 copy × 10 fields | 8× reduction |
| Index arrays | 8 copies × 7 arrays | 1 level array | ~56× reduction |
| Total memory | ~872 × N bytes | ~89 × N bytes | **~9.8× reduction** |
| Allocation | `cudaMalloc` (sync) | `cudaMallocAsync` (stream-ordered) | Zero fragmentation |
| Layout | Vectorized (gather/scatter) | Native 3D (direct indexing) | Simpler, cache-friendly |
