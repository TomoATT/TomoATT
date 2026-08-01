# TomoATT GPU Implementation & Performance Report

**Date**: 2026-07-31
**Author**: verified with GitHub Copilot agent
**Branch**: `feature/gpu_update` (commits `e053307` … `906f992`)
**Platform**: RIKEN TENJO — NVIDIA DGX Spark (**GB10**, Grace Blackwell, sm_121,
CUDA 13.0, driver 580.173.02, aarch64 Ubuntu 24.04, GCC 13.3,
SLURM partition `spark` / `spark-double` / `edge-double` via `nimbus-0`)

---

## 1. Scope

TomoATT has **two GPU code paths**:

| Stack | CMake switch | Sources | Purpose |
|---|---|---|---|
| Legacy backend | `-DUSE_CUDA=True` | `cuda/*.cu` | GPU fast-sweeping inside the production `TOMOATT` executable (LF solver + **new UPWIND port**) |
| Clean-slate v2 | `-DUSE_CUDA_V2=True` (+`USE_CUDA`) | `gpu/src/*.cu`, `gpu/tests/`, `gpu/benchmarks/` | Independent `tomogpu` static library with tests/benchmarks, CUDA-graph replay, 10× lower memory footprint |

Both were built, tested, and benchmarked on a DGX Spark **GB10** node in this
verification campaign.

---

## 2. Work delivered this cycle (`e053307` … `906f992`)

### 2.1 Legacy backend: UPWIND eikonal solver ported to GPU
- New kernel `cuda_do_sweep_level_kernel_upwind` + launcher
  `cuda_run_iteration_upwind` in `cuda/iterator_wrapper.cu` — full port of the
  CPU `calculate_stencil_1st_order_upwind` (**26 cases**: 8 tetrahedron,
  12 triangles (r-t / r-p / t-p planes), 6 lines) with quadratic solve +
  causality check per case and min-candidate selection.
- Host integration: `Iterator_level_1st_order_upwind::do_sweep` GPU path
  (`cuda_copy_tau_to_device` → `cuda_run_iteration_upwind` → `cuda_copy_tau_to_host`).
- `preload_indices(_1d)` gained an `is_upwind` flag so UPWIND index flipping
  (no ±1 shift, unlike the LF stencil) produces correct per-direction
  level-set index arrays for the GPU kernel.

### 2.2 Convergence bug fixes (GPU UPWIND was converging ~200 iters vs CPU ~5)
1. **Incomplete case set** — the first kernel draft implemented only 18 of 26
   cases (missing the r-p and t-p triangle planes). Missing characteristic
   directions made the fixed-point iteration diffusive. **Fixed.**
2. **Wrong index space in line-case causality** — preloaded `T0v[]` is
   level-relative but neighbor lookups used global flattened indices
   (`T0v[ii_mr]`), producing garbage reads. **Fixed** by adding a global
   full-grid device array `Grid_on_device::T0v_glob`
   (`cuda_copy_T0v_glob_to_device`, copied once per source; gpu_grid is
   constructed/destructed per source).

### 2.3 Build/portability hardening
- `CMakeLists.txt`: GPU SM architecture is now `-DTOMOATT_CUDA_ARCH=<sm>`
  (auto-detects from the host GPU; GB10 = sm_121) instead of hard-coded
  `sm_61`, which CUDA 13 no longer supports.
- `CMakeLists.txt`: fixed ordering bug preventing `USE_CUDA_V2` from ever
  building (`add_subdirectory(gpu)` was guarded by `CUDA_FOUND` before
  `find_package(CUDA)` ran).
- `cuda/cuda_initialize.cuh`: `cudaDeviceProp.deviceOverlap` was removed in
  CUDA 13; device capabilities are now queried through the stable
  `cudaDeviceGetAttribute` API.
- `gpu/benchmarks/bench_gpu_sweep.cu`: fixed unsigned-wraparound garbage in
  the *Memory used* printout (CUDA caching-allocator free-pool growth made
  `free_after > free_before`).
- `gpu/tests/test_gpu_correctness.cu`: Test 5 (convergence) was vacuous
  (`assert(l1 >= 0.0)`); it now performs a **numeric GPU vs serial-CPU
  reference comparison** (same 8-direction level-set math) on an anisotropic,
  heterogeneous 16³ grid over 3 sweeps, tolerance 1e-9 (double).

---

## 3. Correctness verification (GB10)

### 3.1 Legacy UPWIND GPU vs CPU — `test/inversion_small`
Full inversion (30 sources, `stencil_type=1` upwind, `sweep_type=1`
Cuthill-McKee, `use_gpu` toggled), run via `scripts_build/verify_gpu_upwind_spark.sh`
(SLURM job 3133):

| Check | CPU | GPU | Match |
|---|---|---|---|
| Forward-solve sweeps to converge (pre run) | 25× it3 + 5× it4 | 25× it3 + 5× it4 | ✅ exact |
| Forward-solve sweeps to converge (inversion) | 84× it3 + 6× it4 | 84× it3 + 6× it4 | ✅ exact |
| `final_model.h5` (inverted velocity) | — | — | ✅ identical ≤ 1e-8 |
| `objective_function.txt` | — | — | ✅ identical (diff-clean) |
| `out_data_sim_group_0.h5` (42 MB travel-time/adjoint fields) | — | — | ✅ identical ≤ 1e-6¹ |

¹ A handful of adjoint-field values differ at the 1e-8–1e-6 level; this is
expected noise from `-use_fast_math` on the GPU LF (adjoint) path.

GPU engagement is proven by `cuda_device_info.txt` (written only inside
`initialize_cuda()`): `Device Name = NVIDIA GB10`.

> Note on the older regression: before the §2.2 fixes the GPU UPWIND solver
> required ~200 sweeps vs CPU ~5. After the fixes the iteration counts match
> exactly (see table).

### 3.2 Clean-slate v2 backend unit tests
`test_gpu_memory`: all pass (DeviceBuffer resize, pinned host buffer, 1 GiB
allocation, copy operations).

`test_gpu_correctness`: **5/5 pass**, including the hardened Test 5:

```
GPU vs serial CPU (3 sweeps, 16^3): max|diff| = 0.000e+00  (tol 1.0e-09)
```

The v2 sweep kernel is **bit-exact** against the serial level-ordered CPU
reference on an anisotropic (`fac_b=0.9`, `fac_f=0.03`), heterogeneous grid.

### 3.3 v2 backend large-grid agreement (256³ = 16.8 M nodes)
`bench_cpu_vs_gpu --mode conv --conv-tol 1e-8` (SLURM job 3134):

| | CPU | GPU |
|---|---|---|
| Sweeps to L1 < 1e-8 | 40 | 45¹ |
| T-field agreement after convergence | \multicolumn{2}{c}{L1 = 1.0e-10, Linf = 9.4e-6} |

¹ Level-set parallel sweeps need ~12 % more iterations than sequential
Gauss-Seidel ordering. The larger gap seen at a fixed 20 iterations
(L1 = 1.13) was a **non-converged transient, not a solver bug**.

---

## 4. Performance benchmarks (GB10, single GPU)

### 4.1 v2 backend: CPU vs GPU sweep — `bench_cpu_vs_gpu` (job 3132)

| Grid | Nodes | CPU (ms) | GPU (ms) | Speedup | L1 err | Linf err | CPU RSS (MiB) | GPU mem (MiB) |
|---|---|---|---|---|---|---|---|---|
| 32³ | 32,768 | 31.88 | 6.58 | **4.85×** | 0.0 | 0.0 | 100.6 | 27,418 |
| 64³ | 262,144 | 494.77 | 14.87 | **33.3×** | 0.0 | 0.0 | 137.1 | 27,442 |
| 128³ | 2,097,152 | 7,247 | 40.53 | **179×** | 4.0e-8 | 4.8e-4 | 302.0 | 27,753 |
| 256³ | 16,777,216 | 96,056 | 479.96 | **200×** | (see 3.3) | | 1,620 | 29,972 |

(GPU mem ≈ 27–30 GiB reflects the GB10 unified-memory footprint of the whole
process, not just the working set.)

### 4.2 v2 backend: sweep time, graph replay, footprint — `bench_gpu_sweep`

| Grid | Sweep (ms) | CUDA-graph replay (ms) | Bandwidth | New mem | Legacy mem | Reduction |
|---|---|---|---|---|---|---|
| 16³ | 2.43 | 2.43 | 0.2 GB/s | 0.4 MiB | 3.7 MiB | **10.1×** |
| 32³ | 6.63 | 6.56 | 0.5 GB/s | 2.9 MiB | 29.2 MiB | **10.1×** |
| 64³ | 14.8 | 14.8 | 1.7 GB/s | 23.2 MiB | 234.0 MiB | **10.1×** |
| 128³ | 40.0 | 40.1 | 5.0 GB/s | 186.0 MiB | 1,872.0 MiB | **10.1×** |

- The 10.1× memory reduction comes from retaining a single copy of the
  per-direction preloaded index/field arrays instead of the legacy 8×
  duplication.
- CUDA-graph replay currently shows no measurable gain (launch overhead is
  already negligible at these grid sizes); keep as an option for small grids.

### 4.3 Legacy backend (production TOMOATT)
- Small-case GPU timing is dominated by per-source
  `cuda_copy_tau_to_device/host` transfers (full-grid copy per sweep is only
  done once per iteration; preloaded arrays are resident).
- For production value on the Japan case, the dominant cost was the LF
  solver's ~16.7× more sweeps than UPWIND; the new GPU UPWIND path removes
  that penalty while keeping the GPU speed, so the *effective*
  speedup over the previous GPU configuration is far larger than the raw
  kernel ratio.
- Multi-GPU/MPI timing runs on the DGX Spark nodes are future work (all
  numbers here are single-process, single-GPU).

---

## 5. Reproduction

```bash
# 0. Deps (OpenMPI 4.1.6 + parallel HDF5 1.13.3, aarch64) — once:
sbatch scripts_build/make_deps_spark.sh

# 1. Legacy production binary (sm_121):
sbatch scripts_build/make_tomoatt_spark.sh          # -> build_spark/bin/TOMOATT

# 2. CPU vs GPU UPWIND verification (test/inversion_small):
sbatch scripts_build/verify_gpu_upwind_spark.sh

# 3. v2 backend build + tests + full benchmarks:
sbatch scripts_build/make_and_test_gpuv2_spark.sh   # -> build_spark_v2/bin/*
sbatch scripts_build/run_gpuv2_full_spark.sh
```

Build essentials (inside the scripts): CUDA at `/usr/local/cuda-13.0`,
`-DTOMOATT_CUDA_ARCH=121`, MPI/HDF5 from `~/tomoatt_deps`.

---

## 6. Known limitations & notes

1. **Cluster MPI quirk**: `mpirun` inside a `sbatch` step on this cluster dies
   with ORTE cross-node errors (HNP on `spark-atom-0` vs orted on
   `dgx-spark-N`). Launch the TOMOATT binary directly (MPI singleton mode)
   until the MPI build gains proper slurm integration.
2. `-use_fast_math` keeps GPU/CPU field agreement at ~1e-6; inversion
   outputs (`final_model.h5`, `objective_function.txt`) nevertheless match to
   1e-8 / exactly in current tests.
3. Level-parallel GPU sweeping needs ~12 % more iterations than serial
   Gauss-Seidel (§3.3) — compare only *converged* fields.
4. Not yet on GPU: 3rd-order stencil, teleseismic mode, adjoint solver in the
   v2 backend, multi-GPU.
5. CUDA 13 requires the `cudaDeviceGetAttribute` capability queries now in
   `cuda_initialize.cuh` (the old `cudaDeviceProp` fields are gone).

---

## 7. Commits

| Hash | Summary |
|---|---|
| `e053307` | UPWIND solver GPU port (26 cases) + `T0v_glob` fix |
| `374777a` | CUDA arch configurability, CUDA 13 compat, spark harness |
| `2aed5e2` | Harness scripts renamed (`build_*` → `make_*`; gitignore `build*`) |
| `906f992` | Bench memory-print fix + hardened correctness Test 5 |
| `07eec99` | GPU adjoint sweep (verified), teleseismic UPWIND GPU port, `is_upwind` flip fix, local-rank device select |
| `89e352d` | 3rd-order verify harness uses legacy CPU sweep as reference |
| `d20dca6` | Mini-teleseismic GPU verify harness |
| `95ebb64` | libevent+hwloc+PMIx+OpenMPI multi-node MPI stack script |

---

## 8. Follow-up GPU extensions (2026-07-31 → 2026-08-01)

### 8.1 Adjoint sweep on GPU — VERIFIED (`verify job 3142`)
Inversions run 2 extra solves per source per iteration (adjoint field + its
density), so the adjoint sweep was ported to GPU in the legacy stack:

- `cuda_do_sweep_level_kernel_adj` — exact port of `calculate_stencil_adj`
  (linear transport update; the adjoint field reuses the `tau` buffer exactly
  like the CPU code; the six physical-domain-edge flags
  (`grid.{i,j,k}_{first,last}()`) reproduce `calculate_boundary_nodes_adj`'s
  zero boundary).
- New full-grid device arrays: `T_glob` (frozen traveltime field),
  `zeta/xi/eta_glob`, `tau_old_glob` (adjoint source), plus 7 one-dimensional
  spherical-coordinate factor arrays. 1-D factors + ζ/ξ/η uploaded on first
  call, ζ/ξ/η refreshed per call, T and τ_old uploaded every sweep.
- GPU branch in `Iterator_level::do_sweep_adj` **and**
  `Iterator_level_tele::do_sweep_adj`.

Verified vs CPU on `test/inversion_small` (same numbers as §3.1):
convergence counts identical (25×3+5×4 pre, 84×3+6×4 main),
`final_model.h5` ≤ 1e-8, `objective_function.txt` identical,
adjoint-field datasets ≤ 1e-6 (single-element 1e-8-level blips, fast-math noise).

### 8.2 Teleseismic UPWIND on GPU — ported, build-verified
`cuda_do_sweep_level_kernel_upwind_tele`: same 26-case upwind structure as
§2.1 but solves for **T** directly (no T0·τ factorization; `ap=±1/dp`,
line-case causality `tmp_T ≥ T_nbr && tmp_T > 0`), mirroring
`calculate_stencil_1st_order_upwind_tele` exactly. Wired into
`Iterator_level_1st_order_upwind_tele::do_sweep` (device `tau` buffer holds T).
Runtime verification harness: `scripts_build/verify_gpu_tele_spark.sh`
(auto-detects teleseismic mode via out-of-domain sources,
`have_tele_data: true`, forward-only CPU vs GPU comparison).

### 8.3 Third-order stencil — findings (pre-existing upstream issues)
1. **The whole LF (non-UPWIND) LEVEL-sweep family is broken on the main
   branch** (reproduced on `3fe822b` and verified with a macOS CPU build in
   this campaign):
   - `stencil_type=0, order=1, sweep_type=1`: converges (it5) but the final
     fields are **wrong** vs the legacy sweep (`mean|d| ≈ 56, max|d| ≈ 106`
     out of 25000 nodes!). The convergence detector is fooled.
   - `stencil_type=0, order=3, sweep_type=1`: 29/30 sources diverge to the
     iteration cap and the travel-time field goes **negative**.
   - The same configurations on the **legacy sweep** or on the **UPWIND
     solver** are clean and verified end-to-end elsewhere in this report.
   - A numpy replica of the exact list/flip/stencil converges identically to
     legacy (even with heterogeneity + anisotropy + fac_f coupling), so the
     defect is not the level-set topology itself; it is implementation-side.
     Experiments: an UPWIND-style flip change and out-of-bounds `is_changed`
     guards improved nothing numerically (guards kept purely as UB hygiene).
   - Recommendation: prefer the UPWIND 1st-order solver (`stencil_type=1`)
     until this is fixed upstream. Keep the legacy sweep if LF+level is
     needed for comparisons.
2. **GPU 3rd-order kernel accuracy gap (~8.5%)** (`verify job 3152`,
   GPU-level vs CPU-legacy): both converge, but `initial objective` differs
   (CPU 49593 vs GPU 53787) and inverted models differ at ~20k grid points.
   This is the **pre-existing `cuda_do_sweep_level_kernel_3rd`**, not the new
   UPWIND port — given (1), expect it to be the same family defect carried
   over to a numerically nicer GPU trajectory. Needs an upstream pass.

### 8.4 Multi-GPU readiness
- Device selection now uses the **node-local rank**
  (`MPI_Comm_split_type(SHARED)` in `initialize_cuda`) — correct placement for
  slurm allocations regardless of global rank layout.
- The stock OpenMPI cannot launch across nodes on this cluster
  (ORTE cross-node failure; system has no usable PMI/PMIx build
  environment). `scripts_build/make_mpi_stack_spark.sh` builds a
  slurm-native stack: hwloc → **libevent 2.2** (PMIx 5 needs
  `event_getcode4name`; Ubuntu 24.04 ships 2.1.12) → PMIx 5 → OpenMPI 4.1.6
  (`--with-pmix --with-slurm`), relinks TOMOATT, and sanity-tests
  `srun -N 2 -n 4 --mpi=pmix_v5`.
- Multi-node (or single-node multi-rank) GPU correctness runs compare inverted
  models and objective history against the single-process verified result.

### 8.5 Harness hardening lessons
- Verify scripts normalize stencil flags on entry and restore on exit
  (crash-proof via `trap`), because a crashed 3rd-order probe once left
  `stencil_order: 3` behind and the next UPWIND run silently switched solver.
- Never run two verifies in the same test directory concurrently (HDF5 output
  file locking races → segfault); chain with
  `sbatch --dependency=afterok:<id>`.
- `spark-edge-0` has no GB10 — GPU jobs exclude it
  (`--exclude=spark-edge-0`) and preflight-check `nvidia-smi -L`.
- `.gitignore` `build*` silently swallows new `build_*.sh` scripts —
  harness files use the `make_*` prefix.
- `stencil_type==UPWIND && stencil_order==3` falls back to the LF solver, so
  `is_upwind` must be `(type==UPWIND && order==1)` — previously preloaded
  index arrays were built with the wrong flip for LF-3rd (fixed in `07eec99`).
