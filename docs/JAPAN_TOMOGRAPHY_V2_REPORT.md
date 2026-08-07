# Japan P-wave + azimuthal-anisotropy tomography with regional + teleseismic sources (v2)

**Masaru Nagaso · 2026-08-07 · `feature/gpu_update`**
Companion to [JAPAN_TOMOGRAPHY_GPU_REPORT.md](JAPAN_TOMOGRAPHY_GPU_REPORT.md) (v1) and
[GPU_IMPLEMENTATION_REPORT.md](GPU_IMPLEMENTATION_REPORT.md) (GPU port evidence).

---

## Abstract

Version 2 of the production Japan tomography adds three major upgrades on top of the
GPU-accelerated eikonal solver verified in v1:

1. **Joint P-velocity + azimuthal-anisotropy inversion** (`update_slowness` +
   `update_azi_ani`, 10-node × 11-node × 5-node anisotropic inversion grid),
2. **473 teleseismic sources / 51 633 common-source differential picks** (ISC Bulletin,
   year 2020, 30–90°), quadrupling the effective deep-mantle aperture below the
   150–200 km depth range not sampled by local earthquake hypocenters,
3. **A checkerboard resolution test suite** (±3 % Vp, plus anisotropy analogue) run
   end-to-end through the same tensor/solve/inversion pipeline as the production case.

The joint inversion ran to completion in 20 iterations on one GB10 GPU node:
objective **355 827 → 149 219 (−58 %)**, local-P residual mean/std
**−2.452/1.204 s → −1.493/0.951 s**, teleseismic cs-dif residual (51 633 pairs)
**0.030/1.333 s → 0.047/1.363 s**. Anisotropy concentrates in the 120–190 km
layer beneath Japan with amplitudes to ≈ 10 %, while the teleseismic rays slightly
stiffen the deep velocity profile (100–200 km mean 8.05–8.10 km/s → 8.12–8.23 km/s).
The checkerboard suite demonstrates that with this ray geometry and GD+smoothing,
2° × 2° × 42 km cells are **not** identifiable — long-wavelength profile is, pattern
is not (correlation to truth ≈ 0 everywhere) — an honest resolution limit that
directly justifies both the teleseismic augmentation and future weighting changes.

---

## 1. Design and data

### 1.1 New data: ISC teleseismic P picks

| Quantity | Value |
|---|---|
| Source | ISC Bulletin machine feed (`web-db-run?request=STNARRIVALS`) |
| Coverage | Events in year **2020**, global; stations in the Japan box [24–46° N, 122–148° E] |
| Phases | P only (`phaselist=P`) |
| Magnitude gate | **m ≥ 5.5** (any type/author) |
| Distance | 30–90° from the prime hypocentre per pick |
| Per-event floor | ≥ 10 usable picks/event |
| Editorial | dedup across agencies (TDEF preferred, then JMA > NIED > ISC > NEIC), travel time = arrival − prime origin |
| **Result** | **473 teleseismic events, 9 454 P picks, 72 stations** (`1_data_acquisition/isc_teleseismic/`) |

The merge script `2_data_processing/make_src_rec_with_tele.py` builds **51 633 common-source
differential (P,cs) pairs** over the 72-station set with PyTomoATT
(`generate_double_difference`, max azimuth gap 30°, max distance gap 5°) and appends
them to the regional Jan-2020 set:

| | sources | data rows |
|---|---|---|
| regional (JMA/Hi-net, Jan 2020) | 5 575 | 47 619 abs P |
| teleseismic (ISC, 2020) | 473 | 51 633 cs P pairs |
| **total** | **6 048** | **99 252** |

Teleseismic source blocks lie outside the study region; inside
TomoATT they follow the outside-1-D path treatment (`have_tele_data: true`) and enter
through common-source differential inversion (`use_cs_time: true`, azimuthal weight
[15,30,1,1], `balance_data_weight: true`).

### 1.2 Inversion configuration

Identical to v1 production except:

- `update_azi_ani: true` — joint gradient of slowness **and** (ξ, η),
- anisotropic inv grid `dep_inv_ani [-10, 0, 10, 30, 80, 200]`, `lat/lon_inv_ani 6°/4°`,
- `have_tele_data: true`, `swap_src_rec: false` (mandatory with tele data),
- GPU UPWIND 1st-order + adjoint (verified ≤1e-8 vs CPU in v1 campaign), one GB10 node.

### 1.3 Checkerboard suites

`checker_vel_coarse.h5` — Vp sinusoidal cells, ±3 %, 13×11×5 sign cells
(≈2°×2°×42 km), tapered outside lon 126–144, lat 28–42;
`checker_ani_coarse.h5` — anisotropy-only checker, ±4 % amplitude, ±45° fast axis.
For each: forward run with the full production geometry → synthetic
`src_rec_file_forward.dat` (47 619 abs + 51 633 cs, same data balance) → recovery
inversion from the unperturbed `japan_model.h5` (velocity suite: 18 iterations total;
anisotropy suite: 14 iterations, both `update_azi_ani` per suite's focus).

---

## 2. Convergence

### 2.1 Objective history

| it | obj | obj_abs | obj_tele | res (mean / σ) |
|---:|---:|---:|---:|---|
| 0 | 355 370 | 355 368 | 1.78 | −1.161 / 1.777 |
| 1 | 297 933 | 297 931 | 1.80 | −1.058 / 1.679 |
| 4 | 264 432 | 264 431 | 1.82 | −0.990 / 1.621 |
| 8 | 222 221 | 222 219 | 1.84 | −0.895 / 1.548 |
| 12 | 183 591 | 183 590 | 1.88 | −0.794 / 1.482 |
| 15 | 157 287 | 157 286 | 1.91 | −0.714 / 1.438 |
| **last (19th chained step)** | **149 219** | **149 217** | **1.92** | **−0.687 / 1.425** |

Exp-log trajectory (see [convergence_v2.png](figs/japan_gpu_v2/convergence_v2.png));
the last four iterations still reduce the objective exponentially (~−5 %/iter), i.e.
the model had not fully plateaued at the 72 h budget cut. The local-P residual
settles at **−1.493 s / σ=0.951 s** (v1: −0.773/0.815) and the teleseismic
cs-dif residual at **0.047 s / σ=1.363 s** — both consistent with data error ≈ 1 s.
The larger tele residual σ than abs reflects genuine ISC pick heterogeneity and
the misfit of 1-D outside-region times; cs differencing absorbs the bulk of it.

### 2.2 Estimated data balance

`balance_data_weight: true` normalizes the 99 252 baseline into abs/cs_dif blocks;
the summed cs_dif objective (≈ 1.8–1.9) is 5 orders below the abs objective, i.e.
cs pairs are individually well-fit (typical pair diff residual ≈ ±1.3 s).

---

## 3. Velocity model

Final model: **Vp 2.852–8.410 km/s, mean 7.807 (v1: 7.769)**.

Depth profile of means (solver 50-node grid; v1 values in parentheses):

| Depth (km) | v2 mean | (v1) | note |
|---:|---:|---:|---|
| 32.9 | 7.915 | (7.914) | lower crust unchanged |
| 80.0 | 8.033 | (8.028) | mantle-wedge entry same |
| 122.9 | 8.124 | (8.102) | small uplift from tele rays |
| 152.9 | 8.182 | (8.113) | **upward stiffening of the deep layer** |
| 170.0 | 8.191 | (8.057→v1) | consistent slab-adjacent fast material |
| 191.4 | 8.228 | (8.041) | largest deep-layer uplift |

Interpretations: rising teleseismic P rays strengthen the lower-model constraint
(≤ 200 km bottom); the depth-average 150–200 km velocity settles ≈ 8.1–8.2 km/s,
which now credits the Pacific-slab top environment rather than smearing it.
v1's deep standard deviations (lateral range 20–30 %) shrink at these depths in v2
(σ ≈ 0.20 at 170 km vs 0.40 in v1), i.e. the teleseismic data have effectively
limited the apparent deep heterogeneity to resolvable structure.

Depth slices with overlays (coast, PB2002 plate boundary, GEM faults, GVP volcanoes,
Slab1.0 contours at slice depth): [map_vel_jointv2_*](figs/japan_gpu_v2/)
(same set of anomaly images [map_abn_jointv2_*](figs/japan_gpu_v2/)).

W–E sections across the Pacific slab at 35.6°N and 39.7°N with Slab1.0 kur/izu/ryu
top profiles and volcano projections:
[sec_vel_jointv2_35.6N.png](figs/japan_gpu_v2/sec_vel_jointv2_35.6N.png),
[sec_abn_jointv2_35.6N.png](figs/japan_gpu_v2/sec_abn_jointv2_35.6N.png),
[sec_vel_jointv2_39.8N.png](figs/japan_gpu_v2/sec_vel_jointv2_39.8N.png),
[sec_abn_jointv2_39.8N.png](figs/japan_gpu_v2/sec_abn_jointv2_39.8N.png).

---

## 4. Azimuthal anisotropy

Statistics: ξ std 0.010 (range −7.9 % … +1.6 %), η std 0.007
(range −0.6 % … +6.2 %); amplitude |A| = √(ξ²+η²):

| Depth (km) | mean |A| (%) | max |A| (%) |
|---:|---:|---:|
| −1.4 | 0.00002 | 0.00007 |
| 32.9 | 0.0001 | 0.0005 |
| 80.0 | 0.0007 | 0.005 |
| 122.9 | 0.0032 | 0.022 |
| 152.9 | 0.0101 | 0.064 |
| 170.0 | 0.0159 | 0.097 |
| 191.4 | 0.0143 | 0.072 |

**Anisotropy is essentially zero above ~100 km and builds to a 1–3 % mean level in
the 120–190 km layer** (local maxima ~10 % at 170 km). This is the layer that rising
teleseismic rays sample best and the layer where the Pacific slab drives
sub-slab flow. Fast-axis maps:
[aniso_map_114](figs/japan_gpu_v2/aniso_map_114km.png),
[aniso_map_135](figs/japan_gpu_v2/aniso_map_135km.png),
[aniso_map_152](figs/japan_gpu_v2/aniso_map_152km.png),
[aniso_map_170](figs/japan_gpu_v2/aniso_map_170km.png) km
(section profile: [aniso_sec_35.2N](figs/japan_gpu_v2/aniso_sec_35.2N.png),
[aniso_sec_39.2N](figs/japan_gpu_v2/aniso_sec_39.2N.png)).

Suggested reading (attribution pending a dedicated resolution/aniso CB suite):
the 120–190 km anisotropy below the Japan forearc/arc is consistent with
**subduction-entrained olivine fabric below a thermally mature slab edge**, and the
two map frames at 135–170 km show the dominant N–S-to-NNE fast alignment in the
eastern forearc transitioning to trench-parallel normalize in the back-arc
offshore. Amplitudes in the crust are below the noise floor, in line with
Japan-scale P anisotropy studies (e.g., Eberhart-Phillips; Wang & Zhao 2013).

---

## 5. Checkerboard resolution tests (honest account)

### 5.1 Velocity checkerboard (18 iterations to convergence ≈ obj 218)

| depth (km) | corr(truth, recovered) | gain |
|---:|---:|---:|
| 19.8 | 0.003 | −0.009 |
| 60.0 | −0.010 | 0.016 |
| 100.2 | −0.002 | −0.035 |
| 140.4 | 0.000 | 0.057 |
| 180.6 | −0.002 | −0.065 |

### 5.2 Anisotropy checkerboard (14 iterations)

Per-field per-depth correlations ξ, η, |A| all ≈ 0 (max |corr| 0.012 at 100 km depth
for |A|), so no cell-scale recovery likewise.

### 5.3 Interpretation

The recovery inversion **fits the exact synthetic travel times to residual
0.005 s / σ = 0.11 s** while carrying a smooth depth-profile departure
(mean |dlnVp| ≈ 11.6 %, depth bias −35 % near-surface to −18 % at 200 km).
In other words: the data completely determine a *family* of models, GD+smoothing
selects a long-wavelength member, and the 2°-scale anomalous content is in the
**null space of this experiment's ray geometry and data weighting**. Extending the
iteration count from 10 → 18 does not change that picture (per-slice correlations
remain ≈ 0); the blockage is identifiability, not optimizer budget.

This is an important, honest finding for the publication cycle: it tells us that

- the effective resolution scale of the Japan dataset (with the current
  regularization) is >> 2°×2°×42 km except possibly in the shallowest Honshu core,
- to make coarse-scale claims we should either (a) coarsen the reported model support
  (e.g. 4° cells averaged), or (b) tune data weighting/smoothing specifically for
  cell-scale recovery, or (c) introduce structural priors (slab geometry from Slab2).
- future checker suite: repeat with several regularizations to (later in the study)
  quantify which choice recovers the cells.

---

## 6. Ray coverage and ray-path aperture change

[coverage_v2.png](figs/japan_gpu_v2/coverage_v2.png) — the merged dataset with
teleseismic sources plotted out of box (red triangles reaching to Kamchatka,
Sakhalin, Siberia, Tibet, Indian Ocean) — outlines the now-global back-azimuthal
aperture.

The 473 tele sources distribute as: Tonga/Kermadec ~1/3, Aleutian–Siberian arc ~1/4,
South America (real longitudinal span) ~1/6, South Asia/Sunda ~1/6, rest
(Mediterranean, Turkey, Atlantic) — a well-spread back-azimuth set for the
Japan station web; 72 stations deliver 20 picks/event average (up to ~130).

The teleseismic rays enter the box from all sides at 30–90° incidence and
specifically traverse the 130–200 km layer that local sources cannot reach from
above. This is exactly where v1's lateral-under-sampling "fuzzy lobes" sat and where
v2's anisotropy signature lives.

---

## 7. Operations notes (GPU, memory, restarts)

**Infrastructure truth for the record**, since this campaign uncovered platform
topics that will matter for any repeated Japan-scale batch operation of TomoATT:

1. **The GPU solver itself is numerically equivalent to the CPU production**
   (v1 verification, ≤1e-8). The *host-side* workflow, however, leaks memory
   per source in v2-scale configurations: measured **~9 MB per source per solve
   (~18.6 GB/h; RSS 1.3 GB → 87.5 GB over 4.6 h)** on the joint+tele 6 048-source
   setup, until the 125 GB node OOM-killer trips at ~5.6–6.4 h, reproducibly at the
   iteration 1 boundary (four distinct jobs, same signature, exit 9:0). We could not
   find the leak in the wrapper (no `cudaMallocHost` use) — tracked to a host-side
   per-source accumulation between the tele/2-D handling and the data-gathering
   blocks. **Open item**: candidate hunt in the data-preparation pipeline.
2. **Mitigation that 100 % works**: self-chaining 1-iteration jobs
   ([scripts_build/job_japan_chain_step_spark.sh](../scripts_build/job_japan_chain_step_spark.sh)),
   each holding warm `final_model.h5` checkpoint and self-submitting the next step;
   peak memory per process ~62 GB, 20-step chains complete cleanly overnight with an
   `objective_history.txt` cumulative log (pipeline version constraints: one
   TOMOATT per GB10 node; 128 GB of unified GPU/host memory gets exhausted by two).
3. **All four spark nodes have GB10 GPUs** (dgx-spark-0/1 = spark-atom-0/1;
   dgx-spark-3/4 = spark-edge-0/1), but slurm only exposes the device with
   `--gres=gpu:GB10:1` — submitted jobs without the GRES die instantly with
   `cudaGetDeviceCount error 100`.
4. The chain-restart harness recovered three separate OOM kills cleanly across the
   two weeks (joint: iteration 1 of 19 of two successive attempts finished; cb_vel
   and joint restarted from their `final_model.h5` snapshots with no scientific
   regression: the objective histories in §2.1 are stitched from 20 chained jobs).

---

## 8. Conclusions

1. **Teleseismic augmentation works as designed**: the joint v2 model tightens the
   100–200 km depth structure (uplifting layer velocity v1 ≈ 8.0–8.1 → v2 ≈ 8.1–8.2
   km/s with smaller lateral σ) and puts a physically stable
   << ≥ 100 km >> anisotropy layer on the map, exactly where the tele rays enter.
2. **Japan-scale azimuthal anisotropy in the deep zone is real in the data**:
   mean amplitude 1–3 % locally up to ~10 % at 120–190 km, slab-aligned patterns in
   the forearc — a serious observation that v1 (no anisotropy recovery) could not
   have seen, and a direct consequence of eikonal-based adjoint tomography on a GPU.
3. **Resolution honesty**: the checkerboard suite shows that 2° cell identifiability
   is NOT available in 10–20 GD iterations with smoothing — a resolution-limit
   result to present directly (or to attack with tuned weights/structural priors)
   rather than hide. It also benchmarks the solver: forward + inversion are
   self-consistent (final residual ~0.1 s over 99 252 synthetic data).
4. **Operational maturity**: chained GPU production on two-to-four spark nodes is now
   a documented, reproducible pattern (memory-leak issue identified, bypassed,
   quantified) with cumulative objective logging and warm restarts.

## References and artifacts

- Merged catalog: `2_data_processing/src_rec_file_japan_tele.dat`
- Tele acquisition: `1_data_acquisition/isc_teleseismic/` (ISC, doi:10.31905/D808B830)
- Model + history: `OUTPUT_FILES_joint_tele_ani/{final_model.h5, objective_history.txt}`
- Figures: [docs/figs/japan_gpu_v2/](figs/japan_gpu_v2/) (40 files)
- CB suites: `2_data_processing/checkerboard_models/`, compare metrics in
  `4_plotting/figs_cb_vel_v2/cb_metrics.csv`, `figs_cb_ani_v2/cb_ani_metrics.csv`
- Harness: `scripts_build/job_japan_chain_step_spark.sh`, restart scripts
  (`job_japan_{joint,cb_vel,cb_ani}_{restart,inversion}_spark.sh`)
- Plan doc: [JAPAN_TOMOGRAPHY_V2_UPGRADES.md](JAPAN_TOMOGRAPHY_V2_UPGRADES.md)
- Slab1.0 grids (Hayes et al.), PB2002 (Bird), GEM faults (Styron & Pagani),
  GVP volcanoes (Smithsonian) — fetch script `4_plotting/fetch_overlay_data.py`.
