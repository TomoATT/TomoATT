# Japan tomography v2 upgrades: joint P + azimuthal anisotropy, teleseismic augmentation, checkerboard tests, geological overlays

Status doc for the upgrade cycle requested 2026-08-02 (user): the four
work items below build on the v1 production run (JAPAN_TOMOGRAPHY_GPU_REPORT.md,
branch `feature/gpu_update`, GB10 `spark` node on RIKEN TENJO).

---

## 1. Joint P-velocity + azimuthal-anisotropy inversion

- `update_azi_ani: true` (with `update_slowness: true`) inverts for ξ, η
  alongside slowness ([input_params.h](../include/input_params.h)).
- Anisotropic inversion grid was already configured in v1
  (`invgrid_ani: true`, `dep_inv_ani [-10,0,10,30,80,200]`, 6×6×5 ani nodes).
- Initial model `japan_model.h5` carries `xi`, `eta` (and we stage a zero-`zeta`
  copy when building checkerboards).
- GPU kernels propagate xi/eta through forward and adjoint; sensitivity
  kernels (Ks, Kxi, Keta) are assembled CPU-side from GPU fields (verified in
  the GPU campaign), so no CUDA change is required.
- Production config: `3_input_params/input_params_joint_tele_ani.yml`.

## 2. Teleseismic source augmentation (deep coverage 150-200 km)

- New downloader `1_data_acquisition/download_teleseismic_isc.py` pulls a full
  year (2020) of ISC-bulletin P arrivals at Japanese stations
  (machine feed `web-db-run?request=STNARRIVALS`, mb ≥ 5.5, 30-90° window,
  ≥10 usable picks/event) → **473 teleseismic events / 9 454 P picks / 72 stations**
  (`1_data_acquisition/isc_teleseismic/`).
- Merger `2_data_processing/make_src_rec_with_tele.py` builds common-source
  differential pairs per tele event with PyTomoATT
  (`generate_double_difference`, max_azi_gap 30° / max_dist_gap 5°) and
  appends them to the Jan-2020 regional set →
  `src_rec_file_japan_tele.dat` (**5 575 local + 473 tele = 6 048 sources**,
  47 619 abs lines + **51 633 cs pair lines**, 14-token `P,cs` records in the
  official TomoATT differential format — sources outside the study region are
  treated as teleseismic with 1-D iasp91 outside-traveltime).
- Verified locally (debug-instrumented forward run, 2026-08-02): full parse,
  tele 2-D boundary solve, and synthesis of all `P,cs` lines in
  `src_rec_file_forward.dat`.
- Data-format lessons:
  - receiver lines are `<src_id> <rec_id> ...` — column 0 is the SOURCE index
    (PyTomoATT's pairing key; the C++ parser ignores it);
  - when slicing blocks out of a catalog, do not key on 14-token line count —
    cs pair lines share it with source headers; detect headers by the
    `ev_`/`tel_` name token. A header with n_data declared but no data lines
    crashes the parser with `bad_alloc`.
- Inversion uses `have_tele_data: true` + `cs_dif_time: true` for the tele
  blocks (kills origin-time/mislocation and the outside-path errors),
  `abs_time` for regional events, `balance_data_weight: true` — schema copied
  from `examples/eg3_joint_inversion/.../input_params_joint_step2.yaml`.
- `swap_src_rec` must remain **false** whenever tele data are present
  (swapping would promote stations to sources and teleport sources outside;
  the v1 GPU run used swap=true only because it had no tele data).
  Consequence: 6 048 sequential GPU solves per inversion iteration
  (~5-6× more than the v1 swapped run) → v2 production run ≈ 2.5-3 days on one
  GB10 node. Job script requests `--time=72:00:00`.

## 3. Checkerboard resolution tests

- Generator `2_data_processing/make_checkerboard_models.py`
  (PyTomoATT `Checker`, sinusoidal cells):
  - `checker_vel_coarse.h5`: ±3 % Vp, 13×11×5 cells (≈2°×2°×42 km), tapered to
    lon 126-144 E / lat 28-42 N;
  - `checker_ani_coarse.h5`: ±4 % anisotropy amplitude, fast directions ±45°;
  - fine suite: `--n-cells 26 22 10 --tag fine` (≈1°×1°×21 km).
- Loop: forward run with checker model (`input_params_cb_forward_*.yml`,
  `run_mode: 0`, merged src-rec file) → synthetic `src_rec_file_forward.dat`;
  inversion from unperturbed `japan_model.h5` (`input_params_cb_inversion_*.yml`,
  10 iterations; velocity-only; anisotropy suite with `update_azi_ani: true`);
  comparison `4_plotting/compare_checkerboard.py` → per-depth correlation and
  gain (least-squares amplitude recovery), three-panel maps and W-E sections.
  Validated on a synthetic identity case (corr = 1.000).

## 4. Geological overlays on slice/section figures

- `4_plotting/fetch_overlay_data.py` downloads, into `overlay_data/` (not in git):
  - PB2002 global plate boundaries (fraxen/tectonicplates digitisation of Bird 2003);
  - GEM Global Active Faults harmonised GeoJSON (Styron & Pagani 2020);
  - Smithsonian GVP Holocene volcano catalogue (TidyTuesday 2020-05-12 mirror;
    106 volcanoes in the study box);
  - USGS Slab1.0 clipped slab-top grids for the Japan system (kur = Pacific slab
    under NE Japan, izu = Izu-Bonin, ryu = Philippine Sea / Ryukyu + Nankai;
    the standalone phi grid only covers Luzon-Taiwan and is unused).
- `4_plotting/overlay_common.py`: loaders + `draw_overlays(ax, dep_km=...)`,
  coastline via cartopy Natural Earth (10 m), slab iso-depth contours per slice
  depth, faults, PB2002 boundaries, volcano markers.
- `4_plotting/plot_slices_overlays.py`: `--mode slices` (vel + dlnVp maps) and
  `--mode section` (W-E sections with slab-top profiles and volcano projections).
  Tested on the checkerboard truth model (slices 19/100/159 km + 35.6 N section).
- GVP/ScienceBase direct endpoints are blocked from some networks; GitHub
  mirrors were used (documented in the fetch script).

## Run order on TENJO (spark, 1-node jobs)

```bash
# after: rsync of the files (see below) to nimbus-0:TomoATT/
sbatch scripts_build/job_japan_cb_vel_spark.sh       # ≈ hours  (forward ~1 h + 10 it)
sbatch scripts_build/job_japan_cb_ani_spark.sh       # ≈ hours  (id., ani recovery)
sbatch scripts_build/job_japan_joint_tele_ani_spark.sh  # ≈ 2.5-3 days (20 it)
```

rsync package (from repo root): the same file list as the git commit for this
upgrade — tele CSVs, merged src_rec, checkerboard models (via rsync, not git),
run YAMLs, job scripts.

Post-processing on the Mac (after rsync back of `OUTPUT_FILES_*/out_data_grid.h5`):

```bash
python 4_plotting/compare_checkerboard.py --truth checker_vel_coarse.h5 \
    --base japan_model.h5 --recovered OUTPUT_FILES_cb_inversion_vel/out_data_grid.h5
python 4_plotting/plot_slices_overlays.py \
    --model OUTPUT_FILES_joint_tele_ani/out_data_grid.h5 --mode slices
python 4_plotting/plot_slices_overlays.py \
    --model OUTPUT_FILES_joint_tele_ani/out_data_grid.h5 --mode section
```

## Known ops blocker (2026-08-02)

`nimbus-0.r-ccs10.riken.jp` does not resolve via public DNS from the current
network — the rsync and job submission are staged but pending connectivity
(VPN route needed). All assets are committed on `feature/gpu_update`; pending
only the cluster-side execution.

## References

- ISC (2024). On-line Bulletin. https://doi.org/10.31905/D808B830
- Hayes, G. P., et al. (2012). Slab1.0. https://doi.org/10.5066/F7PV6JNV
- Bird, P. (2003). An updated digital model of plate boundaries. G-cubed 4(3).
- Styron, R., Pagani, M. (2020). The GEM Global Active Faults Database. EQ Spectra.
- Smithsonian Institution, Global Volcanism Program. Volcanoes of the World.
- Chen, J., et al. (2023). Adjoint-state teleseismic traveltime tomography
  (Thailand case in this repo). JGR 128.
