# Reflected Wave Test

This test validates the multistage eikonal solver for reflected and mode-converted waves.

## Test 1: PmP and PmS (Moho reflection)

**Model**: Two-layer isotropic model
- Layer 1 (0-35 km): Vp=6.0 km/s, Vs=3.5 km/s
- Layer 2 (35-70 km): Vp=8.0 km/s, Vs=4.5 km/s
- Interface (Moho) at 35 km depth

**Source**: depth=5 km, lat=1.0°, lon=1.0°

**Receivers**: 20 surface stations along longitude 0.2°-1.8°

**Phases**:
- `PmP`: P-wave reflected off Moho as P-wave
- `PmS`: P-wave reflected off Moho as S-wave

**Validation**: Compare computed PmP traveltime against analytical formula:
```
T_PmP = 2 * sqrt(h² + (x/2)²) / Vp  (flat-layer approximation)
```
where h = Moho depth - source depth, x = horizontal offset.

## Test 2: pP and sP (Free-surface reflection)

**Phases**:
- `pP`: upgoing P from source reflected at free surface as P
- `sP`: upgoing S from source reflected at free surface as P

**Validation**: pP traveltime should match image-source analytical solution.

## Running

```bash
bash run_test.sh
```

## Output

- `OUTPUT_FILES/`: Contains HDF5 with direct and reflected traveltime fields
  - `src_*/inv_iteration_0/T`: Direct P-wave traveltime
  - `src_*/phase_PmP/inv_iteration_0/T`: PmP reflected traveltime
  - `src_*/phase_PmS/inv_iteration_0/T`: PmS reflected traveltime
