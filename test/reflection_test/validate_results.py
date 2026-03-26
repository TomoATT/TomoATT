"""
Validate reflected-wave traveltimes from TOMOATT against analytical solutions.

Usage:
    python validate_results.py [--surface]

Without --surface: validates PmP traveltimes from the Moho reflection test.
With --surface:    validates pP and sP traveltimes from the free-surface test.
"""

import sys
import os
import math
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available. Will only validate from src_rec output files.")


R_earth = 6371.0


def read_src_rec_output(filename):
    """Read source-receiver output file and return dict of {phase: [(offset_km, t_obs, t_syn)]}."""
    results = {}
    if not os.path.exists(filename):
        print(f"  Output file not found: {filename}")
        return results

    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            # Try to parse receiver line
            try:
                lat = float(parts[3])
                lon = float(parts[4])
                phase = parts[6]
                offset = float(parts[7])
                t_obs = float(parts[8])
                t_syn = float(parts[9]) if len(parts) > 9 else None
            except (ValueError, IndexError):
                continue

            if phase not in results:
                results[phase] = []
            results[phase].append((offset, t_obs, t_syn))

    return results


def analytical_PmP(src_dep, moho_dep, Vp, offset_km):
    """Flat-earth PmP traveltime using image-source for asymmetric geometry.

    Image source at depth (2*moho_dep - src_dep) for a surface receiver.
    T = sqrt((2*z_m - z_s)^2 + x^2) / Vp
    """
    h_img = 2.0 * moho_dep - src_dep
    return math.sqrt(h_img**2 + offset_km**2) / Vp


def analytical_pP(src_dep, Vp, offset_km):
    """Flat-earth pP traveltime for a surface receiver via image-source.

    Image source at depth -src_dep (mirrored across free surface at dep=0).
    For a receiver at the surface (dep=0):
      T_pP = sqrt(src_dep^2 + x^2) / Vp
    Note: this equals the direct P traveltime for surface receivers.
    """
    return math.sqrt(src_dep**2 + offset_km**2) / Vp


def analytical_sP(src_dep, Vs, Vp, offset_km):
    """Flat-earth sP traveltime for a surface receiver.

    S-wave from source to surface, P-wave along surface to receiver.
    T_sP(x) = min over x_r: sqrt(dep^2 + x_r^2)/Vs + |x - x_r|/Vp

    For |x| >= x_crit: T = sqrt(dep^2 + x_crit^2)/Vs + (|x| - x_crit)/Vp
      where x_crit = dep * Vs / sqrt(Vp^2 - Vs^2)
    For |x| < x_crit:  T = sqrt(dep^2 + x^2)/Vs  (= direct S time)
    """
    x_crit = src_dep * Vs / math.sqrt(Vp**2 - Vs**2)
    x = abs(offset_km)
    if x <= x_crit:
        return math.sqrt(src_dep**2 + x**2) / Vs
    else:
        T_crit = math.sqrt(src_dep**2 + x_crit**2) / Vs
        return T_crit + (x - x_crit) / Vp


def validate_moho_test():
    """Validate PmP traveltimes from the Moho reflection test."""
    print("=" * 60)
    print("Validating PmP traveltimes (Moho reflection test)")
    print("=" * 60)

    src_dep = 5.0
    moho_dep = 35.0
    Vp = 6.0
    src_lat = 1.0
    src_lon = 1.0

    # Read the TOMOATT forward output to get computed traveltimes
    forward_file = 'OUTPUT_FILES/src_rec_file_forward.dat'
    if not os.path.exists(forward_file):
        print(f"ERROR: {forward_file} not found. Run TOMOATT first.")
        return False

    print(f"\n{'Receiver':>8s}  {'Offset':>8s}  {'T_analytical':>12s}  {'T_computed':>10s}  {'Residual':>10s}  {'Status':>6s}")
    print("-" * 70)

    n_pass = 0
    n_total = 0
    max_residual = 0.0

    with open(forward_file, 'r') as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue  # skip source line
            parts = line.split()
            if len(parts) < 8:
                continue

            rec_name = parts[2]
            rec_lat = float(parts[3])
            rec_lon = float(parts[4])
            phase = parts[6]
            t_syn = float(parts[7])

            if phase != 'PmP':
                continue

            # Compute horizontal offset
            delta_lon = rec_lon - src_lon
            offset_km = abs(delta_lon * math.pi / 180.0 * R_earth * math.cos(src_lat * math.pi / 180.0))

            t_analytical = analytical_PmP(src_dep, moho_dep, Vp, offset_km)
            residual = t_syn - t_analytical
            max_residual = max(max_residual, abs(residual))
            n_total += 1

            # Allow tolerance for spherical vs flat-earth approximation
            tolerance = 0.5  # seconds — generous for analytical comparison
            status = "PASS" if abs(residual) < tolerance else "FAIL"
            if status == "PASS":
                n_pass += 1

            print(f"{rec_name:>8s}  {offset_km:8.1f}  {t_analytical:12.3f}  {t_syn:10.3f}  {residual:10.3f}  {status:>6s}")

    print(f"\nResults: {n_pass}/{n_total} passed (tolerance={tolerance:.1f}s)")
    print(f"Max residual: {max_residual:.4f} s")

    # Check if TOMOATT output exists
    output_dir = 'OUTPUT_FILES'
    if HAS_H5PY and os.path.exists(output_dir):
        print(f"\nChecking TOMOATT output in {output_dir}/...")
        # Look for HDF5 output files
        for fname in os.listdir(output_dir):
            if fname.endswith('.h5'):
                fpath = os.path.join(output_dir, fname)
                with h5py.File(fpath, 'r') as hf:
                    print(f"  {fname}:")
                    def show_keys(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            print(f"    {name}: shape={obj.shape}, dtype={obj.dtype}")
                    hf.visititems(show_keys)

    return n_pass == n_total


def validate_surface_test():
    """Validate pP and sP traveltimes from the free-surface reflection test."""
    print("=" * 60)
    print("Validating pP/sP traveltimes (free-surface reflection test)")
    print("=" * 60)

    src_dep = 30.0
    src_lat = 1.0
    src_lon = 1.0
    Vp = 6.0
    Vs = 3.5

    # Read the TOMOATT forward output
    forward_file = 'OUTPUT_FILES_SURFACE/src_rec_file_forward.dat'
    if not os.path.exists(forward_file):
        print(f"ERROR: {forward_file} not found. Run TOMOATT first.")
        return False

    for phase_name, analytical_fn in [('pP', lambda x: analytical_pP(src_dep, Vp, x)),
                                       ('sP', lambda x: analytical_sP(src_dep, Vs, Vp, x))]:
        print(f"\n--- Phase: {phase_name} ---")
        print(f"{'Receiver':>8s}  {'Offset':>8s}  {'T_analytical':>12s}  {'T_computed':>10s}  {'Residual':>10s}")
        print("-" * 60)

        with open(forward_file, 'r') as f:
            for line_no, line in enumerate(f):
                if line_no == 0:
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue

                rec_name = parts[2]
                rec_lat = float(parts[3])
                rec_lon = float(parts[4])
                phase = parts[6]
                t_syn = float(parts[7])

                if phase != phase_name:
                    continue

                delta_lon = rec_lon - src_lon
                offset_km = abs(delta_lon * math.pi / 180.0 * R_earth * math.cos(src_lat * math.pi / 180.0))

                t_analytical = analytical_fn(offset_km)
                residual = t_syn - t_analytical
                print(f"{rec_name:>8s}  {offset_km:8.1f}  {t_analytical:12.3f}  {t_syn:10.3f}  {residual:10.3f}")

    return True


if __name__ == '__main__':
    surface_mode = '--surface' in sys.argv

    if surface_mode:
        validate_surface_test()
    else:
        validate_moho_test()
