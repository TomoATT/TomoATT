#!/bin/bash
# Test script for reflected wave computation
# Requires: TOMOATT binary, h5py (Python)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TOMOATT_BIN="${TOMOATT_BIN:-../../build_test/bin/TOMOATT}"
NP="${NP:-1}"

echo "=== Reflection Test: PmP/PmS (Moho) ==="

# Step 1: Generate test model
echo "Generating two-layer test model..."
python3 make_test_model.py

# Step 2: Run forward simulation with reflections (PmP, PmS)
echo "Running forward simulation with PmP/PmS reflections..."
mpirun -np "$NP" "$TOMOATT_BIN" -i input_params.yaml

# Step 3: Validate PmP traveltimes against analytical solution
echo ""
echo "Validating results..."
python3 validate_results.py

echo ""
echo "=== Reflection Test: pP/sP (Free Surface) ==="

# Step 4: Generate surface test data
echo "Generating free-surface test data..."
python3 make_surface_test.py

# Step 5: Run free-surface reflection simulation
echo "Running forward simulation with pP/sP reflections..."
mpirun -np "$NP" "$TOMOATT_BIN" -i input_params_surface.yaml

# Step 6: Validate pP/sP traveltimes
echo ""
echo "Validating surface reflection results..."
python3 validate_results.py --surface

echo ""
echo "=== All Reflection Tests Complete ==="
