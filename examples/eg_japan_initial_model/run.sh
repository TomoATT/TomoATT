#!/bin/bash
# Generate Japan initial velocity model from ETOPO + CRUST1.0 + AK135
#
# Prerequisites:
#   pip install numpy scipy h5py pyyaml pygmt requests
#
# Outputs:
#   japan_model.h5  – TomoATT velocity model (vel, vel_s, xi, eta)
#   japan_topo.h5   – SurfATT-compatible topography (/lon, /lat, /z in metres)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UTILS_DIR="${SCRIPT_DIR}/../../utils"
AK135="${SCRIPT_DIR}/../scripts_of_generate_community_model/ak135.h5"

python "${UTILS_DIR}/make_model_etopo_crust1.py" \
    --param_file "${SCRIPT_DIR}/input_params.yaml" \
    --output "${SCRIPT_DIR}/japan" \
    --ak135 "${AK135}" \
    --smooth

echo ""
echo "To visualise the model, run:"
echo "  python plot_model.py"
