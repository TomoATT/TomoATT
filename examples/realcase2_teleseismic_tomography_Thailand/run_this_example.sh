#!/bin/bash

# Step 1: Generate necessary input files
python prepare_input_files.py

# Step 2: Run inversion
# # for WSL
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_real.yaml
# for Linux
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_real.yaml
# for conda install
mpirun -n 8 TOMOATT -i 3_input_params/input_params_real.yaml

# # Step 3 (Optional): Plot the results
# python plot_output.py

