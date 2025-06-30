#!/bin/bash


# Step 1: Generate necessary input files
echo "Generating TomoATT input files..."
python prepare_input_files.py

# Step 2: Run forward modeling
# # for WSL
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_1dinv_signal.yaml
# # for Linux
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_1dinv_signal.yaml
# for conda install
mpirun -n 8 TOMOATT -i 3_input_params/input_params_1dinv_signal.yaml

# Step 3: Do inversion
# # for WSL
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_1dinv_inv.yaml
# # for Linux
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_1dinv_inv.yaml
# for conda install
mpirun -n 8 TOMOATT -i 3_input_params/input_params_1dinv_inv.yaml

# Step 4 (Optional): Plot the results
echo "Plotting the results..."
python plot_output.py

