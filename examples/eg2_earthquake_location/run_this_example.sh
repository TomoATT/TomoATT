#!/bin/bash


# Step 1: Generate necessary input files
python prepare_input_files.py

# Step 2: Run forward modeling
# # for WSL
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_signal.yaml
# # for Linux
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_signal.yaml
# for conda install
mpirun -n 8 TOMOATT -i 3_input_params/input_params_signal.yaml

# Step 3: Assign data noise and location perturbation to the observational data
python assign_gaussian_noise.py

# Step 4: Do relocation
# # for WSL
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_loc.yaml
# # for Linux
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_loc.yaml
# for conda install
mpirun -n 8 TOMOATT -i 3_input_params/input_params_loc.yaml

# Step 5 (Optional): Plot the results
# python plot_output.py