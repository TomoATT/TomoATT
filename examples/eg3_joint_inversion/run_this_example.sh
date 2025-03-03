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

# Step 4: Do joint inversion
# # for WSL
#     # step 1. relocation for 50 iterations in the initial model, using traveltimes and common-receiver differential arrival times
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_joint_step1.yaml
#     # step 2. simultaneously update model parameters and locations for 40 iterations, 
#     # using traveltimes and common-source differential arrival times for model update
#     # using traveltimes and common-receiver differential arrival times for location
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_joint_step2.yaml
#     # step 3. relocation for 50 iterations in the initial model, using only common-receiver differential arrival times
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_joint_step3.yaml

# # for Linux
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_joint_step1.yaml
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_joint_step2.yaml
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_joint_step3.yaml

# for conda install
mpirun -n 8 TOMOATT -i 3_input_params/input_params_joint_step1.yaml
mpirun -n 8 TOMOATT -i 3_input_params/input_params_joint_step2.yaml
mpirun -n 8 TOMOATT -i 3_input_params/input_params_joint_step3.yaml

# Step 5 (Optional): Plot the results
python plot_output.py