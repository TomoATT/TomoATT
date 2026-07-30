#!/bin/bash

# Step 1: Generate necessary input files
cd ./Script_data_filtering
python prepare_input_files.py    # download original traveltime data file horizontally averaged crust1.0 1D model (both in TomoATT format)
cd ./input_data
tar -xf src_rec_Turkey.tar.gz
cd ..

# Step 2: Data filtering (four sub-steps)
# a) select data in the study region, and rotate the physical coordinates of sources and receivers to a local coordinate ([-2.5, 3.5] in latitude and [-1.5, 1.5] in longitude)
python step1_select_data_in_region_with_rotation.py     
# b) using linear regression to retain reliable data
python step2_retain_reliable_traveltime.py
# c) delete events with records less than a threshold
python step3_limit_min_Ndata.py
# d) generate common source differential arrival time from absolute travel time 
python step4_generate_differential_traveltime.py

cd ..

# Step 3: Run inversion
# # for WSL
# # Do 1-D inversion to obtain a better initial model using absolute travel time
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_step1_1D_inv.yaml
# # Relocate the events using the 1D inverted model using absolute travel time
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_step2_reloc.yaml
# # Perform 3D tomographic inversion to update earthquakes and model parameters simultaneously
# mpirun -n 8 --allow-run-as-root --oversubscribe ../../build/bin/TOMOATT -i 3_input_params/input_params_step3_inv_reloc.yaml

# # for Linux
# # Do 1-D inversion to obtain a better initial model using absolute travel time
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_step1_1D_inv.yaml
# # Relocate the events using the 1D inverted model using absolute travel time
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_step2_reloc.yaml
# # Perform 3D tomographic inversion to update earthquakes and model parameters simultaneously
# mpirun -n 8 ../../build/bin/TOMOATT -i 3_input_params/input_params_step3_inv_reloc.yaml

# for conda install
# Do 1-D inversion to obtain a better initial model using absolute travel time
mpirun -n 8 TOMOATT -i 3_input_params/input_params_step1_1D_inv.yaml
# Relocate the events using the 1D inverted model using absolute travel time
mpirun -n 8 TOMOATT -i 3_input_params/input_params_step2_reloc.yaml
# Perform 3D tomographic inversion to update earthquakes and model parameters simultaneously
mpirun -n 8 TOMOATT -i 3_input_params/input_params_step3_inv_reloc.yaml

# Step 4 (Optional): Plot the results
python plot_output.py

