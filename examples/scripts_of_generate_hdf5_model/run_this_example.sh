#!/bin/bash

# run the script to generate HDF5 model files for TomoATT. Four models will be generated for TomoATT
# 1. constant velocity model
# 2. linear velocity model
# 3. regular checkerboard model based on the linear velocity model
# 4. flexible checkerboard model based on the linear velocity model

python 1_generate_models.py

# run the script to plot the generated models (optional)

python 2_plot_models.py
