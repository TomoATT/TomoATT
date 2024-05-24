# tomography and relocation test 

This is an example to relocate earthquake and recover the checkerbaord model using absolute traveltime data.

1. this example use the model files, src_rec_file and input_params.yml in `0_generate_files_for_TomoATT`
    - `0_generate_files_for_TomoATT/2_models/model_init_N61_61_61.h5`
    - `0_generate_files_for_TomoATT/2_models/model_ckb_N61_61_61.h5`
    - `0_generate_files_for_TomoATT/1_src_rec_files/src_rec_config.dat`
    - `0_generate_files_for_TomoATT/3_input_params/0_input_params_forward_simulation.yaml`
    - `0_generate_files_for_TomoATT/3_input_params/3_input_params_inv_loc.yaml`

You can check the distribution of earthquakes (star) and stations (triangle) in `0_generate_files_for_TomoATT/img/src_rec.jpg`

![](../0_generate_files_for_TomoATT/img/src_rec.jpg)

2. run all cells of `1_generate_input_params.ipynb` to generate the necessary input_params files:
    - `input_params/input_params_signal.yaml`
    - `input_params/input_params_inv_abs_reloc_abs.yaml`

2. run TOMOATT forward with `input_params/input_params_signal.yaml` to compute traveltime data in checkerboard model
``` bash
mpirun --oversubscribe -n 2 ../../build/bin/TOMOATT -i input_params/input_params_signal.yaml
```

3. run all cells of `3_generate_obs_src_rec_data.ipynb` for deviate source location and origin time, generating new src_rec data:
    - `src_rec_obs.dat`

4. run TOMOATT forward with `input_params/input_params_inv_abs_reloc_abs.yaml` to do update model parameters and relocate earthquake simultaneously.
``` bash
mpirun --oversubscribe -n 2 ../../build/bin/TOMOATT -i input_params/input_params_reloc_abs.yml
```

5. finally, you can run all cells of `4_compare_location_result.ipynb` to evaluate the location quantitatively

and run all cells of `5_plot_location_result.ipynb` to show the location result in `img`:

![](img/inv_abs_reloc_abs.jpg)

6. you can run all cells of `6_plot_ckb_model.ipynb` to plot the checkerboard model 

![](img/ckb_model_vel.jpg)

![](img/ckb_model_ani.jpg)

and run all cells of `4_plot_inversion_result.ipynb` to plot the inversion result 

![](img/OUTPUT_FILES_inv_abs_reloc_abs_0040_vel.jpg)

![](img/OUTPUT_FILES_inv_abs_reloc_abs_0040_ani.jpg)


You can run `bash 2_run_this_example.sh` to proceed above Steps if `src_rec_obs.dat` in Step 3 has been generated.


