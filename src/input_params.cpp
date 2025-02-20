#include "input_params.h"
#include <chrono>
#include <ctime>

// Base case of the variadic template function
template <typename T>
void getNodeValue(const YAML::Node& node, const std::string& key, T& value) {
    if (node[key]) {
        try {
            value = node[key].as<T>();
        } catch (const YAML::Exception& e) {
            std::cout << "Error parsing YAML value for key: " << key << ". " << e.what() << std::endl;
        }
    } else {
        std::cout << "Key not found in YAML: " << key << std::endl;
    }
}

// function overload for reading array
template <typename T>
void getNodeValue(const YAML::Node& node, const std::string& key, T& value, const int index) {
    if (node[key]) {
        try {
            value = node[key][index].as<T>();
        } catch (const YAML::Exception& e) {
            std::cout << "Error parsing YAML value for key: " << key << ". " << e.what() << std::endl;
        }
    } else {
        std::cout << "Key not found in YAML: " << key << std::endl;
    }
}


InputParams::InputParams(std::string& input_file){

    if (world_rank == 0) {
        // parse input files
        YAML::Node config = YAML::LoadFile(input_file);

        //
        // domain
        //
        if (config["domain"]) {
            // minimum and maximum depth
            getNodeValue(config["domain"], "min_max_dep", min_dep, 0);
            getNodeValue(config["domain"], "min_max_dep", max_dep, 1);
            // minimum and maximum latitude
            if (config["domain"]["min_max_lat"]) {
                getNodeValue(config["domain"], "min_max_lat", min_lat, 0);
                getNodeValue(config["domain"], "min_max_lat", max_lat, 1);
            }
            // minimum and maximum longitude
            if (config["domain"]["min_max_lon"]) {
                getNodeValue(config["domain"], "min_max_lon", min_lon, 0);
                getNodeValue(config["domain"], "min_max_lon", max_lon, 1);
            }
            // number of grid nodes on each axis r(dep), t(lat), p(lon)
            if (config["domain"]["n_rtp"]) {
                getNodeValue(config["domain"], "n_rtp", ngrid_k, 0);
                getNodeValue(config["domain"], "n_rtp", ngrid_j, 1);
                getNodeValue(config["domain"], "n_rtp", ngrid_i, 2);
            }
        } else {
            std::cout << "domain is not defined. stop." << std::endl;
            exit(1);
        }

        //
        // source
        //
        if (config["source"]) {
            // source depth(km) lat lon
            if (config["source"]["src_dep_lat_lon"]) {
                getNodeValue(config["source"], "src_dep_lat_lon", src_dep, 0);
                getNodeValue(config["source"], "src_dep_lat_lon", src_lat, 1);
                getNodeValue(config["source"], "src_dep_lat_lon", src_lon, 2);
            }
            // src rec file
            if (config["source"]["src_rec_file"]) {
                src_rec_file_exist = true;
                getNodeValue(config["source"], "src_rec_file", src_rec_file);
            }
            // swap src rec
            if (config["source"]["swap_src_rec"]) {
                getNodeValue(config["source"], "swap_src_rec", swap_src_rec);
            }
        } else {
            std::cout << "source is not defined. stop." << std::endl;
            exit(1);
        }

        //
        // model
        //
        if (config["model"]) {
            // model file path
            if (config["model"]["init_model_path"]) {
                getNodeValue(config["model"], "init_model_path", init_model_path);
            }
            // model file path
            if (config["model"]["model_1d_name"]) {
                getNodeValue(config["model"], "model_1d_name", model_1d_name);
            }
        } else {
            std::cout << "model is not defined. stop." << std::endl;
            exit(1);
        }

        //
        // parallel
        //
        if (config["parallel"]) {
            // number of simultaneous runs
            if(config["parallel"]["n_sims"]) {
                getNodeValue(config["parallel"], "n_sims", n_sims);
            }
            // number of subdomains
            if (config["parallel"]["ndiv_rtp"]) {
                getNodeValue(config["parallel"], "ndiv_rtp", ndiv_k, 0);
                getNodeValue(config["parallel"], "ndiv_rtp", ndiv_j, 1);
                getNodeValue(config["parallel"], "ndiv_rtp", ndiv_i, 2);
            }
            // number of processes in each subdomain
            if (config["parallel"]["nproc_sub"]) {
                getNodeValue(config["parallel"], "nproc_sub", n_subprocs);
            }
            // gpu flag
            if (config["parallel"]["use_gpu"]) {
                getNodeValue(config["parallel"], "use_gpu", use_gpu);
            }
        }

        //
        // output setting
        //
        if (config["output_setting"]) {
            // output path
            if (config["output_setting"]["output_dir"])
                getNodeValue(config["output_setting"], "output_dir", output_dir);

            if (config["output_setting"]["output_source_field"])
                getNodeValue(config["output_setting"], "output_source_field", output_source_field);

            if (config["output_setting"]["output_kernel"])
                getNodeValue(config["output_setting"], "output_kernel", output_kernel);

            if (config["output_setting"]["verbose_output_level"]){
                getNodeValue(config["output_setting"], "verbose_output_level", verbose_output_level);
                // currently only 0 or 1 is defined
                if (verbose_output_level > 1) {
                    std::cout << "undefined verbose_output_level. stop." << std::endl;
                    //MPI_Finalize();
                    exit(1);
                }
            }

            if (config["output_setting"]["output_final_model"])
                getNodeValue(config["output_setting"], "output_final_model", output_final_model);

            if (config["output_setting"]["output_middle_model"])
                getNodeValue(config["output_setting"], "output_middle_model", output_middle_model);

            if (config["output_setting"]["output_in_process"])
                getNodeValue(config["output_setting"], "output_in_process", output_in_process);

            if (config["output_setting"]["output_in_process_data"])
                getNodeValue(config["output_setting"], "output_in_process_data", output_in_process_data);

            if (config["output_setting"]["single_precision_output"])
                getNodeValue(config["output_setting"], "single_precision_output", single_precision_output);

            // output file format
            if (config["output_setting"]["output_file_format"]) {
                int ff_flag;
                getNodeValue(config["output_setting"], "output_file_format", ff_flag);
                if (ff_flag == 0){
                    #if USE_HDF5
                        output_format = OUTPUT_FORMAT_HDF5;
                    #else
                        std::cout << "output_file_format is 0, but the code is compiled without HDF5. stop." << std::endl;
                        //MPI_Finalize();
                        exit(1);
                    #endif
                } else if (ff_flag == 1){
                    output_format = OUTPUT_FORMAT_ASCII;
                } else {
                    std::cout << "undefined output_file_format. stop." << std::endl;
                    //MPI_Finalize();
                    exit(1);
                }
            }
        }

        //
        // run mode
        //
        if (config["run_mode"]) {
            getNodeValue(config, "run_mode", run_mode);
            if (run_mode > 4) {
                std::cout << "undefined run_mode. stop." << std::endl;
                //MPI_Finalize();
                exit(1);
            }
        }

        if (config["have_tele_data"]) {
            getNodeValue(config, "have_tele_data", have_tele_data);
        }

        //
        // model update
        //
        if (config["model_update"]) {
            // number of max iteration for inversion
            if (config["model_update"]["max_iterations"]) {
                getNodeValue(config["model_update"], "max_iterations", max_iter_inv);
            }
            // optim_method
            if (config["model_update"]["optim_method"]) {
                getNodeValue(config["model_update"], "optim_method", optim_method);
                if (optim_method > 2) {
                    std::cout << "undefined optim_method. stop." << std::endl;
                    //MPI_Finalize();
                    exit(1);
                }
            }
            // step_length
            if (config["model_update"]["step_length"]) {
                getNodeValue(config["model_update"], "step_length", step_length_init);
            }

            // parameters for optim_method == 0
            if (optim_method == 0) {
                // step method
                if (config["model_update"]["optim_method_0"]["step_method"]) {
                    getNodeValue(config["model_update"]["optim_method_0"], "step_method", step_method);
                }
                // step length decay
                if (config["model_update"]["optim_method_0"]["step_length_decay"]) {
                    getNodeValue(config["model_update"]["optim_method_0"], "step_length_decay", step_length_decay);
                }
                if (config["model_update"]["optim_method_0"]["step_length_gradient_angle"]) {
                    getNodeValue(config["model_update"]["optim_method_0"], "step_length_gradient_angle", step_length_gradient_angle);
                }
                if (config["model_update"]["optim_method_0"]["step_length_change"]) {
                    getNodeValue(config["model_update"]["optim_method_0"], "step_length_change", step_length_down,0);
                    getNodeValue(config["model_update"]["optim_method_0"], "step_length_change", step_length_up,1);
                }
                if (config["model_update"]["optim_method_0"]["Kdensity_coe"]) {
                    getNodeValue(config["model_update"]["optim_method_0"], "Kdensity_coe", Kdensity_coe);
                    if (Kdensity_coe < 0.0){
                        Kdensity_coe = 0.0;
                        std::cout << std::endl;
                        std::cout << "Kdensity_coe: " << Kdensity_coe << " is out of range, which is set to be 0.0 in the inversion." << std::endl;
                        std::cout << std::endl;
                    }
                    if (Kdensity_coe > 0.95){
                        Kdensity_coe = 0.95;
                        std::cout << std::endl;
                        std::cout << "Kdensity_coe: " << Kdensity_coe << " is out of range, which is set to be 0.95 in the inversion." << std::endl;
                        std::cout << std::endl;
                    }
                }
            }

            // parameters for optim_method == 1 or 2
            if (optim_method == 1 || optim_method == 2) {
                // max_sub_iterations
                if (config["model_update"]["optim_method_1_2"]["max_sub_iterations"]) {
                    getNodeValue(config["model_update"]["optim_method_1_2"], "max_sub_iterations", max_sub_iterations);
                }
                // regularization weight
                if (config["model_update"]["optim_method_1_2"]["regularization_weight"]) {
                    getNodeValue(config["model_update"]["optim_method_1_2"], "regularization_weight", regularization_weight);
                }
                // regularization laplacian weights
                if (config["model_update"]["optim_method_1_2"]["coefs_regulalization_rtp"]) {
                    getNodeValue(config["model_update"]["optim_method_1_2"], "coefs_regulalization_rtp", regul_lr, 0);
                    getNodeValue(config["model_update"]["optim_method_1_2"], "coefs_regulalization_rtp", regul_lt, 1);
                    getNodeValue(config["model_update"]["optim_method_1_2"], "coefs_regulalization_rtp", regul_lp, 2);

                    // convert degree to radian
                    regul_lt = regul_lt * DEG2RAD;
                    regul_lp = regul_lp * DEG2RAD;
                }
            }

            // smoothing
            if (config["model_update"]["smoothing"]["smooth_method"]) {
                getNodeValue(config["model_update"]["smoothing"], "smooth_method", smooth_method);
                if (smooth_method > 1) {
                    std::cout << "undefined smooth_method. stop." << std::endl;
                    MPI_Finalize();
                    exit(1);
                }
            }
            // l_smooth_rtp
            if (config["model_update"]["smoothing"]["l_smooth_rtp"]) {
                getNodeValue(config["model_update"]["smoothing"], "l_smooth_rtp", smooth_lr, 0);
                getNodeValue(config["model_update"]["smoothing"], "l_smooth_rtp", smooth_lt, 1);
                getNodeValue(config["model_update"]["smoothing"], "l_smooth_rtp", smooth_lp, 2);

                // convert degree to radian
                smooth_lt = smooth_lt * DEG2RAD;
                smooth_lp = smooth_lp * DEG2RAD;
            }
            // n_inversion_grid
            if (config["model_update"]["n_inversion_grid"]) {
                getNodeValue(config["model_update"], "n_inversion_grid", n_inversion_grid);
            }

            // auto inversion grid
            if (config["model_update"]["uniform_inv_grid_dep"]) {
                getNodeValue(config["model_update"], "uniform_inv_grid_dep", uniform_inv_grid_dep);
            }
            if (config["model_update"]["uniform_inv_grid_lat"]) {
                getNodeValue(config["model_update"], "uniform_inv_grid_lat", uniform_inv_grid_lat);
            }
            if (config["model_update"]["uniform_inv_grid_lon"]) {
                getNodeValue(config["model_update"], "uniform_inv_grid_lon", uniform_inv_grid_lon);
            }

            // number of inversion grid for regular grid
            if (config["model_update"]["n_inv_dep_lat_lon"]) {
                getNodeValue(config["model_update"], "n_inv_dep_lat_lon", n_inv_r, 0);
                getNodeValue(config["model_update"], "n_inv_dep_lat_lon", n_inv_t, 1);
                getNodeValue(config["model_update"], "n_inv_dep_lat_lon", n_inv_p, 2);
            }

                // inversion grid positions
            if (config["model_update"]["min_max_dep_inv"]) {
                getNodeValue(config["model_update"], "min_max_dep_inv", min_dep_inv, 0);
                getNodeValue(config["model_update"], "min_max_dep_inv", max_dep_inv, 1);
            }
            // minimum and maximum latitude
            if (config["model_update"]["min_max_lat_inv"]) {
                getNodeValue(config["model_update"], "min_max_lat_inv", min_lat_inv, 0);
                getNodeValue(config["model_update"], "min_max_lat_inv", max_lat_inv, 1);
            }
            // minimum and maximum longitude
            if (config["model_update"]["min_max_lon_inv"]) {
                getNodeValue(config["model_update"], "min_max_lon_inv", min_lon_inv, 0);
                getNodeValue(config["model_update"], "min_max_lon_inv", max_lon_inv, 1);
            }


            // flexible inversion grid for velocity
            if (config["model_update"]["dep_inv"]){
                n_inv_r_flex = config["model_update"]["dep_inv"].size();
                dep_inv = allocateMemory<CUSTOMREAL>(n_inv_r_flex, 5000);
                for (int i = 0; i < n_inv_r_flex; i++){
                    getNodeValue(config["model_update"], "dep_inv", dep_inv[i], i);
                }
            } else {
                if (!uniform_inv_grid_dep){
                    std::cout << "dep_inv is not defined. stop." << std::endl;
                    exit(1);
                }
            }
            if (config["model_update"]["lat_inv"]) {
                n_inv_t_flex = config["model_update"]["lat_inv"].size();
                lat_inv = allocateMemory<CUSTOMREAL>(n_inv_t_flex, 5001);
                for (int i = 0; i < n_inv_t_flex; i++){
                    getNodeValue(config["model_update"], "lat_inv", lat_inv[i], i);
                }
            } else {
                if (!uniform_inv_grid_lat){
                    std::cout << "lat_inv is not defined. stop." << std::endl;
                    exit(1);
                }
            }
            if (config["model_update"]["lon_inv"]) {
                n_inv_p_flex = config["model_update"]["lon_inv"].size();
                lon_inv = allocateMemory<CUSTOMREAL>(n_inv_p_flex, 5002);
                for (int i = 0; i < n_inv_p_flex; i++){
                    getNodeValue(config["model_update"], "lon_inv", lon_inv[i], i);
                }
            } else {
                if (!uniform_inv_grid_lon){
                    std::cout << "lon_inv is not defined. stop." << std::endl;
                    exit(1);
                }
            }
            if (config["model_update"]["trapezoid"]) {
                for (int i = 0; i < n_trapezoid; i++){
                    getNodeValue(config["model_update"], "trapezoid", trapezoid[i], i);
                }
            }

            // inversion grid for anisotropy
            if (config["model_update"]["invgrid_ani"]) {
                getNodeValue(config["model_update"], "invgrid_ani", invgrid_ani);
            }

            if (config["model_update"]["n_inv_dep_lat_lon_ani"]) {
                getNodeValue(config["model_update"], "n_inv_dep_lat_lon_ani", n_inv_r_ani, 0);
                getNodeValue(config["model_update"], "n_inv_dep_lat_lon_ani", n_inv_t_ani, 1);
                getNodeValue(config["model_update"], "n_inv_dep_lat_lon_ani", n_inv_p_ani, 2);
            }

            if (config["model_update"]["min_max_dep_inv_ani"]) {
                getNodeValue(config["model_update"], "min_max_dep_inv_ani", min_dep_inv_ani, 0);
                getNodeValue(config["model_update"], "min_max_dep_inv_ani", max_dep_inv_ani, 1);
            }

            if (config["model_update"]["min_max_lat_inv_ani"]) {
                getNodeValue(config["model_update"], "min_max_lat_inv_ani", min_lat_inv_ani, 0);
                getNodeValue(config["model_update"], "min_max_lat_inv_ani", max_lat_inv_ani, 1);
            }

            if (config["model_update"]["min_max_lon_inv_ani"]) {
                getNodeValue(config["model_update"], "min_max_lon_inv_ani", min_lon_inv_ani, 0);
                getNodeValue(config["model_update"], "min_max_lon_inv_ani", max_lon_inv_ani, 1);
            }

            // flexible inversion grid for anisotropy
            if (config["model_update"]["dep_inv_ani"]) {
                n_inv_r_flex_ani = config["model_update"]["dep_inv_ani"].size();
                dep_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_r_flex_ani, 5003);
                for (int i = 0; i < n_inv_r_flex_ani; i++){
                    getNodeValue(config["model_update"], "dep_inv_ani", dep_inv_ani[i], i);
                }
            }

            if (config["model_update"]["lat_inv_ani"]) {
                n_inv_t_flex_ani = config["model_update"]["lat_inv_ani"].size();
                lat_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_t_flex_ani, 5004);
                for (int i = 0; i < n_inv_t_flex_ani; i++){
                    getNodeValue(config["model_update"], "lat_inv_ani", lat_inv_ani[i], i);
                }
            }
            if (config["model_update"]["lon_inv_ani"]) {
                n_inv_p_flex_ani = config["model_update"]["lon_inv_ani"].size();
                lon_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_p_flex_ani, 5005);
                for (int i = 0; i < n_inv_p_flex_ani; i++){
                    getNodeValue(config["model_update"], "lon_inv_ani", lon_inv_ani[i], i);
                }
            }
            if (config["model_update"]["trapezoid_ani"]) {
                for (int i = 0; i < n_trapezoid; i++){
                    getNodeValue(config["model_update"], "trapezoid_ani", trapezoid_ani[i], i);
                }
            }

            setup_uniform_inv_grid();

            // if (config["model_update"]["invgrid_volume_rescale"]) {
            //     getNodeValue(config["model_update"], "invgrid_volume_rescale", invgrid_volume_rescale);
            // }

            // station correction (now only for teleseismic data)
            if (config["model_update"]["use_sta_correction"]){
                getNodeValue(config["model_update"], "use_sta_correction", use_sta_correction);
            }
            // sta_correction_file
            if (config["model_update"]["initial_sta_correction_file"]) {
                getNodeValue(config["model_update"], "initial_sta_correction_file", sta_correction_file);
                if (use_sta_correction) {
                    sta_correction_file_exist = true;
                }
            }

            // date usage flags and weights
            // absolute travel times
            if (config["model_update"]["abs_time"]) {
                getNodeValue(config["model_update"]["abs_time"], "use_abs_time", use_abs);
                if (config["model_update"]["abs_time"]["residual_weight"]){
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["model_update"]["abs_time"], "residual_weight", residual_weight_abs[i], i);
                }
                if (config["model_update"]["abs_time"]["distance_weight"]) {
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["model_update"]["abs_time"], "distance_weight", distance_weight_abs[i], i);
                }
            }

            // common source diff travel times
            if (config["model_update"]["cs_dif_time"]) {
                getNodeValue(config["model_update"]["cs_dif_time"], "use_cs_time", use_cs);
                if (config["model_update"]["cs_dif_time"]["residual_weight"]){
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["model_update"]["cs_dif_time"], "residual_weight", residual_weight_cs[i], i);
                }
                if (config["model_update"]["cs_dif_time"]["azimuthal_weight"]) {
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["model_update"]["cs_dif_time"], "azimuthal_weight", azimuthal_weight_cs[i], i);
                }
            }

            // common reciever diff travel times
            if (config["model_update"]["cr_dif_time"]) {
                getNodeValue(config["model_update"]["cr_dif_time"], "use_cr_time", use_cr);
                if (config["model_update"]["cr_dif_time"]["residual_weight"]){
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["model_update"]["cr_dif_time"], "residual_weight", residual_weight_cr[i], i);
                }
                if (config["model_update"]["cr_dif_time"]["azimuthal_weight"]) {
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["model_update"]["cr_dif_time"], "azimuthal_weight", azimuthal_weight_cr[i], i);
                }
            }

            // weight of different types of data
            if (config["model_update"]["global_weight"]){
                if (config["model_update"]["global_weight"]["balance_data_weight"]) {
                    getNodeValue(config["model_update"]["global_weight"], "balance_data_weight", balance_data_weight);
                }
                if (config["model_update"]["global_weight"]["abs_time_weight"]) {
                    getNodeValue(config["model_update"]["global_weight"], "abs_time_weight", abs_time_local_weight);
                }
                if (config["model_update"]["global_weight"]["cs_dif_time_local_weight"]) {
                    getNodeValue(config["model_update"]["global_weight"], "cs_dif_time_local_weight", cs_dif_time_local_weight);
                }
                if (config["model_update"]["global_weight"]["cr_dif_time_local_weight"]) {
                    getNodeValue(config["model_update"]["global_weight"], "cr_dif_time_local_weight", cr_dif_time_local_weight);
                }
                if (config["model_update"]["global_weight"]["teleseismic_weight"]) {
                    getNodeValue(config["model_update"]["global_weight"], "teleseismic_weight", teleseismic_weight);
                }
            }

            // update which model parameters
            if (config["model_update"]["update_slowness"])
                getNodeValue(config["model_update"], "update_slowness", update_slowness);
            if (config["model_update"]["update_azi_ani"])
                getNodeValue(config["model_update"], "update_azi_ani", update_azi_ani);
            if (config["model_udpate"]["update_rad_ani"])
                getNodeValue(config["model_update"], "update_rad_ani", update_rad_ani);

            // taper kernel (now only for teleseismic tomography)
            if (config["model_update"]["depth_taper"]){
                getNodeValue(config["model_update"], "depth_taper", depth_taper[0], 0);
                getNodeValue(config["model_update"], "depth_taper", depth_taper[1], 1);
            }

            // station correction (now only for teleseismic data)
            if (config["model_update"]["use_sta_correction"]){
                getNodeValue(config["model_update"], "use_sta_correction", use_sta_correction);
            }

            // station correction step length
            if (config["model_update"]["step_length_sta_correction"]){
                getNodeValue(config["model_update"], "step_length_sta_correction", step_length_init_sc);
            }
        } // end of model_update

        //
        // relocatoin
        //
        if (config["relocation"]) {
            // the minimum number of data for relocation
            if (config["relocation"]["min_Ndata"]) {
                getNodeValue(config["relocation"], "min_Ndata", min_Ndata_reloc);
            }

            // step size of relocation
            if (config["relocation"]["step_length"]) {
                getNodeValue(config["relocation"], "step_length", step_length_src_reloc);
            }
            // step size decay of relocation
            if (config["relocation"]["step_length_decay"]) {
                getNodeValue(config["relocation"], "step_length_decay", step_length_decay_src_reloc);
            }
            // rescaling values
            if (config["relocation"]["rescaling_dep_lat_lon_ortime"]){
                getNodeValue(config["relocation"], "rescaling_dep_lat_lon_ortime", rescaling_dep, 0);
                getNodeValue(config["relocation"], "rescaling_dep_lat_lon_ortime", rescaling_lat, 1);
                getNodeValue(config["relocation"], "rescaling_dep_lat_lon_ortime", rescaling_lon, 2);
                getNodeValue(config["relocation"], "rescaling_dep_lat_lon_ortime", rescaling_ortime, 3);
            }
            // max change values
            if (config["relocation"]["max_change_dep_lat_lon_ortime"]) {
                getNodeValue(config["relocation"], "max_change_dep_lat_lon_ortime", max_change_dep, 0);
                getNodeValue(config["relocation"], "max_change_dep_lat_lon_ortime", max_change_lat, 1);
                getNodeValue(config["relocation"], "max_change_dep_lat_lon_ortime", max_change_lon, 2);
                getNodeValue(config["relocation"], "max_change_dep_lat_lon_ortime", max_change_ortime, 3);
            }
            // max iteration for relocation
            if (config["relocation"]["max_iterations"]) {
                getNodeValue(config["relocation"], "max_iterations", N_ITER_MAX_SRC_RELOC);
            }
            // norm(grad) threshold of stopping relocation
            if (config["relocation"]["tol_gradient"]) {
                getNodeValue(config["relocation"], "tol_gradient", TOL_SRC_RELOC);
            }
            // data usage
            if (config["relocation"]["abs_time"]) {
                getNodeValue(config["relocation"]["abs_time"], "use_abs_time", use_abs_reloc);
                if (config["relocation"]["abs_time"]["residual_weight"]){
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["relocation"]["abs_time"], "residual_weight", residual_weight_abs_reloc[i], i);
                }
                if (config["relocation"]["abs_time"]["distance_weight"]) {
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["relocation"]["abs_time"], "distance_weight", distance_weight_abs_reloc[i], i);
                }
            }
            if (config["relocation"]["cs_dif_time"]){
                getNodeValue(config["relocation"]["cs_dif_time"], "use_cs_time", use_cs_reloc);
                if (config["relocation"]["cs_dif_time"]["residual_weight"]){
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["relocation"]["cs_dif_time"], "residual_weight", residual_weight_cs_reloc[i], i);
                }
                if (config["relocation"]["cs_dif_time"]["azimuthal_weight"]) {
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["relocation"]["cs_dif_time"], "azimuthal_weight", azimuthal_weight_cs_reloc[i], i);
                }
            }
            if (config["relocation"]["cr_dif_time"]){
                getNodeValue(config["relocation"]["cr_dif_time"], "use_cr_time", use_cr_reloc);
                if (config["relocation"]["cr_dif_time"]["residual_weight"]){
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["relocation"]["cr_dif_time"], "residual_weight", residual_weight_cr_reloc[i], i);
                }
                if (config["relocation"]["cr_dif_time"]["azimuthal_weight"]) {
                    for (int i = 0; i < n_weight; i++)
                        getNodeValue(config["relocation"]["cr_dif_time"], "azimuthal_weight", azimuthal_weight_cr_reloc[i], i);
                }
            }
            // weight of different types of data
            if (config["relocation"]["global_weight"]){
                if (config["relocation"]["global_weight"]["balance_data_weight"]) {
                    getNodeValue(config["relocation"]["global_weight"], "balance_data_weight", balance_data_weight_reloc);
                }
                if (config["relocation"]["global_weight"]["abs_time_weight"]) {
                    getNodeValue(config["relocation"]["global_weight"], "abs_time_weight", abs_time_local_weight_reloc);
                }
                if (config["relocation"]["global_weight"]["cs_dif_time_local_weight"]) {
                    getNodeValue(config["relocation"]["global_weight"], "cs_dif_time_local_weight", cs_dif_time_local_weight_reloc);
                }
                if (config["relocation"]["global_weight"]["cr_dif_time_local_weight"]) {
                    getNodeValue(config["relocation"]["global_weight"], "cr_dif_time_local_weight", cr_dif_time_local_weight_reloc);
                }
            }
        } // end of relocation

        //
        // inversion strategy
        //
        if (config["inversion_strategy"]) {
            // inversion mode, 0: for update model parameters and relocation iteratively
            if (config["inversion_strategy"]["inv_mode"])
                getNodeValue(config["inversion_strategy"], "inv_mode", inv_mode);

            // paramters for inv_mode == 0
            if (config["inversion_strategy"]["inv_mode_0"]){
                // model_update_N_iter
                if (config["inversion_strategy"]["inv_mode_0"]["model_update_N_iter"])
                    getNodeValue(config["inversion_strategy"]["inv_mode_0"], "model_update_N_iter", model_update_N_iter);
                // relocation_N_iter
                if (config["inversion_strategy"]["inv_mode_0"]["relocation_N_iter"])
                    getNodeValue(config["inversion_strategy"]["inv_mode_0"], "relocation_N_iter", relocation_N_iter);
                // max_loop
                if (config["inversion_strategy"]["inv_mode_0"]["max_loop"])
                    getNodeValue(config["inversion_strategy"]["inv_mode_0"], "max_loop", max_loop_mode0);
            }

            // paramters for inv_mode == 1
            if (config["inversion_strategy"]["inv_mode_1"]){
                // max_loop
                if (config["inversion_strategy"]["inv_mode_1"]["max_loop"])
                    getNodeValue(config["inversion_strategy"]["inv_mode_1"], "max_loop", max_loop_mode1);
            }
        }

        //
        // calculation
        //
        if (config["calculation"]) {
            // convergence tolerance
            if (config["calculation"]["convergence_tolerance"]) {
                getNodeValue(config["calculation"], "convergence_tolerance", conv_tol);
            }
            // max number of iterations
            if (config["calculation"]["max_iterations"]) {
                getNodeValue(config["calculation"], "max_iterations", max_iter);
            }
            // stencil order
            if (config["calculation"]["stencil_order"]) {
                getNodeValue(config["calculation"], "stencil_order", stencil_order);
                // check if stencil_order == 999 : hybrid scheme
                if (stencil_order == 999) {
                    hybrid_stencil_order = true;
                    stencil_order = 1;
                }
            }
            // stencil type
            if (config["calculation"]["stencil_type"]) {
                getNodeValue(config["calculation"], "stencil_type", stencil_type);
            }
            // sweep type
            if (config["calculation"]["sweep_type"]) {
                getNodeValue(config["calculation"], "sweep_type", sweep_type);
            }
        }

        if (config["debug"]) {
            if (config["debug"]["debug_mode"]) {
                getNodeValue(config["debug"], "debug_mode", if_test);
            }
        }


        std::cout << "min_max_dep: " << min_dep << " " << max_dep << std::endl;
        std::cout << "min_max_lat: " << min_lat << " " << max_lat << std::endl;
        std::cout << "min_max_lon: " << min_lon << " " << max_lon << std::endl;
        std::cout << "n_rtp: "    << ngrid_k << " " << ngrid_j << " " << ngrid_i << std::endl;
        std::cout << "ndiv_rtp: " << ndiv_k << " "  << ndiv_j  << " " << ndiv_i << std::endl;
        std::cout << "n_subprocs: " << n_subprocs << std::endl;
        std::cout << "n_sims: " << n_sims << std::endl;

    }

    stdout_by_rank_zero("parameter file read done.");

    synchronize_all_world();

    // broadcast all the values read
    broadcast_cr_single(min_dep, 0);
    broadcast_cr_single(max_dep, 0);
    broadcast_cr_single(min_lat, 0);
    broadcast_cr_single(max_lat, 0);
    broadcast_cr_single(min_lon, 0);
    broadcast_cr_single(max_lon, 0);

    broadcast_i_single(ngrid_i, 0);
    broadcast_i_single(ngrid_j, 0);
    broadcast_i_single(ngrid_k, 0);

    broadcast_cr_single(src_dep, 0);
    broadcast_cr_single(src_lat, 0);
    broadcast_cr_single(src_lon, 0);

    broadcast_bool_single(src_rec_file_exist, 0);

    // This have to be done only after broadcast src_rec_file_exist
    if (src_rec_file_exist == false){
        SrcRecInfo src;
        src.id = 0;
        src.name = "s0";
        src.lat    = src_lat;
        src.lon    = src_lon;
        src.dep    = src_dep;
        src_map[src.name] = src;
        SrcRecInfo rec;
        rec.id = 0;
        rec.name = "r0";
        rec_map[rec.name] = rec;
        DataInfo data;
        data_map[src.name][rec.name].push_back(data);
    }

    broadcast_str(src_rec_file, 0);
    broadcast_bool_single(swap_src_rec, 0);


    broadcast_str(init_model_path, 0);
    broadcast_str(model_1d_name, 0);

    broadcast_i_single(n_sims, 0);
    broadcast_i_single(ndiv_i, 0);
    broadcast_i_single(ndiv_j, 0);
    broadcast_i_single(ndiv_k, 0);
    broadcast_i_single(n_subprocs, 0);
    broadcast_bool_single(use_gpu, 0);

    broadcast_str(output_dir, 0);
    broadcast_bool_single(output_source_field, 0);
    broadcast_bool_single(output_kernel, 0);
    broadcast_i_single(verbose_output_level, 0);
    broadcast_bool_single(output_final_model, 0);
    broadcast_bool_single(output_middle_model, 0);
    broadcast_bool_single(output_in_process, 0);
    broadcast_bool_single(output_in_process_data, 0);
    broadcast_bool_single(single_precision_output, 0);
    broadcast_i_single(output_format, 0);

    broadcast_i_single(run_mode, 0);
    broadcast_bool_single(have_tele_data, 0);

    broadcast_i_single(max_iter_inv, 0);
    broadcast_i_single(optim_method, 0);
    broadcast_i_single(step_method, 0);
    broadcast_cr_single(step_length_init, 0);
    broadcast_cr_single(step_length_decay, 0);
    broadcast_cr_single(step_length_gradient_angle, 0);
    broadcast_cr_single(step_length_down, 0);
    broadcast_cr_single(step_length_up, 0);
    broadcast_cr_single(Kdensity_coe, 0);
    broadcast_i_single(max_sub_iterations, 0);
    broadcast_cr_single(regularization_weight, 0);
    broadcast_cr_single(regul_lr, 0);
    broadcast_cr_single(regul_lt, 0);
    broadcast_cr_single(regul_lp, 0);
    broadcast_i_single(smooth_method, 0);
    broadcast_cr_single(smooth_lr, 0);
    broadcast_cr_single(smooth_lt, 0);
    broadcast_cr_single(smooth_lp, 0);

    broadcast_i_single(n_inversion_grid, 0);

    broadcast_bool_single(uniform_inv_grid_dep, 0);
    broadcast_bool_single(uniform_inv_grid_lon, 0);
    broadcast_bool_single(uniform_inv_grid_lat, 0);

    broadcast_i_single(n_inv_r_flex, 0);
    broadcast_i_single(n_inv_t_flex, 0);
    broadcast_i_single(n_inv_p_flex, 0);

    broadcast_i_single(n_inv_r_flex_ani, 0);
    broadcast_i_single(n_inv_t_flex_ani, 0);
    broadcast_i_single(n_inv_p_flex_ani, 0);

    if (world_rank != 0) {
        dep_inv = allocateMemory<CUSTOMREAL>(n_inv_r_flex, 5012);
        lat_inv = allocateMemory<CUSTOMREAL>(n_inv_t_flex, 5013);
        lon_inv = allocateMemory<CUSTOMREAL>(n_inv_p_flex, 5014);
        dep_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_r_flex_ani, 5015);
        lat_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_t_flex_ani, 5016);
        lon_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_p_flex_ani, 5017);
    }
    broadcast_cr(dep_inv,n_inv_r_flex, 0);
    broadcast_cr(lat_inv,n_inv_t_flex, 0);
    broadcast_cr(lon_inv,n_inv_p_flex, 0);
    broadcast_cr(trapezoid,n_trapezoid, 0);
    broadcast_cr(dep_inv_ani,n_inv_r_flex_ani, 0);
    broadcast_cr(lat_inv_ani,n_inv_t_flex_ani, 0);
    broadcast_cr(lon_inv_ani,n_inv_p_flex_ani, 0);
    broadcast_cr(trapezoid_ani,n_trapezoid, 0);

    broadcast_bool_single(invgrid_ani, 0);
    // broadcast_bool_single(invgrid_volume_rescale, 0);

    broadcast_bool_single(use_sta_correction, 0);
    broadcast_cr_single(step_length_init_sc, 0);
    broadcast_bool_single(sta_correction_file_exist, 0);
    broadcast_str(sta_correction_file, 0);

    broadcast_bool_single(use_abs, 0);
    broadcast_cr(residual_weight_abs, n_weight, 0);
    broadcast_cr(distance_weight_abs, n_weight, 0);
    broadcast_bool_single(use_cs, 0);
    broadcast_cr(residual_weight_cs, n_weight, 0);
    broadcast_cr(azimuthal_weight_cs, n_weight, 0);
    broadcast_bool_single(use_cr, 0);
    broadcast_cr(residual_weight_cr, n_weight, 0);
    broadcast_cr(azimuthal_weight_cr, n_weight, 0);

    broadcast_bool_single(balance_data_weight, 0);
    broadcast_cr_single(abs_time_local_weight, 0);
    broadcast_cr_single(cs_dif_time_local_weight, 0);
    broadcast_cr_single(cr_dif_time_local_weight, 0);
    broadcast_cr_single(teleseismic_weight, 0);

    broadcast_bool_single(update_slowness, 0);
    broadcast_bool_single(update_azi_ani, 0);
    broadcast_bool_single(update_rad_ani, 0);
    broadcast_cr(depth_taper,2,0);

    broadcast_i_single(min_Ndata_reloc, 0);
    broadcast_cr_single(step_length_src_reloc, 0);
    broadcast_cr_single(step_length_decay_src_reloc, 0);
    broadcast_cr_single(rescaling_dep, 0);
    broadcast_cr_single(rescaling_lat, 0);
    broadcast_cr_single(rescaling_lon, 0);
    broadcast_cr_single(rescaling_ortime, 0);
    broadcast_cr_single(max_change_dep, 0);
    broadcast_cr_single(max_change_lat, 0);
    broadcast_cr_single(max_change_lon, 0);
    broadcast_cr_single(max_change_ortime, 0);
    broadcast_i_single(N_ITER_MAX_SRC_RELOC, 0);
    broadcast_cr_single(TOL_SRC_RELOC, 0);
    broadcast_bool_single(ortime_local_search, 0);

    broadcast_bool_single(use_abs_reloc, 0);
    broadcast_cr(residual_weight_abs_reloc, n_weight, 0);
    broadcast_cr(distance_weight_abs_reloc, n_weight, 0);
    broadcast_bool_single(use_cs_reloc, 0);
    broadcast_cr(residual_weight_cs_reloc, n_weight, 0);
    broadcast_cr(azimuthal_weight_cs_reloc, n_weight, 0);
    broadcast_bool_single(use_cr_reloc, 0);
    broadcast_cr(residual_weight_cr_reloc, n_weight, 0);
    broadcast_cr(azimuthal_weight_cr_reloc, n_weight, 0);

    broadcast_bool_single(balance_data_weight_reloc, 0);
    broadcast_cr_single(abs_time_local_weight_reloc, 0);
    broadcast_cr_single(cs_dif_time_local_weight_reloc, 0);
    broadcast_cr_single(cr_dif_time_local_weight_reloc, 0);

    broadcast_i_single(inv_mode, 0);
    broadcast_i_single(model_update_N_iter, 0);
    broadcast_i_single(relocation_N_iter, 0);
    broadcast_i_single(max_loop_mode0, 0);
    broadcast_i_single(max_loop_mode1, 0);

    broadcast_cr_single(conv_tol, 0);
    broadcast_i_single(max_iter, 0);
    broadcast_i_single(stencil_order, 0);
    broadcast_bool_single(hybrid_stencil_order, 0);
    broadcast_i_single(stencil_type, 0);
    broadcast_i_single(sweep_type, 0);
    broadcast_bool_single(if_test, 0);

    // check contradictory settings
    check_contradictions();

    // write parameter file to output directory
    if (world_rank == 0)
        write_params_to_file();

    // broadcast the values to all processes
    stdout_by_rank_zero("read input file successfully.");

}


InputParams::~InputParams(){
    // free memory

    // check allocated memory
    if (dep_inv != nullptr) delete [] dep_inv;
    if (lat_inv != nullptr) delete [] lat_inv;
    if (lon_inv != nullptr) delete [] lon_inv;

    if (dep_inv_ani != nullptr) delete [] dep_inv_ani;
    if (lat_inv_ani != nullptr) delete [] lat_inv_ani;
    if (lon_inv_ani != nullptr) delete [] lon_inv_ani;

    // clear all src, rec, data
    src_map.clear();
    src_map_tele.clear();
    src_map_all.clear();
    src_map_back.clear();
    src_map_tele.clear();
    rec_map.clear();
    rec_map_tele.clear();
    rec_map_all.clear();
    rec_map_back.clear();
    data_map.clear();
    data_map_tele.clear();
    data_map_all.clear();
    data_map_back.clear();
}


void InputParams::write_params_to_file() {
    // write all the simulation parameters in another yaml file
    std::string file_name = "params_log.yaml";
    std::ofstream fout(file_name);
    // print boolean as string
    fout << std::boolalpha;

    fout << "version: " << 3 << std::endl;

    fout << std::endl;

    fout << "#################################################" << std::endl;
    fout << "#            computational domian               #" << std::endl;
    fout << "#################################################" << std::endl;
    fout << "domain:" << std::endl;
    fout << "  min_max_dep: [" << min_dep << ", " << max_dep << "] # depth in km" << std::endl;
    fout << "  min_max_lat: [" << min_lat << ", " << max_lat << "] # latitude in degree" << std::endl;
    fout << "  min_max_lon: [" << min_lon << ", " << max_lon << "] # longitude in degree" << std::endl;
    fout << "  n_rtp: [" << ngrid_k << ", " << ngrid_j << ", " << ngrid_i << "] # number of nodes in depth,latitude,longitude direction" << std::endl;
    fout << std::endl;

    fout << "#################################################" << std::endl;
    fout << "#            traveltime data file path          #" << std::endl;
    fout << "#################################################" << std::endl;
    fout << "source:" << std::endl;
    fout << "  src_rec_file: " << src_rec_file     << " # source receiver file path" << std::endl;
    fout << "  swap_src_rec: " << swap_src_rec << " # swap source and receiver (only valid for regional source and receiver, those of tele remain unchanged)" << std::endl;
    fout << std::endl;

    fout << "#################################################" << std::endl;
    fout << "#            initial model file path            #" << std::endl;
    fout << "#################################################" << std::endl;
    fout << "model:" << std::endl;
    fout << "  init_model_path: " << init_model_path << " # path to initial model file " << std::endl;
    // check if model_1d_name has any characters
    if (model_1d_name.size() > 0)
        fout << "  model_1d_name: " << model_1d_name;
    else
        fout << "#   model_1d_name: " << "dummy_model_1d_name";
    fout << " # 1D model name used in teleseismic 2D solver (iasp91, ak135, user_defined is available), defined in include/1d_model.h" << std::endl;
    fout << std::endl;


    fout << "#################################################" << std::endl;
    fout << "#            parallel computation settings      #" << std::endl;
    fout << "#################################################" << std::endl;
    fout << "parallel: # parameters for parallel computation" << std::endl;
    fout << "  n_sims: "    << n_sims << " # number of simultanoues runs (parallel the sources)" << std::endl;
    fout << "  ndiv_rtp: [" << ndiv_k << ", " << ndiv_j << ", " << ndiv_i << "] # number of subdivision on each direction (parallel the computional domain)" << std::endl;
    fout << "  nproc_sub: " << n_subprocs << " # number of processors for sweep parallelization (parallel the fast sweep method)" << std::endl;
    fout << "  use_gpu: "   << use_gpu << " # true if use gpu (EXPERIMENTAL)" << std::endl;
    fout << std::endl;

    fout << "############################################" << std::endl;
    fout << "#            output file setting           #" << std::endl;
    fout << "############################################" << std::endl;
    fout << "output_setting:" << std::endl;
    fout << "  output_dir: "              << output_dir << " # path to output director (default is ./OUTPUT_FILES/)" << std::endl;
    fout << "  output_source_field:     " << output_source_field         << " # True: output the traveltime field and adjoint field of all sources at each iteration. Default: false. File: 'out_data_sim_group_X'." << std::endl;
    fout << "  output_kernel:           " << output_kernel               << " # True: output sensitivity kernel and kernel density. Default: false. File: 'out_data_sim_group_X'." << std::endl;
    fout << "  output_final_model:      " << output_final_model          << " # True: output merged final model. This file can be used as the input model for TomoATT. Default: true. File: 'model_final.h5'." << std::endl;
    fout << "  output_middle_model:     " << output_middle_model         << " # True: output merged intermediate models during inversion. This file can be used as the input model for TomoATT. Default: false. File: 'middle_model_step_XXXX.h5'" << std::endl;
    fout << "  output_in_process:       " << output_in_process           << " # True: output at each inv iteration, otherwise, only output step 0, Niter-1, Niter. Default: true. File: 'out_data_sim_group_0'." << std::endl;
    fout << "  output_in_process_data:  " << output_in_process_data      << " # True: output src_rec_file at each inv iteration, otherwise, only output step 0, Niter-2, Niter-1. Default: true. File: 'src_rec_file_step_XXXX.dat'" << std::endl;
    fout << "  single_precision_output: " << single_precision_output     << " # True: output results in single precision. Default: false.   " << std::endl;
    fout << "  verbose_output_level:    " << verbose_output_level        << " # output internal parameters, (to do)." << std::endl;
    int ff_flag=0;
    if (output_format == OUTPUT_FORMAT_HDF5) ff_flag = 0;
    else if (output_format == OUTPUT_FORMAT_ASCII) ff_flag = 1;
    else {
        std::cout << "Error: output_format is not defined!" << std::endl;
        exit(1);
    }
    fout << "  output_file_format: " << ff_flag << " # 0: hdf5, 1: ascii" << std::endl;
    fout << std::endl;
    fout << "# output files:" << std::endl;
    fout << "# File: 'out_data_grid.h5'. Keys: ['Mesh']['elem_conn'], element index; " << std::endl;
    fout << "#                                 ['Mesh']['node_coords_p'], phi coordinates of nodes; " << std::endl;
    fout << "#                                 ['Mesh']['node_coords_t'], theta coordinates of nodes; " << std::endl;
    fout << "#                                 ['Mesh']['node_coords_r'], r coordinates of nodes;" << std::endl;
    fout << "#                                 ['Mesh']['node_coords_x'], phi coordinates of elements; " << std::endl;
    fout << "#                                 ['Mesh']['node_coords_y'], theta coordinates of elements; " << std::endl;
    fout << "#                                 ['Mesh']['node_coords_z'], r coordinates of elements; " << std::endl;
    fout << "# File: 'out_data_sim_group_0'. Keys: ['model']['vel_inv_XXXX'], velocity model; " << std::endl;
    fout << "#                                     ['model']['xi_inv_XXXX'], xi model; " << std::endl;
    fout << "#                                     ['model']['eta_inv_XXXX'], eta model" << std::endl;
    fout << "#                                     ['model']['Ks_inv_XXXX'], sensitivity kernel related to slowness" << std::endl;
    fout << "#                                     ['model']['Kxi_inv_XXXX'], sensitivity kernel related to xi" << std::endl;
    fout << "#                                     ['model']['Keta_inv_XXXX'], sensitivity kernel related to eta" << std::endl;
    fout << "#                                     ['model']['Ks_density_inv_XXXX'], kernel density of Ks " << std::endl;
    fout << "#                                     ['model']['Kxi_density_inv_XXXX'], kernel density of Kxi " << std::endl;
    fout << "#                                     ['model']['Keta_density_inv_XXXX'], kernel density of Keta " << std::endl;
    fout << "#                                     ['model']['Ks_over_Kden_inv_XXXX'], slowness kernel over kernel density" << std::endl;
    fout << "#                                     ['model']['Kxi_over_Kden_inv_XXXX'], xi kernel over kernel density" << std::endl;
    fout << "#                                     ['model']['Keta_over_Kden_inv_XXXX'], eta kernel over kernel density" << std::endl;
    fout << "#                                     ['model']['Ks_update_inv_XXXX'], slowness kernel over kernel density, smoothed by inversion grid" << std::endl;
    fout << "#                                     ['model']['Kxi_update_inv_XXXX'], xi kernel over kernel density, smoothed by inversion grid" << std::endl;
    fout << "#                                     ['model']['Keta_update_inv_XXXX'], eta kernel over kernel density, smoothed by inversion grid" << std::endl;
    fout << "# File: 'src_rec_file_step_XXXX.dat' or 'src_rec_file_forward.dat'. The synthetic traveltime data file." << std::endl;
    fout << "# File: 'final_model.h5'. Keys: ['eta'], ['xi'], ['vel'], the final model." << std::endl;
    fout << "# File: 'middle_model_step_XXXX.h5'. Keys: ['eta'], ['xi'], ['vel'], the model at step XXXX." << std::endl;
    fout << "# File: 'inversion_grid.txt'. The location of inversion grid nodes" << std::endl;
    fout << "# File: 'objective_function.txt'. The objective function value at each iteration" << std::endl;
    fout << "# File: 'out_data_sim_group_X'. Keys: ['src_$src_name']['time_field_inv_XXXX'], traveltime field for source $src_name at iteration XXXX;" << std::endl;
    fout << "#                                     ['src_$src_name']['adjoint_field_inv_XXXX'], adjoint field for source $src_name at iteration XXXX;" << std::endl;
    fout << std::endl;
    fout << std::endl;

    fout << "#################################################" << std::endl;
    fout << "#          inversion or forward modeling        #" << std::endl;
    fout << "#################################################" << std::endl;
    fout << "# run mode"                                        << std::endl;
    fout << "# 0 for forward simulation only,"                  << std::endl;
    fout << "# 1 for inversion"                                 << std::endl;
    fout << "# 2 for earthquake relocation"                     << std::endl;
    fout << "# 3 for inversion + earthquake relocation"         << std::endl;
    fout << "# 4 for 1d model inversion"                        << std::endl;
    fout << "run_mode: " << run_mode << std::endl;
    fout << std::endl;

    fout << "have_tele_data: " << have_tele_data << " # An error will be reported if false but source out of study region is used. Default: false." << std::endl;
    fout << std::endl;
    fout << std::endl;

    fout << "###################################################" << std::endl;
    fout << "#          model update parameters setting        #" << std::endl;
    fout << "###################################################" << std::endl;
    fout << "model_update:" << std::endl;
    fout << "  max_iterations: " << max_iter_inv << " # maximum number of inversion iterations" << std::endl;
    fout << "  optim_method: "   << optim_method << " # optimization method. 0 : grad_descent, 1 : halve-stepping, 2 : lbfgs (EXPERIMENTAL)" << std::endl;
    fout << std::endl;
    fout << "  #common parameters for all optim methods" << std::endl;
    fout << "  step_length: "             << step_length_init << " # the initial step length of model perturbation. 0.01 means maximum 1% perturbation for each iteration." << std::endl;
    fout << std::endl;
    fout << "  # parameters for optim_method 0 (gradient_descent)" << std::endl;
    fout << "  optim_method_0:" << std::endl;
    fout << "    step_method: " << step_method << "  # the method to modulate step size. 0: according to objective function; 1: according to gradient direction " << std::endl;
    fout << "    # if step_method:0. if objective function increase, step size -> step length * step_length_decay. " << std::endl;
    fout << "    step_length_decay: " << step_length_decay << " # default: 0.9" << std::endl;
    fout << "    # if step_method:1. if the angle between the current and the previous gradients is greater than step_length_gradient_angle, step size -> step length * step_length_change[0]. " << std::endl;
    fout << "    #                                                                                                                otherwise, step size -> step length * step_length_change[1]. " << std::endl;
    fout << "    step_length_gradient_angle: " <<  step_length_gradient_angle << " # default: 120.0 " << std::endl;
    fout << "    step_length_change: [" <<  step_length_down << ", " << step_length_up << "] # default: [0.5,1.2] " << std::endl;
    fout << "    # Kdensity_coe is used to rescale the final kernel:  kernel -> kernel / pow(density of kernel, Kdensity_coe).  if Kdensity_coe > 0, the region with less data will be enhanced during the inversion" << std::endl;
    fout << "    #  e.g., if Kdensity_coe = 0, kernel remains upchanged; if Kdensity_coe = 1, kernel is fully normalized. 0.5 or less is recommended if really required." << std::endl;
    fout << "    Kdensity_coe: " <<  Kdensity_coe << " # default: 0.0,  limited range: 0.0 - 0.95 " << std::endl;
    fout << std::endl;
    fout << "  # parameters for optim_method 1 (halve-stepping) or 2 (lbfgs)" << std::endl;
    fout << "  optim_method_1_2:" << std::endl;
    fout << "    max_sub_iterations: "    << max_sub_iterations << " # maximum number of each sub-iteration" << std::endl;
    fout << "    regularization_weight: " << regularization_weight << " # weight value for regularization (lbfgs mode only)" << std::endl;

    fout << "    coefs_regulalization_rtp: [" << regul_lr << ", " << regul_lt << ", " << regul_lp << "] # regularization coefficients for rtp (lbfgs mode only)" << std::endl;
    fout << std::endl;

    fout << "  # smoothing" << std::endl;
    fout << "  smoothing:" << std::endl;
    fout << "    smooth_method: " << smooth_method << " # 0: multiparametrization, 1: laplacian smoothing (EXPERIMENTAL)" << std::endl;
    fout << "    l_smooth_rtp: ["         << smooth_lr << ", " << smooth_lt << ", " << smooth_lp << "] # smoothing coefficients for laplacian smoothing" << std::endl;
    fout << std::endl;

    fout << "  # parameters for smooth method 0 (multigrid model parametrization)" << std::endl;
    fout << "  # inversion grid can be viewed in OUTPUT_FILES/inversion_grid.txt" << std::endl;
    fout << "  n_inversion_grid: "   << n_inversion_grid << " # number of inversion grid sets" << std::endl;
    fout << std::endl;

    fout << "  uniform_inv_grid_dep: " << uniform_inv_grid_dep << " # true if use uniform inversion grid for dep, false if use flexible inversion grid" << std::endl;
    fout << "  uniform_inv_grid_lat: " << uniform_inv_grid_lat << " # true if use uniform inversion grid for lat, false if use flexible inversion grid" << std::endl;
    fout << "  uniform_inv_grid_lon: " << uniform_inv_grid_lon << " # true if use uniform inversion grid for lon, false if use flexible inversion grid" << std::endl;
    fout << std::endl;

    fout << "  # -------------- uniform inversion grid setting -------------- " << std::endl;
    fout << "  # settings for uniform inversion grid" << std::endl;
    fout << "  n_inv_dep_lat_lon: [" << n_inv_r << ", " << n_inv_t << ", " << n_inv_p << "] # number of the base inversion grid points" << std::endl;
    fout << "  min_max_dep_inv: [" << min_dep_inv << ", " << max_dep_inv << "] # depth in km (Radius of the earth is defined in config.h/R_earth)" << std::endl;
    fout << "  min_max_lat_inv: [" << min_lat_inv << ", " << max_lat_inv << "] # latitude in degree" << std::endl;
    fout << "  min_max_lon_inv: [" << min_lon_inv << ", " << max_lon_inv << "] # longitude in degree" << std::endl;
    fout << std::endl;

    fout << "  # -------------- flexible inversion grid setting -------------- " << std::endl;
    fout << "  # settings for flexible inversion grid" << std::endl;
    fout << "  dep_inv: [";
    for (int i = 0; i < n_inv_r_flex; i++){
        fout << dep_inv[i];
        if (i != n_inv_r_flex-1)
            fout << ", ";
    }
    fout << "] # inversion grid for vel in depth (km)" << std::endl;
    fout << "  lat_inv: [";
    for (int i = 0; i < n_inv_t_flex; i++){
        fout << lat_inv[i];
        if (i != n_inv_t_flex-1)
            fout << ", ";
    }
    fout << "] # inversion grid for vel in latitude (degree)" << std::endl;
    fout << "  lon_inv: [";
    for (int i = 0; i < n_inv_p_flex; i++){
        fout << lon_inv[i];
        if (i != n_inv_p_flex-1)
            fout << ", ";
    }
    fout << "] # inversion grid for vel in longitude (degree)" << std::endl;
    fout << "  trapezoid: [";
    for (int i = 0; i < n_trapezoid; i++){
        fout << trapezoid[i];
        if (i != n_trapezoid-1)
            fout << ", ";
    }
    fout << "]  # usually set as [1.0, 0.0, 50.0] (default)" << std::endl;
    fout << std::endl;

    fout << "  # if we want to use another inversion grid for inverting anisotropy, set invgrid_ani: true (default: false)" << std::endl;
    fout << "  invgrid_ani: " << invgrid_ani << std::endl;
    fout << std::endl;

    fout << "  # ---------- uniform inversion grid setting for anisotropy ----------" << std::endl;
    fout << "  n_inv_dep_lat_lon_ani: [" << n_inv_r_ani << ", " << n_inv_t_ani << ", " << n_inv_p_ani << "] # number of the base inversion grid points" << std::endl;
    fout << "  min_max_dep_inv_ani: [" << min_dep_inv_ani << ", " << max_dep_inv_ani << "] # depth in km (Radius of the earth is defined in config.h/R_earth)" << std::endl;
    fout << "  min_max_lat_inv_ani: [" << min_lat_inv_ani << ", " << max_lat_inv_ani << "] # latitude in degree" << std::endl;
    fout << "  min_max_lon_inv_ani: [" << min_lon_inv_ani << ", " << max_lon_inv_ani << "] # longitude in degree" << std::endl;
    fout << std::endl;

    fout << "  # ---------- flexible inversion grid setting for anisotropy ----------" << std::endl;
    fout << "  # settings for flexible inversion grid for anisotropy" << std::endl;
    fout << "  dep_inv_ani: [";
    for (int i = 0; i < n_inv_r_flex_ani; i++){
        fout << dep_inv_ani[i];
        if (i != n_inv_r_flex_ani-1)
            fout << ", ";
    }
    fout << "] # inversion grid for ani in depth (km)" << std::endl;
    fout << "  lat_inv_ani: [";
    for (int i = 0; i < n_inv_t_flex_ani; i++){
        fout << lat_inv_ani[i];
        if (i != n_inv_t_flex_ani-1)
            fout << ", ";
    }
    fout << "] # inversion grid for ani in latitude (degree)" << std::endl;
    fout << "  lon_inv_ani: [";
    for (int i = 0; i < n_inv_p_flex_ani; i++){
        fout << lon_inv_ani[i];
        if (i != n_inv_p_flex_ani-1)
            fout << ", ";
    }
    fout << "] # inversion grid for ani in longitude (degree)" << std::endl;
    fout << "  trapezoid_ani: [";
    for (int i = 0; i < n_trapezoid; i++){
        fout << trapezoid_ani[i];
        if (i != n_trapezoid-1)
            fout << ", ";
    }
    fout << "]  # usually set as [1.0, 0.0, 50.0] (default)" << std::endl;
    fout << std::endl;

    fout << "  # Carefully change trapezoid and trapezoid_ani, if you really want to use trapezoid inversion grid, increasing the inversion grid spacing with depth to account for the worse data coverage in greater depths." << std::endl;
    fout << "  # The trapezoid_ inversion grid with index (i,j,k) in longitude, latitude, and depth is defined as:"         << std::endl;
    fout << "  # if                 dep_inv[k] < trapezoid[1], lon = lon_inv[i];                   "                        << std::endl;
    fout << "  #                                               lat = lat_inv[j];              "                             << std::endl;
    fout << "  #                                               dep = dep_inv[k];"                                           << std::endl;
    fout << "  # if trapezoid[1] <= dep_inv[k] < trapezoid[2], lon = mid_lon_inv+(lon_inv[i]-mid_lon_inv)*(dep_inv[k]-trapezoid[1])/(trapezoid[2]-trapezoid[1])*trapezoid[0]; " << std::endl;
    fout << "  #                                               lat = mid_lat_inv+(lat_inv[i]-mid_lat_inv)*(dep_inv[k]-trapezoid[1])/(trapezoid[2]-trapezoid[1])*trapezoid[0]; " << std::endl;
    fout << "  #                                               dep = dep_inv[k];"                                           << std::endl;
    fout << "  # if trapezoid[2] <= dep_inv[k],                lon = mid_lon_inv+(lon_inv[i]-mid_lon_inv)*trapezoid[0]; "   << std::endl;
    fout << "  #                                               lat = mid_lat_inv+(lat_inv[i]-mid_lat_inv)*trapezoid[0]; "   << std::endl;
    fout << "  #                                               dep = dep_inv[k];"                                   << std::endl;
    fout << "  # The shape of trapezoid inversion gird (x) looks like:"                                             << std::endl;
    fout << "  #"                                                                                                   << std::endl;
    fout << "  #                                 lon_inv[0]   [1]      [2]      [3]      [4]"                       << std::endl;
    fout << "  #                                  |<-------- (lon_inv[end] - lon_inv[0]) ---->|   "                 << std::endl;
    fout << "  #  dep_inv[0]                      |   x        x        x        x        x   |   "                 << std::endl;
    fout << "  #                                  |                                           |   "                 << std::endl;
    fout << "  #  dep_inv[1]                      |   x        x        x        x        x   |   "                 << std::endl;
    fout << "  #                                  |                                           |   "                 << std::endl;
    fout << "  #  dep_inv[2] = trapezoid[1]      /    x        x        x        x        x    \\ "                 << std::endl;
    fout << "  #                                /                                               \\ "                << std::endl;
    fout << "  #  dep_inv[3]                   /    x         x         x         x         x    \\ "               << std::endl;
    fout << "  #                              /                                                   \\ "              << std::endl;
    fout << "  #  dep_inv[4] = trapezoid[2]  /    x          x          x          x          x    \\ "             << std::endl;
    fout << "  #                            |                                                       | "             << std::endl;
    fout << "  #  dep_inv[5]                |     x          x          x          x          x     |  "            << std::endl;
    fout << "  #                            |                                                       |  "            << std::endl;
    fout << "  #  dep_inv[6]                |     x          x          x          x          x     |  "            << std::endl;
    fout << "  #                            |<---- trapezoid[0]* (lon_inv[end] - lon_inv[0]) ------>|  "            << std::endl;
    fout << std::endl;


    // fout << "  # inversion grid volume rescale (kernel -> kernel / volume of inversion grid mesh)," << std::endl;
    // fout << "  # this precondition may be carefully applied if the sizes of inversion grids are unbalanced" << std::endl;
    // fout << "  invgrid_volume_rescale: " << invgrid_volume_rescale << std::endl;
    fout << std::endl;

    fout << "  # path to station correction file (under development)" << std::endl;
    fout << "  use_sta_correction: " << use_sta_correction << std::endl;
    if (sta_correction_file_exist)
        fout << "  initial_sta_correction_file: " << sta_correction_file;
    else
        fout << "  # initial_sta_correction_file: " << "dummy_sta_correction_file";
    fout << "  # the path of initial station correction " << std::endl;
    fout << "  step_length_sta_correction: " << step_length_init_sc << " # step length relate to the update of station correction terms" << std::endl;
    fout << std::endl;

    fout << std::endl;

    fout << "  # In the following data subsection, XXX_weight means a weight is assigned to the data, influencing the objective function and gradient" << std::endl;
    fout << "  # XXX_weight : [d1,d2,w1,w2] means: " << std::endl;
    fout << "  # if       XXX < d1, weight = w1 " << std::endl;
    fout << "  # if d1 <= XXX < d2, weight = w1 + (XXX-d1)/(d2-d1)*(w2-w1),  (linear interpolation) " << std::endl;
    fout << "  # if d2 <= XXX     , weight = w2 " << std::endl;
    fout << "  # You can easily set w1 = w2 = 1.0 to normalize the weight related to XXX." << std::endl;

    fout << "  # -------------- using absolute traveltime data --------------" << std::endl;
    fout << "  abs_time:" << std::endl;
    fout << "    use_abs_time: " << use_abs << " # 'true' for using absolute traveltime data to update model parameters; 'false' for not using (no need to set parameters in this section)" << std::endl;
    fout << "    residual_weight: [";
    for (int i = 0; i < n_weight; i++){
        fout << residual_weight_abs[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "] # XXX is the absolute traveltime residual (second) = abs(t^{obs}_{n,i} - t^{syn}_{n,j})" << std::endl;
    fout << "    distance_weight: [";
    for (int i = 0; i < n_weight; i++){
        fout << distance_weight_abs[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "] # XXX is epicenter distance (km) between the source and receiver related to the data" << std::endl;
    fout << std::endl;

    fout << "  # -------------- using common source differential traveltime data --------------" << std::endl;
    fout << "  cs_dif_time:" << std::endl;
    fout << "    use_cs_time: " << use_cs << " # 'true' for using common source differential traveltime data to update model parameters; 'false' for not using (no need to set parameters in this section)" << std::endl;
    fout << "    residual_weight: [";
    for (int i = 0; i < n_weight; i++){
        fout << residual_weight_cs[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "] # XXX is the common source differential traveltime residual (second) = abs(t^{obs}_{n,i} - t^{obs}_{n,j} - t^{syn}_{n,i} + t^{syn}_{n,j})." << std::endl;
    fout << "    azimuthal_weight: [";
    for (int i = 0; i < n_weight; i++){
        fout << azimuthal_weight_cs[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "] # XXX is the azimuth difference between two separate stations related to the common source." << std::endl;
    fout << std::endl;

    fout << "  # -------------- using common receiver differential traveltime data --------------" << std::endl;
    fout << "  cr_dif_time:" << std::endl;
    fout << "    use_cr_time: " << use_cr << " # 'true' for using common receiver differential traveltime data to update model parameters; 'false' for not using (no need to set parameters in this section)" << std::endl;
    fout << "    residual_weight: [";
    for (int i = 0; i < n_weight; i++){
        fout << residual_weight_cr[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "] # XXX is the common receiver differential traveltime residual (second) = abs(t^{obs}_{n,i} - t^{obs}_{m,i} - t^{syn}_{n,i} + t^{syn}_{m,i})" << std::endl;
    fout << "    azimuthal_weight: [";
    for (int i = 0; i < n_weight; i++){
        fout << azimuthal_weight_cr[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "] # XXX is the azimuth difference between two separate sources related to the common receiver." << std::endl;
    fout << std::endl;

    fout << "  # -------------- global weight of different types of data (to balance the weight of different data) --------------" << std::endl;
    fout << "  global_weight:" << std::endl;
    fout << "    balance_data_weight: " << balance_data_weight << " # yes: over the total weight of the each type of the data. no: use original weight (below weight for each type of data needs to be set)" << std::endl;
    fout << "    abs_time_weight: " << abs_time_local_weight  << " # weight of absolute traveltime data after balance,                       default: 1.0" << std::endl;
    fout << "    cs_dif_time_local_weight: " << cs_dif_time_local_weight << " # weight of common source differential traveltime data after balance,     default: 1.0" << std::endl;
    fout << "    cr_dif_time_local_weight: " << cr_dif_time_local_weight << " # weight of common receiver differential traveltime data after balance,   default: 1.0" << std::endl;
    fout << "    teleseismic_weight: " << teleseismic_weight << " # weight of teleseismic data after balance,                               default: 1.0  (exclude in this version)" << std::endl;
    fout << std::endl;

    fout << "  # -------------- inversion parameters --------------" << std::endl;
    fout << "  update_slowness : " << update_slowness << " # update slowness (velocity) or not.              default: true" << std::endl;
    fout << "  update_azi_ani  : " << update_azi_ani  << " # update azimuthal anisotropy (xi, eta) or not.   default: false" << std::endl;
    // fout << "  update_rad_ani  : " << update_rad_ani  << " # update radial anisotropy (in future) or not.    default: false" << std::endl;
    fout << std::endl;

    fout << "  # -------------- for teleseismic inversion (under development) --------------" << std::endl;
    fout << "  # depth_taper : [d1,d2] means: " << std::endl;
    fout << "  # if       XXX < d1, kernel <- kernel * 0.0 " << std::endl;
    fout << "  # if d1 <= XXX < d2, kernel <- kernel * (XXX-d1)/(d2-d1),  (linear interpolation) " << std::endl;
    fout << "  # if d2 <= XXX     , kernel <- kernel * 1.0 " << std::endl;
    fout << "  # You can easily set d1 = -200, d1 = -100 to remove this taper." << std::endl;
    fout << "  depth_taper : [" << depth_taper[0] << ", " << depth_taper[1] << "]"  << std::endl;
    fout << std::endl;

    fout << "#################################################" << std::endl;
    fout << "#          relocation parameters setting        #" << std::endl;
    fout << "#################################################" << std::endl;
    fout << "relocation: # update earthquake hypocenter and origin time (when run_mode : 2 and 3)" << std::endl;
    fout << "  min_Ndata: " << min_Ndata_reloc << " # if the number of data of the earthquake is less than <min_Ndata>, the earthquake will not be relocated.  defaut value: 4 " << std::endl;
    fout << std::endl;

    fout << "  # relocation_strategy" << std::endl;
    fout << "  step_length : " << step_length_src_reloc << " # initial step length of relocation perturbation. 0.01 means maximum 1% perturbation for each iteration." << std::endl;
    fout << "  step_length_decay : " << step_length_decay_src_reloc << " # if objective function increase, step size -> step length * step_length_decay. default: 0.9" << std::endl;

    fout << "  rescaling_dep_lat_lon_ortime  : [";
    fout << rescaling_dep << ", " << rescaling_lat << ", " << rescaling_lon << ", " << rescaling_ortime;
    fout << "]  # The perturbation is related to <rescaling_dep_lat_lon_ortime>. Unit: km,km,km,second" << std::endl;

    fout << "  max_change_dep_lat_lon_ortime : [";
    fout << max_change_dep << ", " << max_change_lat << ", " << max_change_lon << ", " << max_change_ortime;
    fout << "]     # the change of dep,lat,lon,ortime do not exceed max_change. Unit: km,km,km,second" << std::endl;
    fout << "  max_iterations : " << N_ITER_MAX_SRC_RELOC <<" # maximum number of iterations for relocation" << std::endl;
    fout << "  tol_gradient : " << TOL_SRC_RELOC << " # if the norm of gradient is smaller than the tolerance, the iteration of relocation terminates" << std::endl;
    fout << std::endl;

    fout << "  # -------------- using absolute traveltime data --------------" << std::endl;
    fout << "  abs_time:" << std::endl;
    fout << "    use_abs_time : " << use_abs_reloc << " # 'yes' for using absolute traveltime data to update ortime and location; 'no' for not using (no need to set parameters in this section)" << std::endl;
    fout << "    residual_weight : [";
    for (int i = 0; i < n_weight; i++){
        fout << residual_weight_abs_reloc[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "]      # XXX is the absolute traveltime residual (second) = abs(t^{obs}_{n,i} - t^{syn}_{n,j})" << std::endl;
    fout << "    distance_weight : [";
    for (int i = 0; i < n_weight; i++){
        fout << distance_weight_abs_reloc[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "]      # XXX is epicenter distance (km) between the source and receiver related to the data" << std::endl;
    fout << std::endl;

    fout << "  # -------------- using common source differential traveltime data --------------" << std::endl;
    fout << "  cs_dif_time:" << std::endl;
    fout << "    use_cs_time : " << use_cs_reloc <<" # 'yes' for using common source differential traveltime data to update ortime and location; 'no' for not using (no need to set parameters in this section)" << std::endl;
    fout << "    residual_weight  : [";
    for (int i = 0; i < n_weight; i++){
        fout << residual_weight_cs_reloc[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "]    # XXX is the common source differential traveltime residual (second) = abs(t^{obs}_{n,i} - t^{obs}_{n,j} - t^{syn}_{n,i} + t^{syn}_{n,j})." << std::endl;
    fout << "    azimuthal_weight : [";
    for (int i = 0; i < n_weight; i++){
        fout << azimuthal_weight_cs_reloc[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "]    # XXX is the azimuth difference between two separate stations related to the common source." << std::endl;
    fout << std::endl;

    fout << "  # -------------- using common receiver differential traveltime data --------------" << std::endl;
    fout << "  cr_dif_time:" << std::endl;
    fout << "    use_cr_time : " << use_cr_reloc <<" # 'yes' for using common receiver differential traveltime data to update ortime and location; 'no' for not using (no need to set parameters in this section)" << std::endl;
    fout << "    residual_weight  : [";
    for (int i = 0; i < n_weight; i++){
        fout << residual_weight_cr_reloc[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "]    # XXX is the common receiver differential traveltime residual (second) = abs(t^{obs}_{n,i} - t^{obs}_{m,i} - t^{syn}_{n,i} + t^{syn}_{m,i})" << std::endl;
    fout << "    azimuthal_weight : [";
    for (int i = 0; i < n_weight; i++){
        fout << azimuthal_weight_cr_reloc[i];
        if (i != n_weight-1)
            fout << ", ";
    }
    fout << "]    # XXX is the azimuth difference between two separate sources related to the common receiver." << std::endl;
    fout << std::endl;

    fout << std::endl;
    fout << "  # -------------- global weight of different types of data (to balance the weight of different data) --------------" << std::endl;
    fout << "  global_weight:" << std::endl;
    fout << "    balance_data_weight: " << balance_data_weight_reloc << " # yes: over the total weight of the each type of the data. no: use original weight (below weight for each type of data needs to be set)" << std::endl;
    fout << "    abs_time_local_weight: " << abs_time_local_weight_reloc << " # weight of absolute traveltime data for relocation after balance,     default: 1.0" << std::endl;
    fout << "    cs_dif_time_local_weight: " << cs_dif_time_local_weight_reloc << " # weight of common source differential traveltime data for relocation after balance,   default: 1.0" << std::endl;
    fout << "    cr_dif_time_local_weight: " << cr_dif_time_local_weight_reloc << " # weight of common receiver differential traveltime data for relocation after balance,   default: 1.0" << std::endl;
    fout << std::endl;


    fout << "####################################################################" << std::endl;
    fout << "#          inversion strategy for tomography and relocation        #" << std::endl;
    fout << "####################################################################" << std::endl;
    fout << "inversion_strategy: # update model parameters and earthquake hypocenter iteratively (when run_mode : 3)" << std::endl;
    fout << std::endl;
    fout << "  inv_mode : " << inv_mode << " # 0 for update model parameters and relocation iteratively. 1 for update model parameters and relocation simultaneously." << std::endl;
    fout << std::endl;
    fout << "  # for inv_mode : 0, parameters below are required" << std::endl;
    fout << "  inv_mode_0: # update model for <model_update_N_iter> steps, then update location for <relocation_N_iter> steps, and repeat the process for <max_loop> loops." << std::endl;
    fout << "    model_update_N_iter : " << model_update_N_iter << std::endl;
    fout << "    relocation_N_iter : " << relocation_N_iter << std::endl;
    fout << "    max_loop : " << max_loop_mode0 << std::endl;
    fout << std::endl;

    fout << "  # for inv_mode : 1, parameters below are required" << std::endl;
    fout << "  inv_mode_1: # update model and location simultaneously for <max_loop> loops." << std::endl;
    fout << "    max_loop : " << max_loop_mode1 << std::endl;
    fout << std::endl;

    fout << "# keep these setting unchanged, unless you are familiar with the eikonal solver in this code" << std::endl;
    fout << "calculation:" << std::endl;
    fout << "   convergence_tolerance: " << conv_tol << " # threshold value for checking the convergence for each forward/adjoint run"<< std::endl;
    fout << "   max_iterations: " << max_iter << " # number of maximum iteration for each forward/adjoint run" << std::endl;
    fout << "   stencil_order: " << stencil_order << " # order of stencil, 1 or 3" << std::endl;
    fout << "   stencil_type: " << stencil_type << " # 0: , 1: first-order upwind scheme (only sweep_type 0 is supported) " << std::endl;
    fout << "   sweep_type: " << sweep_type << " # 0: legacy, 1: cuthill-mckee with shm parallelization" << std::endl;
    fout << std::endl;
    //fout << std::endl;
    //fout << "debug:" << std::endl;
    //fout << "   debug_mode: " << int(if_test) << std::endl;


}

void InputParams::setup_uniform_inv_grid() {
    // set the number of inversion grid points
    if (uniform_inv_grid_dep){
        if (min_dep_inv == -9999.0 || max_dep_inv == -9999.0){
            std::cout << "Error: please setup min_max_dep_inv" << std::endl;
            exit(1);
        }
        if (max_dep_inv < min_dep_inv){
            std::cout << "Error: max_dep_inv should be larger than min_dep_inv" << std::endl;
            exit(1);
        }

        n_inv_r_flex = n_inv_r;

        std::vector<CUSTOMREAL> tmp_dep_inv = linspace(min_dep_inv, max_dep_inv, n_inv_r_flex);

        if (dep_inv != nullptr) delete[] dep_inv;

        dep_inv = allocateMemory<CUSTOMREAL>(n_inv_r_flex, 5000);

        for (int i = 0; i < n_inv_r_flex; i++){
            dep_inv[i] = tmp_dep_inv[i];
        }

        if (invgrid_ani) {
            if (min_dep_inv_ani == -9999.0 || max_dep_inv_ani == -9999.0){
                std::cout << "Error: please setup min_max_dep_inv_ani" << std::endl;
                exit(1);
            }
            if (max_dep_inv_ani < min_dep_inv_ani){
                std::cout << "Error: max_dep_inv_ani should be larger than min_dep_inv_ani" << std::endl;
                exit(1);
            }

            n_inv_r_flex_ani = n_inv_r_ani;

            if (dep_inv_ani != nullptr) delete[] dep_inv_ani;

            dep_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_r_flex_ani, 5003);

            tmp_dep_inv = linspace(min_dep_inv_ani, max_dep_inv_ani, n_inv_r_flex_ani);

            for (int i = 0; i < n_inv_r_flex_ani; i++){
                dep_inv_ani[i] = tmp_dep_inv[i];
            }
        }
        tmp_dep_inv.clear();

    }


    if (uniform_inv_grid_lat){
        if (min_lat_inv == -9999.0 || max_lat_inv == -9999.0){
            std::cout << "Error: please setup min_max_lat_inv_ani" << std::endl;
            exit(1);
        }
        if (max_lat_inv < min_lat_inv){
            std::cout << "Error: max_lat_inv should be larger than min_lat_inv" << std::endl;
            exit(1);
        }
        n_inv_t_flex = n_inv_t;

        std::vector<CUSTOMREAL> tmp_lat_inv = linspace(min_lat_inv, max_lat_inv, n_inv_t_flex);

        if (lat_inv != nullptr) delete[] lat_inv;

        lat_inv = allocateMemory<CUSTOMREAL>(n_inv_t_flex, 5001);

        for (int i = 0; i < n_inv_t_flex; i++){
            lat_inv[i] = tmp_lat_inv[i];
        }

        if (invgrid_ani) {
            if (min_lat_inv_ani == -9999.0 || max_lat_inv_ani == -9999.0){
                std::cout << "Error: please setup min_max_lat_inv_ani" << std::endl;
                exit(1);
            }
            if (max_lat_inv_ani < min_lat_inv_ani){
                std::cout << "Error: max_lat_inv_ani should be larger than min_lat_inv_ani" << std::endl;
                exit(1);
            }

            n_inv_t_flex_ani = n_inv_t_ani;

            if (lat_inv_ani != nullptr) delete[] lat_inv_ani;

            lat_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_t_flex_ani, 5004);

            tmp_lat_inv = linspace(min_lat_inv_ani, max_lat_inv_ani, n_inv_t_flex_ani);

            for (int i = 0; i < n_inv_t_flex_ani; i++){
                lat_inv_ani[i] = tmp_lat_inv[i];
            }
        }
        tmp_lat_inv.clear();
    }

    if (uniform_inv_grid_lon){
        if (min_lon_inv == -9999.0 || max_lon_inv == -9999.0){
            std::cout << "Error: please setup min_max_lon_inv_ani" << std::endl;
            exit(1);
        }
        if (max_lon_inv < min_lon_inv){
            std::cout << "Error: max_lon_inv should be larger than min_lon_inv" << std::endl;
            exit(1);
        }
        n_inv_p_flex = n_inv_p;

        std::vector<CUSTOMREAL> tmp_lon_inv = linspace(min_lon_inv, max_lon_inv, n_inv_p_flex);

        if (lon_inv != nullptr) delete[] lon_inv;

        lon_inv = allocateMemory<CUSTOMREAL>(n_inv_p_flex, 5002);

        for (int i = 0; i < n_inv_p_flex; i++){
            lon_inv[i] = tmp_lon_inv[i];
        }

        if (invgrid_ani) {
            if (min_lon_inv_ani == -9999.0 || max_lon_inv_ani == -9999.0){
                std::cout << "Error: please setup min_max_lon_inv_ani" << std::endl;
                exit(1);
            }
            if (max_lon_inv_ani < min_lon_inv_ani){
                std::cout << "Error: max_lon_inv_ani should be larger than min_lon_inv_ani" << std::endl;
                exit(1);
            }

            n_inv_p_flex_ani = n_inv_p_ani;

            if (lon_inv_ani != nullptr) delete[] lon_inv_ani;

            lon_inv_ani = allocateMemory<CUSTOMREAL>(n_inv_p_flex_ani, 5005);

            tmp_lon_inv = linspace(min_lon_inv_ani, max_lon_inv_ani, n_inv_p_flex_ani);

            for (int i = 0; i < n_inv_p_flex_ani; i++){
                lon_inv_ani[i] = tmp_lon_inv[i];
            }

        }
        tmp_lon_inv.clear();
    }
}



// return radious
CUSTOMREAL InputParams::get_src_radius(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return depth2radius(get_src_point_bcast(name_sim_src).dep);
    else
        return depth2radius(src_dep);
}


CUSTOMREAL InputParams::get_src_lat(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return get_src_point_bcast(name_sim_src).lat*DEG2RAD;
    else
        return src_lat*DEG2RAD;
}


CUSTOMREAL InputParams::get_src_lon(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return get_src_point_bcast(name_sim_src).lon*DEG2RAD;
    else
        return src_lon*DEG2RAD;
}

CUSTOMREAL InputParams::get_src_radius_2d(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return depth2radius(get_src_point_bcast_2d(name_sim_src).dep);
    else
        return depth2radius(src_dep);
}


CUSTOMREAL InputParams::get_src_lat_2d(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return get_src_point_bcast_2d(name_sim_src).lat*DEG2RAD;
    else
        return src_lat*DEG2RAD;
}


CUSTOMREAL InputParams::get_src_lon_2d(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return get_src_point_bcast_2d(name_sim_src).lon*DEG2RAD;
    else
        return src_lon*DEG2RAD;
}


SrcRecInfo& InputParams::get_src_point(const std::string& name_src){
    if (proc_store_srcrec)
        return src_map[name_src];
    else  {
        // exit with error
        std::cout << "Error: non-proc_store_srcrec process should not call the function get_src_point." << std::endl;
        exit(1);
    }
}


SrcRecInfo& InputParams::get_rec_point(const std::string& name_rec){
    if (proc_store_srcrec)
        return rec_map[name_rec];
    else  {
        // exit with error
        std::cout << "Error: non-proc_store_srcrec process should not call the function get_rec_point." << std::endl;
        exit(1);
    }
}


SrcRecInfo InputParams::get_src_point_bcast(const std::string& name_src){
    //
    // This function returns copy of a SrcRecInfo object
    // thus modifying the returned object will not affect the original object
    //

    SrcRecInfo src_tmp;

    if (proc_store_srcrec)
        src_tmp = src_map[name_src];

    // broadcast
    broadcast_src_info_intra_sim(src_tmp, 0);

    // return
    return src_tmp;

}


SrcRecInfo InputParams::get_src_point_bcast_2d(const std::string& name_src){
    //
    // This function returns copy of a SrcRecInfo object
    // thus modifying the returned object will not affect the original object
    //

    SrcRecInfo src_tmp;

    if (proc_store_srcrec)
        src_tmp = src_map_2d[name_src];

    // broadcast
    broadcast_src_info_intra_sim(src_tmp, 0);

    // return
    return src_tmp;

}





SrcRecInfo InputParams::get_rec_point_bcast(const std::string& name_rec) {
    //
    // This function returns copy of a SrcRecInfo object
    // thus modifying the returned object will not affect the original object
    //

    SrcRecInfo rec_tmp;

    if (proc_store_srcrec)
        rec_tmp = rec_map[name_rec];

    // broadcast
    broadcast_rec_info_intra_sim(rec_tmp, 0);

    return rec_tmp;

}


// return source name from in-sim_group id
std::string InputParams::get_src_name(const int& local_id){

    std::string src_name;
    if (proc_store_srcrec)
        src_name = src_id2name[local_id];

    // broadcast
    broadcast_str(src_name, 0);

    return src_name;
}

std::string InputParams::get_src_name_comm(const int& local_id){

    std::string src_name;
    if (proc_store_srcrec)
        src_name = src_id2name_comm_rec[local_id];

    // broadcast
    broadcast_str(src_name, 0);

    return src_name;
}




std::string InputParams::get_rec_name(const int& local_id){

    std::string rec_name;
    if (proc_store_srcrec)
        rec_name = rec_id2name[local_id];

    // broadcast
    broadcast_str(rec_name, 0);

    return rec_name;
}


// return src global id from src name
int InputParams::get_src_id(const std::string& src_name) {
    int src_id;
    if (proc_store_srcrec)
        src_id = src_map[src_name].id;

    // broadcast
    broadcast_i_single(src_id, 0);

    return src_id;
}


bool InputParams::get_if_src_teleseismic(const std::string& src_name) {
    bool if_src_teleseismic = false;

    if (proc_store_srcrec)
        if_src_teleseismic = get_src_point(src_name).is_out_of_region;

    // broadcast to all processes within simultaneous run group
    broadcast_bool_single(if_src_teleseismic, 0);
    broadcast_bool_single_sub(if_src_teleseismic, 0);

    return if_src_teleseismic;
}


bool InputParams::get_is_T_written_into_file(const std::string& src_name) {
    bool is_T_written_into_file = false;

    if (proc_store_srcrec)
        is_T_written_into_file = get_src_point(src_name).is_T_written_into_file;

    // broadcast to all processes within simultaneous run group
    broadcast_bool_single(is_T_written_into_file, 0);
    broadcast_bool_single_sub(is_T_written_into_file, 0);

    return is_T_written_into_file;
}


void InputParams::prepare_src_map(){
    //
    // only the
    // - subdom_main process of the
    // - first subdomain of the
    // - first simultaneous run group
    // (subdom_main==true && id_subdomain==0 && id_sim==0) reads src/rec file
    // and stores entile src/rec list in src_points and rec_points
    // then, the subdom_main process of each simultaneous run group (id_sim==any && subdom_main==true) retains only its own src/rec objects,
    // which are actually calculated in those simultaneous run groups
    //

    // assigne processor roles
    proc_read_srcrec  = (subdom_main && id_subdomain==0 && id_sim==0); // process which reads src/rec file (only one process per entire simulation)
    proc_store_srcrec = (subdom_main && id_subdomain==0); // process which stores src/rec objects (only one process per simultaneous run group)

    // read src rec file only
    if (src_rec_file_exist && proc_read_srcrec) {

        // read source and receiver data, e.g.,
        //      event info:               s0
        //      data info1 (abs):         r0
        //      data info2 (cr_dif):      r1  r2
        //      data info3 (cs_dif):      r3  s1
        //
        // we generate the data structure:
        // |    abs     |    cs_dif     |    cr_dif     |
        // |  s0 - r0   |   s0 - r1     |   s0 - r3     |
        // |            |   |           |        |      |
        // |            |   r2          |        s1     |
        parse_src_rec_file(src_rec_file, \
                           src_map_all, \
                           rec_map_all, \
                           data_map_all, \
                           src_id2name_all, \
                           rec_id2name_back);

        // read station correction file by all processes
        if (sta_correction_file_exist) {
            // store all src/rec info
            parse_sta_correction_file(sta_correction_file,
                                      rec_map_all);
        }

        // copy backups (KEEPED AS THE STATE BEFORE SWAPPING SRC AND REC)
        src_map_back     = src_map_all;
        rec_map_back     = rec_map_all;
        data_map_back    = data_map_all;
        src_id2name_back = src_id2name_all;

        // check if src positions are within the domain or not (teleseismic source)
        // detected teleseismic source is separated into tele_src_points and tele_rec_points

        std::cout << std::endl << "separate regional and teleseismic src/rec points" << std::endl;

        separate_region_and_tele_src_rec_data(src_map_back, rec_map_back, data_map_back,
                                              src_map_all,  rec_map_all,  data_map_all,
                                              src_map_tele, rec_map_tele, data_map_tele,
                                              data_type,
                                              N_abs_local_data,
                                              N_cr_dif_local_data,
                                              N_cs_dif_local_data,
                                              N_teleseismic_data,
                                              N_data,
                                              min_lat, max_lat, min_lon, max_lon, min_dep, max_dep,
                                              have_tele_data);

        if (swap_src_rec) {
            // we swap the source and receviver for regional earthquakes. After that, we have new data structure:
            // Before:
            // |    abs     |    cs_dif     |    cr_dif     |
            // |  s0 - r0   |   s0 - r1     |   s0 - r3     |
            // |            |   |           |        |      |
            // |            |   r2          |        s1     |
            //
            // After:
            // |    abs     |          cr_dif           |    cs_dif     |
            // |  r0 - s0   |   r1 - s0     r2 - s0     |   r3 - s0     |
            // |            |        |           |      |   |           |
            // |            |        r2          r1     |   s1          |
            stdout_by_main("Swapping src and rec. This may take few minutes for a large dataset (only regional events will be processed)\n");
            do_swap_src_rec(src_map_all, rec_map_all, data_map_all, src_id2name_all);
            int tmp = N_cr_dif_local_data;
            N_cr_dif_local_data = N_cs_dif_local_data;
            N_cs_dif_local_data = tmp;
        } else {
            // if we do not swap source and receiver, we need to process cr_dif to include the other source. After that, we have new data structure:
            // Before:
            // |    abs     |    cs_dif     |    cr_dif     |
            // |  s0 - r0   |   s0 - r1     |   s0 - r3     |
            // |            |   |           |        |      |
            // |            |   r2          |        s1     |
            //
            // After:
            // |    abs     |    cs_dif     |          cr_dif           |
            // |  s0 - r0   |   s0 - r1     |   s0 - r3     s1 - r3     |
            // |            |   |           |        |           |      |
            // |            |   r2          |        s1          s0     |
            do_not_swap_src_rec(src_map_all, rec_map_all, data_map_all, src_id2name_all);
        }

        // concatenate resional and teleseismic src/rec points
        //
        // src_map  = src_map_all  + src_map_tele
        // rec_map  = rec_map_all  + rec_map_tele
        // data_map = data_map_all + data_map_tele
        // *_map_tele will be empty after this function

        std::cout << std::endl << "merge regional and teleseismic src/rec points" << std::endl;

        merge_region_and_tele_src(src_map_all,  rec_map_all,  data_map_all, src_id2name_all,
                                  src_map_tele, rec_map_tele, data_map_tele);

        // abort if number of src_points are less than n_sims
        int n_src_points = src_map_all.size();
        if (n_src_points < n_sims){
            std::cout << "Error: number of sources in src_rec_file is less than n_sims. Abort." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

    } // end of if (src_rec_file_exist && proc_read_srcrec)

    // wait
    synchronize_all_world();


    // to all the subdom_main processes of each simultaneous run group
    if (src_rec_file_exist) {

        if (world_rank==0)
            std::cout << "\nsource assign to simultaneous run groups\n" <<std::endl;

        // divide and distribute the data below to each simultaneous run group:
        //  src_map,     this will be used for checking global id in source iteration
        //  rec_map,
        //  data_map,
        //  src_id2name, this will be used for source iteration
        //  rec_id2name, this will be used for adjoint source iteration
        distribute_src_rec_data(src_map_all,
                                rec_map_all,
                                data_map_all,
                                src_id2name_all,
                                src_map,
                                rec_map,
                                data_map,
                                src_id2name,
                                rec_id2name);

        // now src_id2name_all  includes  all src names of after swapping src and rec
        //     src_id2name      includes only src names of this simultaneous run group
        //     src_id2name_back includes only src names of this simultaneous run group before swapping src and rec

        if (world_rank==0)
            std::cout << "\ngenerate src map with common receiver\n" <<std::endl;

        // create source list for common receiver double difference traveltime
        generate_src_map_with_common_receiver(data_map, src_map_comm_rec, src_id2name_comm_rec);

        if (world_rank==0)
            std::cout << "\nprepare src map for 2d solver\n" <<std::endl;

        // prepare source list for teleseismic source
        prepare_src_map_for_2d_solver(src_map_all, src_map, src_id2name_2d, src_map_2d);

        synchronize_all_world();

        // count the number of sources in this simultaneous run group
        if (proc_store_srcrec) {
            n_src_this_sim_group          = (int) src_id2name.size();
            n_src_comm_rec_this_sim_group = (int) src_id2name_comm_rec.size();
            n_src_2d_this_sim_group       = (int) src_id2name_2d.size();
            n_rec_this_sim_group          = (int) rec_id2name.size();
        }
        // broadcast the number of sources to all the processes in this simultaneous run group
        broadcast_i_single_intra_sim(n_src_this_sim_group, 0);
        broadcast_i_single_intra_sim(n_src_comm_rec_this_sim_group, 0);
        broadcast_i_single_intra_sim(n_src_2d_this_sim_group, 0);
        broadcast_i_single_intra_sim(n_rec_this_sim_group, 0);

        if (world_rank==0)
            std::cout << "end parse src_rec file" << std::endl;

    } // end of if src_rec_file_exists
}


// generate a list of events which involve common receiver double difference traveltime
void InputParams::generate_src_map_with_common_receiver(std::map<std::string, std::map<std::string, std::vector<DataInfo>>>& data_map_tmp,
                                                        std::map<std::string, SrcRecInfo>&                                   src_map_comm_rec_tmp,
                                                        std::vector<std::string>&                                            src_id2name_comm_rec_tmp){

    if (proc_store_srcrec) {

        // for earthquake having common receiver differential traveltime, the synthetic traveltime should be computed first at each iteration
        for(auto iter = data_map_tmp.begin(); iter != data_map_tmp.end(); iter++){
            for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++){
                for (auto& data: iter2->second){
                    if (data.is_src_pair){
                        // add this source and turn to the next source
                        src_map_comm_rec_tmp[iter->first] = src_map[iter->first];
                        // add this source to the list of sources that will be looped in each iteration
                        // if the source is not in the list, the synthetic traveltime will not be computed
                        if (std::find(src_id2name_comm_rec_tmp.begin(), src_id2name_comm_rec_tmp.end(), iter->first) == src_id2name_comm_rec_tmp.end())
                            src_id2name_comm_rec_tmp.push_back(iter->first);

                        break;
                    }
                }
            }
        }

        // check if this sim group has common source double difference traveltime
        if (src_map_comm_rec_tmp.size() > 0){
            src_pair_exists = true;
        }

    } // end of if (proc_store_srcrec)

    // flag if any src_pair exists
    allreduce_bool_inplace_inter_sim(&src_pair_exists, 1); // inter-sim
    allreduce_bool_inplace(&src_pair_exists, 1); // intra-sim / inter subdom
    allreduce_bool_inplace_sub(&src_pair_exists, 1); // intra-subdom

}

void InputParams::initialize_adjoint_source(){
    // this funtion should be called by proc_store_srcrec
    if (!proc_store_srcrec){
        std::cout << "initialize_adjoint_source function is called non-proc_store_srcrec process. aborting." << std::endl;
        exit(1);
    }

    for(auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
        iter->second.adjoint_source = _0_CR;
        iter->second.adjoint_source_density = _0_CR;
    }
}

void InputParams::set_adjoint_source(std::string name_rec, CUSTOMREAL adjoint_source){

    // this funtion should be called by proc_store_srcrec
    if (!proc_store_srcrec){
        std::cout << "set_adjoint_source function is called non-proc_store_srcrec process. aborting." << std::endl;
        exit(1);
    }

    if (rec_map.find(name_rec) != rec_map.end()){
        rec_map[name_rec].adjoint_source = adjoint_source;
    } else {
        std::cout << "error !!!, undefined receiver name when adding adjoint source: " << name_rec << std::endl;
    }
}

void InputParams::set_adjoint_source_density(std::string name_rec, CUSTOMREAL adjoint_source_density){

    // this funtion should be called by proc_store_srcrec
    if (!proc_store_srcrec){
        std::cout << "set_adjoint_source_density function is called non-proc_store_srcrec process. aborting." << std::endl;
        exit(1);
    }

    if (rec_map.find(name_rec) != rec_map.end()){
        rec_map[name_rec].adjoint_source_density = adjoint_source_density;
    } else {
        std::cout << "error !!!, undefined receiver name when adding adjoint source: " << name_rec << std::endl;
    }
}

// gather all arrival times to main simultaneous run group
// common source double difference traveltime is also gathered here
// then store them in data_map_all
void InputParams::gather_all_arrival_times_to_main(){

    if (proc_store_srcrec) {

        for (int id_src = 0; id_src < nsrc_total; id_src++){

            // id of simulation group for this source
            int id_sim_group = select_id_sim_for_src(id_src, n_sims);

            // broadcast source name
            std::string name_src;
            if (proc_read_srcrec){
                name_src = src_id2name_all[id_src];
            }
            broadcast_str_inter_sim(name_src, 0);

            if (id_sim_group==0) { // the simultaneous run group == 0 is the main simultaneous run group
                if (id_sim == 0) {
                    // copy arrival time to data_info_back
                    for (auto iter = data_map[name_src].begin(); iter != data_map[name_src].end(); iter++){
                        for(int i_data = 0; i_data < (int)iter->second.size(); i_data++){
                            // store travel time in all datainfo element of each src-rec pair
                            data_map_all[name_src][iter->first].at(i_data).travel_time = iter->second.at(i_data).travel_time;
                            // common source double difference traveltime
                            data_map_all[name_src][iter->first].at(i_data).cs_dif_travel_time = iter->second.at(i_data).cs_dif_travel_time;
                        }
                    }
                } else {
                    // do nothing
                }
            } else { // this source is calculated in the other simultaneous run group than the main

                // the main group receives the arrival time from the other simultaneous run group
                if (id_sim == 0) {
                    // number of receivers
                    int nrec;
                    recv_i_single_sim(&nrec, id_sim_group);

                    // loop over receivers
                    for (int id_rec = 0; id_rec < nrec; id_rec++){
                        // receive name of receiver
                        std::string name_rec;
                        recv_str_sim(name_rec, id_sim_group); ///////

                        // receive number of data
                        int ndata;
                        recv_i_single_sim(&ndata, id_sim_group);

                        // exit if the number of data is not the same with data_map_all
                        if (ndata != (int)data_map_all[name_src][name_rec].size()){
                            std::cout << "ERROR: the number of data calculated sub simultaneous group is not the same with data_map_all" << std::endl;
                            std::cout << "name_src = " << name_src << ", name_rec = " << name_rec << std::endl;
                            std::cout << "ndata = " << ndata << ", data_map_all[name_src][name_rec].size() = " << data_map_all[name_src][name_rec].size() << std::endl;
                            exit(1);   // for rec_3 src_0, ndata = 1   data_map_all[][].size() = 3
                        }

                        // then receive travel time (and common source double difference traveltime)
                        for (auto& data: data_map_all[name_src][name_rec]) {
                            recv_cr_single_sim(&(data.travel_time), id_sim_group);
                            recv_cr_single_sim(&(data.cs_dif_travel_time), id_sim_group);
                        }
                    }

                // the non-main simultaneous run group sends the arrival time to the main group
                } else if (id_sim == id_sim_group) {
                    // send number of receivers
                    int nrec = data_map[name_src].size();
                    send_i_single_sim(&nrec, 0);

                    for (auto iter = data_map[name_src].begin(); iter != data_map[name_src].end(); iter++){
                        // send name of receiver
                        send_str_sim(iter->first, 0);

                        // send number of data
                        int ndata = iter->second.size();
                        send_i_single_sim(&ndata, 0);

                        // then send travel time (and common source double difference traveltime)
                        for (auto& data: iter->second){
                            send_cr_single_sim(&(data.travel_time), 0);
                            send_cr_single_sim(&(data.cs_dif_travel_time), 0);
                        }
                    }
                } else {
                    // do nothing
                }
            }


        } // end for id_src

    } // end if (proc_store_srcrec)

    synchronize_all_world();
}


// gather tau_opt to main simultaneous run group
void InputParams::gather_rec_info_to_main(){

    if (proc_store_srcrec) {

        // broadcast total number of recenver to all procs
        int nrec_total = rec_map_all.size();
        broadcast_i_single_inter_sim(nrec_total, 0);

        std::vector<std::string> name_rec_all;

        if (id_sim==0){
            // assigne tau_opt to rec_map_all from its own rec_map
            for (auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
                rec_map_all[iter->first].tau_opt = iter->second.tau_opt;
                rec_map_all[iter->first].dep = iter->second.dep;
                rec_map_all[iter->first].lat = iter->second.lat;
                rec_map_all[iter->first].lon = iter->second.lon;
            }

            // make a list of receiver names
            for (auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
                name_rec_all.push_back(iter->first);
            }
        }

        for (int irec =  0; irec < nrec_total; irec++){

            // broadcast name of receiver
            std::string name_rec;
            if (id_sim==0){
                name_rec = name_rec_all[irec];
            }
            broadcast_str_inter_sim(name_rec, 0);

            int        rec_counter = 0;
            CUSTOMREAL tau_tmp=0.0;
            CUSTOMREAL dep_tmp=0.0;
            CUSTOMREAL lat_tmp=0.0;
            CUSTOMREAL lon_tmp=0.0;

            // copy value if rec_map[name_rec] exists
            if (rec_map.find(name_rec) != rec_map.end()){
                tau_tmp = rec_map[name_rec].tau_opt;
                dep_tmp = rec_map[name_rec].dep;
                lat_tmp = rec_map[name_rec].lat;
                lon_tmp = rec_map[name_rec].lon;
                rec_counter = 1;
            }

            // reduce counter and tau_tmp
            allreduce_rec_map_var(rec_counter);
            allreduce_rec_map_var(tau_tmp);
            allreduce_rec_map_var(dep_tmp);
            allreduce_rec_map_var(lat_tmp);
            allreduce_rec_map_var(lon_tmp);

            // assign tau_opt to rec_map_all
            if (rec_counter > 0){
                rec_map_all[name_rec].tau_opt = tau_tmp / (CUSTOMREAL)rec_counter;
                rec_map_all[name_rec].dep = dep_tmp / (CUSTOMREAL)rec_counter;
                rec_map_all[name_rec].lat = lat_tmp / (CUSTOMREAL)rec_counter;
                rec_map_all[name_rec].lon = lon_tmp / (CUSTOMREAL)rec_counter;
            }

        } // end for irec
    } // end of proc_store_srcrec
}

// gather traveltimes and calculate synthetic common receiver differential traveltime
void InputParams::gather_traveltimes_and_calc_syn_diff(){

    if (!src_pair_exists) return; // nothing to share

    // gather all synthetic traveltimes to main simultaneous run group
    gather_all_arrival_times_to_main();

    if(subdom_main) {

        int mpi_tag_send=9999;
        int mpi_tag_end=9998;

        // main process calculates differences of synthetic data and send them to other processes
        if (id_sim==0){
            // int n_total_src_pair = 0;

            // calculate differences of synthetic data
            for (auto iter = data_map_all.begin(); iter != data_map_all.end(); iter++){
                for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++){
                    for (auto& data: iter2->second){
                        if (data.is_src_pair){
                            data.cr_dif_travel_time = data_map_all[data.name_src_pair[0]][data.name_rec].at(0).travel_time \
                                                    - data_map_all[data.name_src_pair[1]][data.name_rec].at(0).travel_time;
                            // n_total_src_pair++;
                       }
                    }
                }
            }

            // send differences of synthetic data to other processes
            for (int id_src = 0; id_src < nsrc_total; id_src++){ // MNMN: looping over all of data.id_src_pair[0]

                // id of simulation group for this source
                int id_sim_group = select_id_sim_for_src(id_src, n_sims);

                std::string name_src = src_id2name_all[id_src]; // list of src names after swap

                // iterate over receivers
                for (auto iter = data_map_all[name_src].begin(); iter != data_map_all[name_src].end(); iter++){
                    std::string name_rec = iter->first;

                    // iterate over data
                    for (int i_data = 0; i_data < (int)iter->second.size(); i_data++){
                        auto& data = iter->second.at(i_data);

                        if (data.is_src_pair){
                            std::string name_src1 = data.name_src_pair[0];
                            std::string name_src2 = data.name_src_pair[1];
                            // name_src1 should be the same as name_src for set_cr_dif_to_src_pair (replace otherwise)
                            if (name_src1 != name_src){
                                std::string tmp = name_src1;
                                name_src1 = name_src2;
                                name_src2 = tmp;
                            }

                            if (id_sim_group == 0) {
                                // this source is calculated in the main simultaneous run group
                                set_cr_dif_to_src_pair(data_map, name_src1, name_src2, name_rec, data.cr_dif_travel_time);
                            } else {
                                // send signal with dummy int
                                int dummy = 0;
                                MPI_Send(&dummy, 1, MPI_INT, id_sim_group, mpi_tag_send, inter_sim_comm);

                                // send name_src1
                                send_str_sim(name_src1, id_sim_group);
                                // send name_src2
                                send_str_sim(name_src2, id_sim_group);
                                // send name_rec
                                send_str_sim(name_rec, id_sim_group);
                                // send index of data
                                send_i_single_sim(&i_data, id_sim_group);
                                // send travel time difference
                                send_cr_single_sim(&(data.cr_dif_travel_time), id_sim_group);
                            }
                        }
                    }
                }
            } // end for id_src

            // send end signal (start from 1 because 0 is main process)
            for (int id_sim = 1; id_sim < n_sims; id_sim++){
                // send dummy integer
                int dummy = 0;
                MPI_Send(&dummy, 1, MPI_INT, id_sim, mpi_tag_end, inter_sim_comm);
            }

        // un-main process receives differences of synthetic data from main process
        } else if (id_sim!=0) {
            while (true) {

                // wait with mpi probe
                MPI_Status status;
                MPI_Probe(0, MPI_ANY_TAG, inter_sim_comm, &status);

                // receive signal with dummy int
                int dummy = 0;
                MPI_Recv(&dummy, 1, MPI_INT, 0, MPI_ANY_TAG, inter_sim_comm, &status);

                // if this signal is for sending data
                if (status.MPI_TAG == mpi_tag_send) {

                    std::string name_src1, name_src2, name_rec;

                    // receive src_name1
                    recv_str_sim(name_src1, 0);
                    // receive src_name2
                    recv_str_sim(name_src2, 0);
                    // receive rec_name
                    recv_str_sim(name_rec, 0);
                    // receive index of data
                    int i_data = 0;
                    recv_i_single_sim(&i_data, 0);
                    // receive travel time difference
                    CUSTOMREAL tmp_ttd = 0;
                    recv_cr_single_sim(&(tmp_ttd), 0);
                    set_cr_dif_to_src_pair(data_map, name_src1, name_src2, name_rec, tmp_ttd);


                // if this signal is for terminating the wait loop
                } else if (status.MPI_TAG == mpi_tag_end) {
                    break;
                }

            }
        }
    }

   synchronize_all_world();

}



void InputParams::write_station_correction_file(int i_inv){
    if(use_sta_correction && run_mode == DO_INVERSION) {  // if apply station correction
        station_correction_file_out = output_dir + "/station_correction_file_step_" + int2string_zero_fill(i_inv) +".dat";

        std::ofstream ofs;

        if (proc_read_srcrec){    // main processor of subdomain && the first id of subdoumains

            ofs.open(station_correction_file_out);

            ofs << "# stname " << "   lat   " << "   lon   " << "elevation   " << " station correction (s) " << std::endl;
            for(auto iter = rec_map_back.begin(); iter != rec_map_back.end(); iter++){
                SrcRecInfo  rec      = iter->second;
                std::string name_rec = rec.name;

                // do not consider swap for teleseismic data
                CUSTOMREAL sta_correct = rec_map_all[name_rec].sta_correct;

                ofs << rec.name << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec.lat << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec.lon << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec.dep * -1000.0 << " "
                    << std::fixed << std::setprecision(6) << std::setw(9) << std::right << std::setfill(' ') << sta_correct << " "
                    << std::endl;
            }

            ofs.close();

        }
        synchronize_all_world();
    }
}

void InputParams::write_src_rec_file(int i_inv, int i_iter) {

    if (src_rec_file_exist){

        std::ofstream ofs;

        // gather all arrival time info to the main process (need to call even n_sim=1)
        gather_all_arrival_times_to_main();

        // gather tau_opt, lat, lon, dep info
        if (run_mode == SRC_RELOCATION || run_mode == INV_RELOC){
            gather_rec_info_to_main();

            // modify the source location and ortime
            if (proc_read_srcrec){
                for(auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
                    src_map_back[iter->first].lat   =   iter->second.lat;
                    src_map_back[iter->first].lon   =   iter->second.lon;
                    src_map_back[iter->first].dep   =   iter->second.dep;
                    src_map_back[iter->first].year  =   iter->second.year;
                    src_map_back[iter->first].month =   iter->second.month;
                    src_map_back[iter->first].day   =   iter->second.day;
                    src_map_back[iter->first].hour  =   iter->second.hour;
                    src_map_back[iter->first].min   =   iter->second.min;
                    src_map_back[iter->first].sec   =   iter->second.sec + iter->second.tau_opt;

                    // correct the output ortime:
                    SrcRecInfo  src      = src_map_back[iter->first];
                    // step 1, create a time stamp
                    std::tm timeInfo = {};
                    timeInfo.tm_year = src.year - 1900;
                    timeInfo.tm_mon  = src.month - 1;
                    timeInfo.tm_mday = src.day;
                    timeInfo.tm_hour = src.hour;
                    timeInfo.tm_min  = src.min;

                    // std::cout   << "before, src.hour = " <<  src.hour
                    //             << ", src.min = " <<  src.min
                    //             << ", src.sec = " <<  src.sec << std::endl;

                    if (src.sec >= - 1.0 && src.sec < 0.0)
                        timeInfo.tm_sec  = static_cast<int>(src.sec) - 1.0;
                    else
                        timeInfo.tm_sec  = static_cast<int>(src.sec);

                    // Convert to time_point
                    std::time_t rawTime = timegm(&timeInfo); // use timegm for UTC
                    std::chrono::system_clock::time_point tp = std::chrono::system_clock::from_time_t(rawTime);

                   // Get new time info
                   std::time_t timestamp = std::chrono::system_clock::to_time_t(tp);
                   std::tm newTimeInfo = *std::gmtime(&timestamp); // use gmtime to keep it in UTC

                    // step 2, correct the output ortime:
                    src_map_back[iter->first].year = newTimeInfo.tm_year + 1900;
                    src_map_back[iter->first].month = newTimeInfo.tm_mon + 1;
                    src_map_back[iter->first].day = newTimeInfo.tm_mday;
                    src_map_back[iter->first].hour = newTimeInfo.tm_hour;
                    src_map_back[iter->first].min = newTimeInfo.tm_min;

                    if (src.sec >= - 1.0 && src.sec < 0.0)
                        src_map_back[iter->first].sec = newTimeInfo.tm_sec + (src.sec + 1.0 - static_cast<int>(src.sec));
                    else
                        src_map_back[iter->first].sec = newTimeInfo.tm_sec + (src.sec       - static_cast<int>(src.sec));

                    // std::cout << "after, src.hour = " <<  src_map_back[iter->first].hour
                    //           << ", src.min = " <<  src_map_back[iter->first].min
                    //           << ", src.sec = " <<  src_map_back[iter->first].sec << std::endl;

                }
            }
        }



        // write only by the main processor of subdomain && the first id of subdoumains
        if (proc_read_srcrec){

            if (run_mode == ONLY_FORWARD)
                src_rec_file_out = output_dir + "/src_rec_file_forward.dat";
            else if (run_mode == DO_INVERSION){
                // write out source and receiver points with current inversion iteration number
                src_rec_file_out = output_dir + "/src_rec_file_step_" + int2string_zero_fill(i_inv) +".dat";
            } else if (run_mode == INV_RELOC){
                src_rec_file_out = output_dir + "/src_rec_file_inv_" + int2string_zero_fill(i_inv) +"_reloc_" + int2string_zero_fill(i_iter)+".dat";
            } else if (run_mode == TELESEIS_PREPROCESS) {
                src_rec_file_out = output_dir + "/src_rec_file_teleseis_pre.dat";
            } else if (run_mode == SRC_RELOCATION) {
                src_rec_file_out = output_dir + "/src_rec_file_reloc_" + int2string_zero_fill(i_iter)+".dat";
            } else if (run_mode == ONED_INVERSION){
                src_rec_file_out = output_dir + "/src_rec_file_step_" + int2string_zero_fill(i_inv) +".dat";
            } else {
                std::cerr << "Error: run_mode is not defined" << std::endl;
                exit(1);
            }

            // open file
            ofs.open(src_rec_file_out);

            for (int i_src = 0; i_src < (int)src_id2name_back.size(); i_src++){

                std::string name_src = src_id2name_back[i_src];
                SrcRecInfo  src      = src_map_back[name_src];
                bool        is_tele  = src.is_out_of_region; // true for teleseismic source

                // format should be the same as input src_rec_file
                // source line :  id_src year month day hour min sec lat lon dep_km mag num_recs id_event
                ofs << std::setw(7) << std::right << std::setfill(' ') <<  src.id << " "
                    << src.year << " " << src.month << " " << src.day << " "
                    << src.hour << " " << src.min   << " "
                    << std::fixed << std::setprecision(2) << std::setw(5) << std::right << std::setfill(' ') << src.sec << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.lat << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.lon << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.dep << " "
                    << std::fixed << std::setprecision(2) << std::setw(5) << std::right << std::setfill(' ') << src.mag << " "
                    << std::setw(5) << std::right << std::setfill(' ') << src.n_data << " "
                    << src.name << " "
                    << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << 1.0     // the weight of source is assigned to data
                    << std::endl;


                // iterate data lines of i_src
                for (auto& name_data : rec_id2name_back[i_src]){
                    // name_data has one receiver (r0), or a receiver pair (r0+r1), or one source and one receiver (r0+s1)

                    std::string name_rec1, name_rec2, name_src2; // receivers' (or source's) name (before swap)

                    // data type flag
                    bool src_rec_data  = false;
                    bool src_pair_data = false;
                    bool rec_pair_data = false;

                    // store reference of data to be written
                    DataInfo& data = const_cast<DataInfo&>(data_map_back[name_src][name_data.at(0)].at(0)); // dummy copy

                    if (name_data.size() == 2 && name_data.at(1) == "abs"){   // abs data
                        name_rec1 = name_data.at(0);
                        src_rec_data = true;
                        if (get_is_srcrec_swap() && !is_tele)
                            data = get_data_src_rec(data_map_all[name_rec1][name_src]);
                        else
                            data = get_data_src_rec(data_map_all[name_src][name_rec1]);

                    } else if (name_data.size() == 3 && name_data.at(2) == "cs"){  // cs_dif data
                        name_rec1 = name_data.at(0);
                        name_rec2 = name_data.at(1);
                        rec_pair_data = true;
                        if (get_is_srcrec_swap() && !is_tele)       // cs_dif data -> cr_dif data
                            data = get_data_src_pair(data_map_all, name_rec1, name_rec2, name_src);
                        else                            // cs_dif data
                            data = get_data_rec_pair(data_map_all, name_src, name_rec1, name_rec2);

                    } else if (name_data.size() == 3 && name_data.at(2) == "cr"){   // cr_dif data
                        name_rec1 = name_data.at(0);
                        name_src2 = name_data.at(1);
                        src_pair_data = true;
                        if (get_is_srcrec_swap() && !is_tele)       // cr_dif -> cs_dif
                            data = get_data_rec_pair(data_map_all, name_rec1, name_src, name_src2);
                        else                            // cr_dif
                            data = get_data_src_pair(data_map_all, name_src, name_src2, name_rec1);

                    } else{     // error data type
                        std::cerr << "Error: incorrect data type in rec_id2name_back" << std::endl;
                        exit(1);
                    }


                    // absolute traveltime data
                    if (src_rec_data){
                        SrcRecInfo  rec         = rec_map_back[name_rec1];
                        CUSTOMREAL  travel_time = data.travel_time;

                        // receiver line : id_src id_rec name_rec lat lon elevation_m phase epicentral_distance_km arival_time
                        ofs << std::setw(7) << std::right << std::setfill(' ') << src.id << " "
                            << std::setw(5) << std::right << std::setfill(' ') << rec.id << " "
                            << rec.name << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << rec.lat << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << rec.lon << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << -1.0*rec.dep*1000.0 << " "
                            << data.phase << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << travel_time << " "
                            << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                            << std::endl;

                    // common source differential traveltime
                    } else if (rec_pair_data){
                        // common source differential traveltime data
                        CUSTOMREAL  cs_dif_travel_time;

                        if (get_is_srcrec_swap() && !is_tele){ // reverse swap src and rec
                            cs_dif_travel_time = data.cr_dif_travel_time;
                        } else {// do not swap
                            cs_dif_travel_time = data.cs_dif_travel_time;
                        }

                        SrcRecInfo& rec1 = rec_map_back[name_rec1];
                        SrcRecInfo& rec2 = rec_map_back[name_rec2];

                        // receiver pair line : id_src id_rec1 name_rec1 lat1 lon1 elevation_m1 id_rec2 name_rec2 lat2 lon2 elevation_m2 phase differential_arival_time
                        ofs << std::setw(7) << std::right << std::setfill(' ') <<  src.id << " "
                            << std::setw(5) << std::right << std::setfill(' ') <<  rec1.id << " "
                            << rec1.name << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec1.lat << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec1.lon << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << -1.0*rec1.dep*1000.0 << " "
                            << std::setw(5) << std::right << std::setfill(' ') << rec2.id << " "
                            << rec2.name << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec2.lat << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec2.lon << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << -1.0*rec2.dep*1000.0 << " "
                            << data.phase << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << cs_dif_travel_time << " "
                            << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                            << std::endl;

                    // common receiver differential traveltime
                    } else if (src_pair_data){
                        // common receiver differential traveltime data
                        CUSTOMREAL  cr_dif_travel_time;

                        if (get_is_srcrec_swap() && !is_tele){ // reverse swap src and rec
                            cr_dif_travel_time = data.cs_dif_travel_time;
                        } else {// do not swap
                            cr_dif_travel_time = data.cr_dif_travel_time;
                        }

                        SrcRecInfo& rec1 = rec_map_back[name_rec1];
                        SrcRecInfo& src2 = src_map_back[name_src2];

                        // receiver pair line : id_src id_rec1 name_rec1 lat1 lon1 elevation_m1 id_rec2 name_rec2 lat2 lon2 elevation_m2 phase differential_arival_time
                        ofs << std::setw(7) << std::right << std::setfill(' ') <<  src.id << " "
                            << std::setw(5) << std::right << std::setfill(' ') <<  rec1.id << " "
                            << rec1.name << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec1.lat << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec1.lon << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << -1.0*rec1.dep*1000.0 << " "
                            << std::setw(5) << std::right << std::setfill(' ') << src2.id << " "
                            << src2.name << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src2.lat << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src2.lon << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src2.dep << " "
                            << data.phase << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << cr_dif_travel_time << " "
                            << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                            << std::endl;
                    }

                } // end of rec loop

            } // end of for (int i_src = 0; i_src < (int)src_name_list.size(); i_src++)

            // close file
            ofs.close();

            // only for source relocation, output relocated observational data for tomography
            if (run_mode == SRC_RELOCATION || run_mode == INV_RELOC) {
                if (run_mode == INV_RELOC)
                    src_rec_file_out = output_dir + "/src_rec_file_inv_" + int2string_zero_fill(i_inv) +"_reloc_" + int2string_zero_fill(i_iter)+"_obs.dat";
                else if (run_mode == SRC_RELOCATION)
                    src_rec_file_out = output_dir + "/src_rec_file_reloc_" + int2string_zero_fill(i_iter)+"_obs.dat";

                // open file
                ofs.open(src_rec_file_out);

                for (int i_src = 0; i_src < (int)src_id2name_back.size(); i_src++){

                    std::string name_src = src_id2name_back[i_src];
                    SrcRecInfo  src      = src_map_back[name_src];
                    bool        is_tele  = src.is_out_of_region; // true for teleseismic source

                    // format should be the same as input src_rec_file
                    // source line :  id_src yearm month day hour min sec lat lon dep_km mag num_recs id_event
                    ofs << std::setw(7) << std::right << std::setfill(' ') <<  src.id << " "
                        << src.year << " " << src.month << " " << src.day << " "
                        << src.hour << " " << src.min   << " "
                        << std::fixed << std::setprecision(2) << std::setw(5) << std::right << std::setfill(' ') << src.sec << " "
                        << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.lat << " "
                        << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.lon << " "
                        << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.dep << " "
                        << std::fixed << std::setprecision(2) << std::setw(5) << std::right << std::setfill(' ') << src.mag << " "
                        << std::setw(5) << std::right << std::setfill(' ') << src.n_data << " "
                        << src.name << " "
                        << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << 1.0     // the weight of source is assigned to data
                        << std::endl;

                    // iterate data lines of i_src
                    for (auto& name_data : rec_id2name_back[i_src]){
                        // name_data has one receiver (r0), or a receiver pair (r0+r1), or one source and one receiver (r0+s1)

                        std::string name_rec1, name_rec2, name_src2; // receivers' (or source's) name (before swap)

                        // data type flag
                        bool src_rec_data  = false;
                        bool src_pair_data = false;
                        bool rec_pair_data = false;

                        // store reference of data to be written
                        DataInfo& data = const_cast<DataInfo&>(data_map_back[name_src][name_data.at(0)].at(0)); // dummy copy

                        if (name_data.size() == 2 && name_data.at(1) == "abs"){   // abs data
                            name_rec1 = name_data.at(0);
                            src_rec_data = true;
                            if (get_is_srcrec_swap() && !is_tele)
                                data = get_data_src_rec(data_map_all[name_rec1][name_src]);         // valid, for relocation, sources and receivers are always swapped
                            else
                                data = get_data_src_rec(data_map_all[name_src][name_rec1]);         // invalid
                        } else if (name_data.size() == 3 && name_data.at(2) == "cs"){  // cs_dif data
                            name_rec1 = name_data.at(0);
                            name_rec2 = name_data.at(1);
                            rec_pair_data = true;
                            if (get_is_srcrec_swap() && !is_tele)       // cs_dif data -> cr_dif data
                                data = get_data_src_pair(data_map_all, name_rec1, name_rec2, name_src);     // valid, swapped
                            else                            // cs_dif data
                                data = get_data_rec_pair(data_map_all, name_src, name_rec1, name_rec2);     // invalid
                        } else if (name_data.size() == 3 && name_data.at(2) == "cr"){   // cr_dif data
                            name_rec1 = name_data.at(0);
                            name_src2 = name_data.at(1);
                            src_pair_data = true;
                            if (get_is_srcrec_swap() && !is_tele)       // cr_dif -> cs_dif
                                data = get_data_rec_pair(data_map_all, name_rec1, name_src, name_src2);     // These data are not inverted in relocation. But we need to print them out.
                            else                            // cr_dif
                                data = get_data_src_pair(data_map_all, name_src, name_src2, name_rec1);     // invalid

                        } else{     // error data type
                            std::cerr << "Error: incorrect data type in rec_id2name_back" << std::endl;
                            exit(1);
                        }


                        // original absolute traveltime data
                        if (src_rec_data){

                            // check mandatory condition for earthquake relocation
                            if (!get_is_srcrec_swap()) {
                                std::cerr << "Error: src_rec_data is not swapped in src relocation mode!" << std::endl;
                                exit(1);
                            }

                            SrcRecInfo rec             = rec_map_back[name_rec1];
                            CUSTOMREAL travel_time_obs = data.travel_time_obs - rec_map_all[name_src].tau_opt;

                            // receiver line : id_src id_rec name_rec lat lon elevation_m phase epicentral_distance_km arival_time
                            ofs << std::setw(7) << std::right << std::setfill(' ') << src.id << " "
                                << std::setw(5) << std::right << std::setfill(' ') << rec.id << " "
                                << rec.name << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << rec.lat << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << rec.lon << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << -1.0*rec.dep*1000.0 << " "
                                << data.phase << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << travel_time_obs  << " "
                                << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                                << std::endl;

                        // original common source differential traveltime (not used in relocation currently, but still need to be printed out)
                        } else if (rec_pair_data){

                            // check mandatory condition for earthquake relocation
                            if (!get_is_srcrec_swap()) {
                                std::cerr << "Error: src_rec_data is not swapped in src relocation mode!" << std::endl;
                                exit(1);
                            }

                            // common source differential traveltime data
                            SrcRecInfo& rec1 = rec_map_back[name_rec1];
                            SrcRecInfo& rec2 = rec_map_back[name_rec2];
                            CUSTOMREAL  cs_dif_travel_time_obs = data.cr_dif_travel_time_obs;

                            // receiver pair line : id_src id_rec1 name_rec1 lat1 lon1 elevation_m1 id_rec2 name_rec2 lat2 lon2 elevation_m2 phase differential_arival_time
                            ofs << std::setw(7) << std::right << std::setfill(' ') <<  src.id << " "
                                << std::setw(5) << std::right << std::setfill(' ') <<  rec1.id << " "
                                << rec1.name << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec1.lat << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec1.lon << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << -1.0*rec1.dep*1000.0 << " "
                                << std::setw(5) << std::right << std::setfill(' ') << rec2.id << " "
                                << rec2.name << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec2.lat << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec2.lon << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << -1.0*rec2.dep*1000.0 << " "
                                << data.phase << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << cs_dif_travel_time_obs << " "
                                << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                                << std::endl;

                        // common receiver differential traveltime
                        } else if (src_pair_data){   // common receiver differential traveltime
                            // check mandatory condition for earthquake relocation
                            if (!get_is_srcrec_swap()) {
                                std::cerr << "Error: src_rec_data is not swapped in src relocation mode!" << std::endl;
                                exit(1);
                            }

                            // common receiver differential traveltime data
                            SrcRecInfo& rec1 = rec_map_back[name_rec1];
                            SrcRecInfo& src2 = src_map_back[name_src2];
                            CUSTOMREAL  cr_dif_travel_time_obs = data.cs_dif_travel_time_obs - rec_map_all[name_src].tau_opt + rec_map_all[name_src2].tau_opt;


                            // receiver pair line : id_src id_rec1 name_rec1 lat1 lon1 elevation_m1 id_rec2 name_rec2 lat2 lon2 elevation_m2 phase differential_arival_time
                            ofs << std::setw(7) << std::right << std::setfill(' ') <<  src.id << " "
                                << std::setw(5) << std::right << std::setfill(' ') <<  rec1.id << " "
                                << rec1.name << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec1.lat << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec1.lon << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << -1.0*rec1.dep*1000.0 << " "
                                << std::setw(5) << std::right << std::setfill(' ') << src2.id << " "
                                << src2.name << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src2.lat << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src2.lon << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src2.dep << " "
                                << data.phase << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << cr_dif_travel_time_obs << " "
                                << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                                << std::endl;
                        }
                    } // end of rec loop

                } // end of src loop

                // close file
                ofs.close();

            } // end of run_mode == SRC_RELOCATION
        } // end if (proc_read_srcrec)


    } // end of src_rec_file_exist

    // wait till the main process finishes to write the output file
    synchronize_all_world(); /////////////////////

}


// check contradictory parameters
void InputParams::check_contradictions(){

    // if run_mode == 0 then the max_iter should be 1
    if (run_mode == ONLY_FORWARD && max_iter_inv > 1){
        std::cout << "Warning: run_mode = 0, max_iter should be 1" << std::endl;
        max_iter_inv = 1;
    }

    if (n_subprocs > 1 && sweep_type != SWEEP_TYPE_LEVEL){
        std::cout << "Warning: n_subprocs > 1 but do not use SWEEP_TYPE_LEVEL, sweep_type changes to SWEEP_TYPE_LEVEL" << std::endl;
        sweep_type = SWEEP_TYPE_LEVEL;
    }

    // if run_mode == 4 (1d inversion), only source parallelization is allowed
    if (run_mode == 4 && (ndiv_k > 1 || ndiv_j > 1 || ndiv_i > 1 || n_subprocs > 1)){
        std::cout << "Error: run_mode = 4, only source parallelization is allowed" << std::endl;
        std::cout << "Please set ndiv_rtp: [1,1,1] and nproc_sub: 1 in the input_params.yaml file" << std::endl;
        exit(1);
    }
#ifdef USE_CUDA
    if (use_gpu){

        if (sweep_type != SWEEP_TYPE_LEVEL || n_subprocs != 1){
            if(world_rank == 0) {
                std::cout << "ERROR: In GPU mode, sweep_type must be 1 and n_subprocs must be 1." << std::endl;
                std::cout << "Abort." << std::endl;
            }
            MPI_Finalize();
            exit(1);
        }
    }
#else
    if (use_gpu){
        if(world_rank == 0) {
            std::cout << "ERROR: TOMOATT is not compiled with CUDA." << std::endl;
            std::cout << "Abort." << std::endl;
        }
        MPI_Finalize();
        exit(1);
    }
#endif
}



// station correction kernel
void InputParams::station_correction_update(CUSTOMREAL stepsize){

    if (!use_sta_correction)
        return;

    // station correction kernel is generated in the main process and sent the value to all other processors

    // step 1, gather all arrival time info to the main process
    gather_all_arrival_times_to_main();

    // update station correction in rank0 (proc_read_srcrec = (id_sim = 0 && id_subdomain = 0 && subdom_main = true && ))
    // update rec_map_full, which comprises all the receivers
    if (proc_read_srcrec){

        // step 2 initialize the kernel K_{\hat T_i}
        for (auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
            iter->second.sta_correct_kernel = 0.0;
        }

        CUSTOMREAL max_kernel = 0.0;

        // step 3, calculate the kernel
        for (auto it_src = data_map_all.begin(); it_src != data_map_all.end(); it_src++){
            for (auto  it_rec = it_src->second.begin(); it_rec != it_src->second.end(); it_rec++){

                for (const auto& data : it_rec->second){

                    // absolute traveltime
                    if (data.is_src_rec){
                        std::cout << "teleseismic data, absolute traveltime is not supported now" << std::endl;

                    // common receiver differential traveltime
                    } else if (data.is_src_pair) {
                        std::cout << "teleseismic data, common receiver differential traveltime is not supported now" << std::endl;

                    // common source differential traveltime
                    } else if (data.is_rec_pair) { // here is for checking the flag (if data is swapped, is_rec_pair was originally is_src_pair)
                        std::string name_src   = data.name_src;
                        std::string name_rec1  = data.name_rec_pair[0];
                        std::string name_rec2  = data.name_rec_pair[1];

                        CUSTOMREAL syn_dif_time = data.cs_dif_travel_time;
                        CUSTOMREAL obs_dif_time = data.cs_dif_travel_time_obs;
                        rec_map_all[name_rec1].sta_correct_kernel += _2_CR *(syn_dif_time - obs_dif_time \
                                    + rec_map_all[name_rec1].sta_correct - rec_map_all[name_rec2].sta_correct)*data.weight;
                        rec_map_all[name_rec2].sta_correct_kernel -= _2_CR *(syn_dif_time - obs_dif_time \
                                    + rec_map_all[name_rec1].sta_correct - rec_map_all[name_rec2].sta_correct)*data.weight;

                        max_kernel = std::max(max_kernel,std::abs(rec_map_all[name_rec1].sta_correct_kernel));
                        max_kernel = std::max(max_kernel,std::abs(rec_map_all[name_rec2].sta_correct_kernel));
                    }

                }
            }
        }

        // step 4, update station correction
        for (auto iter = rec_map_all.begin(); iter!=rec_map_all.end(); iter++){
            iter->second.sta_correct -= iter->second.sta_correct_kernel / max_kernel * stepsize;
        }

    } // end of if (proc_read_srcrec)

    // step 5, send the station correction to all procesors. So they can consider station correction when calculating obj and adj source.
    if (proc_store_srcrec){
        CUSTOMREAL sta_correct = 0.0;
        std::vector<std::string> rec_map_all_name;
        std::string sta_name;
        int Nsta = 0;
        // for rank0, who has rec_map_all
        if (id_sim == 0){
            Nsta = rec_map_all.size();
            for (auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
                rec_map_all_name.push_back(iter->first);
            }
        }
        broadcast_i_single_inter_sim(Nsta,0);

        // loop over all receivers
        for (int i = 0; i < Nsta; i++){
            // broadcast sta_name and sta_correction to all processors
            if (id_sim == 0){
                sta_name = rec_map_all_name[i];
                sta_correct = rec_map_all[sta_name].sta_correct;
            }
            broadcast_str_inter_sim(sta_name,0);
            broadcast_cr_single_inter_sim(sta_correct,0);

            // update rec_map
            if (rec_map.find(sta_name) != rec_map.end()){
                rec_map[sta_name].sta_correct = sta_correct;
            }
        }
    }
}

void InputParams::modify_swapped_source_location() {
    if (proc_store_srcrec) {
        for(auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
            src_map_back[iter->first].lat   =   iter->second.lat;
            src_map_back[iter->first].lon   =   iter->second.lon;
            src_map_back[iter->first].dep   =   iter->second.dep;
            src_map_back[iter->first].sec   =   iter->second.sec + iter->second.tau_opt;
        }
    }
}


template <typename T>
void InputParams::allreduce_rec_map_var(T& var){

    T tmp_var = (T)var;

    if (subdom_main){
        // allreduce sum the variable var of rec_map[name_rec] to all
        // some process has rec_map[name_rec], some process does not have it

        // step 1, gather all the variable var

        // allreduce the variable var to the main process
        // if T is CUSTOMREAL
        if (std::is_same<T, CUSTOMREAL>::value){
            CUSTOMREAL v = tmp_var; // for compiler warning
            allreduce_cr_sim_single_inplace(v);
            tmp_var = v; // for compiler warning
        // if T is int
        } else if (std::is_same<T, int>::value){
            int v = tmp_var; // for compiler warning
            allreduce_i_sim_single_inplace(v);
            tmp_var = v; // for compiler warning
        } else {
            //error
            std::cout << "error in allreduce_rec_map_var" << std::endl;
            exit(1);
        }
    }

    // assign the value to the variable var
    var = tmp_var;

}


void InputParams::allreduce_rec_map_vobj_src_reloc(){
    if(proc_store_srcrec){
        // send total number of rec_map_all.size() to all processors
        int n_rec_all;
        std::vector<std::string> name_rec_all;
        if (proc_read_srcrec){
            n_rec_all = rec_map_all.size();
            for (auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
                name_rec_all.push_back(iter->first);
            }
        }

        // broadcast n_rec_all to all processors
        broadcast_i_single_inter_sim(n_rec_all,0);

        for (int i_rec = 0; i_rec < n_rec_all; i_rec++){
            // broadcast name_rec_all[i_rec] to all processors
            std::string name_rec;
            if (id_sim == 0)
                name_rec = name_rec_all[i_rec];

            broadcast_str_inter_sim(name_rec,0);

            // allreduce the vobj_src_reloc of rec_map_all[name_rec] to all processors
            if (rec_map.find(name_rec) != rec_map.end()){
                allreduce_rec_map_var(rec_map[name_rec].vobj_src_reloc);

            } else {
                CUSTOMREAL dummy = 0;
                allreduce_rec_map_var(dummy);

            }
        }
    }
}


void InputParams::allreduce_rec_map_grad_src(){
    if(proc_store_srcrec){
        // send total number of rec_map_all.size() to all processors
        int n_rec_all;
        std::vector<std::string> name_rec_all;
        if (proc_read_srcrec){
            n_rec_all = rec_map_all.size();
            for (auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
                name_rec_all.push_back(iter->first);
            }
        }

        // broadcast n_rec_all to all processors
        broadcast_i_single_inter_sim(n_rec_all,0);

        for (int i_rec = 0; i_rec < n_rec_all; i_rec++){
            // broadcast name_rec_all[i_rec] to all processors
            std::string name_rec;
            if (id_sim == 0)
                name_rec = name_rec_all[i_rec];

            broadcast_str_inter_sim(name_rec,0);

            // allreduce the grad_chi_ijk of rec_map_all[name_rec] to all processors
            if (rec_map.find(name_rec) != rec_map.end()){
                allreduce_rec_map_var(rec_map[name_rec].grad_chi_i);
                allreduce_rec_map_var(rec_map[name_rec].grad_chi_j);
                allreduce_rec_map_var(rec_map[name_rec].grad_chi_k);
                allreduce_rec_map_var(rec_map[name_rec].grad_tau);
                allreduce_rec_map_var(rec_map[name_rec].Ndata);
            } else {
                CUSTOMREAL dummy = 0;
                allreduce_rec_map_var(dummy);
                dummy = 0;
                allreduce_rec_map_var(dummy);
                dummy = 0;
                allreduce_rec_map_var(dummy);
                dummy = 0;
                allreduce_rec_map_var(dummy);
                int dummy_int = 0;
                allreduce_rec_map_var(dummy_int);
            }
        }
    }
}

void InputParams::check_increasing(CUSTOMREAL* arr_inv_grid, int n_grid, std::string name){
    for (int i = 0; i < n_grid-1; i++){
        if(arr_inv_grid[i] >= arr_inv_grid[i+1]){
            std::cout << "ERROR: inversion grid of " << name << " should be monotonically increasing: " << arr_inv_grid[i] << ", " << arr_inv_grid[i+1] << std::endl;
            exit(1);
        }
    }
}

void InputParams::check_inv_grid(){
    // check point 1, inversion grid should be monotonically increasing
    check_increasing(dep_inv, n_inv_r_flex, "depth");
    check_increasing(lat_inv, n_inv_t_flex, "latitude");
    check_increasing(lon_inv, n_inv_p_flex, "longitude");
    if(invgrid_ani){
        check_increasing(dep_inv_ani, n_inv_r_flex_ani, "depth_ani");
        check_increasing(lat_inv_ani, n_inv_t_flex_ani, "latitude_ani");
        check_increasing(lon_inv_ani, n_inv_p_flex_ani, "longitude_ani");
    }

    // check point 2, inversion grid should cover the model
    if(myrank == 0 && id_sim ==0){
        std::cout << "checking the inversion grid ..." << std::endl;
        std::cout << std::endl;
    }

    check_lower_bound(dep_inv, n_inv_r_flex, min_dep, "depth");
    check_upper_bound(dep_inv, n_inv_r_flex, max_dep, "depth");
    check_lower_bound(lat_inv, n_inv_t_flex, min_lat, "latitude");
    check_upper_bound(lat_inv, n_inv_t_flex, max_lat, "latitude");
    check_lower_bound(lon_inv, n_inv_p_flex, min_lon, "longitude");
    check_upper_bound(lon_inv, n_inv_p_flex, max_lon, "longitude");
    if(invgrid_ani){
        check_lower_bound(dep_inv_ani, n_inv_r_flex_ani, min_dep, "depth_ani");
        check_upper_bound(dep_inv_ani, n_inv_r_flex_ani, max_dep, "depth_ani");
        check_lower_bound(lat_inv_ani, n_inv_t_flex_ani, min_lat, "latitude_ani");
        check_upper_bound(lat_inv_ani, n_inv_t_flex_ani, max_lat, "latitude_ani");
        check_lower_bound(lon_inv_ani, n_inv_p_flex_ani, min_lon, "longitude_ani");
        check_upper_bound(lon_inv_ani, n_inv_p_flex_ani, max_lon, "longitude_ani");
    }

}

void InputParams::check_lower_bound(CUSTOMREAL*& arr, int& n_grid, CUSTOMREAL lower_bound, std::string name){

    if (arr[0] > lower_bound){

        CUSTOMREAL grid2 = std::min(lower_bound, 2*arr[0] - arr[1]);
        CUSTOMREAL grid1 = 2*grid2 - arr[0];
        if(myrank == 0 && id_sim ==0){
            std::cout << "lower bound of " << name << " inversion grid " << arr[0] << " is greater than computational domain "
                  << lower_bound << std::endl;
            std::cout << "Two additional inversion grid: " << grid1 << ", " << grid2 << " are added." << std::endl;
            std::cout << std::endl;
        }
        CUSTOMREAL* new_arr = new CUSTOMREAL[n_grid+2];
        new_arr[0] = grid1;
        new_arr[1] = grid2;
        for (int i = 0; i < n_grid; i++){
            new_arr[i+2] = arr[i];
        }
        delete[] arr;
        arr = new_arr;
        n_grid += 2;

    } else if (arr[0] <= lower_bound && arr[1] > lower_bound){

        CUSTOMREAL grid1 = 2*arr[0] - arr[1];
        if(myrank == 0 && id_sim ==0){
            std::cout << "sub lower bound of " << name << " inversion grid " << arr[1] << " is greater than computational domain "
                  << lower_bound << std::endl;
            std::cout << "One additional inversion grid: " << grid1 << " is added." << std::endl;
            std::cout << std::endl;
        }
        CUSTOMREAL* new_arr = new CUSTOMREAL[n_grid+1];
        new_arr[0] = grid1;
        for (int i = 0; i < n_grid; i++){
            new_arr[i+1] = arr[i];
        }
        delete[] arr;
        arr = new_arr;
        n_grid += 1;
    }
}

void InputParams::check_upper_bound(CUSTOMREAL*& arr, int& n_grid, CUSTOMREAL upper_bound, std::string name){
    if (arr[n_grid-1] < upper_bound){

        CUSTOMREAL grid1 = std::max(upper_bound, 2*arr[n_grid-1] - arr[n_grid-2]);
        CUSTOMREAL grid2 = 2*grid1 - arr[n_grid-1];
        if(myrank == 0 && id_sim ==0){
            std::cout << "upper bound of " << name << " inversion grid " << arr[n_grid-1] << " is less than computational domain "
                  << upper_bound << std::endl;
            std::cout << "Two additional inversion grid: " << grid1 << ", " << grid2 << " are added." << std::endl;
            std::cout << std::endl;
        }
        CUSTOMREAL* new_arr = new CUSTOMREAL[n_grid+2];
        for (int i = 0; i < n_grid; i++){
            new_arr[i] = arr[i];
        }
        new_arr[n_grid]     = grid1;
        new_arr[n_grid+1]   = grid2;
        delete[] arr;
        arr = new_arr;
        n_grid += 2;

    } else if (arr[n_grid-1] >= upper_bound && arr[n_grid-2] < upper_bound){

        CUSTOMREAL grid1 = 2*arr[n_grid-1] - arr[n_grid-2];
        if(myrank == 0 && id_sim ==0){
            std::cout << "sub upper bound of " << name << " inversion grid " << arr[n_grid-2] << " is less than computational domain "
                  << upper_bound << std::endl;
            std::cout << "One additional inversion grid: " << grid1 << " is added." << std::endl;
            std::cout << std::endl;
        }
        CUSTOMREAL* new_arr = new CUSTOMREAL[n_grid+1];
        for (int i = 0; i < n_grid; i++){
            new_arr[i] = arr[i];
        }
        new_arr[n_grid] = grid1;
        delete[] arr;
        arr = new_arr;
        n_grid += 1;
    }
}
