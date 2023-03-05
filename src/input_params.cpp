#include "input_params.h"

InputParams::InputParams(std::string& input_file){

    if (world_rank == 0) {
        // parse input files
        YAML::Node config = YAML::LoadFile(input_file);

        // read domain information
        if (config["domain"]) {
            // minimum and maximum depth
            if (config["domain"]["min_max_dep"]) {
                min_dep = config["domain"]["min_max_dep"][0].as<CUSTOMREAL>();
                max_dep = config["domain"]["min_max_dep"][1].as<CUSTOMREAL>();
            }
            // minimum and maximum latitude
            if (config["domain"]["min_max_lat"]) {
                min_lat = config["domain"]["min_max_lat"][0].as<CUSTOMREAL>();
                max_lat = config["domain"]["min_max_lat"][1].as<CUSTOMREAL>();
            }
            // minimum and maximum longitude
            if (config["domain"]["min_max_lon"]) {
                min_lon = config["domain"]["min_max_lon"][0].as<CUSTOMREAL>();
                max_lon = config["domain"]["min_max_lon"][1].as<CUSTOMREAL>();
            }
            // number of grid nodes on each axis r(dep), t(lat), p(lon)
            if (config["domain"]["n_rtp"]) {
                ngrid_k = config["domain"]["n_rtp"][0].as<int>();
                ngrid_j = config["domain"]["n_rtp"][1].as<int>();
                ngrid_i = config["domain"]["n_rtp"][2].as<int>();
            }
        }

        if (config["source"]) {
            // source depth(km) lat lon
            if (config["source"]["src_dep_lat_lon"]) {
                src_dep = config["source"]["src_dep_lat_lon"][0].as<CUSTOMREAL>();
                src_lat = config["source"]["src_dep_lat_lon"][1].as<CUSTOMREAL>();
                src_lon = config["source"]["src_dep_lat_lon"][2].as<CUSTOMREAL>();
            }
            // src rec file
            if (config["source"]["src_rec_file"]) {
                src_rec_file_exist = true;
                src_rec_file = config["source"]["src_rec_file"].as<std::string>();
            }

            if (config["source"]["swap_src_rec"]) {
                int tmp_swap = config["source"]["swap_src_rec"].as<int>();
                if (tmp_swap == 1) {
                    swap_src_rec = true;
                } else {
                    swap_src_rec = false;
                }
            }
        }

        if (config["model"]) {
            // model file path
            if (config["model"]["init_model_path"]) {
                init_model_path = config["model"]["init_model_path"].as<std::string>();
            }
            // model file path
            if (config["model"]["model_1d_name"]) {
                model_1d_name = config["model"]["model_1d_name"].as<std::string>();
            }
        }

        if (config["inversion"]) {
            // do inversion or not
            if (config["inversion"]["run_mode"]) {
                run_mode = config["inversion"]["run_mode"].as<int>();
            }
            // number of inversion grid
            if (config["inversion"]["n_inversion_grid"]) {
                n_inversion_grid = config["inversion"]["n_inversion_grid"].as<int>();
            }

            // number of inversion grid
            if (config["inversion"]["n_inv_dep_lat_lon"]) {
                n_inv_r = config["inversion"]["n_inv_dep_lat_lon"][0].as<int>();
                n_inv_t = config["inversion"]["n_inv_dep_lat_lon"][1].as<int>();
                n_inv_p = config["inversion"]["n_inv_dep_lat_lon"][2].as<int>();
            }

            // output path
            if (config["inversion"]["output_dir"]) {
                output_dir = config["inversion"]["output_dir"].as<std::string>();
            }

            // sta_correction_file
            if (config["inversion"]["sta_correction_file"]) {
                sta_correction_file_exist = true;
                sta_correction_file = config["inversion"]["sta_correction_file"].as<std::string>();
            }

            // type of input inversion grid
            if (config["inversion"]["type_dep_inv"]) {
                type_dep_inv = config["inversion"]["type_dep_inv"].as<int>();
            }
            if (config["inversion"]["type_lat_inv"]) {
                type_lat_inv = config["inversion"]["type_lat_inv"].as<int>();
            }
            if (config["inversion"]["type_lon_inv"]) {
                type_lon_inv = config["inversion"]["type_lon_inv"].as<int>();
            }

            // inversion grid positions
            if (config["inversion"]["min_max_dep_inv"]) {
                min_dep_inv = config["inversion"]["min_max_dep_inv"][0].as<CUSTOMREAL>();
                max_dep_inv = config["inversion"]["min_max_dep_inv"][1].as<CUSTOMREAL>();
            }
            // minimum and maximum latitude
            if (config["inversion"]["min_max_lat_inv"]) {
                min_lat_inv = config["inversion"]["min_max_lat_inv"][0].as<CUSTOMREAL>();
                max_lat_inv = config["inversion"]["min_max_lat_inv"][1].as<CUSTOMREAL>();
            }
            // minimum and maximum longitude
            if (config["inversion"]["min_max_lon_inv"]) {
                min_lon_inv = config["inversion"]["min_max_lon_inv"][0].as<CUSTOMREAL>();
                max_lon_inv = config["inversion"]["min_max_lon_inv"][1].as<CUSTOMREAL>();
            }

            // flexible inversion grid
            if (config["inversion"]["dep_inv"]) {
                n_inv_r_flex = config["inversion"]["dep_inv"].size();
                dep_inv = new CUSTOMREAL[n_inv_r_flex];
                for (int i = 0; i < n_inv_r_flex; i++){
                    dep_inv[i] = config["inversion"]["dep_inv"][i].as<CUSTOMREAL>();
                }
                n_inv_r_flex_read = true;
            }
            if (config["inversion"]["lat_inv"]) {
                n_inv_t_flex = config["inversion"]["lat_inv"].size();
                lat_inv = new CUSTOMREAL[n_inv_t_flex];
                for (int i = 0; i < n_inv_t_flex; i++){
                    lat_inv[i] = config["inversion"]["lat_inv"][i].as<CUSTOMREAL>();
                }
                n_inv_t_flex_read = true;
            }
            if (config["inversion"]["lon_inv"]) {
                n_inv_p_flex = config["inversion"]["lon_inv"].size();
                lon_inv = new CUSTOMREAL[n_inv_p_flex];
                for (int i = 0; i < n_inv_p_flex; i++){
                    lon_inv[i] = config["inversion"]["lon_inv"][i].as<CUSTOMREAL>();
                }
                n_inv_p_flex_read = true;
            }

            // number of max iteration for inversion
            if (config["inversion"]["max_iterations_inv"]) {
                max_iter_inv = config["inversion"]["max_iterations_inv"].as<int>();
            }

            // step_size
            if (config["inversion"]["step_size"]) {
                step_size_init = config["inversion"]["step_size"].as<CUSTOMREAL>();
            }
            if (config["inversion"]["step_size_sc"]) {
                step_size_init_sc = config["inversion"]["step_size_sc"].as<CUSTOMREAL>();
            }
            if (config["inversion"]["step_size_decay"]) {
                step_size_decay = config["inversion"]["step_size_decay"].as<CUSTOMREAL>();
            }
            if (config["inversion"]["step_length_src_reloc"]) {
                step_length_src_reloc = config["inversion"]["step_length_src_reloc"].as<CUSTOMREAL>();
            }


            // smoothing method
            if (config["inversion"]["smooth_method"]) {
                smooth_method = config["inversion"]["smooth_method"].as<int>();
                if (smooth_method > 1) {
                    std::cout << "undefined smooth_method. stop." << std::endl;
                    MPI_Finalize();
                    exit(1);
                }
            }

            // l_smooth_rtp
            if (config["inversion"]["l_smooth_rtp"]) {
                smooth_lr = config["inversion"]["l_smooth_rtp"][0].as<CUSTOMREAL>();
                smooth_lt = config["inversion"]["l_smooth_rtp"][1].as<CUSTOMREAL>();
                smooth_lp = config["inversion"]["l_smooth_rtp"][2].as<CUSTOMREAL>();
            }

            // optim_method
            if (config["inversion"]["optim_method"]) {
                optim_method = config["inversion"]["optim_method"].as<int>();
                if (optim_method > 2) {
                    std::cout << "undefined optim_method. stop." << std::endl;
                    //MPI_Finalize();
                    exit(1);
                }
            }

            // regularization weight
            if (config["inversion"]["regularization_weight"]) {
                regularization_weight = config["inversion"]["regularization_weight"].as<CUSTOMREAL>();
            }

            // max sub iteration
            if (config["inversion"]["max_sub_iterations"]) {
                max_sub_iterations = config["inversion"]["max_sub_iterations"].as<int>();
            }

            // weight of different types of data
            if (config["inversion"]["abs_time_local_weight"]) {
                abs_time_local_weight = config["inversion"]["abs_time_local_weight"].as<CUSTOMREAL>();
            }
            if (config["inversion"]["cr_dif_time_local_weight"]) {
                cr_dif_time_local_weight = config["inversion"]["cr_dif_time_local_weight"].as<CUSTOMREAL>();
            }
            if (config["inversion"]["cs_dif_time_local_weight"]) {
                cs_dif_time_local_weight = config["inversion"]["cs_dif_time_local_weight"].as<CUSTOMREAL>();
            }
            if (config["inversion"]["teleseismic_weight"]) {
                teleseismic_weight = config["inversion"]["teleseismic_weight"].as<CUSTOMREAL>();
            }
            if (config["inversion"]["is_balance_data_weight"]) {
                is_balance_data_weight = config["inversion"]["is_balance_data_weight"].as<int>();
            }

            if (config["inversion"]["step_length_decay"]) {
                step_length_decay = config["inversion"]["step_length_decay"].as<CUSTOMREAL>();
            }
        }

        if (config["inv_strategy"]) {
            // update which model parameters
            if (config["inv_strategy"]["is_inv_slowness"])
                is_inv_slowness = config["inv_strategy"]["is_inv_slowness"].as<int>();
            if (config["inv_strategy"]["is_inv_azi_ani"])
                is_inv_azi_ani = config["inv_strategy"]["is_inv_azi_ani"].as<int>();
            if (config["inv_strategy"]["is_inv_rad_ani"])
                is_inv_rad_ani = config["inv_strategy"]["is_inv_rad_ani"].as<int>();

            // taper kernel (now only for teleseismic tomography)
            if (config["inv_strategy"]["kernel_taper"]){
                kernel_taper[0] = config["inv_strategy"]["kernel_taper"][0].as<CUSTOMREAL>();
                kernel_taper[1] = config["inv_strategy"]["kernel_taper"][1].as<CUSTOMREAL>();
            }

            // station correction (now only for teleseismic data)
            if (config["inv_strategy"]["is_sta_correction"]){
                is_sta_correction = config["inv_strategy"]["is_sta_correction"].as<int>();
            }
        }


        if (config["parallel"]) {
            // number of simultaneous runs
            if(config["parallel"]["n_sims"]) {
                n_sims = config["parallel"]["n_sims"].as<int>();
            }
            // number of subdomains
            if (config["parallel"]["ndiv_rtp"]) {
                ndiv_k = config["parallel"]["ndiv_rtp"][0].as<int>();
                ndiv_j = config["parallel"]["ndiv_rtp"][1].as<int>();
                ndiv_i = config["parallel"]["ndiv_rtp"][2].as<int>();
            }
            // number of processes in each subdomain
            if (config["parallel"]["nproc_sub"]) {
                n_subprocs = config["parallel"]["nproc_sub"].as<int>();
            }
            // gpu flag
            if (config["parallel"]["use_gpu"]) {
                use_gpu = config["parallel"]["use_gpu"].as<int>();
            }
        }

        if (config["calculation"]) {
            // convergence tolerance
            if (config["calculation"]["convergence_tolerance"]) {
                conv_tol = config["calculation"]["convergence_tolerance"].as<CUSTOMREAL>();
            }
            // max number of iterations
            if (config["calculation"]["max_iterations"]) {
                max_iter = config["calculation"]["max_iterations"].as<int>();
            }
            // stencil order
            if (config["calculation"]["stencil_order"]) {
                stencil_order = config["calculation"]["stencil_order"].as<int>();
                // check if stencil_order == 999 : hybrid scheme
                if (stencil_order == 999) {
                    hybrid_stencil_order = true;
                    stencil_order = 1;
                }
            }
            // stencil type
            if (config["calculation"]["stencil_type"]) {
                stencil_type = config["calculation"]["stencil_type"].as<int>();
            }
            // sweep type
            if (config["calculation"]["sweep_type"]) {
                sweep_type = config["calculation"]["sweep_type"].as<int>();
            }
            // output file format
            if (config["calculation"]["output_file_format"]) {
                int ff_flag = config["calculation"]["output_file_format"].as<int>();
                if (ff_flag == 0){
                    #if USE_HDF5
                        output_format = OUTPUT_FORMAT_HDF5;
                    #else
                        std::cout << "output_file_format is 0, but the code is compiled without HDF5. stop." << std::endl;
                        MPI_Finalize();
                        exit(1);
                    #endif
                } else if (ff_flag == 1){
                    output_format = OUTPUT_FORMAT_ASCII;
                } else {
                    std::cout << "undefined output_file_format. stop." << std::endl;
                    MPI_Finalize();
                    exit(1);
                }
            }
        }

        if (config["output_setting"]) {
            if (config["output_setting"]["is_output_source_field"])
                is_output_source_field = config["output_setting"]["is_output_source_field"].as<int>();
            if (config["output_setting"]["is_output_model_dat"])
                is_output_model_dat = config["output_setting"]["is_output_model_dat"].as<int>();
            if (config["output_setting"]["is_verbose_output"])
                is_verbose_output = config["output_setting"]["is_verbose_output"].as<int>();
            if (config["output_setting"]["is_output_final_model"])
                is_output_final_model = config["output_setting"]["is_output_final_model"].as<int>();
            if (config["output_setting"]["is_output_in_process"])
                is_output_in_process = config["output_setting"]["is_output_in_process"].as<int>();
        }

        if (config["debug"]) {
            if (config["debug"]["debug_mode"]) {
                int tmp_test = config["debug"]["debug_mode"].as<int>();
                if (tmp_test == 1) {
                    if_test = true;
                } else {
                    if_test = false;
                }
            }
        }

        std::cout << "min_max_dep: " << min_dep << " " << max_dep << std::endl;
        std::cout << "min_max_lat: " << min_lat << " " << max_lat << std::endl;
        std::cout << "min_max_lon: " << min_lon << " " << max_lon << std::endl;
        std::cout << "n_rtp: "    << ngrid_k << " " << ngrid_j << " " << ngrid_i << std::endl;
        std::cout << "ndiv_rtp: " << ndiv_k << " "  << ndiv_j  << " " << ndiv_i << std::endl;
        std::cout << "n_subprocs: " << n_subprocs << std::endl;
        std::cout << "n_sims: " << n_sims << std::endl;

        // set inversion grid definition if not set by user
        if (min_dep_inv <= -99999) min_dep_inv = min_dep;
        if (max_dep_inv <= -99999) max_dep_inv = max_dep;
        if (min_lat_inv <= -99999) min_lat_inv = min_lat;
        if (max_lat_inv <= -99999) max_lat_inv = max_lat;
        if (min_lon_inv <= -99999) min_lon_inv = min_lon;
        if (max_lon_inv <= -99999) max_lon_inv = max_lon;

        // allocate dummy arrays for flex inv grid
        if (!n_inv_r_flex_read)
            dep_inv = new CUSTOMREAL[n_inv_r_flex];
        if (!n_inv_t_flex_read)
            lat_inv = new CUSTOMREAL[n_inv_t_flex];
        if (!n_inv_p_flex_read)
            lon_inv = new CUSTOMREAL[n_inv_p_flex];

        // write parameter file to output directory
        write_params_to_file();

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
    if (src_rec_file_exist == false){
        SrcRecInfo src;
        src.id = 0;
        src.name = "s0";
        src.lat    = src_lat;
        src.lon    = src_lon;
        src.dep    = src_dep;
        src_map[src.name] = src;
        src_ids_this_sim.push_back(0);
        src_names_this_sim.push_back("s0");
        SrcRecInfo rec;
        rec.id = 0;
        rec.name = "r0";
        rec_map[rec.name] = rec;
        DataInfo data;
        data_info.push_back(data);
    }
    broadcast_bool_single(swap_src_rec, 0);

    broadcast_str(src_rec_file, 0);
    broadcast_str(sta_correction_file, 0);
    broadcast_str(output_dir, 0);
    broadcast_bool_single(src_rec_file_exist, 0);
    broadcast_bool_single(sta_correction_file_exist, 0);
    broadcast_str(init_model_path, 0);
    broadcast_i_single(run_mode, 0);
    broadcast_i_single(n_inversion_grid, 0);
    broadcast_i_single(n_inv_r, 0);
    broadcast_i_single(n_inv_t, 0);
    broadcast_i_single(n_inv_p, 0);
    broadcast_cr_single(min_dep_inv, 0);
    broadcast_cr_single(max_dep_inv, 0);
    broadcast_cr_single(min_lat_inv, 0);
    broadcast_cr_single(max_lat_inv, 0);
    broadcast_cr_single(min_lon_inv, 0);
    broadcast_cr_single(max_lon_inv, 0);

    broadcast_i_single(type_dep_inv, 0);
    broadcast_i_single(type_lat_inv, 0);
    broadcast_i_single(type_lon_inv, 0);
    broadcast_i_single(n_inv_r_flex, 0);
    broadcast_i_single(n_inv_t_flex, 0);
    broadcast_i_single(n_inv_p_flex, 0);

    if (world_rank != 0) {
        dep_inv = new CUSTOMREAL[n_inv_r_flex];
        lat_inv = new CUSTOMREAL[n_inv_t_flex];
        lon_inv = new CUSTOMREAL[n_inv_p_flex];
    }

    broadcast_cr(dep_inv,n_inv_r_flex, 0);
    broadcast_cr(lat_inv,n_inv_t_flex, 0);
    broadcast_cr(lon_inv,n_inv_p_flex, 0);

    broadcast_cr_single(abs_time_local_weight, 0);
    broadcast_cr_single(cs_dif_time_local_weight, 0);
    broadcast_cr_single(cr_dif_time_local_weight, 0);
    broadcast_cr_single(teleseismic_weight, 0);
    broadcast_i_single(is_balance_data_weight, 0);

    broadcast_i_single(smooth_method, 0);
    broadcast_cr_single(smooth_lr, 0);
    broadcast_cr_single(smooth_lt, 0);
    broadcast_cr_single(smooth_lp, 0);
    broadcast_i_single(optim_method, 0);
    broadcast_i_single(max_iter_inv, 0);
    broadcast_cr_single(step_size_init, 0);
    broadcast_cr_single(step_size_init_sc, 0);
    broadcast_cr_single(step_size_decay, 0);
    broadcast_cr_single(step_length_src_reloc, 0);
    broadcast_cr_single(step_length_decay, 0);
    broadcast_cr_single(regularization_weight, 0);
    broadcast_i_single(max_sub_iterations, 0);
    broadcast_i_single(ndiv_i, 0);
    broadcast_i_single(ndiv_j, 0);
    broadcast_i_single(ndiv_k, 0);
    broadcast_i_single(n_subprocs, 0);
    broadcast_i_single(n_sims, 0);
    broadcast_cr_single(conv_tol, 0);
    broadcast_i_single(max_iter, 0);
    broadcast_i_single(stencil_order, 0);
    broadcast_bool_single(hybrid_stencil_order, 0);
    broadcast_i_single(stencil_type, 0);
    broadcast_i_single(sweep_type, 0);
    broadcast_i_single(output_format, 0);
    broadcast_bool_single(if_test, 0);
    broadcast_i_single(use_gpu, 0);

    broadcast_bool_single(is_output_source_field, 0);
    broadcast_bool_single(is_output_model_dat, 0);
    broadcast_bool_single(is_verbose_output, 0);
    broadcast_bool_single(is_output_final_model, 0);
    broadcast_bool_single(is_output_in_process, 0);
    broadcast_bool_single(is_inv_slowness, 0);
    broadcast_bool_single(is_inv_azi_ani, 0);
    broadcast_bool_single(is_inv_rad_ani, 0);
    broadcast_cr(kernel_taper,2,0);
    broadcast_bool_single(is_sta_correction, 0);

    // check contradictory settings
    check_contradictions();

    // broadcast the values to all processes
    stdout_by_rank_zero("read input file successfully.");

}


InputParams::~InputParams(){
    // free memory
    if (subdom_main) {
        for (std::string name_src : src_names_this_sim){
            SrcRecInfo& src = get_src_point(name_src);
            if (src.is_out_of_region){
                if (j_last && src.arr_times_bound_N != nullptr)
                    free(src.arr_times_bound_N);
                if (j_first && src.arr_times_bound_S != nullptr)
                    free(src.arr_times_bound_S);
                if (i_last && src.arr_times_bound_E != nullptr)
                    free(src.arr_times_bound_E);
                if (i_first && src.arr_times_bound_W != nullptr)
                    free(src.arr_times_bound_W);
                if (k_first && src.arr_times_bound_Bot != nullptr)
                    free(src.arr_times_bound_Bot);

                free(src.is_bound_src);
            }
        }
    }

    delete [] dep_inv;
    delete [] lat_inv;
    delete [] lon_inv;

    // clear all src, rec, data
    src_map.clear();
    src_map_back.clear();
    //src_map_prepare.clear();
    src_map_tele.clear();
    rec_map.clear();
    rec_map_back.clear();
    data_info.clear();
    data_info_back.clear();
    syn_time_map_sr.clear();
    //data_info_smap.clear();
}


void InputParams::write_params_to_file() {
    // write all the simulation parameters in another yaml file
    std::string file_name = "params_log.yaml";
    std::ofstream fout(file_name);
    fout << "version: " << 2 << std::endl;

    fout << std::endl;
    fout << "domain:" << std::endl;
    fout << "   min_max_dep: [" << min_dep << ", " << max_dep << "] # depth in km" << std::endl;
    fout << "   min_max_lat: [" << min_lat << ", " << max_lat << "] # latitude in degree" << std::endl;
    fout << "   min_max_lon: [" << min_lon << ", " << max_lon << "] # longitude in degree" << std::endl;
    fout << "   n_rtp: [" << ngrid_k << ", " << ngrid_j << ", " << ngrid_i << "] # number of nodes in depth,latitude,longitude direction" << std::endl;

    fout << std::endl;
    fout << "source:" << std::endl;
    fout << "   src_rec_file: " << src_rec_file     << " # source receiver file path" << std::endl;
    fout << "   swap_src_rec: " << int(swap_src_rec) << " # swap source and receiver (1: yes, 0: no)" << std::endl;

    fout << std::endl;
    fout << "model:" << std::endl;
    fout << "   init_model_path: " << init_model_path << " # path to initial model file " << std::endl;
    // check if model_1d_name has any characters
    if (model_1d_name.size() > 0)
        fout << "   model_1d_name: " << model_1d_name;
    else
        fout << "#   model_1d_name: " << "dummy_model_1d_name";
    fout << " # 1D model name used in teleseismic 2D solver (iasp91, ak135, user_defined is available), defined in include/1d_model.h" << std::endl;

    fout << std::endl;
    fout << "inversion:" << std::endl;
    fout << "   run_mode: "           << run_mode << " # 0 for forward simulation only, 1 for inversion" << std::endl;
    fout << "   output_dir: "         << output_dir << " # path to output director (default is ./OUTPUT_FILES/)" << std::endl;
    fout << "   optim_method: "          << optim_method << " # optimization method. 0 : grad_descent, 1 : halve-stepping, 2 : lbfgs (EXPERIMENTAL)" << std::endl;
    fout << "   max_iterations_inv: "    << max_iter_inv << " # maximum number of inversion iterations" << std::endl;
    fout << "   step_size: "             << step_size_init << " # initial step size for model update" << std::endl;
    fout << "   step_size_sc: "          << step_size_init_sc << " # ..."  << std::endl;
    fout << "   step_size_decay: "       << step_size_decay << " # ..." << std::endl;
    fout << "   smooth_method: "         << smooth_method << " # 0: multiparametrization, 1: laplacian smoothing (EXPERIMENTAL)" << std::endl;

    fout << std::endl;
    fout << "   # parameters for multiparametric inversion" << std::endl;
    fout << "   n_inversion_grid: "   << n_inversion_grid << " # number of inversion grid sets" << std::endl;
    fout << "   n_inv_dep_lat_lon: [" << n_inv_r << ", " << n_inv_t << ", " << n_inv_p << "] # number of the base inversion grid points" << std::endl;
    if (sta_correction_file_exist)
        fout << "   sta_correction_file: " << sta_correction_file;
    else
        fout << "#   sta_correction_file: " << "dummy_sta_correction_file";
    fout << " # station correction file path" << std::endl;
    fout << "   type_dep_inv: " << type_dep_inv << " # 0: uniform inversion grid, 1: flexible grid" <<std::endl;
    fout << "   type_lat_inv: " << type_lat_inv << " # 0: uniform inversion grid, 1: flexible grid" <<std::endl;
    fout << "   type_lon_inv: " << type_lon_inv << " # 0: uniform inversion grid, 1: flexible grid" <<std::endl;

    fout << std::endl;
    fout << "   # parameters for uniform inversion grid" << std::endl;
    fout << "   min_max_dep_inv: [" << min_dep_inv << ", " << max_dep_inv << "]" << " # depth in km (Radius of the earth is defined in config.h/R_earth)"  << std::endl;
    fout << "   min_max_lat_inv: [" << min_lat_inv << ", " << max_lat_inv << "]" << " # latitude in degree"  << std::endl;
    fout << "   min_max_lon_inv: [" << min_lon_inv << ", " << max_lon_inv << "]" << " # longitude in degree" << std::endl;

    fout << std::endl;
    fout << "   # parameters for flexible inversion grid" << std::endl;
    if (n_inv_r_flex_read) {
        fout << "   n_inv_r_flex: " << n_inv_r_flex << std::endl;
        fout << "   dep_inv: [";
        for (int i = 0; i < n_inv_r_flex; i++){
            fout << dep_inv[i];
            if (i != n_inv_r_flex-1)
                fout << ", ";
        }
        fout << "]" << std::endl;
    } else {
        fout << "#   n_inv_r_flex: " << "3" << std::endl;
        fout << "#   dep_inv: " << "[1, 1, 1]" << std::endl;
    }
    if (n_inv_t_flex_read) {
        fout << "   n_inv_t_flex: " << n_inv_t_flex << std::endl;
        fout << "   lat_inv: [";
        for (int i = 0; i < n_inv_t_flex; i++){
            fout << lat_inv[i];
            if (i != n_inv_t_flex-1)
                fout << ", ";
        }
        fout << "]" << std::endl;
    } else {
        fout << "#   n_inv_t_flex: " << "3" << std::endl;
        fout << "#   lat_inv: " << "[1, 1, 1]" << std::endl;
    }
    if (n_inv_p_flex_read) {
        fout << "   n_inv_p_flex: " << n_inv_p_flex << std::endl;
        fout << "   lon_inv: [";
        for (int i = 0; i < n_inv_p_flex; i++){
            fout << lon_inv[i];
            if (i != n_inv_p_flex-1)
                fout << ", ";
        }
        fout << "]" << std::endl;
    } else {
        fout << "#   n_inv_p_flex: " << "3" << std::endl;
        fout << "#   lon_inv: " << "[1, 1, 1]" << std::endl;
    }

    fout << std::endl;
    fout << "   # parameters for halve-stepping or lbfg mode" << std::endl;
    fout << "   max_sub_iterations: "    << max_sub_iterations << " # maximum number of each sub-iteration" << std::endl;
    fout << "   l_smooth_rtp: ["         << smooth_lr << ", " << smooth_lt << ", " << smooth_lp << "] # smoothing coefficients for laplacian smoothing" << std::endl;
    fout << "   regularization_weight: " << regularization_weight << " # weight value for regularization (lbfgs mode only)" << std::endl;

    fout << std::endl;
    fout << "inv_strategy: # flags for selecting the target parameters to be inversed" << std::endl;
    fout << "   is_inv_slowness: "   << int(is_inv_slowness) << " # 1: slowness value will be calculated in inversion, 0: will not be calculated" << std::endl;
    fout << "   is_inv_azi_ani: "    << int(is_inv_azi_ani)  << " # 1: azimuth anisotropy value will be calculated in inversion, 0: will not be calculated"<< std::endl;
    fout << "   is_inv_rad_ani: "    << int(is_inv_rad_ani)  << " # flag for radial anisotropy (Not implemented yet)" << std::endl;
    fout << "   kernel_taper: ["     << kernel_taper[0] << ", " << kernel_taper[1] << "]" << std::endl;
    fout << "   is_sta_correction: " << int(is_sta_correction) << std::endl;

    fout << std::endl;
    fout << "parallel: # parameters for parallel computation" << std::endl;
    fout << "   n_sims: "    << n_sims << " # number of simultanoues runs" << std::endl;
    fout << "   ndiv_rtp: [" << ndiv_k << ", " << ndiv_j << ", " << ndiv_i << "] # number of subdivision on each direction" << std::endl;
    fout << "   nproc_sub: " << n_subprocs << " # number of processors for sweep parallelization" << std::endl;
    fout << "   use_gpu: "   << int(use_gpu) << " # 1 if use gpu (EXPERIMENTAL)" << std::endl;

    fout << std::endl;
    fout << "calculation:" << std::endl;
    fout << "   convergence_tolerance: " << conv_tol << " # threshold value for checking the convergence for each forward/adjoint run"<< std::endl;
    fout << "   max_iterations: " << max_iter << " # number of maximum iteration for each forward/adjoint run" << std::endl;
    fout << "   stencil_order: " << stencil_order << " # order of stencil, 1 or 3" << std::endl;
    fout << "   stencil_type: " << stencil_type << " # 0: , 1: first-order upwind scheme (only sweep_type 0 is supported) " << std::endl;
    fout << "   sweep_type: " << sweep_type << " # 0: legacy, 1: cuthill-mckee with shm parallelization" << std::endl;
    int ff_flag=0;
    if (output_format == OUTPUT_FORMAT_HDF5) ff_flag = 0;
    else if (output_format == OUTPUT_FORMAT_ASCII) ff_flag = 1;
    else {
        std::cout << "Error: output_format is not defined!" << std::endl;
        exit(1);
    }
    fout << "   output_file_format: " << ff_flag << std::endl;

    fout << std::endl;
    fout << "output_setting:" << std::endl;
    fout << "   is_output_source_field: " << int(is_output_source_field) << " # output the calculated field of all sources                            1 for yes; 0 for no;  default: 1" << std::endl;
    fout << "   is_output_model_dat: "    << int(is_output_model_dat)    << " # output model_parameters_inv_0000.dat or not.                          1 for yes; 0 for no;  default: 1" << std::endl;
    fout << "   is_verbose_output: "      << int(is_verbose_output)      << " # output internal parameters, if no, only model parameters are out.     1 for yes; 0 for no;  default: 0" << std::endl;
    fout << "   is_output_final_model: "  << int(is_output_final_model)  << " # output merged final model or not.                                     1 for yes; 0 for no;  default: 1" << std::endl;
    fout << "   is_output_in_process: "   << int(is_output_in_process)   << " # output model at each inv iteration or not.                            1 for yes; 0 for no;  default: 1" << std::endl;

    //fout << std::endl;
    //fout << "debug:" << std::endl;
    //fout << "   debug_mode: " << int(if_test) << std::endl;


}


// return radious
CUSTOMREAL InputParams::get_src_radius(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return depth2radius(get_src_point(name_sim_src).dep);
    else
        return depth2radius(src_dep);
}


CUSTOMREAL InputParams::get_src_lat(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return get_src_point(name_sim_src).lat*DEG2RAD;
    else
        return src_lat*DEG2RAD;
}


CUSTOMREAL InputParams::get_src_lon(const std::string& name_sim_src) {
    if (src_rec_file_exist)
        return get_src_point(name_sim_src).lon*DEG2RAD;
    else
        return src_lon*DEG2RAD;
}


SrcRecInfo& InputParams::get_src_point(std::string name_src){

    if (subdom_main){
        for (auto& src: src_map){
            if (src.second.name == name_src)
                return src.second;
        }

        // if not found, return error
        std::cout << "Error: src name " << name_src << " not found!" << std::endl;
        // assigned src id
        std::cout << "Assigned src names to this simultanous run : ";
        for (auto& src: src_map){
            std::cout << src.second.name << " ";
        }
        std::cout << std::endl;

        exit(1);
    } else {
        // return error because non-subdom_main process should not call this function
        std::cout << "Error: non-subdom_main process should not call this function!" << std::endl;
        exit(1);
    }
}


SrcRecInfo& InputParams::get_rec_point(std::string name_rec) {
    if (subdom_main){
        for (auto& rec: rec_map) {
            if (rec.second.name == name_rec)
                return rec.second;
        }

        // if not found, return error
        std::cout << "Error: rec name " << name_rec << " not found!" << std::endl;
        exit(1);
    } else {
        // return error because non-subdom_main process should not call this function
        std::cout << "Error: non-subdom_main process should not call this function!" << std::endl;
        exit(1);
   }
}


std::vector<std::string> InputParams::get_rec_points(std::string name_src){
    std::vector<std::string> recs;
    for(auto iter = syn_time_map_sr[name_src].begin(); iter != syn_time_map_sr[name_src].end(); iter++){
        recs.push_back(iter->first);
    }
    return recs;
}



bool InputParams::get_if_src_teleseismic(std::string src_name) {
    bool if_src_teleseismic;

    if (subdom_main)
        if_src_teleseismic = get_src_point(src_name).is_out_of_region;

    // broadcast to all processes within simultaneous run group
    broadcast_bool_single_sub(if_src_teleseismic, 0);

    return if_src_teleseismic;
}


void InputParams::prepare_src_map(){
    //
    // only the subdom_main process of the first simultaneous run group (id_sim==0 && sim_rank==any && subdom_main) reads src/rec file
    // and stores entile src/rec list in src_points and rec_points
    // then, the subdom_main process of each simultaneous run group (id_sim==any && subdom_main==true) retains only its own src/rec objects,
    // which are actually calculated in those simultaneous run groups
    //

    // read src rec file
    if (src_rec_file_exist && id_sim==0 && subdom_main) {

        parse_src_rec_file(src_rec_file, \
                           src_map, \
                           rec_map, \
                           data_info, \
                           src_name_list);

        // read station correction file by all processes
        if (sta_correction_file_exist && id_sim==0 && subdom_main) {
            // store all src/rec info
            parse_sta_correction_file(sta_correction_file,
                                      rec_map);
        }

        // copy backups
        src_map_back   = src_map;
        rec_map_back   = rec_map;
        data_info_back = data_info;

        // check if src positions are within the domain or not (teleseismic source)
        // detected teleseismic source is separated into tele_src_points and tele_rec_points
        std::cout << "separate regional and teleseismic src/rec points" << std::endl;
        separate_region_and_tele_src_rec_data(src_map_back, rec_map_back, data_info_back,
                                              src_map,      rec_map,      data_info,
                                              src_map_tele, rec_map_tele, data_info_tele,
                                              data_type,
                                              N_abs_local_data,
                                              N_cr_dif_local_data,
                                              N_cs_dif_local_data,
                                              N_teleseismic_data,
                                              min_lat, max_lat, min_lon, max_lon, min_dep, max_dep);

        if (swap_src_rec) {
            // here only reginal events will be processed
            stdout_by_main("Swapping src and rec. This may take few minutes for a large dataset (only regional events will be processed)\n");
            do_swap_src_rec(src_map, rec_map, data_info);
        }

        // concatenate resional and teleseismic src/rec points
        merge_region_and_tele_src(src_map, rec_map, data_info,
                                  src_map_tele, rec_map_tele, data_info_tele);

        // abort if number of src_points are less than n_sims
        int n_src_points = src_map.size();
        if (n_src_points < n_sims){
            std::cout << "Error: number of sources in src_rec_file is less than n_sims. Abort.1" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

   } // end of if (src_rec_file_exist && id_sim==0 && subdom_main)

    // wait
    synchronize_all_world();


    // to all the subdom_main processes of each simultaneous run group
    // #TODO: not updated yet for new srcrec !!!
    if (src_rec_file_exist) {
        // # TODO: check if this can be placed

        if (world_rank==0)
            std::cout << "\nsource assign to processors\n" <<std::endl;


        // broadcast src_map,
        //           rec_map,
        //           data_info,
        // outuput: src_ids_this_sim, src_names_this_sim includes all src ids/names
        //          which are calculated this simultaneous run group
        distribute_src_rec_data(src_map,
                                rec_map,
                                data_info,
                                src_ids_this_sim,
                                src_names_this_sim);

        std::cout << "initialize syn time map" << std::endl;
        // generate syn_time_map_sr
        initialize_syn_time_map();

        synchronize_all_world();

        // broadcast src_map_tele,
        //           rec_map_tele,
        //           data_info_tele,
        //distribute_src_rec_data(src_map_tele,
        //                        rec_map_tele,
        //                        data_info_tele,
        //                        src_ids_this_sim_tele,
        //                        src_names_this_sim_tele);

        // broadcast src_map_prepare,
        //           rec_map_prepare,
        //           data_info_prepare,
        //distribute_src_rec_data(src_map_prepare,
        //                        rec_map_prepare,
        //                        data_info_prepare,
        //                        src_ids_this_sim_prepare,
        //                        src_names_this_sim_prepare);

        // then finally src_ids_this_sim_tele, src_names_this_sim_tele should be concatenated to src_ids_this_sim, src_names_this_sim
        // and src_ids_this_sim_prepare, src_names_this_sim_prepare should be concatenated to src_ids_this_sim, src_names_this_sim
        //src_ids_this_sim.insert(src_ids_this_sim.end(), src_ids_this_sim_tele.begin(), src_ids_this_sim_tele.end());
        //src_names_this_sim.insert(src_names_this_sim.end(), src_names_this_sim_tele.begin(), src_names_this_sim_tele.end());
        //src_ids_this_sim.insert(src_ids_this_sim.end(), src_ids_this_sim_prepare.begin(), src_ids_this_sim_prepare.end());
        //src_names_this_sim.insert(src_names_this_sim.end(), src_names_this_sim_prepare.begin(), src_names_this_sim_prepare.end());


// Those data management scheme are not used anymore

        // generate data_info_smap,
        //          data_info_smap_reloc,
        //          syn_time_map_sr,
//        std::cout << "rearrange data " << std::endl;
//        // copy/rearrange and analyse "data_info" -> data_info_smap; data_info_smap_reloc
//        rearrange_data_info();
//
//        std::cout << "generate src_map_prepare" << std::endl;
//        // generate src_map_prepare based on data_info_smap
//        generate_src_map_prepare();
//
        // distribute syn_time_map_sr
//        distribute_syn_time_map_sr();

///////////////////////////////////////////////////////////////////////////////////////////////////////


//        /////////////////
//        // for teleseismic earthquakes whose boundary condition should be prepared first
//        /////////////////
//        src_ids_this_sim_tele.clear();
//        src_names_this_sim_tele.clear();
//        // assign elements of src_points
//        int i_src = 0;
//        for (auto iter = src_map_tele.begin(); iter != src_map_tele.end(); iter++){
//            if (i_src % n_sims == id_sim){
//                src_ids_this_sim_tele.push_back(iter->second.id);
//                src_names_this_sim_tele.push_back(iter->second.name);
//            }
//            i_src++;
//        }
//
//        // check IP.src_ids_this_sim for this rank
//        if (myrank==0) {
//            std::cout << id_sim << "(telesmies source boundary) assigned src id(name) : ";
//            for (int i = 0; i < (int)src_ids_this_sim_tele.size(); i++) {
//                std::cout << src_ids_this_sim_tele[i] << "(" << src_names_this_sim_tele[i] << ") ";
//            }
//            std::cout << std::endl;
//        }
//
//        /////////////////
//        // for earthquake having common receiver differential traveltime, the synthetic traveltime should be computed first at each iteration
//        /////////////////
//        src_ids_this_sim_prepare.clear();
//        src_names_this_sim_prepare.clear();
//        // assign elements of src_points
//        i_src = 0;
//        for (auto iter = src_map_prepare.begin(); iter != src_map_prepare.end(); iter++){
//            if (i_src % n_sims == id_sim){
//                src_ids_this_sim_prepare.push_back(iter->second.id);
//                src_names_this_sim_prepare.push_back(iter->second.name);
//            }
//            i_src++;
//        }
//        // check IP.src_ids_this_sim for this rank
//        if (myrank==0) {
//            std::cout << id_sim << "(prepare synthetic time) assigned src id(name) : ";
//            for (int i = 0; i < (int)src_ids_this_sim_prepare.size(); i++) {
//                std::cout << src_ids_this_sim_prepare[i] << "(" << src_names_this_sim_prepare[i] << ") ";
//            }
//            std::cout << std::endl;
//        }
//
//
//        /////////////////
//        // for earthquake that will be looped in each iteraion
//        /////////////////
//        src_ids_this_sim.clear();
//        src_names_this_sim.clear();
//        // assign elements of src_points
//        i_src = 0;
//        for (auto iter = src_map.begin(); iter != src_map.end(); iter++){
//            if (i_src % n_sims == id_sim){
//                src_ids_this_sim.push_back(iter->second.id);
//                src_names_this_sim.push_back(iter->second.name);
//            }
//            i_src++;
//        }
//        // check IP.src_ids_this_sim for this rank
//        if (myrank==0) {
//            std::cout << id_sim << " assigned src id(name) : ";
//            for (int i = 0; i < (int)src_ids_this_sim.size(); i++) {
//                std::cout << src_ids_this_sim[i] << "(" << src_names_this_sim[i] << ") ";
//            }
//            std::cout << std::endl;
//        }
//
//
//        std::cout << std::endl;

////////////////////////////////////////////////////////////////

        std::cout << "end parse src_rec file" << std::endl;
    } // end of if src_rec_file_exists
}


// copy/rearrange and analyse "data_info" -> data_info_smap; data_info_smap_reloc
// dividing the data group by source name
//void InputParams::rearrange_data_info(){
//    for(int i = 0; i < (int)data_info.size(); i++){
//        DataInfo data = data_info[i];
//
//        // add absolute traveltime
//        if(data.is_src_rec){
//            data_info_smap[data.name_src].push_back(data);
//            data_info_smap_reloc[data.name_src].push_back(data);
//
//        // add common source differential traveltime
//        } else if (data.is_rec_pair){
//            data_info_smap[data.name_src_single].push_back(data);
//
//        // add common receiver differential traveltime
//        } else if (data.is_src_pair){
//            data_info_smap[data.name_src_pair[0]].push_back(data);
//            data_info_smap[data.name_src_pair[1]].push_back(data);
//        }
//    }
//}


//// # TODO: src_map_prepare is not clear
//void InputParams::generate_src_map_prepare(){
//    for(auto iter = data_info_smap.begin(); iter != data_info_smap.end(); iter++){
//
//        for (const DataInfo& data : iter->second){
//            // if this source has common source differential traveltime data
//            if (data.is_src_pair){
//                // add this source and turn to the next source
//                src_map_prepare[iter->first] = src_map[iter->first];
//                break;
//            }
//        }
//
//    }
//}

void InputParams::initialize_syn_time_map(){
    for(int i = 0; i < (int)data_info.size(); i++){
        DataInfo data = data_info[i];

        // add absolute traveltime
        if(data.is_src_rec){
            syn_time_map_sr[data.name_src][data.name_rec] = 0.0;

        // add common source differential traveltime
        } else if (data.is_rec_pair){
            syn_time_map_sr[data.name_src_single][data.name_rec_pair[0]] = 0.0;
            syn_time_map_sr[data.name_src_single][data.name_rec_pair[1]] = 0.0;

        // add common receiver differential traveltime
        } else if (data.is_src_pair){
            syn_time_map_sr[data.name_src_pair[0]][data.name_rec_single] = 0.0;
            syn_time_map_sr[data.name_src_pair[1]][data.name_rec_single] = 0.0;
        }
    }
}


void InputParams::distribute_syn_time_map(){




}

void InputParams::initialize_syn_time_list(){
    for(auto iter1 = syn_time_map_sr.begin(); iter1 != syn_time_map_sr.end(); iter1++){
        for(auto iter2 = iter1->second.begin(); iter2 != iter1->second.end(); iter2++){
            iter2->second = 0.0;
        }
    }
}

void InputParams::reduce_syn_time_list(){
    for(auto iter1 = syn_time_map_sr.begin(); iter1 != syn_time_map_sr.end(); iter1++){
        for(auto iter2 = iter1->second.begin(); iter2 != iter1->second.end(); iter2++){
            allreduce_cr_sim_single_inplace(iter2->second);
        }
    }
}

void InputParams::initialize_adjoint_source(){
    for(auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
        iter->second.adjoint_source = 0;
    }
}

void InputParams::set_adjoint_source(std::string name_rec, CUSTOMREAL adjoint_source){
    if (rec_map.find(name_rec) != rec_map.end()){
        rec_map[name_rec].adjoint_source = adjoint_source;
    } else {
        std::cout << "error !!!, undefined receiver name when adding adjoint source: " << name_rec << std::endl;
    }
}

SrcRecInfo& InputParams::get_src_map(std::string name_src){
    if (src_map.find(name_src) != src_map.end()){
        return src_map[name_src];
    } else {
        std::cout << "error !!!, undefined source name when get src " << name_src << " in src_map_nv: " << std::endl;
        abort();
    }
}

SrcRecInfo& InputParams::get_rec_map(std::string name_rec){
    if (rec_map.find(name_rec) != rec_map.end()){
        return rec_map[name_rec];
    } else {
        std::cout << "error !!!, undefined receiver name when get rec " << name_rec << " in rec_map_nv: " << std::endl;
        abort();
    }
}

void InputParams::gather_all_arrival_times_to_main(){
    // std::cout << "ckp0, id_sim: " << id_sim << ", myrank: " << myrank <<std::endl;
    if (id_sim == 1 && myrank == 0){
        for(int id : src_ids_this_sim)
            std::cout << id << ", ";
        std::cout << std::endl;
    }
    int id_src = 0;
    for (auto iter = src_map.begin(); iter != src_map.end(); iter++){
        // int id_src = iter->second.id;
        std::string name_src = iter->second.name;
        // if (id_sim == 0 && myrank == 0)
        //     std::cout << "ckp0, id_sim: " << id_sim << ", name: " << name_src << ", myrank: " << myrank << ", id_subdomain: " << id_subdomain << ", id_src:" << id_src << std::endl;

        if (subdom_main && id_subdomain==0){
            // check if the target source is calculated by this simulation group
            if (std::find(src_names_this_sim.begin(), src_names_this_sim.end(), name_src) != src_names_this_sim.end()) {
                if (id_sim == 0) {
                    // do nothing
                } else {
                    // send to main simulation group
                    for (auto iter2 = syn_time_map_sr[name_src].begin(); iter2 != syn_time_map_sr[name_src].end(); iter2++){
                        send_cr_single_sim(&(iter2->second), 0);
                    }

                }
            } else {
                if (id_sim == 0) {
                    // receive
                    int id_sim_group = id_src % n_sims;
                    for (auto iter2 = syn_time_map_sr[name_src].begin(); iter2 != syn_time_map_sr[name_src].end(); iter2++){
                        recv_cr_single_sim(&(iter2->second), id_sim_group);
                    }

                } else {
                    // do nothing
                }
            }
        }
        id_src++;
    }
}


void InputParams::write_station_correction_file(int i_inv){
    if(is_sta_correction && run_mode == DO_INVERSION) {  // if apply station correction
        station_correction_file_out = output_dir + "/station_correction_file_step_" + int2string_zero_fill(i_inv) +".dat";

        std::ofstream ofs;

        if (world_rank == 0 && subdom_main && id_subdomain==0){    // main processor of subdomain && the first id of subdoumains

            ofs.open(station_correction_file_out);

            ofs << "# stname " << "   lat   " << "   lon   " << "elevation   " << " station correction (s) " << std::endl;
            for(auto iter = rec_map_back.begin(); iter != rec_map_back.end(); iter++){
                SrcRecInfo  rec      = iter->second;
                std::string name_rec = rec.name;

                ofs << rec.name << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec.lat << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec.lon << " "
                    << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << rec.dep * -1000.0 << " "
                    << std::fixed << std::setprecision(6) << std::setw(9) << std::right << std::setfill(' ') << rec.sta_correct << " "
                    << std::endl;
            }

            ofs.close();

        }
        synchronize_all_world();
    }
}

void InputParams::write_src_rec_file(int i_inv) {

    if (src_rec_file_exist){

        std::ofstream ofs;

        // gather all arrival time info to the main process
        if (n_sims > 1)
            gather_all_arrival_times_to_main();

        if (run_mode == ONLY_FORWARD)
            src_rec_file_out = output_dir + "/src_rec_file_forward.dat";
        else if (run_mode == DO_INVERSION){
            // write out source and receiver points with current inversion iteration number
            src_rec_file_out = output_dir + "/src_rec_file_step_" + int2string_zero_fill(i_inv) +".dat";
        } else if (run_mode == TELESEIS_PREPROCESS) {
            src_rec_file_out = output_dir + "/src_rec_file_teleseis_pre.dat";
        } else if (run_mode == SRC_RELOCATION) {
            src_rec_file_out = output_dir + "/src_rec_file_src_reloc_syn.dat";
        } else {
            std::cerr << "Error: run_mode is not defined" << std::endl;
            exit(1);
        }

        int data_count = 0;
        for (int i_src = 0; i_src < (int)src_name_list.size(); i_src++){

            if (world_rank == 0 && subdom_main && id_subdomain==0){    // main processor of subdomain && the first id of subdoumains

                if (i_src == 0)
                    ofs.open(src_rec_file_out);
                else
                    ofs.open(src_rec_file_out, std::ios_base::app);

                // std::string name_src = iter->first;
                // SrcRecInfo src = iter->second;
                std::string name_src = src_name_list[i_src];
                SrcRecInfo src = src_map_back[name_src];

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

                // data line
                for(int i = 0; i < src.n_data; i++){
                    DataInfo data = data_info_back[data_count];

                    if (data.is_src_rec){       // absolute traveltime data

                        std::string name_rec = data.name_rec;
                        SrcRecInfo  rec      = rec_map_back[name_rec];
                        CUSTOMREAL  travel_time;
                        if (get_is_srcrec_swap())     // do swap
                            travel_time = syn_time_map_sr[name_rec][name_src];
                        else // undo swap
                            travel_time = syn_time_map_sr[name_src][name_rec];

                        // receiver line : id_src id_rec name_rec lat lon elevation_m phase epicentral_distance_km arival_time
                        ofs << std::setw(7) << std::right << std::setfill(' ') << i_src << " "
                            << std::setw(5) << std::right << std::setfill(' ') << rec.id << " "
                            << rec.name << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << rec.lat << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << rec.lon << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << -1.0*rec.dep*1000.0 << " "
                            << data.phase << " "
                            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << travel_time << " "
                            << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                            << std::endl;

                    } else if (data.is_rec_pair){   // common source differential traveltime

                        std::string name_rec1 = data.name_rec_pair[0];
                        SrcRecInfo  rec1      = rec_map_back[name_rec1];
                        std::string name_rec2 = data.name_rec_pair[1];
                        SrcRecInfo  rec2      = rec_map_back[name_rec2];
                        CUSTOMREAL cs_dif_travel_time;

                        if (get_is_srcrec_swap())      // do swap
                            cs_dif_travel_time = syn_time_map_sr[name_rec1][name_src] - syn_time_map_sr[name_rec2][name_src];
                        else // undo swap
                            cs_dif_travel_time = syn_time_map_sr[name_src][name_rec1] - syn_time_map_sr[name_src][name_rec2];

                        // receiver pair line : id_src id_rec1 name_rec1 lat1 lon1 elevation_m1 id_rec2 name_rec2 lat2 lon2 elevation_m2 phase differential_arival_time
                        ofs << std::setw(7) << std::right << std::setfill(' ') <<  i_src << " "
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

                    } else if (data.is_src_pair){   // common receiver differential traveltime
                        // not ready
                    }

                    data_count++;
                }

                ofs.close();
            }

            // synchro
            //synchronize_all_world(); // dead lock here because src_name_list is not the same for all processors

        } // end of for (int i_src = 0; i_src < (int)src_name_list.size(); i_src++)

        // only for source relocation, output relocated observational data for tomography
        if (run_mode == SRC_RELOCATION) {
            src_rec_file_out = output_dir + "/src_rec_file_src_reloc_obs.dat";

            int data_count = 0;
            // for (auto iter = src_map_back_nv.begin(); iter != src_map_back_nv.end(); iter++){
            for (int i_src = 0; i_src < (int)src_name_list.size(); i_src++){
                if (world_rank == 0 && subdom_main && id_subdomain==0){    // main processor of subdomain && the first id of subdoumains
                    // if (iter == src_map_back_nv.begin())
                    if (i_src == 0)
                        ofs.open(src_rec_file_out);
                    else
                        ofs.open(src_rec_file_out, std::ios_base::app);

                    // std::string name_src = iter->first;
                    // SrcRecInfo src = iter->second;
                    std::string name_src = src_name_list[i_src];
                    SrcRecInfo src = src_map_back[name_src];

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
                    // data line
                    for(int i = 0; i < src.n_data; i++){
                        DataInfo data = data_info_back[data_count];
                        if (data.is_src_rec){       // absolute traveltime data
                            std::string name_rec = data.name_rec;
                            SrcRecInfo rec = rec_map_back[name_rec];
                            CUSTOMREAL travel_time_obs;
                            // swapped
                            travel_time_obs = data.travel_time_obs - rec_map[name_src].tau_opt;

                            // receiver line : id_src id_rec name_rec lat lon elevation_m phase epicentral_distance_km arival_time
                            ofs << std::setw(7) << std::right << std::setfill(' ') << src.id << " "
                                << std::setw(5) << std::right << std::setfill(' ') << rec.id << " "
                                << rec.name << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << rec.lat << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << rec.lon << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << -1.0*rec.dep*1000.0 << " "
                                << data.phase << " "
                                << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << travel_time_obs << " "
                                << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                                << std::endl;
                        } else if (data.is_rec_pair){   // common source differential traveltime
                            std::string name_rec1 = data.name_rec_pair[0];
                            SrcRecInfo  rec1      = rec_map_back[name_rec1];
                            std::string name_rec2 = data.name_rec_pair[1];
                            SrcRecInfo  rec2      = rec_map_back[name_rec2];
                            CUSTOMREAL  cs_dif_travel_time;

                            // swapped
                            cs_dif_travel_time = data.cs_dif_travel_time_obs;

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
                        } else if (data.is_src_pair){   // common receiver differential traveltime
                            // not ready
                        }

                        data_count++;
                    }

                    ofs.close();

                }
            }

        } // end of run_mode == SRC_RELOCATION
    } // end of src_rec_file_exist

    // synchro
    synchronize_all_world();

}


// check contradictory parameters
void InputParams::check_contradictions(){

    // if run_mode == 0 then the max_iter should be 1
    if (run_mode == ONLY_FORWARD && max_iter_inv > 1){
        std::cout << "Warning: run_mode = 0, max_iter should be 1" << std::endl;
        max_iter_inv = 1;
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


void InputParams::allocate_memory_tele_boundaries(int np, int nt, int nr, std::string name_src, \
        bool i_first_in, bool i_last_in, bool j_first_in, bool j_last_in, bool k_first_in) {

    i_first = i_first_in;
    i_last  = i_last_in;
    j_first = j_first_in;
    j_last  = j_last_in;
    k_first = k_first_in;

    // allocate memory for teleseismic boundary sources
    SrcRecInfo& src = get_src_point(name_src);

    // check if this src is teleseismic source
    if (src.is_out_of_region){
        // North boundary
        if (j_last)
            src.arr_times_bound_N = (CUSTOMREAL*)malloc(sizeof(CUSTOMREAL)*np*nr*N_LAYER_SRC_BOUND);
        // South boundary
        if (j_first)
            src.arr_times_bound_S = (CUSTOMREAL*)malloc(sizeof(CUSTOMREAL)*np*nr*N_LAYER_SRC_BOUND);
        // East boundary
        if (i_last)
            src.arr_times_bound_E = (CUSTOMREAL*)malloc(sizeof(CUSTOMREAL)*nt*nr*N_LAYER_SRC_BOUND);
        // West boundary
        if (i_first)
            src.arr_times_bound_W = (CUSTOMREAL*)malloc(sizeof(CUSTOMREAL)*nt*nr*N_LAYER_SRC_BOUND);
        // Bottom boundary
        if (k_first)
            src.arr_times_bound_Bot = (CUSTOMREAL*)malloc(sizeof(CUSTOMREAL)*nt*np*N_LAYER_SRC_BOUND);

        // boundary source flag
        src.is_bound_src = (bool*)malloc(sizeof(bool)*5);
    }

}

// station correction kernel (need revise)
void InputParams::station_correction_update(CUSTOMREAL stepsize){
    if (!is_sta_correction)
        return;

    // station correction kernel is generated in the main process and sent the value to all other processors

    // step 1, gather all arrival time info to the main process
    if (n_sims > 1){
        gather_all_arrival_times_to_main();
    }

    // do it in the main processor
    if (id_sim == 0){

        // step 2 initialize the kernel K_{\hat T_i}
        for (auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
            iter->second.sta_correct_kernel = 0.0;
        }
        CUSTOMREAL max_kernel = 0.0;

        // step 3, calculate the kernel
        for (auto& data : data_info){     // loop all data related to this source

            if (data.is_src_rec){   // absolute traveltime
                std::cout << "teleseismic data, absolute traveltime is not supported now" << std::endl;
            } else if (data.is_src_pair) {  // common receiver differential traveltime
                std::cout << "teleseismic data, common receiver differential traveltime is not supported now" << std::endl;
            } else if (data.is_rec_pair) {  // common source differential traveltime
                std::string name_src = data.name_src_single;
                std::string name_rec1 = data.name_rec_pair[0];
                std::string name_rec2  = data.name_rec_pair[1];

                CUSTOMREAL syn_dif_time = syn_time_map_sr[name_src][name_rec1] - syn_time_map_sr[name_src][name_rec2];
                CUSTOMREAL obs_dif_time = data.cs_dif_travel_time_obs;
                rec_map[name_rec1].sta_correct_kernel += _2_CR *(syn_dif_time - obs_dif_time \
                            + rec_map[name_rec1].sta_correct - rec_map[name_rec2].sta_correct)*data.weight;
                rec_map[name_rec2].sta_correct_kernel -= _2_CR *(syn_dif_time - obs_dif_time \
                            + rec_map[name_rec1].sta_correct - rec_map[name_rec2].sta_correct)*data.weight;
                max_kernel = std::max(max_kernel,rec_map[name_rec1].sta_correct_kernel);
                max_kernel = std::max(max_kernel,rec_map[name_rec2].sta_correct_kernel);
            }
        }

        // step 4, update station correction
        for (auto iter = rec_map.begin(); iter!=rec_map.end(); iter++){
            iter->second.sta_correct += iter->second.sta_correct_kernel / (-max_kernel) * stepsize;
        }
    }

    // step 5, broadcast the station correction all all procesors
    for (auto iter = rec_map.begin(); iter!=rec_map.end(); iter++){
        broadcast_cr_single_inter_sim(iter->second.sta_correct,0);
        broadcast_cr_single_sub(iter->second.sta_correct,0);
    }

}

void InputParams::modift_swapped_source_location() {
    for(auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
        src_map_back[iter->first].lat   =   iter->second.lat;
        src_map_back[iter->first].lon   =   iter->second.lon;
        src_map_back[iter->first].dep   =   iter->second.dep;
        src_map_back[iter->first].sec   =   iter->second.sec + iter->second.tau_opt;
    }
}


void InputParams::communicate_travel_times() {
    // for common receiver double difference data, the source locates at the different simultaneous group.
    // in this case this function is used to communicate the travel times between different simultaneous groups

    if (subdom_main) {
        // find the sources which are not calculated in this simultaneous group
        // by comparing data_info and src_ids_this_sim

        //

    }

}