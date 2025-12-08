#ifndef MAIN_ROUTINES_CALLING_H
#define MAIN_ROUTINES_CALLING_H

#include <iostream>
#include <memory>
#include "mpi_funcs.h"
#include "config.h"
#include "utils.h"
#include "input_params.h"
#include "grid.h"
#include "io.h"
#include "main_routines_inversion_mode.h"
// #include "model_optimization_routines.h"
#include "main_routines_earthquake_relocation.h"
#include "iterator_selector.h"
#include "iterator.h"
#include "iterator_legacy.h"
#include "iterator_level.h"
#include "source.h"
#include "receiver.h"
// #include "kernel.h"
// #include "model_update.h"
// #include "lbfgs.h"
#include "objective_function_utils.h"
#include "timer.h"
#include "oneD_inversion.h"
#include "optimizer.h"
#include "optimizer_gd.h"
#include "optimizer_bfgs.h"


// run forward-only or inversion mode
// run mode: 0 or run mode: 1
inline void run_forward_only_or_inversion(InputParams &IP, Grid &grid, IO_utils &io) {

    // for check if the current source is the first source
    bool first_src = true;

    if(myrank == 0)
        std::cout << "id_sim: " << id_sim << ", size of src_map: " << IP.src_map.size() << std::endl;

    // estimate running time
    Timer timer("Forward_or_inversion", true);

    // prepare objective_function file
    std::ofstream out_main; // close() is not mandatory
    prepare_header_line(IP, out_main);

    if (id_sim==0) {
        io.prepare_grid_inv_xdmf(0);

        //io.change_xdmf_obj(0); // change xmf file for next src
        io.change_group_name_for_model();

        // write out model info
        io.write_vel(grid, 0);
        io.write_xi( grid, 0);
        io.write_eta(grid, 0);
        //io.write_zeta(grid, i_inv); // TODO

        if (IP.get_verbose_output_level()){
            io.write_a(grid,   0);
            io.write_b(grid,   0);
            io.write_c(grid,   0);
            io.write_f(grid,   0);
            io.write_fun(grid, 0);
        }

        // // output model_parameters_inv_0000.dat
        // if (IP.get_if_output_model_dat())
        //     io.write_concerning_parameters(grid, 0, IP);
    }


    // output station correction file (only for teleseismic differential data)
    IP.write_station_correction_file(0);

    synchronize_all_world();

    /////////////////////
    // loop for inversion
    /////////////////////

    // select an optimizer
    std::unique_ptr<Optimizer> optimizer;  // optimizer pointer
    if (optim_method == GRADIENT_DESCENT){
        optimizer = std::make_unique<Optimizer_gd>(IP);
    } else if (optim_method == HALVE_STEPPING_MODE){
        std::cout << "Halve stepping mode not implemented yet." << std::endl;
        exit(1);
    } else if (optim_method == LBFGS_MODE){
        // std::cout << "LBFGS mode not implemented yet." << std::endl;
        // must make output_kernel: true and output_in_process: true in InputParams, because the previous kernels are needed to calculate the gradient difference
        optimizer = std::make_unique<Optimizer_bfgs>(IP);
    } else {
        std::cout << "Unknown optimization method: " << optim_method << std::endl;
        exit(1);
    }
    

    bool line_search_mode = false; // if true, run_simulation_one_step skips adjoint simulation and only calculates objective function value

    // objective function for all src
    CUSTOMREAL v_obj = 0.0, old_v_obj = 0.0;
    std::vector<CUSTOMREAL> v_obj_misfit(20, 0.0);

    for (int i_inv = 0; i_inv < IP.get_max_iter_inv(); i_inv++) {

        if(myrank == 0 && id_sim ==0){
            std::cout << "iteration " << i_inv << " starting ... " << std::endl;
        }

        old_v_obj = v_obj;

        // prepare inverstion iteration group in xdmf file
        io.prepare_grid_inv_xdmf(i_inv);

        ///////////////////////////////////////////////////////
        // run (forward and adjoint) simulation for each source
        ///////////////////////////////////////////////////////

        // run forward and adjoint simulation and calculate current objective function value and sensitivity kernel for all sources
        line_search_mode = false;
        // skip for the mode with sub-iteration
        // if (i_inv > 0 && optim_method != GRADIENT_DESCENT) {
            
        // } else {
        //     bool is_save_T = false;
        //     v_obj_misfit = run_simulation_one_step(IP, grid, io, i_inv, line_search_mode, is_save_T);
        //     v_obj = v_obj_misfit[0];
        // }
        // (to do) if line search is applied, run_simulation_one_step should be called inside model_update function of optimizer class
        v_obj_misfit = run_simulation_one_step(IP, grid, io, i_inv, false, false);
        v_obj = v_obj_misfit[0];


        // wait for all processes to finish
        synchronize_all_world();

        // check if v_obj is nan
        if (std::isnan(v_obj)) {
            if (myrank == 0)
                std::cout << "v_obj is nan, stop inversion" << std::endl;
            // stop inversion
            break;
        }

        // output src rec file with the result arrival times
        if (IP.get_if_output_in_process_data() || i_inv == IP.get_max_iter_inv()-1 || i_inv==0) {
            IP.write_src_rec_file(i_inv,0);
        }

        ///////////////
        // model update
        ///////////////
        if(myrank == 0 && id_sim ==0)
            std::cout << "model update starting ... " << std::endl;

        if (IP.get_run_mode() == DO_INVERSION) {
            optimizer->model_update(IP, grid, io, i_inv, v_obj, old_v_obj, false);
            // if (optim_method == GRADIENT_DESCENT)
            //     optimizer_gd.model_update(IP, grid, io, i_inv, v_obj, old_v_obj);
            //     // model_optimize(IP, grid, io, i_inv, v_obj, old_v_obj, first_src, out_main);
            // else if (optim_method == HALVE_STEPPING_MODE)
            //     v_obj_misfit = model_optimize_halve_stepping(IP, grid, io, i_inv, v_obj, first_src, out_main);
            // else if (optim_method == LBFGS_MODE) {
            //     bool found_next_step = model_optimize_lbfgs(IP, grid, io, i_inv, v_obj, first_src, out_main);
            //     if (!found_next_step)
            //         goto end_of_inversion;
            // }
        }

        // output station correction file (only for teleseismic differential data)
        IP.write_station_correction_file(i_inv + 1);

        // output objective function
        write_objective_function(IP, i_inv, v_obj_misfit, out_main, "model update");

        // since model is update. The written traveltime field should be discraded.
        // initialize is_T_written_into_file
        for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
            const std::string name_sim_src = IP.get_src_name(i_src);

            if (proc_store_srcrec) // only proc_store_srcrec has the src_map object
                IP.src_map[name_sim_src].is_T_written_into_file = false;
        }

        // output updated model (done in optimizer->model_update)


        // writeout temporary xdmf file
        io.update_xdmf_file();

        // wait for all processes to finish
        synchronize_all_world();


        // estimate running time
        CUSTOMREAL time_elapsed = timer.get_t();
        if (id_sim == 0 && myrank == 0 && i_inv < IP.get_max_iter_inv()-1) {
            const time_t end_time_estimated = time_elapsed / (i_inv + 1) * (IP.get_max_iter_inv() - i_inv - 1) + timer.get_start();
            auto will_run_time = (int)(time_elapsed/(i_inv + 1) * (IP.get_max_iter_inv() - i_inv - 1));

            std::cout << std::endl;
            std::cout << "The program begins at " << timer.get_start_t() << std::endl;
            std::cout << "Iteration (" << i_inv + 1 << "/" << IP.get_max_iter_inv() << ") finished at " << time_elapsed << " seconds" << std::endl;
            std::cout << i_inv + 1 << " iterations run " << timer.get_t() << " seconds, the rest of " << IP.get_max_iter_inv() - i_inv - 1 << " iterations require " << will_run_time << " seconds." << std::endl;
            std::cout << "The program is estimated to stop at " << timer.get_utc_from_time_t(end_time_estimated) << std::endl;
            std::cout << std::endl;
        }

        // output current state of the model
        if (IP.get_if_output_final_model()) {
            // io.write_final_model(grid, IP);
            io.write_merged_model(grid, IP, "final_model.h5");
        }
        if (IP.get_if_output_middle_model()) {
            std::string tmp_fname = "middle_model_step_" + int2string_zero_fill(i_inv + 1) + ".h5";
            io.write_merged_model(grid, IP, tmp_fname);
        }


    } // end loop inverse

// end_of_inversion:

    // close xdmf file
    io.finalize_data_output_file();

    timer.stop_timer();
    if (id_sim == 0 && myrank == 0) {
        std::cout << std::endl;
        std::cout << "The program begin at " << timer.get_start_t() << std::endl;
        std::cout << "Forward_or_inversion end at " << timer.get_end_t() << std::endl;
        std::cout << "It has run " << timer.get_elapsed_t() << " seconds in total." << std::endl;
        std::cout << std::endl;
    }
}


// run earthquake relocation mode
// run mode: 2
inline void run_earthquake_relocation(InputParams& IP, Grid& grid, IO_utils& io) {

    Receiver recs;

    // calculate traveltime for each receiver (swapped from source) and write in output file
    calculate_traveltime_for_all_src_rec(IP, grid, io);

    // prepare output for iteration status
    std::ofstream out_main; // close() is not mandatory
    prepare_header_line(IP, out_main);

    // objective function and its gradient
    CUSTOMREAL v_obj      = 999999999.0;

    int        i_iter     = 0;

    std::vector<CUSTOMREAL> v_obj_misfit;

    // iterate
    while (true) {

        v_obj      = 0.0;

        // calculate gradient of objective function at sources
        v_obj_misfit = calculate_gradient_objective_function(IP, grid, io, i_iter);
        v_obj = v_obj_misfit[0];

        // update source location
        recs.update_source_location(IP, grid);

        synchronize_all_world();

        // check convergence
        bool finished = false;

        if (subdom_main && id_subdomain==0) {
            if (i_iter >= N_ITER_MAX_SRC_RELOC){
                std::cout << "Finished relocation because iteration number exceeds the maximum " << N_ITER_MAX_SRC_RELOC << std::endl;
                finished = true;
            }
            allreduce_bool_single_inplace_sim(finished); //LAND
        }

        synchronize_all_world();

        // check if all processes have finished
        broadcast_bool_inter_and_intra_sim(finished, 0);

        synchronize_all_world();

        // output location information
        if(id_sim == 0 && myrank == 0){
            // write objective function
            std::cout << "iteration: " << i_iter << ", objective function: "              << v_obj << std::endl;
        }

        // write objective functions
        write_objective_function(IP, i_iter, v_obj_misfit, out_main, "relocation");

        // write out new src_rec_file
        if (IP.get_if_output_in_process_data() || i_iter == N_ITER_MAX_SRC_RELOC-1 || i_iter==0){
            IP.write_src_rec_file(0,i_iter);
        }

        // modify the receiver's location for output
        IP.modify_swapped_source_location();


        if (finished)
            break;

        // new iteration
        i_iter++;
    }

    // modify the receiver's location
    IP.modify_swapped_source_location();
    // write out new src_rec_file
    IP.write_src_rec_file(0,i_iter);
    // close xdmf file
    io.finalize_data_output_file();

}


// run earthquake relocation mode
// run mode: 3
inline void run_inversion_and_relocation(InputParams& IP, Grid& grid, IO_utils& io) {

    Timer timer("Inv_and_reloc", true);

    /////////////////////
    // preparation of model update
    /////////////////////

    // for check if the current source is the first source
    bool first_src = true;

    if(myrank == 0)
        std::cout << "id_sim: " << id_sim << ", size of src_map: " << IP.src_map.size() << std::endl;

    // prepare objective_function file
    std::ofstream out_main; // close() is not mandatory
    prepare_header_line(IP, out_main);

    if (id_sim==0) {
        // prepare inverstion iteration group in xdmf file
        io.prepare_grid_inv_xdmf(0);

        //io.change_xdmf_obj(0); // change xmf file for next src
        io.change_group_name_for_model();

        // write out model info
        io.write_vel(grid, 0);
        io.write_xi( grid, 0);
        io.write_eta(grid, 0);
        //io.write_zeta(grid, i_inv); // TODO

        if (IP.get_verbose_output_level()){
            io.write_a(grid,   0);
            io.write_b(grid,   0);
            io.write_c(grid,   0);
            io.write_f(grid,   0);
            io.write_fun(grid, 0);
        }

        // // output model_parameters_inv_0000.dat
        // if (IP.get_if_output_model_dat())
        //     io.write_concerning_parameters(grid, 0, IP);
    }

    // output station correction file (only for teleseismic differential data)
    IP.write_station_correction_file(0);

    synchronize_all_world();

    bool line_search_mode = false; // if true, run_simulation_one_step skips adjoint simulation and only calculates objective function value

    /////////////////////
    // preparation of relocation
    /////////////////////

    Receiver recs;

    /////////////////////
    // initializae objective function
    /////////////////////

    CUSTOMREAL v_obj = 0.0,         old_v_obj = 10000000000.0;
    std::vector<CUSTOMREAL> v_obj_misfit(20, 0.0);

    /////////////////////
    // select an optimizer for model update
    /////////////////////
    std::unique_ptr<Optimizer> optimizer;  // optimizer pointer
    if (optim_method == GRADIENT_DESCENT){
        optimizer = std::make_unique<Optimizer_gd>(IP);
    } else if (optim_method == HALVE_STEPPING_MODE){
        std::cout << "Halve stepping mode not implemented yet." << std::endl;
        exit(1);
    } else if (optim_method == LBFGS_MODE){
        // std::cout << "LBFGS mode not implemented yet." << std::endl;
        // must make output_kernel: true and output_in_process: true in InputParams, because the previous kernels are needed to calculate the gradient difference
        optimizer = std::make_unique<Optimizer_bfgs>(IP);
    } else {
        std::cout << "Unknown optimization method: " << optim_method << std::endl;
        exit(1);
    }



    if (inv_mode == ITERATIVE){

        /////////////////////
        // main loop for model update and relocation
        /////////////////////

        int model_update_step = 0;
        int relocation_step = 0;

        for (int i_loop = 0; i_loop < IP.get_max_loop_mode0(); i_loop++){
            /////////////////////
            // stage 1, update model parameters
            /////////////////////

            for (int one_loop_i_inv = 0; one_loop_i_inv < IP.get_model_update_N_iter(); one_loop_i_inv++){
                // the actural step index of model update
                int i_inv = i_loop * IP.get_model_update_N_iter() + one_loop_i_inv;

                if(myrank == 0 && id_sim ==0){
                    std::cout   << std::endl;
                    std::cout   << "loop " << i_loop+1 << ", model update iteration " << one_loop_i_inv+1
                                << " ( the " << i_inv+1 << "-th model update) starting ... " << std::endl;
                    std::cout   << std::endl;
                }


                // prepare inverstion iteration group in xdmf file
                io.prepare_grid_inv_xdmf(i_inv);

                ///////////////////////////////////////////////////////
                // run (forward and adjoint) simulation for each source
                ///////////////////////////////////////////////////////

                // run forward and adjoint simulation and calculate current objective function value and sensitivity kernel for all sources
                line_search_mode = false;
                // skip for the mode with sub-iteration
                if (i_inv > 0 && optim_method != GRADIENT_DESCENT) {
                } else {
                    bool is_save_T = false;
                    v_obj_misfit = run_simulation_one_step(IP, grid, io, i_inv, line_search_mode, is_save_T);
                    v_obj = v_obj_misfit[0];
                }

                // wait for all processes to finish
                synchronize_all_world();

                // check if v_obj is nan
                if (std::isnan(v_obj)) {
                    if (myrank == 0)
                        std::cout << "v_obj is nan, stop inversion" << std::endl;
                    // stop inversion
                    break;
                }

                ///////////////
                // model update
                ///////////////

                optimizer->model_update(IP, grid, io, i_inv, v_obj, old_v_obj, false);

                // if (optim_method == GRADIENT_DESCENT)
                //     model_optimize(IP, grid, io, i_inv, v_obj, old_v_obj, first_src, out_main);
                // else if (optim_method == HALVE_STEPPING_MODE)
                //     v_obj_misfit = model_optimize_halve_stepping(IP, grid, io, i_inv, v_obj, first_src, out_main);
                // else if (optim_method == LBFGS_MODE) {
                //     bool found_next_step = model_optimize_lbfgs(IP, grid, io, i_inv, v_obj, first_src, out_main);

                //     if (!found_next_step)
                //         break;
                // }

                // define old_v_obj
                old_v_obj = v_obj;

                // output objective function
                write_objective_function(IP, i_inv, v_obj_misfit, out_main, "model update");

                // since model is update. The written traveltime field should be discraded.
                // initialize is_T_written_into_file
                for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
                    const std::string name_sim_src = IP.get_src_name(i_src);

                    if (proc_store_srcrec) // only proc_store_srcrec has the src_map object
                        IP.src_map[name_sim_src].is_T_written_into_file = false;
                }

                // output updated model
                if (id_sim==0) {
                    //io.change_xdmf_obj(0); // change xmf file for next src
                    io.change_group_name_for_model();

                    // write out model info
                    if (IP.get_if_output_in_process() || i_inv >= IP.get_max_loop_mode0()*IP.get_model_update_N_iter() - 2){
                        io.write_vel(grid, i_inv+1);
                        io.write_xi( grid, i_inv+1);
                        io.write_eta(grid, i_inv+1);
                    }
                    //io.write_zeta(grid, i_inv); // TODO

                    if (IP.get_verbose_output_level()){
                        io.write_a(grid,   i_inv+1);
                        io.write_b(grid,   i_inv+1);
                        io.write_c(grid,   i_inv+1);
                        io.write_f(grid,   i_inv+1);
                        io.write_fun(grid, i_inv+1);
                    }

                    // // output model_parameters_inv_0000.dat
                    // if (IP.get_if_output_model_dat()
                    // && (IP.get_if_output_in_process() || i_inv >= IP.get_max_loop_mode0()*IP.get_model_update_N_iter() - 2))
                    //     io.write_concerning_parameters(grid, i_inv + 1, IP);

                } // end output updated model

                // writeout temporary xdmf file
                io.update_xdmf_file();

                // write src rec files
                if (IP.get_if_output_in_process_data()){
                    IP.write_src_rec_file(model_update_step,relocation_step);
                } else if (i_inv == IP.get_max_loop_mode0() * IP.get_model_update_N_iter()-1 || i_inv==0) {
                    IP.write_src_rec_file(model_update_step,relocation_step);
                }
                model_update_step += 1;

                // wait for all processes to finish
                synchronize_all_world();

                // output current state of the model
                if (IP.get_if_output_final_model()) {
                    // io.write_final_model(grid, IP);
                    io.write_merged_model(grid, IP, "final_model.h5");
                }
                if (IP.get_if_output_middle_model()) {
                    std::string tmp_fname = "middle_model_step_" + int2string_zero_fill(i_inv + 1) + ".h5";
                    io.write_merged_model(grid, IP, tmp_fname);
                }

            }   // end model update in one loop


            /////////////////////
            // stage 2, update earthquake locations
            /////////////////////

            if(myrank == 0 && id_sim ==0){
                std::cout   << std::endl;
                std::cout   << "loop " << i_loop+1 << ", computing traveltime field for relocation" << std::endl;
                std::cout   << std::endl;
            }

            // calculate traveltime for each receiver (swapped from source) and write in output file
            if (IP.get_relocation_N_iter() > 0) // we really do relocation
                calculate_traveltime_for_all_src_rec(IP, grid, io);


            // initilize all earthquakes
            if (proc_store_srcrec){
                for(auto iter = IP.rec_map.begin(); iter != IP.rec_map.end(); iter++){
                    iter->second.is_stop = false;
                }
            }

            // iterate
            for (int one_loop_i_iter = 0; one_loop_i_iter < IP.get_relocation_N_iter(); one_loop_i_iter++){
                int i_iter = i_loop * IP.get_relocation_N_iter() + one_loop_i_iter;

                if(myrank == 0 && id_sim ==0){
                    std::cout   << std::endl;
                    std::cout   << "loop " << i_loop+1 << ", relocation iteration " << one_loop_i_iter+1
                                << " ( the " << i_iter+1 << "-th relocation) starting ... " << std::endl;
                    std::cout   << std::endl;
                }


                // calculate gradient of objective function at sources
                v_obj_misfit = calculate_gradient_objective_function(IP, grid, io, i_iter);
                v_obj = v_obj_misfit[0];

                // update source location
                recs.update_source_location(IP, grid);

                synchronize_all_world();


                // output location information
                if(id_sim == 0 && myrank == 0){
                    std::cout << "objective function: "              << v_obj << std::endl;
                }

                // write objective functions
                write_objective_function(IP, i_iter, v_obj_misfit, out_main, "relocation");

                // write out new src_rec_file
                if (IP.get_if_output_in_process_data()){
                    IP.write_src_rec_file(model_update_step,relocation_step);
                } else if (i_iter == IP.get_max_loop_mode0() * IP.get_relocation_N_iter()-1 || i_iter==0) {
                    IP.write_src_rec_file(model_update_step,relocation_step);
                }

                // modify the receiver's location for output
                IP.modify_swapped_source_location();

                relocation_step += 1;
            } // end relocation loop

            grid.rejuvenate_abcf();     // (a,b/r^2,c/(r^2*cos^2),f/(r^2*cos)) -> (a,b,c,f)

            // estimate running time
            CUSTOMREAL time_elapsed = timer.get_t();
            if (id_sim == 0 && myrank == 0 && i_loop < IP.get_max_loop_mode0()-1) {
                const time_t end_time_estimated = time_elapsed / (i_loop + 1) * (IP.get_max_loop_mode0() - i_loop - 1) + timer.get_start();
                auto will_run_time = (int)(time_elapsed/(i_loop + 2) * (IP.get_max_loop_mode0() - i_loop - 1));

                std::cout << std::endl;
                std::cout << "The program begins at " << timer.get_start_t() << std::endl;
                std::cout << "Loop (" << i_loop + 1 << "/" << IP.get_max_loop_mode0() << ") finished at " << time_elapsed << " seconds" << std::endl;
                std::cout << i_loop + 1 << " loop run " << timer.get_t() << " seconds, the rest of " << IP.get_max_loop_mode0() - i_loop - 1 << " iterations require " << will_run_time << " seconds." << std::endl;
                std::cout << "The program is estimated to stop at " << timer.get_utc_from_time_t(end_time_estimated) << std::endl;
                std::cout << std::endl;
            }

        } // end loop for model update and relocation
    } else if (inv_mode == SIMULTANEOUS) {

        for (int i_loop = 0; i_loop < IP.get_max_loop_mode1(); i_loop++){

            if(myrank == 0 && id_sim ==0){
                std::cout   << std::endl;
                std::cout   << "loop " << i_loop+1 << ", model update and relocation starting ... " << std::endl;
                std::cout   << std::endl;
            }

            /////////////////////
            // stage 1, update model parameters
            /////////////////////


            // prepare inversion iteration group in xdmf file
            io.prepare_grid_inv_xdmf(i_loop);

                ///////////////////////////////////////////////////////
                // run (forward and adjoint) simulation for each source
                ///////////////////////////////////////////////////////

            // run forward and adjoint simulation and calculate current objective function value and sensitivity kernel for all sources
            line_search_mode = false;
            // skip for the mode with sub-iteration
            if (i_loop > 0 && optim_method != GRADIENT_DESCENT) {
            } else {
                bool is_save_T = true;
                v_obj_misfit = run_simulation_one_step(IP, grid, io, i_loop, line_search_mode, is_save_T);
                v_obj = v_obj_misfit[0];
            }

            // wait for all processes to finish
            synchronize_all_world();

            // check if v_obj is nan
            if (std::isnan(v_obj)) {
                if (myrank == 0)
                    std::cout << "v_obj is nan, stop inversion" << std::endl;

                // stop inversion
                break;
            }

            ///////////////
            // model update
            ///////////////

            optimizer->model_update(IP, grid, io, i_loop, v_obj, old_v_obj, false);

            // if (optim_method == GRADIENT_DESCENT)
            //     model_optimize(IP, grid, io, i_loop, v_obj, old_v_obj, first_src, out_main);
            // else if (optim_method == HALVE_STEPPING_MODE)
            //     v_obj_misfit = model_optimize_halve_stepping(IP, grid, io, i_loop, v_obj, first_src, out_main);
            // else if (optim_method == LBFGS_MODE) {
            //     bool found_next_step = model_optimize_lbfgs(IP, grid, io, i_loop, v_obj, first_src, out_main);

            //     if (!found_next_step)
            //         break;
            // }


            // output objective function (model update part)
            write_objective_function(IP, i_loop, v_obj_misfit, out_main, "model update");

            /////////////////////
            // stage 2, update earthquake locations
            /////////////////////


            // initilize all earthquakes
            if (proc_store_srcrec){
                for(auto iter = IP.rec_map.begin(); iter != IP.rec_map.end(); iter++){
                    iter->second.is_stop = false;
                }
            }

            // calculate gradient of objective function at sources
            v_obj_misfit = calculate_gradient_objective_function(IP, grid, io, i_loop);

            // update source location
            recs.update_source_location(IP, grid);

            synchronize_all_world();

            // output objective function (relocation part)
            write_objective_function(IP, i_loop, v_obj_misfit, out_main, "relocation");


            // define old_v_obj
            old_v_obj   = v_obj;


            // since model is update. The written traveltime field should be discraded.
            // initialize is_T_written_into_file
            for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
                const std::string name_sim_src = IP.get_src_name(i_src);

                if (proc_store_srcrec) // only proc_store_srcrec has the src_map object
                    IP.src_map[name_sim_src].is_T_written_into_file = false;
            }

            // output updated model
            if (id_sim==0) {
                //io.change_xdmf_obj(0); // change xmf file for next src
                io.change_group_name_for_model();

                // write out model info
                if (IP.get_if_output_in_process() || i_loop >= IP.get_max_loop_mode1() - 2){
                    io.write_vel(grid, i_loop+1);
                    io.write_xi( grid, i_loop+1);
                    io.write_eta(grid, i_loop+1);
                }
                //io.write_zeta(grid, i_inv); // TODO

                if (IP.get_verbose_output_level()){
                    io.write_a(grid,   i_loop+1);
                    io.write_b(grid,   i_loop+1);
                    io.write_c(grid,   i_loop+1);
                    io.write_f(grid,   i_loop+1);
                    io.write_fun(grid, i_loop+1);
                }

                // // output model_parameters_inv_0000.dat
                // if (IP.get_if_output_model_dat()
                // && (IP.get_if_output_in_process() || i_loop >= IP.get_max_loop_mode1() - 2))
                //     io.write_concerning_parameters(grid, i_loop + 1, IP);

            } // end output updated model

            // writeout temporary xdmf file
            io.update_xdmf_file();

            // write src rec files
            if (IP.get_if_output_in_process_data()){
                IP.write_src_rec_file(i_loop,i_loop);
            } else if (i_loop == IP.get_max_loop_mode1() -1 || i_loop==0) {
                IP.write_src_rec_file(i_loop,i_loop);
            }

            // modify the receiver's location for output
            IP.modify_swapped_source_location();

            // wait for all processes to finish
            synchronize_all_world();

            grid.rejuvenate_abcf();     // (a,b/r^2,c/(r^2*cos^2),f/(r^2*cos)) -> (a,b,c,f)

            // estimate running time
            CUSTOMREAL time_elapsed = timer.get_t();
            if (id_sim == 0 && myrank == 0 && i_loop < IP.get_max_loop_mode1()-1) {
                const time_t end_time_estimated = time_elapsed / (i_loop + 1) * (IP.get_max_loop_mode1() - i_loop - 1) + timer.get_start();
                auto will_run_time = (int)(time_elapsed/(i_loop + 2) * (IP.get_max_loop_mode1() - i_loop - 1));

                std::cout << std::endl;
                std::cout << "The program begins at " << timer.get_start_t() << std::endl;
                std::cout << "Loop (" << i_loop + 1 << "/" << IP.get_max_loop_mode1() << ") finished at " << time_elapsed << " seconds" << std::endl;
                std::cout << i_loop + 1 << " loop run " << timer.get_t() << " seconds, the rest of " << IP.get_max_loop_mode1() - i_loop - 1 << " iterations require " << will_run_time << " seconds." << std::endl;
                std::cout << "The program is estimated to stop at " << timer.get_utc_from_time_t(end_time_estimated) << std::endl;
                std::cout << std::endl;
            }

            // output current state of the model
            if (IP.get_if_output_final_model()) {
                // io.write_final_model(grid, IP);
                io.write_merged_model(grid, IP, "final_model.h5");
            }
            if (IP.get_if_output_middle_model()) {
                std::string tmp_fname = "middle_model_step_" + int2string_zero_fill(i_loop + 1) + ".h5";
                io.write_merged_model(grid, IP, tmp_fname);
            }
        } // end loop for model update and relocation

    } else {
        std::cout << "Error: inv_mode is not defined" << std::endl;
        exit(1);
    }
    // close xdmf file
    io.finalize_data_output_file();

    timer.stop_timer();
    if (id_sim == 0 && myrank == 0) {
        std::cout << std::endl;
        std::cout << "The program begin at " << timer.get_start_t();
        std::cout << "Inv_and_reloc end at " << timer.get_end_t();
        std::cout << "It has run " << timer.get_elapsed_t() << " seconds in total." << std::endl;
        std::cout << std::endl;
    }

}


// run 1D inversion mode
// run mode: 4
inline void run_1d_inversion(InputParams& IP, Grid& grid, IO_utils& io) {
    OneDInversion oneDInv(IP, grid);

    if(myrank == 0)
        std::cout << "id_sim: " << id_sim << ", size of src_map: " << IP.src_map.size() << std::endl;

    // estimate running time
    Timer timer("1D_inversion", true);

    // prepare objective_function file
    std::ofstream out_main; // close() is not mandatory
    prepare_header_line(IP, out_main);

    synchronize_all_world();

    /////////////////////
    // loop for inversion
    /////////////////////

    // objective function for all src
    std::vector<CUSTOMREAL> v_obj_misfit(20, 0.0);

    for (int i_inv = 0; i_inv < IP.get_max_iter_inv(); i_inv++) {
        if(myrank == 0 && id_sim ==0){
            std::cout << "iteration " << i_inv << " starting ... " << std::endl;
        }

        // prepare inverstion iteration group in xdmf file
        io.prepare_grid_inv_xdmf(i_inv);

        ///////////////////////////////////////////////////////
        // run (forward and adjoint) simulation for each source
        ///////////////////////////////////////////////////////
        v_obj_misfit = oneDInv.run_simulation_one_step_1dinv(IP, io, i_inv);

        // wait for all processes to finish
        synchronize_all_world();

        // check if v_obj is nan
        if (std::isnan(v_obj_misfit[0])) {
            if (myrank == 0)
                std::cout << "v_obj is nan, stop inversion" << std::endl;
            // stop inversion
            break;
        }

        // output src rec file with the result arrival times
        if (IP.get_if_output_in_process_data()){
            IP.write_src_rec_file(i_inv,0);
        } else if (i_inv == IP.get_max_iter_inv()-1 || i_inv==0) {
            IP.write_src_rec_file(i_inv,0);
        }

        ///////////////
        // model update
        ///////////////
        if(myrank == 0 && id_sim ==0)
            std::cout << "model update starting ... " << std::endl;
        oneDInv.model_optimize_1dinv(IP, grid, io, i_inv);

        // output objective function
        write_objective_function(IP, i_inv, v_obj_misfit, out_main, "1d inversion");

        // output updated model
        if (id_sim==0) {
            //io.change_xdmf_obj(0); // change xmf file for next src
            io.change_group_name_for_model();

            // write out model info
            if (IP.get_if_output_in_process() || i_inv >= IP.get_max_iter_inv() - 2){
                io.write_vel(grid, i_inv+1);
                io.write_xi( grid, i_inv+1);
                io.write_eta(grid, i_inv+1);
            }
        }

        // writeout temporary xdmf file
        io.update_xdmf_file();

        // wait for all processes to finish
        synchronize_all_world();

        // estimate running time
        CUSTOMREAL time_elapsed = timer.get_t();
        if (id_sim == 0 && myrank == 0 && i_inv < IP.get_max_iter_inv()-1) {
            const time_t end_time_estimated = time_elapsed / (i_inv + 1) * (IP.get_max_iter_inv() - i_inv - 1) + timer.get_start();
            auto will_run_time = (int)(time_elapsed/(i_inv + 1) * (IP.get_max_iter_inv() - i_inv - 1));

            std::cout << std::endl;
            std::cout << "The program begins at " << timer.get_start_t() << std::endl;
            std::cout << "Iteration (" << i_inv + 1 << "/" << IP.get_max_iter_inv() << ") finished at " << time_elapsed << " seconds" << std::endl;
            std::cout << i_inv + 1 << " iterations run " << timer.get_t() << " seconds, the rest of " << IP.get_max_iter_inv() - i_inv - 1 << " iterations require " << will_run_time << " seconds." << std::endl;
            std::cout << "The program is estimated to stop at " << timer.get_utc_from_time_t(end_time_estimated) << std::endl;
            std::cout << std::endl;
        }

        // output current state of the model
        if (IP.get_if_output_final_model()) {
            // io.write_final_model(grid, IP);
            io.write_merged_model(grid, IP, "final_model.h5");
        }
        if (IP.get_if_output_middle_model()) {
            std::string tmp_fname = "middle_model_step_" + int2string_zero_fill(i_inv + 1) + ".h5";
            io.write_merged_model(grid, IP, tmp_fname);
        }

        // wait for all processes to finish
        synchronize_all_world();
    } // end loop inverse

    // close xdmf file
    io.finalize_data_output_file();

    timer.stop_timer();
    if (id_sim == 0 && myrank == 0) {
        std::cout << std::endl;
        std::cout << "The program begin at " << timer.get_start_t() << std::endl;
        std::cout << "1d model inversion ended at " << timer.get_end_t() << std::endl;
        std::cout << "It has run " << timer.get_elapsed_t() << " seconds in total." << std::endl;
        std::cout << std::endl;
    }
}

#endif // MAIN_ROUTINES_CALLING_H