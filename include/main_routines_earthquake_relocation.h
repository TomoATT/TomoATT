#ifndef MAIN_ROUTINES_EARTHQUAKE_RELOCATION_H
#define MAIN_ROUTINES_EARTHQUAKE_RELOCATION_H

#include <iostream>
#include <memory>
#include "mpi_funcs.h"
#include "config.h"
#include "utils.h"
#include "input_params.h"
#include "grid.h"
#include "io.h"
#include "iterator_selector.h"
#include "iterator.h"
#include "iterator_legacy.h"
#include "iterator_level.h"
#include "source.h"
#include "receiver.h"
#include "main_routines_inversion_mode.h"


// calculate traveltime field for all sources-receivers and write to file
void calculate_traveltime_for_all_src_rec(InputParams& IP, Grid& grid, IO_utils& io){

    // check if this run is in source receiver swap mode
    // if not, stop the program
    if (IP.get_is_srcrec_swap() == false){
        std::cout << "Error: Source relocation mode must run with swap_src_rec = true" << std::endl;
        exit(1);
    }

    // reinitialize factors
    grid.rejuvenate_abcf();     // vel, xi, eta -> a, b, c, f
    grid.reinitialize_abcf();   // a, b, c, f -> a, b/r^2, c/(r^2*cos^2), f/(r^2*cos)


    // prepare inverstion iteration group in xdmf file
    io.prepare_grid_inv_xdmf(0);

    ///////////////////////
    // loop for each source
    ///////////////////////

    Source src;

    // iterate over sources
    for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){

        const std::string name_sim_src = IP.get_src_name(i_src);
        const int         id_sim_src   = IP.get_src_id(name_sim_src); // global source id

        // check if the source is teleseismic or not
        // because teleseismic source is not supported in this mode
        bool is_teleseismic = IP.get_if_src_teleseismic(name_sim_src);

        if (is_teleseismic){
            std::cout << "Error: Teleseismic source is not supported in source relocation mode." << std::endl;
            exit(1);
        }

        // set simu group id and source name for output files/dataset names
        io.reset_source_info(id_sim_src, name_sim_src);

        // set source position
        src.set_source_position(IP, grid, is_teleseismic, name_sim_src);

        // initialize iterator object
        bool first_init = (i_src==0);

        // initialize iterator object
        std::unique_ptr<Iterator> It;
        select_iterator(IP, grid, src, io, name_sim_src, first_init, is_teleseismic, It, false);

        /////////////////////////
        // run forward simulation
        /////////////////////////

        if (proc_store_srcrec){
            auto srcmap_this = IP.get_src_point(name_sim_src);

            std::cout << "calculating source (" << i_src+1 << "/" << IP.n_src_this_sim_group << "), name: "
                      << name_sim_src << ", lat: " << srcmap_this.lat
                      << ", lon: " << srcmap_this.lon << ", dep: " << srcmap_this.dep
                      << std::endl;
        }

        bool write_tmp_T = true;
        calculate_or_read_traveltime_field(IP, grid, io, i_src, IP.n_src_this_sim_group, first_init, It, name_sim_src, write_tmp_T);

    }

    // wait for all processes to finish traveltime calculation
    synchronize_all_world();
}


std::vector<CUSTOMREAL> calculate_gradient_objective_function(InputParams& IP, Grid& grid, IO_utils& io, int i_iter){

    Receiver recs; // here the source is swapped to receiver

    // initialize source parameters (obj, kernel, )
    recs.init_vars_src_reloc(IP);

    // iterate over sources (to obtain gradient of traveltime field T and absolute traveltime, common source differential traveltime, and common receiver differential traveltime)
    for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){

        const std::string name_sim_src = IP.get_src_name(i_src);
        const int         id_sim_src   = IP.get_src_id(name_sim_src); // global source id

        // set simu group id and source name for output files/dataset names
        io.reset_source_info(id_sim_src, name_sim_src);

        // load travel time field on grid.T_loc
        io.read_T_tmp(grid);

        // calculate travel time at the actual source location (absolute traveltime, common source differential traveltime)
        recs.interpolate_and_store_arrival_times_at_rec_position(IP, grid, name_sim_src);

        // calculate gradient at the actual source location
        recs.calculate_T_gradient(IP, grid, name_sim_src);
    }

    // wait for all processes to finish
    synchronize_all_world();

    // gather all the traveltime to the main process and distribute to all processes
    // for calculating the synthetic (common receiver differential traveltime)
    IP.gather_traveltimes_and_calc_syn_diff();


    // iterate over sources for calculating gradient of objective function
    for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
        const std::string name_sim_src = IP.get_src_name(i_src);
             
        // calculate gradient of objective function with respect to location and ortime
        recs.calculate_grad_obj_src_reloc(IP, name_sim_src);
    }

    // compute the objective function
    std::vector<CUSTOMREAL> obj_residual = recs.calculate_obj_reloc(IP, i_iter);

    // sum grad_tau and grad_chi_k of all simulation groups
    IP.allreduce_rec_map_grad_src();

    //synchronize_all_world(); // not necessary here because allreduce is already synchronizing communication

    return obj_residual;
}



#endif // MAIN_ROUTINES_EARTHQUAKE_RELOCATION_H