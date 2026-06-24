#ifndef MAIN_ROUTINES_INVERSION_MODE_H
#define MAIN_ROUTINES_INVERSION_MODE_H

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
// #include "kernel.h"
// #include "model_update.h"
// #include "lbfgs.h"


inline void calculate_or_read_traveltime_field(InputParams& IP, Grid& grid, IO_utils& io, const int i_src, const int N_src, bool first_init,
                                               std::unique_ptr<Iterator>& It, const int id_src_att, const bool& prerun=false){

    if (IP.get_is_T_written_into_file(id_src_att)){ // if field has been written into file, read it from file
        // load travel time field on grid.T_loc
        if (myrank == 0){  // main of level 2 and 3
            std::cout << "id_sim: " << id_sim << ", reading source (" << i_src+1 << "/" << N_src
                    << "), name: "
                    << IP.src_map[id_src_att].name << ", lat: " << IP.src_map[id_src_att].lat
                    << ", lon: " << IP.src_map[id_src_att].lon << ", dep: " << IP.src_map[id_src_att].dep
                    << std::endl;
        }

        io.read_T_tmp(grid);

    } else {
        // We need to solve eikonal equation
        if (myrank == 0){
            std::cout << "id_sim: " << id_sim << ", calculating source (" << i_src+1 << "/" << N_src
                    << "), name: "
                    << IP.src_map[id_src_att].name << ", lat: " << IP.src_map[id_src_att].lat
                    << ", lon: " << IP.src_map[id_src_att].lon << ", dep: " << IP.src_map[id_src_att].dep
                    << std::endl;
        }

        // solve travel time field on grid.T_loc
        It->run_iteration_forward(IP, grid, io, first_init);

        // writeout travel time field
        if (prerun) {
            // write the temporary traveltime field into file for later use
            io.write_T_tmp(grid);

            if (proc_store_srcrec) // only proc_store_srcrec has the src_map object
                IP.src_map[id_src_att].is_T_written_into_file = true;
        }
   }
}


inline void pre_run_forward_only(InputParams& IP, Grid& grid, IO_utils& io, int i_inv){
    if(world_rank == 0)
        std::cout << "preparing traveltimes of common receiver data ..." << std::endl;

    Source src;
    Receiver recs;

    // i_src (1:N) -> id_src_att (key of src_map_comm_rec)
    std::vector<int> id_src_att_vec = srcrec_id_2_id_att(IP.src_map_comm_rec);      // main of level 2 and 3
    if (proc_store_srcrec){
        if ((int)id_src_att_vec.size() != IP.n_src_comm_rec_this_sim_group){
            std::cout << "Error: number of sources in this simulation group is not equal to the number of sources in src_map." << std::endl;
            exit(1);
        }
    }
    for (int i_src = 0; i_src < IP.n_src_comm_rec_this_sim_group; i_src++){

        // check if this is the first iteration of entire inversion process
        bool first_init = (i_inv == 0 && i_src==0);

        // get source info
        int         id_src_att     = IP.get_id_src_att(i_src, id_src_att_vec);       // level 2 and level 3
        std::string name_src       = IP.get_src_name(i_src, id_src_att_vec);         // level 2 and level 3
        bool        is_teleseismic = IP.get_if_src_teleseismic(id_src_att); // get is_teleseismic flag  (level 2 and level 3)

        // set source name for output files/dataset names
        io.reset_source_info(name_src);

        // set source position
        src.set_source_position(IP, grid, is_teleseismic, id_src_att, false); 

        // initialize iterator object
        std::unique_ptr<Iterator> It;

        select_iterator(IP, grid, src, io, first_init, is_teleseismic, It, false);

        // calculate or read traveltime field
        bool prerun_mode = true;
        calculate_or_read_traveltime_field(IP, grid, io, i_src, IP.n_src_comm_rec_this_sim_group, first_init, It, id_src_att, prerun_mode);

        // interpolate and store  traveltime, cs_dif. For cr_dif, only store the traveltime.
        recs.interpolate_and_store_arrival_times_at_rec_position(IP, grid, id_src_att);
        
    }


    // wait for all processes to finish
    synchronize_all_world();

    if(world_rank == 0)
        std::cout << "synthetic traveltimes of common receiver data have been prepared." << std::endl;

    // gather all the traveltime to the main process and distribute to all processes
    // for calculating cr_dif
    IP.gather_traveltimes_and_calc_syn_diff();


}


// calculate sensitivity kernel
inline void calculate_sensitivity_kernel(Grid& grid, InputParams& IP, const int id_src_att){
    // calculate sensitivity kernel

    // kernel calculation will be done only by the subdom_main
    if (subdom_main) {
        // get the necessary parameters
        int np             = loc_I;
        int nt             = loc_J;
        int nr             = loc_K;
        CUSTOMREAL dr      = grid.dr;
        CUSTOMREAL dt      = grid.dt;
        CUSTOMREAL dp      = grid.dp;
        CUSTOMREAL src_lon = IP.get_src_lon(   id_src_att);
        CUSTOMREAL src_lat = IP.get_src_lat(   id_src_att);
        CUSTOMREAL src_r   = IP.get_src_radius(id_src_att);

        CUSTOMREAL weight   = _1_CR;

        // inner points
        for (int kkr = 1; kkr < nr-1; kkr++) {
            for (int jjt = 1; jjt < nt-1; jjt++) {
                for (int iip = 1; iip < np-1; iip++) {

                    // calculate the kernel
                    CUSTOMREAL Tr_km     = (grid.T_loc[I2V(iip,jjt,kkr+1)] - grid.T_loc[I2V(iip,jjt,kkr-1)]) / (_2_CR * dr);
                    CUSTOMREAL Ttheta_km = (grid.T_loc[I2V(iip,jjt+1,kkr)] - grid.T_loc[I2V(iip,jjt-1,kkr)]) / (_2_CR * dt) / grid.r_loc_1d[kkr];
                    CUSTOMREAL Tphi_km   = (grid.T_loc[I2V(iip+1,jjt,kkr)] - grid.T_loc[I2V(iip-1,jjt,kkr)]) / (_2_CR * dp) / (grid.r_loc_1d[kkr]*std::cos(grid.t_loc_1d[jjt]));

                    CUSTOMREAL azi_ratio = std::sqrt((my_square(Ttheta_km) + my_square(Tphi_km))/(my_square(Tr_km) + my_square(Ttheta_km) + my_square(Tphi_km)));
                    
                    // mask within one grid around the source
                    if ((std::abs(grid.r_loc_1d[kkr] - src_r) >= dr) ||
                        (std::abs(grid.t_loc_1d[jjt] - src_lat) >= dt) ||
                        (std::abs(grid.p_loc_1d[iip] - src_lon) >= dp)) {
                        // density of ks
                        grid.Ks_density_loc[I2V(iip,jjt,kkr)]   += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)];  
                        
                        // density of kxi
                        // grid.Kxi_density_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)]; 
                        grid.Kxi_density_loc[I2V(iip,jjt,kkr)]  += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)] * azi_ratio;
                                                                
                        // density of keta
                        // grid.Keta_density_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)]; 
                        grid.Keta_density_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)] * azi_ratio;
        

                        if (IP.get_update_slowness()==1){      // we need to update slowness
                            // Kernel w r t slowness s
                            grid.Ks_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] * my_square(grid.fun_loc[I2V(iip,jjt,kkr)]);                                                  
                        } else {
                            grid.Ks_loc[I2V(iip,jjt,kkr)] = _0_CR;
                        }


                        if (IP.get_update_azi_ani()){      // we need to update azimuthal anisotropy
                            // Kernel w r t anisotrophy xi
                            if (isZero(std::sqrt(my_square(grid.xi_loc[I2V(iip,jjt,kkr)])+my_square(grid.eta_loc[I2V(iip,jjt,kkr)])))) {
                                grid.Kxi_loc[I2V(iip,jjt,kkr)]  += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] \
                                                                * (my_square(Ttheta_km) - my_square(Tphi_km));

                                grid.Keta_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] \
                                                                * ( -_2_CR * Ttheta_km * Tphi_km );

                            } else {
                                grid.Kxi_loc[I2V(iip,jjt,kkr)]  += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] \
                                                            * ((- GAMMA * grid.xi_loc[I2V(iip,jjt,kkr)] / \
                                                                        std::sqrt(my_square(grid.xi_loc[ I2V(iip,jjt,kkr)]) + my_square(grid.eta_loc[I2V(iip,jjt,kkr)]))) * my_square(Tr_km) \
                                                                + my_square(Ttheta_km) 
                                                                - my_square(Tphi_km));

                                grid.Keta_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] \
                                                            * (( - GAMMA * grid.eta_loc[I2V(iip,jjt,kkr)]/ \
                                                                            std::sqrt(my_square(grid.xi_loc[I2V(iip,jjt,kkr)]) + my_square(grid.eta_loc[I2V(iip,jjt,kkr)]))) * my_square(Tr_km) \
                                                                    - _2_CR * Ttheta_km * Tphi_km );
                            }
                        } else {
                            grid.Kxi_loc[I2V(iip,jjt,kkr)]  = _0_CR;
                            grid.Keta_loc[I2V(iip,jjt,kkr)] = _0_CR;
                        }

                    } else{
                        grid.Ks_loc[I2V(iip,jjt,kkr)]   += _0_CR;
                        grid.Kxi_loc[I2V(iip,jjt,kkr)]  += _0_CR;
                        grid.Keta_loc[I2V(iip,jjt,kkr)] += _0_CR;

                        grid.Ks_density_loc[I2V(iip,jjt,kkr)]   += _0_CR;
                        grid.Kxi_density_loc[I2V(iip,jjt,kkr)]  += _0_CR;
                        grid.Keta_density_loc[I2V(iip,jjt,kkr)] += _0_CR;
        
                    }
                }
            }
        }

        // boundary
        for (int kkr = 0; kkr < nr; kkr++) {
            for (int jjt = 0; jjt < nt; jjt++) {
                // set Ks Kxi Keta to zero
                if (grid.i_first()){
                    grid.Ks_loc[I2V(0,jjt,kkr)]             = _0_CR;
                    grid.Kxi_loc[I2V(0,jjt,kkr)]            = _0_CR;
                    grid.Keta_loc[I2V(0,jjt,kkr)]           = _0_CR;
                    grid.Ks_density_loc[I2V(0,jjt,kkr)]     = _0_CR;
                    grid.Kxi_density_loc[I2V(0,jjt,kkr)]    = _0_CR;
                    grid.Keta_density_loc[I2V(0,jjt,kkr)]   = _0_CR;
                }
                if (grid.i_last()){
                    grid.Ks_loc[I2V(np-1,jjt,kkr)]          = _0_CR;
                    grid.Kxi_loc[I2V(np-1,jjt,kkr)]         = _0_CR;
                    grid.Keta_loc[I2V(np-1,jjt,kkr)]        = _0_CR;
                    grid.Ks_density_loc[I2V(np-1,jjt,kkr)]  = _0_CR;
                    grid.Kxi_density_loc[I2V(np-1,jjt,kkr)] = _0_CR;
                    grid.Keta_density_loc[I2V(np-1,jjt,kkr)]= _0_CR;
                }
           }
        }
        for (int kkr = 0; kkr < nr; kkr++) {
            for (int iip = 0; iip < np; iip++) {
                // set Ks Kxi Keta to zero
                if (grid.j_first()){
                    grid.Ks_loc[I2V(iip,0,kkr)]             = _0_CR;
                    grid.Kxi_loc[I2V(iip,0,kkr)]            = _0_CR;
                    grid.Keta_loc[I2V(iip,0,kkr)]           = _0_CR;
                    grid.Ks_density_loc[I2V(iip,0,kkr)]     = _0_CR;
                    grid.Kxi_density_loc[I2V(iip,0,kkr)]    = _0_CR;
                    grid.Keta_density_loc[I2V(iip,0,kkr)]   = _0_CR;
                }
                if (grid.j_last()){
                    grid.Ks_loc[I2V(iip,nt-1,kkr)]          = _0_CR;
                    grid.Kxi_loc[I2V(iip,nt-1,kkr)]         = _0_CR;
                    grid.Keta_loc[I2V(iip,nt-1,kkr)]        = _0_CR;
                    grid.Ks_density_loc[I2V(iip,nt-1,kkr)]  = _0_CR;
                    grid.Kxi_density_loc[I2V(iip,nt-1,kkr)] = _0_CR;
                    grid.Keta_density_loc[I2V(iip,nt-1,kkr)]= _0_CR;
                }
            }
        }
        for (int jjt = 0; jjt < nt; jjt++) {
            for (int iip = 0; iip < np; iip++) {
                // set Ks Kxi Keta to zero
                if (grid.k_first()){
                    grid.Ks_loc[I2V(iip,jjt,0)]             = _0_CR;
                    grid.Kxi_loc[I2V(iip,jjt,0)]            = _0_CR;
                    grid.Keta_loc[I2V(iip,jjt,0)]           = _0_CR;
                    grid.Ks_density_loc[I2V(iip,jjt,0)]     = _0_CR;
                    grid.Kxi_density_loc[I2V(iip,jjt,0)]    = _0_CR;
                    grid.Keta_density_loc[I2V(iip,jjt,0)]   = _0_CR;
                }
                if (grid.k_last()){
                    grid.Ks_loc[I2V(iip,jjt,nr-1)]          = _0_CR;
                    grid.Kxi_loc[I2V(iip,jjt,nr-1)]         = _0_CR;
                    grid.Keta_loc[I2V(iip,jjt,nr-1)]        = _0_CR;
                    grid.Ks_density_loc[I2V(iip,jjt,nr-1)]  = _0_CR;
                    grid.Kxi_density_loc[I2V(iip,jjt,nr-1)] = _0_CR;
                    grid.Keta_density_loc[I2V(iip,jjt,nr-1)]= _0_CR;
                }
            }
        }

    } // end if subdom_main
}


// check kernel density
inline void check_kernel_density(InputParams& IP, Grid& grid) {
    if(subdom_main){
        // check local kernel density positivity
        for (int i_loc = 0; i_loc < loc_I; i_loc++) {
            for (int j_loc = 0; j_loc < loc_J; j_loc++) {
                for (int k_loc = 0; k_loc < loc_K; k_loc++) {
                    if (isNegative(grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)])){
                        std::cout   << "Warning, id_sim: " << id_sim << ", grid.Ks_density_loc[I2V(" << i_loc << "," << j_loc << "," << k_loc << ")] is less than 0, = " 
                                    << grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)]   
                                    << std::endl;
                    }
                }
            }
        }
    }
}


// sum up kernels from all simulateous group (level 1)
inline void sumup_kernels(Grid& grid) {
    if(subdom_main){
        int n_grids = loc_I*loc_J*loc_K;

        allreduce_cr_sim_inplace(grid.Ks_loc, n_grids);
        allreduce_cr_sim_inplace(grid.Kxi_loc, n_grids);
        allreduce_cr_sim_inplace(grid.Keta_loc, n_grids);
        allreduce_cr_sim_inplace(grid.Ks_density_loc, n_grids);
        allreduce_cr_sim_inplace(grid.Kxi_density_loc, n_grids);
        allreduce_cr_sim_inplace(grid.Keta_density_loc, n_grids);

        // share the values on boundary
        grid.send_recev_boundary_data(grid.Ks_loc);
        grid.send_recev_boundary_data(grid.Kxi_loc);
        grid.send_recev_boundary_data(grid.Keta_loc);
        grid.send_recev_boundary_data(grid.Ks_density_loc);
        grid.send_recev_boundary_data(grid.Kxi_density_loc);
        grid.send_recev_boundary_data(grid.Keta_density_loc);

        grid.send_recev_boundary_data_kosumi(grid.Ks_loc);
        grid.send_recev_boundary_data_kosumi(grid.Kxi_loc);
        grid.send_recev_boundary_data_kosumi(grid.Keta_loc);
        grid.send_recev_boundary_data_kosumi(grid.Ks_density_loc);
        grid.send_recev_boundary_data_kosumi(grid.Kxi_density_loc);
        grid.send_recev_boundary_data_kosumi(grid.Keta_density_loc);
    }

    synchronize_all_world();
}


// run forward and adjoint simulation and calculate current objective function value and sensitivity kernel if requested
inline std::vector<CUSTOMREAL> run_simulation_one_step(InputParams& IP, Grid& grid, IO_utils& io, int i_inv, bool line_search_mode, bool is_save_T){
    // line_search_mode: if true, time field and adjoint field will not be written into file
    // is_save_T: if true, save temperory traveltime field into file (just for earthquake)

    int n_src_map_size = IP.src_map.size();
    int n_rec_map_size = IP.rec_map.size();


    // initialize kernel arrays
    if (IP.get_run_mode() == DO_INVERSION || IP.get_run_mode() == INV_RELOC)
        grid.initialize_kernels();

    // reinitialize factors
    // grid.rejuvenate_abcf();     // vel, xi, eta -> a, b, c, f
    // grid.reinitialize_abcf();   // a, b, c, f -> a, b/r^2, c/(r^2*cos^2), f/(r^2*cos)

    ///////////////////////////////////////////////////////////////////////
    //  compute the synthetic common receiver differential traveltime first
    ///////////////////////////////////////////////////////////////////////

    // prepare synthetic traveltime for all earthquakes, if
    //  1. common receiver data exists;
    //  2. we use common receiver data to update model; (cr + not swap) or (cs + swap)
    //  3. we do inversion
    if ( src_pair_exists &&
              ((IP.get_use_cr() && !IP.get_is_srcrec_swap())
            || (IP.get_use_cs() && IP.get_is_srcrec_swap()) )
        && (IP.get_run_mode() == DO_INVERSION || IP.get_run_mode() == INV_RELOC)){
        pre_run_forward_only(IP, grid, io, i_inv);
    }
    //
    // loop over all sources
    //


    Source src;
    Receiver recs;

    if(world_rank == 0)
        std::cout << "computing traveltime field, adjoint field and kernel ..." << std::endl;

    // i_src (1:N) -> id_src_att (key of src_map)
    std::vector<int> id_src_att_vec = srcrec_id_2_id_att(IP.src_map);      // main of level 2 and 3
    if (proc_store_srcrec){
        if ((int)id_src_att_vec.size() != IP.n_src_this_sim_group){
            std::cout << "Error: number of sources in this simulation group is not equal to the number of sources in src_map." << std::endl;
            exit(1);
        }
    }
    // iterate over sources
    for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
        if (id_subdomain == 0 && subdom_main){  // main of level 2 and 3
            std::cout << "ckp1, id_sim: " << id_sim << ", i_src: " << i_src << std::endl;
        }

        // check if this is the first iteration of entire inversion process
        bool first_init = (i_inv == 0 && i_src==0);

        // get source info
        int         id_src_att     = IP.get_id_src_att(i_src, id_src_att_vec);       // level 2 and level 3
        std::string name_src       = IP.get_src_name(i_src, id_src_att_vec);         // level 2 and level 3
        bool        is_teleseismic = IP.get_if_src_teleseismic(id_src_att);         // get is_teleseismic flag

        // set source name for output files/dataset names
        io.reset_source_info(name_src);


        /////////////////////////
        // run forward simulation
        /////////////////////////

        // (re) initialize source object and set to grid
        src.set_source_position(IP, grid, is_teleseismic, id_src_att, false);

        // initialize iterator object
        std::unique_ptr<Iterator> It;

        if (id_subdomain == 0 && subdom_main){  // main of level 2 and 3
            std::cout << "ckp2, id_sim: " << id_sim << ", i_src: " << i_src << ", id_src_att: " << id_src_att << std::endl;
        }

        if (!hybrid_stencil_order){
            select_iterator(IP, grid, src, io, first_init, is_teleseismic, It, false);

            // if traveltime field has been wriiten into the file, we choose to read the traveltime data.
            calculate_or_read_traveltime_field(IP, grid, io, i_src, IP.n_src_this_sim_group, first_init, It, id_src_att, is_save_T);

        } else {
            // hybrid stencil mode
            std::cout << "\nrunnning in hybrid stencil mode\n" << std::endl;

            // run 1st order forward simulation
            std::unique_ptr<Iterator> It_pre;
            IP.set_stencil_order(1);
            IP.set_conv_tol(IP.get_conv_tol()*100.0);
            select_iterator(IP, grid, src, io, first_init, is_teleseismic, It_pre, false);
            calculate_or_read_traveltime_field(IP, grid, io, i_src, IP.n_src_this_sim_group, first_init, It_pre, id_src_att, is_save_T);

            // run 3rd order forward simulation
            IP.set_stencil_order(3);
            IP.set_conv_tol(IP.get_conv_tol()/100.0);
            select_iterator(IP, grid, src, io, first_init, is_teleseismic, It, true);
            calculate_or_read_traveltime_field(IP, grid, io, i_src, IP.n_src_this_sim_group, first_init, It, id_src_att, is_save_T);
        }

        if (id_subdomain == 0 && subdom_main){  // main of level 2 and 3
            std::cout << "ckp3, id_sim: " << id_sim << ", i_src: " << i_src << ", id_src_att: " << id_src_att << std::endl;
        }

        // output the result of forward simulation
        // ignored for inversion mode.
        if (!line_search_mode && IP.get_if_output_source_field()) {

            // output T (result timetable)
            io.write_T(grid, i_inv);

            // output T0v
            //io.write_T0v(grid,i_inv); // initial Timetable
            // output u (true solution)
            //if (if_test)
            //   io.write_u(grid);  // true Timetable
            // output tau
            //io.write_tau(grid, i_inv); // calculated deviation
            // output residual (residual = true_solution - result)
            //if (if_test)
            //    io.write_residual(grid); // this will over write the u_loc, so we need to call write_u_h5 first
        }
        // calculate the arrival times at each receivers
        recs.interpolate_and_store_arrival_times_at_rec_position(IP, grid, id_src_att);

        if (id_subdomain == 0 && subdom_main){  // main of level 2 and 3
            std::cout << "ckp4, id_sim: " << id_sim << ", i_src: " << i_src << ", id_src_att: " << id_src_att << std::endl;
        }

        /////////////////////////
        // run adjoint simulation
        /////////////////////////

        if (IP.get_run_mode()==DO_INVERSION || IP.get_run_mode()==INV_RELOC){
            // calculate adjoint source
            if (id_subdomain == 0 && subdom_main){  // main of level 2 and 3
                std::cout << "ckp5, id_sim: " << id_sim << ", i_src: " << i_src << ", id_src_att: " << id_src_att << std::endl;
            }
            recs.calculate_adjoint_source(IP, id_src_att);
            // run iteration for adjoint field calculation
            if (id_subdomain == 0 && subdom_main){  // main of level 2 and 3
                std::cout << "ckp6, id_sim: " << id_sim << ", i_src: " << i_src << ", id_src_att: " << id_src_att << std::endl;
            }
            int adj_type = 0;   // compute adjoint field
            It->run_iteration_adjoint(IP, grid, io, adj_type);
            // run iteration for density of the adjoint field
            adj_type = 1;   // compute adjoint field
            It->run_iteration_adjoint(IP, grid, io, adj_type);
            // calculate sensitivity kernel
            calculate_sensitivity_kernel(grid, IP, id_src_att);
            if (subdom_main && !line_search_mode && IP.get_if_output_source_field()) {
                // adjoint field will be output only at the end of subiteration
                // output the result of adjoint simulation
                io.write_adjoint_field(grid,i_inv);
            }

            // io.write_adjoint_field(grid,i_inv);

            // check adjoint source
            // if (proc_store_srcrec){
            //     for (auto iter = IP.rec_map.begin(); iter != IP.rec_map.end(); iter++){
            //         std::cout << "rec id: " << iter->second.id << ", rec name: " << iter->second.name << ", adjoint source: " << iter->second.adjoint_source << std::endl;
            //     }
            // }

        } // end if run_mode == DO_INVERSION

        if (id_subdomain == 0 && subdom_main){  // main of level 2 and 3
            std::cout << "ckp7, id_sim: " << id_sim << ", i_src: " << i_src << ", id_src_att: " << id_src_att << std::endl;
        }

        // wait for all processes to finish
        // this should not be called here, for the case that the simultaneous run group has different number of sources
        //synchronize_all_world();

    } // end for i_src
    // synchronize all processes
    synchronize_all_world();


    // gather all the traveltime to the main process and distribute to all processes
    // for calculating the synthetic common receiver differential traveltime
    if ( IP.get_run_mode()==ONLY_FORWARD ||                 // case 1. if we are doing forward modeling, traveltime is not prepared for computing cr_dif data. Now we need to compute it
        (!IP.get_use_cr() && !IP.get_is_srcrec_swap())  ||   // case 2-1, we do inversion, but we do not use cr data (cr + no swap)
        (!IP.get_use_cs() &&  IP.get_is_srcrec_swap())){     // case 2-2, we do inversion, but we do not use cr data (cs +    swap)
        IP.gather_traveltimes_and_calc_syn_diff();
    }


    // compute all residual and obj
    std::vector<CUSTOMREAL> obj_residual = recs.calculate_obj_and_residual(IP);


    // check kernel density and sum up kernels from all simulateous group (level 1)
    if (IP.get_run_mode() == DO_INVERSION || IP.get_run_mode() == INV_RELOC){
        check_kernel_density(IP, grid);     // check kernel density
        sumup_kernels(grid); // allreduce kernels from all simulateous group (level 1) Ks_loc, Ks_density_loc
    }

    // check src_map and rec_map are not changed after the simulation
    if (proc_store_srcrec) {
       if(n_src_map_size != (int)IP.src_map.size()){
            std::cout << "ERROR, id_sim: " << id_sim << ", size of src_map has been changed after simulation, before: " << n_src_map_size << ", after: " << IP.src_map.size() << std::endl;
            exit(1);
        }
        if(n_rec_map_size != (int)IP.rec_map.size()){
            std::cout << "ERROR, id_sim: " << id_sim << ", size of rec_map has been changed after simulation, before: " << n_rec_map_size << ", after: " << IP.rec_map.size() << std::endl;
            exit(1);
        }
    }

    // return current objective function value
    return obj_residual;
}





#endif // MAIN_ROUTINES_INVERSION_MODE_H