#include "iterator_legacy.h"


Iterator_legacy:: Iterator_legacy(InputParams& IP, Grid& grid, Source& src, IO_utils& io,
                                  bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                : Iterator(IP, grid, src, io, first_init, is_teleseismic_in, is_second_run_in) {
    // do nothing
}

void Iterator_legacy::do_sweep_adj(int iswp, Grid& grid, InputParams& IP) {
    if (subdom_main) {
        // set sweep direction
        set_sweep_direction(iswp);

        int r_start, t_start, p_start, r_end, t_end, p_end;

        // set loop range
        if (r_dirc < 0) {
            r_start = nr-2;
            r_end   = 0;
        } else {
            r_start = 1;
            r_end   = nr-1;
        }
        if (t_dirc < 0) {
            t_start = nt-2;
            t_end   = 0;
        } else {
            t_start = 1;
            t_end   = nt-1;
        }
        if (p_dirc < 0) {
            p_start = np-2;
            p_end   = 0;
        } else {
            p_start = 1;
            p_end   = np-1;
        }

        for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
            for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
                for (int iip = p_start; iip != p_end; iip+=p_dirc) {
                    // calculate stencils
                    calculate_stencil_adj(grid, iip, jjt, kkr);
                }
            }
        }

   } // end if subdom_main

}


Iterator_legacy_tele::Iterator_legacy_tele(InputParams& IP, Grid& grid, Source& src, IO_utils& io,
                                           bool first_init, bool is_teleseismic_in, bool is_second_run_in)
                : Iterator(IP, grid, src, io, first_init, is_teleseismic_in, is_second_run_in) {
    // do nothing
}


void Iterator_legacy_tele::do_sweep_adj(int iswp, Grid& grid, InputParams& IP) {
    if (subdom_main) {
        if(if_verbose) std::cout << "sweeping: " << iswp << std::endl;
        // set sweep direction
        set_sweep_direction(iswp);

        int r_start, t_start, p_start, r_end, t_end, p_end;

        // set loop range
        if (r_dirc < 0) {
            r_start = nr-1;
            r_end   = -1;
        } else {
            r_start = 0;
            r_end   = nr;
        }
        if (t_dirc < 0) {
            t_start = nt-1;
            t_end   = -1;
        } else {
            t_start = 0;
            t_end   = nt;
        }
        if (p_dirc < 0) {
            p_start = np-1;
            p_end   = -1;
        } else {
            p_start = 0;
            p_end   = np;
        }

        for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
            for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
                for (int iip = p_start; iip != p_end; iip+=p_dirc) {
                    if (iip != 0    && jjt != 0    && kkr != 0 \
                     && iip != np-1 && jjt != nt-1 && kkr != nr-1) {
                        // calculate stencils
                        calculate_stencil_adj(grid, iip, jjt, kkr);
                    } else {
                        calculate_boundary_nodes_adj(grid, iip, jjt, kkr);
                    }
                }
            }
        }

    } // end if subdom_main
}


Iterator_legacy_1st_order::Iterator_legacy_1st_order(InputParams& IP, Grid& grid, Source& src, IO_utils& io,
                                                     bool first_init, bool is_teleseismic_in, bool is_second_run_in)
                         : Iterator_legacy(IP, grid, src, io, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}


void Iterator_legacy_1st_order::do_sweep(int iswp, Grid& grid, InputParams& IP) {
    if (subdom_main) {

        // set sweep direction
        set_sweep_direction(iswp);

        // set loop range
        int r_start, t_start, p_start, r_end, t_end, p_end;

        if (r_dirc < 0) {
            r_start = nr-2;
            r_end   = 0;
        } else {
            r_start = 1;
            r_end   = nr-1;
        }
        if (t_dirc < 0) {
            t_start = nt-2;
            t_end   = 0;
        } else {
            t_start = 1;
            t_end   = nt-1;
        }
        if (p_dirc < 0) {
            p_start = np-2;
            p_end   = 0;
        } else {
            p_start = 1;
            p_end   = np-1;
        }

        for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
            for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
                for (int iip = p_start; iip != p_end; iip+=p_dirc) {
                    if (grid.is_changed[I2V(iip, jjt, kkr)]) {
                        // 1st order, calculate stencils
                        calculate_stencil_1st_order(grid, iip, jjt, kkr);
                    }
                }
            }
        }

        // update boundary
        //calculate_boundary_nodes_maxmin(grid);
        calculate_boundary_nodes(grid);

    } // end if subdom_main
}



Iterator_legacy_3rd_order::Iterator_legacy_3rd_order(InputParams& IP, Grid& grid, Source& src, IO_utils& io, 
                                                     bool first_init, bool is_teleseismic_in, bool is_second_run_in)
                         : Iterator_legacy(IP, grid, src, io, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}


void Iterator_legacy_3rd_order::do_sweep(int iswp, Grid& grid, InputParams& IP) {
    // std::cout << "cp1.5, my world rank is: " << world_rank << std::endl;
    if (subdom_main) {

        // set sweep direction
        set_sweep_direction(iswp);

        // set loop range
        int r_start, t_start, p_start, r_end, t_end, p_end;

        if (r_dirc < 0) {
            r_start = nr-2;
            r_end   = 0;
        } else {
            r_start = 1;
            r_end   = nr-1;
        }
        if (t_dirc < 0) {
            t_start = nt-2;
            t_end   = 0;
        } else {
            t_start = 1;
            t_end   = nt-1;
        }
        if (p_dirc < 0) {
            p_start = np-2;
            p_end   = 0;
        } else {
            p_start = 1;
            p_end   = np-1;
        }

        for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
            for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
                for (int iip = p_start; iip != p_end; iip+=p_dirc) {
                    if (grid.is_changed[I2V(iip, jjt, kkr)]) {
                        // 1st order, calculate stencils
                        calculate_stencil_3rd_order(grid, iip, jjt, kkr);
                    }
                }
            }
        }

        // update boundary
        calculate_boundary_nodes(grid);
        // std::cout << "cp2, my world rank is: " << world_rank << std::endl;
        // for (int i_proc = 0; i_proc < world_nprocs; i_proc++){
        //     std::cout << "my world rank is: " << world_rank << ", i_proc is: " << i_proc
        //               << ", world_nprocs: " << world_nprocs  << std::endl;

        //     if (i_proc == world_rank ){
        //         for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
        //             for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
        //                 for (int iip = p_start; iip != p_end; iip+=p_dirc) {

        //                     std::cout << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ')
        //                             << grid.tau_loc[I2V(iip,jjt,kkr)]*grid.T0v_loc[I2V(iip,jjt,kkr)];
        //                 }
        //                 std::cout << std::endl;
        //             }
        //             std::cout << std::endl;
        //         }
        //     }
        //     synchronize_all_world();
        // }

    } // end if subdom_main
}


Iterator_legacy_1st_order_upwind::Iterator_legacy_1st_order_upwind(InputParams& IP, Grid& grid, Source& src, IO_utils& io,
                                                                   bool first_init, bool is_teleseismic_in, bool is_second_run_in) 
                         : Iterator_legacy(IP, grid, src, io, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}

void Iterator_legacy_1st_order_upwind::do_sweep(int iswp, Grid& grid, InputParams& IP) {
    if (subdom_main) {

        // set sweep direction
        set_sweep_direction(iswp);

        // set loop range
        int r_start, t_start, p_start, r_end, t_end, p_end;

        if (r_dirc < 0) {
            r_start = nr-1;
            r_end   = -1;
        } else {
            r_start = 0;
            r_end   = nr;
        }
        if (t_dirc < 0) {
            t_start = nt-1;
            t_end   = -1;
        } else {
            t_start = 0;
            t_end   = nt;
        }
        if (p_dirc < 0) {
            p_start = np-1;
            p_end   = -1;
        } else {
            p_start = 0;
            p_end   = np;
        }

        for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
            for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
                for (int iip = p_start; iip != p_end; iip+=p_dirc) {
                    if (grid.is_changed[I2V(iip, jjt, kkr)]) {
                        // 1st order, calculate stencils
                        calculate_stencil_1st_order_upwind(grid, iip, jjt, kkr);
                    }
                }
            }
        }
        // std::cout << "cp2, my world rank is: " << world_rank << std::endl;
        // for (int i_proc = 0; i_proc < world_nprocs; i_proc++){
        //     std::cout << "my world rank is: " << world_rank << ", i_proc is: " << i_proc
        //               << ", world_nprocs: " << world_nprocs  << std::endl;

        //     if (i_proc == world_rank ){
        //         for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
        //             for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
        //                 for (int iip = p_start; iip != p_end; iip+=p_dirc) {

        //                     std::cout << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ')
        //                             << grid.tau_loc[I2V(iip,jjt,kkr)]*grid.T0v_loc[I2V(iip,jjt,kkr)];
        //                 }
        //                 std::cout << std::endl;
        //             }
        //             std::cout << std::endl;
        //         }
        //     }
        //     // synchronize_all_world();
        // }


        // no boundary process is needed for 1st order upwind method
        // // update boundary
        // calculate_boundary_nodes(grid);

    } // end if subdom_main
}



Iterator_legacy_1st_order_tele::Iterator_legacy_1st_order_tele(InputParams& IP, Grid& grid, Source& src, IO_utils& io, 
                                                               bool first_init, bool is_teleseismic_in, bool is_second_run_in) 
                         : Iterator_legacy_tele(IP, grid, src, io, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}

// overloading virtual function
void Iterator_legacy_1st_order_tele::do_sweep(int iswp, Grid& grid, InputParams& IP) {
    if (subdom_main) {

        // set sweep direction
        set_sweep_direction(iswp);

        // set loop range
        int r_start, t_start, p_start, r_end, t_end, p_end;

        if (r_dirc < 0) {
            r_start = nr-1;
            r_end   = -1;
        } else {
            r_start = 0;
            r_end   = nr;
        }
        if (t_dirc < 0) {
            t_start = nt-1;
            t_end   = -1;
        } else {
            t_start = 0;
            t_end   = nt;
        }
        if (p_dirc < 0) {
            p_start = np-1;
            p_end   = -1;
        } else {
            p_start = 0;
            p_end   = np;
        }

        for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
            for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
                for (int iip = p_start; iip != p_end; iip+=p_dirc) {
                    // this if statement is not necessary because always only the outer layer
                    // is !is_changed in teleseismic case
                    //if (grid.is_changed[I2V(iip, jjt, kkr)]) {
                    if (iip != 0 && iip != np-1 && jjt != 0 && jjt != nt-1 && kkr != 0 && kkr != nr-1) {
                        // 1st order, calculate stencils
                        calculate_stencil_1st_order_tele(grid, iip, jjt, kkr);
                    } else {
                        // update boundary
                        calculate_boundary_nodes_tele(grid, iip, jjt, kkr);
                    }
                }
            }
        }

    } // end if subdom_main
}



Iterator_legacy_3rd_order_tele::Iterator_legacy_3rd_order_tele(InputParams& IP, Grid& grid, Source& src, IO_utils& io,
                                                               bool first_init, bool is_teleseismic_in, bool is_second_run_in)
                         : Iterator_legacy_tele(IP, grid, src, io, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}


void Iterator_legacy_3rd_order_tele::do_sweep(int iswp, Grid& grid, InputParams& IP) {
    if (subdom_main) {

        // set sweep direction
        set_sweep_direction(iswp);

        // set loop range
        int r_start, t_start, p_start, r_end, t_end, p_end;

        if (r_dirc < 0) {
            r_start = nr-1;
            r_end   = -1;
        } else {
            r_start = 0;
            r_end   = nr;
        }
        if (t_dirc < 0) {
            t_start = nt-1;
            t_end   = -1;
        } else {
            t_start = 0;
            t_end   = nt;
        }
        if (p_dirc < 0) {
            p_start = np-1;
            p_end   = -1;
        } else {
            p_start = 0;
            p_end   = np;
        }

        for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
            for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
                for (int iip = p_start; iip != p_end; iip+=p_dirc) {
                    if (iip != 0 && iip != np-1 && jjt != 0 && jjt != nt-1 && kkr != 0 && kkr != nr-1) {
                        // calculate stencils and update tau
                        calculate_stencil_3rd_order_tele(grid, iip, jjt, kkr);
                    } else {
                        // update boundary
                        calculate_boundary_nodes_tele(grid, iip, jjt, kkr);
                    }
                }
            }
        }
    } // end if subdom_main
}

Iterator_legacy_1st_order_upwind_tele::Iterator_legacy_1st_order_upwind_tele(InputParams& IP, Grid& grid, Source& src, IO_utils& io,
                                                                             bool first_init, bool is_teleseismic_in, bool is_second_run_in)
                         : Iterator_legacy_tele(IP, grid, src, io, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}

void Iterator_legacy_1st_order_upwind_tele::do_sweep(int iswp, Grid& grid, InputParams& IP) {
    if (subdom_main) {
        // set sweep direction
        set_sweep_direction(iswp);

        // set loop range
        int r_start, t_start, p_start, r_end, t_end, p_end;

        if (r_dirc < 0) {
            r_start = nr-1;
            r_end   = -1;
        } else {
            r_start = 0;
            r_end   = nr;
        }
        if (t_dirc < 0) {
            t_start = nt-1;
            t_end   = -1;
        } else {
            t_start = 0;
            t_end   = nt;
        }
        if (p_dirc < 0) {
            p_start = np-1;
            p_end   = -1;
        } else {
            p_start = 0;
            p_end   = np;
        }

        for (int kkr = r_start; kkr != r_end; kkr+=r_dirc) {
            for (int jjt = t_start; jjt != t_end; jjt+=t_dirc) {
                for (int iip = p_start; iip != p_end; iip+=p_dirc) {
                    // this if statement is not necessary because always only the outer layer
                    // is !is_changed in teleseismic case
                    // if (grid.is_changed[I2V(iip, jjt, kkr)]) {
                    // if (iip != 0 && iip != np-1 && jjt != 0 && jjt != nt-1 && kkr != 0 && kkr != nr-1) {
                    // 1st order, calculate stencils
                    if (iip != 0 && iip != np-1 && jjt != 0 && jjt != nt-1 && kkr != 0) {   // top layer is not fixed, otherwise, the top layer will be 2000
                        calculate_stencil_1st_order_upwind_tele(grid, iip, jjt, kkr);       // no need to consider the boundary for upwind scheme
                    }
                }
            }
        }

    } // end if subdom_main
}

