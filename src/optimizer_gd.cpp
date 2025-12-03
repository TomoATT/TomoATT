#include "optimizer_gd.h"
#include <iostream>
#include "kernel_postprocessing.h"

Optimizer_gd::Optimizer_gd() {
}

Optimizer_gd::~Optimizer_gd() {
}

void Optimizer_gd::model_update(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv) {
    std::cout << "Gradient descent update\n";

    // check kernel density
    check_kernel_density(grid, IP);

    // sum up kernels from all simulateous group (level 1) 
    sumup_kernels(grid);

    // write out original kernels
    // Ks, Kxi, Keta, Ks_den, Kxi_den, Keta_den
    write_original_kernels(grid, IP, io, i_inv);

    // process kernels
    // Ks_loc, Keta_loc, Kxi_loc
    // --> 
    // Ks_update_loc, Keta_update_loc, Kxi_update_loc
    processing_kernels(grid, IP);

    // write out modified kernels (descent direction)
    // Ks_update, Kxi_update, Keta_update, Ks_density_update, Kxi_density_update, Keta_density_update
    write_modified_kernels(grid, IP, io, i_inv);

    // determine step length
    determine_step_length();

    // set new model
    set_new_model(grid, step_length_init);

    // make station correction
    IP.station_correction_update(step_length_init_sc);

    // writeout temporary xdmf file
    if (IP.get_verbose_output_level())
        io.update_xdmf_file();

    synchronize_all_world();
}


// smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
void Optimizer_gd::processing_kernels(Grid& grid, InputParams& IP) {
    
    // initialize and backup modified kernels
    initialize_and_backup_modified_kernels(grid);

    // check kernel value range
    check_kernel_value_range(grid);

    // post-processing kernels, depending on the optimization method in the .yaml file
    // 1. multigrid smoothing + kernel density normalization
    // 2. XXX (to do)
    Kernel_postprocessing::process_kernels(grid, IP);


}


// initialize and backup modified kernels
void Optimizer_gd::initialize_and_backup_modified_kernels(Grid& grid) {
    if (subdom_main){ // parallel level 3
        if (id_sim==0){ // parallel level 1

            // initiaize and backup modified kernel (XX_update_loc) params
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {

                        // backup previous smoothed kernel
                        grid.Ks_update_loc_previous[I2V(i,j,k)]   = grid.Ks_update_loc[I2V(i,j,k)];
                        grid.Keta_update_loc_previous[I2V(i,j,k)] = grid.Keta_update_loc[I2V(i,j,k)];
                        grid.Kxi_update_loc_previous[I2V(i,j,k)]  = grid.Kxi_update_loc[I2V(i,j,k)];

                        // initialize modified kernel
                        grid.Ks_update_loc[I2V(i,j,k)]   = _0_CR;
                        grid.Keta_update_loc[I2V(i,j,k)] = _0_CR;
                        grid.Kxi_update_loc[I2V(i,j,k)]  = _0_CR;

                    }
                }
            }

        }
    }

    // synchronize all processes
    synchronize_all_world();
}


// check kernel value range
void Optimizer_gd::check_kernel_value_range(Grid& grid) {
    if (subdom_main){ // parallel level 3
        if (id_sim==0){ // parallel level 1
            // check kernel
            CUSTOMREAL max_kernel = _0_CR;
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        max_kernel = std::max(max_kernel, std::abs(grid.Ks_loc[I2V(i,j,k)]));
                    }
                }
            }

            if (max_kernel <= eps) {    
                std::cout << "Error: max_kernel is near zero (less than 10^-12), check data residual and whether no data is used" << std::endl;
                exit(1);
            }
        }
    }
}


// determine step length
void Optimizer_gd::determine_step_length() {
    
}