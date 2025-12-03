#include "optimizer_gd.h"
#include <iostream>

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
    write_original_kernels(grid, IP, io, i_inv);

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    processing_kernels(grid, IP);

    // write out modified kernels (descent direction)
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
    
}


// determine step length
void Optimizer_gd::determine_step_length() {
    
}