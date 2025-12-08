#include "optimizer_gd.h"
#include <iostream>
#include "kernel_postprocessing.h"

Optimizer_gd::Optimizer_gd(InputParams& IP) : Optimizer(IP) {

    // for gradient descent method, 
    need_write_model = false;
    need_write_original_kernel = false;

}

Optimizer_gd::~Optimizer_gd() {
}

// ---------------------------------------------------------
// ------------------ specified functions ------------------
// ---------------------------------------------------------


// smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
void Optimizer_gd::processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv) {
    
    // initialize and backup modified kernels
    initialize_and_backup_modified_kernels(grid);

    // check kernel value range
    check_kernel_value_range(grid);

    // post-processing kernels, depending on the optimization method in the .yaml file
    // 1. multigrid smoothing + kernel density normalization
    // 2. XXX (to do)
    Kernel_postprocessing::process_kernels(IP, grid);
}







