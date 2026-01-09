#ifndef KERNEL_POSTPROCESSING_H
#define KERNEL_POSTPROCESSING_H

#include "config.h"
#include "grid.h"
#include "input_params.h"

namespace Kernel_postprocessing {

    // ---------------------------------------------------
    // ------------------ main function ------------------
    // ---------------------------------------------------

    // process kernels (multigrid, normalization ...):
    // Ks_loc, Keta_loc, Kxi_loc
    // --> 
    // Ks_update_loc, Keta_update_loc, Kxi_update_loc
    void process_kernels(InputParams& IP, Grid& grid);
    
    // normalize kernels to -1 ~ 1
    void normalize_kernels(Grid& grid);

    // assign processing kernels to modified kernels for model update
    void assign_to_modified_kernels(Grid& grid);

    // ---------------------------------------------------
    // ------------------ sub functions ------------------
    // ---------------------------------------------------

    // initialize processing kernels
    void initialize_processing_kernels(Grid& grid);

    // eliminate overlapping kernels on ghost layers when performing multigrid smoothing (just for MULTI_GRID_SMOOTHING)
    void eliminate_ghost_layer_multigrid(Grid& grid);

    // make the boundary values of updated kernels consistent among subdomains
    void shared_boundary_of_processing_kernels(Grid& grid);

    // METHOD 1, multigrid parameterization + kernel density normalization
    void multigrid_parameterization_density_normalization(InputParams& IP, Grid& grid);

    // (kernel rescaling to -1.0 - +1.0)
    void kernel_rescaling_to_unit(Grid& grid); 
}


#endif // KERNEL_POSTPROCESSING_H