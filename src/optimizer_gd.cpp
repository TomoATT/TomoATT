#include "optimizer_gd.h"
#include <iostream>
#include "kernel_postprocessing.h"

Optimizer_gd::Optimizer_gd() {
}

Optimizer_gd::~Optimizer_gd() {
}

void Optimizer_gd::model_update(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj) {
    if(id_sim == 0 && myrank == 0)
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
    determine_step_length(grid, i_inv, v_obj_inout, old_v_obj);

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


// calculate the angle between previous and current model update directions
CUSTOMREAL Optimizer_gd::direction_change_of_model_update(Grid& grid){
    CUSTOMREAL norm_grad = _0_CR;
    CUSTOMREAL norm_grad_previous = _0_CR;
    CUSTOMREAL inner_product = _0_CR;
    CUSTOMREAL cos_angle = _0_CR;
    CUSTOMREAL angle = _0_CR;
    if (subdom_main) {
        // initiaize update params
        inner_product      = dot_product(grid.Ks_update_loc_previous, grid.Ks_update_loc, loc_I*loc_J*loc_K);
        norm_grad          = dot_product(grid.Ks_update_loc, grid.Ks_update_loc, loc_I*loc_J*loc_K);
        norm_grad_previous = dot_product(grid.Ks_update_loc_previous, grid.Ks_update_loc_previous, loc_I*loc_J*loc_K);

        CUSTOMREAL tmp;
        allreduce_cr_single(inner_product,tmp);
        inner_product = tmp;

        allreduce_cr_single(norm_grad,tmp);
        norm_grad = tmp;

        allreduce_cr_single(norm_grad_previous,tmp);
        norm_grad_previous = tmp;

        cos_angle = inner_product / (std::sqrt(norm_grad) * std::sqrt(norm_grad_previous));
        angle     = acos(cos_angle) * RAD2DEG;
    }
    return angle;
}


// determine step length
void Optimizer_gd::determine_step_length(Grid& grid, int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj) {

    // change stepsize
    // Option 1: the step length is modulated when obj changes.
    if (step_method == OBJ_DEFINED){
        if(i_inv != 0){
            if (v_obj_inout < old_v_obj) {
                step_length_init    = std::min((CUSTOMREAL)0.02, step_length_init);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The obj keeps decreasing, from " << old_v_obj << " to " << v_obj_inout
                            << ", the step length is " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            } else if (v_obj_inout >= old_v_obj) {
                step_length_init    = std::max((CUSTOMREAL)0.0001, step_length_init*step_length_decay);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The obj keep increases, from " << old_v_obj << " to " << v_obj_inout
                            << ", the step length decreases from " << step_length_init/step_length_decay
                            << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            }
        } else {
            if(myrank == 0 && id_sim == 0){
                std::cout << std::endl;
                std::cout << "At the first iteration, the step length is " << step_length_init << std::endl;
                std::cout << std::endl;
            }
        }
    } else if (step_method == GRADIENT_DEFINED){
        // Option 2: we modulate the step length according to the angle between the previous and current gradient directions.
        // If the angle is less than XX degree, which means the model update direction is successive, we should enlarge the step size
        // Otherwise, the step length should decrease
        CUSTOMREAL angle = direction_change_of_model_update(grid);
        if(i_inv != 0){
            if (angle > step_length_gradient_angle){
                CUSTOMREAL old_step_length = step_length_init;
                step_length_init    = std::max((CUSTOMREAL)0.0001, step_length_init * step_length_down);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The angle between two update darections is " << angle
                            << ". Because the angle is greater than " << step_length_gradient_angle << " degree, the step length decreases from "
                            << old_step_length << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            } else if (angle <= step_length_gradient_angle) {
                CUSTOMREAL old_step_length = step_length_init;
                step_length_init    = std::min((CUSTOMREAL)0.02, step_length_init * step_length_up);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The angle between two update darections is " << angle
                            << ". Because the angle is less than " << step_length_gradient_angle << " degree, the step length increases from "
                            << old_step_length << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            }
        } else {
            if(myrank == 0 && id_sim == 0){
                std::cout << std::endl;
                std::cout << "At the first iteration, the step length is " << step_length_init << std::endl;
                std::cout << std::endl;
            }
        }
    } else {
        std::cout << std::endl;
        std::cout << "No supported method for step size change, step keep the same: " << step_length_init << std::endl;
        std::cout << std::endl;
    }
    
}