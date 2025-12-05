#include "optimizer.h"

Optimizer::Optimizer(InputParams& IP){}

Optimizer::~Optimizer(){}


void Optimizer::model_update(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj, bool is_line_search) {
    if(id_sim == 0 && myrank == 0)
        std::cout << "Gradient descent update\n";

    // check kernel density
    check_kernel_density(grid, IP);

    // sum up kernels from all simulateous group (level 1) 
    sumup_kernels(grid);

    // write out original kernels
    // Ks, Kxi, Keta, Ks_den, Kxi_den, Keta_den
    write_original_kernels(grid, IP, io, i_inv);

    // process kernels (specialized in derived classes)
    // Ks_loc, Keta_loc, Kxi_loc
    // --> 
    // Ks_update_loc, Keta_update_loc, Kxi_update_loc
    processing_kernels(grid, io, IP, i_inv);

    // write out modified kernels (descent direction)
    // Ks_update, Kxi_update, Keta_update, Ks_density_update, Kxi_density_update, Keta_density_update
    write_modified_kernels(grid, IP, io, i_inv);

    // determine step length (specialized in derived classes)
    determine_step_length(grid, i_inv, v_obj_inout, old_v_obj, is_line_search);

    // set new model
    set_new_model(grid, step_length_init);

    // write new model
    write_new_model(grid, IP, io, i_inv);

    // make station correction
    IP.station_correction_update(step_length_init_sc);

    // writeout temporary xdmf file
    if (IP.get_verbose_output_level())
        io.update_xdmf_file();

    synchronize_all_world();
}

// ---------------------------------------------------
// ------------------ main functions ------------------
// ---------------------------------------------------


// check kernel density
void Optimizer::check_kernel_density(Grid& grid, InputParams& IP) {
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
void Optimizer::sumup_kernels(Grid& grid) {
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


// write out original kernels
void Optimizer::write_original_kernels(Grid& grid, InputParams& IP, IO_utils& io, int& i_inv){
    if (is_write_original_kernel(IP, i_inv)) {
        // store kernel only in the first src datafile
        io.change_group_name_for_model();

        // write original kernel
        io.write_Ks(grid, i_inv);
        io.write_Keta(grid, i_inv);
        io.write_Kxi(grid, i_inv);

        // write original kernel density
        io.write_Ks_density(grid, i_inv);
        io.write_Kxi_density(grid, i_inv);
        io.write_Keta_density(grid, i_inv);
    }
}


// VIRTUAL function, to be specialized in derived classes
void Optimizer::processing_kernels(Grid& grid, IO_utils& io, InputParams& IP, int& i_inv){}


// write out modified kernels (descent direction)
void Optimizer::write_modified_kernels(Grid& grid, InputParams& IP, IO_utils& io, int& i_inv){
    if (is_write_modified_kernel(IP, i_inv)) {
        // store kernel only in the first src datafile
        io.change_group_name_for_model();

        // write descent direction
        io.write_Ks_update(grid, i_inv);
        io.write_Keta_update(grid, i_inv);
        io.write_Kxi_update(grid, i_inv);

        // write kernel density with smoothing
        io.write_Ks_density_update(grid, i_inv);
        io.write_Kxi_density_update(grid, i_inv);
        io.write_Keta_density_update(grid, i_inv);
    }
}


// determine step length
void Optimizer::determine_step_length(Grid& grid, int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj, bool is_line_search) {

    if (!is_line_search) {
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
    } else {
        if(myrank == 0 && id_sim == 0){
            std::cout << std::endl;
            std::cout << "(to do)" << std::endl;
        }
    }
}


// set new model
void Optimizer::set_new_model(Grid& grid, CUSTOMREAL step_length){

    if (subdom_main) {
        for (int k = 0; k < loc_K; k++) {
            for (int j = 0; j < loc_J; j++) {
                for (int i = 0; i < loc_I; i++) {
                    // update
                    grid.fun_loc[I2V(i,j,k)] *= (_1_CR - grid.Ks_update_loc[I2V(i,j,k)  ] * step_length);
                    grid.xi_loc[I2V(i,j,k)]  -=          grid.Kxi_update_loc[I2V(i,j,k) ] * step_length;
                    grid.eta_loc[I2V(i,j,k)] -=          grid.Keta_update_loc[I2V(i,j,k)] * step_length;

                }
            }
        }

        grid.rejuvenate_abcf();

        // shared values on the boundary
        grid.send_recev_boundary_data(grid.fun_loc);
        grid.send_recev_boundary_data(grid.xi_loc);
        grid.send_recev_boundary_data(grid.eta_loc);
        grid.send_recev_boundary_data(grid.fac_b_loc);
        grid.send_recev_boundary_data(grid.fac_c_loc);
        grid.send_recev_boundary_data(grid.fac_f_loc);

    } // end if subdom_main
}


// write out new models
void Optimizer::write_new_model(Grid& grid, InputParams& IP, IO_utils& io, int& i_inv){
    if (is_write_model(IP, i_inv)) {
        //io.change_xdmf_obj(0); // change xmf file for next src
        io.change_group_name_for_model();

        // write out model info
        if (IP.get_if_output_in_process() || i_inv >= IP.get_max_iter_inv() - 2){
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
    }
}

// ---------------------------------------------------
// ------------------ sub functions ------------------
// ---------------------------------------------------

// calculate the angle between previous and current model update directions
CUSTOMREAL Optimizer::direction_change_of_model_update(Grid& grid){
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


// initialize and backup modified kernels
void Optimizer::initialize_and_backup_modified_kernels(Grid& grid) {
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
void Optimizer::check_kernel_value_range(Grid& grid) {
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


// check write model and kernel condition
bool Optimizer::is_write_model(InputParams& IP, int& i_inv){
    bool is_write = false;

    is_write =  (id_sim == 0 && subdom_main) &&         //  read and write can only be done by main process (level 3) in main simulateous group (level 1)
                (need_write_model ||                    // if this method needs to read and write model,
                IP.get_if_output_in_process() ||        // or users request to write the model at intermediate iterations
                (i_inv >= IP.get_max_iter_inv() - 2));  // or at the last two iterations
    return is_write;
}


bool Optimizer::is_write_modified_kernel(InputParams& IP, int& i_inv){
    bool is_write = false;

    is_write =  (id_sim == 0 && subdom_main) &&             //  read and write can only be done by main process (level 3) in main simulateous group (level 1)
                IP.get_if_output_kernel() &&                // if users request to write the kernel
                (IP.get_if_output_in_process() ||           // if users request to writeat intermediate iterations
                (i_inv >= IP.get_max_iter_inv() - 2));      // or at the last two iterations

    return is_write;
}


bool Optimizer::is_write_original_kernel(InputParams& IP, int& i_inv){
    bool is_write = false;

    is_write =  (id_sim == 0 && subdom_main) &&                                         // read and write can only be done by main process (level 3) in main simulateous group (level 1)
                (need_write_original_kernel ||                                          // if this method needs to read and write kernel,
                (IP.get_if_output_kernel() && IP.get_if_output_in_process()) ||         // or users request to write the kernel at intermediate iterations
                (IP.get_if_output_kernel() && (i_inv >= IP.get_max_iter_inv() - 2)));   // or users request to write the kernel, but at the last two iterations

    return is_write;
}

