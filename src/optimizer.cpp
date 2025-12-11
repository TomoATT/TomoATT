#include "optimizer.h"

Optimizer::Optimizer(InputParams& IP){

    n_total_loc_grid_points = loc_I * loc_J * loc_K;

    // (to do) only true if line search is applied
    if(true){    
        fun_loc_backup.resize(n_total_loc_grid_points);
        xi_loc_backup.resize(n_total_loc_grid_points);
        eta_loc_backup.resize(n_total_loc_grid_points);
    }
}

Optimizer::~Optimizer(){}


std::vector<CUSTOMREAL> Optimizer::model_update(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj, bool is_line_search) {

    std::vector<CUSTOMREAL> v_obj_misfit_line_search;

    // write out original kernels
    // Ks, Kxi, Keta, Ks_den, Kxi_den, Keta_den
    write_original_kernels(IP, grid, io, i_inv);

    // process kernels (specialized in derived classes)
    // Ks_loc, Keta_loc, Kxi_loc
    // --> 
    // Ks_update_loc, Keta_update_loc, Kxi_update_loc
    processing_kernels(IP, grid, io, i_inv);    // kernels have been broadcasted to all simultaneous groups 

    // write out modified kernels (descent direction)
    // Ks_update, Kxi_update, Keta_update, Ks_density_update, Kxi_density_update, Keta_density_update
    write_modified_kernels(IP, grid, io, i_inv);


    // determine step length, and set new model
    if (!is_line_search){
        // step-size controlled implementation
        determine_step_length_controlled(IP, grid, i_inv, v_obj_inout, old_v_obj);
    } else {
        // line search implementation
        v_obj_misfit_line_search = determine_step_length_line_search(IP, grid, io, i_inv, v_obj_inout);
    }

    // write new model
    write_new_model(IP, grid, io, i_inv);

    // make station correction
    IP.station_correction_update(step_length_init_sc);

    // writeout temporary xdmf file
    if (IP.get_verbose_output_level())
        io.update_xdmf_file();

    synchronize_all_world();

    return v_obj_misfit_line_search;
}

// ---------------------------------------------------
// ------------------ main functions ------------------
// ---------------------------------------------------


// write out original kernels
void Optimizer::write_original_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv){
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
    synchronize_all_world();
}


// VIRTUAL function, to be specialized in derived classes
void Optimizer::processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv){}


// write out modified kernels (descent direction)
void Optimizer::write_modified_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv){
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
void Optimizer::determine_step_length_controlled(InputParams& IP, Grid& grid, int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj) {

    if(subdom_main && id_sim == 0){     // main of level 1 and level 3 determine steo
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
            CUSTOMREAL angle = calculate_angle_between_grid_values(grid.Ks_update_loc_previous, grid.Ks_update_loc, n_total_loc_grid_points);
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

    // broadcast the step_length
    broadcast_cr_single(step_length_init,0);

    // set new model
    set_new_model(IP, grid, step_length_init);
}


// determine step length (line search method)
// 用二次函数近似的方式，最多给1次试探步。
// 设置最大步长
std::vector<CUSTOMREAL> Optimizer::determine_step_length_line_search(InputParams& IP, Grid& grid, IO_utils& io, int i_inv, CUSTOMREAL& v_obj_inout) {
    
    std::vector<CUSTOMREAL> v_obj_misfit_line_search;

    if (myrank == 0 && id_sim == 0){
        std::cout << "Line search to determine step length starting ... " << std::endl;
    }

    // ----------------------- step 1, backup current model -----------------------
    if (subdom_main){   // main of level 3 can backup model
        fun_loc_backup.assign(grid.fun_loc, grid.fun_loc + n_total_loc_grid_points);
        xi_loc_backup.assign(grid.xi_loc, grid.xi_loc + n_total_loc_grid_points);
        eta_loc_backup.assign(grid.eta_loc, grid.eta_loc + n_total_loc_grid_points);
    }

    // ----------------------- step 2, do line search -----------------------
    CUSTOMREAL alpha = step_length_init;        // initial step length
    int quit_sub_iter = 2;                      // maximum sub-iteration number to quit (avoid too many sub-iterations)
    CUSTOMREAL alpha_R = _0_CR;                 // upper bound of step length
    CUSTOMREAL alpha_L = _0_CR;                 // lower bound of step length
    std::vector<CUSTOMREAL> alpha_sub_iter(quit_sub_iter+1);     // store tried step lengths
    std::vector<CUSTOMREAL> v_obj_sub_iter(quit_sub_iter+1);     // store objective function values at tried step lengths


    // constant value for curvature condition (p_k is descent direction, Ks_update_loc is ascent direction, so use negative value)
    CUSTOMREAL c2_inner_product_old = - 0.9 * grid_value_dot_product(grid.Ks_update_loc, grid.Ks_loc, n_total_loc_grid_points);
    CUSTOMREAL angle_old = calculate_angle_between_grid_values(grid.Ks_update_loc, grid.Ks_loc, n_total_loc_grid_points);

    // main line search iteration
    for(int sub_iter = 0; sub_iter < 100; sub_iter++){

        if (myrank == 0 && id_sim == 0){
            std::cout << "Line search sub-iteration " << sub_iter << ", try step length alpha = " << alpha << std::endl << std::endl;
        }

        // substep 1, --------- back to the original model ---------
        if (subdom_main){
            std::copy(fun_loc_backup.begin(), fun_loc_backup.end(), grid.fun_loc);
            std::copy(xi_loc_backup.begin(), xi_loc_backup.end(), grid.xi_loc);
            std::copy(eta_loc_backup.begin(), eta_loc_backup.end(), grid.eta_loc);
        }

        // substep 2, --------- set new model with current alpha ---------
        if (subdom_main){
            set_new_model(IP, grid, alpha);
        }
        synchronize_all_world();

        // substep 3, --------- forward modeling + adjoint field + kernel  ---------
        v_obj_misfit_line_search = run_simulation_one_step(IP, grid, io, i_inv, true, false);
        CUSTOMREAL v_obj_try = v_obj_misfit_line_search[0];

        // substep 4, --------- evaluate the update performance ---------     
          
        bool cond_armijo    = check_armijo_condition(v_obj_inout, v_obj_try);      // condition 1, Armijo condition
        // bool cond_curvature = check_curvature_condition(grid, c2_inner_product_old); // condition 2, curvature condition
        bool cond_angle     = false;
        CUSTOMREAL angle_new    = calculate_angle_between_grid_values(grid.Ks_update_loc, grid.Ks_loc, n_total_loc_grid_points);    // have broadcasted within one subdomain
        if (std::abs(angle_new - angle_old) > 20.0 &&  std::abs(angle_new - angle_old) < 160.0){
            cond_angle = true;
        }

        // check conditions
        if(myrank == 0 && id_sim == 0){
            std::cout << std::endl; 
            std::cout << "Evaluate conditions at sub-iteration " << sub_iter << ": " << std::endl;
            std::cout << "    Tried step length alpha = " << alpha << std::endl;
            std::cout << "    Objective function value at current model: " << v_obj_inout << std::endl;
            std::cout << "    Objective function value at tried model: " << v_obj_try << std::endl;
            std::cout << "    Armijo condition satisfied: " << std::boolalpha << cond_armijo << std::endl;
            std::cout << "    Angle between descent direction and gradient at current model: " << angle_old << " degree" << std::endl;
            std::cout << "    Angle between descent direction and gradient at tried model: " << angle_new << " degree" << std::endl;
            std::cout << "    Angle condition satisfied (dif > 20 degree): " << std::boolalpha << cond_angle << std::endl;
        }

        alpha_sub_iter[sub_iter] = alpha;
        v_obj_sub_iter[sub_iter] = v_obj_try;

        // adjust step length
        if(sub_iter == quit_sub_iter-1){
            // must 
        } else {

        }


        if (cond_armijo && cond_curvature){
            // satisfy Wolfe conditions
            if(myrank == 0 && id_sim == 0){
                std::cout << "Satisfy Wolfe conditions at sub-iteration " << sub_iter 
                        << ", step length alpha = " << alpha 
                        << ", obj = " << v_obj_try << std::endl;
            }
            break;
        } else if (!cond_armijo && cond_curvature){
            // only satisfy curvature condition, step length is too large
            alpha_R = alpha;
            alpha = (alpha_L + alpha_R) / 2.0;
            if (sub_iter == quit_sub_iter) {
                if (myrank == 0 && id_sim == 0){
                    std::cout   << "Quit line search due to too many tries. Armijo condition not satisfied. "
                                << "Reduce the initial step length to " << alpha << " at the next iteration." << std::endl;
                }
                break;
            } else {
                if (myrank == 0 && id_sim == 0){
                    std::cout << "Armijo condition not satisfied" 
                            << ", step length may be too large, Reduce the searching step length from " << alpha_R 
                            << " to " << alpha << std::endl;
                }
            }
            
        } else if (cond_armijo && !cond_curvature){
            // only satisfy Armijo condition, step length is too small
            alpha_L =  alpha;
            if(alpha_R != _0_CR){
                alpha = (alpha_L + alpha_R) / 2.0;
            } else {
                alpha = 2.0 * alpha;
            }
            if (sub_iter == quit_sub_iter) {
                if (myrank == 0 && id_sim == 0){
                    std::cout << "Quit line search due to too many tries. Curvature condition not satisfied. "
                                << "Increase the initial step length to " << alpha << " at the next iteration." << std::endl;
                }
                break;
            } else {
                if (myrank == 0 && id_sim == 0){
                    mpi_debug_print_ranks();
                    std::cout << "Curvature condition not satisfied" 
                            << ", step length may be too small, Increase the searching step length from " << alpha_L 
                            << " to " << alpha << std::endl;
                }
            }
        } else {
            // neither condition is satisfied
            alpha_R = alpha;
            alpha = (alpha_L + alpha_R) / 2.0;
            if (sub_iter == quit_sub_iter) {
                if (myrank == 0 && id_sim == 0){
                    std::cout   << "Quit line search due to too many tries. Curvature and Armijo conditions not satisfied. "
                                << "Reduce the initial step length to " << alpha << " at the next iteration." << std::endl;
                }
                break;
            } else {
                if (myrank == 0 && id_sim == 0){
                    std::cout << "Curvature and Armijo conditions not satisfied" 
                            << ", step length may be too large, Reduce the searching step length from " << alpha_R 
                            << " to " << alpha << std::endl;
                }
            }
        }

    } // end for sub_iter

    // broadcast the step_length
    step_length_init = alpha;
    broadcast_cr_single(step_length_init,0);

    return v_obj_misfit_line_search;
}


// set new model
void Optimizer::set_new_model(InputParams& IP, Grid& grid, CUSTOMREAL step_length){

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

        // grid.rejuvenate_abcf();

        // shared values on the boundary
        grid.send_recev_boundary_data(grid.fun_loc);
        grid.send_recev_boundary_data(grid.xi_loc);
        grid.send_recev_boundary_data(grid.eta_loc);
        // grid.send_recev_boundary_data(grid.fac_a_loc);
        // grid.send_recev_boundary_data(grid.fac_b_loc);
        // grid.send_recev_boundary_data(grid.fac_c_loc);
        // grid.send_recev_boundary_data(grid.fac_f_loc);

    } // end if subdom_main

    // since model is update. The written traveltime field should be discraded
    // initialize is_T_written_into_file
    for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
        const std::string name_sim_src = IP.get_src_name(i_src);

        if (proc_store_srcrec) // only proc_store_srcrec has the src_map object
            IP.src_map[name_sim_src].is_T_written_into_file = false;
    }
}


// write out new models
void Optimizer::write_new_model(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv){
    if (is_write_model(IP, i_inv)) {
        //io.change_xdmf_obj(0); // change xmf file for next src
        io.change_group_name_for_model();

        // write out model info
        io.write_vel(grid, i_inv+1);
        io.write_xi( grid, i_inv+1);
        io.write_eta(grid, i_inv+1);

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
CUSTOMREAL Optimizer::calculate_angle_between_grid_values(CUSTOMREAL* vec1, CUSTOMREAL* vec2, int n){
    CUSTOMREAL norm_grad = _0_CR;
    CUSTOMREAL norm_grad_previous = _0_CR;
    CUSTOMREAL inner_product = _0_CR;
    CUSTOMREAL cos_angle = _0_CR;
    CUSTOMREAL angle = _0_CR;
    if (subdom_main) {
        inner_product      = grid_value_dot_product(vec1, vec2, n);
        norm_grad          = grid_value_dot_product(vec2, vec2, n);
        norm_grad_previous = grid_value_dot_product(vec1, vec1, n);

        // initiaize update params
        // inner_product      = grid_value_dot_product(grid.Ks_update_loc_previous, grid.Ks_update_loc, loc_I*loc_J*loc_K);
        // norm_grad          = grid_value_dot_product(grid.Ks_update_loc, grid.Ks_update_loc, loc_I*loc_J*loc_K);
        // norm_grad_previous = grid_value_dot_product(grid.Ks_update_loc_previous, grid.Ks_update_loc_previous, loc_I*loc_J*loc_K);
        // inner_product      = dot_product(grid.Ks_update_loc_previous, grid.Ks_update_loc, loc_I*loc_J*loc_K);
        // norm_grad          = dot_product(grid.Ks_update_loc, grid.Ks_update_loc, loc_I*loc_J*loc_K);
        // norm_grad_previous = dot_product(grid.Ks_update_loc_previous, grid.Ks_update_loc_previous, loc_I*loc_J*loc_K);

        // CUSTOMREAL tmp;
        // allreduce_cr_single(inner_product,tmp);
        // inner_product = tmp;

        // allreduce_cr_single(norm_grad,tmp);
        // norm_grad = tmp;

        // allreduce_cr_single(norm_grad_previous,tmp);
        // norm_grad_previous = tmp;

        cos_angle = inner_product / (std::sqrt(norm_grad) * std::sqrt(norm_grad_previous));
        angle     = acos(cos_angle) * RAD2DEG;
    }
    broadcast_cr_single_sub(angle, 0);  // broadcast within one subdomain
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

        // send the previous updated model to all the simultaneous run
        broadcast_cr_inter_sim(grid.Ks_update_loc_previous, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Kxi_update_loc_previous, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_update_loc_previous, loc_I*loc_J*loc_K, 0);
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

            // gather max_kernel from all processes
            CUSTOMREAL tmp;
            allreduce_cr_single_max(max_kernel, tmp);
            max_kernel = tmp;

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


// vector dot product
// (to do) remove boundary ghost points
CUSTOMREAL Optimizer::grid_value_dot_product(CUSTOMREAL* vec1, CUSTOMREAL* vec2, int n){
    CUSTOMREAL local_sum = dot_product(vec1, vec2, n);
    CUSTOMREAL global_sum;
    allreduce_cr_single(local_sum, global_sum);
    return global_sum;
}


// Armijo condition (c1 = 0)
bool Optimizer::check_armijo_condition(CUSTOMREAL v_obj_inout, CUSTOMREAL v_obj_try){
    // Armijo condition, sufficient decrease condition (modified)
    // f(x + alpha*p) <= f(x) + c1*alpha*grad_f(x)^T*p
    // However, the magnitude of kernel is perhaps not accurate. The direction of kernel is accurate.
    // So, we only check if f(x + alpha*p) < f(x)
    // This considtion ensures the step length is not too large.
    
    bool cond_armijo = false;
    if(subdom_main){    // check condition only for main of level 3
        if (v_obj_try < v_obj_inout){
            cond_armijo = true;
        }
    }
    //  broadcast the condition result to all processes within one subdomain
    broadcast_bool_single_sub(cond_armijo, 0);      // '_sub' means within one subdomain, subdom_main tell others
    return cond_armijo;
}

// curvature condition
bool Optimizer::check_curvature_condition(Grid& grid, CUSTOMREAL c2_inner_product_old){
    // curvature condition
    // grad_f(x + alpha*p)^T*p >= c2*grad_f(x)^T*p
    // This condition ensures the step length is not too small.
    
    bool cond_curvature = false;
    if(subdom_main){    // check condition only for main of level 3
        CUSTOMREAL inner_product_new = - grid_value_dot_product(grid.Ks_update_loc, grid.Ks_loc, n_total_loc_grid_points);
        if (inner_product_new >= c2_inner_product_old){
            cond_curvature = true;
        }
    }
    //  broadcast the condition result to all processes within one subdomain
    broadcast_bool_single_sub(cond_curvature, 0);      // '_sub' means within one subdomain, subdom_main tell others
    return cond_curvature;
}
