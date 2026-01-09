#include "optimizer_bfgs.h"
#include <iostream>
#include "kernel_postprocessing.h"
    
Optimizer_bfgs::Optimizer_bfgs(InputParams& IP) : Optimizer(IP) {
    // for bfgs method, model and kernel should be written
    need_write_model = true;
    need_write_original_kernel = true;

    // initialize sizes
    array_3d_forward.resize(n_total_loc_grid_points);
    array_3d_backward.resize(n_total_loc_grid_points);

    // vectors in bfgs
    sk_s.resize(n_total_loc_grid_points);
    sk_xi.resize(n_total_loc_grid_points);
    sk_eta.resize(n_total_loc_grid_points);
    yk_s.resize(n_total_loc_grid_points);
    yk_xi.resize(n_total_loc_grid_points);
    yk_eta.resize(n_total_loc_grid_points);

    Ks_bfgs_loc.resize(n_total_loc_grid_points);
    Kxi_bfgs_loc.resize(n_total_loc_grid_points);
    Keta_bfgs_loc.resize(n_total_loc_grid_points);

    // scalars in bfgs
    alpha_bfgs.resize(10000);
    rho.resize(10000);


    if(line_search_mode){
        alpha_sub_iter.resize(100);
        v_obj_sub_iter.resize(100);
        proj_sub_iter.resize(100);
        armijo_sub_iter.resize(100);
        curvature_sub_iter.resize(100);
    }
}

Optimizer_bfgs::~Optimizer_bfgs() {

}

// ---------------------------------------------------------
// ------------------ specified functions ------------------
// ---------------------------------------------------------

// smooth kernels (multigrid) + kernel normalization (kernel density normalization)
void Optimizer_bfgs::processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv) {
    
    // initialize and backup modified kernels
    initialize_and_backup_modified_kernels(grid);

    // check kernel value range
    check_kernel_value_range(grid);

    // multigrid parameterization + kernel density normalization
    // Ks_loc, Keta_loc, Kxi_loc
    // --> 
    // Ks_processing_loc, Keta_processing_loc, Kxi_processing_loc
    Kernel_postprocessing::process_kernels(IP, grid); 

    // write bfgs gradient
    write_bfgs_gradient(grid, io, i_inv);

    // backup bfgs gradient
    backup_bfgs_gradient(grid);
    
    // calculate bfgs descent direction
    calculate_bfgs_descent_direction(grid, io, i_inv);

    // normalize kernels to -1 ~ 1
    Kernel_postprocessing::normalize_kernels(grid);
    
    // assign processing kernels to modified kernels for model update
    // Ks_processing_loc, Keta_processing_loc, Kxi_processing_loc
    // -->
    // Ks_update_loc, Keta_update_loc, Kxi_update_loc
    Kernel_postprocessing::assign_to_modified_kernels(grid);
}


// evaluate line search performance
// (to do) allow users to adjust the step length change
bool Optimizer_bfgs::check_conditions_for_line_search(InputParams& IP, Grid& grid, int sub_iter, int quit_sub_iter, CUSTOMREAL v_obj_inout, CUSTOMREAL v_obj_try){
    bool exit_flag = false;

    // --------------- Armijo condition, sufficient decrease condition (modified) ---------------
    // f(x + alpha*p) <= f(x) + c1*alpha*grad_f(x)^T*p
    // However, the magnitude of kernel is perhaps not accurate. The direction of kernel is accurate.
    // So, we only check if f(x + alpha*p) < f(x)
    // This considtion ensures the decrease of objective function
    bool cond_armijo = false;
    if(subdom_main){    // check condition only for main of level 3
        if (v_obj_try < v_obj_inout){
            cond_armijo = true;
        }
    }
    //  broadcast the condition result to all processes within one subdomain
    broadcast_bool_single_sub(cond_armijo, 0);      // '_sub' means within one subdomain, subdom_main tell others


    // --------------- curvature condition ---------------
    // sk^T yk >= 0, to ensure positive definiteness of Hessian approximation (m_k+1 - m_k) * (g_k+1 - g_k) >= 0
    // (m_k+1 - m_k) is descent direction, so grid.Ks_update_loc is -p.
    // 

    // grad_f(x + alpha*p)^T*p >= grad_f(x)^T*p


    bool cond_curvature = false;
    CUSTOMREAL proj_current = _0_CR;   // grad_f(x_k)^T * p_k, projection of gradient on descent direction at current model
    CUSTOMREAL proj_tried = _0_CR;   // grad_f(x_k+1)^T * p_k, projection of gradient on descent direction at tried model
    if (subdom_main){    // check condition only for main of level 3
        // Ks_update_loc is -p = - alpha * (m_k+1 - m_k)
        // Ks_bfgs_loc is the backup of gradient at current model = g_k, that is, grad_f(x_k)
        proj_current -= grid_value_dot_product(grid.Ks_update_loc, Ks_bfgs_loc.data(), n_total_loc_grid_points);
        proj_current -= grid_value_dot_product(grid.Kxi_update_loc, Kxi_bfgs_loc.data(), n_total_loc_grid_points);
        proj_current -= grid_value_dot_product(grid.Keta_update_loc, Keta_bfgs_loc.data(), n_total_loc_grid_points);

        // Ks_processing_loc is gradient at tried model = g_k+1 = grad_f(x_k+1)
        proj_tried -= grid_value_dot_product(grid.Ks_update_loc, grid.Ks_processing_loc.data(), n_total_loc_grid_points);
        proj_tried -= grid_value_dot_product(grid.Kxi_update_loc, grid.Kxi_processing_loc.data(), n_total_loc_grid_points);
        proj_tried -= grid_value_dot_product(grid.Keta_update_loc, grid.Keta_processing_loc.data(), n_total_loc_grid_points);

        if (proj_tried >= proj_current){
            cond_curvature = true;
        }
    }
    //  broadcast the condition result to all processes within one subdomain
    broadcast_bool_single_sub(cond_curvature, 0);       // '_sub' means within one subdomain, subdom_main tell others


    alpha_sub_iter[sub_iter] = alpha;
    v_obj_sub_iter[sub_iter] = v_obj_try;
    proj_sub_iter[sub_iter] = proj_tried;
    armijo_sub_iter[sub_iter] = cond_armijo;
    curvature_sub_iter[sub_iter] = cond_curvature;

    // --------------- evaluate exit flag ---------------
    if(myrank == 0 && id_sim == 0){
        std::cout << std::endl;
        std::cout << "Evaluate conditions at sub-iteration " << sub_iter << ": " << std::endl;
        std::cout << "    Tried step length alpha = " << alpha << std::endl;
        std::cout << "    Objective function value at current model: " << v_obj_inout << std::endl;
        std::cout << "    Objective function value at tried model: " << v_obj_try << std::endl;
        std::cout << "    Armijo condition satisfied: " << std::boolalpha << cond_armijo << std::endl;
        std::cout << "    projection of gradient on descent direcction at current model: "  << proj_current << std::endl;
        std::cout << "    projection of gradient on descent direcction at tried model: "    << proj_tried << std::endl;
        std::cout << "    Curvature condition satisfied: " << std::boolalpha << cond_curvature << std::endl << std::endl;
    }

    if (cond_armijo && cond_curvature){
        // satisfy Wolfe conditions
        if(myrank == 0 && id_sim == 0){
            std::cout << "Satisfy Wolfe conditions at sub-iteration " << sub_iter 
                    << ", step length alpha = " << alpha 
                    << ", obj = " << v_obj_try << std::endl;
            std::cout << "In the next iteration, the initial step length increases set to " << std::min(1.2*alpha, step_length_max) << std::endl << std::endl;
        }
        alpha = std::min(1.2*alpha, step_length_max);
        exit_flag = true;
    } else if (!cond_armijo && cond_curvature){
        // only satisfy curvature condition, step length is too large
        alpha_R = alpha;
        alpha = (alpha_L + alpha_R) / 2.0;
        if (sub_iter == quit_sub_iter) {
            if (myrank == 0 && id_sim == 0){
                std::cout   << "Quit line search due to too many tries. Armijo condition not satisfied. "
                            << "Reduce the initial step length to " << alpha << " at the next iteration." << std::endl;
            }
            exit_flag = true;
        } else {
            if (myrank == 0 && id_sim == 0){
                std::cout << "Armijo condition not satisfied " << sub_iter 
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
            exit_flag = true;
        } else {
            if (myrank == 0 && id_sim == 0){
                mpi_debug_print_ranks();
                std::cout << "Curvature condition not satisfied" << sub_iter 
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
            exit_flag = true;
        } else {
            if (myrank == 0 && id_sim == 0){
                std::cout << "Curvature and Armijo conditions not satisfied " << sub_iter 
                        << ", step length may be too large, Reduce the searching step length from " << alpha_R 
                        << " to " << alpha << std::endl;
            }
        }
    }


    if (exit_flag){
        if (myrank == 0 && id_sim == 0){
            std::cout << std::endl;
            std::cout << "Line search ends. The search process contains  " << sub_iter+1 << " sub-iterations." << std::endl;
            std::cout << "The information is summarized as follows: " << std::endl;
            std::cout       << std::setw(25)        << "    Current, ";
            std::cout       << " step length = " << std::setw(4)    << 0;
            std::cout       << ", obj value = "  << std::setw(10)   << v_obj_inout << "        ";
            std::cout       << ", projection = " << std::setw(10)   << proj_current << std::endl;
            for(int tmp_i = 0; tmp_i <= sub_iter; tmp_i++){
                std::cout   << std::setw(23)    << "    Sub-iter "  << tmp_i+1 << ",";
                std::cout   << " step length = " << std::setw(4)    << alpha_sub_iter[tmp_i];
                std::cout   << ", obj value = " << std::setw(10)    << v_obj_sub_iter[tmp_i] << std::boolalpha << " (" << armijo_sub_iter[tmp_i] << ")";
                std::cout   << ", projection = " << std::setw(10)   << proj_sub_iter[tmp_i] << std::boolalpha << " (" << curvature_sub_iter[tmp_i] << ")" << std::endl;
            }
            std::cout << std::endl;
        }
    }



    return exit_flag;
}



// ---------------------------------------------------
// ------------------ sub functions ------------------
// ---------------------------------------------------

// calculate bfgs descent direction 
// y_i = g_{i+1} - g_i, gradient difference
// s_i = m_{i+1} - m_i, model difference
void Optimizer_bfgs::calculate_bfgs_descent_direction(Grid& grid, IO_utils& io, int& i_inv) {
    if(subdom_main){
        if(id_sim == 0){
            if (i_inv > 0) {
                // --------------- step 1,  initialize q = g_k (in this step, descent_dir is q)
                // do nothing. 
                // grid.Ks_processing_loc is the gradient at current model

                // --------------- step 2, loop for i = k-1, k-2, ..., k-Mbfgs  (in this step, descent_dir is q)
                // alpha_i = rho_i * s_i^T * q
                // q = q - alpha_i * y_i
                int n_stored = std::min(i_inv, Mbfgs);
                for (int i_bfgs = i_inv-1; i_bfgs >= i_inv-n_stored; i_bfgs--) {
                    // --------------- substep 1, calculate rho_i = 1 / (y_i^T * s_i)
                    get_model_dif(grid, io, i_bfgs);        // obtain s_i (model difference)
                    get_gradient_dif(grid, io, i_bfgs);     // obtain y_i (gradient difference)
                    rho[i_bfgs]     = grid_value_dot_product(yk_s.data(), sk_s.data(), n_total_loc_grid_points);      // for slowness
                    rho[i_bfgs]    += grid_value_dot_product(yk_xi.data(), sk_xi.data(), n_total_loc_grid_points);   // for xi
                    rho[i_bfgs]    += grid_value_dot_product(yk_eta.data(), sk_eta.data(), n_total_loc_grid_points);  // for eta
                    rho[i_bfgs]     = 1.0 / rho[i_bfgs];

                    // --------------- substep 2, calculate alpha_i = rho_i * s_i^T * q
                    alpha_bfgs[i_bfgs]  = grid_value_dot_product(sk_s.data(), grid.Ks_processing_loc.data(), n_total_loc_grid_points);
                    alpha_bfgs[i_bfgs] += grid_value_dot_product(sk_xi.data(), grid.Kxi_processing_loc.data(), n_total_loc_grid_points);
                    alpha_bfgs[i_bfgs] += grid_value_dot_product(sk_eta.data(), grid.Keta_processing_loc.data(), n_total_loc_grid_points);  
                    alpha_bfgs[i_bfgs] *= rho[i_bfgs];

                    // --------------- substep 3, q = q - alpha_i * y_i
                    for (int idx = 0; idx < n_total_loc_grid_points; idx++) {
                        grid.Ks_processing_loc[idx]   -= alpha_bfgs[i_bfgs] * yk_s[idx];
                        grid.Kxi_processing_loc[idx]  -= alpha_bfgs[i_bfgs] * yk_xi[idx];
                        grid.Keta_processing_loc[idx] -= alpha_bfgs[i_bfgs] * yk_eta[idx];
                    }
                    
                }

                // --------------- step 3, scaling of initial Hessian H0_k  (in this step, descent_dir is z)
                // substep 1, calculate gamma_k = (s_{k-1}^T * y_{k-1}) / (y_{k-1}^T * y_{k-1})
                int i_bfgs = i_inv - 1;
                get_model_dif(grid, io, i_bfgs);     // obtain s_{k-1} (model difference)
                get_gradient_dif(grid, io, i_bfgs);  // obtain y_{k-1} (gradient difference)
                CUSTOMREAL sT_y = _0_CR;
                sT_y  = grid_value_dot_product(sk_s.data(), yk_s.data(), n_total_loc_grid_points);
                sT_y += grid_value_dot_product(sk_xi.data(), yk_xi.data(), n_total_loc_grid_points);
                sT_y += grid_value_dot_product(sk_eta.data(), yk_eta.data(), n_total_loc_grid_points);
                CUSTOMREAL yT_y = _0_CR;
                yT_y  = grid_value_dot_product(yk_s.data(), yk_s.data(), n_total_loc_grid_points);
                yT_y += grid_value_dot_product(yk_xi.data(), yk_xi.data(), n_total_loc_grid_points);
                yT_y += grid_value_dot_product(yk_eta.data(), yk_eta.data(), n_total_loc_grid_points);
                CUSTOMREAL gamma_k = sT_y / yT_y;

                // substep 2, z = gamma_k * q (because H0_k = gamma_k * I)
                for (int idx = 0; idx < n_total_loc_grid_points; idx++) {
                    grid.Ks_processing_loc[idx]   *= gamma_k;
                    grid.Kxi_processing_loc[idx]  *= gamma_k;
                    grid.Keta_processing_loc[idx] *= gamma_k;
                }

                // --------------- step 4, loop for i = k-Mbfgs, k-Mbfgs+1, ..., k-1  (in this step, descent_dir is z)
                for (int i_bfgs = i_inv-n_stored; i_bfgs <= i_inv-1; i_bfgs++) {
                    // --------------- substep 1, beta = rho_i * y_i^T * z
                    get_gradient_dif(grid, io, i_bfgs); // obtain y_i (gradient difference)
                    CUSTOMREAL beta = _0_CR;
                    beta  = grid_value_dot_product(yk_s.data(), grid.Ks_processing_loc.data(), n_total_loc_grid_points);
                    beta += grid_value_dot_product(yk_xi.data(), grid.Kxi_processing_loc.data(), n_total_loc_grid_points);
                    beta += grid_value_dot_product(yk_eta.data(), grid.Keta_processing_loc.data(), n_total_loc_grid_points);
                    beta *= rho[i_bfgs];

                    // --------------- substep 2, z = z + s_i * (alpha_i - beta)
                    get_model_dif(grid, io, i_bfgs);     // obtain s_i (model difference)
                    for (int idx = 0; idx < n_total_loc_grid_points; idx++) {
                        grid.Ks_processing_loc[idx]   += sk_s[idx] * (alpha_bfgs[i_bfgs] - beta);
                        grid.Kxi_processing_loc[idx]  += sk_xi[idx] * (alpha_bfgs[i_bfgs] - beta);
                        grid.Keta_processing_loc[idx] += sk_eta[idx] * (alpha_bfgs[i_bfgs] - beta);
                    }
                }

                // --------------- final step, set modified kernels
                // do nothing. Now, grid.Ks_processing_loc is the descent direction

            } else {
                // for the first iteration, use steepest descent (do nothing)
            }
        }

        broadcast_cr_inter_sim(grid.Ks_processing_loc.data(), loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Kxi_processing_loc.data(), loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_processing_loc.data(), loc_I*loc_J*loc_K, 0);
    }
    
    synchronize_all_world();
}


// write bfgs gradient ()
void Optimizer_bfgs::write_bfgs_gradient(Grid& grid, IO_utils& io, int& i_inv){
    if (id_sim == 0 && subdom_main){
        // store kernel only in the first src datafile
        io.change_group_name_for_model();

        // write descent direction
        io.write_Ks_bfgs(grid, i_inv);
        io.write_Keta_bfgs(grid, i_inv);
        io.write_Kxi_bfgs(grid, i_inv);
    }
}


// backup bfgs gradient
void Optimizer_bfgs::backup_bfgs_gradient(Grid& grid){
    if (subdom_main){
        // backup bfgs gradient
        Ks_bfgs_loc = grid.Ks_processing_loc;
        Kxi_bfgs_loc = grid.Kxi_processing_loc;
        Keta_bfgs_loc = grid.Keta_processing_loc;
    }
    synchronize_all_world();
}


// read histrorical model difference
void Optimizer_bfgs::get_model_dif(Grid& grid, IO_utils& io, int& i_inv){
    // make h5_group_name_data to be "model"
    io.change_group_name_for_model();

    // slowness perturbation, delta ln(1/vel)
    io.read_vel(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_vel(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        sk_s[i] = std::log(1.0/array_3d_forward[i]) - std::log(1.0/array_3d_backward[i]);
    
    // delta xi
    io.read_xi(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_xi(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        sk_xi[i] = array_3d_forward[i] - array_3d_backward[i];

    // delta eta
    io.read_eta(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_eta(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        sk_eta[i] = array_3d_forward[i] - array_3d_backward[i];
}


// read histrorical gradient difference
void Optimizer_bfgs::get_gradient_dif(Grid& grid, IO_utils& io, int& i_inv){
    // make h5_group_name_data to be "model"
    io.change_group_name_for_model();

    // slowness gradient difference
    io.read_Ks_bfgs(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_Ks_bfgs(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        yk_s[i] = array_3d_forward[i] - array_3d_backward[i];

    // xi gradient difference
    io.read_Kxi_bfgs(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_Kxi_bfgs(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        yk_xi[i] = array_3d_forward[i] - array_3d_backward[i];
        
    // eta gradient difference
    io.read_Keta_bfgs(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_Keta_bfgs(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        yk_eta[i] = array_3d_forward[i] - array_3d_backward[i];
}



