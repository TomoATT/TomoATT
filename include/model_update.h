#ifndef MODEL_UPDATE_H
#define MODEL_UPDATE_H

#include <cmath>
#include "config.h"
#include "grid.h"
#include "smooth.h"
#include "smooth_descent_dir.h"
#include "smooth_grad_regul.h"
#include "lbfgs.h"


// generate smoothed kernels (K*_update_loc) from the kernels (K*_loc)
// before doing this, K*_loc should be summed up among all simultaneous runs (by calling sumup_kernels)
// before doing this, K*_update_loc has no meaning (unavailable)
void smooth_kernels(Grid& grid, InputParams& IP) {

    if (subdom_main){ // parallel level 3

        if (id_sim==0){ // parallel level 1
            // initiaize update params
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        grid.Ks_update_loc_previous[I2V(i,j,k)]   = grid.Ks_update_loc[I2V(i,j,k)];
                        grid.Keta_update_loc_previous[I2V(i,j,k)] = grid.Keta_update_loc[I2V(i,j,k)];
                        grid.Kxi_update_loc_previous[I2V(i,j,k)]  = grid.Kxi_update_loc[I2V(i,j,k)];
                        grid.Ks_update_loc[I2V(i,j,k)]   = _0_CR;
                        grid.Keta_update_loc[I2V(i,j,k)] = _0_CR;
                        grid.Kxi_update_loc[I2V(i,j,k)]  = _0_CR;
                        // grid.Ks_density_update_loc[I2V(i,j,k)]  = _0_CR;
                        // grid.Kxi_density_update_loc[I2V(i,j,k)] = _0_CR;
                        // grid.Keta_density_update_loc[I2V(i,j,k)] = _0_CR;
                    }
                }
            }

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

            if (smooth_method == 0) {
                // Ks_loc, Keta_loc, Kxi_loc to be 0 for ghost layers to eliminate the effect of overlapping boundary
                for (int k = 0; k < loc_K; k++) {
                    for (int j = 0; j < loc_J; j++) {
                        for (int i = 0; i < loc_I; i++) {

                            if (i == 0 && !grid.i_first()) {
                                grid.Ks_loc[I2V(i,j,k)]         = _0_CR;
                                grid.Keta_loc[I2V(i,j,k)]       = _0_CR;
                                grid.Kxi_loc[I2V(i,j,k)]        = _0_CR;
                                grid.Ks_density_loc[I2V(i,j,k)]     = _0_CR;
                                grid.Kxi_density_loc[I2V(i,j,k)]    = _0_CR;
                                grid.Keta_density_loc[I2V(i,j,k)]   = _0_CR;
                            }
                            if (i == loc_I-1 && !grid.i_last()) {
                                grid.Ks_loc[I2V(i,j,k)]         = _0_CR;
                                grid.Keta_loc[I2V(i,j,k)]       = _0_CR;
                                grid.Kxi_loc[I2V(i,j,k)]        = _0_CR;
                                grid.Ks_density_loc[I2V(i,j,k)]     = _0_CR;
                                grid.Kxi_density_loc[I2V(i,j,k)]    = _0_CR;
                                grid.Keta_density_loc[I2V(i,j,k)]   = _0_CR;
                            }
                            if (j == 0 && !grid.j_first()) {
                                grid.Ks_loc[I2V(i,j,k)]         = _0_CR;
                                grid.Keta_loc[I2V(i,j,k)]       = _0_CR;
                                grid.Kxi_loc[I2V(i,j,k)]        = _0_CR;
                                grid.Ks_density_loc[I2V(i,j,k)]     = _0_CR;
                                grid.Kxi_density_loc[I2V(i,j,k)]    = _0_CR;
                                grid.Keta_density_loc[I2V(i,j,k)]   = _0_CR;
                            }
                            if (j == loc_J-1 && !grid.j_last()) {
                                grid.Ks_loc[I2V(i,j,k)]         = _0_CR;
                                grid.Keta_loc[I2V(i,j,k)]       = _0_CR;
                                grid.Kxi_loc[I2V(i,j,k)]        = _0_CR;
                                grid.Ks_density_loc[I2V(i,j,k)]     = _0_CR;
                                grid.Kxi_density_loc[I2V(i,j,k)]    = _0_CR;
                                grid.Keta_density_loc[I2V(i,j,k)]   = _0_CR;
                            }
                            if (k == 0 && !grid.k_first()) {
                                grid.Ks_loc[I2V(i,j,k)]         = _0_CR;
                                grid.Keta_loc[I2V(i,j,k)]       = _0_CR;
                                grid.Kxi_loc[I2V(i,j,k)]        = _0_CR;
                                grid.Ks_density_loc[I2V(i,j,k)]     = _0_CR;
                                grid.Kxi_density_loc[I2V(i,j,k)]    = _0_CR;
                                grid.Keta_density_loc[I2V(i,j,k)]   = _0_CR;
                            }
                            if (k == loc_K-1 && !grid.k_last()) {
                                grid.Ks_loc[I2V(i,j,k)]         = _0_CR;
                                grid.Keta_loc[I2V(i,j,k)]       = _0_CR;
                                grid.Kxi_loc[I2V(i,j,k)]        = _0_CR;
                                grid.Ks_density_loc[I2V(i,j,k)]     = _0_CR;
                                grid.Kxi_density_loc[I2V(i,j,k)]    = _0_CR;
                                grid.Keta_density_loc[I2V(i,j,k)]   = _0_CR;
                            }
                        }
                    }
                }

                // grid based smoothing
                smooth_inv_kernels_orig(grid, IP);
            } else if (smooth_method == 1) {
                // CG smoothing
                smooth_inv_kernels_CG(grid, smooth_lr, smooth_lt, smooth_lp);
            }

            // shared values on the boundary
            grid.send_recev_boundary_data(grid.Ks_update_loc);
            grid.send_recev_boundary_data(grid.Keta_update_loc);
            grid.send_recev_boundary_data(grid.Kxi_update_loc);
            grid.send_recev_boundary_data(grid.Ks_density_update_loc);
            grid.send_recev_boundary_data(grid.Kxi_density_update_loc);
            grid.send_recev_boundary_data(grid.Keta_density_update_loc);

            grid.send_recev_boundary_data_kosumi(grid.Ks_update_loc);
            grid.send_recev_boundary_data_kosumi(grid.Keta_update_loc);
            grid.send_recev_boundary_data_kosumi(grid.Kxi_update_loc);
            grid.send_recev_boundary_data_kosumi(grid.Ks_density_update_loc);
            grid.send_recev_boundary_data_kosumi(grid.Kxi_density_update_loc);
            grid.send_recev_boundary_data_kosumi(grid.Keta_density_update_loc);

        } // end if id_sim == 0

        // send the updated model to all the simultaneous run
        broadcast_cr_inter_sim(grid.Ks_update_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Kxi_update_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_update_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Ks_density_update_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Kxi_density_update_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_density_update_loc, loc_I*loc_J*loc_K, 0);

        // send the previous updated model to all the simultaneous run
        broadcast_cr_inter_sim(grid.Ks_update_loc_previous, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Kxi_update_loc_previous, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_update_loc_previous, loc_I*loc_J*loc_K, 0);

    } // end if subdom_main

    synchronize_all_world();
}


void normalize_gradient(Grid& grid){
    if (subdom_main){
        CUSTOMREAL Linf_Ks, Linf_Keta, Linf_Kxi;
        Linf_Ks = find_absmax(grid.Ks_update_loc, loc_I*loc_J*loc_K);
        Linf_Keta = find_absmax(grid.Keta_update_loc, loc_I*loc_J*loc_K);
        Linf_Kxi = find_absmax(grid.Kxi_update_loc, loc_I*loc_J*loc_K);

        CUSTOMREAL Linf_tmp;
        allreduce_cr_single_max(Linf_Ks, Linf_tmp);   Linf_Ks = Linf_tmp;
        allreduce_cr_single_max(Linf_Keta, Linf_tmp); Linf_Keta = Linf_tmp;
        allreduce_cr_single_max(Linf_Kxi, Linf_tmp);  Linf_Kxi = Linf_tmp;

        CUSTOMREAL Linf_all = _0_CR;
        Linf_all = std::max(Linf_Ks, std::max(Linf_Keta, Linf_Kxi));

        for (int i = 0; i < loc_I*loc_J*loc_K; i++) {
            grid.Ks_update_loc[i]   /= Linf_all;
            grid.Keta_update_loc[i] /= Linf_all;
            grid.Kxi_update_loc[i]  /= Linf_all;
        }
    }
    synchronize_all_world();
}

// smooth gradient regularization term
void smooth_gradient_regularization(Grid& grid) {

    if (subdom_main && id_sim==0){ // only id_sim==0 has values for these arrays
        if (smooth_method == 0) {
            // grid based smoothing
            smooth_gradient_regularization_orig(grid);
        } else if (smooth_method == 1) {
            // CG smoothing
            smooth_gradient_regularization_CG(grid, smooth_lr, smooth_lt, smooth_lp);
        }

        grid.send_recev_boundary_data(grid.fun_gradient_regularization_penalty_loc);
        grid.send_recev_boundary_data(grid.eta_gradient_regularization_penalty_loc);
        grid.send_recev_boundary_data(grid.xi_gradient_regularization_penalty_loc);
    }
}


void smooth_descent_direction(Grid& grid){
    if (subdom_main){
       if (id_sim == 0) { // calculation of the update model is only done in the main simultaneous run
            if (smooth_method == 0) {
                // grid based smoothing
                smooth_descent_dir(grid);
            } else if (smooth_method == 1) {
                // CG smoothing
                smooth_descent_dir_CG(grid, smooth_lr, smooth_lt, smooth_lp);
            }

            // shared values on the boundary
            grid.send_recev_boundary_data(  grid.Ks_descent_dir_loc);
            grid.send_recev_boundary_data(grid.Keta_descent_dir_loc);
            grid.send_recev_boundary_data( grid.Kxi_descent_dir_loc);

        } // end if id_sim == 0

        // send the updated model to all the simultaneous run
        broadcast_cr_inter_sim(  grid.Ks_descent_dir_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim( grid.Kxi_descent_dir_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_descent_dir_loc, loc_I*loc_J*loc_K, 0);

    }
}


void calc_descent_direction(Grid& grid, int i_inv, InputParams& IP) {

    if (subdom_main) {

        // routines for LBFGS
        if (optim_method == LBFGS_MODE) {

            // calculate the descent direction
            if (i_inv > 0) {
                calculate_descent_direction_lbfgs(grid, i_inv);
            // use gradient for the first iteration
            } else {
                int n_grid = loc_I*loc_J*loc_K;

                // first time, descent direction = - precond * gradient
                // inverse the gradient to fit the update scheme for LBFGS
                for (int i = 0; i < n_grid; i++){
                    grid.Ks_descent_dir_loc[i]   = - _1_CR* grid.Ks_update_loc[i];
                    grid.Keta_descent_dir_loc[i] = - _1_CR* grid.Keta_update_loc[i];
                    grid.Kxi_descent_dir_loc[i]  = - _1_CR* grid.Kxi_update_loc[i];
                    //grid.Ks_descent_dir_loc[i]   = - _1_CR* grid.Ks_loc[i];
                    //grid.Keta_descent_dir_loc[i] = - _1_CR* grid.Keta_loc[i];
                    //grid.Kxi_descent_dir_loc[i]  = - _1_CR* grid.Kxi_loc[i];

                }
            }
        } else {
            // return error
            std::cout << "Error: optim_method is not set to LBFGS_MODE (=2)" << std::endl;
            exit(1);
        }


    } // end if subdom_main

    synchronize_all_world();

}

CUSTOMREAL direction_change_of_model_update(Grid& grid){
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


void set_new_model(Grid& grid, CUSTOMREAL step_length_new, bool init_bfgs=false) {

    if (subdom_main) {

        // for LBFGS mode. K*_update_loc is not directly the descent direction but smoothed gradient
        // so for non LBFGS_mode, nextstep will be calculated with K*_update_loc
        if (optim_method != LBFGS_MODE) {
            // // get the scaling factor
            // CUSTOMREAL Linf_Ks = _0_CR, Linf_Keta = _0_CR, Linf_Kxi = _0_CR;
            // for (int k = 1; k < loc_K-1; k++) {
            //     for (int j = 1; j < loc_J-1; j++) {
            //         for (int i = 1; i < loc_I-1; i++) {
            //             Linf_Ks   = std::max(Linf_Ks,   std::abs(grid.Ks_update_loc[I2V(i,j,k)]));
            //             Linf_Keta = std::max(Linf_Keta, std::abs(grid.Keta_update_loc[I2V(i,j,k)]));
            //             Linf_Kxi  = std::max(Linf_Kxi,  std::abs(grid.Kxi_update_loc[I2V(i,j,k)]));
            //         }
            //     }
            // }

            // // get the maximum scaling factor among subdomains
            // CUSTOMREAL Linf_tmp;
            // allreduce_cr_single_max(Linf_Ks, Linf_tmp);   Linf_Ks = Linf_tmp;
            // allreduce_cr_single_max(Linf_Keta, Linf_tmp); Linf_Keta = Linf_tmp;
            // allreduce_cr_single_max(Linf_Kxi, Linf_tmp);  Linf_Kxi = Linf_tmp;

            // CUSTOMREAL Linf_all = _0_CR;
            // Linf_all = std::max(Linf_Ks, std::max(Linf_Keta, Linf_Kxi));

            // // if (myrank == 0 && id_sim == 0)
            // //    std::cout << "Scaling factor for all kernels: " << Linf_all << std::endl;
            // //    std::cout << "Scaling factor for model update for Ks, Keta, Kx, stepsize: " << Linf_Ks << ", " << Linf_Keta << ", " << Linf_Kxi << ", " << step_length_new << std::endl;


            // Linf_Ks = Linf_all;
            // Linf_Keta = Linf_all;
            // Linf_Kxi = Linf_all;

            // kernel update has been rescaled in "smooth_inv_kernels_orig"

            // update the model
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {

                        // // update
                        // grid.fun_loc[I2V(i,j,k)] *= (_1_CR - grid.Ks_update_loc[I2V(i,j,k)  ] / (Linf_Ks   /step_length_new) );
                        // grid.xi_loc[I2V(i,j,k)]  -=          grid.Kxi_update_loc[I2V(i,j,k) ] / (Linf_Kxi  /step_length_new)  ;
                        // grid.eta_loc[I2V(i,j,k)] -=          grid.Keta_update_loc[I2V(i,j,k)] / (Linf_Keta /step_length_new)  ;

                        // update
                        grid.fun_loc[I2V(i,j,k)] *= (_1_CR - grid.Ks_update_loc[I2V(i,j,k)  ] * step_length_new);
                        grid.xi_loc[I2V(i,j,k)]  -=          grid.Kxi_update_loc[I2V(i,j,k) ] * step_length_new;
                        grid.eta_loc[I2V(i,j,k)] -=          grid.Keta_update_loc[I2V(i,j,k)] * step_length_new;


                        // grid.fac_b_loc[I2V(i,j,k)] = _1_CR - _2_CR * grid.xi_loc[I2V(i,j,k)];
                        // grid.fac_c_loc[I2V(i,j,k)] = _1_CR + _2_CR * grid.xi_loc[I2V(i,j,k)];
                        // grid.fac_f_loc[I2V(i,j,k)] =       - _2_CR * grid.eta_loc[I2V(i,j,k)];

                        // grid.fun_loc[I2V(i,j,k)] *= (_1_CR);
                        // grid.xi_loc[I2V(i,j,k)]  -= (_0_CR);
                        // grid.eta_loc[I2V(i,j,k)] -= (_0_CR);

                        // grid.fac_b_loc[I2V(i,j,k)] = _1_CR - _2_CR * grid.xi_loc[I2V(i,j,k)];
                        // grid.fac_c_loc[I2V(i,j,k)] = _1_CR + _2_CR * grid.xi_loc[I2V(i,j,k)];
                        // grid.fac_f_loc[I2V(i,j,k)] =       - _2_CR * grid.eta_loc[I2V(i,j,k)];

                    }
                }
            }

            grid.rejuvenate_abcf();


        } else { // for LBFGS routine

            // here all the simultaneous runs have the same values used in this routine.
            // thus we don't need to if(id_sim==0)

            CUSTOMREAL step_length = step_length_new;
            const CUSTOMREAL factor = - _1_CR;

            // update the model
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        // update
                        grid.fun_loc[I2V(i,j,k)] *= (_1_CR - factor * grid.Ks_descent_dir_loc[I2V(i,j,k)]   * step_length);
                        grid.xi_loc[I2V(i,j,k)]  -=          factor * grid.Kxi_descent_dir_loc[I2V(i,j,k) ] * step_length;
                        grid.eta_loc[I2V(i,j,k)] -=          factor * grid.Keta_descent_dir_loc[I2V(i,j,k)] * step_length;
                        //grid.fun_loc[I2V(i,j,k)] += grid.Ks_descent_dir_loc[I2V(i,j,k)]  *step_length_new;
                        //grid.xi_loc[I2V(i,j,k)]  += grid.Kxi_descent_dir_loc[I2V(i,j,k)] *step_length_new;
                        //grid.eta_loc[I2V(i,j,k)] += grid.Keta_descent_dir_loc[I2V(i,j,k)]*step_length_new;

                        grid.fac_b_loc[I2V(i,j,k)] = _1_CR - _2_CR * grid.xi_loc[I2V(i,j,k)];
                        grid.fac_c_loc[I2V(i,j,k)] = _1_CR + _2_CR * grid.xi_loc[I2V(i,j,k)];
                        grid.fac_f_loc[I2V(i,j,k)] =       - _2_CR * grid.eta_loc[I2V(i,j,k)];
                    }
                }
            }

        } // end LBFGS model update

        // shared values on the boundary
        grid.send_recev_boundary_data(grid.fun_loc);
        grid.send_recev_boundary_data(grid.xi_loc);
        grid.send_recev_boundary_data(grid.eta_loc);
        grid.send_recev_boundary_data(grid.fac_b_loc);
        grid.send_recev_boundary_data(grid.fac_c_loc);
        grid.send_recev_boundary_data(grid.fac_f_loc);

    } // end if subdom_main
}



// compute p_k * grad(f_k)
CUSTOMREAL compute_q_k(Grid& grid) {

    CUSTOMREAL tmp_qk = _0_CR;

    if (subdom_main && id_sim == 0) {

        // grad * descent_direction
        for (int k = 0; k < loc_K; k++) {
            for (int j = 0; j < loc_J; j++) {
                for (int i = 0; i < loc_I; i++) {
                    // we can add here a precondition (here use *_update_loc for storing descent direction)
                    tmp_qk += grid.Ks_update_loc[I2V(i,j,k)]   * grid.Ks_descent_dir_loc[I2V(i,j,k)]
                            + grid.Keta_update_loc[I2V(i,j,k)] * grid.Keta_descent_dir_loc[I2V(i,j,k)]
                            + grid.Kxi_update_loc[I2V(i,j,k)]  * grid.Kxi_descent_dir_loc[I2V(i,j,k)];
                }
            }
        }

        // add tmp_qk among all subdomain
        allreduce_cr_single(tmp_qk, tmp_qk);
    }

    // share tmp_qk among all simultaneous runs
    if (subdom_main)
        broadcast_cr_single_inter_sim(tmp_qk,0);

    return tmp_qk;

}


// check if the wolfe conditions are satisfied
bool check_wolfe_cond(Grid& grid, \
                    CUSTOMREAL q_0, CUSTOMREAL q_t, \
                    CUSTOMREAL qp_0, CUSTOMREAL qp_t, \
                    CUSTOMREAL& td, CUSTOMREAL& tg, CUSTOMREAL& step_length_sub) {
    /*
    q_0 : initial cost function
    q_t : current cost function
    q_p0 : initial grad * descent_dir
    q_pt : current grad * descent_dir
    td : right step size
    tg : left step size
    step_length_sub : current step size
    */

    bool step_accepted = false;

    // check  if the wolfe conditions are satisfied and update the step_length_sub
    CUSTOMREAL qpt = (q_t - q_0) / step_length_sub;

    if (subdom_main) {
        // good step size
        if (qpt   <= wolfe_c1*qp_0 \
         && wolfe_c2*qp_0 <= qp_t \
         ){
            if (myrank==0)
                std::cout << "Wolfe rules:  step accepted" << std::endl;
            step_accepted = true;
        } else {
            // modify the stepsize
            if (wolfe_c1*qp_0 < qpt) {
                td = step_length_sub;
                if (myrank==0)
                    std::cout << "Wolfe rules:  right step size updated." << std::endl;
            }
            if (qpt <= wolfe_c1*qp_0 && qp_t < wolfe_c2*qp_0) {
                tg = step_length_sub;
                if (myrank==0)
                    std::cout << "Wolfe rules:  left step size updated." << std::endl;
            }
            //if (isZero(td)) {
            if (td == _0_CR) {
                step_length_sub = _2_CR * step_length_sub;
                if (myrank==0)
                    std::cout << "Wolfe rules:  step size too small. Increased to " << step_length_sub << std::endl;
            } else {
                step_length_sub = _0_5_CR * (td+tg);
                if (myrank==0)
                    std::cout << "Wolfe rules:  step size too large. Decreased to " << step_length_sub << std::endl;
            }
        }

        // share step_accepted among all subdomains
        broadcast_bool_single(step_accepted,0);
        broadcast_cr_single(step_length_sub,0);
    } // end if subdom_main

    broadcast_bool_single_sub(step_accepted,0);
    broadcast_cr_single_sub(step_length_sub,0);

    return step_accepted;
}



#endif // MODEL_UPDATE_H