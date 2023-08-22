#ifndef LBFGS_H
#define LBFGS_H

#include "config.h"
#include "utils.h"
#include "grid.h"
#include "mpi_funcs.h"

inline CUSTOMREAL volume_domain = _0_CR;          // volume of the domain
inline CUSTOMREAL weight_Tikonov = _0_CR;         // weight of the regularization term
inline int        N_params = 3;                   // number of parameters to invert
inline CUSTOMREAL q_0           = _0_CR;          // initial cost function
inline CUSTOMREAL qp_0          = _0_CR;          // store initial p_k * grad(f_k) (descent_direction * gradient)


void store_model_and_gradient(Grid& grid, int i_inv) {

    if (subdom_main && id_sim==0) {

        if (i_inv < Mbfgs){
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        grid.Ks_model_store_loc[  I2VLBFGS(i_inv,i,j,k)] = grid.fun_loc[I2V(i,j,k)];
                        grid.Keta_model_store_loc[I2VLBFGS(i_inv,i,j,k)] = grid.eta_loc[I2V(i,j,k)];
                        grid.Kxi_model_store_loc[ I2VLBFGS(i_inv,i,j,k)] = grid.xi_loc[ I2V(i,j,k)];
                        grid.Ks_grad_store_loc[   I2VLBFGS(i_inv,i,j,k)] = grid.Ks_update_loc[  I2V(i,j,k)];
                        grid.Keta_grad_store_loc[ I2VLBFGS(i_inv,i,j,k)] = grid.Keta_update_loc[I2V(i,j,k)];
                        grid.Kxi_grad_store_loc[  I2VLBFGS(i_inv,i,j,k)] = grid.Kxi_update_loc[ I2V(i,j,k)];
                    }
                }
            }

        } else {
            // replace the current stored models and gradients
            for (int istore = 0; istore < Mbfgs-1; istore++){
                for (int k = 0; k < loc_K; k++) {
                    for (int j = 0; j < loc_J; j++) {
                        for (int i = 0; i < loc_I; i++) {
                            grid.Ks_model_store_loc[  I2VLBFGS(istore,i,j,k)] = grid.Ks_model_store_loc[  I2VLBFGS(istore+1,i,j,k)];
                            grid.Keta_model_store_loc[I2VLBFGS(istore,i,j,k)] = grid.Keta_model_store_loc[I2VLBFGS(istore+1,i,j,k)];
                            grid.Kxi_model_store_loc[ I2VLBFGS(istore,i,j,k)] = grid.Kxi_model_store_loc[ I2VLBFGS(istore+1,i,j,k)];
                            grid.Ks_grad_store_loc[   I2VLBFGS(istore,i,j,k)] = grid.Ks_grad_store_loc[   I2VLBFGS(istore+1,i,j,k)];
                            grid.Keta_grad_store_loc[ I2VLBFGS(istore,i,j,k)] = grid.Keta_grad_store_loc[ I2VLBFGS(istore+1,i,j,k)];
                            grid.Kxi_grad_store_loc[  I2VLBFGS(istore,i,j,k)] = grid.Kxi_grad_store_loc[  I2VLBFGS(istore+1,i,j,k)];
                        }
                    }
                }
            }

            // store new model and gradient
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        grid.Ks_model_store_loc[  I2VLBFGS(Mbfgs-1,i,j,k)] = grid.fun_loc[I2V(i,j,k)];
                        grid.Keta_model_store_loc[I2VLBFGS(Mbfgs-1,i,j,k)] = grid.eta_loc[I2V(i,j,k)];
                        grid.Kxi_model_store_loc[ I2VLBFGS(Mbfgs-1,i,j,k)] = grid.xi_loc[ I2V(i,j,k)];
                        grid.Ks_grad_store_loc[   I2VLBFGS(Mbfgs-1,i,j,k)] = grid.Ks_update_loc[  I2V(i,j,k)];
                        grid.Keta_grad_store_loc[ I2VLBFGS(Mbfgs-1,i,j,k)] = grid.Keta_update_loc[I2V(i,j,k)];
                        grid.Kxi_grad_store_loc[  I2VLBFGS(Mbfgs-1,i,j,k)] = grid.Kxi_update_loc[ I2V(i,j,k)];
                    }
                }
            }
        }

   } // end subdom main and id_sim==0

}


void calculate_descent_direction_lbfgs(Grid& grid, int i_inv) {

    if (subdom_main && id_sim==0) {
        int imin = 0;
        int imax = 0;
        if (i_inv >= Mbfgs)
            imax = Mbfgs-2;
        else
            imax = i_inv-1;

        CUSTOMREAL* ak_store = new CUSTOMREAL[imax-imin+1];
        CUSTOMREAL* pk_store = new CUSTOMREAL[imax-imin+1];

        CUSTOMREAL* desc_wks_Ks   = new CUSTOMREAL[loc_I*loc_J*loc_K];
        CUSTOMREAL* desc_wks_Keta = new CUSTOMREAL[loc_I*loc_J*loc_K];
        CUSTOMREAL* desc_wks_Kxi  = new CUSTOMREAL[loc_I*loc_J*loc_K];

        CUSTOMREAL* wks_1_Ks   = new CUSTOMREAL[loc_I*loc_J*loc_K];
        CUSTOMREAL* wks_1_Keta = new CUSTOMREAL[loc_I*loc_J*loc_K];
        CUSTOMREAL* wks_1_Kxi  = new CUSTOMREAL[loc_I*loc_J*loc_K];
        CUSTOMREAL* wks_2_Ks   = new CUSTOMREAL[loc_I*loc_J*loc_K];
        CUSTOMREAL* wks_2_Keta = new CUSTOMREAL[loc_I*loc_J*loc_K];
        CUSTOMREAL* wks_2_Kxi  = new CUSTOMREAL[loc_I*loc_J*loc_K];

        std::cout << "imax, imin: " << imax << ", " << imin << std::endl;

        // initialize
        for (int i = 0; i < imax-imin+1; i++) {
            ak_store[i] = 0.0; // alpha
            pk_store[i] = 0.0; //rho
        }

        for (int k = 0; k < loc_K; k++) {
            for (int j = 0; j < loc_J; j++) {
                for (int i = 0; i < loc_I; i++) {
                    desc_wks_Ks[I2V(i,j,k)]   = grid.Ks_grad_store_loc[  I2VLBFGS(imax+1,i,j,k)]; // d
                    desc_wks_Keta[I2V(i,j,k)] = grid.Keta_grad_store_loc[I2VLBFGS(imax+1,i,j,k)]; // d
                    desc_wks_Kxi[I2V(i,j,k)]  = grid.Kxi_grad_store_loc[ I2VLBFGS(imax+1,i,j,k)]; // d
                    wks_1_Ks[I2V(i,j,k)]   = grid.Ks_grad_store_loc[  I2VLBFGS(imax+1,i,j,k)] - grid.Ks_grad_store_loc[  I2VLBFGS(imax,i,j,k)];
                    wks_1_Keta[I2V(i,j,k)] = grid.Keta_grad_store_loc[I2VLBFGS(imax+1,i,j,k)] - grid.Keta_grad_store_loc[I2VLBFGS(imax,i,j,k)];
                    wks_1_Kxi[I2V(i,j,k)]  = grid.Kxi_grad_store_loc[ I2VLBFGS(imax+1,i,j,k)] - grid.Kxi_grad_store_loc[ I2VLBFGS(imax,i,j,k)];
                }
            }
        }

        //calculate l2 norms
        CUSTOMREAL norm_yiter = 0.0;

        norm_yiter += calc_l2norm(wks_1_Ks,loc_I*loc_J*loc_K);
        norm_yiter += calc_l2norm(wks_1_Keta,loc_I*loc_J*loc_K);
        norm_yiter += calc_l2norm(wks_1_Kxi,loc_I*loc_J*loc_K);

        CUSTOMREAL tmp = norm_yiter;
        allreduce_cr_single(tmp, norm_yiter);

        for (int iinv = imax; iinv >= imin; iinv--){

            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        wks_1_Ks[I2V(i,j,k)]   = grid.Ks_grad_store_loc[  I2VLBFGS(iinv+1,i,j,k)]   - grid.Ks_grad_store_loc[I2VLBFGS(iinv,i,j,k)];
                        wks_1_Keta[I2V(i,j,k)] = grid.Keta_grad_store_loc[I2VLBFGS(iinv+1,i,j,k)] - grid.Keta_grad_store_loc[I2VLBFGS(iinv,i,j,k)];
                        wks_1_Kxi[I2V(i,j,k)]  = grid.Kxi_grad_store_loc[ I2VLBFGS(iinv+1,i,j,k)]  - grid.Kxi_grad_store_loc[I2VLBFGS(iinv,i,j,k)];

                        wks_2_Ks[I2V(i,j,k)]   = grid.Ks_model_store_loc[  I2VLBFGS(iinv+1,i,j,k)]   - grid.Ks_model_store_loc[I2VLBFGS(iinv,i,j,k)];
                        wks_2_Keta[I2V(i,j,k)] = grid.Keta_model_store_loc[I2VLBFGS(iinv+1,i,j,k)] - grid.Keta_model_store_loc[I2VLBFGS(iinv,i,j,k)];
                        wks_2_Kxi[I2V(i,j,k)]  = grid.Kxi_model_store_loc[ I2VLBFGS(iinv+1,i,j,k)]  - grid.Kxi_model_store_loc[I2VLBFGS(iinv,i,j,k)];
                    }
                }
            }

            CUSTOMREAL pk = _0_CR;
            pk += dot_product(wks_1_Ks,wks_2_Ks,loc_I*loc_J*loc_K);
            pk += dot_product(wks_1_Keta,wks_2_Keta,loc_I*loc_J*loc_K);
            pk += dot_product(wks_1_Kxi,wks_2_Kxi,loc_I*loc_J*loc_K);
            CUSTOMREAL tmp = pk;
            allreduce_cr_single(tmp, pk);
            pk_store[iinv] = _1_CR / pk;

            CUSTOMREAL ak = _0_CR;
            ak += dot_product(wks_2_Ks, desc_wks_Ks,loc_I*loc_J*loc_K);
            ak += dot_product(wks_2_Keta, desc_wks_Keta,loc_I*loc_J*loc_K);
            ak += dot_product(wks_2_Kxi, desc_wks_Kxi,loc_I*loc_J*loc_K);
            // print ak
            if(myrank== 0) std::cout << "ak: " << ak << std::endl;
            tmp = ak;
            allreduce_cr_single(tmp, ak);
            // print ak
            if(myrank== 0) std::cout << "ak gathered: " << ak << std::endl;
            ak_store[iinv] = pk_store[iinv] * ak;

            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        desc_wks_Ks[I2V(i,j,k)]   -= ak_store[iinv] * wks_1_Ks[I2V(i,j,k)];
                        desc_wks_Keta[I2V(i,j,k)] -= ak_store[iinv] * wks_1_Keta[I2V(i,j,k)];
                        desc_wks_Kxi[I2V(i,j,k)]  -= ak_store[iinv] * wks_1_Kxi[I2V(i,j,k)];
                    }
                }
            }

        } // end loop iinv

        // Nocedal's default preconditionning
        CUSTOMREAL pk = _1_CR / (pk_store[imax] * norm_yiter);
        for (int k = 0; k < loc_K; k++) {
            for (int j = 0; j < loc_J; j++) {
                for (int i = 0; i < loc_I; i++) {
                    desc_wks_Ks[I2V(i,j,k)]   *= pk;
                    desc_wks_Keta[I2V(i,j,k)] *= pk;
                    desc_wks_Kxi[I2V(i,j,k)]  *= pk;
                }
            }
        }

        // print ak_store and pk_store
        if (myrank == 0) {
            std::cout << "ak_store: ";
            for (int i = 0; i < imax+1; i++) {
                std::cout << ak_store[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "pk_store: ";
            for (int i = 0; i < imax+1; i++) {
                std::cout << pk_store[i] << " ";
            }
            std::cout << std::endl;
        }


        // custom preconditionning
        // diagonal preconditionner

        for (int iinv = imin; iinv <= imax; iinv++){

            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        wks_1_Ks[I2V(i,j,k)]   = grid.Ks_grad_store_loc[  I2VLBFGS(iinv+1,i,j,k)]   - grid.Ks_grad_store_loc[I2VLBFGS(iinv,i,j,k)];
                        wks_1_Keta[I2V(i,j,k)] = grid.Keta_grad_store_loc[I2VLBFGS(iinv+1,i,j,k)] - grid.Keta_grad_store_loc[I2VLBFGS(iinv,i,j,k)];
                        wks_1_Kxi[I2V(i,j,k)]  = grid.Kxi_grad_store_loc[ I2VLBFGS(iinv+1,i,j,k)]  - grid.Kxi_grad_store_loc[I2VLBFGS(iinv,i,j,k)];

                        wks_2_Ks[I2V(i,j,k)]   = grid.Ks_model_store_loc[  I2VLBFGS(iinv+1,i,j,k)]   - grid.Ks_model_store_loc[I2VLBFGS(iinv,i,j,k)];
                        wks_2_Keta[I2V(i,j,k)] = grid.Keta_model_store_loc[I2VLBFGS(iinv+1,i,j,k)] - grid.Keta_model_store_loc[I2VLBFGS(iinv,i,j,k)];
                        wks_2_Kxi[I2V(i,j,k)]  = grid.Kxi_model_store_loc[ I2VLBFGS(iinv+1,i,j,k)]  - grid.Kxi_model_store_loc[I2VLBFGS(iinv,i,j,k)];
                    }
                }
            }

            CUSTOMREAL beta = _0_CR;
            beta += dot_product(wks_1_Ks,desc_wks_Ks,loc_I*loc_J*loc_K);
            beta += dot_product(wks_1_Keta,desc_wks_Keta,loc_I*loc_J*loc_K);
            beta += dot_product(wks_1_Kxi,desc_wks_Kxi,loc_I*loc_J*loc_K);
            CUSTOMREAL tmp = beta;
            allreduce_cr_single(tmp, beta);
            beta = pk_store[iinv] * beta;

            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        desc_wks_Ks[I2V(i,j,k)]   += (ak_store[iinv]-beta) * wks_2_Ks[I2V(i,j,k)];
                        desc_wks_Keta[I2V(i,j,k)] += (ak_store[iinv]-beta) * wks_2_Keta[I2V(i,j,k)];
                        desc_wks_Kxi[I2V(i,j,k)]  += (ak_store[iinv]-beta) * wks_2_Kxi[I2V(i,j,k)];
                    }
                }
            }
        }

        // descent directions
        for (int k = 0; k < loc_K; k++) {
            for (int j = 0; j < loc_J; j++) {
                for (int i = 0; i < loc_I; i++) {
                    grid.Ks_descent_dir_loc[I2V(i,j,k)]   = - _1_CR * desc_wks_Ks[I2V(i,j,k)];
                    grid.Keta_descent_dir_loc[I2V(i,j,k)] = - _1_CR * desc_wks_Keta[I2V(i,j,k)];
                    grid.Kxi_descent_dir_loc[I2V(i,j,k)]  = - _1_CR * desc_wks_Kxi[I2V(i,j,k)];
                    //grid.Ks_descent_dir_loc[I2V(i,j,k)]   = desc_wks_Ks[I2V(i,j,k)];
                    //grid.Keta_descent_dir_loc[I2V(i,j,k)] = desc_wks_Keta[I2V(i,j,k)];
                    //grid.Kxi_descent_dir_loc[I2V(i,j,k)]  = desc_wks_Kxi[I2V(i,j,k)];

                }
            }
        }


        delete [] ak_store;
        delete [] pk_store;
        delete [] desc_wks_Ks;
        delete [] desc_wks_Keta;
        delete [] desc_wks_Kxi;
        delete [] wks_1_Ks;
        delete [] wks_1_Keta;
        delete [] wks_1_Kxi;
        delete [] wks_2_Ks;
        delete [] wks_2_Keta;
        delete [] wks_2_Kxi;


    } // end of subdom_main

    // share with all simultaneous run
    if (subdom_main){
        broadcast_cr_inter_sim(grid.Ks_descent_dir_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_descent_dir_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Kxi_descent_dir_loc, loc_I*loc_J*loc_K, 0);
    }

}


inline void calc_laplacian_field(Grid& grid, CUSTOMREAL* arr_in, CUSTOMREAL* arr_res) {
    if (subdom_main) {

        //CUSTOMREAL lr = 1.0;
        //CUSTOMREAL lt = 1.0;
        //CUSTOMREAL lp = 1.0;

        CUSTOMREAL lr = smooth_lr;
        CUSTOMREAL lt = smooth_lt;
        CUSTOMREAL lp = smooth_lp;


        // calculate L(m)
        for (int k = 1; k < loc_K-1; k++) {
            for (int j = 1; j < loc_J-1; j++) {
                for (int i = 1; i < loc_I-1; i++) {
                    arr_res[I2V(i,j,k)] = \
                     - _2_CR * (lr+lt+lp) * arr_in[I2V(i,j,k)] \
                        + lr * (arr_in[I2V(i,j,k+1)] + arr_in[I2V(i,j,k-1)]) \
                        + lt * (arr_in[I2V(i,j+1,k)] + arr_in[I2V(i,j-1,k)]) \
                        + lp * (arr_in[I2V(i+1,j,k)] + arr_in[I2V(i-1,j,k)]);
                }
            }
        }

        // communicate with adjacent subdomains
        grid.send_recev_boundary_data(arr_res);
    }
}

// add 1/2 * L(m)^2 to the objective function
inline CUSTOMREAL calculate_regularization_obj(Grid& grid) {

    CUSTOMREAL regularization_term = _0_CR;

    if (subdom_main && id_sim==0) {
        int ngrid = loc_I*loc_J*loc_K;

        regularization_term += calc_l2norm(grid.fun_regularization_penalty_loc, ngrid);
        regularization_term += calc_l2norm(grid.eta_regularization_penalty_loc, ngrid);
        regularization_term += calc_l2norm(grid.xi_regularization_penalty_loc, ngrid);

        // gather from all subdomain
        CUSTOMREAL tmp = regularization_term;
        allreduce_cr_single(tmp, regularization_term);
    }

    // share retularization_term with all simultaneous run (may be unnecessary)
    broadcast_cr_single_inter_sim(regularization_term, 0);

    return _0_5_CR * weight_Tikonov * regularization_term;

}


inline void init_regularization_penalty(Grid& grid) {
    if (subdom_main && id_sim==0) {
        int ngrid = loc_I*loc_J*loc_K;

        // initialize regularization penalties
        for (int i = 0; i < ngrid; i++) {
            grid.fun_regularization_penalty_loc[i]          = _0_CR;
            grid.eta_regularization_penalty_loc[i]          = _0_CR;
            grid.xi_regularization_penalty_loc[i]           = _0_CR;
            grid.fun_gradient_regularization_penalty_loc[i] = _0_CR;
            grid.eta_gradient_regularization_penalty_loc[i] = _0_CR;
            grid.xi_gradient_regularization_penalty_loc[i]  = _0_CR;
        }
    }
}

inline void init_regulaization_penalty_with_one(Grid& grid) {
    if (subdom_main && id_sim==0) {
        int ngrid = loc_I*loc_J*loc_K;

        // initialize regularization penalties
        for (int i = 0; i < ngrid; i++) {
            grid.fun_regularization_penalty_loc[i]          = _1_CR;
            grid.eta_regularization_penalty_loc[i]          = _1_CR;
            grid.xi_regularization_penalty_loc[i]           = _1_CR;
            grid.fun_gradient_regularization_penalty_loc[i] = _1_CR;
            grid.eta_gradient_regularization_penalty_loc[i] = _1_CR;
            grid.xi_gradient_regularization_penalty_loc[i]  = _1_CR;
        }
    }
}


inline void calculate_regularization_penalty(Grid& grid) {
    if (subdom_main && id_sim==0) {
        init_regularization_penalty(grid);

        // calculate LL(m) on fun (Ks)
        calc_laplacian_field(grid, grid.fun_loc, grid.fun_regularization_penalty_loc);
        calc_laplacian_field(grid, grid.fun_regularization_penalty_loc, grid.fun_gradient_regularization_penalty_loc);

        // calculate LL(m) on eta (Keta)
        calc_laplacian_field(grid, grid.eta_loc, grid.eta_regularization_penalty_loc);
        calc_laplacian_field(grid, grid.eta_regularization_penalty_loc, grid.eta_gradient_regularization_penalty_loc);

        // calculate LL(m) on xi (Kxi)
        calc_laplacian_field(grid, grid.xi_loc, grid.xi_regularization_penalty_loc);
        calc_laplacian_field(grid, grid.xi_regularization_penalty_loc, grid.xi_gradient_regularization_penalty_loc);

        //
        int n_grid = loc_I*loc_J*loc_K;

        for (int i = 0; i < n_grid; i++){
            grid.fun_regularization_penalty_loc[i] += grid.fun_loc[i] - grid.fun_regularization_penalty_loc[i];
            grid.eta_regularization_penalty_loc[i] += grid.eta_loc[i] - grid.eta_regularization_penalty_loc[i];
            grid.xi_regularization_penalty_loc[i]  += grid.xi_loc[i]  - grid.xi_regularization_penalty_loc[i];

            grid.fun_gradient_regularization_penalty_loc[i] += grid.fun_loc[i] - _2_CR * grid.fun_regularization_penalty_loc[i] + grid.fun_gradient_regularization_penalty_loc[i];
            grid.eta_gradient_regularization_penalty_loc[i] += grid.eta_loc[i] - _2_CR * grid.eta_regularization_penalty_loc[i] + grid.eta_gradient_regularization_penalty_loc[i];
            grid.xi_gradient_regularization_penalty_loc[i]  += grid.xi_loc[i]  - _2_CR * grid.xi_regularization_penalty_loc[i]  + grid.xi_gradient_regularization_penalty_loc[i];
        }
    }
}


// add grad(L(m)) to gradient
inline void add_regularization_grad(Grid& grid) {
    if (subdom_main){
        if(id_sim==0){
            // add LL(m) to gradient
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {

                        //grid.fun_gradient_regularization_penalty_loc[I2V(i,j,k)]   /= Linf_Ks;
                        //grid.eta_gradient_regularization_penalty_loc[I2V(i,j,k)] /= Linf_Keta;
                        //grid.xi_gradient_regularization_penalty_loc[I2V(i,j,k)]  /= Linf_Kxi;

                        grid.Ks_update_loc[I2V(i,j,k)]   += weight_Tikonov * grid.fun_gradient_regularization_penalty_loc[I2V(i,j,k)];
                        grid.Keta_update_loc[I2V(i,j,k)] += weight_Tikonov * grid.eta_gradient_regularization_penalty_loc[I2V(i,j,k)];
                        grid.Kxi_update_loc[I2V(i,j,k)]  += weight_Tikonov * grid.xi_gradient_regularization_penalty_loc[I2V(i,j,k)];
                    }
                }
            }
        }// end id_sim==0

        // share with all simultaneous run (may be unnecessary)
        broadcast_cr_inter_sim(grid.Ks_update_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_update_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Kxi_update_loc, loc_I*loc_J*loc_K, 0);

    }// end subdom_main
}


// compute initial guess for step length to try for line search
void initial_guess_step(Grid& grid, CUSTOMREAL& step_length, CUSTOMREAL SC_VAL) {
    // find the max value in descent_direction
    CUSTOMREAL max_val = _0_CR;

    if(subdom_main && id_sim==0) {
        for (int k = 0; k < loc_K; k++) {
            for (int j = 0; j < loc_J; j++) {
                for (int i = 0; i < loc_I; i++) {
                    if (std::abs(grid.Ks_descent_dir_loc[I2V(i,j,k)]  ) > max_val) max_val = std::abs(grid.Ks_descent_dir_loc[I2V(i,j,k)]);
                    if (std::abs(grid.Keta_descent_dir_loc[I2V(i,j,k)]) > max_val) max_val = std::abs(grid.Keta_descent_dir_loc[I2V(i,j,k)]);
                    if (std::abs(grid.Kxi_descent_dir_loc[I2V(i,j,k)] ) > max_val) max_val = std::abs(grid.Kxi_descent_dir_loc[I2V(i,j,k)]);
                }
            }
        }

        // reduce
        allreduce_cr_single_max(max_val, max_val);

        // debug print max_val
        std::cout << "max_val: " << max_val << std::endl;

        // step size
        step_length = SC_VAL / max_val;
    }

    // broadcast to all simultaneous run (may be unnecessary)
    if (subdom_main)
        broadcast_cr_single_inter_sim(step_length, 0);

}


inline CUSTOMREAL compute_volume_domain(Grid& grid) {
    CUSTOMREAL volume = _0_CR;

    if (subdom_main && id_sim==0) {
        volume += calc_l2norm(grid.fun_regularization_penalty_loc, loc_I*loc_J*loc_K);
        volume += calc_l2norm(grid.eta_regularization_penalty_loc, loc_I*loc_J*loc_K);
        volume += calc_l2norm(grid.xi_regularization_penalty_loc, loc_I*loc_J*loc_K);

        // reduce
        allreduce_cr_single(volume, volume);
    }

    // broadcast to all simultaneous run (may be unnecessary)
    broadcast_cr_single_inter_sim(volume, 0);

    return volume;
}


#endif