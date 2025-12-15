#ifndef SMOOTH_DESCENT_DIR_H
#define SMOOTH_DESCENT_DIR_H

#include <iostream>
#include "grid.h"
#include "config.h"
#include "smooth.h"


inline void smooth_descent_dir_CG(Grid& grid, const CUSTOMREAL lr, const CUSTOMREAL lt, const CUSTOMREAL lp) {
    // smoothing kernels with Conjugate Gradient method
    // lr,lt,lp: smoothing length in r, theta, phi direction

    // smooth Ks
    CG_smooth(grid, grid.Ks_descent_dir_loc, grid.Ks_descent_dir_loc, lr, lt, lp);
    // smooth Keta
    CG_smooth(grid, grid.Keta_descent_dir_loc, grid.Keta_descent_dir_loc, lr, lt, lp);
    // smooth Kxi
    CG_smooth(grid, grid.Kxi_descent_dir_loc, grid.Kxi_descent_dir_loc, lr, lt, lp);

}


// original method for smoothing kernels
inline void smooth_descent_dir(Grid& grid){
    // necessary params
    CUSTOMREAL r_r, r_t, r_p;
    int kdr, jdt, idp;

    int k_start = 0;
    int j_start = 0;
    int i_start = 0;
    int k_end   = ngrid_k;
    int j_end   = ngrid_j;
    int i_end   = ngrid_i;

    //
    // sum the kernel values on the inversion grid
    //

    for (int i_grid = 0; i_grid < n_inv_grids; i_grid++) {

        // init coarser kernel arrays
        for (int k = 0; k < n_inv_K_loc; k++) {
            for (int j = 0; j < n_inv_J_loc; j++) {
                for (int i = 0; i < n_inv_I_loc; i++) {
                    grid.Ks_inv_loc[  I2V_INV_KNL(i,j,k)] = _0_CR;
                    grid.Keta_inv_loc[I2V_INV_KNL(i,j,k)] = _0_CR;
                    grid.Kxi_inv_loc[ I2V_INV_KNL(i,j,k)] = _0_CR;
                }
            }
        }

        for (int k = k_start; k < k_end; k++) {
            // CUSTOMREAL r_glob = grid.get_r_min() + k*grid.get_delta_r(); // global coordinate of r
            // r_r = -_1_CR;
            // for (int ii_invr = 0; ii_invr < n_inv_K_loc-1; ii_invr++){
            //     if (r_glob > grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)] && r_glob <= grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)]) {
            //         kdr = ii_invr;
            //         r_r = (r_glob - grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)]) / (grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)] - grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)]);
            //         break;
            //     }
            // }
            CUSTOMREAL r_glob = grid.get_r_min() + k*grid.get_delta_r(); // global coordinate of r
            r_r = -_1_CR;
            for (int ii_invr = 0; ii_invr < n_inv_K_loc-1; ii_invr++){
                // increasing or decreasing order
                if (in_between(r_glob, grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)])) {
                    kdr = ii_invr;
                    r_r = calc_ratio_between(r_glob, grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)]);
                    break;
                }
            }
            // continue if r is out of the inversion grid
            if (r_r < 0) continue;

            for (int j = j_start; j < j_end; j++) {
                // CUSTOMREAL t_glob = grid.get_lat_min() + j*grid.get_delta_lat();
                // r_t = -_1_CR;
                // for (int ii_invt = 0; ii_invt < n_inv_J_loc-1; ii_invt++){
                //     if (t_glob > grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,i_grid)] && t_glob <= grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt+1,i_grid)]) {
                //         jdt = ii_invt;
                //         r_t = (t_glob - grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,i_grid)]) / (grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt+1,i_grid)] - grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,i_grid)]);
                //         break;
                //     }
                // }
                CUSTOMREAL t_glob = grid.get_lat_min() + j*grid.get_delta_lat(); // global coordinate of t (latitude)
                r_t = -_1_CR;
                for (int ii_invt = 0; ii_invt < n_inv_J_loc-1; ii_invt++){
                    CUSTOMREAL left  = grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,  kdr,i_grid)]*(1-r_r) + grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,  kdr+1,i_grid)]*(r_r);
                    CUSTOMREAL right = grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt+1,kdr,i_grid)]*(1-r_r) + grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt+1,kdr+1,i_grid)]*(r_r);
                    if (in_between(t_glob, left, right)) {
                        jdt = ii_invt;
                        r_t = calc_ratio_between(t_glob, left, right);
                        break;
                    }
                }

                // continue if t is out of the inversion grid
                if (r_t < 0) continue;

                for (int i = i_start; i < i_end; i++) {
                    // CUSTOMREAL p_glob = grid.get_lon_min() + i*grid.get_delta_lon();
                    // r_p = -_1_CR;
                    // for (int ii_invp = 0; ii_invp < n_inv_I_loc-1; ii_invp++){
                    //     if (p_glob > grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,i_grid)] && p_glob <= grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,i_grid)]) {
                    //         idp = ii_invp;
                    //         r_p = (p_glob - grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,i_grid)]) / (grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,i_grid)] - grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,i_grid)]);
                    //         break;
                    //     }
                    // }
                    CUSTOMREAL p_glob = grid.get_lon_min() + i*grid.get_delta_lon();    // global coordinate of p (longitude)
                    r_p = -_1_CR;
                    for (int ii_invp = 0; ii_invp < n_inv_I_loc-1; ii_invp++){
                        CUSTOMREAL left  = grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,  kdr,i_grid)]*(1-r_r) + grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,  kdr,i_grid)]*(r_r);
                        CUSTOMREAL right = grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,kdr,i_grid)]*(1-r_r) + grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,kdr,i_grid)]*(r_r);
                        if (in_between(p_glob, left, right)) {
                            idp = ii_invp;
                            r_p = calc_ratio_between(p_glob, left, right);
                            break;
                        }
                    }

                    // continue if p is out of the inversion grid
                    if (r_p < 0) continue;

                    // global grid id to local id
                    int k_loc = k - grid.get_offset_k();
                    int j_loc = j - grid.get_offset_j();
                    int i_loc = i - grid.get_offset_i();

                    // check if *_loc are inside the local subdomain
                    if (k_loc < 0 || k_loc > loc_K-1) continue;
                    if (j_loc < 0 || j_loc > loc_J-1) continue;
                    if (i_loc < 0 || i_loc > loc_I-1) continue;

                    // update Ks_inv Keta_inv Kxi_inv
                    grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt,kdr)] += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)];
                    grid.Keta_inv_loc[I2V_INV_KNL(idp,jdt,kdr)] += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)];
                    grid.Kxi_inv_loc[ I2V_INV_KNL(idp,jdt,kdr)] += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt,kdr+1)] += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)];
                    grid.Keta_inv_loc[I2V_INV_KNL(idp,jdt,kdr+1)] += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)];
                    grid.Kxi_inv_loc[ I2V_INV_KNL(idp,jdt,kdr+1)] += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr)] += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)];
                    grid.Keta_inv_loc[I2V_INV_KNL(idp,jdt+1,kdr)] += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)];
                    grid.Kxi_inv_loc[ I2V_INV_KNL(idp,jdt+1,kdr)] += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr)] += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)];
                    grid.Keta_inv_loc[I2V_INV_KNL(idp+1,jdt,kdr)] += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)];
                    grid.Kxi_inv_loc[ I2V_INV_KNL(idp+1,jdt,kdr)] += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr+1)] += r_r*r_t*(_1_CR-r_p)*grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)];
                    grid.Keta_inv_loc[I2V_INV_KNL(idp,jdt+1,kdr+1)] += r_r*r_t*(_1_CR-r_p)*grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)];
                    grid.Kxi_inv_loc[ I2V_INV_KNL(idp,jdt+1,kdr+1)] += r_r*r_t*(_1_CR-r_p)*grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr+1)] += r_r*(_1_CR-r_t)*r_p*grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)];
                    grid.Keta_inv_loc[I2V_INV_KNL(idp+1,jdt,kdr+1)] += r_r*(_1_CR-r_t)*r_p*grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)];
                    grid.Kxi_inv_loc[ I2V_INV_KNL(idp+1,jdt,kdr+1)] += r_r*(_1_CR-r_t)*r_p*grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr)] += (_1_CR-r_r)*r_t*r_p*grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)];
                    grid.Keta_inv_loc[I2V_INV_KNL(idp+1,jdt+1,kdr)] += (_1_CR-r_r)*r_t*r_p*grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)];
                    grid.Kxi_inv_loc[ I2V_INV_KNL(idp+1,jdt+1,kdr)] += (_1_CR-r_r)*r_t*r_p*grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr+1)] += r_r*r_t*r_p*grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)];
                    grid.Keta_inv_loc[I2V_INV_KNL(idp+1,jdt+1,kdr+1)] += r_r*r_t*r_p*grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)];
                    grid.Kxi_inv_loc[ I2V_INV_KNL(idp+1,jdt+1,kdr+1)] += r_r*r_t*r_p*grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)];

                } // end for i
            } // end for j
        } // end for k

        // sum over all sub-domains
        allreduce_cr_inplace(grid.Ks_inv_loc,   n_inv_I_loc*n_inv_J_loc*n_inv_K_loc);
        allreduce_cr_inplace(grid.Keta_inv_loc, n_inv_I_loc*n_inv_J_loc*n_inv_K_loc);
        allreduce_cr_inplace(grid.Kxi_inv_loc,  n_inv_I_loc*n_inv_J_loc*n_inv_K_loc);

        //
        // update factors
        //
        for (int k = k_start; k < k_end; k++) {
            // CUSTOMREAL r_glob = grid.get_r_min() + k*grid.get_delta_r(); // global coordinate of r
            // r_r = -_1_CR;
            // for (int ii_invr = 0; ii_invr < n_inv_K_loc-1; ii_invr++){
            //     if (r_glob > grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)] && r_glob <= grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)]) {
            //         kdr = ii_invr;
            //         r_r = (r_glob - grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)]) / (grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)] - grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)]);
            //         break;
            //     }
            // }
            CUSTOMREAL r_glob = grid.get_r_min() + k*grid.get_delta_r(); // global coordinate of r
            r_r = -_1_CR;
            for (int ii_invr = 0; ii_invr < n_inv_K_loc-1; ii_invr++){
                if (in_between(r_glob, grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)])) {
                    kdr = ii_invr;
                    r_r = calc_ratio_between(r_glob, grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)]);
                    break;
                }
            }

            // continue if r is out of the inversion grid
            if (r_r < 0) continue;

            for (int j = j_start; j < j_end; j++) {
                // CUSTOMREAL t_glob = grid.get_lat_min() + j*grid.get_delta_lat();
                // r_t = -_1_CR;
                // for (int ii_invt = 0; ii_invt < n_inv_J_loc-1; ii_invt++){
                //     if (t_glob > grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,i_grid)] && t_glob <= grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt+1,i_grid)]) {
                //         jdt = ii_invt;
                //         r_t = (t_glob - grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,i_grid)]) / (grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt+1,i_grid)] - grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,i_grid)]);
                //         break;
                //     }
                // }
                CUSTOMREAL t_glob = grid.get_lat_min() + j*grid.get_delta_lat();
                r_t = -_1_CR;
                for (int ii_invt = 0; ii_invt < n_inv_J_loc-1; ii_invt++){
                    CUSTOMREAL left  = grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,  kdr,i_grid)]*(1-r_r) + grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt,  kdr+1,i_grid)]*(r_r);
                    CUSTOMREAL right = grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt+1,kdr,i_grid)]*(1-r_r) + grid.inv_grid->t.arr[I2V_INV_GRIDS_2DJ(ii_invt+1,kdr+1,i_grid)]*(r_r);
                    if (in_between(t_glob, left, right)) {
                        jdt = ii_invt;
                        r_t = calc_ratio_between(t_glob, left, right);
                        break;
                    }
                }


                // continue if t is out of the inversion grid
                if (r_t < 0) continue;

                for (int i = i_start; i < i_end; i++) {
                    // CUSTOMREAL p_glob = grid.get_lon_min() + i*grid.get_delta_lon();
                    // r_p = -_1_CR;
                    // for (int ii_invp = 0; ii_invp < n_inv_I_loc-1; ii_invp++){
                    //     if (p_glob > grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,i_grid)] && p_glob <= grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,i_grid)]) {
                    //         idp = ii_invp;
                    //         r_p = (p_glob - grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,i_grid)]) / (grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,i_grid)] - grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,i_grid)]);
                    //         break;
                    //     }
                    // }
                    CUSTOMREAL p_glob = grid.get_lon_min() + i*grid.get_delta_lon();
                    r_p = -_1_CR;
                    for (int ii_invp = 0; ii_invp < n_inv_I_loc-1; ii_invp++){
                        CUSTOMREAL left  = grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,  kdr,i_grid)]*(1-r_r) + grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,  kdr,i_grid)]*(r_r);
                        CUSTOMREAL right = grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,kdr,i_grid)]*(1-r_r) + grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,kdr,i_grid)]*(r_r);
                        if (in_between(p_glob, left, right)) {
                            idp = ii_invp;
                            r_p = calc_ratio_between(p_glob, left, right);
                            break;
                        }
                    }

                    // continue if p is out of the inversion grid
                    if (r_p < 0) continue;

                    // global grid id to local id
                    int k_loc = k - grid.get_offset_k();
                    int j_loc = j - grid.get_offset_j();
                    int i_loc = i - grid.get_offset_i();

                    // check if *_loc are inside the local subdomain
                    if (k_loc < 0 || k_loc > loc_K-1) continue;
                    if (j_loc < 0 || j_loc > loc_J-1) continue;
                    if (i_loc < 0 || i_loc > loc_I-1) continue;

                    CUSTOMREAL pert_Ks = 0.0;
                    CUSTOMREAL pert_Keta = 0.0;
                    CUSTOMREAL pert_Kxi = 0.0;

                    pert_Ks   += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt,kdr)];
                    pert_Keta += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Keta_inv_loc[I2V_INV_KNL(idp,jdt,kdr)];
                    pert_Kxi  += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Kxi_inv_loc[ I2V_INV_KNL(idp,jdt,kdr)];

                    pert_Ks    += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt,kdr+1)];
                    pert_Keta  += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Keta_inv_loc[I2V_INV_KNL(idp,jdt,kdr+1)];
                    pert_Kxi   += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Kxi_inv_loc[ I2V_INV_KNL(idp,jdt,kdr+1)];

                    pert_Ks    += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr)];
                    pert_Keta  += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Keta_inv_loc[I2V_INV_KNL(idp,jdt+1,kdr)];
                    pert_Kxi   += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Kxi_inv_loc[ I2V_INV_KNL(idp,jdt+1,kdr)];

                    pert_Ks    += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr)];
                    pert_Keta  += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Keta_inv_loc[I2V_INV_KNL(idp+1,jdt,kdr)];
                    pert_Kxi   += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Kxi_inv_loc[ I2V_INV_KNL(idp+1,jdt,kdr)];

                    pert_Ks    += r_r*r_t*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr+1)];
                    pert_Keta  += r_r*r_t*(_1_CR-r_p)*grid.Keta_inv_loc[I2V_INV_KNL(idp,jdt+1,kdr+1)];
                    pert_Kxi   += r_r*r_t*(_1_CR-r_p)*grid.Kxi_inv_loc[ I2V_INV_KNL(idp,jdt+1,kdr+1)];

                    pert_Ks    += r_r*(_1_CR-r_t)*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr+1)];
                    pert_Keta  += r_r*(_1_CR-r_t)*r_p*grid.Keta_inv_loc[I2V_INV_KNL(idp+1,jdt,kdr+1)];
                    pert_Kxi   += r_r*(_1_CR-r_t)*r_p*grid.Kxi_inv_loc[ I2V_INV_KNL(idp+1,jdt,kdr+1)];

                    pert_Ks    += (_1_CR-r_r)*r_t*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr)];
                    pert_Keta  += (_1_CR-r_r)*r_t*r_p*grid.Keta_inv_loc[I2V_INV_KNL(idp+1,jdt+1,kdr)];
                    pert_Kxi   += (_1_CR-r_r)*r_t*r_p*grid.Kxi_inv_loc[ I2V_INV_KNL(idp+1,jdt+1,kdr)];

                    pert_Ks    += r_r*r_t*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr+1)];
                    pert_Keta  += r_r*r_t*r_p*grid.Keta_inv_loc[I2V_INV_KNL(idp+1,jdt+1,kdr+1)];
                    pert_Kxi   += r_r*r_t*r_p*grid.Kxi_inv_loc[ I2V_INV_KNL(idp+1,jdt+1,kdr+1)];


                    // update para
                    grid.Ks_descent_dir_loc[  I2V(i_loc,j_loc,k_loc)] += pert_Ks;
                    grid.Keta_descent_dir_loc[I2V(i_loc,j_loc,k_loc)] += pert_Keta;
                    grid.Kxi_descent_dir_loc[ I2V(i_loc,j_loc,k_loc)] += pert_Kxi;


                } // end for i
            } // end for j
        } // end for k

    } // end i_grid

}



#endif