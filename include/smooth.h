#ifndef SMOOTH_H
#define SMOOTH_H

#include <iostream>
#include "grid.h"
#include "config.h"
#include "utils.h"

void calc_inversed_laplacian(CUSTOMREAL* d, CUSTOMREAL* Ap,
                             const int i, const int j, const int k,
                             const CUSTOMREAL lr, const CUSTOMREAL lt, const CUSTOMREAL lp,
                             const CUSTOMREAL dr, const CUSTOMREAL dt, const CUSTOMREAL dp) {
    // calculate inversed laplacian operator
    CUSTOMREAL termx = _0_CR, termy = _0_CR, termz = _0_CR;

    if (i==0) {
        termx = dp*lp/3.0 * (1/(lp*dp)*(d[I2V(i,j,k)]) - lp/(dp*dp*dp)*(-2.0*d[I2V(i,j,k)]+2.0*d[I2V(i+1,j,k)]));
    } else if (i==loc_I-1) {
        termx = dp*lp/3.0 * (1/(lp*dp)*(d[I2V(i,j,k)]) - lp/(dp*dp*dp)*(-2.0*d[I2V(i,j,k)]+2.0*d[I2V(i-1,j,k)]));
    } else {
        termx = dp*lp/3.0 * (1/(lp*dp)*(d[I2V(i,j,k)]) - lp/(dp*dp*dp)*(-2.0*d[I2V(i,j,k)]+d[I2V(i-1,j,k)]+d[I2V(i+1,j,k)]));
    }

    if (j==0) {
        termy = dt*lt/3.0 * (1/(lt*dt)*(d[I2V(i,j,k)]) - lt/(dt*dt*dt)*(-2.0*d[I2V(i,j,k)]+2.0*d[I2V(i,j+1,k)]));
    } else if (j==loc_J-1) {
        termy = dt*lt/3.0 * (1/(lt*dt)*(d[I2V(i,j,k)]) - lt/(dt*dt*dt)*(-2.0*d[I2V(i,j,k)]+2.0*d[I2V(i,j-1,k)]));
    } else {
        termy = dt*lt/3.0 * (1/(lt*dt)*(d[I2V(i,j,k)]) - lt/(dt*dt*dt)*(-2.0*d[I2V(i,j,k)]+d[I2V(i,j-1,k)]+d[I2V(i,j+1,k)]));
    }

    if (k==0) {
        termz = dr*lr/3.0 * (1/(lr*dr)*(d[I2V(i,j,k)]) - lr/(dr*dr*dr)*(-2.0*d[I2V(i,j,k)]+2.0*d[I2V(i,j,k+1)]));
    } else if (k==loc_K-1) {
        termz = dr*lr/3.0 * (1/(lr*dr)*(d[I2V(i,j,k)]) - lr/(dr*dr*dr)*(-2.0*d[I2V(i,j,k)]+2.0*d[I2V(i,j,k-1)]));
    } else {
        termz = dr*lr/3.0 * (1/(lr*dr)*(d[I2V(i,j,k)]) - lr/(dr*dr*dr)*(-2.0*d[I2V(i,j,k)]+d[I2V(i,j,k-1)]+d[I2V(i,j,k+1)]));
    }

    Ap[I2V(i,j,k)] = termx+termy+termz;
}



void CG_smooth(Grid& grid, CUSTOMREAL* arr_in, CUSTOMREAL* arr_out, CUSTOMREAL lr, CUSTOMREAL lt, CUSTOMREAL lp) {
    // arr: array to be smoothed
    // lr: smooth length on r
    // lt: smooth length on theta
    // lp: smooth length on phi

    // arrays :
    // x_array  model
    // g_array  gradient
    // d_array  descent direction
    // Ap  stiffness scalar (laplacian)
    // pAp = d_array*Ap
    // rr = g_array * g_array (dot product)

    const int max_iter_cg = 1000;
    const CUSTOMREAL xtol = 0.001;
    const bool use_scaling = true;

    CUSTOMREAL dr=grid.dr, dt=grid.dt, dp=grid.dp; // in km, rad, rad
    //CUSTOMREAL dr=_1_CR,dt=_1_CR,dp=_1_CR;
    //debug
    std::cout << "dr, dt, dp, lr, lt, lp = " << dr << " " << dt << " " << dp << " " << lr << " " << lt << " " << lp << std::endl;

    // allocate memory
    CUSTOMREAL* x_array = allocateMemory<CUSTOMREAL>(loc_I*loc_J*loc_K, 3000);
    CUSTOMREAL* r_array = allocateMemory<CUSTOMREAL>(loc_I*loc_J*loc_K, 3001);
    CUSTOMREAL* p_array = allocateMemory<CUSTOMREAL>(loc_I*loc_J*loc_K, 3002);
    CUSTOMREAL* Ap = allocateMemory<CUSTOMREAL>(loc_I*loc_J*loc_K, 3003);
    CUSTOMREAL pAp=_0_CR, rr_0=_0_CR, rr=_0_CR, rr_new=_0_CR, aa=_0_CR, bb=_0_CR, tmp=_0_CR;

    CUSTOMREAL scaling_A=_1_CR, scaling_coeff = _1_CR;

    if (use_scaling) {
        // calculate scaling factor
        //scaling_A = std::sqrt(_1_CR / (_8_CR * PI * lr * lt * lp));
        // scaling coefficient for gradient
        scaling_coeff = find_absmax(arr_in, loc_I*loc_J*loc_K);
        tmp = scaling_coeff;
        allreduce_cr_single_max(tmp, scaling_coeff);
        //if (scaling_coeff == _0_CR)
        if (isZero(scaling_coeff))
            scaling_coeff = _1_CR;
    }
    // std out scaling factors
    if (myrank == 0) {
        std::cout << "scaling_A = " << scaling_A << std::endl;
        std::cout << "scaling_coeff = " << scaling_coeff << std::endl;
    }


    // array initialization
    for (int i=0; i<loc_I*loc_J*loc_K; i++) {
        x_array[i] = _0_CR;                   // x
        r_array[i] = arr_in[i]/scaling_coeff; // r = b - Ax
        p_array[i] = r_array[i];              // p
        Ap[i] = _0_CR;
    }

    // initial rr
    rr = dot_product(r_array, r_array, loc_I*loc_J*loc_K);
    tmp = rr;
    // sum all rr among all processors
    allreduce_cr_single(tmp, rr);
    rr_0 = rr; // record initial rr

    int k_start=0, k_end=loc_K, j_start=0, j_end=loc_J, i_start=0, i_end=loc_I;

    // CG loop
    for (int iter=0; iter<max_iter_cg; iter++) {

        // calculate laplacian
        for (int k = k_start; k < k_end; k++) {
            for (int j = j_start; j < j_end; j++) {
                for (int i = i_start; i < i_end; i++) {
                    //scaling_coeff = std::max(scaling_coeff, std::abs(arr[I2V(i,j,k)]));

                    // calculate inversed laplacian operator
                    calc_inversed_laplacian(p_array,Ap,i,j,k,lr,lt,lp,dr,dt,dp);

                    // scaling
                    Ap[I2V(i,j,k)] = Ap[I2V(i,j,k)]*scaling_A;
                }
            }
        }

        // get the values on the boundaries
        grid.send_recev_boundary_data(Ap);

        // calculate pAp
        pAp = dot_product(p_array, Ap, loc_I*loc_J*loc_K);
        tmp = pAp;
        // sum all pAp among all processors
        allreduce_cr_single(tmp, pAp);

        // compute step length
        aa = rr / pAp;

        // update x_array (model)
        for (int i=0; i<loc_I*loc_J*loc_K; i++) {
            x_array[i] += aa * p_array[i];
        }

        // update r_array (gradient)
        for (int i=0; i<loc_I*loc_J*loc_K; i++) {
            r_array[i] -= aa * Ap[i];
        }

        // update rr
        rr_new = dot_product(r_array, r_array, loc_I*loc_J*loc_K);
        tmp = rr_new;
        // sum all rr among all processors
        allreduce_cr_single(tmp, rr_new);

        // update d_array (descent direction)
        bb = rr_new / rr;
        for (int i=0; i<loc_I*loc_J*loc_K; i++) {
            p_array[i] = r_array[i] + bb * p_array[i];
        }

        if (myrank == 0 && iter%100==0){//} && if_verbose) {
            std::cout << "iter: " << iter << " rr: " << rr << " rr_new: " << rr_new << " rr/rr_0: " << rr/rr_0 << " pAp: " << pAp << " aa: " << aa << " bb: " << bb << std::endl;
        }

        // update rr
        rr = rr_new;

        // check convergence
        if (rr / rr_0 < xtol) {
            std::cout << "CG converged in " << iter << " iterations." << std::endl;
            break;
        }

    } // end of CG loop


    // copy x_array to arr_out
    for (int i=0; i<loc_I*loc_J*loc_K; i++) {
        arr_out[i] = x_array[i]*scaling_coeff;
    }

    // deallocate
    delete[] x_array;
    delete[] r_array;
    delete[] p_array;
    delete[] Ap;
    //delete[] Ap_tmp;

}




void smooth_inv_kernels_CG(Grid& grid, CUSTOMREAL lr, CUSTOMREAL lt, CUSTOMREAL lp) {
    // smoothing kernels with Conjugate Gradient method
    // lr,lt,lp: smoothing length in r, theta, phi direction

    // smooth Ks
    CG_smooth(grid, grid.Ks_loc, grid.Ks_update_loc, lr, lt, lp);
    // smooth Keta
    CG_smooth(grid, grid.Keta_loc, grid.Keta_update_loc, lr, lt, lp);
    // smooth Kxi
    CG_smooth(grid, grid.Kxi_loc, grid.Kxi_update_loc, lr, lt, lp);

}





// original method for smoothing kernels
inline void smooth_inv_kernels_orig(Grid& grid, InputParams& IP) {
    // necessary params
    CUSTOMREAL r_r, r_t, r_p;
    CUSTOMREAL r_r_ani, r_t_ani, r_p_ani;

    int kdr = 0, jdt = 0, idp = 0;
    int kdr_ani = 0, jdt_ani = 0, idp_ani = 0;

    int k_start = 0;
    int j_start = 0;
    int i_start = 0;
    int k_end   = ngrid_k;
    int j_end   = ngrid_j;
    int i_end   = ngrid_i;

    CUSTOMREAL weight   = _1_CR;
    CUSTOMREAL * taper  = IP.get_depth_taper();


    // kernel density normalization
    for (int i_loc = 0; i_loc < loc_I; i_loc++) {
        for (int j_loc = 0; j_loc < loc_J; j_loc++) {
            for (int k_loc = 0; k_loc < loc_K; k_loc++) {
                if(isZero(grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)])){
                    // do nothing
                } else {
                    if (grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)] < 0){
                        std::cout   << "Warning: grid.Ks_density_loc[I2V(" << i_loc << "," << j_loc << "," << k_loc << ")] is less than 0, = " 
                                    << grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)]                      
                                    << std::endl;
                    }
                    grid.Ks_loc[I2V(i_loc,j_loc,k_loc)] /= std::pow(std::abs(grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)]),Kdensity_coe);
                }
                
                if(isZero(grid.Kxi_density_loc[I2V(i_loc,j_loc,k_loc)])){
                    // do nothing
                } else {
                    if (grid.Kxi_density_loc[I2V(i_loc,j_loc,k_loc)] < 0){
                        std::cout << "Warning: grid.Kxi_density_loc[I2V(" << i_loc << "," << j_loc << "," << k_loc << ")] is less than 0, = " << grid.Kxi_density_loc[I2V(i_loc,j_loc,k_loc)] << std::endl;
                    }
                    grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] /= std::pow(std::abs(grid.Kxi_density_loc[I2V(i_loc,j_loc,k_loc)]),Kdensity_coe);
                }

                if(isZero(grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)])){
                    // do nothing
                } else {
                    if (grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] < 0){
                        std::cout << "Warning: grid.Keta_density_loc[I2V(" << i_loc << "," << j_loc << "," << k_loc << ")] is less than 0, = " << grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] << std::endl;
                    }
                    grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] /= std::pow(std::abs(grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)]),Kdensity_coe);
                }
            }
        }
    }

    //
    // sum the kernel values on the inversion grid
    //

    for (int i_grid = 0; i_grid < n_inv_grids; i_grid++) {

        // init coarser kernel arrays
        // velocity
        for (int k = 0; k < n_inv_K_loc; k++) {
            for (int j = 0; j < n_inv_J_loc; j++) {
                for (int i = 0; i < n_inv_I_loc; i++) {
                    grid.Ks_inv_loc[  I2V_INV_KNL(i,j,k)] = _0_CR;
                }
            }
        }
        // anisotropy
        for (int k = 0; k < n_inv_K_loc_ani; k++) {
            for (int j = 0; j < n_inv_J_loc_ani; j++) {
                for (int i = 0; i < n_inv_I_loc_ani; i++) {
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(i,j,k)] = _0_CR;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(i,j,k)] = _0_CR;
                }
            }
        }


        for (int k = k_start; k < k_end; k++) {
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
            r_r_ani = -_1_CR;
            for (int ii_invr = 0; ii_invr < n_inv_K_loc_ani-1; ii_invr++){
                // increasing or decreasing order
                if (in_between(r_glob, grid.inv_grid->r_ani.arr[I2V_INV_ANI_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r_ani.arr[I2V_INV_ANI_GRIDS_1DK(ii_invr+1,i_grid)])) {
                    kdr_ani = ii_invr;
                    r_r_ani = calc_ratio_between(r_glob, grid.inv_grid->r_ani.arr[I2V_INV_ANI_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r_ani.arr[I2V_INV_ANI_GRIDS_1DK(ii_invr+1,i_grid)]);
                    break;
                }
            }

            // continue if r is out of the inversion grid
            if (r_r < _0_CR || r_r_ani < _0_CR) continue;

            for (int j = j_start; j < j_end; j++) {
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

                r_t_ani = -_1_CR;
                for (int ii_invt = 0; ii_invt < n_inv_J_loc_ani-1; ii_invt++){
                    //if ((t_glob - grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt,i_grid)]) * (t_glob - grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt+1,i_grid)]) <= _0_CR) {
                    CUSTOMREAL left  = grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt,  kdr_ani,i_grid)]*(1-r_r_ani) + grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt,  kdr_ani+1,i_grid)]*(r_r_ani);
                    CUSTOMREAL right = grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt+1,kdr_ani,i_grid)]*(1-r_r_ani) + grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt+1,kdr_ani+1,i_grid)]*(r_r_ani);
                    if (in_between(t_glob, left, right)) {
                        jdt_ani = ii_invt;
                        r_t_ani = calc_ratio_between(t_glob, left, right);
                        break;
                    }
                }

                // continue if t is out of the inversion grid
                if (r_t < _0_CR || r_t_ani < _0_CR) continue;

                for (int i = i_start; i < i_end; i++) {
                    CUSTOMREAL p_glob = grid.get_lon_min() + i*grid.get_delta_lon();    // global coordinate of p (longitude)
                    r_p = -_1_CR;
                    for (int ii_invp = 0; ii_invp < n_inv_I_loc-1; ii_invp++){
                        CUSTOMREAL left  = grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,  kdr,i_grid)]*(1-r_r) + grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,  kdr+1,i_grid)]*(r_r);
                        CUSTOMREAL right = grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,kdr,i_grid)]*(1-r_r) + grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,kdr+1,i_grid)]*(r_r);
                        if (in_between(p_glob, left, right)) {
                            idp = ii_invp;
                            r_p = calc_ratio_between(p_glob, left, right);
                            break;
                        }
                    }

                    r_p_ani = -_1_CR;
                    for (int ii_invp = 0; ii_invp < n_inv_I_loc_ani-1; ii_invp++){
                        CUSTOMREAL left  = grid.inv_grid->p_ani.arr[I2V_INV_ANI_GRIDS_1DI(ii_invp,  kdr_ani,i_grid)]*(1-r_r_ani) + grid.inv_grid->p_ani.arr[I2V_INV_ANI_GRIDS_1DI(ii_invp,  kdr_ani+1,i_grid)]*(r_r_ani);
                        CUSTOMREAL right = grid.inv_grid->p_ani.arr[I2V_INV_ANI_GRIDS_1DI(ii_invp+1,kdr_ani,i_grid)]*(1-r_r_ani) + grid.inv_grid->p_ani.arr[I2V_INV_ANI_GRIDS_1DI(ii_invp+1,kdr_ani+1,i_grid)]*(r_r_ani);
                        if (in_between(p_glob, left, right)) {
                            idp_ani = ii_invp;
                            r_p_ani = calc_ratio_between(p_glob, left, right);
                            break;
                        }
                    }

                    // continue if p is out of the inversion grid
                    if (r_p < _0_CR || r_p_ani < _0_CR) continue;

                    // global grid id to local id
                    int k_loc = k - grid.get_offset_k();
                    int j_loc = j - grid.get_offset_j();
                    int i_loc = i - grid.get_offset_i();

                    // check if *_loc are inside the local subdomain
                    if (k_loc < 0 || k_loc > loc_K-1) continue;
                    if (j_loc < 0 || j_loc > loc_J-1) continue;
                    if (i_loc < 0 || i_loc > loc_I-1) continue;

                    // check if Ks_density_loc is 0 (no contributary kernel here)
                    // if(isZero(grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)])) continue;

                    CUSTOMREAL dxdydz = (grid.dr) * (r_glob*grid.dt) * (r_glob*cos(t_glob)*grid.dp);    // dx*dy*dz  = r^2*cos(t)dr*dt*dp

                    // update Ks_inv Keta_inv Kxi_inv
                    grid.Ks_inv_loc[  I2V_INV_KNL(    idp,    jdt,    kdr)]     += (_1_CR-r_r)    *(_1_CR-r_t)    *(_1_CR-r_p)    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    // grid.Kdensity_inv_loc[  I2V_INV_KNL(  idp,jdt,    kdr)]     += (_1_CR-r_r)    *(_1_CR-r_t)    *(_1_CR-r_p)    *grid.Kdensity_loc[  I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(    idp,    jdt,    kdr    +1)] += r_r    *(_1_CR-r_t)    *(_1_CR-r_p)    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    // grid.Kdensity_inv_loc[  I2V_INV_KNL(  idp,jdt,    kdr    +1)] += r_r    *(_1_CR-r_t)    *(_1_CR-r_p)    *grid.Kdensity_loc[  I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(    idp,    jdt    +1,kdr)]     += (_1_CR-r_r)    *r_t    *(_1_CR-r_p)    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    // grid.Kdensity_inv_loc[  I2V_INV_KNL(  idp,jdt    +1,kdr)]     += (_1_CR-r_r)    *r_t    *(_1_CR-r_p)    *grid.Kdensity_loc[  I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(    idp    +1,jdt,    kdr)]     += (_1_CR-r_r)    *(_1_CR-r_t)    *r_p    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    // grid.Kdensity_inv_loc[  I2V_INV_KNL(  idp+1,jdt,    kdr)]     += (_1_CR-r_r)    *(_1_CR-r_t)    *r_p    *grid.Kdensity_loc[  I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(    idp,    jdt    +1,kdr    +1)] += r_r    *r_t    *(_1_CR-r_p)    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    // grid.Kdensity_inv_loc[  I2V_INV_KNL(  idp,jdt    +1,kdr    +1)] += r_r    *r_t    *(_1_CR-r_p)    *grid.Kdensity_loc[  I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(    idp    +1,jdt,    kdr    +1)] += r_r    *(_1_CR-r_t)    *r_p    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    // grid.Kdensity_inv_loc[  I2V_INV_KNL(  idp+1,jdt,    kdr    +1)] += r_r    *(_1_CR-r_t)    *r_p    *grid.Kdensity_loc[  I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(    idp    +1,jdt    +1,kdr)]     += (_1_CR-r_r)    *r_t    *r_p    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    // grid.Kdensity_inv_loc[  I2V_INV_KNL(  idp+1,jdt    +1,kdr)]     += (_1_CR-r_r)    *r_t    *r_p    *grid.Kdensity_loc[  I2V(i_loc,j_loc,k_loc)];

                    grid.Ks_inv_loc[  I2V_INV_KNL(    idp    +1,jdt    +1,kdr    +1)] += r_r    *r_t    *r_p    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*r_p_ani*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*r_p_ani*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                    // grid.Kdensity_inv_loc[  I2V_INV_KNL(  idp+1,jdt    +1,kdr    +1)] += r_r    *r_t    *r_p    *grid.Kdensity_loc[  I2V(i_loc,j_loc,k_loc)];
                } // end for i
            } // end for j
        } // end for k


        // sum over all sub-domains
        allreduce_cr_inplace(grid.Ks_inv_loc,       n_inv_I_loc*n_inv_J_loc*n_inv_K_loc);
        allreduce_cr_inplace(grid.Keta_inv_loc,     n_inv_I_loc_ani*n_inv_J_loc_ani*n_inv_K_loc_ani);
        allreduce_cr_inplace(grid.Kxi_inv_loc,      n_inv_I_loc_ani*n_inv_J_loc_ani*n_inv_K_loc_ani);
        // allreduce_cr_inplace(grid.Kdensity_inv_loc, n_inv_I_loc*n_inv_J_loc*n_inv_K_loc);


        //
        // update factors
        //
        for (int k = k_start; k < k_end; k++) {
            CUSTOMREAL r_glob = grid.get_r_min() + k*grid.get_delta_r(); // global coordinate of r
            r_r = -_1_CR;
            for (int ii_invr = 0; ii_invr < n_inv_K_loc-1; ii_invr++){
                if (in_between(r_glob, grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)])) {
                    kdr = ii_invr;
                    r_r = calc_ratio_between(r_glob, grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)]);
                    break;
                }
            }

            r_r_ani = -_1_CR;
            for (int ii_invr = 0; ii_invr < n_inv_K_loc_ani-1; ii_invr++){
                // increasing or decreasing order
                if (in_between(r_glob, grid.inv_grid->r_ani.arr[I2V_INV_ANI_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r_ani.arr[I2V_INV_ANI_GRIDS_1DK(ii_invr+1,i_grid)])) {
                    kdr_ani = ii_invr;
                    r_r_ani = calc_ratio_between(r_glob, grid.inv_grid->r_ani.arr[I2V_INV_ANI_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r_ani.arr[I2V_INV_ANI_GRIDS_1DK(ii_invr+1,i_grid)]);
                    break;
                }
            }
            // depth model update taper
            CUSTOMREAL depth = radius2depth(r_glob);
            if (depth < taper[0]) {     // weight = 0;
                weight = _0_CR;
            } else if (depth < taper[1]) {
                weight = (_1_CR - std::cos(PI*(depth - taper[0])/(taper[1] - taper[0]))) / _2_CR;
            } else {
                weight = _1_CR;
            }

            // continue if r is out of the inversion grid
            if (r_r < _0_CR || r_r_ani < _0_CR) continue;

            for (int j = j_start; j < j_end; j++) {
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

                r_t_ani = -_1_CR;
                for (int ii_invt = 0; ii_invt < n_inv_J_loc_ani-1; ii_invt++){
                    CUSTOMREAL left  = grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt,  kdr_ani,i_grid)]*(1-r_r_ani) + grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt,  kdr_ani+1,i_grid)]*(r_r_ani);
                    CUSTOMREAL right = grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt+1,kdr_ani,i_grid)]*(1-r_r_ani) + grid.inv_grid->t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(ii_invt+1,kdr_ani+1,i_grid)]*(r_r_ani);
                    if (in_between(t_glob, left, right)) {
                        jdt_ani = ii_invt;
                        r_t_ani = calc_ratio_between(t_glob, left, right);
                        break;
                    }
                }

                // continue if t is out of the inversion grid
                if (r_t < _0_CR || r_t_ani < _0_CR) continue;

                for (int i = i_start; i < i_end; i++) {
                    CUSTOMREAL p_glob = grid.get_lon_min() + i*grid.get_delta_lon();
                    r_p = -_1_CR;
                    for (int ii_invp = 0; ii_invp < n_inv_I_loc-1; ii_invp++){
                        CUSTOMREAL left  = grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,  kdr,i_grid)]*(1-r_r) + grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp,  kdr+1,i_grid)]*(r_r);
                        CUSTOMREAL right = grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,kdr,i_grid)]*(1-r_r) + grid.inv_grid->p.arr[I2V_INV_GRIDS_2DI(ii_invp+1,kdr+1,i_grid)]*(r_r);
                        if (in_between(p_glob, left, right)) {
                            idp = ii_invp;
                            r_p = calc_ratio_between(p_glob, left, right);
                            break;
                        }
                    }

                    r_p_ani = -_1_CR;
                    for (int ii_invp = 0; ii_invp < n_inv_I_loc_ani-1; ii_invp++){
                        CUSTOMREAL left  = grid.inv_grid->p_ani.arr[I2V_INV_ANI_GRIDS_1DI(ii_invp,  kdr_ani,i_grid)]*(1-r_r_ani) + grid.inv_grid->p_ani.arr[I2V_INV_ANI_GRIDS_1DI(ii_invp,  kdr_ani+1,i_grid)]*(r_r_ani);
                        CUSTOMREAL right = grid.inv_grid->p_ani.arr[I2V_INV_ANI_GRIDS_1DI(ii_invp+1,kdr_ani,i_grid)]*(1-r_r_ani) + grid.inv_grid->p_ani.arr[I2V_INV_ANI_GRIDS_1DI(ii_invp+1,kdr_ani+1,i_grid)]*(r_r_ani);
                        if (in_between(p_glob, left, right)) {
                            idp_ani = ii_invp;
                            r_p_ani = calc_ratio_between(p_glob, left, right);
                            break;
                        }
                    }

                    // continue if p is out of the inversion grid
                    if (r_p < _0_CR || r_p_ani < _0_CR) continue;

                    // global grid id to local id
                    int k_loc = k - grid.get_offset_k();
                    int j_loc = j - grid.get_offset_j();
                    int i_loc = i - grid.get_offset_i();

                    // check if *_loc are inside the local subdomain
                    if (k_loc < 0 || k_loc > loc_K-1) continue;
                    if (j_loc < 0 || j_loc > loc_J-1) continue;
                    if (i_loc < 0 || i_loc > loc_I-1) continue;

                    CUSTOMREAL pert_Ks   = 0.0;
                    CUSTOMREAL pert_Keta = 0.0;
                    CUSTOMREAL pert_Kxi  = 0.0;
                    // CUSTOMREAL pert_Kdensity  = 0.0;

                    pert_Ks         += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt,kdr)];
                    pert_Keta       += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)];
                    pert_Kxi        += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)];
                    // pert_Kdensity   += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Kdensity_inv_loc[  I2V_INV_KNL(idp,jdt,kdr)];

                    pert_Ks         += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt,kdr+1)];
                    pert_Keta       += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)];
                    pert_Kxi        += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)];
                    // pert_Kdensity   += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Kdensity_inv_loc[  I2V_INV_KNL(idp,jdt,kdr+1)];

                    pert_Ks         += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr)];
                    pert_Keta       += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)];
                    pert_Kxi        += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)];
                    // pert_Kdensity   += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Kdensity_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr)];

                    pert_Ks         += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr)];
                    pert_Keta       += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)];
                    pert_Kxi        += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)];
                    // pert_Kdensity   += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Kdensity_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr)];

                    pert_Ks         += r_r*r_t*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr+1)];
                    pert_Keta       += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)];
                    pert_Kxi        += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)];
                    // pert_Kdensity   += r_r*r_t*(_1_CR-r_p)*grid.Kdensity_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr+1)];

                    pert_Ks         += r_r*(_1_CR-r_t)*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr+1)];
                    pert_Keta       += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)];
                    pert_Kxi        += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)];
                    // pert_Kdensity   += r_r*(_1_CR-r_t)*r_p*grid.Kdensity_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr+1)];

                    pert_Ks         += (_1_CR-r_r)*r_t*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr)];
                    pert_Keta       += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)];
                    pert_Kxi        += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)];
                    // pert_Kdensity   += (_1_CR-r_r)*r_t*r_p*grid.Kdensity_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr)];

                    pert_Ks         += r_r*r_t*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr+1)];
                    pert_Keta       += r_r_ani*r_t_ani*r_p_ani*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)];
                    pert_Kxi        += r_r_ani*r_t_ani*r_p_ani*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)];
                    // pert_Kdensity   += r_r*r_t*r_p*grid.Kdensity_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr+1)];


                    // update para
                    grid.Ks_update_loc[         I2V(i_loc,j_loc,k_loc)] += weight * pert_Ks;
                    grid.Keta_update_loc[       I2V(i_loc,j_loc,k_loc)] += weight * pert_Keta;
                    grid.Kxi_update_loc[        I2V(i_loc,j_loc,k_loc)] += weight * pert_Kxi;
                    // grid.Kdensity_update_loc[   I2V(i_loc,j_loc,k_loc)] += pert_Kdensity;

                } // end for i
            } // end for j
        } // end for k

    } // end i_grid


    //
    // rescale kernel update to -1 ~ 1
    //
    // get the scaling factor
    CUSTOMREAL Linf_Ks = _0_CR, Linf_Keta = _0_CR, Linf_Kxi = _0_CR;
    // CUSTOMREAL Linf_Kden = _0_CR;
    for (int k = 1; k < loc_K-1; k++) {
        for (int j = 1; j < loc_J-1; j++) {
            for (int i = 1; i < loc_I-1; i++) {
                Linf_Ks   = std::max(Linf_Ks,   std::abs(grid.Ks_update_loc[I2V(i,j,k)]));
                Linf_Keta = std::max(Linf_Keta, std::abs(grid.Keta_update_loc[I2V(i,j,k)]));
                Linf_Kxi  = std::max(Linf_Kxi,  std::abs(grid.Kxi_update_loc[I2V(i,j,k)]));
                // Linf_Kden = std::max(Linf_Kden, std::abs(grid.Kdensity_update_loc[I2V(i,j,k)]));
            }
        }
    }

    // get the maximum scaling factor among subdomains
    CUSTOMREAL Linf_tmp;
    allreduce_cr_single_max(Linf_Ks, Linf_tmp);   Linf_Ks = Linf_tmp;
    allreduce_cr_single_max(Linf_Keta, Linf_tmp); Linf_Keta = Linf_tmp;
    allreduce_cr_single_max(Linf_Kxi, Linf_tmp);  Linf_Kxi = Linf_tmp;
    // CUSTOMREAL Linf_Kden_tmp;
    // allreduce_cr_single_max(Linf_Kden, Linf_Kden_tmp); Linf_Kden = Linf_Kden_tmp;

    CUSTOMREAL Linf_all = _0_CR;
    Linf_all = std::max(Linf_Ks, std::max(Linf_Keta, Linf_Kxi));

    Linf_Ks = Linf_all;
    Linf_Keta = Linf_all;
    Linf_Kxi = Linf_all;

    // rescale the kernel update
    for (int k = 0; k < loc_K; k++) {
        for (int j = 0; j < loc_J; j++) {
            for (int i = 0; i < loc_I; i++) {
                grid.Ks_update_loc[I2V(i,j,k)]      /= Linf_Ks;
                grid.Keta_update_loc[I2V(i,j,k)]    /= Linf_Keta;
                grid.Kxi_update_loc[I2V(i,j,k)]     /= Linf_Kxi;
                // grid.Kdensity_update_loc[I2V(i,j,k)] /= Linf_Kden;
            }
        }
    }
}


#endif