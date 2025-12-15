#include "kernel_postprocessing.h"

namespace Kernel_postprocessing {

    // ---------------------------------------------------
    // ------------------ main function ------------------
    // ---------------------------------------------------

    // process kernels (multigrid, normalization ...)
    // Ks_loc, Keta_loc, Kxi_loc
    // --> 
    // Ks_processing_loc, Keta_processing_loc, Kxi_processing_loc
    void process_kernels(InputParams& IP, Grid& grid) {
        
        if (subdom_main){ // parallel level 3
            if (id_sim==0){ // parallel level 1
                initialize_processing_kernels(grid);

                if (smooth_method == MULTI_GRID_SMOOTHING) {
                    // eliminate overlapping kernels on ghost layers when performing multigrid smoothing
                    eliminate_ghost_layer_multigrid(grid);

                    // multigrid parameterization + kernel density normalization
                    multigrid_parameterization_density_normalization(IP, grid);
                }
                // make the boundary values of updated kernels consistent among subdomains
                shared_boundary_of_processing_kernels(grid);
            } // end if id_sim == 0

            // boardcast modified kernels to all simultaneous runs
            broadcast_cr_inter_sim(grid.Ks_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Kxi_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Keta_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Ks_density_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Kxi_density_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Keta_density_processing_loc.data(), loc_I*loc_J*loc_K, 0);
        } // end if subdom_main

        synchronize_all_world();
    }


    // normalize kernels to -1 ~ 1
    void normalize_kernels(Grid& grid){
        if (subdom_main){ // parallel level 3
            if (id_sim==0){ // parallel level 1
                kernel_rescaling_to_unit(grid);
            }
            // boardcast modified kernels to all simultaneous runs
            broadcast_cr_inter_sim(grid.Ks_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Kxi_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Keta_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Ks_density_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Kxi_density_processing_loc.data(), loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Keta_density_processing_loc.data(), loc_I*loc_J*loc_K, 0);
        }
        synchronize_all_world();
    }
    

    // assign processing kernels to modified kernels for model update
    void assign_to_modified_kernels(Grid& grid){
        if (subdom_main){ // parallel level 3
            if (id_sim==0){ // parallel level 1
                // assign processing kernels to modified kernels for model update
                std::copy(grid.Ks_processing_loc.begin(), grid.Ks_processing_loc.end(), grid.Ks_update_loc);
                std::copy(grid.Kxi_processing_loc.begin(), grid.Kxi_processing_loc.end(), grid.Kxi_update_loc);
                std::copy(grid.Keta_processing_loc.begin(), grid.Keta_processing_loc.end(), grid.Keta_update_loc);

                std::copy(grid.Ks_density_processing_loc.begin(), grid.Ks_density_processing_loc.end(), grid.Ks_density_update_loc);
                std::copy(grid.Kxi_density_processing_loc.begin(), grid.Kxi_density_processing_loc.end(), grid.Kxi_density_update_loc);
                std::copy(grid.Keta_density_processing_loc.begin(), grid.Keta_density_processing_loc.end(), grid.Keta_density_update_loc);
            }
            // boardcast modified kernels to all simultaneous runs
            broadcast_cr_inter_sim(grid.Ks_update_loc, loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Kxi_update_loc, loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Keta_update_loc, loc_I*loc_J*loc_K, 0);

            broadcast_cr_inter_sim(grid.Ks_density_update_loc, loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Kxi_density_update_loc, loc_I*loc_J*loc_K, 0);
            broadcast_cr_inter_sim(grid.Keta_density_update_loc, loc_I*loc_J*loc_K, 0);
        }
        synchronize_all_world();
    }

    // ---------------------------------------------------
    // ------------------ sub functions ------------------
    // ---------------------------------------------------

    // initialize processing kernels
    void initialize_processing_kernels(Grid& grid){
        if (subdom_main && id_sim==0 ){ // main of level 3 and level 1   
            for (int k = 0; k < loc_K; k++) {
                for (int j = 0; j < loc_J; j++) {
                    for (int i = 0; i < loc_I; i++) {
                        // initialize processing kernel
                        grid.Ks_processing_loc[I2V(i,j,k)]           = _0_CR;
                        grid.Keta_processing_loc[I2V(i,j,k)]         = _0_CR;
                        grid.Kxi_processing_loc[I2V(i,j,k)]          = _0_CR;
                        grid.Ks_density_processing_loc[I2V(i,j,k)]   = _0_CR;
                        grid.Kxi_density_processing_loc[I2V(i,j,k)]  = _0_CR;
                        grid.Keta_density_processing_loc[I2V(i,j,k)] = _0_CR;
                    }
                }
            }
        }
    }


    // eliminate overlapping kernels on ghost layers when performing multigrid smoothing
    void eliminate_ghost_layer_multigrid(Grid& grid){
        if (!subdom_main){
            std::cout << "Error: eliminate_ghost_layer_multigrid should be called only when subdom_main==true (main proc of level 3)" << std::endl;
            exit(1);
        }
        if (id_sim != 0){
            std::cout << "Error: eliminate_ghost_layer_multigrid should be called only when id_sim==0 (by the first simultaneous group of level 1)" << std::endl;
            exit(1);
        }
        if (smooth_method != MULTI_GRID_SMOOTHING){ // parallel level 3
            std::cout << "Error: eliminate_ghost_layer_multigrid should be called only when smooth_method==MULTI_GRID_SMOOTHING" << std::endl;
            exit(1);
        }

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
        
    }


    // update boundary and corners between subdomains
    void shared_boundary_of_processing_kernels(Grid& grid){
        // 6 faces
        grid.send_recev_boundary_data(grid.Ks_processing_loc.data());
        grid.send_recev_boundary_data(grid.Keta_processing_loc.data());
        grid.send_recev_boundary_data(grid.Kxi_processing_loc.data());
        grid.send_recev_boundary_data(grid.Ks_density_processing_loc.data());
        grid.send_recev_boundary_data(grid.Kxi_density_processing_loc.data());
        grid.send_recev_boundary_data(grid.Keta_density_processing_loc.data());

        // 20 edges and corners
        grid.send_recev_boundary_data_kosumi(grid.Ks_processing_loc.data());
        grid.send_recev_boundary_data_kosumi(grid.Keta_processing_loc.data());
        grid.send_recev_boundary_data_kosumi(grid.Kxi_processing_loc.data());
        grid.send_recev_boundary_data_kosumi(grid.Ks_density_processing_loc.data());
        grid.send_recev_boundary_data_kosumi(grid.Kxi_density_processing_loc.data());
        grid.send_recev_boundary_data_kosumi(grid.Keta_density_processing_loc.data());
    }


    // METHOD 1, multigrid parameterization + kernel density normalization
    void multigrid_parameterization_density_normalization(InputParams& IP, Grid& grid) {
        if (!subdom_main){
            std::cout << "Error: eliminate_ghost_layer_multigrid should be called only when subdom_main==true (main proc of level 3)" << std::endl;
            exit(1);
        }
        if (id_sim != 0){
            std::cout << "Error: eliminate_ghost_layer_multigrid should be called only when id_sim==0 (by the first simultaneous group of level 1)" << std::endl;
            exit(1);
        }
        if (smooth_method != MULTI_GRID_SMOOTHING){ // parallel level 3
            std::cout << "Error: eliminate_ghost_layer_multigrid should be called only when smooth_method==MULTI_GRID_SMOOTHING" << std::endl;
            exit(1);
        }

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

        // check final kernel density
        for (int i_loc = 0; i_loc < loc_I; i_loc++) {
            for (int j_loc = 0; j_loc < loc_J; j_loc++) {
                for (int k_loc = 0; k_loc < loc_K; k_loc++) {
                    if (isNegative(grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)])){
                        std::cout   << "Warning: grid.Ks_density_loc[I2V(" << i_loc << "," << j_loc << "," << k_loc << ")] is less than 0, = " << grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)] << std::endl;
                    }
                    if (isNegative(grid.Kxi_density_loc[I2V(i_loc,j_loc,k_loc)])){
                        std::cout << "Warning: grid.Kxi_density_loc[I2V(" << i_loc << "," << j_loc << "," << k_loc << ")] is less than 0, = " << grid.Kxi_density_loc[I2V(i_loc,j_loc,k_loc)] << std::endl;
                    }
                    if (isNegative(grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)])){
                        std::cout << "Warning: grid.Keta_density_loc[I2V(" << i_loc << "," << j_loc << "," << k_loc << ")] is less than 0, = " << grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] << std::endl;
                    }
                }
            }
        }


        //  --------------- multigrid parameterization ---------------
        for (int i_grid = 0; i_grid < n_inv_grids; i_grid++) {      // loop over several sets of inversion grids

            // --------------- init coarser kernel arrays
            // velocity
            for (int k = 0; k < n_inv_K_loc; k++) {
                for (int j = 0; j < n_inv_J_loc; j++) {
                    for (int i = 0; i < n_inv_I_loc; i++) {
                        grid.Ks_inv_loc[  I2V_INV_KNL(i,j,k)] = _0_CR;
                        grid.Ks_density_inv_loc[I2V_INV_KNL(i,j,k)] = _0_CR;
                    }
                }
            }
            // anisotropy
            for (int k = 0; k < n_inv_K_loc_ani; k++) {
                for (int j = 0; j < n_inv_J_loc_ani; j++) {
                    for (int i = 0; i < n_inv_I_loc_ani; i++) {
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(i,j,k)] = _0_CR;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)] = _0_CR;

                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(i,j,k)] = _0_CR;
                        grid.Kxi_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)] = _0_CR;
                    }
                }
            }

            // --------------- project fine kernels to coarser inversion grid (integration)
            for (int k = k_start; k < k_end; k++) {     // loop over depth
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

                for (int j = j_start; j < j_end; j++) {     // loop over latitude
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

                    for (int i = i_start; i < i_end; i++) {     // loop over longitude
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



                        // ------------ accumulate differential elements
                        CUSTOMREAL dxdydz = (grid.dr) * (r_glob*grid.dt) * (r_glob*cos(t_glob)*grid.dp);    // dx*dy*dz  = r^2*cos(t)dr*dt*dp

                        // -p, -t, -r
                        grid.Ks_inv_loc[  I2V_INV_KNL(    idp,    jdt,    kdr)]     += (_1_CR-r_r)    *(_1_CR-r_t)    *(_1_CR-r_p)    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        grid.Ks_density_inv_loc[  I2V_INV_KNL(    idp,    jdt,    kdr)]     += (_1_CR-r_r)    *(_1_CR-r_t)    *(_1_CR-r_p)    *grid.Ks_density_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_density_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        // -p, -t, +r
                        grid.Ks_inv_loc[  I2V_INV_KNL(    idp,    jdt,    kdr    +1)] += r_r    *(_1_CR-r_t)    *(_1_CR-r_p)    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        grid.Ks_density_inv_loc[  I2V_INV_KNL(    idp,    jdt,    kdr    +1)] += r_r    *(_1_CR-r_t)    *(_1_CR-r_p)    *grid.Ks_density_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_density_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;

                        // -p, +t, -r
                        grid.Ks_inv_loc[  I2V_INV_KNL(    idp,    jdt    +1,kdr)]     += (_1_CR-r_r)    *r_t    *(_1_CR-r_p)    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        grid.Ks_density_inv_loc[  I2V_INV_KNL(    idp,    jdt    +1,kdr)]     += (_1_CR-r_r)    *r_t    *(_1_CR-r_p)    *grid.Ks_density_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_density_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        // +p, -t, -r
                        grid.Ks_inv_loc[  I2V_INV_KNL(    idp    +1,jdt,    kdr)]     += (_1_CR-r_r)    *(_1_CR-r_t)    *r_p    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        grid.Ks_density_inv_loc[  I2V_INV_KNL(    idp    +1,jdt,    kdr)]     += (_1_CR-r_r)    *(_1_CR-r_t)    *r_p    *grid.Ks_density_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)] += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_density_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        // -p, +t, +r
                        grid.Ks_inv_loc[  I2V_INV_KNL(    idp,    jdt    +1,kdr    +1)] += r_r    *r_t    *(_1_CR-r_p)    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        grid.Ks_density_inv_loc[  I2V_INV_KNL(    idp,    jdt    +1,kdr    +1)] += r_r    *r_t    *(_1_CR-r_p)    *grid.Ks_density_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_density_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        // +p, -t, +r
                        grid.Ks_inv_loc[  I2V_INV_KNL(    idp    +1,jdt,    kdr    +1)] += r_r    *(_1_CR-r_t)    *r_p    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        grid.Ks_density_inv_loc[  I2V_INV_KNL(    idp    +1,jdt,    kdr    +1)] += r_r    *(_1_CR-r_t)    *r_p    *grid.Ks_density_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)] += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_density_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        // +p, +t, -r
                        grid.Ks_inv_loc[  I2V_INV_KNL(    idp    +1,jdt    +1,kdr)]     += (_1_CR-r_r)    *r_t    *r_p    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        grid.Ks_density_inv_loc[  I2V_INV_KNL(    idp    +1,jdt    +1,kdr)]     += (_1_CR-r_r)    *r_t    *r_p    *grid.Ks_density_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)] += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Kxi_density_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        // +p, +t, +r
                        grid.Ks_inv_loc[  I2V_INV_KNL(    idp    +1,jdt    +1,kdr    +1)] += r_r    *r_t    *r_p    *grid.Ks_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*r_p_ani*grid.Keta_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*r_p_ani*grid.Kxi_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                        grid.Ks_density_inv_loc[  I2V_INV_KNL(    idp    +1,jdt    +1,kdr    +1)] += r_r    *r_t    *r_p    *grid.Ks_density_loc[  I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*r_p_ani*grid.Keta_density_loc[I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)] += r_r_ani*r_t_ani*r_p_ani*grid.Kxi_density_loc[ I2V(i_loc,j_loc,k_loc)] * dxdydz;
                        
                    } // end for i
                } // end for j
            } // end for k

            // ----------------
            // Here, we have projected fine kernels to coarser inversion grid by doing integration
            // ----------------


            // sum over all sub-domains (level 2)
            allreduce_cr_inplace(grid.Ks_inv_loc,       n_inv_I_loc*n_inv_J_loc*n_inv_K_loc);
            allreduce_cr_inplace(grid.Keta_inv_loc,     n_inv_I_loc_ani*n_inv_J_loc_ani*n_inv_K_loc_ani);
            allreduce_cr_inplace(grid.Kxi_inv_loc,      n_inv_I_loc_ani*n_inv_J_loc_ani*n_inv_K_loc_ani);

            allreduce_cr_inplace(grid.Ks_density_inv_loc,       n_inv_I_loc*n_inv_J_loc*n_inv_K_loc);
            allreduce_cr_inplace(grid.Keta_density_inv_loc,     n_inv_I_loc_ani*n_inv_J_loc_ani*n_inv_K_loc_ani);
            allreduce_cr_inplace(grid.Kxi_density_inv_loc,      n_inv_I_loc_ani*n_inv_J_loc_ani*n_inv_K_loc_ani);


            // ----------- kernel density normalization -----------
            // velocity
            for (int k = 0; k < n_inv_K_loc; k++) {
                for (int j = 0; j < n_inv_J_loc; j++) {
                    for (int i = 0; i < n_inv_I_loc; i++) {
                        if(isZero(grid.Ks_density_inv_loc[I2V_INV_KNL(i,j,k)])){
                            // do nothing
                        } else {
                            if (isNegative(grid.Ks_density_inv_loc[I2V_INV_KNL(i,j,k)])){
                                std::cout   << "Warning: grid.Ks_density_inv_loc[I2V_INV_KNL(" << i << "," << j << "," << k << ")] is less than 0, = " 
                                            << grid.Ks_density_inv_loc[I2V_INV_KNL(i,j,k)]   
                                            << ", using absolute value instead."                   
                                            << std::endl;
                            }
                            grid.Ks_inv_loc[I2V_INV_KNL(i,j,k)] /= std::pow(std::abs(grid.Ks_density_inv_loc[I2V_INV_KNL(i,j,k)]),Kdensity_coe);
                        }
                    }
                }
            }
            // anisotropy
            for (int k = 0; k < n_inv_K_loc_ani; k++) {
                for (int j = 0; j < n_inv_J_loc_ani; j++) {
                    for (int i = 0; i < n_inv_I_loc_ani; i++) {
                        if(isZero(grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)])){
                            // do nothing
                        } else {
                            if (isNegative(grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)])){
                                std::cout   << "Warning: grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(" << i << "," << j << "," << k << ")] is less than 0, = " 
                                            << grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)]   
                                            << ", using absolute value instead."                   
                                            << std::endl;
                            }
                            grid.Keta_inv_loc[I2V_INV_ANI_KNL(i,j,k)] /= std::pow(std::abs(grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)]),Kdensity_coe);
                        }

                        if(isZero(grid.Kxi_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)])){
                            // do nothing
                        } else {
                            if (isNegative(grid.Kxi_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)])){
                                std::cout   << "Warning: grid.Kxi_density_inv_loc[I2V_INV_ANI_KNL(" << i << "," << j << "," << k << ")] is less than 0, = " 
                                            << grid.Kxi_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)]   
                                            << ", using absolute value instead."                   
                                            << std::endl;
                            }
                            grid.Kxi_inv_loc[I2V_INV_ANI_KNL(i,j,k)] /= std::pow(std::abs(grid.Kxi_density_inv_loc[I2V_INV_ANI_KNL(i,j,k)]),Kdensity_coe);
                        }
                    }
                }
            }

            // --------- back-project kernels from inversion grid to fine grid (interpolation) -----------
            for (int k = k_start; k < k_end; k++) {     // loop over depth
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

                for (int j = j_start; j < j_end; j++) {   // loop over latitude
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

                    for (int i = i_start; i < i_end; i++) {  // loop over longitude
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

                        // consider the surrounding 8 inversion nodes for trilinear interpolation
                        CUSTOMREAL pert_Ks   = 0.0;
                        CUSTOMREAL pert_Keta = 0.0;
                        CUSTOMREAL pert_Kxi  = 0.0;
                        CUSTOMREAL pert_Ks_density  = 0.0;
                        CUSTOMREAL pert_Keta_density = 0.0;
                        CUSTOMREAL pert_Kxi_density  = 0.0;

                        pert_Ks             += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt,kdr)];
                        pert_Keta           += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)];
                        pert_Kxi            += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)];
                        pert_Ks_density     += (_1_CR-r_r)*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_density_inv_loc[  I2V_INV_KNL(idp,jdt,kdr)];
                        pert_Keta_density   += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)];
                        pert_Kxi_density    += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani)];

                        pert_Ks             += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt,kdr+1)];
                        pert_Keta           += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)];
                        pert_Kxi            += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)];
                        pert_Ks_density     += r_r*(_1_CR-r_t)*(_1_CR-r_p)*grid.Ks_density_inv_loc[  I2V_INV_KNL(idp,jdt,kdr+1)];
                        pert_Keta_density   += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)];
                        pert_Kxi_density    += r_r_ani*(_1_CR-r_t_ani)*(_1_CR-r_p_ani)*grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani,kdr_ani+1)];
                        
                        pert_Ks             += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr)];
                        pert_Keta           += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)];
                        pert_Kxi            += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)];
                        pert_Ks_density     += (_1_CR-r_r)*r_t*(_1_CR-r_p)*grid.Ks_density_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr)];
                        pert_Keta_density   = (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)];
                        pert_Kxi_density    += (_1_CR-r_r_ani)*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani)];
                        
                        pert_Ks             += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr)];
                        pert_Keta           += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)];
                        pert_Kxi            += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)];
                        pert_Ks_density     += (_1_CR-r_r)*(_1_CR-r_t)*r_p*grid.Ks_density_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr)];
                        pert_Keta_density   += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)];
                        pert_Kxi_density    += (_1_CR-r_r_ani)*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani)];
                        
                        pert_Ks             += r_r*r_t*(_1_CR-r_p)*grid.Ks_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr+1)];
                        pert_Keta           += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)];
                        pert_Kxi            += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)];
                        pert_Ks_density     += r_r*r_t*(_1_CR-r_p)*grid.Ks_density_inv_loc[  I2V_INV_KNL(idp,jdt+1,kdr+1)];
                        pert_Keta_density   += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)];
                        pert_Kxi_density    += r_r_ani*r_t_ani*(_1_CR-r_p_ani)*grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani,jdt_ani+1,kdr_ani+1)];
                        
                        pert_Ks             += r_r*(_1_CR-r_t)*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr+1)];
                        pert_Keta           += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)];
                        pert_Kxi            += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)];
                        pert_Ks_density     += r_r*(_1_CR-r_t)*r_p*grid.Ks_density_inv_loc[  I2V_INV_KNL(idp+1,jdt,kdr+1)];
                        pert_Keta_density   += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)];
                        pert_Kxi_density    += r_r_ani*(_1_CR-r_t_ani)*r_p_ani*grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani,kdr_ani+1)];
                        
                        pert_Ks             += (_1_CR-r_r)*r_t*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr)];
                        pert_Keta           += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)];
                        pert_Kxi            += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)];
                        pert_Ks_density     += (_1_CR-r_r)*r_t*r_p*grid.Ks_density_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr)];
                        pert_Keta_density   += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)];
                        pert_Kxi_density    += (_1_CR-r_r_ani)*r_t_ani*r_p_ani*grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani)];

                        pert_Ks             += r_r*r_t*r_p*grid.Ks_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr+1)];
                        pert_Keta           += r_r_ani*r_t_ani*r_p_ani*grid.Keta_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)];
                        pert_Kxi            += r_r_ani*r_t_ani*r_p_ani*grid.Kxi_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)];
                        pert_Ks_density     += r_r*r_t*r_p*grid.Ks_density_inv_loc[  I2V_INV_KNL(idp+1,jdt+1,kdr+1)];   
                        pert_Keta_density   += r_r_ani*r_t_ani*r_p_ani*grid.Keta_density_inv_loc[I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)];
                        pert_Kxi_density    += r_r_ani*r_t_ani*r_p_ani*grid.Kxi_density_inv_loc[ I2V_INV_ANI_KNL(idp_ani+1,jdt_ani+1,kdr_ani+1)];

                        // update para
                        grid.Ks_processing_loc[             I2V(i_loc,j_loc,k_loc)] += weight * pert_Ks;
                        grid.Keta_processing_loc[           I2V(i_loc,j_loc,k_loc)] += weight * pert_Keta;
                        grid.Kxi_processing_loc[            I2V(i_loc,j_loc,k_loc)] += weight * pert_Kxi;
                        grid.Ks_density_processing_loc[     I2V(i_loc,j_loc,k_loc)] += weight * pert_Ks_density;
                        grid.Keta_density_processing_loc[   I2V(i_loc,j_loc,k_loc)] += weight * pert_Keta_density;
                        grid.Kxi_density_processing_loc[    I2V(i_loc,j_loc,k_loc)] += weight * pert_Kxi_density;

                    } // end for i
                } // end for j
            } // end for k

        } // end i_grid

        // ----------------
        // Here, we have obtained processed kernels
        // grid.Ks_update_loc, grid.Keta_update_loc, grid.Kxi_update_loc
        // grid.Ks_density_update_loc, grid.Keta_density_update_loc, grid.Kxi_density_update_loc
        // ----------------
    }


    // rescale kernel updates to -1 ~ 1
    void kernel_rescaling_to_unit(Grid& grid){
        //
        // rescale kernel update to -1 ~ 1
        //
        // get the scaling factor
        CUSTOMREAL Linf_Ks = _0_CR, Linf_Keta = _0_CR, Linf_Kxi = _0_CR;
        CUSTOMREAL Linf_Ks_den = _0_CR, Linf_Keta_den = _0_CR, Linf_Kxi_den = _0_CR;

        // CUSTOMREAL Linf_Kden = _0_CR;
        for (int k = 1; k < loc_K-1; k++) {
            for (int j = 1; j < loc_J-1; j++) {
                for (int i = 1; i < loc_I-1; i++) {
                    Linf_Ks   = std::max(Linf_Ks,   std::abs(grid.Ks_processing_loc[I2V(i,j,k)]));
                    Linf_Keta = std::max(Linf_Keta, std::abs(grid.Keta_processing_loc[I2V(i,j,k)]));
                    Linf_Kxi  = std::max(Linf_Kxi,  std::abs(grid.Kxi_processing_loc[I2V(i,j,k)]));

                    Linf_Ks_den   = std::max(Linf_Ks_den,   std::abs(grid.Ks_density_processing_loc[I2V(i,j,k)]));
                    Linf_Keta_den = std::max(Linf_Keta_den, std::abs(grid.Keta_density_processing_loc[I2V(i,j,k)]));
                    Linf_Kxi_den  = std::max(Linf_Kxi_den,  std::abs(grid.Kxi_density_processing_loc[I2V(i,j,k)]));
                }
            }
        }

        // get the maximum scaling factor among subdomains
        CUSTOMREAL Linf_tmp;
        allreduce_cr_single_max(Linf_Ks, Linf_tmp);         Linf_Ks = Linf_tmp;
        allreduce_cr_single_max(Linf_Keta, Linf_tmp);       Linf_Keta = Linf_tmp;
        allreduce_cr_single_max(Linf_Kxi, Linf_tmp);        Linf_Kxi = Linf_tmp;
        allreduce_cr_single_max(Linf_Ks_den, Linf_tmp);     Linf_Ks_den = Linf_tmp;
        allreduce_cr_single_max(Linf_Keta_den, Linf_tmp);   Linf_Keta_den = Linf_tmp;
        allreduce_cr_single_max(Linf_Kxi_den, Linf_tmp);    Linf_Kxi_den = Linf_tmp;  

        CUSTOMREAL Linf_all = _0_CR, Linf_all_den = _0_CR;
        Linf_all = std::max(Linf_Ks, std::max(Linf_Keta, Linf_Kxi));
        Linf_all_den = std::max(Linf_Ks_den, std::max(Linf_Keta_den, Linf_Kxi_den));

        // rescale the kernel processing
        for (int k = 0; k < loc_K; k++) {
            for (int j = 0; j < loc_J; j++) {
                for (int i = 0; i < loc_I; i++) {
                    grid.Ks_processing_loc[I2V(i,j,k)]      /= Linf_all;
                    grid.Keta_processing_loc[I2V(i,j,k)]    /= Linf_all;
                    grid.Kxi_processing_loc[I2V(i,j,k)]     /= Linf_all;

                    grid.Ks_density_processing_loc[I2V(i,j,k)]    /= Linf_all_den;
                    grid.Keta_density_processing_loc[I2V(i,j,k)]  /= Linf_all_den;
                    grid.Kxi_density_processing_loc[I2V(i,j,k)]   /= Linf_all_den;
                }
            }
        }
    }


}