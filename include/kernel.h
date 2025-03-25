#ifndef KERNEL_H
#define KERNEL_H

#include "config.h"
#include "grid.h"
#include "input_params.h"

void calculate_sensitivity_kernel(Grid& grid, InputParams& IP, const std::string& name_sim_src){
    // calculate sensitivity kernel

    // kernel calculation will be done only by the subdom_main
    if (subdom_main) {
        // get the necessary parameters
        int np             = loc_I;
        int nt             = loc_J;
        int nr             = loc_K;
        CUSTOMREAL dr      = grid.dr;
        CUSTOMREAL dt      = grid.dt;
        CUSTOMREAL dp      = grid.dp;
        CUSTOMREAL src_lon = IP.get_src_lon(   name_sim_src);
        CUSTOMREAL src_lat = IP.get_src_lat(   name_sim_src);
        CUSTOMREAL src_r   = IP.get_src_radius(name_sim_src);

        CUSTOMREAL weight   = _1_CR;

        // inner points
        for (int kkr = 1; kkr < nr-1; kkr++) {
            for (int jjt = 1; jjt < nt-1; jjt++) {
                for (int iip = 1; iip < np-1; iip++) {

                    // calculate the kernel
                    CUSTOMREAL Tr_km     = (grid.T_loc[I2V(iip,jjt,kkr+1)] - grid.T_loc[I2V(iip,jjt,kkr-1)]) / (_2_CR * dr);
                    CUSTOMREAL Ttheta_km = (grid.T_loc[I2V(iip,jjt+1,kkr)] - grid.T_loc[I2V(iip,jjt-1,kkr)]) / (_2_CR * dt) / grid.r_loc_1d[kkr];
                    CUSTOMREAL Tphi_km   = (grid.T_loc[I2V(iip+1,jjt,kkr)] - grid.T_loc[I2V(iip-1,jjt,kkr)]) / (_2_CR * dp) / (grid.r_loc_1d[kkr]*std::cos(grid.t_loc_1d[jjt]));
                    
                    CUSTOMREAL azi_ratio = std::sqrt((my_square(Ttheta_km) + my_square(Tphi_km))/(my_square(Tr_km) + my_square(Ttheta_km) + my_square(Tphi_km)));
                    
                    // density of ks
                    grid.Ks_density_loc[I2V(iip,jjt,kkr)]   += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)];  
                    
                    // density of kxi
                    // grid.Kxi_density_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)]; 
                    grid.Kxi_density_loc[I2V(iip,jjt,kkr)]  += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)] * azi_ratio;
                                                            

                    // density of keta
                    // grid.Keta_density_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)]; 
                    grid.Keta_density_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_density_loc[I2V(iip,jjt,kkr)] * azi_ratio;
                                                            
                    // mask within one grid around the source
                    if (std::abs(grid.r_loc_1d[kkr]-src_r)   >= dr \
                     || std::abs(grid.t_loc_1d[jjt]-src_lat) >= dt \
                     || std::abs(grid.p_loc_1d[iip]-src_lon) >= dp) {
                        

                        if (IP.get_update_slowness()==1){      // we need to update slowness
                            // Kernel w r t slowness s
                            grid.Ks_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] * my_square(grid.fun_loc[I2V(iip,jjt,kkr)]);                                                  
                        } else {
                            grid.Ks_loc[I2V(iip,jjt,kkr)] = _0_CR;
                        }


                        if (IP.get_update_azi_ani()){      // we need to update azimuthal anisotropy
                            // Kernel w r t anisotrophy xi
                            if (isZero(std::sqrt(my_square(grid.xi_loc[I2V(iip,jjt,kkr)])+my_square(grid.eta_loc[I2V(iip,jjt,kkr)])))) {
                                grid.Kxi_loc[I2V(iip,jjt,kkr)]  += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] \
                                                                * (my_square(Ttheta_km) - my_square(Tphi_km));

                                grid.Keta_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] \
                                                                * ( -_2_CR * Ttheta_km * Tphi_km );

                            } else {
                                grid.Kxi_loc[I2V(iip,jjt,kkr)]  += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] \
                                                            * ((- GAMMA * grid.xi_loc[I2V(iip,jjt,kkr)] / \
                                                                        std::sqrt(my_square(grid.xi_loc[ I2V(iip,jjt,kkr)]) + my_square(grid.eta_loc[I2V(iip,jjt,kkr)]))) * my_square(Tr_km) \
                                                                + my_square(Ttheta_km) 
                                                                - my_square(Tphi_km));

                                grid.Keta_loc[I2V(iip,jjt,kkr)] += weight * grid.Tadj_loc[I2V(iip,jjt,kkr)] \
                                                            * (( - GAMMA * grid.eta_loc[I2V(iip,jjt,kkr)]/ \
                                                                            std::sqrt(my_square(grid.xi_loc[I2V(iip,jjt,kkr)]) + my_square(grid.eta_loc[I2V(iip,jjt,kkr)]))) * my_square(Tr_km) \
                                                                    - _2_CR * Ttheta_km * Tphi_km );
                            }
                        } else {
                            grid.Kxi_loc[I2V(iip,jjt,kkr)]  = _0_CR;
                            grid.Keta_loc[I2V(iip,jjt,kkr)] = _0_CR;
                        }

                    } else{
                        grid.Ks_loc[I2V(iip,jjt,kkr)]   += _0_CR;
                        grid.Kxi_loc[I2V(iip,jjt,kkr)]  += _0_CR;
                        grid.Keta_loc[I2V(iip,jjt,kkr)] += _0_CR;

                    }
                }
            }
        }

        // boundary
        for (int kkr = 0; kkr < nr; kkr++) {
            for (int jjt = 0; jjt < nt; jjt++) {
                // set Ks Kxi Keta to zero
                if (grid.i_first()){
                    grid.Ks_loc[I2V(0,jjt,kkr)]             = _0_CR;
                    grid.Kxi_loc[I2V(0,jjt,kkr)]            = _0_CR;
                    grid.Keta_loc[I2V(0,jjt,kkr)]           = _0_CR;
                    grid.Ks_density_loc[I2V(0,jjt,kkr)]     = _0_CR;
                    grid.Kxi_density_loc[I2V(0,jjt,kkr)]    = _0_CR;
                    grid.Keta_density_loc[I2V(0,jjt,kkr)]   = _0_CR;
                }
                if (grid.i_last()){
                    grid.Ks_loc[I2V(np-1,jjt,kkr)]          = _0_CR;
                    grid.Kxi_loc[I2V(np-1,jjt,kkr)]         = _0_CR;
                    grid.Keta_loc[I2V(np-1,jjt,kkr)]        = _0_CR;
                    grid.Ks_density_loc[I2V(np-1,jjt,kkr)]  = _0_CR;
                    grid.Kxi_density_loc[I2V(np-1,jjt,kkr)] = _0_CR;
                    grid.Keta_density_loc[I2V(np-1,jjt,kkr)]= _0_CR;
                }
           }
        }
        for (int kkr = 0; kkr < nr; kkr++) {
            for (int iip = 0; iip < np; iip++) {
                // set Ks Kxi Keta to zero
                if (grid.j_first()){
                    grid.Ks_loc[I2V(iip,0,kkr)]             = _0_CR;
                    grid.Kxi_loc[I2V(iip,0,kkr)]            = _0_CR;
                    grid.Keta_loc[I2V(iip,0,kkr)]           = _0_CR;
                    grid.Ks_density_loc[I2V(iip,0,kkr)]     = _0_CR;
                    grid.Kxi_density_loc[I2V(iip,0,kkr)]    = _0_CR;
                    grid.Keta_density_loc[I2V(iip,0,kkr)]   = _0_CR;
                }
                if (grid.j_last()){
                    grid.Ks_loc[I2V(iip,nt-1,kkr)]          = _0_CR;
                    grid.Kxi_loc[I2V(iip,nt-1,kkr)]         = _0_CR;
                    grid.Keta_loc[I2V(iip,nt-1,kkr)]        = _0_CR;
                    grid.Ks_density_loc[I2V(iip,nt-1,kkr)]  = _0_CR;
                    grid.Kxi_density_loc[I2V(iip,nt-1,kkr)] = _0_CR;
                    grid.Keta_density_loc[I2V(iip,nt-1,kkr)]= _0_CR;
                }
            }
        }
        for (int jjt = 0; jjt < nt; jjt++) {
            for (int iip = 0; iip < np; iip++) {
                // set Ks Kxi Keta to zero
                if (grid.k_first()){
                    grid.Ks_loc[I2V(iip,jjt,0)]             = _0_CR;
                    grid.Kxi_loc[I2V(iip,jjt,0)]            = _0_CR;
                    grid.Keta_loc[I2V(iip,jjt,0)]           = _0_CR;
                    grid.Ks_density_loc[I2V(iip,jjt,0)]     = _0_CR;
                    grid.Kxi_density_loc[I2V(iip,jjt,0)]    = _0_CR;
                    grid.Keta_density_loc[I2V(iip,jjt,0)]   = _0_CR;
                }
                if (grid.k_last()){
                    grid.Ks_loc[I2V(iip,jjt,nr-1)]          = _0_CR;
                    grid.Kxi_loc[I2V(iip,jjt,nr-1)]         = _0_CR;
                    grid.Keta_loc[I2V(iip,jjt,nr-1)]        = _0_CR;
                    grid.Ks_density_loc[I2V(iip,jjt,nr-1)]  = _0_CR;
                    grid.Kxi_density_loc[I2V(iip,jjt,nr-1)] = _0_CR;
                    grid.Keta_density_loc[I2V(iip,jjt,nr-1)]= _0_CR;
                }
            }
        }

    } // end if subdom_main
}


void check_kernel_density(Grid& grid, InputParams& IP) {
    if(subdom_main){
        // check local kernel density positivity
        for (int i_loc = 0; i_loc < loc_I; i_loc++) {
            for (int j_loc = 0; j_loc < loc_J; j_loc++) {
                for (int k_loc = 0; k_loc < loc_K; k_loc++) {
                    if (grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)] < 0){
                        std::cout   << "Warning, id_sim: " << id_sim << ", grid.Ks_density_loc[I2V(" << i_loc << "," << j_loc << "," << k_loc << ")] is less than 0, = " 
                                    << grid.Ks_density_loc[I2V(i_loc,j_loc,k_loc)]   
                                    << std::endl;
                        // print the source name
                        for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
                            // get source info
                            const std::string name_sim_src   = IP.get_src_name(i_src);
                            std::cout   << "id_sim: " << id_sim << ", "
                                        << i_src + 1 << "/" << IP.n_src_this_sim_group << ", source name: " << name_sim_src << std::endl;
                        }
                    }
                }
            }
        }
    }
}


void sumup_kernels(Grid& grid) {
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

#endif