#ifndef GRID_WRAPPER_CUH
#define GRID_WRAPPER_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <vector>
#include <iostream>

//#include "config.h"
#include "cuda_constants.cuh"
#include "cuda_utils.cuh"

// structure for storing grid information on device
typedef struct Grid_on_device {

    // parameters
    int loc_I_host, loc_J_host, loc_K_host;
    int n_nodes_total_host;
    int n_nodes_max_host=0;
    int n_levels_host;
    CUSTOMREAL dr_host, dt_host, dp_host;

    // index storage
    int* n_nodes_on_levels, *n_nodes_on_levels_host;

    int* vv_i__j__k___0, *vv_i__j__k___1, *vv_i__j__k___2, *vv_i__j__k___3, *vv_i__j__k___4, *vv_i__j__k___5, *vv_i__j__k___6, *vv_i__j__k___7;
    int* vv_ip1j__k___0, *vv_ip1j__k___1, *vv_ip1j__k___2, *vv_ip1j__k___3, *vv_ip1j__k___4, *vv_ip1j__k___5, *vv_ip1j__k___6, *vv_ip1j__k___7;
    int* vv_im1j__k___0, *vv_im1j__k___1, *vv_im1j__k___2, *vv_im1j__k___3, *vv_im1j__k___4, *vv_im1j__k___5, *vv_im1j__k___6, *vv_im1j__k___7;
    int* vv_i__jp1k___0, *vv_i__jp1k___1, *vv_i__jp1k___2, *vv_i__jp1k___3, *vv_i__jp1k___4, *vv_i__jp1k___5, *vv_i__jp1k___6, *vv_i__jp1k___7;
    int* vv_i__jm1k___0, *vv_i__jm1k___1, *vv_i__jm1k___2, *vv_i__jm1k___3, *vv_i__jm1k___4, *vv_i__jm1k___5, *vv_i__jm1k___6, *vv_i__jm1k___7;
    int* vv_i__j__kp1_0, *vv_i__j__kp1_1, *vv_i__j__kp1_2, *vv_i__j__kp1_3, *vv_i__j__kp1_4, *vv_i__j__kp1_5, *vv_i__j__kp1_6, *vv_i__j__kp1_7;
    int* vv_i__j__km1_0, *vv_i__j__km1_1, *vv_i__j__km1_2, *vv_i__j__km1_3, *vv_i__j__km1_4, *vv_i__j__km1_5, *vv_i__j__km1_6, *vv_i__j__km1_7;
    int* vv_ip2j__k___0, *vv_ip2j__k___1, *vv_ip2j__k___2, *vv_ip2j__k___3, *vv_ip2j__k___4, *vv_ip2j__k___5, *vv_ip2j__k___6, *vv_ip2j__k___7;
    int* vv_im2j__k___0, *vv_im2j__k___1, *vv_im2j__k___2, *vv_im2j__k___3, *vv_im2j__k___4, *vv_im2j__k___5, *vv_im2j__k___6, *vv_im2j__k___7;
    int* vv_i__jp2k___0, *vv_i__jp2k___1, *vv_i__jp2k___2, *vv_i__jp2k___3, *vv_i__jp2k___4, *vv_i__jp2k___5, *vv_i__jp2k___6, *vv_i__jp2k___7;
    int* vv_i__jm2k___0, *vv_i__jm2k___1, *vv_i__jm2k___2, *vv_i__jm2k___3, *vv_i__jm2k___4, *vv_i__jm2k___5, *vv_i__jm2k___6, *vv_i__jm2k___7;
    int* vv_i__j__kp2_0, *vv_i__j__kp2_1, *vv_i__j__kp2_2, *vv_i__j__kp2_3, *vv_i__j__kp2_4, *vv_i__j__kp2_5, *vv_i__j__kp2_6, *vv_i__j__kp2_7;
    int* vv_i__j__km2_0, *vv_i__j__km2_1, *vv_i__j__km2_2, *vv_i__j__km2_3, *vv_i__j__km2_4, *vv_i__j__km2_5, *vv_i__j__km2_6, *vv_i__j__km2_7;

    // constants
    CUSTOMREAL* vv_fac_a_0, *vv_fac_a_1, *vv_fac_a_2, *vv_fac_a_3, *vv_fac_a_4, *vv_fac_a_5, *vv_fac_a_6, *vv_fac_a_7;
    CUSTOMREAL* vv_fac_b_0, *vv_fac_b_1, *vv_fac_b_2, *vv_fac_b_3, *vv_fac_b_4, *vv_fac_b_5, *vv_fac_b_6, *vv_fac_b_7;
    CUSTOMREAL* vv_fac_c_0, *vv_fac_c_1, *vv_fac_c_2, *vv_fac_c_3, *vv_fac_c_4, *vv_fac_c_5, *vv_fac_c_6, *vv_fac_c_7;
    CUSTOMREAL* vv_fac_f_0, *vv_fac_f_1, *vv_fac_f_2, *vv_fac_f_3, *vv_fac_f_4, *vv_fac_f_5, *vv_fac_f_6, *vv_fac_f_7;
    CUSTOMREAL* vv_T0v_0, *vv_T0v_1, *vv_T0v_2, *vv_T0v_3, *vv_T0v_4, *vv_T0v_5, *vv_T0v_6, *vv_T0v_7;
    CUSTOMREAL* vv_T0r_0, *vv_T0r_1, *vv_T0r_2, *vv_T0r_3, *vv_T0r_4, *vv_T0r_5, *vv_T0r_6, *vv_T0r_7;
    CUSTOMREAL* vv_T0t_0, *vv_T0t_1, *vv_T0t_2, *vv_T0t_3, *vv_T0t_4, *vv_T0t_5, *vv_T0t_6, *vv_T0t_7;
    CUSTOMREAL* vv_T0p_0, *vv_T0p_1, *vv_T0p_2, *vv_T0p_3, *vv_T0p_4, *vv_T0p_5, *vv_T0p_6, *vv_T0p_7;
    CUSTOMREAL* vv_fun_0, *vv_fun_1, *vv_fun_2, *vv_fun_3, *vv_fun_4, *vv_fun_5, *vv_fun_6, *vv_fun_7;
    bool*       vv_change_0, *vv_change_1, *vv_change_2, *vv_change_3, *vv_change_4, *vv_change_5, *vv_change_6, *vv_change_7;

    // temporary variables
    CUSTOMREAL* tau;

    // full-grid (global) T0v, indexed by flattened global index. Used by UPWIND line-case neighbor causality.
    CUSTOMREAL* T0v_glob = nullptr;

    // full-grid (global) fields for the adjoint solve (adjoint field itself reuses `tau`)
    CUSTOMREAL* T_glob = nullptr;        // current traveltime field T
    CUSTOMREAL* zeta_glob = nullptr;     // anisotropy factor zeta
    CUSTOMREAL* xi_glob = nullptr;       // anisotropy factor xi
    CUSTOMREAL* eta_glob = nullptr;      // anisotropy factor eta
    CUSTOMREAL* tau_old_glob = nullptr;  // tau_old (delta field used as adjoint source)
    // 1-D spherical coordinate factor arrays for the adjoint stencil
    CUSTOMREAL* adj_one_over_r = nullptr;      // one_over_r_loc_1d     [loc_K]
    CUSTOMREAL* adj_one_over_r_sq = nullptr;   // one_over_r_loc_1d_sq  [loc_K]
    CUSTOMREAL* adj_one_over_cos_t = nullptr;  // one_over_cos_t_loc    [loc_J]
    CUSTOMREAL* adj_one_over_cos_t_sq = nullptr;// one_over_cos_t_loc_sq [loc_J]
    CUSTOMREAL* adj_sin_t = nullptr;           // sin_t_loc             [loc_J]
    CUSTOMREAL* adj_cos_t_m0p5 = nullptr;      // cos_t_loc_m0p5        [loc_J]
    CUSTOMREAL* adj_cos_t_p0p5 = nullptr;      // cos_t_loc_p0p5        [loc_J]
    bool adj_static_loaded = false;

    bool if_3rd_order = false;

    // thead and grid for sweeping
    dim3 grid_sweep_host, threads_sweep_host;
    // array of streams
    cudaStream_t* level_streams;


} Grid_on_device;


void cuda_initialize_grid_1st(std::vector< std::vector<int> >& ijk, Grid_on_device* grid_dv, int const& loc_I, int const& loc_J, int const& loc_K,
                CUSTOMREAL const& dp, CUSTOMREAL const& dt, CUSTOMREAL const& dr, \
                std::vector<std::vector<int*>>        & vv_i__j__k__, \
                std::vector<std::vector<int*>>        & vv_ip1j__k__, \
                std::vector<std::vector<int*>>        & vv_im1j__k__, \
                std::vector<std::vector<int*>>        & vv_i__jp1k__, \
                std::vector<std::vector<int*>>        & vv_i__jm1k__, \
                std::vector<std::vector<int*>>        & vv_i__j__kp1, \
                std::vector<std::vector<int*>>        & vv_i__j__km1, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fac_a, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fac_b, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fac_c, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fac_f, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_T0v, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_T0r, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_T0t, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_T0p, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fun, \
                std::vector<std::vector<bool*>>       & vv_change);

void cuda_initialize_grid_3rd(std::vector< std::vector<int> >& ijk, Grid_on_device* grid_dv, int const& loc_I, int const& loc_J, int const& loc_K,
                CUSTOMREAL const& dp, CUSTOMREAL const& dt, CUSTOMREAL const& dr, \
                std::vector<std::vector<int*>>        & vv_i__j__k__, \
                std::vector<std::vector<int*>>        & vv_ip1j__k__, \
                std::vector<std::vector<int*>>        & vv_im1j__k__, \
                std::vector<std::vector<int*>>        & vv_i__jp1k__, \
                std::vector<std::vector<int*>>        & vv_i__jm1k__, \
                std::vector<std::vector<int*>>        & vv_i__j__kp1, \
                std::vector<std::vector<int*>>        & vv_i__j__km1, \
                std::vector<std::vector<int*>>        & vv_ip2j__k__, \
                std::vector<std::vector<int*>>        & vv_im2j__k__, \
                std::vector<std::vector<int*>>        & vv_i__jp2k__, \
                std::vector<std::vector<int*>>        & vv_i__jm2k__, \
                std::vector<std::vector<int*>>        & vv_i__j__kp2, \
                std::vector<std::vector<int*>>        & vv_i__j__km2, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fac_a, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fac_b, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fac_c, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fac_f, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_T0v, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_T0r, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_T0t, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_T0p, \
                std::vector<std::vector<CUSTOMREAL*>> & vv_fun, \
                std::vector<std::vector<bool*>>       & vv_change);


// finalize
void cuda_finalize_grid(Grid_on_device* grid_dv);

// copy tau from host to device
void cuda_copy_tau_to_device(Grid_on_device* grid_dv, CUSTOMREAL* tau_h);
// copy tau from device to host
void cuda_copy_tau_to_host(Grid_on_device* grid_dv, CUSTOMREAL* tau_h);
// copy full-grid (global) T0v to device (allocate on first call). Used by UPWIND solver.
void cuda_copy_T0v_glob_to_device(Grid_on_device* grid_dv, CUSTOMREAL* T0v_h);

// adjoint solve helpers (full-grid globals; allocate on first call)
void cuda_copy_adj_static_to_device(Grid_on_device* grid_dv,
                                    CUSTOMREAL* zeta_h, CUSTOMREAL* xi_h, CUSTOMREAL* eta_h,
                                    CUSTOMREAL* oor_h, CUSTOMREAL* oor_sq_h,
                                    CUSTOMREAL* ooc_h, CUSTOMREAL* ooc_sq_h,
                                    CUSTOMREAL* sint_h, CUSTOMREAL* cos_m_h, CUSTOMREAL* cos_p_h,
                                    int n_total, int nr, int nt);
void cuda_copy_adj_T_to_device(Grid_on_device* grid_dv, CUSTOMREAL* T_h);
void cuda_copy_adj_tau_old_to_device(Grid_on_device* grid_dv, CUSTOMREAL* tau_old_h);

#endif // GRID_WRAPPER_CUH