#ifndef ITERATOR_H
#define ITERATOR_H

#include <algorithm>
#include <initializer_list>
#include <vector>
#include <math.h>
#include "grid.h"
#include "input_params.h"
#include "utils.h"
#include "source.h"
#include "io.h"
#include "timer.h"
#include "eikonal_solver_2d.h"

#ifdef USE_CUDA
#include "grid_wrapper.cuh"
#include "iterator_wrapper.cuh"
#endif

//#ifdef USE_SIMD
#include "simd_conf.h"
//#endif

class Iterator {
public:
    Iterator(InputParams&, Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
    virtual ~Iterator();
    // regional source
    void run_iteration_forward(InputParams&, Grid&, IO_utils&, bool&); // run forward iteration till convergence
    void run_iteration_adjoint(InputParams&, Grid&, IO_utils&, int);        // run adjoint iteration till convergence

    void initialize_arrays(InputParams&, IO_utils&, Grid&, Source&, const std::string&); // initialize factors etc.

protected:
    void assign_processes_for_levels(Grid&, InputParams&); // assign intra-node processes for each sweeping level
    void set_sweep_direction(int);                         // set sweep direction
    // regional source
    virtual void do_sweep(int, Grid&, InputParams&){};                // do sweeping with ordinal method
    void calculate_stencil_1st_order(Grid&, int&, int&, int&);        // calculate stencil for 1st order
    void calculate_stencil_3rd_order(Grid&, int&, int&, int&);        // calculate stencil for 3rd order
    void calculate_stencil_1st_order_upwind(Grid&, int&, int&, int&); // calculate stencil for 1st order in upwind form
    void calculate_boundary_nodes(Grid&);                             // calculate boundary values
    // teleseismic source
    void calculate_stencil_1st_order_tele(Grid&, int&, int&, int&);        // calculate stencil for 1st order
    void calculate_stencil_3rd_order_tele(Grid&, int&, int&, int&);        // calculate stencil for 3rd order
    void calculate_stencil_1st_order_upwind_tele(Grid&, int&, int&, int&); // calculate stencil for 1st order in upwind form
    void calculate_boundary_nodes_tele(Grid&, int&, int&, int&);           // calculate boundary values for teleseismic source

    void calculate_boundary_nodes_adj(Grid&, int&, int&, int&);       // calculate boundary values for adjoint source (all zeros)

    // Hamiltonian calculation
    inline CUSTOMREAL calc_LF_Hamiltonian(Grid&, CUSTOMREAL& ,CUSTOMREAL& , \
                                                 CUSTOMREAL& ,CUSTOMREAL& , \
                                                 CUSTOMREAL& ,CUSTOMREAL&, \
                                                 int&, int&, int& );
    inline CUSTOMREAL calc_LF_Hamiltonian_tele(Grid&, CUSTOMREAL& ,CUSTOMREAL& , \
                                                 CUSTOMREAL& ,CUSTOMREAL& , \
                                                 CUSTOMREAL& ,CUSTOMREAL&, \
                                                 int&, int&, int& );

    // methods for adjoint field calculation
    void init_delta_and_Tadj(Grid&, InputParams&);                     // initialize delta and Tadj
    void init_delta_and_Tadj_density(Grid&, InputParams&);             // initialize delta and Tadj_density
    void fix_boundary_Tadj(Grid&);                                     // fix boundary values for Tadj
    virtual void do_sweep_adj(int, Grid&, InputParams&){};             // do sweeping with ordinal method for adjoint field
    void calculate_stencil_adj(Grid&, int&, int&, int&);               // calculate stencil for 1st order for adjoint field

    // grid point information
    int* _nr, *_nt, *_np;           // number of grid points on the direction r, theta, phi
    CUSTOMREAL* _dr, *_dt, *_dp;    // grid spacing on the direction r, theta, phi
    int nr, nt, np;                 // number of grid points on the direction r, theta, phi
    CUSTOMREAL dr, dt, dp;          // grid spacing on the direction r, theta, phi
    int st_level;                   // start level for sweeping
    int ed_level;                   // end level for sweeping
    std::vector< std::vector<int> > ijk_for_this_subproc; // ijk=I2V(i,j,k) for this process (level, ijk)
    int max_n_nodes_plane;                                // maximum number of nodes on a plane


#if defined USE_SIMD || defined USE_CUDA
    // stencil dumps
    // first orders
    CUSTOMREAL *dump_c__;// center of C
    // all grid data expect tau pre-load strategy (iswap, ilevel, inodes)
#if USE_AVX512 || USE_AVX || USE_NEON || defined USE_CUDA
    std::vector<std::vector<int*>> vv_i__j__k__, vv_ip1j__k__, vv_im1j__k__, vv_i__jp1k__, vv_i__jm1k__, vv_i__j__kp1, vv_i__j__km1;
    std::vector<std::vector<int*>>               vv_ip2j__k__, vv_im2j__k__, vv_i__jp2k__, vv_i__jm2k__, vv_i__j__kp2, vv_i__j__km2;
#elif USE_ARM_SVE
    std::vector<std::vector<uint64_t*>> vv_i__j__k__, vv_ip1j__k__, vv_im1j__k__, vv_i__jp1k__, vv_i__jm1k__, vv_i__j__kp1, vv_i__j__km1;
    std::vector<std::vector<uint64_t*>>               vv_ip2j__k__, vv_im2j__k__, vv_i__jp2k__, vv_i__jm2k__, vv_i__j__kp2, vv_i__j__km2;
#endif
    std::vector<std::vector<CUSTOMREAL*>> vv_iip, vv_jjt, vv_kkr;

    std::vector<std::vector<CUSTOMREAL*>> vv_fac_a, vv_fac_b, vv_fac_c, vv_fac_f, vv_T0v, vv_T0r, vv_T0t, vv_T0p, vv_fun;
    std::vector<std::vector<CUSTOMREAL*>>vv_change;
    std::vector<std::vector<bool*>>vv_change_bl;

    template <typename T>
    void preload_indices(std::vector<std::vector<T*>> &vi, std::vector<std::vector<T*>> &, std::vector<std::vector<T*>> &, int, int, int);
    template <typename T>
    void preload_indices_1d(std::vector<std::vector<T*>> &, int, int, int);
    template <typename T>
    std::vector<std::vector<CUSTOMREAL*>> preload_array(T* a);
    std::vector<std::vector<bool*>> preload_array_bl(bool* a);
    template <typename T>
    void free_preloaded_array(std::vector<std::vector<T*>> &vvv){
        for (int iswap = 0; iswap < 8; iswap++){
            for (auto& vv : vvv.at(iswap)) free(vv);
        }
    }
    // flag for deallocation
    bool simd_allocated     = false;
    bool simd_allocated_3rd = false;

#endif // USE_SIMD || USE_CUDA

#ifdef USE_CUDA
    Grid_on_device *gpu_grid;
#endif


    const int nswp = 8;          // number of sweeping directions
    int r_dirc, t_dirc, p_dirc;  // sweeping directions
    CUSTOMREAL sigr, sigt, sigp; //
    CUSTOMREAL coe;
    CUSTOMREAL wp1, pp1, wp2, pp2;
    CUSTOMREAL wt1, pt1, wt2, pt2;
    CUSTOMREAL wr1, pr1, wr2, pr2;
    CUSTOMREAL Htau, tpT;

    CUSTOMREAL ap1, bp1, ap2, bp2, ap, bp;
    CUSTOMREAL at1, bt1, at2, bt2, at, bt;
    CUSTOMREAL ar1, br1, ar2, br2, ar, br;

    CUSTOMREAL bc_f2, eqn_a, eqn_b, eqn_c, eqn_Delta;
    CUSTOMREAL tmp_tau, tmp_T;
    CUSTOMREAL T_r, T_t, T_p, charact_r, charact_t, charact_p;
    bool is_causality;
    int count_cand;
    std::vector<CUSTOMREAL> canditate = std::vector<CUSTOMREAL>(60);


    // iteration control
    int iter_count = 0;
    CUSTOMREAL ini_diff_L1 = HUGE_VAL, ini_diff_Linf = HUGE_VAL;
    CUSTOMREAL ini_err_L1  = HUGE_VAL, ini_err_Linf  = HUGE_VAL;
    CUSTOMREAL cur_diff_L1 = HUGE_VAL, cur_diff_Linf = HUGE_VAL;
    CUSTOMREAL cur_err_L1  = HUGE_VAL, cur_err_Linf  = HUGE_VAL;

    // teleseismic flag
    bool is_teleseismic = false;

    // second run for hybrid order method
    bool is_second_run = false;

};

// define derived classes for each iteration scheme


#endif // ITERATOR_H