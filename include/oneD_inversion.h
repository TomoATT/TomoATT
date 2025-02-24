#ifndef ONED_INVERSION_H
#define ONED_INVERSION_H

#include <fstream>
#include <string>
#include <iostream>

#include "input_params.h"
#include "grid.h"
#include "utils.h"
#include "config.h"
class OneDInversion {

public:
    OneDInversion(InputParams&, Grid&);
    ~OneDInversion();

    // grid parameters
    int             nr_1dinv, nt_1dinv;    // number of grid points in r and epicenter distance
    CUSTOMREAL      *r_1dinv, *t_1dinv;    // coordinates
    CUSTOMREAL      dr_1dinv, dt_1dinv;    // grid spacing

    // parameters on grid nodes (for model)
    CUSTOMREAL      *slowness_1dinv;       // slowness
    CUSTOMREAL      *fac_a_1dinv;          // fac_a
    CUSTOMREAL      *fac_b_1dinv;          // fac_b

    // parameters on grid nodes (for eikonal solver and adjoint solver)
    CUSTOMREAL      *tau_1dinv;            // tau
    CUSTOMREAL      *tau_old_1dinv;        // tau_old
    CUSTOMREAL      *T_1dinv;              // T
    CUSTOMREAL      *Tadj_1dinv;           // Tadj
    CUSTOMREAL      *Tadj_density_1dinv;           // Tadj
    CUSTOMREAL      *T0v_1dinv;            // T0v
    CUSTOMREAL      *T0r_1dinv;            // T0r
    CUSTOMREAL      *T0t_1dinv;            // T0t
    bool            *is_changed_1dinv;     // is_changed
    CUSTOMREAL      *delta_1dinv;          // delta

    // parameters on grid nodes (for inversion)
    CUSTOMREAL      *Ks_1dinv;              // Ks
    CUSTOMREAL      *Ks_density_1dinv;      // Ks_density
    CUSTOMREAL      *Ks_over_Kden_1dinv;    // Ks_over_Kdensity
    CUSTOMREAL      *Ks_multigrid;          // (Ks_over_Kdensity) parameterized by multigrid
    CUSTOMREAL      *Ks_multigrid_previous; // Ks_multigrid at previous iteration 
    CUSTOMREAL      *Ks_update;             // Ks update at current iteration (rescaled to be [-1,1])

    // parameters for optimization
    CUSTOMREAL      v_obj;
    CUSTOMREAL      old_v_obj;

    // functions
    std::vector<CUSTOMREAL> run_simulation_one_step_1dinv(InputParams&);
    void model_optimize_1dinv(Grid&, const int&); 

private:

    // for eikonal solver
    int count_cand;
    int ii,ii_nr,ii_n2r,ii_pr,ii_p2r,ii_nt,ii_n2t,ii_pt,ii_p2t;
    CUSTOMREAL ar,br,at,bt,ar1,ar2,at1,at2,br1,br2,bt1,bt2;
    CUSTOMREAL eqn_a, eqn_b, eqn_c, eqn_Delta;
    CUSTOMREAL tmp_tau, T_r, T_t, charact_r, charact_t;
    bool is_causality;
    std::vector<CUSTOMREAL> canditate = std::vector<CUSTOMREAL>(60);


    // member functions
    // class functions
    void generate_2d_mesh(InputParams&);
    void deallocate_arrays();
    void allocate_arrays();
    void load_1d_model(Grid&);
    int I2V_1DINV(const int&,const int&);
    
    // "run_simulation_one_step_1dinv" subfunctions:
    void eikonal_solver_2d(InputParams&, int& );
    void initialize_eikonal_array(CUSTOMREAL);
    void FSM_2d();
    void calculate_stencil(const int&, const int&);
    
    void calculate_synthetic_traveltime_and_adjoint_source(InputParams&, int& );
    CUSTOMREAL interpolate_2d_traveltime(const CUSTOMREAL&, const CUSTOMREAL&);

    void adjoint_solver_2d(InputParams&, const int&, const int&);
    void initialize_adjoint_array(InputParams&, const int&, const int&);
    void FSM_2d_adjoint(const int&);
    void calculate_stencil_adj(const int&, const int&);

    void initialize_kernel_1d();
    void calculate_kernel_1d();
    std::vector<CUSTOMREAL> calculate_obj_and_residual_1dinv(InputParams&);

    // "model_optimize_1dinv" subfunctions:
    void kernel_processing_1dinv(Grid&);
    void density_normalization_1dinv();
    void multi_grid_parameterization_1dinv(Grid&);
    void model_update_1dinv(const int&);
    void determine_step_size_1dinv(const int&);
    CUSTOMREAL norm_1dinv(const CUSTOMREAL*, const int&);
    void generate_3d_model(Grid&);
};

#endif // ONED_INVERSION_H