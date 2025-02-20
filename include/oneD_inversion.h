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
    CUSTOMREAL      *T0v_1dinv;            // T0v
    CUSTOMREAL      *T0r_1dinv;            // T0r
    CUSTOMREAL      *T0t_1dinv;            // T0t
    bool            *is_changed_1dinv;     // is_changed

    // parameters on grid nodes (for inversion)
    CUSTOMREAL      *Ks_1dinv;             // Ks


    // functions
    std::vector<CUSTOMREAL> run_simulation_one_step_1dinv();
    void model_optimize_1dinv(); 

private:

    void generate_2d_mesh(InputParams&);
    void deallocate_arrays();
    void allocate_arrays();
    void load_1d_model(Grid&);
    int I2V_1DINV(const int&,const int&);
};

#endif // ONED_INVERSION_H