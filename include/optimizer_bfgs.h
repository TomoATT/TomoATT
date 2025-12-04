#ifndef OPTIMIZER_BFGS_H
#define OPTIMIZER_BFGS_H

#include "optimizer.h"  

class Optimizer_bfgs : public Optimizer {
public:
    Optimizer_bfgs();
    ~Optimizer_bfgs();

private:

    const int Mbfgs = 5; // number of previous steps to store
    int n_total_loc_grid_points;    // number of local grid points

    // historical model and gradient
    CUSTOMREAL* array_3d_forward;
    CUSTOMREAL* array_3d_backward;

    CUSTOMREAL* sk_s;         // s_k = m_{k+1} - m_k, model difference
    CUSTOMREAL* sk_xi;
    CUSTOMREAL* sk_eta;
    CUSTOMREAL* yk_s;         // y_k = g_{k+1} - g_k, gradient difference
    CUSTOMREAL* yk_xi;
    CUSTOMREAL* yk_eta;


    // ---------------------------------------------------------
    // ------------------ specified functions ------------------
    // ---------------------------------------------------------

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    void processing_kernels(Grid&, InputParams&) override;

    // determine step length
    // void determine_step_length(int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj) override;
    void determine_step_length(Grid& grid, int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj) override;


    // ---------------------------------------------------
    // ------------------ sub functions ------------------
    // ---------------------------------------------------

    // calculate bfgs descent direction
    void calculate_bfgs_descent_direction(Grid& grid);

    // read and write histrorical model and gradient
    void get_model_dif(Grid& grid, IO_utils& io, int& i_inv);
    void get_gradient_dif(Grid& grid, IO_utils& io, int& i_inv);
    

};


#endif // OPTIMIZER_BFGS_H