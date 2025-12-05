#ifndef OPTIMIZER_BFGS_H
#define OPTIMIZER_BFGS_H

#include "optimizer.h"  

class Optimizer_bfgs : public Optimizer {
public:
    Optimizer_bfgs(InputParams& IP);
    ~Optimizer_bfgs();

private:

    const int Mbfgs = 5; // number of previous steps to store
    int n_total_loc_grid_points;    // number of local grid points

    // historical model and gradient
    std::vector<CUSTOMREAL> array_3d_forward;
    std::vector<CUSTOMREAL> array_3d_backward;

    // vectors in bfgs
    std::vector<CUSTOMREAL> sk_s;       // s_k = m_{k+1} - m_k, model difference
    std::vector<CUSTOMREAL> sk_xi;
    std::vector<CUSTOMREAL> sk_eta;
    std::vector<CUSTOMREAL> yk_s;       // y_k = g_{k+1} - g_k, gradient difference
    std::vector<CUSTOMREAL> yk_xi;
    std::vector<CUSTOMREAL> yk_eta;

    std::vector<CUSTOMREAL> descent_dir_s;    // bfgs descent direction
    std::vector<CUSTOMREAL> descent_dir_xi;
    std::vector<CUSTOMREAL> descent_dir_eta;

    // scalars in bfgs
    std::vector<CUSTOMREAL> alpha;  // store alpha_i in two-loop recursion
    std::vector<CUSTOMREAL> rho;    // store rho_i in two-loop recursion

    // ---------------------------------------------------------
    // ------------------ specified functions ------------------
    // ---------------------------------------------------------

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    void processing_kernels(Grid& grid, IO_utils& io, InputParams& IP, int& i_inv) override;


    // ---------------------------------------------------
    // ------------------ sub functions ------------------
    // ---------------------------------------------------

    // calculate bfgs descent direction
    void calculate_bfgs_descent_direction(Grid& grid, IO_utils& io, int& i_inv);

    // read and write histrorical model and gradient
    void get_model_dif(Grid& grid, IO_utils& io, int& i_inv);
    void get_gradient_dif(Grid& grid, IO_utils& io, int& i_inv);
    

};


#endif // OPTIMIZER_BFGS_H