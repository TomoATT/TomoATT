#ifndef OPTIMIZER_BFGS_H
#define OPTIMIZER_BFGS_H

#include "optimizer.h"  

class Optimizer_bfgs : public Optimizer {
public:
    Optimizer_bfgs(InputParams& IP);
    ~Optimizer_bfgs();

private:

    const int Mbfgs = 5; // number of previous steps to store

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

    std::vector<CUSTOMREAL> Ks_bfgs_loc;      // backup of bfgs gradient
    std::vector<CUSTOMREAL> Kxi_bfgs_loc;
    std::vector<CUSTOMREAL> Keta_bfgs_loc;

    // scalars in bfgs
    std::vector<CUSTOMREAL> alpha_bfgs;  // store alpha_i in two-loop recursion
    std::vector<CUSTOMREAL> rho;    // store rho_i in two-loop recursion



    // for line search
    std::vector<CUSTOMREAL> alpha_sub_iter;     // store tried step lengths
    std::vector<CUSTOMREAL> v_obj_sub_iter;     // store objective function values at tried step lengths
    std::vector<CUSTOMREAL> proj_sub_iter;      // store projection values at tried step lengths  
    std::vector<bool> armijo_sub_iter;                  // store Armijo condition results
    std::vector<bool> curvature_sub_iter;              // store curvature condition results


    // ---------------------------------------------------------
    // ------------------ specified functions ------------------
    // ---------------------------------------------------------

    // smooth kernels (multigrid)) + kernel normalization (kernel density normalization)
    void processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv) override;


    // ---------------------------------------------------
    // ------------------ sub functions ------------------
    // ---------------------------------------------------

    // write bfgs gradient ()
    void write_bfgs_gradient(Grid& grid, IO_utils& io, int& i_inv);

    // calculate bfgs descent direction
    void calculate_bfgs_descent_direction(Grid& grid, IO_utils& io, int& i_inv);

    // backup bfgs gradient
    void backup_bfgs_gradient(Grid& grid);

    // read and write histrorical model and gradient
    void get_model_dif(Grid& grid, IO_utils& io, int& i_inv);
    void get_gradient_dif(Grid& grid, IO_utils& io, int& i_inv);


    // evaluate line search performance
    bool check_conditions_for_line_search(InputParams& IP, Grid& grid, int sub_iter, int quit_sub_iter, CUSTOMREAL v_obj_inout, CUSTOMREAL v_obj_try) override;


};


#endif // OPTIMIZER_BFGS_H