#include "optimizer_bfgs.h"
#include <iostream>
#include "kernel_postprocessing.h"
    
Optimizer_bfgs::Optimizer_bfgs(InputParams& IP) : Optimizer(IP) {
    // for bfgs method, model and kernel should be written
    need_write_model = true;
    need_write_original_kernel = true;

    // initialize sizes
    array_3d_forward.resize(n_total_loc_grid_points);
    array_3d_backward.resize(n_total_loc_grid_points);

    // vectors in bfgs
    sk_s.resize(n_total_loc_grid_points);
    sk_xi.resize(n_total_loc_grid_points);
    sk_eta.resize(n_total_loc_grid_points);
    yk_s.resize(n_total_loc_grid_points);
    yk_xi.resize(n_total_loc_grid_points);
    yk_eta.resize(n_total_loc_grid_points);

    descent_dir_s.resize(n_total_loc_grid_points);
    descent_dir_xi.resize(n_total_loc_grid_points);
    descent_dir_eta.resize(n_total_loc_grid_points);

    // scalars in bfgs
    alpha.resize(10000);
    rho.resize(10000);
}

Optimizer_bfgs::~Optimizer_bfgs() {

}

// ---------------------------------------------------------
// ------------------ specified functions ------------------
// ---------------------------------------------------------


// smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
void Optimizer_bfgs::processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv) {
    
    // initialize and backup modified kernels
    initialize_and_backup_modified_kernels(grid);

    // check kernel value range
    check_kernel_value_range(grid);

    // calculate bfgs descent direction
    calculate_bfgs_descent_direction(grid, io, i_inv);

    // (to do) BFGS should be equipped with proper kernel smoothing and normalization
    Kernel_postprocessing::process_kernels(IP, grid);       

}



// ---------------------------------------------------
// ------------------ sub functions ------------------
// ---------------------------------------------------

// calculate bfgs descent direction 
// (to do) process for xi and eta kernels
// (to do) be careful for small kernel (zero division)
// y_i = g_{i+1} - g_i, gradient difference
// s_i = m_{i+1} - m_i, model difference
void Optimizer_bfgs::calculate_bfgs_descent_direction(Grid& grid, IO_utils& io, int& i_inv) {
    if(subdom_main){
        if(id_sim == 0){
            if (i_inv > 0) {
                // --------------- step 1,  initialize q = g_k (in this step, descent_dir is q)
                descent_dir_s.assign(grid.Ks_loc, grid.Ks_loc + n_total_loc_grid_points);
                descent_dir_xi.assign(grid.Kxi_loc, grid.Kxi_loc + n_total_loc_grid_points);
                descent_dir_eta.assign(grid.Keta_loc, grid.Keta_loc + n_total_loc_grid_points);

                // --------------- step 2, loop for i = k-1, k-2, ..., k-Mbfgs  (in this step, descent_dir is q)
                // alpha_i = rho_i * s_i^T * q
                // q = q - alpha_i * y_i
                int n_stored = std::min(i_inv, Mbfgs);
                for (int i_bfgs = i_inv-1; i_bfgs >= i_inv-n_stored; i_bfgs--) {
                    // --------------- substep 1, calculate rho_i = 1 / (y_i^T * s_i)
                    get_model_dif(grid, io, i_bfgs);        // obtain s_i (model difference)
                    get_gradient_dif(grid, io, i_bfgs);     // obtain y_i (gradient difference)
                    rho[i_bfgs] = 1.0 / vector_dot_product(yk_s.data(), sk_s.data(), n_total_loc_grid_points);

                    // --------------- substep 2, calculate alpha_i = rho_i * s_i^T * q
                    alpha[i_bfgs] = rho[i_bfgs] * vector_dot_product(sk_s.data(), descent_dir_s.data(), n_total_loc_grid_points);

                    // --------------- substep 3, q = q - alpha_i * y_i
                    for (int idx = 0; idx < n_total_loc_grid_points; idx++) {
                        descent_dir_s[idx]   -= alpha[i_bfgs] * yk_s[idx];
                    }
                    
                }

                // --------------- step 3, scaling of initial Hessian H0_k  (in this step, descent_dir is z)
                // substep 1, calculate gamma_k = (s_{k-1}^T * y_{k-1}) / (y_{k-1}^T * y_{k-1})
                int i_bfgs = i_inv - 1;
                get_model_dif(grid, io, i_bfgs);     // obtain s_{k-1} (model difference)
                get_gradient_dif(grid, io, i_bfgs);  // obtain y_{k-1} (gradient difference)
                CUSTOMREAL sT_y = vector_dot_product(sk_s.data(), yk_s.data(), n_total_loc_grid_points);
                CUSTOMREAL yT_y = vector_dot_product(yk_s.data(), yk_s.data(), n_total_loc_grid_points);
                CUSTOMREAL gamma_k = sT_y / yT_y;

                // substep 2, z = gamma_k * q (because H0_k = gamma_k * I)
                for (int idx = 0; idx < n_total_loc_grid_points; idx++) {
                    descent_dir_s[idx]   *= gamma_k;
                }

                // --------------- step 4, loop for i = k-Mbfgs, k-Mbfgs+1, ..., k-1  (in this step, descent_dir is z)
                for (int i_bfgs = i_inv-n_stored; i_bfgs <= i_inv-1; i_bfgs++) {
                    // --------------- substep 1, beta = rho_i * y_i^T * z
                    get_gradient_dif(grid, io, i_bfgs); // obtain y_i (gradient difference)
                    CUSTOMREAL beta = rho[i_bfgs] * vector_dot_product(yk_s.data(), descent_dir_s.data(), n_total_loc_grid_points);

                    // --------------- substep 2, z = z + s_i * (alpha_i - beta)
                    get_model_dif(grid, io, i_bfgs);     // obtain s_i (model difference)
                    for (int idx = 0; idx < n_total_loc_grid_points; idx++) {
                        descent_dir_s[idx]   += sk_s[idx] * (alpha[i_bfgs] - beta);
                    }
                }

                // --------------- final step, set modified kernels
                // assign minus descent direction (modified kernels) to grid.Ks_loc
                std::copy(descent_dir_s.begin(), descent_dir_s.end(), grid.Ks_loc);
                std::copy(descent_dir_xi.begin(), descent_dir_xi.end(), grid.Kxi_loc);
                std::copy(descent_dir_eta.begin(), descent_dir_eta.end(), grid.Keta_loc);
                
            } else {
                // for the first iteration, use steepest descent (do nothing)
            }
        }

        broadcast_cr_inter_sim(grid.Ks_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Kxi_loc, loc_I*loc_J*loc_K, 0);
        broadcast_cr_inter_sim(grid.Keta_loc, loc_I*loc_J*loc_K, 0);
    }
    
    synchronize_all_world();
}

// read histrorical model difference
void Optimizer_bfgs::get_model_dif(Grid& grid, IO_utils& io, int& i_inv){
    // make h5_group_name_data to be "model"
    io.change_group_name_for_model();

    // slowness perturbation, delta ln(1/vel)
    io.read_vel(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_vel(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        sk_s[i] = std::log(1.0/array_3d_forward[i]) - std::log(1.0/array_3d_backward[i]);
    
    // delta xi
    io.read_xi(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_xi(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        sk_xi[i] = array_3d_forward[i] - array_3d_backward[i];

    // delta eta
    io.read_eta(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_eta(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        sk_eta[i] = array_3d_forward[i] - array_3d_backward[i];
}

// read histrorical gradient difference
void Optimizer_bfgs::get_gradient_dif(Grid& grid, IO_utils& io, int& i_inv){
    // make h5_group_name_data to be "model"
    io.change_group_name_for_model();

    // slowness gradient difference
    io.read_Ks(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_Ks(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        yk_s[i] = array_3d_forward[i] - array_3d_backward[i];

    // xi gradient difference
    io.read_Kxi(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_Kxi(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        yk_xi[i] = array_3d_forward[i] - array_3d_backward[i];
        
    // eta gradient difference
    io.read_Keta(grid, i_inv + 1);
    grid.set_array_from_vis(array_3d_forward.data());
    io.read_Keta(grid, i_inv);
    grid.set_array_from_vis(array_3d_backward.data());
    for (int i = 0; i < n_total_loc_grid_points; i++)
        yk_eta[i] = array_3d_forward[i] - array_3d_backward[i];
}

// vector dot product
// (to do) remove boundary ghost points
CUSTOMREAL Optimizer_bfgs::vector_dot_product(CUSTOMREAL* vec1, CUSTOMREAL* vec2, int n){
    CUSTOMREAL local_sum = dot_product(vec1, vec2, n);
    CUSTOMREAL global_sum;
    allreduce_cr_single(local_sum, global_sum);
    return global_sum;
}
    