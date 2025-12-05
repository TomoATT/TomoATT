#include "optimizer_bfgs.h"
#include <iostream>
#include "kernel_postprocessing.h"
    
Optimizer_bfgs::Optimizer_bfgs() {
    n_total_loc_grid_points = loc_I * loc_J * loc_K;
    
    array_3d_forward.resize(n_total_loc_grid_points);
    array_3d_backward.resize(n_total_loc_grid_points);

    sk_s.resize(n_total_loc_grid_points);
    sk_xi.resize(n_total_loc_grid_points);
    sk_eta.resize(n_total_loc_grid_points);
    yk_s.resize(n_total_loc_grid_points);
    yk_xi.resize(n_total_loc_grid_points);
    yk_eta.resize(n_total_loc_grid_points);
}

Optimizer_bfgs::~Optimizer_bfgs() {

}

// ---------------------------------------------------------
// ------------------ specified functions ------------------
// ---------------------------------------------------------


// smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
void Optimizer_bfgs::processing_kernels(Grid& grid, InputParams& IP) {
    
}


// determine step length
void Optimizer_bfgs::determine_step_length(Grid& grid, int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj) {

}


// ---------------------------------------------------
// ------------------ sub functions ------------------
// ---------------------------------------------------

// calculate bfgs descent direction
void Optimizer_bfgs::calculate_bfgs_descent_direction(Grid& grid) {
    // to be implemented


}


// read histrorical model difference
void Optimizer_bfgs::get_model_dif(Grid& grid, IO_utils& io, int& i_inv){
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

