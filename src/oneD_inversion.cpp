#include "oneD_inversion.h"

OneDInversion::OneDInversion(InputParams& IP, Grid& grid) {
    // constructor
    // std::cout << "OneDInversion constructor" << std::endl;

    // generate 2D mesh for forward modeling and inversion
    generate_2d_mesh(IP);

    // allocate arrays (slowness, fac_a, fac_b, tau, ...)
    allocate_arrays();

    // initialize arrays (slowness, fac_a, fac_b, tau, ...)
    load_1d_model(grid);
}


OneDInversion::~OneDInversion() {
    // destructor
    // std::cout << "OneDInversion destructor" << std::endl;
    deallocate_arrays();
}


int OneDInversion::I2V_1DINV(const int& it, const int& ir) {
    // convert 2D index to 1D index
    return ir*nt_1dinv+it;
}


void OneDInversion::generate_2d_mesh(InputParams& IP) {
    // generate 2D mesh for forward modeling and inversion
    // std::cout << "generate_2d_mesh" << std::endl;

    // obtain the maximum epicenter distance
    CUSTOMREAL      tmin_3d = IP.get_min_lat();
    CUSTOMREAL      tmax_3d = IP.get_max_lat();
    CUSTOMREAL      pmin_3d = IP.get_min_lon();
    CUSTOMREAL      pmax_3d = IP.get_max_lon();
    CUSTOMREAL      rmin_3d = radius2depth(IP.get_max_dep());
    CUSTOMREAL      rmax_3d = radius2depth(IP.get_min_dep());
    CUSTOMREAL      max_distance;
    Epicentral_distance_sphere(tmin_3d, pmin_3d, tmax_3d, pmax_3d, max_distance);

    // determine r_1dinv and t_1dinv:
    // r_1dinv: from rmin_3d to rmax_3d
    // t_1dinv: from -0.2*max_distance to 1.2*max_distance
    // dr_1dinv = dr/2;
    // dt_1dinv = dt/2;
    dr_1dinv = (rmax_3d-rmin_3d)/(ngrid_k-1)/2;
    dt_1dinv = (tmax_3d-tmin_3d)/(ngrid_j-1)/2;
    nr_1dinv = std::floor((rmax_3d-rmin_3d)/dr_1dinv)+1;
    nt_1dinv = std::floor(1.4*max_distance/dt_1dinv)+1;
    // initialize arrays
    r_1dinv = allocateMemory<CUSTOMREAL>(nr_1dinv, 4000);
    t_1dinv = allocateMemory<CUSTOMREAL>(nt_1dinv, 4001);
    // fill arrays
    for(int i=0; i<nr_1dinv; i++){
        r_1dinv[i] = rmin_3d + i*dr_1dinv;
    }
    for(int i=0; i<nt_1dinv; i++){
        t_1dinv[i] = -0.2*max_distance + i*dt_1dinv;
    }

}


void OneDInversion::allocate_arrays() {

    // parameters on grid nodes (for model)
    slowness_1dinv      = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4000);
    fac_a_1dinv         = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4001);
    fac_b_1dinv         = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4002);

    // parameters on grid nodes (for eikonal solver and adjoint solver)
    tau_1dinv           = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4003);
    tau_old_1dinv       = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4004);
    T_1dinv             = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4005);
    Tadj_1dinv          = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4006);
    T0v_1dinv           = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4007);
    T0r_1dinv           = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4008);
    T0t_1dinv           = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4009);
    is_changed_1dinv    = allocateMemory<bool>(nr_1dinv*nt_1dinv, 4010);

    // parameters on grid nodes (for inversion)
    Ks_1dinv            = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4011);
}


void OneDInversion::load_1d_model(Grid& grid) {

    // 1. ---------------- load slowness ----------------
    CUSTOMREAL *model_1d_input;
    model_1d_input = allocateMemory<CUSTOMREAL>(loc_K, 4000);
    for(int k=0; k<loc_K; k++){
        int center_index_i = std::floor(loc_I/2)+1;
        int center_index_j = std::floor(loc_J/2)+1;
        model_1d_input[k] = grid.fun_loc[I2V(center_index_i,center_index_j,k)];
    }
    // 1d interpolation
    // interpolate slowness from (grid.r_loc_1d, model_1d_input[k]) to (r_1dinv, *)
    CUSTOMREAL *tmp_slowness;
    tmp_slowness = allocateMemory<CUSTOMREAL>(nr_1dinv, 4000);
    linear_interpolation_1d_sorted(grid.r_loc_1d, model_1d_input, loc_K, r_1dinv, tmp_slowness, nr_1dinv);
    for (int ir = 0; ir < nr_1dinv; ir++){
        for (int it = 0; it < nt_1dinv; it++){
            slowness_1dinv[I2V_1DINV(it,ir)] = tmp_slowness[ir];
        }
    }

    // 2. ---------------- initilize fac_a and fac_b ----------------
    // initialize other field
    for (int ir = 0; ir < nr_1dinv; ir++){
        for (int it = 0; it < nt_1dinv; it++){
            fac_a_1dinv[I2V_1DINV(it,ir)] = _1_CR;
            fac_b_1dinv[I2V_1DINV(it,ir)] = _1_CR/my_square(r_1dinv[ir]);
        }
    }

}


void OneDInversion::deallocate_arrays(){
    
    // grid parameters
    delete[] r_1dinv;
    delete[] t_1dinv;

    // parameters on grid nodes (for model)
    delete[] slowness_1dinv;
    delete[] fac_a_1dinv;
    delete[] fac_b_1dinv;

    // parameters on grid nodes (for eikonal solver and adjoint solver)
    delete[] tau_1dinv;
    delete[] tau_old_1dinv;
    delete[] T_1dinv;
    delete[] Tadj_1dinv;
    delete[] T0v_1dinv;
    delete[] T0r_1dinv;
    delete[] T0t_1dinv;
    delete[] is_changed_1dinv;

    // parameters on grid nodes (for inversion)
    delete[] Ks_1dinv;

}


std::vector<CUSTOMREAL> OneDInversion::run_simulation_one_step_1dinv() {

    // begin from here





    // compute all residual and obj
    std::vector<CUSTOMREAL> obj_residual;

    // return current objective function value
    return obj_residual;
}

void OneDInversion::model_optimize_1dinv() {

}





