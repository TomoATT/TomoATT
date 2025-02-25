#include "oneD_inversion.h"

// ###############################################
// ########     OneDInversion class     ##########
// ###############################################

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
    Tadj_density_1dinv  = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4006);
    T0v_1dinv           = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4007);
    T0r_1dinv           = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4008);
    T0t_1dinv           = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4009);
    is_changed_1dinv    = allocateMemory<bool>(nr_1dinv*nt_1dinv, 4010);
    delta_1dinv         = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4011);

    // parameters on grid nodes (for inversion)
    Ks_1dinv                    = allocateMemory<CUSTOMREAL>(nr_1dinv, 4011);
    Ks_density_1dinv            = allocateMemory<CUSTOMREAL>(nr_1dinv, 4012);
    Ks_over_Kden_1dinv          = allocateMemory<CUSTOMREAL>(nr_1dinv, 4013);
    Ks_multigrid_1dinv          = allocateMemory<CUSTOMREAL>(nr_1dinv, 4014);
    Ks_multigrid_previous_1dinv = allocateMemory<CUSTOMREAL>(nr_1dinv, 4015);

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

    delete[] model_1d_input;
    delete[] tmp_slowness;
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
    delete[] Tadj_density_1dinv;
    delete[] T0v_1dinv;
    delete[] T0r_1dinv;
    delete[] T0t_1dinv;
    delete[] is_changed_1dinv;
    delete[] delta_1dinv;

    // parameters on grid nodes (for inversion)
    delete[] Ks_1dinv;
    delete[] Ks_density_1dinv;
    delete[] Ks_over_Kden_1dinv;
    delete[] Ks_multigrid_1dinv;
    delete[] Ks_multigrid_previous_1dinv;

}

// ########################################################
// ########           main member function         ########
// ########      run_simulation_one_step_1dinv     ########
// ########           and sub functions            ########
// ########################################################

std::vector<CUSTOMREAL> OneDInversion::run_simulation_one_step_1dinv(InputParams& IP, IO_utils& io, const int& i_inv) {

    // begin from here
    if(world_rank == 0)
        std::cout << "computing traveltime field, adjoint field and kernel for 1d inversion ..." << std::endl;

    // initialize misfit kernel
    initialize_kernel_1d();

    // iterate over sources
    for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
        
        if (myrank == 0){
            std::string name_sim_src  = IP.get_src_name(i_src);
            std::cout << "id_sim: " << id_sim << ", calculating source (" << i_src+1 << "/" << IP.n_src_this_sim_group
                    << "), name: "
                    << name_sim_src << ", lat: " << IP.src_map[name_sim_src].lat
                    << ", lon: " << IP.src_map[name_sim_src].lon << ", dep: " << IP.src_map[name_sim_src].dep
                    << std::endl;
        }

        ///////////////// run forward //////////////////

        // solver 2d eikonal equation for the i-th source for traveltime field
        eikonal_solver_2d(IP, i_src);   // now traveltime field has been stored in T_1dinv.

        // calculate synthetic traveltime and adjoint source
        calculate_synthetic_traveltime_and_adjoint_source(IP, i_src);  // now data in data_map, the data.traveltime has been updated.

        if(IP.get_if_output_source_field() && (IP.get_if_output_in_process() || i_inv >= IP.get_max_iter_inv() - 2 || i_inv == 0)){
            // write out the traveltime field
            write_T_1dinv(io, IP.get_src_name(i_src), i_inv);
        }

        ///////////////// run adjoint //////////////////

        // solver for 2d adjoint field
        int adj_type = 0;  // 0: adjoint_field, 1: adjoint_density
        adjoint_solver_2d(IP, i_src, adj_type);  // now adjoint field has been stored in Tadj_1dinv.
        adj_type = 1;  
        adjoint_solver_2d(IP, i_src, adj_type);  // now adjoint field has been stored in Tadj_density_1dinv.
        
        if(IP.get_if_output_source_field() && (IP.get_if_output_in_process() || i_inv >= IP.get_max_iter_inv() - 2 || i_inv == 0)){
            // write out the adjoint field
            write_Tadj_1dinv(io, IP.get_src_name(i_src), i_src);
        }

        // calculate event sensitivity kernel
        calculate_kernel_1d();
    }
    // synchronize all processes
    synchronize_all_world();

    // calculate objective function and residual
    std::vector<CUSTOMREAL> obj_residual = calculate_obj_and_residual_1dinv(IP);

    // return current objective function value
    return obj_residual;
}


void OneDInversion::eikonal_solver_2d(InputParams& IP, int& i_src){

    // get the source r 
    const std::string name_sim_src  = IP.get_src_name(i_src);
    CUSTOMREAL src_r                = IP.get_src_radius(name_sim_src);
    // source in the 2D grid is r = src_r, t = 0;

    // initialize T0v_1dinv, T0r_1dinv, T0t_1dinv, tau_1dinv, tau_old_1dinv is_changed_1dinv
    initialize_eikonal_array(src_r);

    // fast sweeping method for 2d space
    FSM_2d();
}


void OneDInversion::initialize_eikonal_array(CUSTOMREAL src_r) {
    // discretize the src position
    CUSTOMREAL src_t_dummy = _0_CR;
    int src_r_i = std::floor((src_r - r_1dinv[0])/dr_1dinv);
    int src_t_i = std::floor((src_t_dummy - t_1dinv[0])/dt_1dinv);

    // error of discretized source position
    CUSTOMREAL src_r_err = std::min(_1_CR, (src_r - r_1dinv[src_r_i])/dr_1dinv);
    CUSTOMREAL src_t_err = std::min(_1_CR, (src_t_dummy - t_1dinv[src_t_i])/dt_1dinv);

    // check precision error for floor
    if (src_r_err == _1_CR) {
        src_r_err = _0_CR;
        src_r_i++;
    }
    if (src_t_err == _1_CR) {
        src_t_err = _0_CR;
        src_t_i++;
    }

    // initialize the initial fields
    CUSTOMREAL a0           = (_1_CR - src_r_err)*(_1_CR - src_t_err)*fac_a_1dinv[I2V_1DINV(src_t_i  ,src_r_i  )] \
                            + (_1_CR - src_r_err)*         src_t_err *fac_a_1dinv[I2V_1DINV(src_t_i+1,src_r_i  )] \
                            +           src_r_err*(_1_CR - src_t_err)*fac_a_1dinv[I2V_1DINV(src_t_i  ,src_r_i+1)] \
                            +           src_r_err*         src_t_err *fac_a_1dinv[I2V_1DINV(src_t_i+1,src_r_i+1)];

    CUSTOMREAL b0           = (_1_CR - src_r_err)*(_1_CR - src_t_err)*fac_b_1dinv[I2V_1DINV(src_t_i  ,src_r_i  )] \
                            + (_1_CR - src_r_err)*         src_t_err *fac_b_1dinv[I2V_1DINV(src_t_i+1,src_r_i  )] \
                            +          src_r_err *(_1_CR - src_t_err)*fac_b_1dinv[I2V_1DINV(src_t_i  ,src_r_i+1)] \
                            +          src_r_err *         src_t_err *fac_b_1dinv[I2V_1DINV(src_t_i+1,src_r_i+1)];

    CUSTOMREAL slowness0    = (_1_CR - src_r_err)*(_1_CR - src_t_err)*slowness_1dinv[I2V_1DINV(src_t_i  ,src_r_i  )] \
                            + (_1_CR - src_r_err)*         src_t_err *slowness_1dinv[I2V_1DINV(src_t_i+1,src_r_i  )] \
                            +          src_r_err *(_1_CR - src_t_err)*slowness_1dinv[I2V_1DINV(src_t_i  ,src_r_i+1)] \
                            +          src_r_err *         src_t_err *slowness_1dinv[I2V_1DINV(src_t_i+1,src_r_i+1)];


    for (int ir = 0; ir < nr_1dinv; ir++){
        for (int it = 0; it < nt_1dinv; it++){

            int irt = I2V_1DINV(it,ir);
            T0v_1dinv[irt] = slowness0 * std::sqrt((_1_CR/a0) * my_square((r_1dinv[ir]-src_r))
                                          + _1_CR/b0  * my_square((t_1dinv[it]-src_t_dummy)));

            if (isZero(T0v_1dinv[irt])) {
                T0r_1dinv[irt] = _0_CR;
                T0t_1dinv[irt] = _0_CR;
            } else {
                T0r_1dinv[irt] = my_square(slowness0) * (_1_CR/a0 * (r_1dinv[ir]-src_r))       / T0v_1dinv[irt];
                T0t_1dinv[irt] = my_square(slowness0) * (_1_CR/b0 * (t_1dinv[it]-src_t_dummy)) / T0v_1dinv[irt];
            }

            if (std::abs((r_1dinv[ir]-src_r)/dr_1dinv)       <= _1_CR-0.1 \
             && std::abs((t_1dinv[it]-src_t_dummy)/dt_1dinv) <= _1_CR-0.1) {
                tau_1dinv[irt] = TAU_INITIAL_VAL;
                is_changed_1dinv[irt] = false;
            } else {
                tau_1dinv[irt] = TAU_INF_VAL;  // upwind scheme, initial tau should be large enough
                is_changed_1dinv[irt] = true;
            }
            tau_old_1dinv[irt] = _0_CR;
        }
    }
}


void OneDInversion::FSM_2d() {
    if (myrank==0)
        std::cout << "Running 2d eikonal solver..." << std::endl;

    CUSTOMREAL L1_dif  =1000000000;
    CUSTOMREAL Linf_dif=1000000000;

    int iter = 0;

    while (true) {

        // update tau_old
        std::memcpy(tau_old_1dinv, tau_1dinv, sizeof(CUSTOMREAL)*nr_1dinv*nt_1dinv);

        int r_start, r_end;
        int t_start, t_end;
        int r_dirc, t_dirc;

        // sweep direction
        for (int iswp = 0; iswp < 4; iswp++){
            if (iswp == 0){
                r_start = nr_1dinv-1;
                r_end   = -1;
                t_start = nt_1dinv-1;
                t_end   = -1;
                r_dirc  = -1;
                t_dirc  = -1;
            } else if (iswp==1){
                r_start = nr_1dinv-1;
                r_end   = -1;
                t_start = 0;
                t_end   = nt_1dinv;
                r_dirc  = -1;
                t_dirc  = 1;
            } else if (iswp==2){
                r_start = 0;
                r_end   = nr_1dinv;
                t_start = nt_1dinv-1;
                t_end   = -1;
                r_dirc  = 1;
                t_dirc  = -1;
            } else {
                r_start = 0;
                r_end   = nr_1dinv;
                t_start = 0;
                t_end   = nt_1dinv;
                r_dirc  = 1;
                t_dirc  = 1;
            }

            for (int ir = r_start; ir != r_end; ir += r_dirc) {
                for (int it = t_start; it != t_end; it += t_dirc) {

                    if (is_changed_1dinv[I2V_1DINV(it,ir)]){
                        calculate_stencil(it,ir);
                    }
                }
            }
        } // end of iswp

        // calculate L1 and Linf error
        L1_dif  = _0_CR;
        Linf_dif= _0_CR;

        for (int ii = 0; ii < nr_1dinv*nt_1dinv; ii++){
            L1_dif  += std::abs(tau_1dinv[ii]-tau_old_1dinv[ii]);
            Linf_dif = std::max(Linf_dif, std::abs(tau_1dinv[ii]-tau_old_1dinv[ii]));
        }
        L1_dif /= nr_1dinv*nt_1dinv;


        // check convergence
        if (std::abs(L1_dif) < TOL_2D_SOLVER && std::abs(Linf_dif) < TOL_2D_SOLVER){
            if (myrank==0)
                std::cout << "Eikonal solver converged at iteration " << iter << std::endl;
            break;
        } else if (iter > MAX_ITER_2D_SOLVER){
            if (myrank==0)
                std::cout << "Eikonal solver reached the maximum iteration at iteration " << iter << std::endl;
            break;
        } else {
            if (myrank==0 && if_verbose)
                std::cout << "Iteration " << iter << ": L1 dif = " << L1_dif << ", Linf dif = " << Linf_dif << std::endl;
            iter++;
        }
    } // end of wile

    for (int ii = 0; ii < nr_1dinv*nt_1dinv; ii++){
        T_1dinv[ii] = tau_1dinv[ii]*T0v_1dinv[ii];
    }

}


void OneDInversion::calculate_stencil(const int& it, const int& ir) {
    
    count_cand = 0;
    ii     = I2V_1DINV(it,ir);
    ii_nr  = I2V_1DINV(it,ir-1);
    ii_pr  = I2V_1DINV(it,ir+1);
    ii_nt  = I2V_1DINV(it-1,ir);
    ii_pt  = I2V_1DINV(it+1,ir);
    

    // (T0*tau)_r = ar*tau(iix,iiy)+br; (T0*tau)_t = at*tau(iix,iiy)+bt
    if (it > 0){
        at1 =  T0t_1dinv[ii] + T0v_1dinv[ii]/dt_1dinv;
        bt1 = -T0v_1dinv[ii]/dt_1dinv*tau_1dinv[ii_nt];
    }
    if (it < nt_1dinv-1){
        at2 =  T0t_1dinv[ii] - T0v_1dinv[ii]/dt_1dinv;
        bt2 =  T0v_1dinv[ii]/dt_1dinv*tau_1dinv[ii_pt];
    }
    if (ir > 0){
        ar1 =  T0r_1dinv[ii] + T0v_1dinv[ii]/dr_1dinv;
        br1 = -T0v_1dinv[ii]/dr_1dinv*tau_1dinv[ii_nr];
    }
    if (ir < nr_1dinv-1){
        ar2 =  T0r_1dinv[ii] - T0v_1dinv[ii]/dr_1dinv;
        br2 =  T0v_1dinv[ii]/dr_1dinv*tau_1dinv[ii_pr];
    }

    // start to find candidate solutions

    // first catalog: characteristic travels through sector in 2D volume (4 cases)
    for (int i_case = 0; i_case < 4; i_case++){
        // determine discretization of T_t,T_r
        switch (i_case) {
            case 0:    // characteristic travels from -t, -r
                if (it == 0 || ir == 0)
                    continue;
                at = at1; bt = bt1;
                ar = ar1; br = br1;
                break;
            case 1:     // characteristic travels from -t, +r
                if (it == 0 || ir == nr_1dinv-1)
                    continue;
                at = at1; bt = bt1;
                ar = ar2; br = br2;
                break;
            case 2:    // characteristic travels from +t, -r
                if (it == nt_1dinv-1 || ir == 0)
                    continue;
                at = at2; bt = bt2;
                ar = ar1; br = br1;
                break;
            case 3:     // characteristic travels from +t, +r
                if (it == nt_1dinv-1 || ir == nr_1dinv-1)
                    continue;
                at = at2; bt = bt2;
                ar = ar2; br = br2;
                break;
        }

        // plug T_t, T_r into eikonal equation, solving the quadratic equation with respect to tau(iip,jjt,kkr)
        // that is a*(ar*tau+br)^2 + b*(at*tau+bt)^2 = s^2
        eqn_a = fac_a_1dinv[ii] * std::pow(ar, _2_CR) + fac_b_1dinv[ii] * std::pow(at, _2_CR);
        eqn_b = fac_a_1dinv[ii] * _2_CR * ar * br     + fac_b_1dinv[ii] * _2_CR * at * bt;
        eqn_c = fac_a_1dinv[ii] * std::pow(br, _2_CR) + fac_b_1dinv[ii] * std::pow(bt, _2_CR) - std::pow(slowness_1dinv[ii], _2_CR);
        eqn_Delta = std::pow(eqn_b, _2_CR) - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0){    // one or two real solutions
            for (int i_solution = 0; i_solution < 2; i_solution++){
                // solutions
                switch (i_solution){
                    case 0:
                        tmp_tau = (-eqn_b + std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                    case 1:
                        tmp_tau = (-eqn_b - std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                }

                // check the causality condition: the characteristic passing through (it,ir) is in between used two sides
                // characteristic direction is (dr/dt, dtheta/dt, tphi/dt) = (H_p1,H_p2,H_p3), p1 = T_r, p2 = T_t, p3 = T_p
                T_r = ar*tmp_tau + br;
                T_t = at*tmp_tau + bt;

                charact_r = T_r;
                charact_t = T_t;

                is_causality = false;
                switch (i_case){
                    case 0:  //characteristic travels from -t, -r
                        if (charact_t >= 0 && charact_r >= 0 && tmp_tau > 0){
                            is_causality = true;
                        }
                        break;
                    case 1:  //characteristic travels from -t, +r
                        if (charact_t >= 0 && charact_r <= 0 && tmp_tau > 0){
                            is_causality = true;
                        }
                        break;
                    case 2:  //characteristic travels from +t, -r
                        if (charact_t <= 0 && charact_r >= 0 && tmp_tau > 0){
                            is_causality = true;
                        }
                        break;
                    case 3:  //characteristic travels from +t, +r
                        if (charact_t <= 0 && charact_r <= 0 && tmp_tau > 0){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_tau;
                    count_cand += 1;
                }
            }
        }
    }

    // second catalog: characteristic travels through lines in 1D volume (4 cases)
    // case: 1-2
    // characteristic travels along r-axis, force H_p2, H_p3 = 0, that is, T_t = 0
    // plug the constraint into eikonal equation, we have the equation:   a*T_r^2 = s^2
    for (int i_case = 0; i_case < 2; i_case++){
        switch (i_case){
            case 0:     //characteristic travels from  -r
                if (ir ==  0){
                    continue;
                }
                ar = ar1; br = br1;
                break;
            case 1:     //characteristic travels from  +r
                if (ir ==  nr_1dinv-1){
                    continue;
                }
                ar = ar2; br = br2;
                break;
        }

        // plug T_t, T_r into eikonal equation, solve the quadratic equation:  (ar*tau+br)^2 = s^2
        // simply, we have two solutions
        for (int i_solution = 0; i_solution < 2; i_solution++){
            // solutions
            switch (i_solution){
                case 0:
                    tmp_tau = ( std::sqrt(std::pow(slowness_1dinv[ii],_2_CR)/fac_a_1dinv[ii]) - br)/ar;
                    break;
                case 1:
                    tmp_tau = (-std::sqrt(std::pow(slowness_1dinv[ii],_2_CR)/fac_a_1dinv[ii]) - br)/ar;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 0:  //characteristic travels from -r (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_tau * T0v_1dinv[ii] > tau_1dinv[ii_nr] * T0v_1dinv[ii_nr]
                        && tmp_tau > tau_1dinv[ii_nr]/_2_CR && tmp_tau > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 1:  //characteristic travels from +r
                    if (tmp_tau * T0v_1dinv[ii] > tau_1dinv[ii_pr] * T0v_1dinv[ii_pr]
                        && tmp_tau > tau_1dinv[ii_pr]/_2_CR && tmp_tau > 0){
                        is_causality = true;
                    }
                    break;
            }

            // if satisfying the causility condition, retain it as a canditate solution
            if (is_causality) {
                canditate[count_cand] = tmp_tau;
                count_cand += 1;
            }
        }

    }


    // case: 3-4
    // characteristic travels along t-axis, force H_p1, H_p3 = 0, that is, T_r = 0;
    // plug the constraint into eikonal equation, we have the equation:   T_t^2 = s^2
    for (int i_case = 2; i_case < 4; i_case++){
        switch (i_case){
            case 2:     //characteristic travels from  -t
                if (it ==  0){
                    continue;
                }
                at = at1; bt = bt1;
                break;
            case 3:     //characteristic travels from  +t
                if (it ==  nt_1dinv-1){
                    continue;
                }
                at = at2; bt = bt2;
                break;
        }

        // plug T_t into eikonal equation, solve the quadratic equation:  b*(at*tau+bt)^2 = s^2
        // simply, we have two solutions
        for (int i_solution = 0; i_solution < 2; i_solution++){
            // solutions
            switch (i_solution){
                case 0:
                    tmp_tau = ( std::sqrt(std::pow(slowness_1dinv[ii],_2_CR)/fac_b_1dinv[ii]) - bt)/at;
                    break;
                case 1:
                    tmp_tau = (-std::sqrt(std::pow(slowness_1dinv[ii],_2_CR)/fac_b_1dinv[ii]) - bt)/at;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 2:  //characteristic travels from -t (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_tau * T0v_1dinv[ii] > tau_1dinv[ii_nt] * T0v_1dinv[ii_nt]
                        && tmp_tau > tau_1dinv[ii_nt]/_2_CR && tmp_tau > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 3:  //characteristic travels from +t
                    if (tmp_tau * T0v_1dinv[ii] > tau_1dinv[ii_pt]  * T0v_1dinv[ii_pt]
                        && tmp_tau > tau_1dinv[ii_pt]/_2_CR && tmp_tau > 0){
                        is_causality = true;
                    }
                    break;
            }

            // if satisfying the causility condition, retain it as a canditate solution
            if (is_causality) {
                canditate[count_cand] = tmp_tau;
                count_cand += 1;
            }
        }
    }

    // final, choose the minimum candidate solution as the updated value
    for (int i_cand = 0; i_cand < count_cand; i_cand++){
        tau_1dinv[ii] = std::min(tau_1dinv[ii], canditate[i_cand]);
        if (tau_1dinv[ii] < 0 ){
            std::cout << "error, tau_loc < 0. ir: " << ir << ", it: " << it << std::endl;
            exit(1);
        }
    }

}


void OneDInversion::calculate_synthetic_traveltime_and_adjoint_source(InputParams& IP, int& i_src) {

    // get the (r,t,p) of the real source
    const std::string name_src  = IP.get_src_name(i_src);
    // CUSTOMREAL src_r   = IP.get_src_radius(name_src);
    CUSTOMREAL src_lon = IP.get_src_lon(   name_src); // in radian
    CUSTOMREAL src_lat = IP.get_src_lat(   name_src); // in radian

    // rec.adjoint_source = 0 && rec.adjoint_source_density = 0
    IP.initialize_adjoint_source();

    // loop all data 
    for (auto it_rec = IP.data_map[name_src].begin(); it_rec != IP.data_map[name_src].end(); ++it_rec) {
        for (auto& data: it_rec->second){

            const std::string name_rec = data.name_rec;
            // get position of the receiver
            CUSTOMREAL rec_r = depth2radius(IP.rec_map[name_rec].dep);
            CUSTOMREAL rec_lon = IP.rec_map[name_rec].lon*DEG2RAD;   // in radian
            CUSTOMREAL rec_lat = IP.rec_map[name_rec].lat*DEG2RAD;   // in radian

            // calculate epicentral distance
            CUSTOMREAL distance =0.0;
            Epicentral_distance_sphere(src_lat, src_lon, rec_lat, rec_lon, distance);

            // 2d interporlation, to find the traveltime at (distance, rec_r) on the field of T_1dinv on the mesh meshgrid(t_1dinv, r_1dinv)
            CUSTOMREAL traveltime = interpolate_2d_traveltime(distance, rec_r);
            data.travel_time = traveltime;

            // calculate adjoint source
            if (data.is_src_rec){
                CUSTOMREAL syn_time       = data.travel_time;
                CUSTOMREAL obs_time       = data.travel_time_obs;

                // assign local weight
                CUSTOMREAL  local_weight = _1_CR;

                // evaluate residual_weight_abs （If run_mode == DO_INVERSION, tau_opt always equal 0. But when run_mode == INV_RELOC, we need to consider the change of ortime of earthquakes (swapped receiver)）
                CUSTOMREAL  local_residual = abs(syn_time - obs_time + IP.rec_map[name_rec].tau_opt);
                CUSTOMREAL* res_weight = IP.get_residual_weight_abs();

                if      (local_residual < res_weight[0])    local_weight *= res_weight[2];
                else if (local_residual > res_weight[1])    local_weight *= res_weight[3];
                else                                        local_weight *= ((local_residual - res_weight[0])/(res_weight[1] - res_weight[0]) * (res_weight[3] - res_weight[2]) + res_weight[2]);

                // evaluate distance_weight_abs
                CUSTOMREAL  local_dis    =   _0_CR;
                Epicentral_distance_sphere(IP.get_rec_point(name_rec).lat*DEG2RAD, IP.get_rec_point(name_rec).lon*DEG2RAD, IP.get_src_point(name_src).lat*DEG2RAD, IP.get_src_point(name_src).lon*DEG2RAD, local_dis);
                local_dis *= R_earth;       // rad to km
                CUSTOMREAL* dis_weight = IP.get_distance_weight_abs();

                if      (local_dis < dis_weight[0])         local_weight *= dis_weight[2];
                else if (local_dis > dis_weight[1])         local_weight *= dis_weight[3];
                else                                        local_weight *= ((local_dis - dis_weight[0])/(dis_weight[1] - dis_weight[0]) * (dis_weight[3] - dis_weight[2]) + dis_weight[2]);

                // assign adjoint source
                CUSTOMREAL adjoint_source = IP.get_rec_point(name_rec).adjoint_source + (syn_time - obs_time + IP.rec_map[name_rec].tau_opt) * data.weight * local_weight;
                IP.set_adjoint_source(name_rec, adjoint_source); // set adjoint source to rec_map[name_rec]

                // assign adjoint source density
                CUSTOMREAL adjoint_source_density = IP.get_rec_point(name_rec).adjoint_source_density + _1_CR;
                IP.set_adjoint_source_density(name_rec, adjoint_source_density);
            }
        }
    }
}


CUSTOMREAL OneDInversion::interpolate_2d_traveltime(const CUSTOMREAL& distance, const CUSTOMREAL& r) {

    int r_index = std::floor((r         - r_1dinv[0])/dr_1dinv);
    int t_index = std::floor((distance  - t_1dinv[0])/dt_1dinv);

    // error of discretized source position
    CUSTOMREAL r_err = std::min(_1_CR, (r           - r_1dinv[r_index])/dr_1dinv);
    CUSTOMREAL t_err = std::min(_1_CR, (distance    - t_1dinv[t_index])/dt_1dinv);

    // check precision error for floor
    if (r_err == _1_CR) {
        r_err = _0_CR;
        r_err++;
    }
    if (t_err == _1_CR) {
        t_err = _0_CR;
        t_err++;
    }

    if(t_index < 0 || t_index > nt_1dinv-2 || r_index < 0 || r_index > nr_1dinv-2){
        std::cout << "error, out of range. t_index: " << t_index << ", r_index: " << r_index << std::endl;
        exit(1);
    }

    // initialize the initial fields
    CUSTOMREAL traveltime   = (_1_CR - r_err)*(_1_CR - t_err)*T_1dinv[I2V_1DINV(t_index  ,r_index  )] \
                            + (_1_CR - r_err)*         t_err *T_1dinv[I2V_1DINV(t_index+1,r_index  )] \
                            +          r_err *(_1_CR - t_err)*T_1dinv[I2V_1DINV(t_index  ,r_index+1)] \
                            +          r_err *         t_err *T_1dinv[I2V_1DINV(t_index+1,r_index+1)];
            
    return traveltime;
}


void OneDInversion::adjoint_solver_2d(InputParams& IP, const int& i_src, const int& adj_type){
    
    // initialize adjoint arrays: Tadj_1dinv, is_changed_1dinv, delta_1dinv
    initialize_adjoint_array(IP, i_src, adj_type);

    // fast sweeping method for 2d space
    FSM_2d_adjoint(adj_type);

}   


void OneDInversion::initialize_adjoint_array(InputParams& IP, const int& i_src, const int& adj_type) {
    // adj_type: 0, adjoint_source; 1, adjoint_source_density
    
    // initialize adjoint field
    for (int ir = 0; ir < nr_1dinv; ir++){
        for (int it = 0; it < nt_1dinv; it++){
            tau_1dinv[I2V_1DINV(it,ir)]    = _0_CR;
        }
    }

    // initialize is_changed_1dinv
    for (int ir = 0; ir < nr_1dinv; ir++){
        for (int it = 0; it < nt_1dinv; it++){
            is_changed_1dinv[I2V_1DINV(it,ir)] = true;
        }
    }
    for (int ir = 0; ir < nr_1dinv; ir++){      // boundary are false
        is_changed_1dinv[I2V_1DINV(0,ir)] = false;
        is_changed_1dinv[I2V_1DINV(nt_1dinv-1,ir)] = false;
    }
    for (int it = 0; it < nt_1dinv; it++){
        is_changed_1dinv[I2V_1DINV(it,0)] = false;
        is_changed_1dinv[I2V_1DINV(it,nr_1dinv-1)] = false;
    }

    // initialize delta_1dinv
    for (int ir = 0; ir < nr_1dinv; ir++){
        for (int it = 0; it < nt_1dinv; it++){
            delta_1dinv[I2V_1DINV(it,ir)] = _0_CR;
        }
    }

    // loop all receivers to assign adjoint source

    // get the (r,t,p) of the real source
    const std::string name_src  = IP.get_src_name(i_src);
    // CUSTOMREAL src_r   = IP.get_src_radius(name_src);
    CUSTOMREAL src_lon = IP.get_src_lon(   name_src); // in radian
    CUSTOMREAL src_lat = IP.get_src_lat(   name_src); // in radian

    // loop all receivers
    for (int irec = 0; irec < IP.n_rec_this_sim_group; irec++) {

        // get receiver information
        std::string name_rec = IP.get_rec_name(irec);
        CUSTOMREAL adjoint_source; 
        if (adj_type == 0){
            adjoint_source = IP.rec_map[name_rec].adjoint_source;
        } else if (adj_type == 1) {
            adjoint_source = IP.rec_map[name_rec].adjoint_source_density;
        } else {
            std::cout << "error, adj_type is not defined." << std::endl;
            exit(1);
        }
        
        CUSTOMREAL rec_r = depth2radius(IP.rec_map[name_rec].dep);
        CUSTOMREAL rec_lon = IP.rec_map[name_rec].lon*DEG2RAD;   // in radian
        CUSTOMREAL rec_lat = IP.rec_map[name_rec].lat*DEG2RAD;   // in radian

        if (adjoint_source == 0){
            continue;
        }

        // descretize receiver position
        CUSTOMREAL distance =0.0;
        Epicentral_distance_sphere(src_lat, src_lon, rec_lat, rec_lon, distance);
        int r_index = std::floor((rec_r     - r_1dinv[0])/dr_1dinv);
        int t_index = std::floor((distance  - t_1dinv[0])/dt_1dinv);

        // error of discretized source position
        CUSTOMREAL r_err = std::min(_1_CR, (rec_r       - r_1dinv[r_index])/dr_1dinv);
        CUSTOMREAL t_err = std::min(_1_CR, (distance    - t_1dinv[t_index])/dt_1dinv);

        // check precision error for floor
        if (r_err == _1_CR) {
            r_err = _0_CR;
            r_err++;
        }
        if (t_err == _1_CR) {
            t_err = _0_CR;
            t_err++;
        }

        if(t_index < 0 || t_index > nt_1dinv-2 || r_index < 0 || r_index > nr_1dinv-2){
            std::cout << "error, out of range. t_index: " << t_index << ", r_index: " << r_index << std::endl;
            exit(1);
        }
        
        // assign to delta function
        delta_1dinv[I2V_1DINV(t_index  ,r_index  )] += adjoint_source*(1.0-r_err)*(1.0-t_err)/(dr_1dinv*dt_1dinv*rec_r);
        delta_1dinv[I2V_1DINV(t_index+1,r_index  )] += adjoint_source*(1.0-r_err)*     t_err /(dr_1dinv*dt_1dinv*rec_r);
        delta_1dinv[I2V_1DINV(t_index  ,r_index+1)] += adjoint_source*     r_err *(1.0-t_err)/(dr_1dinv*dt_1dinv*rec_r);
        delta_1dinv[I2V_1DINV(t_index+1,r_index+1)] += adjoint_source*     r_err *     t_err /(dr_1dinv*dt_1dinv*rec_r);
    }

}


void OneDInversion::FSM_2d_adjoint(const int& adj_type) {
    if (myrank==0)
        std::cout << "Running 2d adjoint solver..." << std::endl;

    CUSTOMREAL L1_dif  =1000000000;
    CUSTOMREAL Linf_dif=1000000000;

    int iter = 0;

    while (true) {

        // update tau_old
        std::memcpy(tau_old_1dinv, tau_1dinv, sizeof(CUSTOMREAL)*nr_1dinv*nt_1dinv);

        int r_start, r_end;
        int t_start, t_end;
        int r_dirc, t_dirc;

        // sweep direction
        for (int iswp = 0; iswp < 4; iswp++){
            if (iswp == 0){
                r_start = nr_1dinv-1;
                r_end   = -1;
                t_start = nt_1dinv-1;
                t_end   = -1;
                r_dirc  = -1;
                t_dirc  = -1;
            } else if (iswp==1){
                r_start = nr_1dinv-1;
                r_end   = -1;
                t_start = 0;
                t_end   = nt_1dinv;
                r_dirc  = -1;
                t_dirc  = 1;
            } else if (iswp==2){
                r_start = 0;
                r_end   = nr_1dinv;
                t_start = nt_1dinv-1;
                t_end   = -1;
                r_dirc  = 1;
                t_dirc  = -1;
            } else {
                r_start = 0;
                r_end   = nr_1dinv;
                t_start = 0;
                t_end   = nt_1dinv;
                r_dirc  = 1;
                t_dirc  = 1;
            }

            for (int ir = r_start; ir != r_end; ir += r_dirc) {
                for (int it = t_start; it != t_end; it += t_dirc) {

                    if (is_changed_1dinv[I2V_1DINV(it,ir)])
                        calculate_stencil_adj(it,ir);

                }
            }
        } // end of iswp

        // calculate L1 and Linf error
        L1_dif  = _0_CR;
        Linf_dif= _0_CR;

        for (int ii = 0; ii < nr_1dinv*nt_1dinv; ii++){
            L1_dif  += std::abs(tau_1dinv[ii]-tau_old_1dinv[ii]);
            Linf_dif = std::max(Linf_dif, std::abs(tau_1dinv[ii]-tau_old_1dinv[ii]));
        }
        L1_dif /= nr_1dinv*nt_1dinv;


        // check convergence
        if (std::abs(L1_dif) < TOL_2D_SOLVER && std::abs(Linf_dif) < TOL_2D_SOLVER){
            if (myrank==0)
                std::cout << "Adjoint solver converged at iteration " << iter << std::endl;
            break;
        } else if (iter > MAX_ITER_2D_SOLVER){
            if (myrank==0)
                std::cout << "Adjoint solver reached the maximum iteration at Iteration " << iter << std::endl;
            break;
        } else {
            if (myrank==0 && if_verbose)
                std::cout << "Iteration " << iter << ": L1 dif = " << L1_dif << ", Linf dif = " << Linf_dif << std::endl;
            iter++;
        }
    } // end of wile

    if (adj_type == 0){
        for (int ii = 0; ii < nr_1dinv*nt_1dinv; ii++){
            Tadj_1dinv[ii] = tau_1dinv[ii];
        }
    } else if (adj_type == 1){
        for (int ii = 0; ii < nr_1dinv*nt_1dinv; ii++){
            Tadj_density_1dinv[ii] = tau_1dinv[ii];
        }
    } else {
        std::cout << "error, adj_type is not defined." << std::endl;
        exit(1);
    }
}


void OneDInversion::calculate_stencil_adj(const int& it, const int& ir) {
    // function \nabla \cdot ( P (-\nabla T) M ) = \sum delta * adj
    // matrix:
    // |  a  0 |
    // |  0  b |

    // related coefficients
    CUSTOMREAL a1  = - (T_1dinv[I2V_1DINV(it  ,ir  )] - T_1dinv[I2V_1DINV(it  ,ir-1)]) / dr_1dinv * (fac_a_1dinv[I2V_1DINV(it  ,ir  )] + fac_a_1dinv[I2V_1DINV(it  ,ir-1)])/2.0;
    CUSTOMREAL a1m = (a1 - std::abs(a1))/2.0;
    CUSTOMREAL a1p = (a1 + std::abs(a1))/2.0;

    CUSTOMREAL a2  = - (T_1dinv[I2V_1DINV(it  ,ir+1)] - T_1dinv[I2V_1DINV(it  ,ir  )]) / dr_1dinv * (fac_a_1dinv[I2V_1DINV(it  ,ir  )] + fac_a_1dinv[I2V_1DINV(it  ,ir+1)])/2.0;
    CUSTOMREAL a2m = (a2 - std::abs(a2))/2.0;
    CUSTOMREAL a2p = (a2 + std::abs(a2))/2.0;

    CUSTOMREAL b1  = - (T_1dinv[I2V_1DINV(it  ,ir  )] - T_1dinv[I2V_1DINV(it-1,ir  )]) / dt_1dinv * (fac_b_1dinv[I2V_1DINV(it  ,ir  )] + fac_b_1dinv[I2V_1DINV(it-1,ir  )])/2.0;
    CUSTOMREAL b1m = (b1 - std::abs(b1))/2.0;
    CUSTOMREAL b1p = (b1 + std::abs(b1))/2.0;

    CUSTOMREAL b2  = - (T_1dinv[I2V_1DINV(it+1,ir  )] - T_1dinv[I2V_1DINV(it  ,ir  )]) / dt_1dinv * (fac_b_1dinv[I2V_1DINV(it+1,ir  )] + fac_b_1dinv[I2V_1DINV(it  ,ir  )])/2.0;
    CUSTOMREAL b2m = (b2 - std::abs(b2))/2.0;
    CUSTOMREAL b2p = (b2 + std::abs(b2))/2.0;

    CUSTOMREAL coe = (a2p - a1m)/dr_1dinv + (b2p - b1m)/dt_1dinv;

    if (isZero(coe)){
        tau_1dinv[I2V_1DINV(it,ir)] = 0.0;
    } else {
        // Hamiltonian
        CUSTOMREAL Hadj = (a1p * tau_1dinv[I2V_1DINV(it,ir-1)] - a2m * tau_1dinv[I2V_1DINV(it,ir+1)])/dr_1dinv 
                        + (b1p * tau_1dinv[I2V_1DINV(it-1,ir)] - b2m * tau_1dinv[I2V_1DINV(it+1,ir)])/dt_1dinv;

        tau_1dinv[I2V_1DINV(it,ir)] = (delta_1dinv[I2V_1DINV(it,ir)] + Hadj) / coe;
    }
}


void OneDInversion::initialize_kernel_1d() {
    for (int ir=0; ir<nr_1dinv; ir++){
        Ks_1dinv[ir]                    = _0_CR;
        Ks_density_1dinv[ir]            = _0_CR;
    }
}


void OneDInversion::calculate_kernel_1d() {
    for (int ir=0; ir<nr_1dinv; ir++){
        for (int it=0; it<nt_1dinv; it++){
            Ks_1dinv[ir]            += Tadj_1dinv[I2V_1DINV(it,ir)]         * my_square(slowness_1dinv[I2V_1DINV(it,ir)]) * dt_1dinv * dr_1dinv;
            Ks_density_1dinv[ir]    += Tadj_density_1dinv[I2V_1DINV(it,ir)] * my_square(slowness_1dinv[I2V_1DINV(it,ir)]) * dt_1dinv * dr_1dinv;
        }
    }
    
}


std::vector<CUSTOMREAL> OneDInversion::calculate_obj_and_residual_1dinv(InputParams& IP) {

    CUSTOMREAL obj           = 0.0;
    CUSTOMREAL obj_abs       = 0.0;
    CUSTOMREAL obj_cs_dif    = 0.0;
    CUSTOMREAL obj_cr_dif    = 0.0;
    CUSTOMREAL obj_tele      = 0.0;

    CUSTOMREAL res           = 0.0;
    CUSTOMREAL res_sq        = 0.0;
    CUSTOMREAL res_abs       = 0.0;
    CUSTOMREAL res_abs_sq    = 0.0;
    CUSTOMREAL res_cs_dif    = 0.0;
    CUSTOMREAL res_cs_dif_sq = 0.0;
    CUSTOMREAL res_cr_dif    = 0.0;
    CUSTOMREAL res_cr_dif_sq = 0.0;
    CUSTOMREAL res_tele      = 0.0;
    CUSTOMREAL res_tele_sq   = 0.0;

    std::vector<CUSTOMREAL> obj_residual;

    // iterate over sources
    for (int i_src = 0; i_src < IP.n_src_this_sim_group; i_src++){
        const std::string name_src  = IP.get_src_name(i_src);

        // loop all data 
        for (auto it_rec = IP.data_map[name_src].begin(); it_rec != IP.data_map[name_src].end(); ++it_rec) {
            for (auto& data: it_rec->second){
                if (data.is_src_rec){
                    CUSTOMREAL syn_time       = data.travel_time;
                    CUSTOMREAL obs_time       = data.travel_time_obs;
                    std::string name_rec      = data.name_rec;

                    // contribute misfit of specific type of data
                    obj             +=  1.0 * my_square(syn_time - obs_time + IP.rec_map[name_rec].tau_opt) * data.weight;
                    res             += 1.0 *          (syn_time - obs_time + IP.rec_map[name_rec].tau_opt);
                    res_sq          += 1.0 * my_square(syn_time - obs_time + IP.rec_map[name_rec].tau_opt);

                    obj_abs         +=  1.0 * my_square(syn_time - obs_time + IP.rec_map[name_rec].tau_opt) * data.weight;
                    res_abs         +=  1.0 *          (syn_time - obs_time + IP.rec_map[name_rec].tau_opt);
                    res_abs_sq      +=  1.0 * my_square(syn_time - obs_time + IP.rec_map[name_rec].tau_opt);
                    
                } else {
                    // pass, only consider absolute time residual for 1D inversion
                }
            }
        }
    }


    obj_residual = {obj, obj_abs, obj_cs_dif, obj_cr_dif, obj_tele, res, res_sq, res_abs, res_abs_sq, res_cs_dif, res_cs_dif_sq, res_cr_dif, res_cr_dif_sq, res_tele, res_tele_sq};

    for(int i = 0; i < (int)obj_residual.size(); i++){
        allreduce_cr_sim_single_inplace(obj_residual[i]);
    }

    // update obj
    old_v_obj       = v_obj;
    v_obj           = obj_residual[0];

    return obj_residual;
}

// ########################################################
// ########           main member function         ########
// ########           model_optimize_1dinv         ########
// ########             and sub functions          ########
// ########################################################

void OneDInversion::model_optimize_1dinv(InputParams& IP, Grid& grid, IO_utils& io, const int& i_inv) {

    // kernel aggregation
    allreduce_cr_sim_inplace(Ks_1dinv, nr_1dinv);
    allreduce_cr_sim_inplace(Ks_density_1dinv, nr_1dinv);

    // // write kernel
    // if (id_sim==0 && subdom_main && IP.get_if_output_kernel() && (IP.get_if_output_in_process() || i_inv >= IP.get_max_iter_inv() - 2 || i_inv == 0)) {
    //     // store kernel only in the first src datafile
    //     io.change_group_name_for_model();
    
    //     write_Ks_1dinv(io, i_inv);
    // }

    // update kernel by rank 0
    if (id_sim==0){
        // kernel processing (multi-grid parameterization, density normalization)
        kernel_processing_1dinv(grid);

        // model_update
        model_update_1dinv(i_inv);

        // write slowness
        if(IP.get_if_output_in_process() || i_inv >= IP.get_max_iter_inv() - 2 || i_inv == 0){
            write_vel_1dinv(io, i_inv);
        }
        
    }

    // broadcast new slowness
    broadcast_cr_inter_sim(slowness_1dinv, nr_1dinv*nt_1dinv, 0);

    // generate 3d model from the 1d model
    generate_3d_model(grid);

}   


void OneDInversion::kernel_processing_1dinv(Grid& grid) {

    // density normalization
    density_normalization_1dinv();

    // multi-grid parameterization
    multi_grid_parameterization_1dinv(grid);

}


void OneDInversion::density_normalization_1dinv() {
    // kernel density normalization
    for (int ir=0; ir<nr_1dinv; ir++){
        if(isZero(Ks_density_1dinv[ir])){
            // do nothing
            Ks_over_Kden_1dinv[ir] = Ks_1dinv[ir];
        } else {
            if(Ks_density_1dinv[ir] < 0) {
                std::cout << "WARNING!!!  Ks_density_1dinv < 0. ir: " << ir << ", Ks_density_1dinv[ir]: " << Ks_density_1dinv[ir] << std::endl;
            }
            Ks_over_Kden_1dinv[ir] = Ks_1dinv[ir] / std::pow(std::abs(Ks_density_1dinv[ir]),Kdensity_coe);
        }
    }
}


void OneDInversion::multi_grid_parameterization_1dinv(Grid& grid) {

    // store the previous Ks_multigrid and initialize Ks_multigrid
    for (int ir = 0; ir < nr_1dinv; ir++) {
        Ks_multigrid_previous_1dinv[ir] = Ks_multigrid_1dinv[ir];
        Ks_multigrid_1dinv[ir] = _0_CR;
    }

    const int ref_idx_t = 0;
    const int ref_idx_p = 0;
    int kdr = -1;
    CUSTOMREAL ratio_r = -_1_CR;

    


    // loop over all inversion grids
    for (int i_grid = 0; i_grid < n_inv_grids; i_grid++) {
        // initialize the Ks on inversion grid 
        // Here we borrow the definition of Ks_inv_loc on 3D inversion grid, but lon, lat index are set to be ref_idx_t = 0 and ref_idx_p = 0
        for (int k = 0; k < n_inv_K_loc; k++) { 
            grid.Ks_inv_loc[I2V_INV_KNL(ref_idx_p,ref_idx_t,k)] = _0_CR;
        }

        // part 1, project r_1dinv onto the inversion grid
        for (int ir = 0; ir < nr_1dinv; ir++) {
            ratio_r = -_1_CR;
            for (int ii_invr = 0; ii_invr < n_inv_K_loc-1; ii_invr++){
                // increasing or decreasing order
                if (in_between(r_1dinv[ir], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)])) {
                    kdr = ii_invr;
                    ratio_r = calc_ratio_between(r_1dinv[ir], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)]);
                    break;
                }
            }
            // continue if r is out of the inversion grid
            if (ratio_r < _0_CR) continue;

            CUSTOMREAL dtdr = dt_1dinv * r_1dinv[ir] * dr_1dinv;

            grid.Ks_inv_loc[I2V_INV_KNL(ref_idx_p,ref_idx_t,kdr  )] += (_1_CR-ratio_r) * Ks_over_Kden_1dinv[ir] * dtdr;
            grid.Ks_inv_loc[I2V_INV_KNL(ref_idx_p,ref_idx_t,kdr+1)] += (ratio_r)       * Ks_over_Kden_1dinv[ir] * dtdr;
                    
        }

        // part 2, project Ks_inv_loc back to r_1dinv
        for (int ir = 0; ir < nr_1dinv; ir++) {
            ratio_r = -_1_CR;
            for (int ii_invr = 0; ii_invr < n_inv_K_loc-1; ii_invr++){
                // increasing or decreasing order
                if (in_between(r_1dinv[ir], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)])) {
                    kdr = ii_invr;
                    ratio_r = calc_ratio_between(r_1dinv[ir], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr,i_grid)], grid.inv_grid->r.arr[I2V_INV_GRIDS_1DK(ii_invr+1,i_grid)]);
                    break;
                }
            }
            // continue if r is out of the inversion grid
            if (ratio_r < _0_CR) continue;

            CUSTOMREAL dtdr = dt_1dinv * r_1dinv[ir] * dr_1dinv;

            Ks_multigrid_1dinv[ir] += (_1_CR-ratio_r) * grid.Ks_inv_loc[I2V_INV_KNL(ref_idx_p,ref_idx_t,kdr  )] * dtdr;
            Ks_multigrid_1dinv[ir] += (ratio_r)       * grid.Ks_inv_loc[I2V_INV_KNL(ref_idx_p,ref_idx_t,kdr+1)] * dtdr;
        }
    }

    // normalize Ks_miltigrid
    CUSTOMREAL Linf_Ks = _0_CR;
    for(int ir = 0; ir < nr_1dinv; ir++) {
        Linf_Ks = std::max(Linf_Ks, std::abs(Ks_multigrid_1dinv[ir]));
    }
    if(isZero(Linf_Ks)){
        std::cout << "ERROR!!!  max value of kernel = 0. Please check data and input parameter" << std::endl;
        exit(1);
    }
    for(int ir = 0; ir < nr_1dinv; ir++) {
        Ks_multigrid_1dinv[ir] /= Linf_Ks;
    }
}


void OneDInversion::model_update_1dinv(const int& i_inv ) {
    
    // determine step size
    determine_step_size_1dinv(i_inv);

    // update model
    for (int it=0; it<nt_1dinv; it++){
        for (int ir=0; ir<nr_1dinv; ir++){
            slowness_1dinv[I2V_1DINV(it,ir)] *= (_1_CR - Ks_multigrid_1dinv[ir] * step_length_init);
        }
    }
}


void OneDInversion::determine_step_size_1dinv(const int& i_inv) {

    // change stepsize
    // Option 1: the step length is modulated when obj changes.
    if (step_method == OBJ_DEFINED){
        if(i_inv != 0){
            if (v_obj < old_v_obj) {
                step_length_init    = std::min((CUSTOMREAL)0.02, step_length_init);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The obj keeps decreasing, from " << old_v_obj << " to " << v_obj
                            << ", the step length is " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            } else if (v_obj >= old_v_obj) {
                step_length_init    = std::max((CUSTOMREAL)0.0001, step_length_init*step_length_decay);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The obj keep increases, from " << old_v_obj << " to " << v_obj
                            << ", the step length decreases from " << step_length_init/step_length_decay
                            << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            }
        } else {
            if(myrank == 0 && id_sim == 0){
                std::cout << std::endl;
                std::cout << "At the first iteration, the step length is " << step_length_init << std::endl;
                std::cout << std::endl;
            }
        }
    } else if (step_method == GRADIENT_DEFINED){
        // Option 2: we modulate the step length according to the angle between the previous and current gradient directions.
        // If the angle is less than XX degree, which means the model update direction is successive, we should enlarge the step size
        // Otherwise, the step length should decrease
        if(i_inv != 0){
            // calculate the angle between the previous and current gradient directions
            CUSTOMREAL angle = 0;
            for (int ir = 0; ir < nr_1dinv; ir++){
                angle += Ks_multigrid_1dinv[ir] * Ks_multigrid_previous_1dinv[ir];
            }
            angle = std::acos(angle/(norm_1dinv(Ks_multigrid_1dinv, nr_1dinv)*norm_1dinv(Ks_multigrid_previous_1dinv, nr_1dinv)))*RAD2DEG;

            if (angle > step_length_gradient_angle){
                CUSTOMREAL old_step_length = step_length_init;
                step_length_init    = std::max((CUSTOMREAL)0.0001, step_length_init * step_length_down);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The angle between two update darections is " << angle
                            << ". Because the angle is greater than " << step_length_gradient_angle << " degree, the step length decreases from "
                            << old_step_length << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            } else if (angle <= step_length_gradient_angle) {
                CUSTOMREAL old_step_length = step_length_init;
                step_length_init    = std::min((CUSTOMREAL)0.02, step_length_init * step_length_up);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The angle between two update darections is " << angle
                            << ". Because the angle is less than " << step_length_gradient_angle << " degree, the step length increases from "
                            << old_step_length << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            }
        } else {
            if(myrank == 0 && id_sim == 0){
                std::cout << std::endl;
                std::cout << "At the first iteration, the step length is " << step_length_init << std::endl;
                std::cout << std::endl;
            }
        }
    } else {
        std::cout << std::endl;
        std::cout << "No supported method for step size change, step keep the same: " << step_length_init << std::endl;
        std::cout << std::endl;
    }
}


CUSTOMREAL OneDInversion::norm_1dinv(const CUSTOMREAL* vec, const int& n) {
    CUSTOMREAL norm = 0.0;
    for (int i = 0; i < n; i++){
        norm += vec[i]*vec[i];
    }
    norm = std::sqrt(norm);
    
    return norm;
}


void OneDInversion::generate_3d_model(Grid& grid){
    
    // 1. ---------------- load slowness ----------------
    CUSTOMREAL *tmp_slowness;
    tmp_slowness = allocateMemory<CUSTOMREAL>(nr_1dinv, 4000);
    for (int ir = 0; ir < nr_1dinv; ir++){
        tmp_slowness[ir] = slowness_1dinv[I2V_1DINV(0,ir)];
    }
    // 1d interpolation
    // interpolate slowness from (r_1dinv, slowness_1dinv) to (grid.r_loc_1d, *)
    CUSTOMREAL *model_1d_output;
    model_1d_output = allocateMemory<CUSTOMREAL>(loc_K, 4000);
    linear_interpolation_1d_sorted(r_1dinv, tmp_slowness, nr_1dinv, grid.r_loc_1d, model_1d_output, loc_K);

    // 1d model to 3d model
    for (int k = 0; k < loc_K; k++) {
        for (int j = 0; j < loc_J; j++) {
            for (int i = 0; i < loc_I; i++) {
                grid.fun_loc[I2V(i,j,k)] = model_1d_output[k];
            }
        }
    }
}

// ########################################################
// ########           HDF5 output functions         #######
// ########################################################

void OneDInversion::write_T_1dinv(IO_utils& io, const std::string& name_src, const int& i_inv) {
    std::string field_name = "time_field_1dinv_" + name_src + "_inv_" + int2string_zero_fill(i_inv);
    io.write_1dinv_field(T_1dinv, r_1dinv, t_1dinv, nr_1dinv, nt_1dinv, field_name);
}

void OneDInversion::write_Tadj_1dinv(IO_utils& io, const std::string& name_src, const int& i_inv) {
    std::string field_name = "adjoint_field_1dinv_" + name_src + "_inv_" + int2string_zero_fill(i_inv);
    io.write_1dinv_field(Tadj_1dinv, r_1dinv, t_1dinv, nr_1dinv, nt_1dinv, field_name);
}

void OneDInversion::write_vel_1dinv(IO_utils& io, const int& i_inv) {
    std::string field_name = "vel_1dinv_inv_" + int2string_zero_fill(i_inv);
    CUSTOMREAL *vel_1dinv;
    vel_1dinv = allocateMemory<CUSTOMREAL>(nr_1dinv*nt_1dinv, 4000);
    for (int ii = 0; ii < nr_1dinv*nt_1dinv; ii++){
        vel_1dinv[ii] = _1_CR/slowness_1dinv[ii];
    }
    io.write_1dinv_field(vel_1dinv, r_1dinv, t_1dinv, nr_1dinv, nt_1dinv, field_name);
    delete[] vel_1dinv;
}

