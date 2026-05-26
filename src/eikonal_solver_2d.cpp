#include "eikonal_solver_2d.h"

void prepare_teleseismic_boundary_conditions(InputParams& IP, Grid& grid, IO_utils& io) {
    bool if_teleseismic_event_exists=false;

    // 2D travel time fields vary only by the depth of the source.
    // So here we make another src_id2name list for dropping the sources
    // that have the same depth, in onder to reduce the number of 2D eikonal solver runs.

    // so we use
    // - src_id2name_2d
    // - src_map_2d
    // for 2D eikonal solver runs

    // in the current implentation, we only run the 2d solver with the proc_store_srcrec process
    // thus all the processes assigned to subdomain and sweep parallelization will not used.
    // For efficiency, it is strongly recommended to use pre-run mode of teleseismic source calculation,
    // with no domain decomposition and sweep parallelization.

    //
    // pre calculation of 2d travel time fields
    //

    for (int i_src = 0; i_src < IP.n_src_2d_this_sim_group; i_src++){

        std::string name_sim_src;

        if (proc_store_srcrec)
            name_sim_src = IP.src_id2name_2d[i_src];
        broadcast_str(name_sim_src, 0);

        // get source info
        bool is_teleseismic = true; // the object in src_id2name_2d is always teleseismic
        // #BUG: src in src_id2name_2d includes the srcs in other sim groups.
        bool for_2d_solver = true;

        Source src;
        src.set_source_position(IP, grid, is_teleseismic, name_sim_src, for_2d_solver);

        // run 2d eikonal solver for teleseismic boundary conditions if teleseismic event
        if (proc_store_srcrec){
            if (myrank==0)
                std::cout << "solve 2d eikonal equation for src: " << name_sim_src << std::endl;
            run_2d_solver(IP, src, io);
        }
    }

    synchronize_all_world();

    // boundary travel time data should not be loaded here because the memory requirement will be
    // too large for large scale simulations.
    // instead, we load the boundary travel time data at the initialization of the iterator class.

    if (IP.n_src_2d_this_sim_group > 0)
        if_teleseismic_event_exists = true;

    if (myrank==0 && if_teleseismic_event_exists)
        std::cout << "done preparing teleseismic boundary conditions" << std::endl;
}



PlainGrid::PlainGrid(Source& src, InputParams& IP) {

    //stdout_by_main("PlainGrid initialization start.");

    // get source info
    src_r = src.get_src_r();
    src_t = src.get_src_t();
    src_p = src.get_src_p();
    src_t_dummy = _0_CR; // dummy src_t
    rmin_3d = radius2depth(IP.get_max_dep());
    rmax_3d = radius2depth(IP.get_min_dep());
    tmin_3d = IP.get_min_lat();
    tmax_3d = IP.get_max_lat();
    pmin_3d = IP.get_min_lon();
    pmax_3d = IP.get_max_lon();

    // allocate arrays
    allocate_arrays();

    // calculate epicentral distance and azimuth between source and simulation domain
    // S-W
    Epicentral_distance_sphere(tmin_3d, pmin_3d, src_t, src_p, epicentral_distance_2d[0]);
    Azimuth_sphere(tmin_3d, pmin_3d, src_t, src_p, azimuth_2d[0]);
    // S-E
    Epicentral_distance_sphere(tmin_3d, pmax_3d, src_t, src_p, epicentral_distance_2d[1]);
    Azimuth_sphere(tmin_3d, pmax_3d, src_t, src_p, azimuth_2d[1]);
    // N-W
    Epicentral_distance_sphere(tmax_3d, pmin_3d, src_t, src_p, epicentral_distance_2d[2]);
    Azimuth_sphere(tmax_3d, pmin_3d, src_t, src_p, azimuth_2d[2]);
    // N-E
    Epicentral_distance_sphere(tmax_3d, pmax_3d, src_t, src_p, epicentral_distance_2d[3]);
    Azimuth_sphere(tmax_3d, pmax_3d, src_t, src_p, azimuth_2d[3]);

    // determine which boundaries are activated
    max_degree = std::max({
        std::abs(epicentral_distance_2d[0]),
        std::abs(epicentral_distance_2d[1]),
        std::abs(epicentral_distance_2d[2]),
        std::abs(epicentral_distance_2d[3]),
    });

    //activated_boundaries[0] = true; // N
    //activated_boundaries[1] = true; // E
    //activated_boundaries[2] = true; // W
    //activated_boundaries[3] = true; // S
    //activated_boundaries[4] = true; // Bot

    // grid setup
    nr_2d = std::floor((rmax_2d-rmin_2d)/dr_2d)+1;
    nt_2d = std::floor((tmax_2d-tmin_2d)/dt_2d)+1;

    // initialize arrays
    r_2d = allocateMemory<CUSTOMREAL>(nr_2d, 4000);
    t_2d = allocateMemory<CUSTOMREAL>(nt_2d, 4001);

    // fill arrays
    for (int i = 0; i < nr_2d; i++) {
        r_2d[i] = rmin_2d + i*dr_2d;
    }
    for (int i = 0; i < nt_2d; i++) {
        t_2d[i] = tmin_2d + i*dt_2d;
    }

    // output summary of 2d grid
    if (myrank==0 && if_verbose) {
        std::cout << "\n\n2d grid info :" << std::endl;
        std::cout << "nr_2d: " << nr_2d << std::endl;
        std::cout << "nt_2d: " << nt_2d << std::endl;
        std::cout << "rmin_2d: " << rmin_2d << std::endl;
        std::cout << "rmax_2d: " << rmax_2d << std::endl;
        std::cout << "tmin_2d: " << tmin_2d << std::endl;
        std::cout << "tmax_2d: " << tmax_2d << std::endl;
        std::cout << "dr_2d: " << dr_2d << std::endl;
        std::cout << "dt_2d: " << dt_2d << std::endl;
        std::cout << "r_2d: " << r_2d[0] << " " << r_2d[nr_2d-1] << std::endl;
        std::cout << "t_2d: " << t_2d[0] << " " << t_2d[nt_2d-1] << std::endl;
        std::cout << "\n\n" << std::endl;
    }

    // field setup
    fun_2d        = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4002);
    fac_a_2d      = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4003);
    fac_b_2d      = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4004);
    u_2d          = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4005);
    T_2d          = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4006);
    T0v_2d        = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4007);
    T0r_2d        = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4008);
    T0t_2d        = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4009);
    tau_2d        = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4010);
    tau_old_2d    = allocateMemory<CUSTOMREAL>(nr_2d*nt_2d, 4011);
    is_changed_2d = allocateMemory<bool>(nr_2d*nt_2d, 4012);

    // load 1d model to 2d field
    load_1d_model(IP.get_model_1d_name());

    // initialize other field
    for (int ir = 0; ir < nr_2d; ir++){
        for (int it = 0; it < nt_2d; it++){
            fac_a_2d[ir*nt_2d+it] = _1_CR;
            fac_b_2d[ir*nt_2d+it] = _1_CR/my_square(r_2d[ir]);
            u_2d[ir*nt_2d+it] = _0_CR;
            T_2d[ir*nt_2d+it] = _0_CR;
        }
    }

    // discretize the src position
    int src_r_i = std::floor((src_r      -rmin_2d)/dr_2d);
    int src_t_i = std::floor((src_t_dummy-tmin_2d)/dt_2d);

    // error of discretized source position
    CUSTOMREAL src_r_err = std::min(_1_CR, (src_r       - r_2d[src_r_i])/dr_2d);
    CUSTOMREAL src_t_err = std::min(_1_CR, (src_t_dummy - t_2d[src_t_i])/dt_2d);

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
    CUSTOMREAL a0 = (_1_CR - src_r_err)*(_1_CR - src_t_err)*fac_a_2d[src_r_i*nt_2d+src_t_i] \
                  + (_1_CR - src_r_err)*         src_t_err *fac_a_2d[src_r_i*nt_2d+src_t_i+1] \
                  +           src_r_err*(_1_CR - src_t_err)*fac_a_2d[(src_r_i+1)*nt_2d+src_t_i] \
                  +           src_r_err*         src_t_err *fac_a_2d[(src_r_i+1)*nt_2d+src_t_i+1];

    CUSTOMREAL b0 = (_1_CR - src_r_err)*(_1_CR - src_t_err)*fac_b_2d[src_r_i*nt_2d+src_t_i] \
                  + (_1_CR - src_r_err)*         src_t_err *fac_b_2d[src_r_i*nt_2d+src_t_i+1] \
                  +          src_r_err *(_1_CR - src_t_err)*fac_b_2d[(src_r_i+1)*nt_2d+src_t_i] \
                  +          src_r_err *         src_t_err *fac_b_2d[(src_r_i+1)*nt_2d+src_t_i+1];

    CUSTOMREAL fun0 = (_1_CR - src_r_err)*(_1_CR - src_t_err)*fun_2d[src_r_i*nt_2d+src_t_i] \
                    + (_1_CR - src_r_err)*         src_t_err *fun_2d[src_r_i*nt_2d+src_t_i+1] \
                    +          src_r_err *(_1_CR - src_t_err)*fun_2d[(src_r_i+1)*nt_2d+src_t_i] \
                    +          src_r_err *         src_t_err *fun_2d[(src_r_i+1)*nt_2d+src_t_i+1];


    for (int ir = 0; ir < nr_2d; ir++){
        for (int it = 0; it < nt_2d; it++){

            int irt = ir*nt_2d+it;
            T0v_2d[irt] = fun0 * std::sqrt((_1_CR/a0) * my_square((r_2d[ir]-src_r))
                                          + _1_CR/b0  * my_square((t_2d[it]-src_t_dummy)));

            if (isZero(T0v_2d[irt])) {
                T0r_2d[irt] = _0_CR;
                T0t_2d[irt] = _0_CR;
            } else {
                T0r_2d[irt] = my_square(fun0) * (_1_CR/a0 * (r_2d[ir]-src_r))       / T0v_2d[irt];
                T0t_2d[irt] = my_square(fun0) * (_1_CR/b0 * (t_2d[it]-src_t_dummy)) / T0v_2d[irt];
            }

            if (std::abs((r_2d[ir]-src_r)/dr_2d)       <= _2_CR \
             && std::abs((t_2d[it]-src_t_dummy)/dt_2d) <= _2_CR) {
                tau_2d[irt] = TAU_INITIAL_VAL;
                is_changed_2d[irt] = false;
            } else {
                tau_2d[irt] = TAU_INF_VAL;  // upwind scheme, initial tau should be large enough
                is_changed_2d[irt] = true;
            }

            tau_old_2d[irt] = _0_CR;
        }
    }
}


void PlainGrid::allocate_arrays(){
    azimuth_2d             = allocateMemory<CUSTOMREAL>(4, 4013);
    epicentral_distance_2d = allocateMemory<CUSTOMREAL>(4, 4014);
}


void PlainGrid::deallocate_arrays(){
    delete[] azimuth_2d;
    delete[] epicentral_distance_2d;
    delete[] r_2d;
    delete[] t_2d;
    delete[] fun_2d;
    delete[] fac_a_2d;
    delete[] fac_b_2d;
    delete[] u_2d;
    delete[] T_2d;
    delete[] T0v_2d;
    delete[] T0r_2d;
    delete[] T0t_2d;
    delete[] tau_2d;
    delete[] tau_old_2d;
    delete[] is_changed_2d;
}


PlainGrid::~PlainGrid() {
    deallocate_arrays();
}


void PlainGrid::select_1d_model(std::string model_1d_name){
    if(model_1d_name.compare("iasp91")==0) {
        model_1d_ref = model_1d_iasp91;
    } else if (model_1d_name.compare("ak135")==0) {
        model_1d_ref = model_1d_ak135;
    } else if (model_1d_name.compare("user_defined")==0) {
        model_1d_ref = model_1d_prem;
    } else {
        model_1d_ref = model_1d_prem;
    }
}


void PlainGrid::load_1d_model(std::string model_1d_name){

    // debug output 1d model to a file
    std::ofstream file_1d_model;
    file_1d_model.open(output_dir + "/model_1d.txt");

    // load 1d model to 2d field
    for (int i = 0; i < nr_2d; i++) {

        // interpolate from model_1d_ref by linear interpolation
        CUSTOMREAL r = r_2d[i];

        // select 1d model
        select_1d_model(model_1d_name);

        // find 2 closest r in model_1d_ref
        int ir = 0;
        for (auto& dep_vel : model_1d_ref) {
            CUSTOMREAL r_model = depth2radius(dep_vel[0]);
            if (r > r_model)
                break;
            ir++;
        }

        // linear interpolation
        CUSTOMREAL v_interp;

        if(ir != 0){
            CUSTOMREAL r_model_1 = depth2radius(model_1d_ref[ir-1][0]);
            CUSTOMREAL r_model_2 = depth2radius(model_1d_ref[ir][0]);
            CUSTOMREAL v_model_1 = model_1d_ref[ir-1][1];
            CUSTOMREAL v_model_2 = model_1d_ref[ir][1];

            v_interp = v_model_1 + (v_model_2 - v_model_1) * (r - r_model_1) / (r_model_2 - r_model_1);

            if (isZero((r_model_2 - r_model_1)))
                v_interp = model_1d_ref[ir][1];
        } else {
            v_interp = model_1d_ref[ir][1];
        }

        // debug output 1d model to a file
        file_1d_model << r << ",   " << v_interp << std::endl;

        for (int j = 0; j < nt_2d; j++)
            fun_2d[i*nt_2d+j] = _1_CR / v_interp;

    }

    // close file
    file_1d_model.close();
}


void PlainGrid::run_iteration(InputParams& IP){
    // solve Tau, H(tau) = a tau_x^2+ b tau_y^2 + (2aTx-2cTy) tau_x + (2bTy-2cTx) tau_y
    //                   -2c tau_x tau_y + (aTx^2+bTy^2-2cTxTy) = f^2

    if (myrank==0)
        std::cout << "Running 2d eikonal solver..." << std::endl;

    CUSTOMREAL L1_dif  =1000000000;
    CUSTOMREAL Linf_dif=1000000000;
    CUSTOMREAL L1_err  =_0_CR;
    CUSTOMREAL Linf_err=_0_CR;

    int iter = 0;

    while (true) {

        // update tau_old
        std::memcpy(tau_old_2d, tau_2d, sizeof(CUSTOMREAL)*nr_2d*nt_2d);

        int r_start, r_end;
        int t_start, t_end;
        int r_dirc, t_dirc;

        // sweep direction
        for (int iswp = 0; iswp < 4; iswp++){
            if (iswp == 0){
                r_start = nr_2d-2;
                r_end   = 0;
                t_start = nt_2d-2;
                t_end   = 0;
                r_dirc  = -1;
                t_dirc  = -1;
            } else if (iswp==1){
                r_start = nr_2d-2;
                r_end   = 0;
                t_start = 1;
                t_end   = nt_2d-1;
                r_dirc  = -1;
                t_dirc  = 1;
            } else if (iswp==2){
                r_start = 1;
                r_end   = nr_2d-1;
                t_start = nt_2d-2;
                t_end   = 0;
                r_dirc  = 1;
                t_dirc  = -1;
            } else {
                r_start = 1;
                r_end   = nr_2d-1;
                t_start = 1;
                t_end   = nt_2d-1;
                r_dirc  = 1;
                t_dirc  = 1;
            }

            for (int ir = r_start; ir != r_end; ir += r_dirc) {
                for (int it = t_start; it != t_end; it += t_dirc) {

                    // if the point is not in the boundary, skip it
                    if (is_changed_2d[ir*nt_2d+it])
                        calculate_stencil_2d(ir, it);

                }
            }

            // boundary
            for (int ir = 0; ir < nr_2d; ir++) {
                tau_2d[ir*nt_2d]         = std::max(_2_CR*tau_2d[ir*nt_2d+1]      -tau_2d[ir*nt_2d+2],       tau_2d[ir*nt_2d+2]);
                tau_2d[ir*nt_2d+nt_2d-1] = std::max(_2_CR*tau_2d[ir*nt_2d+nt_2d-2]-tau_2d[ir*nt_2d+nt_2d-3], tau_2d[ir*nt_2d+nt_2d-3]);
            }
            for (int it = 0; it < nt_2d; it++) {
                tau_2d[it]                 = std::max(_2_CR*tau_2d[1*nt_2d+it]        -tau_2d[2*nt_2d+it],         tau_2d[2*nt_2d+it]);
                tau_2d[(nr_2d-1)*nt_2d+it] = std::max(_2_CR*tau_2d[(nr_2d-2)*nt_2d+it]-tau_2d[(nr_2d-3)*nt_2d+it], tau_2d[(nr_2d-3)*nt_2d+it]);
            }

        } // end of iswp

        // calculate L1 and Linf error
        L1_dif  = _0_CR;
        Linf_dif= _0_CR;

        for (int ii = 0; ii < nr_2d*nt_2d; ii++){
            L1_dif  += std::abs(tau_2d[ii]-tau_old_2d[ii]);
            Linf_dif = std::max(Linf_dif, std::abs(tau_2d[ii]-tau_old_2d[ii]));
        }
        L1_dif /= nr_2d*nt_2d;

        // calculate L1 and Linf error
        L1_err  = _0_CR;
        Linf_err= _0_CR;

        for (int ii = 0; ii < nr_2d*nt_2d; ii++){
            L1_err  += std::abs(tau_2d[ii]*T0v_2d[ii] - u_2d[ii]);
            Linf_err = std::max(Linf_err, std::abs(tau_2d[ii]*T0v_2d[ii] - u_2d[ii]));
        }
        L1_err /= nr_2d*nt_2d;

        // check convergence
        if (std::abs(L1_dif) < TOL_2D_SOLVER && std::abs(Linf_dif) < TOL_2D_SOLVER){
            if (myrank==0)
                std::cout << "Converged at iteration " << iter << std::endl;
            goto iter_end;
        } else if (iter > MAX_ITER_2D_SOLVER){
            if (myrank==0)
                std::cout << "Maximum iteration reached at iteration " << iter << std::endl;
            goto iter_end;
        } else {
            if (myrank==0 && if_verbose)
                std::cout << "Iteration " << iter << "L_1(Tnew-Told)= " << L1_dif << " , L_inf(Tnew-Told) = " << Linf_dif << std::endl;
            iter++;
        }
        // if (myrank==0){
        //     std::cout << "Iteration " << iter << "L_1(Tnew-Told)= " << L1_dif << " , L_inf(Tnew-Told) = " << Linf_dif << std::endl;
        //     std::cout << "Iteration " << iter << ": L1 error = " << L1_err << ", Linf error = " << Linf_err << std::endl;
        // }
    } // end of wile

iter_end:
    if (myrank==0)
        std::cout << "Iteration " << iter << " finished." << std::endl;

    for (int ii = 0; ii < nr_2d*nt_2d; ii++){
        T_2d[ii] = tau_2d[ii]*T0v_2d[ii];
    }

}


void PlainGrid::calculate_stencil_2d(int& ir, int& it){
    ii     = ir*nt_2d+it;
    ii_nr  = (ir-1)*nt_2d+it;
    ii_n2r = (ir-2)*nt_2d+it;
    ii_nt  = ir*nt_2d+(it-1);
    ii_n2t = ir*nt_2d+(it-2);
    ii_pr  = (ir+1)*nt_2d+it;
    ii_p2r = (ir+2)*nt_2d+it;
    ii_pt  = ir*nt_2d+(it+1);
    ii_p2t = ir*nt_2d+(it+2);

    CUSTOMREAL sigr = std::sqrt(fac_a_2d[ii])*T0v_2d[ii];
    CUSTOMREAL sigt = std::sqrt(fac_b_2d[ii])*T0v_2d[ii];
    CUSTOMREAL coe  = _1_CR/(sigr/dr_2d + sigt/dt_2d);

    CUSTOMREAL px1, px2, wx1=0, wx2=0;
    CUSTOMREAL py1, py2, wy1=0, wy2=0;

    if(ir==1){
        px1=(tau_2d[ii]-tau_2d[ii_nr])/dr_2d;
        wx2=_1_CR/(_1_CR+_2_CR*my_square( (eps_2d+my_square(tau_2d[ii]     - _2_CR*tau_2d[ii_pr]+ tau_2d[ii_p2r])) \
                                       /  (eps_2d +my_square(tau_2d[ii_nr] - _2_CR*tau_2d[ii]   + tau_2d[ii_pr] ))));
        px2=(_1_CR-wx2)*(       tau_2d[ii_pr]                 -tau_2d[ii_nr]) /_2_CR/dr_2d \
                 + wx2 *(-_3_CR*tau_2d[ii]+_4_CR*tau_2d[ii_pr]-tau_2d[ii_p2r])/_2_CR/dr_2d;

    }
    else if (ir == nr_2d-2){
        wx1=_1_CR/(_1_CR+_2_CR*my_square( (eps_2d+my_square(tau_2d[ii]    - _2_CR*tau_2d[ii_nr]+ tau_2d[ii_n2r])) \
                                        / (eps_2d+my_square(tau_2d[ii_pr] - _2_CR*tau_2d[ii]   + tau_2d[ii_nr] ))));
        px1=(_1_CR-wx1)*(       tau_2d[ii_pr]                 -tau_2d[ii_nr]) /_2_CR/dr_2d \
                 + wx1 *( _3_CR*tau_2d[ii]-_4_CR*tau_2d[ii_nr]+tau_2d[ii_n2r])/_2_CR/dr_2d;
        px2=(tau_2d[ii_pr]-tau_2d[ii_nr])/dr_2d;
    }
    else {
        wx1=_1_CR/(_1_CR+_2_CR*my_square( (eps_2d+my_square(tau_2d[ii]    - _2_CR*tau_2d[ii_nr]+ tau_2d[ii_n2r])) \
                                        / (eps_2d+my_square(tau_2d[ii_pr] - _2_CR*tau_2d[ii]   + tau_2d[ii_nr] ))));
        px1=(_1_CR-wx1)*(       tau_2d[ii_pr]                 -tau_2d[ii_nr]) /_2_CR/dr_2d \
                 + wx1 *( _3_CR*tau_2d[ii]-_4_CR*tau_2d[ii_nr]+tau_2d[ii_n2r])/_2_CR/dr_2d;
        wx2=_1_CR/(_1_CR+_2_CR*my_square( (eps_2d+my_square(tau_2d[ii]    - _2_CR*tau_2d[ii_pr]+ tau_2d[ii_p2r])) \
                                        / (eps_2d+my_square(tau_2d[ii_nr] - _2_CR*tau_2d[ii]   + tau_2d[ii_pr] ))));
        px2=(_1_CR-wx2)*(       tau_2d[ii_pr]                 -tau_2d[ii_nr]) /_2_CR/dr_2d \
                 + wx2 *(-_3_CR*tau_2d[ii]+_4_CR*tau_2d[ii_pr]-tau_2d[ii_p2r])/_2_CR/dr_2d;
    }

    if(it==1){
        py1=(tau_2d[ii]-tau_2d[ii_nt])/dt_2d;
        wy2=_1_CR/(_1_CR+_2_CR*my_square( (eps_2d+my_square(tau_2d[ii]    - _2_CR*tau_2d[ii_pt]+ tau_2d[ii_p2t])) \
                                        / (eps_2d+my_square(tau_2d[ii_nt] - _2_CR*tau_2d[ii]   + tau_2d[ii_pt] ))));
        py2=(_1_CR-wy2)*(       tau_2d[ii_pt]                 -tau_2d[ii_nt]) /_2_CR/dt_2d \
                 + wy2 *(-_3_CR*tau_2d[ii]+_4_CR*tau_2d[ii_pt]-tau_2d[ii_p2t])/_2_CR/dt_2d;
    }
    else if (it == nt_2d-2){
        wy1=_1_CR/(_1_CR+_2_CR*my_square( (eps_2d+my_square(tau_2d[ii]    - _2_CR*tau_2d[ii_nt]+ tau_2d[ii_n2t])) \
                                        / (eps_2d+my_square(tau_2d[ii_pt] - _2_CR*tau_2d[ii]   + tau_2d[ii_nt] ))));
        py1=(_1_CR-wy1)*(       tau_2d[ii_pt]                 -tau_2d[ii_nt]) /_2_CR/dt_2d \
                 + wy1 *( _3_CR*tau_2d[ii]-_4_CR*tau_2d[ii_nt]+tau_2d[ii_n2t])/_2_CR/dt_2d;
        py2=(tau_2d[ii_pt]-tau_2d[ii])/dt_2d;
    }
    else {
        wy1=_1_CR/(_1_CR+_2_CR*my_square( (eps_2d+my_square(tau_2d[ii]    - _2_CR*tau_2d[ii_nt]+ tau_2d[ii_n2t])) \
                                        / (eps_2d+my_square(tau_2d[ii_pt] - _2_CR*tau_2d[ii]   + tau_2d[ii_nt] ))));
        py1=(_1_CR-wy1)*(       tau_2d[ii_pt]                 -tau_2d[ii_nt]) /_2_CR/dt_2d \
                 + wy1 *( _3_CR*tau_2d[ii]-_4_CR*tau_2d[ii_nt]+tau_2d[ii_n2t])/_2_CR/dt_2d;
        wy2=_1_CR/(_1_CR+_2_CR*my_square( (eps_2d+my_square(tau_2d[ii]    - _2_CR*tau_2d[ii_pt]+ tau_2d[ii_p2t])) \
                                        / (eps_2d+my_square(tau_2d[ii_nt] - _2_CR*tau_2d[ii]   + tau_2d[ii_pt] ))));
        py2=(_1_CR-wy2)*(       tau_2d[ii_pt]                 -tau_2d[ii_nt]) /_2_CR/dt_2d \
                 + wy2 *(-_3_CR*tau_2d[ii]+_4_CR*tau_2d[ii_pt]-tau_2d[ii_p2t])/_2_CR/dt_2d;
    }

    CUSTOMREAL Htau = std::sqrt(fac_a_2d[ii]*my_square(T0r_2d[ii]*tau_2d[ii]+T0v_2d[ii]*(px1+px2)/_2_CR) \
                               +fac_b_2d[ii]*my_square(T0t_2d[ii]*tau_2d[ii]+T0v_2d[ii]*(py1+py2)/_2_CR));

    tau_2d[ii] = coe * ( (fun_2d[ii] - Htau) + (sigr*(px2-px1)/_2_CR + sigt*(py2-py1)/_2_CR) ) + tau_2d[ii];

}


void PlainGrid::run_iteration_upwind(InputParams& IP){
    // solve Tau, H(tau) = a tau_x^2+ b tau_y^2 + (2aTx-2cTy) tau_x + (2bTy-2cTx) tau_y
    //                   -2c tau_x tau_y + (aTx^2+bTy^2-2cTxTy) = f^2

    if (myrank==0)
        std::cout << "Running 2d eikonal solver..." << std::endl;

    CUSTOMREAL L1_dif  =1000000000;
    CUSTOMREAL Linf_dif=1000000000;
    CUSTOMREAL L1_err  =_0_CR;
    CUSTOMREAL Linf_err=_0_CR;

    int iter = 0;

    while (true) {

        // update tau_old
        std::memcpy(tau_old_2d, tau_2d, sizeof(CUSTOMREAL)*nr_2d*nt_2d);

        int r_start, r_end;
        int t_start, t_end;
        int r_dirc, t_dirc;

        // sweep direction
        for (int iswp = 0; iswp < 4; iswp++){
            if (iswp == 0){
                r_start = nr_2d-1;
                r_end   = -1;
                t_start = nt_2d-1;
                t_end   = -1;
                r_dirc  = -1;
                t_dirc  = -1;
            } else if (iswp==1){
                r_start = nr_2d-1;
                r_end   = -1;
                t_start = 0;
                t_end   = nt_2d;
                r_dirc  = -1;
                t_dirc  = 1;
            } else if (iswp==2){
                r_start = 0;
                r_end   = nr_2d;
                t_start = nt_2d-1;
                t_end   = -1;
                r_dirc  = 1;
                t_dirc  = -1;
            } else {
                r_start = 0;
                r_end   = nr_2d;
                t_start = 0;
                t_end   = nt_2d;
                r_dirc  = 1;
                t_dirc  = 1;
            }

            for (int ir = r_start; ir != r_end; ir += r_dirc) {
                for (int it = t_start; it != t_end; it += t_dirc) {

                    // if the point is not in the boundary, skip it
                    if (is_changed_2d[ir*nt_2d+it])
                        calculate_stencil_upwind_2d(ir, it);

                }
            }
        } // end of iswp

        // calculate L1 and Linf error
        L1_dif  = _0_CR;
        Linf_dif= _0_CR;

        for (int ii = 0; ii < nr_2d*nt_2d; ii++){
            L1_dif  += std::abs(tau_2d[ii]-tau_old_2d[ii]);
            Linf_dif = std::max(Linf_dif, std::abs(tau_2d[ii]-tau_old_2d[ii]));
        }
        L1_dif /= nr_2d*nt_2d;

        // calculate L1 and Linf error
        L1_err  = _0_CR;
        Linf_err= _0_CR;

        for (int ii = 0; ii < nr_2d*nt_2d; ii++){
            L1_err  += std::abs(tau_2d[ii]*T0v_2d[ii] - u_2d[ii]);
            Linf_err = std::max(Linf_err, std::abs(tau_2d[ii]*T0v_2d[ii] - u_2d[ii]));
        }
        L1_err /= nr_2d*nt_2d;

        // check convergence
        if (std::abs(L1_dif) < TOL_2D_SOLVER && std::abs(Linf_dif) < TOL_2D_SOLVER){
            if (myrank==0)
                std::cout << "Converged at iteration " << iter << std::endl;
            goto iter_end;
        } else if (iter > MAX_ITER_2D_SOLVER){
            if (myrank==0)
                std::cout << "Maximum iteration reached at iteration " << iter << std::endl;
            goto iter_end;
        } else {
            if (myrank==0 && if_verbose)
                std::cout << "Iteration " << iter << ": L1 error = " << L1_err << ", Linf error = " << Linf_err << std::endl;
            iter++;

        }
        // if (myrank==0){
        //     std::cout << "Iteration " << iter << "L_1(Tnew-Told)= " << L1_dif << " , L_inf(Tnew-Told) = " << Linf_dif << std::endl;
        //     // std::cout << "Iteration " << iter << ": L1 error = " << L1_err << ", Linf error = " << Linf_err << std::endl;
        // }
    } // end of wile

iter_end:
    if (myrank==0)
        std::cout << "Iteration " << iter << " finished." << std::endl;

    for (int ii = 0; ii < nr_2d*nt_2d; ii++){
        T_2d[ii] = tau_2d[ii]*T0v_2d[ii];
    }

}


void PlainGrid::calculate_stencil_upwind_2d(int& ir, int& it){

    count_cand = 0;
    ii     = ir*nt_2d+it;
    ii_nr  = (ir-1)*nt_2d+it;
    ii_pr  = (ir+1)*nt_2d+it;
    ii_nt  = ir*nt_2d+(it-1);
    ii_pt  = ir*nt_2d+(it+1);


    // (T0*tau)_r = ar*tau(iix,iiy)+br; (T0*tau)_t = at*tau(iix,iiy)+bt
    if (it > 0){
        at1 =  T0t_2d[ii] + T0v_2d[ii]/dt_2d;
        bt1 = -T0v_2d[ii]/dt_2d*tau_2d[ii_nt];
    }
    if (it < nt_2d-1){
        at2 =  T0t_2d[ii] - T0v_2d[ii]/dt_2d;
        bt2 =  T0v_2d[ii]/dt_2d*tau_2d[ii_pt];
    }
    if (ir > 0){
        ar1 =  T0r_2d[ii] + T0v_2d[ii]/dr_2d;
        br1 = -T0v_2d[ii]/dr_2d*tau_2d[ii_nr];
    }
    if (ir < nr_2d-1){
        ar2 =  T0r_2d[ii] - T0v_2d[ii]/dr_2d;
        br2 =  T0v_2d[ii]/dr_2d*tau_2d[ii_pr];
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
                if (it == 0 || ir == nr_2d-1)
                    continue;
                at = at1; bt = bt1;
                ar = ar2; br = br2;
                break;
            case 2:    // characteristic travels from +t, -r
                if (it == nt_2d-1 || ir == 0)
                    continue;
                at = at2; bt = bt2;
                ar = ar1; br = br1;
                break;
            case 3:     // characteristic travels from +t, +r
                if (it == nt_2d-1 || ir == nr_2d-1)
                    continue;
                at = at2; bt = bt2;
                ar = ar2; br = br2;
                break;
        }

        // plug T_t, T_r into eikonal equation, solving the quadratic equation with respect to tau(iip,jjt,kkr)
        // that is a*(ar*tau+br)^2 + b*(at*tau+bt)^2 = s^2
        eqn_a = fac_a_2d[ii] * std::pow(ar, _2_CR) + fac_b_2d[ii] * std::pow(at, _2_CR);
        eqn_b = fac_a_2d[ii] * _2_CR * ar * br     + fac_b_2d[ii] * _2_CR * at * bt;
        eqn_c = fac_a_2d[ii] * std::pow(br, _2_CR) + fac_b_2d[ii] * std::pow(bt, _2_CR) - std::pow(fun_2d[ii], _2_CR);
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
                if (ir ==  nr_2d-1){
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
                    tmp_tau = ( std::sqrt(std::pow(fun_2d[ii],_2_CR)/fac_a_2d[ii]) - br)/ar;
                    break;
                case 1:
                    tmp_tau = (-std::sqrt(std::pow(fun_2d[ii],_2_CR)/fac_a_2d[ii]) - br)/ar;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 0:  //characteristic travels from -r (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_tau * T0v_2d[ii] > tau_2d[ii_nr] * T0v_2d[ii_nr]
                        && tmp_tau > tau_2d[ii_nr]/_2_CR && tmp_tau > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 1:  //characteristic travels from +r
                    if (tmp_tau * T0v_2d[ii] > tau_2d[ii_pr] * T0v_2d[ii_pr]
                        && tmp_tau > tau_2d[ii_pr]/_2_CR && tmp_tau > 0){
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
                if (it ==  nt_2d-1){
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
                    tmp_tau = ( std::sqrt(std::pow(fun_2d[ii],_2_CR)/fac_b_2d[ii]) - bt)/at;
                    break;
                case 1:
                    tmp_tau = (-std::sqrt(std::pow(fun_2d[ii],_2_CR)/fac_b_2d[ii]) - bt)/at;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 2:  //characteristic travels from -t (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_tau * T0v_2d[ii] > tau_2d[ii_nt] * T0v_2d[ii_nt]
                        && tmp_tau > tau_2d[ii_nt]/_2_CR && tmp_tau > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 3:  //characteristic travels from +t
                    if (tmp_tau * T0v_2d[ii] > tau_2d[ii_pt]  * T0v_2d[ii_pt]
                        && tmp_tau > tau_2d[ii_pt]/_2_CR && tmp_tau > 0){
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
        tau_2d[ii] = std::min(tau_2d[ii], canditate[i_cand]);
        if (tau_2d[ii] < 0 ){
            std::cout << "error, tele tau_loc < 0. ir: " << ir << ", it: " << it << std::endl;
            exit(1);
        }
    }

}


CUSTOMREAL interp2d(PlainGrid& pg, CUSTOMREAL t, CUSTOMREAL r) {
    // interpolate 2d travel time field on the boundaries
    // pg is the grid
    // t is distance between source and boundary point
    // r is the radius of the boundary point

    // find the index of the closest point on the grid
    int it_n=0, it_p=0, ir_n=0, ir_p=0;

    CUSTOMREAL dt = pg.t_2d[1] - pg.t_2d[0];
    CUSTOMREAL dr = pg.r_2d[1] - pg.r_2d[0];

    // here t is the distance between source and boundary point
    // which is ok because t values of 2d grid is also relative + margin from source
    // location
    it_n = std::floor((t - pg.t_2d[0]) / dt);
    it_p = it_n + 1;
    ir_n = std::floor((r - pg.r_2d[0]) / dr);
    ir_p = ir_n + 1;

    // interpolate
    CUSTOMREAL T_tnrn = pg.T_2d[ir_n*pg.nt_2d+it_n];
    CUSTOMREAL T_tnrp = pg.T_2d[ir_p*pg.nt_2d+it_n];
    CUSTOMREAL T_tprn = pg.T_2d[ir_n*pg.nt_2d+it_p];
    CUSTOMREAL T_tprp = pg.T_2d[ir_p*pg.nt_2d+it_p];

    CUSTOMREAL t_err = std::min(_1_CR, (t-pg.t_2d[it_n])/dt);
    CUSTOMREAL r_err = std::min(_1_CR, (r-pg.r_2d[ir_n])/dr);

    return (1-t_err)*(1-r_err)*T_tnrn + t_err*(1-r_err)*T_tprn + (1-t_err)*r_err*T_tnrp + t_err*r_err*T_tprp;
}


std::string get_2d_tt_filename(const std::string& dir_2d, Source& src) {

    std::string fname_2d_src;

    if (output_format == OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        // add the depth of the source to the file name
        // to share the travel time field if sources have the same depth of hypocenter
        auto str = std::to_string(src.get_src_dep());
        fname_2d_src = dir_2d + "/2d_travel_time_field_dep_" + str.substr(0,str.find(".")+4) +".h5";
#else
        std::cout << "HDF5 is not enabled. Please recompile with HDF5" << std::endl;
#endif
    } else if (output_format == OUTPUT_FORMAT_ASCII) {
        auto str = std::to_string(src.get_src_dep()); // use depth value of the source to the file name
        fname_2d_src = dir_2d + "/2d_travel_time_field_dep_" + str.substr(0,str.find(".")+4) +".dat";
    }

    return fname_2d_src;

}


void run_2d_solver(InputParams& IP, Source& src, IO_utils& io) {
    // calculate 2d travel time field by only the first process of each simulation group

    // initialize grid for all processes
    PlainGrid plain_grid(src,IP);

    // check if pre-calculated 2d travel time field exists

    // create output directory for 2d solver
    std::string dir_2d = output_dir+"/"+OUTPUT_DIR_2D;
    create_output_dir(dir_2d);

    std::string fname_2d_src = get_2d_tt_filename(dir_2d, src);

    if (is_file_exist(fname_2d_src.c_str())) {
        std::cout << "2d solver skipped the src because traveltime data already exists: " << fname_2d_src << std::endl;
    } else { // if not, calculate and store it
        // run iteration
        std::cout << "start run iteration myrank: " << myrank << ", for file: " << fname_2d_src << std::endl;
        plain_grid.run_iteration_upwind(IP);

        // write out calculated 2D travel time field
        io.write_2d_travel_time_field(plain_grid.T_2d, plain_grid.r_2d, plain_grid.t_2d, plain_grid.nr_2d, plain_grid.nt_2d, src.get_src_dep());
    }

 }


void load_2d_traveltime(InputParams& IP, Source& src, Grid& grid, IO_utils& io) {
    // initialize grid for all processes
    PlainGrid plain_grid(src,IP);

    // check if pre-calculated 2d travel time field exists
    if(myrank == 0) {

        // output directory for 2d solver
        std::string dir_2d = output_dir+"/"+OUTPUT_DIR_2D;
        //create_output_dir(dir_2d);

        std::string fname_2d_src = get_2d_tt_filename(dir_2d, src);

        if (is_file_exist(fname_2d_src.c_str())) {
            // remark :: TODO :: we need to check whether the 2d_traveltime file has the same size of T_2d here.
            // Otherwise, we need to recalculate the 2d traveltime field and write it out.

            // read 2d travel time field from file
            io.read_2d_travel_time_field(fname_2d_src, plain_grid.T_2d, plain_grid.nt_2d, plain_grid.nr_2d);

            if (if_verbose)
                std::cout << "2d travel time field has been read from file: " << fname_2d_src << std::endl;
        } else { // if not, stop the program
            std::cout << "2d travel time field does not exist. Please verify if the 2d solver finished normally." << std::endl;
            std::cout << "missing : " << fname_2d_src << std::endl;
            exit(1);
        }
    }

    // share T_2d, r_2d and t_2d to all processes
    broadcast_cr(plain_grid.T_2d, plain_grid.nr_2d*plain_grid.nt_2d, 0);
    broadcast_cr(plain_grid.r_2d, plain_grid.nr_2d, 0);
    broadcast_cr(plain_grid.t_2d, plain_grid.nt_2d, 0);

    // interpolate results on the boundaries and store SrcRec object

    CUSTOMREAL tmp_dist;

    for (int l = 0; l < N_LAYER_SRC_BOUND; l++){
        for (int k = 0; k < loc_K; k++) {
            for (int i = 0; i < loc_I; i++) {
                // North boundary
                if (grid.j_last()){
                    Epicentral_distance_sphere(plain_grid.src_t, plain_grid.src_p, grid.get_lat_by_index(loc_J-1-l), grid.get_lon_by_index(i), tmp_dist);
                    if(tmp_dist >= tmax_2d){
                        std::cout << "epicenter distance exceeds tmax_2d:" << tmp_dist/PI*180.0 << "degree" << std::endl;
                        exit(1);
                    }
                    grid.T_loc[I2V(i, loc_J-1-l, k)] = interp2d(plain_grid, tmp_dist, grid.get_r_by_index(k));
                    grid.is_changed[I2V(i, loc_J-1-l, k)] = false;
                }
                // South boundary
                if (grid.j_first()){
                    Epicentral_distance_sphere(plain_grid.src_t, plain_grid.src_p, grid.get_lat_by_index(l), grid.get_lon_by_index(i), tmp_dist);
                    if(tmp_dist >= tmax_2d){
                        std::cout << "epicenter distance exceeds tmax_2d:" << tmp_dist/PI*180.0 << "degree" << std::endl;
                        exit(1);
                    }
                    grid.T_loc[I2V(i, l, k)] = interp2d(plain_grid, tmp_dist, grid.get_r_by_index(k));
                    grid.is_changed[I2V(i, l, k)] = false;
                }
            }
        }

        for (int k = 0; k < loc_K; k++) {
            for (int j = 0; j < loc_J; j++) {
                // East boundary
                if (grid.i_last()){
                    Epicentral_distance_sphere(plain_grid.src_t, plain_grid.src_p, grid.get_lat_by_index(j), grid.get_lon_by_index(loc_I-1-l), tmp_dist);
                    if(tmp_dist >= tmax_2d){
                        std::cout << "epicenter distance exceeds tmax_2d:" << tmp_dist/PI*180.0 << "degree" << std::endl;
                        exit(1);
                    }
                    grid.T_loc[I2V(loc_I-1-l, j, k)] = interp2d(plain_grid, tmp_dist, grid.get_r_by_index(k));
                    grid.is_changed[I2V(loc_I-1-l, j, k)] = false;
                }
                // West boundary
                if (grid.i_first()){
                    Epicentral_distance_sphere(plain_grid.src_t, plain_grid.src_p, grid.get_lat_by_index(j), grid.get_lon_by_index(l), tmp_dist);
                    if(tmp_dist >= tmax_2d){
                        std::cout << "epicenter distance exceeds tmax_2d:" << tmp_dist/PI*180.0 << "degree" << std::endl;
                        exit(1);
                    }
                    grid.T_loc[I2V(l, j, k)] = interp2d(plain_grid, tmp_dist, grid.get_r_by_index(k));
                    grid.is_changed[I2V(l, j, k)] = false;
                }
            }
        }

        for (int i = 0; i < loc_I; i++) {
            for (int j = 0; j < loc_J; j++) {
                // Bottom boundary
                if (grid.k_first()){
                    Epicentral_distance_sphere(plain_grid.src_t, plain_grid.src_p, grid.get_lat_by_index(j), grid.get_lon_by_index(i), tmp_dist);
                    if(tmp_dist >= tmax_2d){
                        std::cout << "epicenter distance exceeds tmax_2d:" << tmp_dist/PI*180.0 << "degree" << std::endl;
                        exit(1);
                    }
                    grid.T_loc[I2V(i, j, l)] = interp2d(plain_grid, tmp_dist, grid.get_r_by_index(l));
                    grid.is_changed[I2V(i, j, l)] = false;
                }
            }
        }
    }

}


