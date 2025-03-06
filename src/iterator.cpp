#include "iterator.h"

// test no valence anymore

Iterator::Iterator(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, \
                   bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
         : is_second_run(is_second_run_in) {

    if(n_subprocs > 1) {

        // share necessary values between subprocs
        np = loc_I;
        nt = loc_J;
        nr = loc_K;
        dr = grid.dr;
        dt = grid.dt;
        dp = grid.dp;

        if (first_init) {

            broadcast_i_single_sub(nr,0);
            broadcast_i_single_sub(nt,0);
            broadcast_i_single_sub(np,0);
            broadcast_cr_single_sub(dr,0);
            broadcast_cr_single_sub(dt,0);
            broadcast_cr_single_sub(dp,0);

            // initialize n_elms for subprocs
            if (!subdom_main){
                loc_I = np;
                loc_J = nt;
                loc_K = nr;
                grid.dr = dr;
                grid.dt = dt;
                grid.dp = dp;
            }
        }

        // check if teleseismic source
        is_teleseismic = is_teleseismic_in;
        //broadcast_bool_single_sub(is_teleseismic, 0); // done in IP.get_if_teleseismic()

        // check initialized values
        if (if_verbose){
            std::cout << "nr: " << nr << " nt: " << nt << " np: " << np << std::endl;
            std::cout << "dr: " << dr << " dt: " << dt << " dp: " << dp << std::endl;
            std::cout << "loc_I: " << loc_I << " loc_J: " << loc_J << " loc_K: " << loc_K << std::endl;
        }


    } else {
        np = loc_I;
        nt = loc_J;
        nr = loc_K;
        dr = grid.dr;
        dt = grid.dt;
        dp = grid.dp;
        is_teleseismic = is_teleseismic_in;
    }

    // // set initial and end indices of level set
    // if (!is_teleseismic) {
    //     st_level = 6;
    //     ed_level = nr+nt+np-3;
    // } else {
    //     st_level = 0;
    //     ed_level = nr+nt+np;
    // }
    // if (IP.get_stencil_type() == UPWIND) {
    //     st_level = 0;
    //     ed_level = nr+nt+np-3;
    // }

    st_level = 0;
    ed_level = nr+nt+np-3;

    // initialize factors etc.
    initialize_arrays(IP, io, grid, src, src_name);



}


Iterator::~Iterator() {

#if defined USE_SIMD || defined USE_CUDA

    if (simd_allocated){
        // if use_gpu == false, simd_allocated is also false here

        free(dump_c__);// center of stencil

        free_preloaded_array(vv_iip);
        free_preloaded_array(vv_jjt);
        free_preloaded_array(vv_kkr);

        // free vv_* preloaded arrays
        free_preloaded_array(vv_i__j__k__);
        free_preloaded_array(vv_ip1j__k__);
        free_preloaded_array(vv_im1j__k__);
        free_preloaded_array(vv_i__jp1k__);
        free_preloaded_array(vv_i__jm1k__);
        free_preloaded_array(vv_i__j__kp1);
        free_preloaded_array(vv_i__j__km1);

        if(simd_allocated_3rd || is_teleseismic){
            free_preloaded_array(vv_ip2j__k__);
            free_preloaded_array(vv_im2j__k__);
            free_preloaded_array(vv_i__jp2k__);
            free_preloaded_array(vv_i__jm2k__);
            free_preloaded_array(vv_i__j__kp2);
            free_preloaded_array(vv_i__j__km2);
        }

        free_preloaded_array(vv_fac_a);
        free_preloaded_array(vv_fac_b);
        free_preloaded_array(vv_fac_c);
        free_preloaded_array(vv_fac_f);
        free_preloaded_array(vv_fun);
        if (!is_teleseismic){
            free_preloaded_array(vv_T0v);
            free_preloaded_array(vv_T0r);
            free_preloaded_array(vv_T0t);
            free_preloaded_array(vv_T0p);
        }
        if(!use_gpu)
            free_preloaded_array(vv_change);
        else
            free_preloaded_array(vv_change_bl);
    }
#endif

#ifdef USE_CUDA
    // free memory on device before host arrays
    // otherwise the program crashes
    if (use_gpu) {
        // free memory on device
        cuda_finalize_grid(gpu_grid);

        delete gpu_grid;
    }
#endif

}


void Iterator::initialize_arrays(InputParams& IP, IO_utils& io, Grid& grid, Source& src, const std::string& name_sim_src) {
    if(if_verbose && myrank == 0) std::cout << "(re) initializing arrays" << std::endl;

    // std::cout << "source lat: " << src.get_src_t()*RAD2DEG << ", source lon: " << src.get_src_p()*RAD2DEG << ", source dep: " << src.get_src_r() << std::endl;

    if (!is_second_run) { // field initialization has already been done in the second run
        if (subdom_main) {
            if (!is_teleseismic) {
                // set initial a b c and calculate a0 b0 c0 f0
                grid.setup_factors(src);

                // calculate T0 T0r T0t T0p and initialize tau
                grid.initialize_fields(src, IP);
            } else {
                // copy T_loc arrival time on domain's boundaries.
                grid.initialize_fields_teleseismic();

                // load 2d traveltime on boundary from file
                load_2d_traveltime(IP, src, grid, io);
            }
        }
    }

    // assign processes for each sweeping level
    if (IP.get_sweep_type() == SWEEP_TYPE_LEVEL) assign_processes_for_levels(grid, IP);

#ifdef USE_CUDA
    if(use_gpu){
        gpu_grid = new Grid_on_device();
        if (IP.get_stencil_order() == 1){
            cuda_initialize_grid_1st(ijk_for_this_subproc, gpu_grid, loc_I, loc_J, loc_K, dp, dt, dr, \
                vv_i__j__k__, vv_ip1j__k__, vv_im1j__k__, vv_i__jp1k__, vv_i__jm1k__, vv_i__j__kp1, vv_i__j__km1, \
                vv_fac_a, vv_fac_b, vv_fac_c, vv_fac_f, vv_T0v, vv_T0r, vv_T0t, vv_T0p, vv_fun, vv_change_bl);
        } else {
            cuda_initialize_grid_3rd(ijk_for_this_subproc, gpu_grid, loc_I, loc_J, loc_K, dp, dt, dr, \
                vv_i__j__k__, vv_ip1j__k__, vv_im1j__k__, vv_i__jp1k__, vv_i__jm1k__, vv_i__j__kp1, vv_i__j__km1, \
                              vv_ip2j__k__, vv_im2j__k__, vv_i__jp2k__, vv_i__jm2k__, vv_i__j__kp2, vv_i__j__km2, \
                vv_fac_a, vv_fac_b, vv_fac_c, vv_fac_f, vv_T0v, vv_T0r, vv_T0t, vv_T0p, vv_fun, vv_change_bl);
        }

        //std::cout << "gpu grid initialization done." << std::endl;
    }
#endif

}


// assign intra-node processes for each sweeping level
void Iterator::assign_processes_for_levels(Grid& grid, InputParams& IP) {
    // allocate memory for process range
    std::vector<int> n_nodes_of_levels;

    // count the number of nodes on each level
    for (int level = st_level; level <= ed_level; level++) {
        int count_n_nodes = 0; // the number of nodes in this level

        // the range of index k (rr)
        int kleft  = std::max(0, level - (np-1) - (nt-1));
        int kright = std::min(nr-1, level);

        for (int kk = kleft; kk <= kright; kk++) {

            // the range of index j (tt)
            int jleft  = std::max(0, level - kk - (np-1));
            int jright = std::min(nt-1, level-kk);

            // now i = level - kk - jj
            int n_jlines = jright - jleft + 1;
            count_n_nodes += n_jlines;
        } // end loop kk

        n_nodes_of_levels.push_back(count_n_nodes); // store the number of nodes of each level
    } // end loop level


    // find max in n_nodes_of_levels
    max_n_nodes_plane = 0;
    for (size_t i = 0; i < n_nodes_of_levels.size(); i++)
        if (n_nodes_of_levels[i] > max_n_nodes_plane)
            max_n_nodes_plane = n_nodes_of_levels[i];


    int n_grids_this_subproc = 0; // count the number of grids calculated by this subproc

    // assign nodes on each level plane to processes
    for (int level = st_level; level <= ed_level; level++) {

        // the range of index k (rr)
        int kleft  = std::max(0, level - (np-1) - (nt-1));
        int kright = std::min(nr-1, level);


        std::vector<int> asigned_nodes_on_this_level;
        int grid_count = 0;

        // // n grids calculated by each subproc
        // int n_grids_each = static_cast<int>(n_nodes_of_levels[level] / n_subprocs);
        // int n_grids_by_this = n_grids_each;
        // int i_grid_start = n_grids_each*sub_rank;

        // // add modulo for last sub_rank
        // if (sub_rank == n_subprocs-1)
        //     n_grids_by_this += static_cast<int>(n_nodes_of_levels[level] % n_subprocs);

        // determine the beginnig index and the number of node for each subproc
        int n_grids_each = static_cast<int>(n_nodes_of_levels[level] / n_subprocs);
        int remainder = n_nodes_of_levels[level] % n_subprocs;

        int n_grids_by_this = 0; // the number of grids calculated by this subproc
        int i_grid_start    = 0; // the starting index of grids calculated by this subproc
        if(sub_rank < remainder){
            n_grids_by_this = n_grids_each + 1;
            i_grid_start    = n_grids_each*sub_rank + sub_rank;
        } else{
            n_grids_by_this = n_grids_each;
            i_grid_start    = n_grids_each*sub_rank + remainder;
        }

        // std::cout << "sub_rank " << sub_rank << ", n_grids_by_this: " << n_grids_by_this << ", idx: " << i_grid_start << std::endl;

        for (int kk = kleft; kk <= kright; kk++) {

            // the range of index j (tt)
            int jleft  = std::max(0, level - kk - (np-1));
            int jright = std::min(nt-1, level-kk);

            for (int jj = jleft; jj <= jright; jj++) {
                int ii = level - kk - jj;

                // check if this node should be assigned to this process
                if (grid_count >= i_grid_start && grid_count < i_grid_start+n_grids_by_this) {
                    int tmp_ijk = I2V(ii,jj,kk);
                    asigned_nodes_on_this_level.push_back(tmp_ijk);
                    n_grids_this_subproc++;
                }

                grid_count++;

            } // end loop jj
        } // end loop kk

        // store the node ids of each level
        ijk_for_this_subproc.push_back(asigned_nodes_on_this_level);
    } // end loop level

    if(if_verbose)
        std::cout << "n total grids calculated by sub_rank " << sub_rank << ": " << n_grids_this_subproc << std::endl;

    // std::cout << "n total grids calculated by sub_rank " << sub_rank << ": " << n_grids_this_subproc << std::endl;

    //
    // TODO: upwind scheme + sweep parallelization does not support SIMD yet
    //
    return;


#if defined USE_SIMD || defined USE_CUDA

    preload_indices(vv_iip, vv_jjt, vv_kkr,  0, 0, 0);
    preload_indices_1d(vv_i__j__k__, 0, 0, 0);
    preload_indices_1d(vv_ip1j__k__, 1, 0, 0);
    preload_indices_1d(vv_i__jp1k__, 0, 1, 0);
    preload_indices_1d(vv_i__j__kp1, 0, 0, 1);
    preload_indices_1d(vv_im1j__k__,-1, 0, 0);
    preload_indices_1d(vv_i__jm1k__, 0,-1, 0);
    preload_indices_1d(vv_i__j__km1, 0, 0,-1);

    if(IP.get_stencil_order() == 3 || is_teleseismic){
        preload_indices_1d(vv_ip2j__k__, 2, 0, 0);
        preload_indices_1d(vv_i__jp2k__, 0, 2, 0);
        preload_indices_1d(vv_i__j__kp2, 0, 0, 2);
        preload_indices_1d(vv_im2j__k__,-2, 0, 0);
        preload_indices_1d(vv_i__jm2k__, 0,-2, 0);
        preload_indices_1d(vv_i__j__km2, 0, 0,-2);
        simd_allocated_3rd = true;
    }

    int dump_length = NSIMD;
    // stencil dumps
    // first orders
    dump_c__ = (CUSTOMREAL*) aligned_alloc(ALIGN, dump_length*sizeof(CUSTOMREAL));// center of stencil

    // preload the stencil dumps
    vv_fac_a = preload_array(grid.fac_a_loc);
    vv_fac_b = preload_array(grid.fac_b_loc);
    vv_fac_c = preload_array(grid.fac_c_loc);
    vv_fac_f = preload_array(grid.fac_f_loc);
    vv_fun   = preload_array(grid.fun_loc);
    if(!is_teleseismic) {
        vv_T0v   = preload_array(grid.T0v_loc);
        vv_T0r   = preload_array(grid.T0r_loc);
        vv_T0t   = preload_array(grid.T0t_loc);
        vv_T0p   = preload_array(grid.T0p_loc);
    }
    if(!use_gpu)
        vv_change = preload_array(grid.is_changed);
    else
        vv_change_bl = preload_array_bl(grid.is_changed);

    // flag for preloading
    simd_allocated = true;
#endif // USE_SIMD

}

#if defined USE_SIMD || defined USE_CUDA

template <typename T>
std::vector<std::vector<CUSTOMREAL*>> Iterator::preload_array(T* a){

    std::vector<std::vector<CUSTOMREAL*>> vvv;

    for (int iswap=0; iswap<8; iswap++){
        int n_levels = ijk_for_this_subproc.size();
        std::vector<CUSTOMREAL*> vv;

        for (int i_level=0; i_level<n_levels; i_level++){

            int nnodes = ijk_for_this_subproc.at(i_level).size();
            int nnodes_tmp;
            if(!use_gpu)
                nnodes_tmp = nnodes + ((nnodes % NSIMD == 0) ? 0 : (NSIMD - nnodes % NSIMD)); // length needs to be multiple of NSIMD
            else
                nnodes_tmp = nnodes;

            CUSTOMREAL* v = (CUSTOMREAL*) aligned_alloc(ALIGN, nnodes_tmp*sizeof(CUSTOMREAL));

            // asign values
            for (int i_node=0; i_node<nnodes; i_node++){
                v[i_node] = (CUSTOMREAL) a[vv_i__j__k__.at(iswap).at(i_level)[i_node]];
            }
            // assign dummy
            for (int i_node=nnodes; i_node<nnodes_tmp; i_node++){
                v[i_node] = 0.0;
            }
            vv.push_back(v);
        }
        vvv.push_back(vv);
    }

    return vvv;
}

std::vector<std::vector<bool*>> Iterator::preload_array_bl(bool* a){

    std::vector<std::vector<bool*>> vvv;

    for (int iswap=0; iswap<8; iswap++){
        int n_levels = ijk_for_this_subproc.size();
        std::vector<bool*> vv;

        for (int i_level=0; i_level<n_levels; i_level++){

            int nnodes = ijk_for_this_subproc.at(i_level).size();
            int nnodes_tmp;
            if(!use_gpu)
                nnodes_tmp = nnodes + ((nnodes % NSIMD == 0) ? 0 : (NSIMD - nnodes % NSIMD)); // length needs to be multiple of NSIMD
            else
                nnodes_tmp = nnodes;

            bool* v = (bool*) aligned_alloc(ALIGN, nnodes_tmp*sizeof(bool));

            // asign values
            for (int i_node=0; i_node<nnodes; i_node++){
                v[i_node] = (bool) a[vv_i__j__k__.at(iswap).at(i_level)[i_node]];
            }
            // assign dummy
            for (int i_node=nnodes; i_node<nnodes_tmp; i_node++){
                v[i_node] = 0.0;
            }
            vv.push_back(v);
        }
        vvv.push_back(vv);
    }

    return vvv;
}


template <typename T>
void Iterator::preload_indices(std::vector<std::vector<T*>> &vvvi, \
                               std::vector<std::vector<T*>> &vvvj, \
                               std::vector<std::vector<T*>> &vvvk, \
                               int shift_i, int shift_j, int shift_k) {

    int iip, jjt, kkr;
    for (int iswp=0; iswp < 8; iswp++){
        set_sweep_direction(iswp);

        std::vector<T*> vvi, vvj, vvk;

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc[i_level].size();
            int n_nodes_tmp;
            if(!use_gpu)
                n_nodes_tmp = n_nodes + ((n_nodes % NSIMD == 0) ? 0 : (NSIMD - n_nodes % NSIMD)); // length needs to be multiple of NSIMD
            else
                n_nodes_tmp = n_nodes;

            T*  vi = (T*) aligned_alloc(ALIGN, n_nodes_tmp*sizeof(T));
            T*  vj = (T*) aligned_alloc(ALIGN, n_nodes_tmp*sizeof(T));
            T*  vk = (T*) aligned_alloc(ALIGN, n_nodes_tmp*sizeof(T));

            for (int i_node = 0; i_node < n_nodes; i_node++) {
                int tmp_ijk = ijk_for_this_subproc[i_level][i_node];
                V2I(tmp_ijk, iip, jjt, kkr);
                if (r_dirc < 0) kkr = loc_K-kkr; //kk-1;
                else            kkr = kkr-1;  //nr-kk;
                if (t_dirc < 0) jjt = loc_J-jjt; //jj-1;
                else            jjt = jjt-1;  //nt-jj;
                if (p_dirc < 0) iip = loc_I-iip; //ii-1;
                else            iip = iip-1;  //np-ii;

                kkr += shift_k;
                jjt += shift_j;
                iip += shift_i;

                if (kkr >= loc_K) kkr = 0;
                if (jjt >= loc_J) jjt = 0;
                if (iip >= loc_I) iip = 0;

                if (kkr < 0) kkr = 0;
                if (jjt < 0) jjt = 0;
                if (iip < 0) iip = 0;

                vi[i_node] = (T)iip;
                vj[i_node] = (T)jjt;
                vk[i_node] = (T)kkr;

            }
            // assign dummy
            for (int i=n_nodes; i<n_nodes_tmp; i++){
                vi[i] = (T)0;
                vj[i] = (T)0;
                vk[i] = (T)0;
            }

            vvi.push_back(vi);
            vvj.push_back(vj);
            vvk.push_back(vk);
        }
        vvvi.push_back(vvi);
        vvvj.push_back(vvj);
        vvvk.push_back(vvk);
    }
}


template <typename T>
void Iterator::preload_indices_1d(std::vector<std::vector<T*>> &vvv, \
                               int shift_i, int shift_j, int shift_k) {

    int iip, jjt, kkr;
    for (int iswp=0; iswp < 8; iswp++){
        set_sweep_direction(iswp);

        std::vector<T*> vv;

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc[i_level].size();
            int n_nodes_tmp;
            if(!use_gpu)
                n_nodes_tmp = n_nodes + ((n_nodes % NSIMD == 0) ? 0 : (NSIMD - n_nodes % NSIMD)); // length needs to be multiple of NSIMD
            else
                n_nodes_tmp = n_nodes;

            T* v = (T*) aligned_alloc(ALIGN, n_nodes_tmp*sizeof(T));

            for (int i_node = 0; i_node < n_nodes; i_node++) {
                int tmp_ijk = ijk_for_this_subproc[i_level][i_node];
                V2I(tmp_ijk, iip, jjt, kkr);
                if (r_dirc < 0) kkr = loc_K-kkr; //kk-1;
                else            kkr = kkr-1;  //nr-kk;
                if (t_dirc < 0) jjt = loc_J-jjt; //jj-1;
                else            jjt = jjt-1;  //nt-jj;
                if (p_dirc < 0) iip = loc_I-iip; //ii-1;
                else            iip = iip-1;  //np-ii;

                kkr += shift_k;
                jjt += shift_j;
                iip += shift_i;

                if (kkr >= loc_K) kkr = 0;
                if (jjt >= loc_J) jjt = 0;
                if (iip >= loc_I) iip = 0;

                if (kkr < 0) kkr = 0;
                if (jjt < 0) jjt = 0;
                if (iip < 0) iip = 0;

                v[i_node] = (T)I2V(iip, jjt, kkr);

            }
            // assign dummy
            for (int i=n_nodes; i<n_nodes_tmp; i++){
                v[i] = (T)0;
            }

            vv.push_back(v);
        }

        vvv.push_back(vv);
    }
}


#endif // USE_SIMD || USE_CUDA

void Iterator::run_iteration_forward(InputParams& IP, Grid& grid, IO_utils& io, bool& first_init) {

    if(if_verbose) stdout_by_main("--- start iteration forward. ---");

    // start timer
    std::string iter_str = "iteration_forward";
    Timer timer_iter(iter_str);

    // in cpu mode
    iter_count = 0; cur_diff_L1 = HUGE_VAL; cur_diff_Linf = HUGE_VAL;

    // calculate the differcence from the true solution
    if (if_test && subdom_main) {
        if (!is_teleseismic)
            grid.calc_L1_and_Linf_error(ini_err_L1, ini_err_Linf);
        else
            grid.calc_L1_and_Linf_diff_tele(cur_diff_L1, cur_diff_Linf);

        if (myrank==0)
            std::cout << "initial err values L1, inf: " << ini_err_L1 << ", " << ini_err_Linf << std::endl;
    }

    if (subdom_main) {
        grid.calc_L1_and_Linf_diff(cur_diff_L1, cur_diff_Linf);
        if (myrank==0 && if_verbose)
            std::cout << "initial diff values L1, inf: " << cur_diff_L1 << ", " << cur_diff_Linf << std::endl;
    }

    // start iteration
    while (true) {

        // store tau for comparison
        if (subdom_main){
            if(!is_teleseismic)
                grid.tau2tau_old();
            else
                grid.T2tau_old();
        }

        // do sweeping for all direction
        for (int iswp = nswp-1; iswp > -1; iswp--) {
            do_sweep(iswp, grid, IP);

#ifdef FREQ_SYNC_GHOST
            // synchronize ghost cells everytime after sweeping of each direction
            if (subdom_main){
                if (!is_teleseismic)
                    grid.send_recev_boundary_data(grid.tau_loc);
                else
                    grid.send_recev_boundary_data(grid.T_loc);

            }
#endif
        }


#ifndef FREQ_SYNC_GHOST
        // synchronize ghost cells everytime after sweeping of all directions
        // as the same method with Detrixhe2016
        if (subdom_main){
            if (!is_teleseismic)
                grid.send_recev_boundary_data(grid.tau_loc);
            else
                grid.send_recev_boundary_data(grid.T_loc);
        }
#endif

        // calculate the objective function
        // if converged, break the loop
        if (subdom_main) {
            if (!is_teleseismic)
                grid.calc_L1_and_Linf_diff(cur_diff_L1, cur_diff_Linf);
            else
                grid.calc_L1_and_Linf_diff_tele(cur_diff_L1, cur_diff_Linf);

            if(if_test) {
                grid.calc_L1_and_Linf_error(cur_err_L1, cur_err_Linf);
                //std::cout << "Theoretical difference: " << cur_err_L1 << ' ' << cur_err_Linf << std::endl;
            }
        }

        // broadcast the diff values
        broadcast_cr_single_sub(cur_diff_L1, 0);
        broadcast_cr_single_sub(cur_diff_Linf, 0); // dead lock
        if (if_test) {
            broadcast_cr_single_sub(cur_err_L1, 0);
            broadcast_cr_single_sub(cur_err_Linf, 0);
        }

        //if (iter_count==0)
        // MNMN: please don't leave this line active when pushing your commit.
        // MNMN: because std::cout/endl is very slow when called in the loop.
        // std::cout << "id_sim, sub_rank, cur_diff_L1, cur_diff_Linf: " << id_sim << ", " << sub_rank << ", " << cur_diff_L1 << ", " << cur_diff_Linf << std::endl;

        // debug store temporal T fields
        //io.write_tmp_tau_h5(grid, iter_count);


        iter_count++;

        //if (cur_diff_L1 < IP.get_conv_tol() && cur_diff_Linf < IP.get_conv_tol()) { // MNMN: let us use only L1 because Linf stop decreasing when using numbers of subdomains.
        if (cur_diff_L1 < IP.get_conv_tol()) {
            //stdout_by_main("--- iteration converged. ---");
            goto iter_end;
        } else if (IP.get_max_iter() <= iter_count) {
            stdout_by_main("--- iteration reached to the maximum number of iterations. ---");
            goto iter_end;
        } else {
            if(myrank==0 && if_verbose)
                std::cout << "iteration " << iter_count << ": " << cur_diff_L1 << ", " << cur_diff_Linf << ", " << timer_iter.get_t_delta() << "\n";
        }
    }

iter_end:
    if (myrank==0){
        if (if_verbose){
            std::cout << "Converged at iteration " << iter_count << ": " << cur_diff_L1 << ", " << cur_diff_Linf << std::endl;
        }
        if (if_test)
            std::cout << "errors at iteration " << iter_count << ": " << cur_err_L1 << ", " << cur_err_Linf << std::endl;
        std::cout << "id_sim: " << id_sim << ", converged at iteration " << iter_count << std::endl;
    }

    // calculate T
    // teleseismic case will update T_loc, so T = T0*tau is not necessary.
    if (subdom_main && !is_teleseismic) grid.calc_T_plus_tau();

    // corner nodes are not used in the sweeping but used in the interpolation and visualization
    if (subdom_main) grid.send_recev_boundary_data_kosumi(grid.T_loc);

    // check the time for iteration
    if (inter_sub_rank==0 && subdom_main) {
        timer_iter.stop_timer();

        // output time in file
        std::ofstream ofs;
        ofs.open("time.txt", std::ios::app);
        ofs << "Converged at iteration " << iter_count << ", L1 " << cur_diff_L1 << ", Linf " << cur_diff_Linf << ", total time[s] " << timer_iter.get_t() << std::endl;

    }

}


void Iterator::run_iteration_adjoint(InputParams& IP, Grid& grid, IO_utils& io, int adj_type) {

    if(if_verbose) stdout_by_main("--- start iteration adjoint. ---");
    iter_count = 0;
    CUSTOMREAL cur_diff_L1_dummy = HUGE_VAL;
               cur_diff_Linf = HUGE_VAL;

    // initialize delta and Tadj_loc (here we use the array tau_old instead of delta for memory saving)
    if (subdom_main){
        if (adj_type == 0)  init_delta_and_Tadj(grid, IP);          // run iteration for adjoint field
        if (adj_type == 1)  init_delta_and_Tadj_density(grid, IP);  // run iteration for the density of adjoint field (adjoint source -> abs(adjoint source))
    }


    if(if_verbose) std::cout << "checker point 1, myrank: " << myrank << ", id_sim: " << id_sim << ", id_subdomain: " << id_subdomain
               << ", subdom_main:" << subdom_main << ", world_rank: " << world_rank << std::endl;
    // fix the boundary of the adjoint field
    if (subdom_main){
        fix_boundary_Tadj(grid);
    }

    // start timer
    std::string iter_str = "iteration_adjoint";
    Timer timer_iter(iter_str);
    if(if_verbose) stdout_by_main("--- adjoint fast sweeping ... ---");

    // start iteration
    while (true) {
        // do sweeping for all direction
        for (int iswp = 0; iswp < nswp; iswp++) {
            do_sweep_adj(iswp, grid, IP);

#ifdef FREQ_SYNC_GHOST
            // copy the values of communication nodes and ghost nodes
            if (subdom_main)
                grid.send_recev_boundary_data(grid.tau_loc);
#endif
        }

#ifndef FREQ_SYNC_GHOST
        // copy the values of communication nodes and ghost nodes
        if (subdom_main)
            grid.send_recev_boundary_data(grid.tau_loc);
#endif

        // calculate the objective function
        // if converged, break the loop
        if (subdom_main) {
            if(adj_type == 0)   grid.calc_L1_and_Linf_diff_adj(cur_diff_L1_dummy, cur_diff_Linf);
            if(adj_type == 1)   grid.calc_L1_and_Linf_diff_adj_density(cur_diff_L1_dummy, cur_diff_Linf);
        }

        // broadcast the diff values
        //broadcast_cr_single_sub(cur_diff_L1, 0);
        broadcast_cr_single_sub(cur_diff_Linf, 0);

        // debug store temporal T fields
        //io.write_tmp_tau_h5(grid, iter_count);

        // store tau -> Tadj
        if (subdom_main){
            if(adj_type == 0)   grid.update_Tadj();
            if(adj_type == 1)   grid.update_Tadj_density();
        }

        if (cur_diff_Linf < IP.get_conv_tol()) {
            //stdout_by_main("--- adjoint iteration converged. ---");
            goto iter_end;
        } else if (IP.get_max_iter() <= iter_count) {
            stdout_by_main("--- adjoint iteration reached the maximum number of iterations. ---");
            goto iter_end;
        } else {
            if(myrank==0 && if_verbose)
                std::cout << "iteration adj. " << iter_count << ": " << cur_diff_Linf << std::endl;
            iter_count++;
        }


    }

iter_end:
    if (myrank==0){
        if (if_verbose)
            std::cout << "Converged at adjoint iteration " << iter_count << ": "  << cur_diff_Linf << std::endl;
        if (if_test)
            std::cout << "errors at iteration " << iter_count << ": " << cur_err_Linf << std::endl;
    }

    // corner nodes are not used in the sweeping but used in the interpolation and visualization
    if (subdom_main) {
        if(adj_type == 0)   grid.send_recev_boundary_data_kosumi(grid.Tadj_loc);
        if(adj_type == 1)   grid.send_recev_boundary_data_kosumi(grid.Tadj_density_loc);
    }


    // check the time for iteration
    if (inter_sub_rank==0 && subdom_main) timer_iter.stop_timer();

}



void Iterator::init_delta_and_Tadj(Grid& grid, InputParams& IP) {
    if(if_verbose) std::cout << "initializing delta and Tadj" << std::endl;

    for (int k = 0; k < nr; k++) {
        for (int j = 0; j < nt; j++) {
            for (int i = 0; i < np; i++) {
                grid.tau_old_loc[I2V(i,j,k)] = _0_CR; // use tau_old_loc for delta
                grid.tau_loc[I2V(i,j,k)]     = _0_CR; // use tau_loc for Tadj_loc (later copy to Tadj_loc)
                grid.Tadj_loc[I2V(i,j,k)]    = 9999999.9;
            }
        }
    }

    // loop all receivers
    for (int irec = 0; irec < IP.n_rec_this_sim_group; irec++) {

        // get receiver information
        std::string rec_name = IP.get_rec_name(irec);
        auto rec = IP.get_rec_point_bcast(rec_name);

        // "iter->second" is the receiver, with the class SrcRecInfo
        if (rec.adjoint_source == 0){
            continue;
        }

        CUSTOMREAL delta_lon = grid.get_delta_lon();
        CUSTOMREAL delta_lat = grid.get_delta_lat();
        CUSTOMREAL delta_r   = grid.get_delta_r();


        // get positions
        CUSTOMREAL rec_lon = rec.lon*DEG2RAD;
        CUSTOMREAL rec_lat = rec.lat*DEG2RAD;
        CUSTOMREAL rec_r = depth2radius(rec.dep);

        // check if the receiver is in this subdomain
        if (grid.get_lon_min_loc() <= rec_lon && rec_lon <= grid.get_lon_max_loc()  && \
            grid.get_lat_min_loc() <= rec_lat && rec_lat <= grid.get_lat_max_loc()  && \
            grid.get_r_min_loc()   <= rec_r   && rec_r   <= grid.get_r_max_loc()   ) {

            // descretize receiver position (LOCAL ID)
            int i_rec_loc =  std::floor((rec_lon - grid.get_lon_min_loc()) / delta_lon);
            int j_rec_loc =  std::floor((rec_lat - grid.get_lat_min_loc()) / delta_lat);
            int k_rec_loc =  std::floor((rec_r   - grid.get_r_min_loc())   / delta_r);

            if(i_rec_loc +1 >= loc_I)
                i_rec_loc = loc_I - 2;
            if(j_rec_loc + 1 >= loc_J)
                j_rec_loc = loc_J - 2;
            if(k_rec_loc + 1 >= loc_K)
                k_rec_loc = loc_K - 2;

            // discretized receiver position
            CUSTOMREAL dis_rec_lon = grid.p_loc_1d[i_rec_loc];
            CUSTOMREAL dis_rec_lat = grid.t_loc_1d[j_rec_loc];
            CUSTOMREAL dis_rec_r   = grid.r_loc_1d[k_rec_loc];

            // relative position errors
            CUSTOMREAL e_lon = std::min(_1_CR,(rec_lon - dis_rec_lon)/delta_lon);
            CUSTOMREAL e_lat = std::min(_1_CR,(rec_lat - dis_rec_lat)/delta_lat);
            CUSTOMREAL e_r   = std::min(_1_CR,(rec_r   - dis_rec_r)  /delta_r);

            // set delta values
            grid.tau_old_loc[I2V(i_rec_loc,j_rec_loc,k_rec_loc)]       += rec.adjoint_source*(1.0-e_lon)*(1.0-e_lat)*(1.0-e_r)/(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc,j_rec_loc,k_rec_loc+1)]     += rec.adjoint_source*(1.0-e_lon)*(1.0-e_lat)*     e_r /(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc,j_rec_loc+1,k_rec_loc)]     += rec.adjoint_source*(1.0-e_lon)*     e_lat* (1.0-e_r)/(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc,j_rec_loc+1,k_rec_loc+1)]   += rec.adjoint_source*(1.0-e_lon)*     e_lat*      e_r /(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc+1,j_rec_loc,k_rec_loc)]     += rec.adjoint_source*     e_lon *(1.0-e_lat)*(1.0-e_r)/(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc+1,j_rec_loc,k_rec_loc+1)]   += rec.adjoint_source*     e_lon *(1.0-e_lat)*     e_r /(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc+1,j_rec_loc+1,k_rec_loc)]   += rec.adjoint_source*     e_lon *     e_lat *(1.0-e_r)/(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc+1,j_rec_loc+1,k_rec_loc+1)] += rec.adjoint_source*     e_lon *     e_lat *     e_r /(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
        }
    } // end of loop all receivers

    // communicate tau_old_loc to all processors
    grid.send_recev_boundary_data(grid.tau_old_loc);

}


void Iterator::init_delta_and_Tadj_density(Grid& grid, InputParams& IP) {
    if(if_verbose) std::cout << "initializing delta and Tadj" << std::endl;

    for (int k = 0; k < nr; k++) {
        for (int j = 0; j < nt; j++) {
            for (int i = 0; i < np; i++) {
                grid.tau_old_loc[I2V(i,j,k)] = _0_CR; // use tau_old_loc for delta
                grid.tau_loc[I2V(i,j,k)]     = _0_CR; // use tau_loc for Tadj_density_loc (later copy to Tadj_density_loc)
                grid.Tadj_density_loc[I2V(i,j,k)]    = 9999999.9;
            }
        }
    }

    // loop all receivers
    for (int irec = 0; irec < IP.n_rec_this_sim_group; irec++) {

        // get receiver information
        std::string rec_name = IP.get_rec_name(irec);
        auto rec = IP.get_rec_point_bcast(rec_name);

        // "iter->second" is the receiver, with the class SrcRecInfo
        if (rec.adjoint_source_density == 0){
            continue;
        }

        CUSTOMREAL delta_lon = grid.get_delta_lon();
        CUSTOMREAL delta_lat = grid.get_delta_lat();
        CUSTOMREAL delta_r   = grid.get_delta_r();


        // get positions
        CUSTOMREAL rec_lon = rec.lon*DEG2RAD;
        CUSTOMREAL rec_lat = rec.lat*DEG2RAD;
        CUSTOMREAL rec_r = depth2radius(rec.dep);

        // check if the receiver is in this subdomain
        if (grid.get_lon_min_loc() <= rec_lon && rec_lon <= grid.get_lon_max_loc()  && \
            grid.get_lat_min_loc() <= rec_lat && rec_lat <= grid.get_lat_max_loc()  && \
            grid.get_r_min_loc()   <= rec_r   && rec_r   <= grid.get_r_max_loc()   ) {

            // descretize receiver position (LOCAL ID)
            int i_rec_loc =  std::floor((rec_lon - grid.get_lon_min_loc()) / delta_lon);
            int j_rec_loc =  std::floor((rec_lat - grid.get_lat_min_loc()) / delta_lat);
            int k_rec_loc =  std::floor((rec_r   - grid.get_r_min_loc())   / delta_r);

            if(i_rec_loc +1 >= loc_I)
                i_rec_loc = loc_I - 2;
            if(j_rec_loc + 1 >= loc_J)
                j_rec_loc = loc_J - 2;
            if(k_rec_loc + 1 >= loc_K)
                k_rec_loc = loc_K - 2;

            // discretized receiver position
            CUSTOMREAL dis_rec_lon = grid.p_loc_1d[i_rec_loc];
            CUSTOMREAL dis_rec_lat = grid.t_loc_1d[j_rec_loc];
            CUSTOMREAL dis_rec_r   = grid.r_loc_1d[k_rec_loc];

            // relative position errors
            CUSTOMREAL e_lon = std::min(_1_CR,(rec_lon - dis_rec_lon)/delta_lon);
            CUSTOMREAL e_lat = std::min(_1_CR,(rec_lat - dis_rec_lat)/delta_lat);
            CUSTOMREAL e_r   = std::min(_1_CR,(rec_r   - dis_rec_r)  /delta_r);

            // set delta values
            grid.tau_old_loc[I2V(i_rec_loc,j_rec_loc,k_rec_loc)]       += rec.adjoint_source_density*(1.0-e_lon)*(1.0-e_lat)*(1.0-e_r)/(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc,j_rec_loc,k_rec_loc+1)]     += rec.adjoint_source_density*(1.0-e_lon)*(1.0-e_lat)*     e_r /(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc,j_rec_loc+1,k_rec_loc)]     += rec.adjoint_source_density*(1.0-e_lon)*     e_lat* (1.0-e_r)/(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc,j_rec_loc+1,k_rec_loc+1)]   += rec.adjoint_source_density*(1.0-e_lon)*     e_lat*      e_r /(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc+1,j_rec_loc,k_rec_loc)]     += rec.adjoint_source_density*     e_lon *(1.0-e_lat)*(1.0-e_r)/(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc+1,j_rec_loc,k_rec_loc+1)]   += rec.adjoint_source_density*     e_lon *(1.0-e_lat)*     e_r /(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc+1,j_rec_loc+1,k_rec_loc)]   += rec.adjoint_source_density*     e_lon *     e_lat *(1.0-e_r)/(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
            grid.tau_old_loc[I2V(i_rec_loc+1,j_rec_loc+1,k_rec_loc+1)] += rec.adjoint_source_density*     e_lon *     e_lat *     e_r /(delta_lon*delta_lat*delta_r*my_square(rec_r)*std::cos(rec_lat));
        }
    } // end of loop all receivers

    // communicate tau_old_loc to all processors
    grid.send_recev_boundary_data(grid.tau_old_loc);

}

void Iterator::fix_boundary_Tadj(Grid& grid) {

    if (!is_teleseismic){
        // r, theta boundary
        if (grid.i_first())
            for (int ir = 0; ir < nr; ir++)
                for (int it = 0; it < nt; it++)
                    grid.tau_loc[I2V(0,it,ir)]    = _0_CR;
        if (grid.i_last())
            for (int ir = 0; ir < nr; ir++)
                for (int it = 0; it < nt; it++)
                    grid.tau_loc[I2V(np-1,it,ir)] = _0_CR;
        // r, phi boundary
        if (grid.j_first())
            for (int ir = 0; ir < nr; ir++)
                for (int ip = 0; ip < np; ip++)
                    grid.tau_loc[I2V(ip,0,ir)]    = _0_CR;
        if (grid.j_last())
            for (int ir = 0; ir < nr; ir++)
                for (int ip = 0; ip < np; ip++)
                    grid.tau_loc[I2V(ip,nt-1,ir)] = _0_CR;
        // theta, phi boundary
        if (grid.k_first())
            for (int it = 0; it < nt; it++)
                for (int ip = 0; ip < np; ip++)
                    grid.tau_loc[I2V(ip,it,0)]    = _0_CR;
        if (grid.k_last())
            for (int it = 0; it < nt; it++)
                for (int ip = 0; ip < np; ip++)
                    grid.tau_loc[I2V(ip,it,nr-1)] = _0_CR;
    } else {
        for (int ir = 0; ir < nr; ir++)
            for (int it = 0; it < nt; it++)
                for (int ip = 0; ip < np; ip++)
                    calculate_boundary_nodes_adj(grid,ip,it,ir);
    }

}


void Iterator::calculate_stencil_1st_order_upwind(Grid&grid, int&iip, int&jjt, int&kkr){

    bool check_out = false;
    int iip_out = 48;
    int jjt_out = 122;
    int kkr_out = 49;
    // preparations

    count_cand = 0;
    // forward and backward partial differential discretization
    //  T_p = (T0*tau)_p = T0p*tau + T0v*tau_p = ap*tau(iip, jjt, kkr)+bp;
    //  T_t = (T0*tau)_t = T0t*tau + T0v*tau_t = at*tau(iip, jjt, kkr)+bt;
    //  T_r = (T0*tau)_r = T0r*tau + T0v*tau_r = ar*tau(iip, jjt, kkr)+br;
    if (iip > 0){
        ap1 =  grid.T0p_loc[I2V(iip, jjt, kkr)] + grid.T0v_loc[I2V(iip, jjt, kkr)]/dp;
        bp1 = -grid.T0v_loc[I2V(iip, jjt, kkr)]/dp*grid.tau_loc[I2V(iip-1, jjt, kkr)];
    }
    if (iip < np-1){
        ap2 =  grid.T0p_loc[I2V(iip, jjt, kkr)] - grid.T0v_loc[I2V(iip, jjt, kkr)]/dp;
        bp2 =  grid.T0v_loc[I2V(iip, jjt, kkr)]/dp*grid.tau_loc[I2V(iip+1, jjt, kkr)];
    }

    if (jjt > 0){
        at1 =  grid.T0t_loc[I2V(iip, jjt, kkr)] + grid.T0v_loc[I2V(iip, jjt, kkr)]/dt;
        bt1 = -grid.T0v_loc[I2V(iip, jjt, kkr)]/dt*grid.tau_loc[I2V(iip, jjt-1, kkr)];
    }
    if (jjt < nt-1){
        at2 =  grid.T0t_loc[I2V(iip, jjt, kkr)] - grid.T0v_loc[I2V(iip, jjt, kkr)]/dt;
        bt2 =  grid.T0v_loc[I2V(iip, jjt, kkr)]/dt*grid.tau_loc[I2V(iip, jjt+1, kkr)];
    }

    if (kkr > 0){
        ar1 =  grid.T0r_loc[I2V(iip, jjt, kkr)] + grid.T0v_loc[I2V(iip, jjt, kkr)]/dr;
        br1 = -grid.T0v_loc[I2V(iip, jjt, kkr)]/dr*grid.tau_loc[I2V(iip, jjt, kkr-1)];
    }
    if (kkr < nr-1){
        ar2 =  grid.T0r_loc[I2V(iip, jjt, kkr)] - grid.T0v_loc[I2V(iip, jjt, kkr)]/dr;
        br2 =  grid.T0v_loc[I2V(iip, jjt, kkr)]/dr*grid.tau_loc[I2V(iip, jjt, kkr+1)];
    }
    bc_f2 = grid.fac_b_loc[I2V(iip,jjt,kkr)]*grid.fac_c_loc[I2V(iip,jjt,kkr)] - std::pow(grid.fac_f_loc[I2V(iip,jjt,kkr)],_2_CR);

    // start to find candidate solutions

    // first catalog: characteristic travels through tetrahedron in 3D volume (8 cases)
    for (int i_case = 0; i_case < 8; i_case++){

        // determine discretization of T_p,T_t,T_r
        switch (i_case) {
            case 0:    // characteristic travels from -p, -t, -r
                if (iip == 0 || jjt == 0 || kkr == 0)
                    continue;
                ap = ap1; bp = bp1;
                at = at1; bt = bt1;
                ar = ar1; br = br1;
                break;
            case 1:     // characteristic travels from -p, -t, +r
                if (iip == 0 || jjt == 0 || kkr == nr-1)
                    continue;
                ap = ap1; bp = bp1;
                at = at1; bt = bt1;
                ar = ar2; br = br2;
                break;
            case 2:    // characteristic travels from -p, +t, -r
                if (iip == 0 || jjt == nt-1 || kkr == 0)
                    continue;
                ap = ap1; bp = bp1;
                at = at2; bt = bt2;
                ar = ar1; br = br1;
                break;
            case 3:     // characteristic travels from -p, +t, +r
                if (iip == 0 || jjt == nt-1 || kkr == nr-1)
                    continue;
                ap = ap1; bp = bp1;
                at = at2; bt = bt2;
                ar = ar2; br = br2;
                break;
            case 4:    // characteristic travels from +p, -t, -r
                if (iip == np-1 || jjt == 0 || kkr == 0)
                    continue;
                ap = ap2; bp = bp2;
                at = at1; bt = bt1;
                ar = ar1; br = br1;
                break;
            case 5:     // characteristic travels from +p, -t, +r
                if (iip == np-1 || jjt == 0 || kkr == nr-1)
                    continue;
                ap = ap2; bp = bp2;
                at = at1; bt = bt1;
                ar = ar2; br = br2;
                break;
            case 6:    // characteristic travels from +p, +t, -r
                if (iip == np-1 || jjt == nt-1 || kkr == 0)
                    continue;
                ap = ap2; bp = bp2;
                at = at2; bt = bt2;
                ar = ar1; br = br1;
                break;
            case 7:     // characteristic travels from +p, +t, +r
                if (iip == np-1 || jjt == nt-1 || kkr == nr-1)
                    continue;
                ap = ap2; bp = bp2;
                at = at2; bt = bt2;
                ar = ar2; br = br2;
                break;
        }

        // plug T_p, T_t, T_r into eikonal equation, solving the quadratic equation with respect to tau(iip,jjt,kkr)
        // that is a*(ar*tau+br)^2 + b*(at*tau+bt)^2 + c*(ap*tau+bp)^2 - 2*f*(at*tau+bt)*(ap*tau+bp) = s^2
        eqn_a = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(ar, _2_CR) + grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(at, _2_CR)
              + grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(ap, _2_CR) - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * at * ap;
        eqn_b = _2_CR*grid.fac_a_loc[I2V(iip,jjt,kkr)] * ar * br + _2_CR*grid.fac_b_loc[I2V(iip,jjt,kkr)] * at * bt
              + _2_CR*grid.fac_c_loc[I2V(iip,jjt,kkr)] * ap * bp - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * (at*bp + bt*ap);
        eqn_c = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(br, _2_CR) + grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(bt, _2_CR)
              + grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(bp, _2_CR) - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * bt * bp
              - std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR);
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

                // check the causality condition: the characteristic passing through (iip,jjt,kkr) is in between used three sides
                // characteristic direction is (dr/dt, dtheta/dt, tphi/dt) = (H_p1,H_p2,H_p3), p1 = T_r, p2 = T_t, p3 = T_p
                T_r = ar*tmp_tau + br;
                T_t = at*tmp_tau + bt;
                T_p = ap*tmp_tau + bp;

                charact_r = grid.fac_a_loc[I2V(iip,jjt,kkr)]*T_r;
                charact_t = grid.fac_b_loc[I2V(iip,jjt,kkr)]*T_t - grid.fac_f_loc[I2V(iip,jjt,kkr)]*T_p;
                charact_p = grid.fac_c_loc[I2V(iip,jjt,kkr)]*T_p - grid.fac_f_loc[I2V(iip,jjt,kkr)]*T_t;

                is_causality = false;
                switch (i_case){
                    case 0:  //characteristic travels from -p, -t, -r
                        if (charact_p >= 0 && charact_t >= 0 && charact_r >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip-1,jjt,kkr)] * grid.tau_loc[I2V(iip-1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt-1,kkr)] * grid.tau_loc[I2V(iip,jjt-1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr-1)] * grid.tau_loc[I2V(iip,jjt,kkr-1)]){
                                // the additional constrains is only valid for weak anisotropy. it ensures correctness near the source.
                            is_causality = true;
                        }
                        break;
                    case 1:  //characteristic travels from -p, -t, +r
                        if (charact_p >= 0 && charact_t >= 0 && charact_r <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip-1,jjt,kkr)] * grid.tau_loc[I2V(iip-1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt-1,kkr)] * grid.tau_loc[I2V(iip,jjt-1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr+1)] * grid.tau_loc[I2V(iip,jjt,kkr+1)]){
                            is_causality = true;
                        }
                        break;
                    case 2:  //characteristic travels from -p, +t, -r
                        if (charact_p >= 0 && charact_t <= 0 && charact_r >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip-1,jjt,kkr)] * grid.tau_loc[I2V(iip-1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt+1,kkr)] * grid.tau_loc[I2V(iip,jjt+1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr-1)] * grid.tau_loc[I2V(iip,jjt,kkr-1)]){
                            is_causality = true;
                        }
                        break;
                    case 3:  //characteristic travels from -p, +t, +r
                        if (charact_p >= 0 && charact_t <= 0 && charact_r <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip-1,jjt,kkr)] * grid.tau_loc[I2V(iip-1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt+1,kkr)] * grid.tau_loc[I2V(iip,jjt+1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr+1)] * grid.tau_loc[I2V(iip,jjt,kkr+1)]){
                            is_causality = true;
                        }
                        break;
                    case 4:  //characteristic travels from +p, -t, -r
                        if (charact_p <= 0 && charact_t >= 0 && charact_r >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip+1,jjt,kkr)] * grid.tau_loc[I2V(iip+1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt-1,kkr)] * grid.tau_loc[I2V(iip,jjt-1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr-1)] * grid.tau_loc[I2V(iip,jjt,kkr-1)]){
                            is_causality = true;
                        }
                        break;
                    case 5:  //characteristic travels from +p, -t, +r
                        if (charact_p <= 0 && charact_t >= 0 && charact_r <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip+1,jjt,kkr)] * grid.tau_loc[I2V(iip+1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt-1,kkr)] * grid.tau_loc[I2V(iip,jjt-1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr+1)] * grid.tau_loc[I2V(iip,jjt,kkr+1)]){
                            is_causality = true;
                        }
                        break;
                    case 6:  //characteristic travels from +p, +t, -r
                        if (charact_p <= 0 && charact_t <= 0 && charact_r >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip+1,jjt,kkr)] * grid.tau_loc[I2V(iip+1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt+1,kkr)] * grid.tau_loc[I2V(iip,jjt+1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr-1)] * grid.tau_loc[I2V(iip,jjt,kkr-1)]){
                            is_causality = true;
                        }
                        break;
                    case 7:  //characteristic travels from +p, +t, +r
                        if (charact_p <= 0 && charact_t <= 0 && charact_r <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip+1,jjt,kkr)] * grid.tau_loc[I2V(iip+1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt+1,kkr)] * grid.tau_loc[I2V(iip,jjt+1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr+1)] * grid.tau_loc[I2V(iip,jjt,kkr+1)]){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_tau;
                    count_cand += 1;
                }

                if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
                    std::cout   << "body case, i_case, i_solution, is_causality, tau, T0, tau*t0: "
                                << i_case << ", " << i_solution << ", " << is_causality << ", " << tmp_tau << ", "
                                << grid.T0v_loc[I2V(iip,jjt,kkr)] << ", " << grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau
                                << ", " << std::endl;
                    switch (i_case) {
                        case 0: //characteristic travels from -p, -t, -r
                            std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", "
                                        << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", "
                                        << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 1: //characteristic travels from -p, -t, +r
                            std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", "
                                        << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", "
                                        << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 2: //characteristic travels from -p, +t, -r
                            std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", "
                                        << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", "
                                        << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 3: //characteristic travels from -p, +t, +r
                            std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", "
                                        << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", "
                                        << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 4: //characteristic travels from +p, -t, -r
                            std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", "
                                        << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", "
                                        << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 5: //characteristic travels from +p, -t, +r
                            std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", "
                                        << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", "
                                        << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 6: //characteristic travels from +p, +t, -r
                            std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", "
                                        << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", "
                                        << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 7: //characteristic travels from +p, +t, +r
                            std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", "
                                        << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", "
                                        << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                    }
                    
                }


            }

        }
    }

    // second catalog: characteristic travels through triangles in 2D volume (12 cases)
    // case: 1-4
    // characteristic on r-t plane, force H_p3 = H_(T_p) = 0, that is, c*T_p-f*T_t = 0
    // plug the constraint into eikonal equation, we have the equation:  a*T_r^2 + (bc-f^2)/c*T_t^2 = s^2
    for (int i_case = 0; i_case < 4; i_case++){
        switch (i_case){
            case 0:     //characteristic travels from  -t, -r
                if (jjt ==  0 || kkr ==  0){
                    continue;
                }
                at = at1; bt = bt1;
                ar = ar1; br = br1;
                break;
            case 1:     //characteristic travels from  -t, +r
                if (jjt ==  0 || kkr ==  nr-1){
                    continue;
                }
                at = at1; bt = bt1;
                ar = ar2; br = br2;
                break;
            case 2:     //characteristic travels from  +t, -r
                if (jjt ==  nt-1 || kkr ==  0){
                    continue;
                }
                at = at2; bt = bt2;
                ar = ar1; br = br1;
                break;
            case 3:     //characteristic travels from  +t, +r
                if (jjt ==  nt-1 || kkr ==  nr-1){
                    continue;
                }
                at = at2; bt = bt2;
                ar = ar2; br = br2;
                break;
        }

        // plug T_t, T_r into eikonal equation, solve the quadratic equation:  a*(ar*tau+br)^2 + (bc-f^2)/c*(at*tau+bt)^2 = s^2
        eqn_a = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(ar, _2_CR) + bc_f2/grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(at, _2_CR);
        eqn_b = _2_CR*grid.fac_a_loc[I2V(iip,jjt,kkr)] * ar * br + _2_CR*bc_f2/grid.fac_c_loc[I2V(iip,jjt,kkr)] * at * bt;
        eqn_c = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(br, _2_CR) + bc_f2/grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(bt, _2_CR)
              - std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR);
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

                // check the causality condition:
                T_r = ar*tmp_tau + br;
                T_t = at*tmp_tau + bt;

                charact_r = grid.fac_a_loc[I2V(iip,jjt,kkr)]*T_r;
                charact_t = bc_f2/grid.fac_c_loc[I2V(iip,jjt,kkr)]*T_t;

                is_causality = false;
                switch (i_case){
                    case 0:  //characteristic travels from -t, -r
                        if (charact_t >= 0 && charact_r >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt-1,kkr)] * grid.tau_loc[I2V(iip,jjt-1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr-1)] * grid.tau_loc[I2V(iip,jjt,kkr-1)]){
                            is_causality = true;
                        }
                        break;
                    case 1:  //characteristic travels from -t, +r
                        if (charact_t >= 0 && charact_r <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt-1,kkr)] * grid.tau_loc[I2V(iip,jjt-1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr+1)] * grid.tau_loc[I2V(iip,jjt,kkr+1)]){
                            is_causality = true;
                        }
                        break;
                    case 2:  //characteristic travels from +t, -r
                        if (charact_t <= 0 && charact_r >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt+1,kkr)] * grid.tau_loc[I2V(iip,jjt+1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr-1)] * grid.tau_loc[I2V(iip,jjt,kkr-1)]){
                            is_causality = true;
                        }
                        break;
                    case 3:  //characteristic travels from +t, +r
                        if (charact_t <= 0 && charact_r <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt+1,kkr)] * grid.tau_loc[I2V(iip,jjt+1,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr+1)] * grid.tau_loc[I2V(iip,jjt,kkr+1)]){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_tau;
                    count_cand += 1;
                }

                if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
                    std::cout   << "surface case, i_case, i_solution, is_causality, tau, T0, tau*t0: "
                                << i_case << ", " << i_solution << ", " << is_causality << ", " << tmp_tau << ", "
                                << grid.T0v_loc[I2V(iip,jjt,kkr)] << ", " << grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau
                                << ", " << std::endl;
                    switch (i_case) {
                        case 0: //characteristic travels from -t, -r
                            std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", "
                                        << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 1: //characteristic travels from -t, +r
                            std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", "
                                        << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 2: //characteristic travels from +t, -r
                            std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", "
                                        << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 3: //characteristic travels from +t, +r
                            std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", "
                                        << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", "
                                        << std::endl;
                            break;
                    }
                }

            }
        }

    }

    // case: 5-8
    // characteristic on r-p plane, force H_p2 = H_(T_t) = 0, that is, b*T_t-f*T_p = 0
    // plug the constraint into eikonal equation, we have the equation:  a*T_r^2 + (bc-f^2)/b*T_p^2 = s^2
    for (int i_case = 4; i_case < 8; i_case++){
         switch (i_case){
            case 4:     //characteristic travels from  -p, -r
                if (iip ==  0 || kkr ==  0){
                    continue;
                }
                ap = ap1; bp = bp1;
                ar = ar1; br = br1;
                break;
            case 5:     //characteristic travels from  -p, +r
                if (iip ==  0 || kkr ==  nr-1){
                    continue;
                }
                ap = ap1; bp = bp1;
                ar = ar2; br = br2;
                break;
            case 6:     //characteristic travels from  +p, -r
                if (iip ==  np-1 || kkr ==  0){
                    continue;
                }
                ap = ap2; bp = bp2;
                ar = ar1; br = br1;
                break;
            case 7:     //characteristic travels from  +p, +r
                if (iip ==  np-1 || kkr ==  nr-1){
                    continue;
                }
                ap = ap2; bp = bp2;
                ar = ar2; br = br2;
                break;
        }

        // plug T_p, T_r into eikonal equation, solve the quadratic equation:  a*(ar*tau+br)^2 + (bc-f^2)/b*(ap*tau+bp)^2 = s^2
        eqn_a = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(ar, _2_CR) + bc_f2/grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(ap, _2_CR);
        eqn_b = _2_CR*grid.fac_a_loc[I2V(iip,jjt,kkr)] * ar * br + _2_CR*bc_f2/grid.fac_b_loc[I2V(iip,jjt,kkr)] * ap * bp;
        eqn_c = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(br, _2_CR) + bc_f2/grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(bp, _2_CR)
              - std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR);
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

                // check the causality condition:
                T_r = ar*tmp_tau + br;
                T_p = ap*tmp_tau + bp;

                charact_r = grid.fac_a_loc[I2V(iip,jjt,kkr)]*T_r;
                charact_p = bc_f2/grid.fac_b_loc[I2V(iip,jjt,kkr)]*T_p;

                is_causality = false;
                switch (i_case){
                    case 4:  //characteristic travels from -p, -r
                        if (charact_p >= 0 && charact_r >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip-1,jjt,kkr)] * grid.tau_loc[I2V(iip-1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr-1)] * grid.tau_loc[I2V(iip,jjt,kkr-1)]){
                            is_causality = true;
                        }
                        break;
                    case 5:  //characteristic travels from -p, +r
                        if (charact_p >= 0 && charact_r <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip-1,jjt,kkr)] * grid.tau_loc[I2V(iip-1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr+1)] * grid.tau_loc[I2V(iip,jjt,kkr+1)]){
                            is_causality = true;
                        }
                        break;
                    case 6:  //characteristic travels from +p, -r
                        if (charact_p <= 0 && charact_r >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip+1,jjt,kkr)] * grid.tau_loc[I2V(iip+1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr-1)] * grid.tau_loc[I2V(iip,jjt,kkr-1)]){
                            is_causality = true;
                        }
                        break;
                    case 7:  //characteristic travels from +p, +r
                        if (charact_p <= 0 && charact_r <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip+1,jjt,kkr)] * grid.tau_loc[I2V(iip+1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt,kkr+1)] * grid.tau_loc[I2V(iip,jjt,kkr+1)]){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_tau;
                    count_cand += 1;
                }

                if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
                    std::cout   << "surface, case, i_case, i_solution, is_causality, tau, T0, tau*t0: "
                            << i_case << ", " << i_solution << ", " << is_causality << ", " << tmp_tau << ", "
                            << grid.T0v_loc[I2V(iip,jjt,kkr)] << ", " << grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau
                            << ", " << std::endl;
                    switch (i_case) {
                        case 4: // characteristic travels from -p, -r
                            std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", "
                                        << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 5: // characteristic travels from -p, +r
                            std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", "
                                        << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 6: // characteristic travels from +p, -r
                            std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", "
                                        << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 7: // characteristic travels from +p, +r
                            std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", "
                                        << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                    }
                }

            }
        }
    }

    // case: 9-12
    // characteristic on t-p plane, force H_p1 = H_(T_t) = 0, that is, T_r = 0
    // plug the constraint into eikonal equation, we have the equation:  b*T_t^2 + c*T_p^2 - 2f*T_t*T_p = s^2
    for (int i_case = 8; i_case < 12; i_case++){
        switch (i_case){
            case 8:     //characteristic travels from  -p, -t
                if (iip ==  0 || jjt ==  0){
                    continue;
                }
                ap = ap1; bp = bp1;
                at = at1; bt = bt1;
                break;
            case 9:     //characteristic travels from  -p, +t
                if (iip ==  0 || jjt ==  nt-1){
                    continue;
                }
                ap = ap1; bp = bp1;
                at = at2; bt = bt2;
                break;
            case 10:     //characteristic travels from  +p, -t
                if (iip ==  np-1 || jjt ==  0){
                    continue;
                }
                ap = ap2; bp = bp2;
                at = at1; bt = bt1;
                break;
            case 11:     //characteristic travels from  +p, +t
                if (iip ==  np-1 || jjt ==  nt-1){
                    continue;
                }
                ap = ap2; bp = bp2;
                at = at2; bt = bt2;
                break;
        }

        // plug T_p, T_t into eikonal equation, solve the quadratic equation:  b*(at*tau+bt)^2 + c*(ap*tau+bp)^2 - 2f*(at*tau+bt)*(ap*tau+bp) = s^2
        eqn_a = grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(at, _2_CR)
              + grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(ap, _2_CR) - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * at * ap;
        eqn_b = _2_CR*grid.fac_b_loc[I2V(iip,jjt,kkr)] * at * bt
              + _2_CR*grid.fac_c_loc[I2V(iip,jjt,kkr)] * ap * bp - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * (at*bp + bt*ap);
        eqn_c = grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(bt, _2_CR)
              + grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(bp, _2_CR) - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * bt * bp
              - std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR);
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

                // check the causality condition:
                T_t = at*tmp_tau + bt;
                T_p = ap*tmp_tau + bp;

                charact_t = grid.fac_b_loc[I2V(iip,jjt,kkr)]*T_t - grid.fac_f_loc[I2V(iip,jjt,kkr)]*T_p;
                charact_p = grid.fac_c_loc[I2V(iip,jjt,kkr)]*T_p - grid.fac_f_loc[I2V(iip,jjt,kkr)]*T_t;

                is_causality = false;
                switch (i_case){
                    case 8:  //characteristic travels from -p, -t
                        if (charact_p >= 0 && charact_t >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip-1,jjt,kkr)] * grid.tau_loc[I2V(iip-1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt-1,kkr)] * grid.tau_loc[I2V(iip,jjt-1,kkr)]){
                            
                            is_causality = true;
                        }
                        break;
                    case 9:  //characteristic travels from -p, +t
                        if (charact_p >= 0 && charact_t <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip-1,jjt,kkr)] * grid.tau_loc[I2V(iip-1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt+1,kkr)] * grid.tau_loc[I2V(iip,jjt+1,kkr)]){
                            is_causality = true;
                        }
                        break;
                    case 10:  //characteristic travels from +p, -t
                        if (charact_p <= 0 && charact_t >= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip+1,jjt,kkr)] * grid.tau_loc[I2V(iip+1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt-1,kkr)] * grid.tau_loc[I2V(iip,jjt-1,kkr)]){
                            is_causality = true;
                        }
                        break;
                    case 11:  //characteristic travels from +p, +t
                        if (charact_p <= 0 && charact_t <= 0 && tmp_tau > 0
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip+1,jjt,kkr)] * grid.tau_loc[I2V(iip+1,jjt,kkr)]
                            && tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.T0v_loc[I2V(iip,jjt+1,kkr)] * grid.tau_loc[I2V(iip,jjt+1,kkr)]){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_tau;
                    count_cand += 1;
                }

                if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
                    std::cout   << "surface case, i_case, i_solution, is_causality, tau, T0, tau*t0: "
                            << i_case << ", " << i_solution << ", " << is_causality << ", " << tmp_tau << ", "
                            << grid.T0v_loc[I2V(iip,jjt,kkr)] << ", " << grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau
                            << ", " << std::endl;
                    switch (i_case) {
                        case 8: // characteristic travels from -p, -t
                            std::cout   << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", "
                                        << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 9: // characteristic travels from -p, +t
                            std::cout   << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", "
                                        << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 10: // characteristic travels from +p, -t
                            std::cout   << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", "
                                        << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", "
                                        << std::endl;
                            break;
                        case 11: // characteristic travels from +p, +t
                            std::cout   << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", "
                                        << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", "
                                        << "T_t: " << T_t << ", T_p: " << T_p << ", "
                                        << std::endl;
                            // T_t = (T0v * tau)_t = 
                            std::cout << " check T_t. " << std::endl;
                            std::cout << "T_+ = T0v_loc[I2V(iip,jjt+1,kkr)] * tau_loc[I2V(iip,jjt+1,kkr)]:" << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << std::endl;
                            std::cout << "T   = T0v_loc[I2V(iip,jjt,kkr)]   * tmp_tau  :" << grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau << std::endl;
                            std::cout << "T_t_1 = (T_+ - T)/dt = " 
                                      <<  (grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] \
                                        - grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau) / grid.dt << std::endl;
                            std::cout << "T_t_2 = T0v * (tau_+ - tau)/dt + T0t * tau " << std::endl; 
                            std::cout << "T0v_+ : " << grid.T0v_loc[I2V(iip,jjt+1,kkr)] << std::endl;
                            std::cout << "T0v   : " << grid.T0v_loc[I2V(iip  ,jjt,kkr)] << std::endl;
                            std::cout << "tau_+ : " << grid.tau_loc[I2V(iip, jjt+1, kkr)] << std::endl;
                            std::cout << "tau : " << tmp_tau << std::endl;
                            std::cout << "T0t : " << grid.T0t_loc[I2V(iip,jjt,kkr)] << std::endl;
                            std::cout << "T_t_2 = "
                                      << grid.T0v_loc[I2V(iip,jjt,kkr)] * (grid.tau_loc[I2V(iip, jjt+1, kkr)] - tmp_tau) / grid.dt
                                      + grid.T0t_loc[I2V(iip,jjt,kkr)] * tmp_tau << std::endl;
                                      break;
                    }
                }

            }
        }
    }

    // third catalog: characteristic travels through lines in 1D volume (6 cases)
    // case: 1-2
    // characteristic travels along r-axis, force H_p2, H_p3 = 0, that is, T_p = T_t = 0
    // plug the constraint into eikonal equation, we have the equation:   a*T_r^2 = s^2
    for (int i_case = 0; i_case < 2; i_case++){
        switch (i_case){
            case 0:     //characteristic travels from  -r
                if (kkr ==  0){
                    continue;
                }
                ar = ar1; br = br1;
                break;
            case 1:     //characteristic travels from  +r
                if (kkr ==  nr-1){
                    continue;
                }
                ar = ar2; br = br2;
                break;
        }

        // plug T_t, T_r into eikonal equation, solve the quadratic equation:  a*(ar*tau+br)^2 = s^2
        // simply, we have two solutions
        for (int i_solution = 0; i_solution < 2; i_solution++){
            // solutions
            switch (i_solution){
                case 0:
                    tmp_tau = ( std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)/grid.fac_a_loc[I2V(iip,jjt,kkr)]) - br)/ar;
                    break;
                case 1:
                    tmp_tau = (-std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)/grid.fac_a_loc[I2V(iip,jjt,kkr)]) - br)/ar;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 0:  //characteristic travels from -r (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.tau_loc[I2V(iip,jjt,kkr-1)] * grid.T0v_loc[I2V(iip,jjt,kkr-1)]
                        && tmp_tau > grid.tau_loc[I2V(iip,jjt,kkr-1)]/_2_CR && tmp_tau > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 1:  //characteristic travels from +r
                    if (tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.tau_loc[I2V(iip,jjt,kkr+1)] * grid.T0v_loc[I2V(iip,jjt,kkr+1)]
                        && tmp_tau > grid.tau_loc[I2V(iip,jjt,kkr+1)]/_2_CR && tmp_tau > 0){
                        is_causality = true;
                    }
                    break;
            }

            // if satisfying the causility condition, retain it as a canditate solution
            if (is_causality) {
                canditate[count_cand] = tmp_tau;
                count_cand += 1;
            }

            if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
                std::cout   << "line case, i_case, i_solution, is_causality, tau, T0, tau*t0: "
                            << i_case << ", " << i_solution << ", " << is_causality << ", " << tmp_tau << ", "
                            << grid.T0v_loc[I2V(iip,jjt,kkr)] << ", " << grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau
                            << ", " << std::endl;
                switch (i_case) {
                    case 0: //characteristic travels from -r
                        std::cout   << "-r: " << grid.T0v_loc[I2V(iip,jjt,kkr-1)]*grid.tau_loc[I2V(iip, jjt, kkr-1)] << ", " << std::endl;
                        break;
                    case 1: //characteristic travels from +r
                        std::cout   << "+r: " << grid.T0v_loc[I2V(iip,jjt,kkr+1)]*grid.tau_loc[I2V(iip, jjt, kkr+1)] << ", " << std::endl;
                        break;
                }
            }

        }

    }

    // case: 3-4
    // characteristic travels along t-axis, force H_p1, H_p3 = 0, that is, T_r = 0; c*T_p-f*T_t = 0
    // plug the constraint into eikonal equation, we have the equation:   (bc-f^2)/c*T_t^2 = s^2
    for (int i_case = 2; i_case < 4; i_case++){
        switch (i_case){
            case 2:     //characteristic travels from  -t
                if (jjt ==  0){
                    continue;
                }
                at = at1; bt = bt1;
                break;
            case 3:     //characteristic travels from  +t
                if (jjt ==  nt-1){
                    continue;
                }
                at = at2; bt = bt2;
                break;
        }

        // plug T_p, T_r into eikonal equation, solve the quadratic equation:  (bc-f^2)/c*(at*tau+bt)^2 = s^2
        // simply, we have two solutions
        for (int i_solution = 0; i_solution < 2; i_solution++){
            // solutions
            switch (i_solution){
                case 0:
                    tmp_tau = ( std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)*grid.fac_c_loc[I2V(iip,jjt,kkr)]/bc_f2) - bt)/at;
                    break;
                case 1:
                    tmp_tau = (-std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)*grid.fac_c_loc[I2V(iip,jjt,kkr)]/bc_f2) - bt)/at;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 2:  //characteristic travels from -t (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.tau_loc[I2V(iip,jjt-1,kkr)] * grid.T0v_loc[I2V(iip,jjt-1,kkr)]
                        && tmp_tau > grid.tau_loc[I2V(iip,jjt-1,kkr)]/_2_CR && tmp_tau > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 3:  //characteristic travels from +t
                    if (tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.tau_loc[I2V(iip,jjt+1,kkr)] * grid.T0v_loc[I2V(iip,jjt+1,kkr)]
                        && tmp_tau > grid.tau_loc[I2V(iip,jjt+1,kkr)]/_2_CR && tmp_tau > 0){
                        is_causality = true;
                    }
                    break;

            }

            // if satisfying the causility condition, retain it as a canditate solution
            if (is_causality) {
                canditate[count_cand] = tmp_tau;
                count_cand += 1;
            }

            if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
                std::cout   << "line case, i_case, i_solution, is_causality, tau, T0, tau*t0: "
                            << i_case << ", " << i_solution << ", " << is_causality << ", " << tmp_tau << ", "
                            << grid.T0v_loc[I2V(iip,jjt,kkr)] << ", " << grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau
                            << ", " << std::endl;
                switch (i_case) {
                    case 2: //characteristic travels from -t
                        std::cout   << "-t: " << grid.T0v_loc[I2V(iip,jjt-1,kkr)]*grid.tau_loc[I2V(iip, jjt-1, kkr)] << ", " << std::endl;
                        break;
                    case 3: //characteristic travels from +t
                        std::cout   << "+t: " << grid.T0v_loc[I2V(iip,jjt+1,kkr)]*grid.tau_loc[I2V(iip, jjt+1, kkr)] << ", " << std::endl;
                        break;
                }
            }

        }
    }

    // case: 5-6
    // characteristic travels along p-axis, force H_p1, H_p2 = 0, that is, T_r = 0; b*T_t-f*T_p = 0
    // plug the constraint into eikonal equation, we have the equation:   (bc-f^2)/b*T_p^2 = s^2
    for (int i_case = 4; i_case < 6; i_case++){
        switch (i_case){
            case 4:     //characteristic travels from  -p
                if (iip ==  0){
                    continue;
                }
                ap = ap1; bp = bp1;
                break;
            case 5:     //characteristic travels from  +p
                if (iip ==  np-1){
                    continue;
                }
                ap = ap2; bp = bp2;
                break;

        }

        // plug T_t, T_r into eikonal equation, solve the quadratic equation:  (bc-f^2)/b*(ap*tau+bp)^2 = s^2
        // simply, we have two solutions
        for (int i_solution = 0; i_solution < 2; i_solution++){
            // solutions
            switch (i_solution){
                case 0:
                    tmp_tau = ( std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)*grid.fac_b_loc[I2V(iip,jjt,kkr)]/bc_f2) - bp)/ap;
                    break;
                case 1:
                    tmp_tau = (-std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)*grid.fac_b_loc[I2V(iip,jjt,kkr)]/bc_f2) - bp)/ap;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 4:  //characteristic travels from -p (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.tau_loc[I2V(iip-1,jjt,kkr)] * grid.T0v_loc[I2V(iip-1,jjt,kkr)]
                        && tmp_tau > grid.tau_loc[I2V(iip-1,jjt,kkr)]/_2_CR && tmp_tau > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 5:  //characteristic travels from +p
                    if (tmp_tau * grid.T0v_loc[I2V(iip,jjt,kkr)] > grid.tau_loc[I2V(iip+1,jjt,kkr)] * grid.T0v_loc[I2V(iip+1,jjt,kkr)]
                        && tmp_tau > grid.tau_loc[I2V(iip+1,jjt,kkr)]/_2_CR && tmp_tau > 0){
                        is_causality = true;
                    }
                    break;

            }

            if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
                std::cout   << "line case, i_case, i_solution, is_causality, tau, T0, tau*t0: "
                            << i_case << ", " << i_solution << ", " << is_causality << ", " << tmp_tau << ", "
                            << grid.T0v_loc[I2V(iip,jjt,kkr)] << ", " << grid.T0v_loc[I2V(iip,jjt,kkr)]*tmp_tau
                            << ", " << std::endl;
                switch (i_case) {
                    case 4: //characteristic travels from -p
                        std::cout   << "-p: " << grid.T0v_loc[I2V(iip-1,jjt,kkr)]*grid.tau_loc[I2V(iip-1, jjt, kkr)] << ", " << std::endl;
                        break;
                    case 5: //characteristic travels from +p
                        std::cout   << "+p: " << grid.T0v_loc[I2V(iip+1,jjt,kkr)]*grid.tau_loc[I2V(iip+1, jjt, kkr)] << ", " << std::endl;
                        break;
                }
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
        grid.tau_loc[I2V(iip, jjt, kkr)] = std::min(grid.tau_loc[I2V(iip, jjt, kkr)], canditate[i_cand]);

        if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
            std::cout << "tau_loc: " << grid.tau_loc[I2V(iip, jjt, kkr)] << ", i_cand: " << i_cand << ", cand: " << canditate[i_cand] << std::endl;
        }

        if (grid.tau_loc[I2V(iip, jjt, kkr)] < 0 ){
            std::cout << "error, tau_loc < 0. iip: " << iip << ", jjt: " << jjt << ", kkr: " << kkr << std::endl;
            exit(1);
        }

    }



    if (check_out && iip == iip_out && jjt == jjt_out && kkr == kkr_out){
            std::cout << "tau_loc: " << grid.tau_loc[I2V(iip, jjt, kkr)]
                      << ",  T_loc: " << grid.T0v_loc[I2V(iip,jjt,kkr)] * grid.tau_loc[I2V(iip, jjt, kkr)]
                      << std::endl;
    }
}

void Iterator::calculate_stencil_1st_order(Grid& grid, int& iip, int& jjt, int&kkr){
    sigr = SWEEPING_COEFF*std::sqrt(grid.fac_a_loc[I2V(iip, jjt, kkr)])*grid.T0v_loc[I2V(iip, jjt, kkr)];
    sigt = SWEEPING_COEFF*std::sqrt(grid.fac_b_loc[I2V(iip, jjt, kkr)])*grid.T0v_loc[I2V(iip, jjt, kkr)];
    sigp = SWEEPING_COEFF*std::sqrt(grid.fac_c_loc[I2V(iip, jjt, kkr)])*grid.T0v_loc[I2V(iip, jjt, kkr)];
    coe  = _1_CR/((sigr/dr)+(sigt/dt)+(sigp/dp));

    pp1 = (grid.tau_loc[I2V(iip  , jjt  , kkr  )] - grid.tau_loc[I2V(iip-1, jjt  , kkr  )])/dp;
    pp2 = (grid.tau_loc[I2V(iip+1, jjt  , kkr  )] - grid.tau_loc[I2V(iip  , jjt  , kkr  )])/dp;

    pt1 = (grid.tau_loc[I2V(iip  , jjt  , kkr  )] - grid.tau_loc[I2V(iip  , jjt-1, kkr  )])/dt;
    pt2 = (grid.tau_loc[I2V(iip  , jjt+1, kkr  )] - grid.tau_loc[I2V(iip  , jjt  , kkr  )])/dt;

    pr1 = (grid.tau_loc[I2V(iip  , jjt  , kkr  )] - grid.tau_loc[I2V(iip  , jjt  , kkr-1)])/dr;
    pr2 = (grid.tau_loc[I2V(iip  , jjt  , kkr+1)] - grid.tau_loc[I2V(iip  , jjt  , kkr  )])/dr;

    // LF Hamiltonian
    Htau = calc_LF_Hamiltonian(grid, pp1, pp2, pt1, pt2, pr1, pr2, iip, jjt, kkr);

    grid.tau_loc[I2V(iip, jjt, kkr)] += coe*(grid.fun_loc[I2V(iip, jjt, kkr)] - Htau) \
                                      + coe*(sigr*(pr2-pr1)/_2_CR + sigt*(pt2-pt1)/_2_CR + sigp*(pp2-pp1)/_2_CR);

}


void Iterator::calculate_stencil_3rd_order(Grid& grid, int& iip, int& jjt, int&kkr){
    sigr = SWEEPING_COEFF*std::sqrt(grid.fac_a_loc[I2V(iip, jjt, kkr)])*grid.T0v_loc[I2V(iip, jjt, kkr)];
    sigt = SWEEPING_COEFF*std::sqrt(grid.fac_b_loc[I2V(iip, jjt, kkr)])*grid.T0v_loc[I2V(iip, jjt, kkr)];
    sigp = SWEEPING_COEFF*std::sqrt(grid.fac_c_loc[I2V(iip, jjt, kkr)])*grid.T0v_loc[I2V(iip, jjt, kkr)];
    coe  = _1_CR/((sigr/dr)+(sigt/dt)+(sigp/dp));


    // direction p
    if (iip == 1) {
        pp1 = (grid.tau_loc[I2V(iip,jjt,kkr)] \
             - grid.tau_loc[I2V(iip-1,jjt,kkr)]) / dp;

        wp2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip+2,jjt,kkr)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip-1,jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip+1,jjt,kkr)]) )) );

        pp2 = (_1_CR - wp2) * (         grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                              -         grid.tau_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp \
                   + wp2  * ( - _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                              + _4_CR * grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                              -         grid.tau_loc[I2V(iip+2,jjt,kkr)] ) / _2_CR / dp;

    } else if (iip == np-2) {
        wp1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip-1,jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip-2,jjt,kkr)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip-1,jjt,kkr)]) )) );

        pp1 = (_1_CR - wp1) * (           grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                                -         grid.tau_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp \
                     + wp1  * ( + _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                                - _4_CR * grid.tau_loc[I2V(iip-1,jjt,kkr)] \
                                +         grid.tau_loc[I2V(iip-2,jjt,kkr)] ) / _2_CR / dp;

        pp2 = (grid.tau_loc[I2V(iip+1,jjt,kkr)] - grid.tau_loc[I2V(iip,jjt,kkr)]) / dp;

    } else {
        wp1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip-1,jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip-2,jjt,kkr)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip-1,jjt,kkr)]) )) );

        pp1 = (_1_CR - wp1) * (         grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                              -         grid.tau_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp \
                   + wp1  * ( + _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                              - _4_CR * grid.tau_loc[I2V(iip-1,jjt,kkr)] \
                              +         grid.tau_loc[I2V(iip-2,jjt,kkr)] ) / _2_CR / dp;

        wp2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip+2,jjt,kkr)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip-1,jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip+1,jjt,kkr)]) )) );

        pp2 = (_1_CR - wp2) * (           grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                                -         grid.tau_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp \
                     + wp2  * ( - _3_CR * grid.tau_loc[I2V(iip  ,jjt,kkr)] \
                                + _4_CR * grid.tau_loc[I2V(iip+1,jjt,kkr)] \
                                -         grid.tau_loc[I2V(iip+2,jjt,kkr)] ) / _2_CR / dp;

    }

    // direction t
    if (jjt == 1) {
        pt1 = (grid.tau_loc[I2V(iip,jjt  ,kkr)] \
             - grid.tau_loc[I2V(iip,jjt-1,kkr)]) / dt;

        wt2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt+2,kkr)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip,jjt-1,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt+1,kkr)]) )) );

        pt2 = (_1_CR - wt2) * (         grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                              -         grid.tau_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                   + wt2  * ( - _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                              + _4_CR * grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                              -         grid.tau_loc[I2V(iip,jjt+2,kkr)] ) / _2_CR / dt;

    } else if (jjt == nt-2) {
        wt1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt-1,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt-2,kkr)]) ) \

                                         / (eps + my_square(       grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                                                            -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                            +      grid.tau_loc[I2V(iip,jjt-1,kkr)]) )) );

        pt1 = (_1_CR - wt1) * (           grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                                -         grid.tau_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                     + wt1  * ( + _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                                - _4_CR * grid.tau_loc[I2V(iip,jjt-1,kkr)] \
                                +         grid.tau_loc[I2V(iip,jjt-2,kkr)] ) / _2_CR / dt;

        pt2 = (grid.tau_loc[I2V(iip,jjt+1,kkr)] - grid.tau_loc[I2V(iip,jjt,kkr)]) / dt;

    } else {
        wt1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt-1,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt-2,kkr)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt-1,kkr)]) )) );

        pt1 = (_1_CR - wt1) * (           grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                                -         grid.tau_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                     + wt1  * ( + _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                                - _4_CR * grid.tau_loc[I2V(iip,jjt-1,kkr)] \
                                +         grid.tau_loc[I2V(iip,jjt-2,kkr)] ) / _2_CR / dt;

        wt2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt+2,kkr)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip,jjt-1,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt+1,kkr)]) )) );

        pt2 = (_1_CR - wt2) * (           grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                                -         grid.tau_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                     + wt2  * ( - _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                                + _4_CR * grid.tau_loc[I2V(iip,jjt+1,kkr)] \
                                -         grid.tau_loc[I2V(iip,jjt+2,kkr)] ) / _2_CR / dt;

    }

    // direction r
    if (kkr == 1) {
        pr1 = (grid.tau_loc[I2V(iip,jjt,kkr  )] \
             - grid.tau_loc[I2V(iip,jjt,kkr-1)]) / dr;

        wr2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                                           +      grid.tau_loc[I2V(iip,jjt,kkr+2)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip,jjt,kkr-1)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt,kkr+1)]) )) );

        pr2 = (_1_CR - wr2) * (           grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.tau_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                     + wr2  * ( - _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                                + _4_CR * grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.tau_loc[I2V(iip,jjt,kkr+2)] ) / _2_CR / dr;

    } else if (kkr == nr - 2) {
        wr1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt,kkr-1)] \
                                                           +      grid.tau_loc[I2V(iip,jjt,kkr-2)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt,kkr-1)]) )) );

        pr1 = (_1_CR - wr1) * (            grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                -          grid.tau_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                      + wr1  * ( + _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                                 - _4_CR * grid.tau_loc[I2V(iip,jjt,kkr-1)] \
                                 +         grid.tau_loc[I2V(iip,jjt,kkr-2)] ) / _2_CR / dr;

        pr2 = (grid.tau_loc[I2V(iip,jjt,kkr+1)] - grid.tau_loc[I2V(iip,jjt,kkr)]) / dr;

    } else {
        wr1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt,kkr-1)] \
                                                           +      grid.tau_loc[I2V(iip,jjt,kkr-2)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt,kkr-1)]) )) );

        pr1 = (_1_CR - wr1) * (           grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.tau_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                     + wr1  * ( + _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                                - _4_CR * grid.tau_loc[I2V(iip,jjt,kkr-1)] \
                                +         grid.tau_loc[I2V(iip,jjt,kkr-2)] ) / _2_CR / dr;

        wr2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.tau_loc[I2V(iip,jjt,kkr)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                                           +      grid.tau_loc[I2V(iip,jjt,kkr+2)]) ) \

                                         / (eps + my_square(      grid.tau_loc[I2V(iip,jjt,kkr-1)] \
                                                           -_2_CR*grid.tau_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.tau_loc[I2V(iip,jjt,kkr+1)]) )) );

        pr2 = (_1_CR - wr2) * (           grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.tau_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                     + wr2  * ( - _3_CR * grid.tau_loc[I2V(iip,jjt,kkr)] \
                                + _4_CR * grid.tau_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.tau_loc[I2V(iip,jjt,kkr+2)] ) / _2_CR / dr;

    }

    // LF Hamiltonian
    Htau = calc_LF_Hamiltonian(grid, pp1, pp2, pt1, pt2, pr1, pr2, iip, jjt, kkr);

    // update tau
    grid.tau_loc[I2V(iip,jjt,kkr)] += coe * ((grid.fun_loc[I2V(iip,jjt,kkr)] - Htau) + (sigr*(pr2-pr1) + sigt*(pt2-pt1) + sigp*(pp2-pp1))/_2_CR);

}


void Iterator::calculate_stencil_adj(Grid& grid, int& iip, int& jjt, int& kkr){
    //CUSTOMREAL tmpr1 = (grid.r_loc_1d[kkr-1]+grid.r_loc_1d[kkr])/_2_CR;
    //CUSTOMREAL tmpr2 = (grid.r_loc_1d[kkr+1]+grid.r_loc_1d[kkr])/_2_CR;
    CUSTOMREAL tmpt1 = (grid.t_loc_1d[jjt-1]+grid.t_loc_1d[jjt])/_2_CR;
    CUSTOMREAL tmpt2 = (grid.t_loc_1d[jjt]  +grid.t_loc_1d[jjt+1])/_2_CR;

    CUSTOMREAL a1  = - (_1_CR+grid.zeta_loc[I2V(iip,jjt,kkr-1)]+grid.zeta_loc[I2V(iip,jjt,kkr)]) * (grid.T_loc[I2V(iip,jjt,kkr)]-grid.T_loc[I2V(iip,jjt,kkr-1)]) / dr;
    CUSTOMREAL a1m = (a1 - std::abs(a1))/_2_CR;
    CUSTOMREAL a1p = (a1 + std::abs(a1))/_2_CR;

    CUSTOMREAL a2  = - (_1_CR+grid.zeta_loc[I2V(iip,jjt,kkr)]+grid.zeta_loc[I2V(iip,jjt,kkr+1)]) * (grid.T_loc[I2V(iip,jjt,kkr+1)]-grid.T_loc[I2V(iip,jjt,kkr)]) / dr;
    CUSTOMREAL a2m = (a2 - std::abs(a2))/_2_CR;
    CUSTOMREAL a2p = (a2 + std::abs(a2))/_2_CR;

    CUSTOMREAL b1  = - (_1_CR-grid.xi_loc[ I2V(iip,jjt-1,kkr)]-grid.xi_loc[ I2V(iip,jjt,kkr)]) /  my_square(grid.r_loc_1d[kkr]) * (grid.T_loc[I2V(iip,jjt,kkr)]-grid.T_loc[I2V(iip,jjt-1,kkr)]) / dt \
                     - (      grid.eta_loc[I2V(iip,jjt-1,kkr)]+grid.eta_loc[I2V(iip,jjt,kkr)]) / (my_square(grid.r_loc_1d[kkr])*std::cos(tmpt1)) / (_4_CR*dp) \
                     * ((grid.T_loc[I2V(iip+1,jjt-1,kkr)]-grid.T_loc[I2V(iip-1,jjt-1,kkr)]) \
                     + (grid.T_loc[I2V(iip+1,jjt,kkr)]  -grid.T_loc[I2V(iip-1,jjt,kkr)]));
    CUSTOMREAL b1m = (b1 - std::abs(b1))/_2_CR;
    CUSTOMREAL b1p = (b1 + std::abs(b1))/_2_CR;

    CUSTOMREAL b2  = - (_1_CR-grid.xi_loc[ I2V(iip,jjt,kkr)]-grid.xi_loc[ I2V(iip,jjt+1,kkr)]) /  my_square(grid.r_loc_1d[kkr]) * (grid.T_loc[I2V(iip,jjt+1,kkr)]-grid.T_loc[I2V(iip,jjt,kkr)]) / dt \
                     - (      grid.eta_loc[I2V(iip,jjt,kkr)]+grid.eta_loc[I2V(iip,jjt+1,kkr)]) / (my_square(grid.r_loc_1d[kkr])*std::cos(tmpt2)) / (_4_CR*dp) \
                     * ((grid.T_loc[I2V(iip+1,jjt,kkr)]-grid.T_loc[I2V(iip-1,jjt,kkr)]) \
                     + (grid.T_loc[I2V(iip+1,jjt+1,kkr)]-grid.T_loc[I2V(iip-1,jjt+1,kkr)]));
    CUSTOMREAL b2m = (b2 - std::abs(b2))/_2_CR;
    CUSTOMREAL b2p = (b2 + std::abs(b2))/_2_CR;

    CUSTOMREAL c1  =  - (_1_CR+grid.xi_loc[I2V(iip-1,jjt,kkr)]+grid.xi_loc[I2V(iip,jjt,kkr)]) / (my_square(grid.r_loc_1d[kkr])*my_square(std::cos(grid.t_loc_1d[jjt]))) \
                      * (grid.T_loc[I2V(iip,jjt,kkr)]-grid.T_loc[I2V(iip-1,jjt,kkr)]) / dp \
                      - (grid.eta_loc[I2V(iip-1,jjt,kkr)]+grid.eta_loc[I2V(iip,jjt,kkr)]) / (my_square(grid.r_loc_1d[kkr])*std::cos(grid.t_loc_1d[jjt])) / (_4_CR*dt) \
                      * ((grid.T_loc[I2V(iip-1,jjt+1,kkr)]-grid.T_loc[I2V(iip-1,jjt-1,kkr)]) \
                      + (grid.T_loc[I2V(iip,  jjt+1,kkr)]-grid.T_loc[I2V(iip,  jjt-1,kkr)]));
    CUSTOMREAL c1m = (c1 - std::abs(c1))/_2_CR;
    CUSTOMREAL c1p = (c1 + std::abs(c1))/_2_CR;

    CUSTOMREAL c2  =  - (_1_CR+grid.xi_loc[I2V(iip,jjt,kkr)]+grid.xi_loc[I2V(iip+1,jjt,kkr)]) / (my_square(grid.r_loc_1d[kkr])*my_square(std::cos(grid.t_loc_1d[jjt]))) \
                      * (grid.T_loc[I2V(iip+1,jjt,kkr)]-grid.T_loc[I2V(iip,jjt,kkr)]) / dp \
                      - (grid.eta_loc[I2V(iip,jjt,kkr)]+grid.eta_loc[I2V(iip+1,jjt,kkr)]) / (my_square(grid.r_loc_1d[kkr])*std::cos(grid.t_loc_1d[jjt])) / (_4_CR*dt) \
                      * ((grid.T_loc[I2V(iip,jjt+1,kkr)]-grid.T_loc[I2V(iip,jjt-1,kkr)]) \
                      + (grid.T_loc[I2V(iip+1,jjt+1,kkr)]-grid.T_loc[I2V(iip+1,jjt-1,kkr)]));
    CUSTOMREAL c2m = (c2 - std::abs(c2))/_2_CR;
    CUSTOMREAL c2p = (c2 + std::abs(c2))/_2_CR;

    // additional terms of divergence in spherical cooridinate
    CUSTOMREAL d   = - _2_CR * (_1_CR+_2_CR*grid.zeta_loc[I2V(iip,jjt,kkr)]) / grid.r_loc_1d[kkr] \
                     * (grid.T_loc[I2V(iip,jjt,kkr+1)]-grid.T_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                     + (_1_CR-_2_CR*grid.xi_loc[I2V(iip,jjt,kkr)]) * std::sin(grid.t_loc_1d[jjt]) / (my_square(grid.r_loc_1d[kkr])*std::cos(grid.t_loc_1d[jjt])) \
                     * (grid.T_loc[I2V(iip,jjt+1,kkr)]-grid.T_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                     + _2_CR * grid.eta_loc[I2V(iip,jjt,kkr)] * std::sin(grid.t_loc_1d[jjt]) / (my_square(grid.r_loc_1d[kkr])*my_square(std::cos(grid.t_loc_1d[jjt]))) \
                     * (grid.T_loc[I2V(iip+1,jjt,kkr)]-grid.T_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp;

    // stabilize the calculation on the boundary
    // if (kkr == 1 && grid.k_first() && a1p > 0 ){
    //     a1m = -a1p;
    //     a1p = 0;
    // }

    // coe
    CUSTOMREAL coe = (a2p-a1m)/dr + (b2p-b1m)/dt + (c2p-c1m)/dp;

    if (isZeroAdj(coe)) {   // here the traveltime is larger than surrounding
        grid.tau_loc[I2V(iip,jjt,kkr)] = _0_CR;
    } else {
        // hamiltonian
        CUSTOMREAL Hadj = (a1p*grid.tau_loc[I2V(iip,jjt,kkr-1)] - a2m*grid.tau_loc[I2V(iip,jjt,kkr+1)]) / dr \
                        + (b1p*grid.tau_loc[I2V(iip,jjt-1,kkr)] - b2m*grid.tau_loc[I2V(iip,jjt+1,kkr)]) / dt \
                        + (c1p*grid.tau_loc[I2V(iip-1,jjt,kkr)] - c2m*grid.tau_loc[I2V(iip+1,jjt,kkr)]) / dp;

        grid.tau_loc[I2V(iip,jjt,kkr)] = (grid.tau_old_loc[I2V(iip,jjt,kkr)] + Hadj) / (coe + d);
    }
}

// no multiplicative factorization
void Iterator::calculate_stencil_1st_order_upwind_tele(Grid&grid, int&iip, int&jjt, int&kkr){

    // preparations

    count_cand = 0;
    // forward and backward partial differential discretization
    //  T_p = ap*T(iip, jjt, kkr)+bp; two cases:  1/dp * T(iip,jjt,kkr) - T(iip-1,jjt,kkr)/dp  or  -1/dp * T(iip,jjt,kkr) + T(iip+1,jjt,kkr)/dp
    //  T_t = at*T(iip, jjt, kkr)+bt;
    //  T_r = ar*T(iip, jjt, kkr)+br;
    if (iip > 0){
        ap1 =  _1_CR/dp;
        bp1 = -grid.T_loc[I2V(iip-1, jjt, kkr)]/dp;
    }
    if (iip < np-1){
        ap2 = -_1_CR/dp;
        bp2 =  grid.T_loc[I2V(iip+1, jjt, kkr)]/dp;
    }

    if (jjt > 0){
        at1 =  _1_CR/dt;
        bt1 = -grid.T_loc[I2V(iip, jjt-1, kkr)]/dt;
    }
    if (jjt < nt-1){
        at2 = -_1_CR/dt;
        bt2 =  grid.T_loc[I2V(iip, jjt+1, kkr)]/dt;
    }

    if (kkr > 0){
        ar1 =  _1_CR/dr;
        br1 = -grid.T_loc[I2V(iip, jjt, kkr-1)]/dr;
    }
    if (kkr < nr-1){
        ar2 = -_1_CR/dr;
        br2 =  grid.T_loc[I2V(iip, jjt, kkr+1)]/dr;
    }
    bc_f2 = grid.fac_b_loc[I2V(iip,jjt,kkr)]*grid.fac_c_loc[I2V(iip,jjt,kkr)] - std::pow(grid.fac_f_loc[I2V(iip,jjt,kkr)],_2_CR);

    // start to find candidate solutions

    // first catalog: characteristic travels through tetrahedron in 3D volume (8 cases)
    for (int i_case = 0; i_case < 8; i_case++){

        // determine discretization of T_p,T_t,T_r
        switch (i_case) {
            case 0:    // characteristic travels from -p, -t, -r
                if (iip == 0 || jjt == 0 || kkr == 0)
                    continue;
                ap = ap1; bp = bp1;
                at = at1; bt = bt1;
                ar = ar1; br = br1;
                break;
            case 1:     // characteristic travels from -p, -t, +r
                if (iip == 0 || jjt == 0 || kkr == nr-1)
                    continue;
                ap = ap1; bp = bp1;
                at = at1; bt = bt1;
                ar = ar2; br = br2;
                break;
            case 2:    // characteristic travels from -p, +t, -r
                if (iip == 0 || jjt == nt-1 || kkr == 0)
                    continue;
                ap = ap1; bp = bp1;
                at = at2; bt = bt2;
                ar = ar1; br = br1;
                break;
            case 3:     // characteristic travels from -p, +t, +r
                if (iip == 0 || jjt == nt-1 || kkr == nr-1)
                    continue;
                ap = ap1; bp = bp1;
                at = at2; bt = bt2;
                ar = ar2; br = br2;
                break;
            case 4:    // characteristic travels from +p, -t, -r
                if (iip == np-1 || jjt == 0 || kkr == 0)
                    continue;
                ap = ap2; bp = bp2;
                at = at1; bt = bt1;
                ar = ar1; br = br1;
                break;
            case 5:     // characteristic travels from +p, -t, +r
                if (iip == np-1 || jjt == 0 || kkr == nr-1)
                    continue;
                ap = ap2; bp = bp2;
                at = at1; bt = bt1;
                ar = ar2; br = br2;
                break;
            case 6:    // characteristic travels from +p, +t, -r
                if (iip == np-1 || jjt == nt-1 || kkr == 0)
                    continue;
                ap = ap2; bp = bp2;
                at = at2; bt = bt2;
                ar = ar1; br = br1;
                break;
            case 7:     // characteristic travels from +p, +t, +r
                if (iip == np-1 || jjt == nt-1 || kkr == nr-1)
                    continue;
                ap = ap2; bp = bp2;
                at = at2; bt = bt2;
                ar = ar2; br = br2;
                break;
        }

        // plug T_p, T_t, T_r into eikonal equation, solving the quadratic equation with respect to tau(iip,jjt,kkr)
        // that is a*(ar*T+br)^2 + b*(at*T+bt)^2 + c*(ap*T+bp)^2 - 2*f*(at*T+bt)*(ap*T+bp) = s^2
        eqn_a = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(ar, _2_CR) + grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(at, _2_CR)
              + grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(ap, _2_CR) - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * at * ap;
        eqn_b = _2_CR*grid.fac_a_loc[I2V(iip,jjt,kkr)] * ar * br + _2_CR*grid.fac_b_loc[I2V(iip,jjt,kkr)] * at * bt
              + _2_CR*grid.fac_c_loc[I2V(iip,jjt,kkr)] * ap * bp - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * (at*bp + bt*ap);
        eqn_c = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(br, _2_CR) + grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(bt, _2_CR)
              + grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(bp, _2_CR) - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * bt * bp
              - std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR);
        eqn_Delta = std::pow(eqn_b, _2_CR) - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0){    // one or two real solutions
            for (int i_solution = 0; i_solution < 2; i_solution++){
                // solutions
                switch (i_solution){
                    case 0:
                        tmp_T = (-eqn_b + std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                    case 1:
                        tmp_T = (-eqn_b - std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                }

                // check the causality condition: the characteristic passing through (iip,jjt,kkr) is in between used three sides
                // characteristic direction is (dr/dt, dtheta/dt, tphi/dt) = (H_p1,H_p2,H_p3), p1 = T_r, p2 = T_t, p3 = T_p
                T_r = ar*tmp_T + br;
                T_t = at*tmp_T + bt;
                T_p = ap*tmp_T + bp;

                charact_r = grid.fac_a_loc[I2V(iip,jjt,kkr)]*T_r;
                charact_t = grid.fac_b_loc[I2V(iip,jjt,kkr)]*T_t - grid.fac_f_loc[I2V(iip,jjt,kkr)]*T_p;
                charact_p = grid.fac_c_loc[I2V(iip,jjt,kkr)]*T_p - grid.fac_f_loc[I2V(iip,jjt,kkr)]*T_t;

                is_causality = false;
                switch (i_case){
                    case 0:  //characteristic travels from -p, -t, -r
                        if (charact_p >= 0 && charact_t >= 0 && charact_r >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 1:  //characteristic travels from -p, -t, +r
                        if (charact_p >= 0 && charact_t >= 0 && charact_r <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 2:  //characteristic travels from -p, +t, -r
                        if (charact_p >= 0 && charact_t <= 0 && charact_r >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 3:  //characteristic travels from -p, +t, +r
                        if (charact_p >= 0 && charact_t <= 0 && charact_r <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 4:  //characteristic travels from +p, -t, -r
                        if (charact_p <= 0 && charact_t >= 0 && charact_r >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 5:  //characteristic travels from +p, -t, +r
                        if (charact_p <= 0 && charact_t >= 0 && charact_r <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 6:  //characteristic travels from +p, +t, -r
                        if (charact_p <= 0 && charact_t <= 0 && charact_r >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 7:  //characteristic travels from +p, +t, +r
                        if (charact_p <= 0 && charact_t <= 0 && charact_r <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_T;
                    count_cand += 1;
                    // if (iip == 1 && jjt ==1 && kkr ==1){
                    //     std::cout << "id_sim: " << id_sim << ", cd volumn icase " << i_case << ": " << canditate[count_cand-1] << std::endl;
                    // }
                }

            }

        }
    }

    // second catalog: characteristic travels through triangles in 2D volume (12 cases)
    // case: 1-4
    // characteristic on r-t plane, force H_p3 = H_(T_p) = 0, that is, c*T_p-f*T_t = 0
    // plug the constraint into eikonal equation, we have the equation:  a*T_r^2 + (bc-f^2)/c*T_t^2 = s^2
    for (int i_case = 0; i_case < 4; i_case++){
        switch (i_case){
            case 0:     //characteristic travels from  -t, -r
                if (jjt ==  0 || kkr ==  0){
                    continue;
                }
                at = at1; bt = bt1;
                ar = ar1; br = br1;
                break;
            case 1:     //characteristic travels from  -t, +r
                if (jjt ==  0 || kkr ==  nr-1){
                    continue;
                }
                at = at1; bt = bt1;
                ar = ar2; br = br2;
                break;
            case 2:     //characteristic travels from  +t, -r
                if (jjt ==  nt-1 || kkr ==  0){
                    continue;
                }
                at = at2; bt = bt2;
                ar = ar1; br = br1;
                break;
            case 3:     //characteristic travels from  +t, +r
                if (jjt ==  nt-1 || kkr ==  nr-1){
                    continue;
                }
                at = at2; bt = bt2;
                ar = ar2; br = br2;
                break;
        }

        // plug T_t, T_r into eikonal equation, solve the quadratic equation:  a*(ar*tau+br)^2 + (bc-f^2)/c*(at*tau+bt)^2 = s^2
        eqn_a = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(ar, _2_CR) + bc_f2/grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(at, _2_CR);
        eqn_b = _2_CR*grid.fac_a_loc[I2V(iip,jjt,kkr)] * ar * br + _2_CR*bc_f2/grid.fac_c_loc[I2V(iip,jjt,kkr)] * at * bt;
        eqn_c = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(br, _2_CR) + bc_f2/grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(bt, _2_CR)
              - std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR);
        eqn_Delta = std::pow(eqn_b, _2_CR) - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0){    // one or two real solutions
            for (int i_solution = 0; i_solution < 2; i_solution++){
                // solutions
                switch (i_solution){
                    case 0:
                        tmp_T = (-eqn_b + std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                    case 1:
                        tmp_T = (-eqn_b - std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                }

                // check the causality condition:
                T_r = ar*tmp_T + br;
                T_t = at*tmp_T + bt;

                charact_r = grid.fac_a_loc[I2V(iip,jjt,kkr)]*T_r;
                charact_t = bc_f2/grid.fac_c_loc[I2V(iip,jjt,kkr)]*T_t;

                is_causality = false;
                switch (i_case){
                    case 0:  //characteristic travels from -t, -r
                        if (charact_t >= 0 && charact_r >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 1:  //characteristic travels from -t, +r
                        if (charact_t >= 0 && charact_r <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 2:  //characteristic travels from +t, -r
                        if (charact_t <= 0 && charact_r >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 3:  //characteristic travels from +t, +r
                        if (charact_t <= 0 && charact_r <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_T;
                    count_cand += 1;
                    // if (iip == 1 && jjt ==1 && kkr ==1){
                    //     std::cout << "id_sim: " << id_sim << ", cd plane icase " << i_case << ": " << canditate[count_cand-1] << std::endl;
                    // }
                }

            }
        }

    }

    // case: 5-8
    // characteristic on r-p plane, force H_p2 = H_(T_t) = 0, that is, b*T_t-f*T_p = 0
    // plug the constraint into eikonal equation, we have the equation:  a*T_r^2 + (bc-f^2)/b*T_p^2 = s^2
    for (int i_case = 4; i_case < 8; i_case++){
         switch (i_case){
            case 4:     //characteristic travels from  -p, -r
                if (iip ==  0 || kkr ==  0){
                    continue;
                }
                ap = ap1; bp = bp1;
                ar = ar1; br = br1;
                break;
            case 5:     //characteristic travels from  -p, +r
                if (iip ==  0 || kkr ==  nr-1){
                    continue;
                }
                ap = ap1; bp = bp1;
                ar = ar2; br = br2;
                break;
            case 6:     //characteristic travels from  +p, -r
                if (iip ==  np-1 || kkr ==  0){
                    continue;
                }
                ap = ap2; bp = bp2;
                ar = ar1; br = br1;
                break;
            case 7:     //characteristic travels from  +p, +r
                if (iip ==  np-1 || kkr ==  nr-1){
                    continue;
                }
                ap = ap2; bp = bp2;
                ar = ar2; br = br2;
                break;
        }

        // plug T_p, T_r into eikonal equation, solve the quadratic equation:  a*(ar*tau+br)^2 + (bc-f^2)/b*(ap*tau+bp)^2 = s^2
        eqn_a = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(ar, _2_CR) + bc_f2/grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(ap, _2_CR);
        eqn_b = _2_CR*grid.fac_a_loc[I2V(iip,jjt,kkr)] * ar * br + _2_CR*bc_f2/grid.fac_b_loc[I2V(iip,jjt,kkr)] * ap * bp;
        eqn_c = grid.fac_a_loc[I2V(iip,jjt,kkr)] * std::pow(br, _2_CR) + bc_f2/grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(bp, _2_CR)
              - std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR);
        eqn_Delta = std::pow(eqn_b, _2_CR) - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0){    // one or two real solutions
            for (int i_solution = 0; i_solution < 2; i_solution++){
                // solutions
                switch (i_solution){
                    case 0:
                        tmp_T = (-eqn_b + std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                    case 1:
                        tmp_T = (-eqn_b - std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                }

                // check the causality condition:
                T_r = ar*tmp_T + br;
                T_p = ap*tmp_T + bp;

                charact_r = grid.fac_a_loc[I2V(iip,jjt,kkr)]*T_r;
                charact_p = bc_f2/grid.fac_b_loc[I2V(iip,jjt,kkr)]*T_p;

                is_causality = false;
                switch (i_case){
                    case 4:  //characteristic travels from -p, -r
                        if (charact_p >= 0 && charact_r >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 5:  //characteristic travels from -p, +r
                        if (charact_p >= 0 && charact_r <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 6:  //characteristic travels from +p, -r
                        if (charact_p <= 0 && charact_r >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 7:  //characteristic travels from +p, +r
                        if (charact_p <= 0 && charact_r <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_T;
                    count_cand += 1;
                    // if (iip == 1 && jjt ==1 && kkr ==1){
                    //     std::cout << "id_sim: " << id_sim << ", cd plane icase " << i_case << ": " << canditate[count_cand-1] << std::endl;
                    // }
                }

            }
        }
    }

    // case: 9-12
    // characteristic on t-p plane, force H_p1 = H_(T_t) = 0, that is, T_r = 0
    // plug the constraint into eikonal equation, we have the equation:  b*T_t^2 + c*T_p^2 - 2f*T_t*T_p = s^2
    for (int i_case = 8; i_case < 12; i_case++){
        switch (i_case){
            case 8:     //characteristic travels from  -p, -t
                if (iip ==  0 || jjt ==  0){
                    continue;
                }
                ap = ap1; bp = bp1;
                at = at1; bt = bt1;
                break;
            case 9:     //characteristic travels from  -p, +t
                if (iip ==  0 || jjt ==  nt-1){
                    continue;
                }
                ap = ap1; bp = bp1;
                at = at2; bt = bt2;
                break;
            case 10:     //characteristic travels from  +p, -t
                if (iip ==  np-1 || jjt ==  0){
                    continue;
                }
                ap = ap2; bp = bp2;
                at = at1; bt = bt1;
                break;
            case 11:     //characteristic travels from  +p, +t
                if (iip ==  np-1 || jjt ==  nt-1){
                    continue;
                }
                ap = ap2; bp = bp2;
                at = at2; bt = bt2;
                break;
        }

        // plug T_p, T_t into eikonal equation, solve the quadratic equation:  b*(at*tau+bt)^2 + c*(ap*tau+bp)^2 - 2f*(at*tau+bt)*(ap*tau+bp) = s^2
        eqn_a = grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(at, _2_CR)
              + grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(ap, _2_CR) - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * at * ap;
        eqn_b = _2_CR*grid.fac_b_loc[I2V(iip,jjt,kkr)] * at * bt
              + _2_CR*grid.fac_c_loc[I2V(iip,jjt,kkr)] * ap * bp - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * (at*bp + bt*ap);
        eqn_c = grid.fac_b_loc[I2V(iip,jjt,kkr)] * std::pow(bt, _2_CR)
              + grid.fac_c_loc[I2V(iip,jjt,kkr)] * std::pow(bp, _2_CR) - _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * bt * bp
              - std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR);
        eqn_Delta = std::pow(eqn_b, _2_CR) - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0){    // one or two real solutions
            for (int i_solution = 0; i_solution < 2; i_solution++){
                // solutions
                switch (i_solution){
                    case 0:
                        tmp_T = (-eqn_b + std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                    case 1:
                        tmp_T = (-eqn_b - std::sqrt(eqn_Delta))/(_2_CR*eqn_a);
                        break;
                }

                // check the causality condition:
                T_t = at*tmp_T + bt;
                T_p = ap*tmp_T + bp;

                charact_t = grid.fac_b_loc[I2V(iip,jjt,kkr)]*T_t - grid.fac_f_loc[I2V(iip,jjt,kkr)]*T_p;
                charact_p = grid.fac_c_loc[I2V(iip,jjt,kkr)]*T_p - grid.fac_f_loc[I2V(iip,jjt,kkr)]*T_t;

                is_causality = false;
                switch (i_case){
                    case 8:  //characteristic travels from -p, -t
                        if (charact_p >= 0 && charact_t >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 9:  //characteristic travels from -p, +t
                        if (charact_p >= 0 && charact_t <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 10:  //characteristic travels from +p, -t
                        if (charact_p <= 0 && charact_t >= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                    case 11:  //characteristic travels from +p, +t
                        if (charact_p <= 0 && charact_t <= 0 && tmp_T > 0){
                            is_causality = true;
                        }
                        break;
                }

                // if satisfying the causility condition, retain it as a canditate solution
                if (is_causality) {
                    canditate[count_cand] = tmp_T;
                    count_cand += 1;
                    // if (iip == 1 && jjt ==1 && kkr ==1){
                    //     std::cout << "id_sim: " << id_sim << ", cd plane icase " << i_case << ": " << canditate[count_cand-1] << std::endl;
                    // }
                }

            }
        }
    }

    // third catalog: characteristic travels through lines in 1D volume (6 cases)
    // case: 1-2
    // characteristic travels along r-axis, force H_p2, H_p3 = 0, that is, T_p = T_t = 0
    // plug the constraint into eikonal equation, we have the equation:   a*T_r^2 = s^2
    for (int i_case = 0; i_case < 2; i_case++){
        switch (i_case){
            case 0:     //characteristic travels from  -r
                if (kkr ==  0){
                    continue;
                }
                ar = ar1; br = br1;
                break;
            case 1:     //characteristic travels from  +r
                if (kkr ==  nr-1){
                    continue;
                }
                ar = ar2; br = br2;
                break;
        }

        // plug T_t, T_r into eikonal equation, solve the quadratic equation:  a*(ar*tau+br)^2 = s^2
        // simply, we have two solutions
        for (int i_solution = 0; i_solution < 2; i_solution++){
            // solutions
            switch (i_solution){
                case 0:
                    tmp_T = ( std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)/grid.fac_a_loc[I2V(iip,jjt,kkr)]) - br)/ar;
                    break;
                case 1:
                    tmp_T = (-std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)/grid.fac_a_loc[I2V(iip,jjt,kkr)]) - br)/ar;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 0:  //characteristic travels from -r (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_T >= grid.T_loc[I2V(iip,jjt,kkr-1)] && tmp_T > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 1:  //characteristic travels from +r
                    if (tmp_T >= grid.T_loc[I2V(iip,jjt,kkr+1)] && tmp_T > 0){
                        is_causality = true;
                    }
                    break;
            }

            // if satisfying the causility condition, retain it as a canditate solution
            if (is_causality) {
                canditate[count_cand] = tmp_T;
                count_cand += 1;
                // if (iip == 1 && jjt ==1 && kkr ==1){
                //     std::cout << "id_sim: " << id_sim << ", cd line icase " << i_case << ": " << canditate[count_cand-1] << std::endl;
                // }
            }

        }

    }

    // case: 3-4
    // characteristic travels along t-axis, force H_p1, H_p3 = 0, that is, T_r = 0; c*T_p-f*T_t = 0
    // plug the constraint into eikonal equation, we have the equation:   (bc-f^2)/c*T_t^2 = s^2
    for (int i_case = 2; i_case < 4; i_case++){
        switch (i_case){
            case 2:     //characteristic travels from  -t
                if (jjt ==  0){
                    continue;
                }
                at = at1; bt = bt1;
                break;
            case 3:     //characteristic travels from  +t
                if (jjt ==  nt-1){
                    continue;
                }
                at = at2; bt = bt2;
                break;
        }

        // plug T_p, T_r into eikonal equation, solve the quadratic equation:  (bc-f^2)/c*(at*tau+bt)^2 = s^2
        // simply, we have two solutions
        for (int i_solution = 0; i_solution < 2; i_solution++){
            // solutions
            switch (i_solution){
                case 0:
                    tmp_T = ( std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)*grid.fac_c_loc[I2V(iip,jjt,kkr)]/bc_f2) - bt)/at;
                    break;
                case 1:
                    tmp_T = (-std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)*grid.fac_c_loc[I2V(iip,jjt,kkr)]/bc_f2) - bt)/at;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 2:  //characteristic travels from -t (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_T >= grid.T_loc[I2V(iip,jjt-1,kkr)] && tmp_T > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 3:  //characteristic travels from +t
                    if (tmp_T >= grid.T_loc[I2V(iip,jjt+1,kkr)] && tmp_T > 0){
                        is_causality = true;
                    }
                    break;

            }

            // if satisfying the causility condition, retain it as a canditate solution
            if (is_causality) {
                canditate[count_cand] = tmp_T;
                count_cand += 1;
                // if (iip == 1 && jjt ==1 && kkr ==1){
                //     std::cout << "id_sim: " << id_sim << ", cd line icase " << i_case << ": " << canditate[count_cand-1] << std::endl;
                // }
            }

        }
    }

    // case: 5-6
    // characteristic travels along p-axis, force H_p1, H_p2 = 0, that is, T_r = 0; b*T_t-f*T_p = 0
    // plug the constraint into eikonal equation, we have the equation:   (bc-f^2)/b*T_p^2 = s^2
    for (int i_case = 4; i_case < 6; i_case++){
        switch (i_case){
            case 4:     //characteristic travels from  -p
                if (iip ==  0){
                    continue;
                }
                ap = ap1; bp = bp1;
                break;
            case 5:     //characteristic travels from  +p
                if (iip ==  np-1){
                    continue;
                }
                ap = ap2; bp = bp2;
                break;

        }

        // plug T_t, T_r into eikonal equation, solve the quadratic equation:  (bc-f^2)/b*(ap*tau+bp)^2 = s^2
        // simply, we have two solutions
        for (int i_solution = 0; i_solution < 2; i_solution++){
            // solutions
            switch (i_solution){
                case 0:
                    tmp_T = ( std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)*grid.fac_b_loc[I2V(iip,jjt,kkr)]/bc_f2) - bp)/ap;
                    break;
                case 1:
                    tmp_T = (-std::sqrt(std::pow(grid.fun_loc[I2V(iip,jjt,kkr)], _2_CR)*grid.fac_b_loc[I2V(iip,jjt,kkr)]/bc_f2) - bp)/ap;
                    break;
            }

            // check the causality condition:

            is_causality = false;
            switch (i_case){
                case 4:  //characteristic travels from -p (we can simply compare the traveltime, which is the same as check the direction of characteristic)
                    if (tmp_T >= grid.T_loc[I2V(iip-1,jjt,kkr)] && tmp_T > 0){   // this additional condition ensures the causality near the source
                        is_causality = true;
                    }
                    break;
                case 5:  //characteristic travels from +p
                    if (tmp_T >= grid.T_loc[I2V(iip+1,jjt,kkr)] && tmp_T > 0){
                        is_causality = true;
                    }
                    break;

            }

            // if satisfying the causility condition, retain it as a canditate solution
            if (is_causality) {
                canditate[count_cand] = tmp_T;
                count_cand += 1;
                // if (iip == 1 && jjt ==1 && kkr ==1){
                //     std::cout << "id_sim: " << id_sim << ", cd line icase " << i_case << ": " << canditate[count_cand-1] << std::endl;
                // }
            }
        }
    }

    // final, choose the minimum candidate solution as the updated value
    for (int i_cand = 0; i_cand < count_cand; i_cand++){
        grid.T_loc[I2V(iip, jjt, kkr)] = std::min(grid.T_loc[I2V(iip, jjt, kkr)], canditate[i_cand]);
    }

    // if (iip == 1 && jjt ==1 && kkr ==1){
    //     std::cout << "id_sim: " << id_sim << ", ckp 2, "
    //               << "T values: " << grid.T_loc[I2V(iip, jjt, kkr)] << ", "
    //               << grid.T_loc[I2V(iip-1, jjt, kkr)] << ", "
    //               << grid.T_loc[I2V(iip+1, jjt, kkr)] << ", "
    //               << grid.T_loc[I2V(iip, jjt-1, kkr)] << ", "
    //               << grid.T_loc[I2V(iip, jjt+1, kkr)] << ", "
    //               << grid.T_loc[I2V(iip, jjt, kkr-1)] << ", "
    //               << grid.T_loc[I2V(iip, jjt, kkr+1)] << std::endl;
    // }
}


void Iterator::calculate_stencil_1st_order_tele(Grid& grid, int& iip, int& jjt, int&kkr){
    sigr = SWEEPING_COEFF_TELE*std::sqrt(grid.fac_a_loc[I2V(iip, jjt, kkr)]);
    sigt = SWEEPING_COEFF_TELE*std::sqrt(grid.fac_b_loc[I2V(iip, jjt, kkr)]);
    sigp = SWEEPING_COEFF_TELE*std::sqrt(grid.fac_c_loc[I2V(iip, jjt, kkr)]);
    coe  = _1_CR/((sigr/dr)+(sigt/dt)+(sigp/dp));

    pp1 = (grid.T_loc[I2V(iip  , jjt  , kkr  )] - grid.T_loc[I2V(iip-1, jjt  , kkr  )])/dp;
    pp2 = (grid.T_loc[I2V(iip+1, jjt  , kkr  )] - grid.T_loc[I2V(iip  , jjt  , kkr  )])/dp;
    pt1 = (grid.T_loc[I2V(iip  , jjt  , kkr  )] - grid.T_loc[I2V(iip  , jjt-1, kkr  )])/dt;
    pt2 = (grid.T_loc[I2V(iip  , jjt+1, kkr  )] - grid.T_loc[I2V(iip  , jjt  , kkr  )])/dt;
    pr1 = (grid.T_loc[I2V(iip  , jjt  , kkr  )] - grid.T_loc[I2V(iip  , jjt  , kkr-1)])/dr;
    pr2 = (grid.T_loc[I2V(iip  , jjt  , kkr+1)] - grid.T_loc[I2V(iip  , jjt  , kkr  )])/dr;

    // LF Hamiltonian
    Htau = calc_LF_Hamiltonian_tele(grid, pp1, pp2, pt1, pt2, pr1, pr2, iip, jjt, kkr);

    grid.T_loc[I2V(iip, jjt, kkr)] += coe*(grid.fun_loc[I2V(iip, jjt, kkr)] - Htau) \
                                    + coe*(sigr*(pr2-pr1)/_2_CR + sigt*(pt2-pt1)/_2_CR + sigp*(pp2-pp1)/_2_CR);

}


void Iterator::calculate_stencil_3rd_order_tele(Grid& grid, int& iip, int& jjt, int&kkr){
    sigr = SWEEPING_COEFF_TELE*std::sqrt(grid.fac_a_loc[I2V(iip, jjt, kkr)]);
    sigt = SWEEPING_COEFF_TELE*std::sqrt(grid.fac_b_loc[I2V(iip, jjt, kkr)]);
    sigp = SWEEPING_COEFF_TELE*std::sqrt(grid.fac_c_loc[I2V(iip, jjt, kkr)]);
    coe  = _1_CR/((sigr/dr)+(sigt/dt)+(sigp/dp));

    // direction p
    if (iip == 1) {
        pp1 = (grid.T_loc[I2V(iip,  jjt,kkr)] \
             - grid.T_loc[I2V(iip-1,jjt,kkr)]) / dp;

        wp2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip+1,jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip+2,jjt,kkr)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip-1,jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip+1,jjt,kkr)]) )) );

        pp2 = (_1_CR - wp2) * (         grid.T_loc[I2V(iip+1,jjt,kkr)] \
                              -         grid.T_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp \
                   + wp2  * ( - _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                              + _4_CR * grid.T_loc[I2V(iip+1,jjt,kkr)] \
                              -         grid.T_loc[I2V(iip+2,jjt,kkr)] ) / _2_CR / dp;

    } else if (iip == np-2) {
        wp1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip-1,jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip-2,jjt,kkr)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip+1,jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip-1,jjt,kkr)]) )) );

        pp1 = (_1_CR - wp1) * (           grid.T_loc[I2V(iip+1,jjt,kkr)] \
                                -         grid.T_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp \
                     + wp1  * ( + _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                                - _4_CR * grid.T_loc[I2V(iip-1,jjt,kkr)] \
                                +         grid.T_loc[I2V(iip-2,jjt,kkr)] ) / _2_CR / dp;

        pp2 = (grid.T_loc[I2V(iip+1,jjt,kkr)] \
             - grid.T_loc[I2V(iip,  jjt,kkr)]) / dp;

    } else {
        wp1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip-1,jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip-2,jjt,kkr)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip+1,jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                             +    grid.T_loc[I2V(iip-1,jjt,kkr)]) )) );

        pp1 = (_1_CR - wp1) * (         grid.T_loc[I2V(iip+1,jjt,kkr)] \
                              -         grid.T_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp \
                   + wp1  * ( + _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                              - _4_CR * grid.T_loc[I2V(iip-1,jjt,kkr)] \
                              +         grid.T_loc[I2V(iip-2,jjt,kkr)] ) / _2_CR / dp;

        wp2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip+1,jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip+2,jjt,kkr)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip-1,jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip+1,jjt,kkr)]) )) );

        pp2 = (_1_CR - wp2) * (           grid.T_loc[I2V(iip+1,jjt,kkr)] \
                                -         grid.T_loc[I2V(iip-1,jjt,kkr)]) / _2_CR / dp \
                     + wp2  * ( - _3_CR * grid.T_loc[I2V(iip  ,jjt,kkr)] \
                                + _4_CR * grid.T_loc[I2V(iip+1,jjt,kkr)] \
                                -         grid.T_loc[I2V(iip+2,jjt,kkr)] ) / _2_CR / dp;

    }

    // direction t
    if (jjt == 1) {
        pt1 = (grid.T_loc[I2V(iip,jjt  ,kkr)] \
             - grid.T_loc[I2V(iip,jjt-1,kkr)]) / dt;

        wt2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt+1,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt+2,kkr)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip,jjt-1,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt+1,kkr)]) )) );

        pt2 = (_1_CR - wt2) * (         grid.T_loc[I2V(iip,jjt+1,kkr)] \
                              -         grid.T_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                   + wt2  * ( - _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                              + _4_CR * grid.T_loc[I2V(iip,jjt+1,kkr)] \
                              -         grid.T_loc[I2V(iip,jjt+2,kkr)] ) / _2_CR / dt;

    } else if (jjt == nt-2) {
        wt1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt-1,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt-2,kkr)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip,jjt+1,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt-1,kkr)]) )) );

        pt1 = (_1_CR - wt1) * (           grid.T_loc[I2V(iip,jjt+1,kkr)] \
                                -         grid.T_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                     + wt1  * ( + _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                                - _4_CR * grid.T_loc[I2V(iip,jjt-1,kkr)] \
                                +         grid.T_loc[I2V(iip,jjt-2,kkr)] ) / _2_CR / dt;

        pt2 = (grid.T_loc[I2V(iip,jjt+1,kkr)] \
             - grid.T_loc[I2V(iip,jjt,  kkr)]) / dt;

    } else {
        wt1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt-1,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt-2,kkr)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip,jjt+1,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt-1,kkr)]) )) );

        pt1 = (_1_CR - wt1) * (           grid.T_loc[I2V(iip,jjt+1,kkr)] \
                                -         grid.T_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                     + wt1  * ( + _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                                - _4_CR * grid.T_loc[I2V(iip,jjt-1,kkr)] \
                                +         grid.T_loc[I2V(iip,jjt-2,kkr)] ) / _2_CR / dt;

        wt2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt+1,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt+2,kkr)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip,jjt-1,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt+1,kkr)]) )) );

        pt2 = (_1_CR - wt2) * (           grid.T_loc[I2V(iip,jjt+1,kkr)] \
                                -         grid.T_loc[I2V(iip,jjt-1,kkr)]) / _2_CR / dt \
                     + wt2  * ( - _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                                + _4_CR * grid.T_loc[I2V(iip,jjt+1,kkr)] \
                                -         grid.T_loc[I2V(iip,jjt+2,kkr)] ) / _2_CR / dt;

    }

    // direction r
    if (kkr == 1) {
        pr1 = (grid.T_loc[I2V(iip,jjt,kkr  )] \
             - grid.T_loc[I2V(iip,jjt,kkr-1)]) / dr;

        wr2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                                           +      grid.T_loc[I2V(iip,jjt,kkr+2)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip,jjt,kkr-1)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt,kkr+1)]) )) );

        pr2 = (_1_CR - wr2) * (           grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.T_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                     + wr2  * ( - _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                                + _4_CR * grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.T_loc[I2V(iip,jjt,kkr+2)] ) / _2_CR / dr;

    } else if (kkr == nr - 2) {
        wr1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt,kkr-1)] \
                                                           +      grid.T_loc[I2V(iip,jjt,kkr-2)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt,kkr-1)]) )) );

        pr1 = (_1_CR - wr1) * (            grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                -          grid.T_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                      + wr1  * ( + _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                                 - _4_CR * grid.T_loc[I2V(iip,jjt,kkr-1)] \
                                 +         grid.T_loc[I2V(iip,jjt,kkr-2)] ) / _2_CR / dr;

        pr2 = (grid.T_loc[I2V(iip,jjt,kkr+1)] \
             - grid.T_loc[I2V(iip,jjt,kkr)]) / dr;

    } else {
        wr1 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt,kkr-1)] \
                                                           +      grid.T_loc[I2V(iip,jjt,kkr-2)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt,kkr-1)]) )) );

        pr1 = (_1_CR - wr1) * (           grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.T_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                     + wr1  * ( + _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                                - _4_CR * grid.T_loc[I2V(iip,jjt,kkr-1)] \
                                +         grid.T_loc[I2V(iip,jjt,kkr-2)] ) / _2_CR / dr;

        wr2 = _1_CR/(_1_CR+_2_CR*my_square((eps + my_square(      grid.T_loc[I2V(iip,jjt,kkr)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                                           +      grid.T_loc[I2V(iip,jjt,kkr+2)]) ) \

                                         / (eps + my_square(      grid.T_loc[I2V(iip,jjt,kkr-1)] \
                                                           -_2_CR*grid.T_loc[I2V(iip,  jjt,kkr)] \
                                                           +      grid.T_loc[I2V(iip,jjt,kkr+1)]) )) );

        pr2 = (_1_CR - wr2) * (           grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.T_loc[I2V(iip,jjt,kkr-1)]) / _2_CR / dr \
                     + wr2  * ( - _3_CR * grid.T_loc[I2V(iip,jjt,kkr)] \
                                + _4_CR * grid.T_loc[I2V(iip,jjt,kkr+1)] \
                                -         grid.T_loc[I2V(iip,jjt,kkr+2)] ) / _2_CR / dr;

    }

    // LF Hamiltonian
    Htau = calc_LF_Hamiltonian_tele(grid, pp1, pp2, pt1, pt2, pr1, pr2, iip, jjt, kkr);

    // update tau
    grid.T_loc[I2V(iip,jjt,kkr)] += coe * (grid.fun_loc[I2V(iip,jjt,kkr)] - Htau) \
        + coe * (sigr*(pr2-pr1)/_2_CR + sigt*(pt2-pt1)/_2_CR + sigp*(pp2-pp1)/_2_CR);

    // check value
    // if(kkr==1 && jjt==1 && iip==1){
    //     std::cout   << sigr << " " << sigt << " " << sigp << " " << coe << " "
    //                 << grid.fun_loc[I2V(iip,jjt,kkr)] << " " << Htau << " " << grid.T_loc[I2V(iip,jjt,kkr)] << " "
    //                 << std::endl;
    // }
}


inline CUSTOMREAL Iterator::calc_LF_Hamiltonian(Grid& grid, \
                                         CUSTOMREAL& pp1, CUSTOMREAL& pp2, \
                                         CUSTOMREAL& pt1, CUSTOMREAL& pt2, \
                                         CUSTOMREAL& pr1, CUSTOMREAL& pr2, \
                                         int& iip, int& jjt, int& kkr) {
    // LF Hamiltonian
    return sqrt(
              grid.fac_a_loc[I2V(iip,jjt,kkr)] * my_square(grid.T0r_loc[I2V(iip,jjt,kkr)] * grid.tau_loc[I2V(iip,jjt,kkr)] + grid.T0v_loc[I2V(iip,jjt,kkr)] * (pr1+pr2)/_2_CR) \
    +         grid.fac_b_loc[I2V(iip,jjt,kkr)] * my_square(grid.T0t_loc[I2V(iip,jjt,kkr)] * grid.tau_loc[I2V(iip,jjt,kkr)] + grid.T0v_loc[I2V(iip,jjt,kkr)] * (pt1+pt2)/_2_CR) \
    +         grid.fac_c_loc[I2V(iip,jjt,kkr)] * my_square(grid.T0p_loc[I2V(iip,jjt,kkr)] * grid.tau_loc[I2V(iip,jjt,kkr)] + grid.T0v_loc[I2V(iip,jjt,kkr)] * (pp1+pp2)/_2_CR) \
    -   _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * (grid.T0t_loc[I2V(iip,jjt,kkr)] * grid.tau_loc[I2V(iip,jjt,kkr)] + grid.T0v_loc[I2V(iip,jjt,kkr)] * (pt1+pt2)/_2_CR) \
                                               * (grid.T0p_loc[I2V(iip,jjt,kkr)] * grid.tau_loc[I2V(iip,jjt,kkr)] + grid.T0v_loc[I2V(iip,jjt,kkr)] * (pp1+pp2)/_2_CR) \
    );

}


inline CUSTOMREAL Iterator::calc_LF_Hamiltonian_tele(Grid& grid, \
                                         CUSTOMREAL& pp1, CUSTOMREAL& pp2, \
                                         CUSTOMREAL& pt1, CUSTOMREAL& pt2, \
                                         CUSTOMREAL& pr1, CUSTOMREAL& pr2, \
                                         int& iip, int& jjt, int& kkr) {
    // LF Hamiltonian for teleseismic source
    return std::sqrt(
              grid.fac_a_loc[I2V(iip,jjt,kkr)] * my_square((pr1+pr2)/_2_CR) \
    +         grid.fac_b_loc[I2V(iip,jjt,kkr)] * my_square((pt1+pt2)/_2_CR) \
    +         grid.fac_c_loc[I2V(iip,jjt,kkr)] * my_square((pp1+pp2)/_2_CR) \
    -   _2_CR*grid.fac_f_loc[I2V(iip,jjt,kkr)] * ((pt1+pt2)/_2_CR) \
                                               * ((pp1+pp2)/_2_CR) \
    );

}


void Iterator::calculate_boundary_nodes(Grid& grid){
    CUSTOMREAL v0, v1;

    //plane
    for (int jjt = 0; jjt < nt; jjt++){
        for (int iip = 0; iip < np; iip++){
            v0 = _2_CR * grid.tau_loc[I2V(iip,jjt,1)] - grid.tau_loc[I2V(iip,jjt,2)];
            v1 = grid.tau_loc[I2V(iip,jjt,2)];
            grid.tau_loc[I2V(iip,jjt,0)] = std::max({v0,v1});
            v0 = _2_CR * grid.tau_loc[I2V(iip,jjt,nr-2)] - grid.tau_loc[I2V(iip,jjt,nr-3)];
            v1 = grid.tau_loc[I2V(iip,jjt,nr-3)];
            grid.tau_loc[I2V(iip,jjt,nr-1)] = std::max({v0,v1});
        }
    }

    for (int kkr = 0; kkr < nr; kkr++){
        for (int iip = 0; iip < np; iip++){
            v0 = _2_CR * grid.tau_loc[I2V(iip,1,kkr)] - grid.tau_loc[I2V(iip,2,kkr)];
            v1 = grid.tau_loc[I2V(iip,2,kkr)];
            grid.tau_loc[I2V(iip,0,kkr)] = std::max({v0,v1});
            v0 = _2_CR * grid.tau_loc[I2V(iip,nt-2,kkr)] - grid.tau_loc[I2V(iip,nt-3,kkr)];
            v1 = grid.tau_loc[I2V(iip,nt-3,kkr)];
            grid.tau_loc[I2V(iip,nt-1,kkr)] = std::max({v0,v1});
        }
    }

    for (int kkr = 0; kkr < nr; kkr++){
        for (int jjt = 0; jjt < nt; jjt++){
            v0 = _2_CR * grid.tau_loc[I2V(1,jjt,kkr)] - grid.tau_loc[I2V(2,jjt,kkr)];
            v1 = grid.tau_loc[I2V(2,jjt,kkr)];
            grid.tau_loc[I2V(0,jjt,kkr)] = std::max({v0,v1});
            v0 = _2_CR * grid.tau_loc[I2V(np-2,jjt,kkr)] - grid.tau_loc[I2V(np-3,jjt,kkr)];
            v1 = grid.tau_loc[I2V(np-3,jjt,kkr)];
            grid.tau_loc[I2V(np-1,jjt,kkr)] = std::max({v0,v1});
        }
    }

}


void Iterator::calculate_boundary_nodes_tele(Grid& grid, int& iip, int& jjt, int& kkr){
    CUSTOMREAL v0, v1;

    if (grid.is_changed[I2V(iip,jjt,kkr)]) {

        // Bottom
        if (kkr == 0 && grid.k_first()){
            v0 = _2_CR * grid.T_loc[I2V(iip,jjt,1)] - grid.T_loc[I2V(iip,jjt,2)];
            v1 = grid.T_loc[I2V(iip,jjt,2)];
            grid.T_loc[I2V(iip,jjt,0)] = std::max({v0,v1});
        }

        // Top
        if (kkr == nr-1 && grid.k_last()){
            v0 = _2_CR * grid.T_loc[I2V(iip,jjt,nr-2)] - grid.T_loc[I2V(iip,jjt,nr-3)];
            v1 = grid.T_loc[I2V(iip,jjt,nr-3)];
            grid.T_loc[I2V(iip,jjt,nr-1)] = std::max({v0,v1});
        }

        // South
        if (jjt == 0 && grid.j_first()) {
            v0 = _2_CR * grid.T_loc[I2V(iip,1,kkr)] - grid.T_loc[I2V(iip,2,kkr)];
            v1 = grid.T_loc[I2V(iip,2,kkr)];
            grid.T_loc[I2V(iip,0,kkr)] = std::max({v0,v1});
        }

        // North
        if (jjt == nt-1 && grid.j_last()) {
            v0 = _2_CR * grid.T_loc[I2V(iip,nt-2,kkr)] - grid.T_loc[I2V(iip,nt-3,kkr)];
            v1 = grid.T_loc[I2V(iip,nt-3,kkr)];
            grid.T_loc[I2V(iip,nt-1,kkr)] = std::max({v0,v1});
        }

        // West
        if (iip == 0 && grid.i_first()) {
            v0 = _2_CR * grid.T_loc[I2V(1,jjt,kkr)] - grid.T_loc[I2V(2,jjt,kkr)];
            v1 = grid.T_loc[I2V(2,jjt,kkr)];
            grid.T_loc[I2V(0,jjt,kkr)] = std::max({v0,v1});
        }

        // East
        if (iip == np-1 && grid.i_last()){
            v0 = _2_CR * grid.T_loc[I2V(np-2,jjt,kkr)] - grid.T_loc[I2V(np-3,jjt,kkr)];
            v1 = grid.T_loc[I2V(np-3,jjt,kkr)];
            grid.T_loc[I2V(np-1,jjt,kkr)] = std::max({v0,v1});
        }

    } // end if is_changed
}


void Iterator::calculate_boundary_nodes_adj(Grid& grid, int& iip, int& jjt, int& kkr){
    // West
    if (iip == 0 && grid.i_first()) {
        grid.tau_loc[I2V(0,jjt,kkr)] = _0_CR;
    }

    // East
    if (iip == np-1 && grid.i_last()) {
        grid.tau_loc[I2V(np-1,jjt,kkr)] = _0_CR;
    }

    // South
    if (jjt == 0 && grid.j_first()) {
        grid.tau_loc[I2V(iip,0,kkr)] = _0_CR;
    }

    // North
    if (jjt == nt-1 && grid.j_last()) {
        grid.tau_loc[I2V(iip,nt-1,kkr)] = _0_CR;
    }

    // Bottom
    if (kkr == 0 && grid.k_first()) {
        grid.tau_loc[I2V(iip,jjt,0)] = _0_CR;
    }

    // Top
    if (kkr == nr-1 && grid.k_last()) {
        grid.tau_loc[I2V(iip,jjt,nr-1)] = _0_CR;
    }
}


void Iterator::set_sweep_direction(int iswp) {
    // set sweep direction
    if (iswp == 0) {
        r_dirc = -1;
        t_dirc = -1;
        p_dirc = -1;
    } else if (iswp == 1) {
        r_dirc = -1;
        t_dirc = -1;
        p_dirc =  1;
    } else if (iswp == 2) {
        r_dirc = -1;
        t_dirc =  1;
        p_dirc = -1;
    } else if (iswp == 3) {
        r_dirc = -1;
        t_dirc =  1;
        p_dirc =  1;
    } else if (iswp == 4) {
        r_dirc =  1;
        t_dirc = -1;
        p_dirc = -1;
    } else if (iswp == 5) {
        r_dirc =  1;
        t_dirc = -1;
        p_dirc =  1;
    } else if (iswp == 6) {
        r_dirc =  1;
        t_dirc =  1;
        p_dirc = -1;
    } else if (iswp == 7) {
        r_dirc =  1;
        t_dirc =  1;
        p_dirc =  1;
    }
}

