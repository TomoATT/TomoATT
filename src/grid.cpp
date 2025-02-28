#include "grid.h"

Grid::Grid(InputParams& IP, IO_utils& io) {
    stdout_by_main("--- grid object initialization starts. ---");

    // initialize grid parameters are done by only the main process of each subdomain
    if (subdom_main) {
        // domain decomposition
        init_decomposition(IP);

        // memory allocation
        memory_allocation();
    }

    // allocate memory for shm arrays if necessary
    if (n_subprocs > 1){
        shm_memory_allocation();
    }

    synchronize_all_world();

    // setup grid parameters
    if (subdom_main)
        setup_grid_params(IP, io);

}

Grid::~Grid() {
    // deallocate memory from all arrays
    if (subdom_main) memory_deallocation();

    if (n_subprocs > 1) {
        // deallocate memory from shm arrays
        shm_memory_deallocation();
    }
}

void Grid::init_decomposition(InputParams& IP) {

    int tmp_rank = 0;

    for (int kk = 0; kk < ndiv_k; kk++)
        for (int jj = 0; jj < ndiv_j; jj++)
            for (int ii = 0; ii < ndiv_i; ii++) {
                tmp_rank = kk * ndiv_i * ndiv_j + jj * ndiv_i + ii;
                if (tmp_rank == myrank) {
                    // check neighbors on i axis
                    if (ndiv_i > 1) {
                        if (ii == 0) {                 // neighbor on the bound +i
                            neighbors_id[1] = tmp_rank + 1;
                        } else if (ii == ndiv_i - 1) { // neighbor on the bound -i
                            neighbors_id[0] = tmp_rank - 1;
                        } else {                       // middle layers
                            neighbors_id[0] = tmp_rank - 1;
                            neighbors_id[1] = tmp_rank + 1;
                        }
                    }

                    // check neighbors on j axis
                    if (ndiv_j > 1) {
                        if (jj == 0) {                 // neighbor on the bound +j
                            neighbors_id[3] = tmp_rank + ndiv_i;
                        } else if (jj == ndiv_j - 1) { // neighbor on the bound -j
                            neighbors_id[2] = tmp_rank - ndiv_i;
                        } else {                       // middle layers
                            neighbors_id[2] = tmp_rank - ndiv_i;
                            neighbors_id[3] = tmp_rank + ndiv_i;
                        }
                    }

                    // check neighbors on k axis
                    if (ndiv_k > 1) {
                        if (kk == 0) {                 // neighbor on the bound +k
                            neighbors_id[5] = tmp_rank + ndiv_i * ndiv_j;
                        } else if (kk == ndiv_k - 1) { // neighbor on the bound -k
                            neighbors_id[4] = tmp_rank - ndiv_i * ndiv_j;
                        } else {                       // middle layers
                            neighbors_id[4] = tmp_rank - ndiv_i * ndiv_j;
                            neighbors_id[5] = tmp_rank + ndiv_i * ndiv_j;
                        }
                    }

                    // store the layer id
                    domain_i = ii;
                    domain_j = jj;
                    domain_k = kk;

                    // stop searching immediately if this rank found the local ids
                    goto end_pos_detected;
                }
            }

    end_pos_detected:

    // calculate the number of grid points for this subdomain
    loc_I = (int)ngrid_i/ndiv_i;
    loc_J = (int)ngrid_j/ndiv_j;
    loc_K = (int)ngrid_k/ndiv_k;

    // calculate the number of grid points for this subdomain
    offset_i = domain_i * loc_I;
    offset_j = domain_j * loc_J;
    offset_k = domain_k * loc_K;

    // add modulus to the last rank
    if (i_last()) loc_I += ngrid_i % ndiv_i;
    if (j_last()) loc_J += ngrid_j % ndiv_j;
    if (k_last()) loc_K += ngrid_k % ndiv_k;

    // number of grid points on the boundary surfaces
    n_grid_bound_i = loc_J * loc_K;
    n_grid_bound_j = loc_I * loc_K;
    n_grid_bound_k = loc_I * loc_J;

    // store the number of grids excluding the ghost grids
    loc_I_excl_ghost = loc_I;
    loc_J_excl_ghost = loc_J;
    loc_K_excl_ghost = loc_K;

    // initialize iteration start/end
    i_start_loc = 0;
    j_start_loc = 0;
    k_start_loc = 0;
    i_end_loc   = loc_I-1; // end id needs to be included for the iterations
    j_end_loc   = loc_J-1; // thus please pay attaintion to the for loop
    k_end_loc   = loc_K-1; // to be e.g. for (i = i_start_loc; i <= i_end_loc; i++)

    // add ghost nodes to comunicate with neighbor domains
    // i axis
    if (ndiv_i > 1) {
        if (i_first() || i_last()) {
            loc_I += n_ghost_layers;
        } else {
            loc_I += n_ghost_layers * 2;
        }
        // store the start/end ids of iteration for axis i
        if (!i_first()) i_start_loc = n_ghost_layers;
        if (!i_last())  i_end_loc   = loc_I - n_ghost_layers - 1;
        if (i_last())   i_end_loc   = loc_I - 1;
    }

    // j axis
    if (ndiv_j > 1) {
        if (j_first() || j_last()) {
            loc_J += n_ghost_layers;
        } else {
            loc_J += n_ghost_layers * 2;
        }
        // store the start/end ids of iteration for axis j
        if (!j_first()) j_start_loc = n_ghost_layers;
        if (!j_last())  j_end_loc   = loc_J - n_ghost_layers - 1;
        if (j_last())   j_end_loc   = loc_J - 1;
    }
    // k axis
    if (ndiv_k > 1) {
        if (k_first() || k_last()) {
            loc_K += n_ghost_layers;
        } else {
            loc_K += n_ghost_layers * 2;
        }
        // store the start/end ids of iteration for axis k
        if (!k_first()) k_start_loc = n_ghost_layers;
        if (!k_last())  k_end_loc   = loc_K - n_ghost_layers - 1;
        if (k_last())   k_end_loc   = loc_K - 1;
    }

    // calculate the nodes' and elms' offset for data output
    loc_nnodes =  loc_I_excl_ghost      *  loc_J_excl_ghost      *  loc_K_excl_ghost;
    loc_nelms  = (loc_I_excl_ghost - 1) * (loc_J_excl_ghost - 1) * (loc_K_excl_ghost - 1);

    nnodes = allocateMemory<int>(nprocs, 1001);
    nelms  = allocateMemory<int>(nprocs, 1002);

    // gather the number of nodes and elements
    allgather_i_single(&loc_nnodes, nnodes);
    allgather_i_single(&loc_nelms, nelms);

    if (myrank > 0){
        // sum nnodes and nelms
        for (int i = 0; i < myrank; i++) {
            offset_nnodes += nnodes[i];
            offset_nelms  += nelms[i];
        }
    }

    // setup number of elements and nodes for visualization (additinoal 1 layer for connection)
    loc_I_vis = loc_I_excl_ghost;
    loc_J_vis = loc_J_excl_ghost;
    loc_K_vis = loc_K_excl_ghost;
    if (!i_last()) loc_I_vis++;
    if (!j_last()) loc_J_vis++;
    if (!k_last()) loc_K_vis++;

    loc_nnodes_vis = loc_I_vis * loc_J_vis * loc_K_vis;
    loc_nelms_vis  = (loc_I_vis - 1) * (loc_J_vis - 1) * (loc_K_vis - 1);
    nnodes_vis = allocateMemory<int>(nprocs, 1003);
    nelms_vis  = allocateMemory<int>(nprocs, 1004);
    allgather_i_single(&loc_nnodes_vis, nnodes_vis);
    allgather_i_single(&loc_nelms_vis, nelms_vis);
    if (myrank > 0){
        // sum nnodes and nelms
        for (int i = 0; i < myrank; i++) {
            offset_nnodes_vis += nnodes_vis[i];
            offset_nelms_vis  += nelms_vis[i];
        }
    }
    i_start_vis = i_start_loc;
    j_start_vis = j_start_loc;
    k_start_vis = k_start_loc;
    i_end_vis = i_end_loc;
    j_end_vis = j_end_loc;
    k_end_vis = k_end_loc;
    // add one layer for connection to the next subdomain
    if (!i_last()) i_end_vis++;
    if (!j_last()) j_end_vis++;
    if (!k_last()) k_end_vis++;

    // inversion setup
    // check if inversion grids are needed
    if (IP.get_run_mode()==DO_INVERSION || IP.get_run_mode()==INV_RELOC || IP.get_run_mode()==ONED_INVERSION){
        inverse_flag = true;
        inv_grid     = new InvGrid(IP);
    } else {
        inverse_flag = false;
    }

    // check subdomains contacting lines and points
    // ij nn
    if (neighbors_id[0] != -1 && neighbors_id[2] != -1)
        neighbors_id_ij[0] = neighbors_id[2]-1;
    // ij np
    if (neighbors_id[0] != -1 && neighbors_id[3] != -1)
        neighbors_id_ij[1] = neighbors_id[3]-1;
    // ij pn
    if (neighbors_id[1] != -1 && neighbors_id[2] != -1)
        neighbors_id_ij[2] = neighbors_id[2]+1;
    // ij pp
    if (neighbors_id[1] != -1 && neighbors_id[3] != -1)
        neighbors_id_ij[3] = neighbors_id[3]+1;
    // jk nn
    if (neighbors_id[2] != -1 && neighbors_id[4] != -1)
        neighbors_id_jk[0] = neighbors_id[4]-ndiv_i;
    // jk np
    if (neighbors_id[2] != -1 && neighbors_id[5] != -1)
        neighbors_id_jk[1] = neighbors_id[5]-ndiv_i;
    // jk pn
    if (neighbors_id[3] != -1 && neighbors_id[4] != -1)
        neighbors_id_jk[2] = neighbors_id[4]+ndiv_i;
    // jk pp
    if (neighbors_id[3] != -1 && neighbors_id[5] != -1)
        neighbors_id_jk[3] = neighbors_id[5]+ndiv_i;
    // ik nn
    if (neighbors_id[0] != -1 && neighbors_id[4] != -1)
        neighbors_id_ik[0] = neighbors_id[4]-1;
    // ik np
    if (neighbors_id[0] != -1 && neighbors_id[5] != -1)
        neighbors_id_ik[1] = neighbors_id[5]-1;
    // ik pn
    if (neighbors_id[1] != -1 && neighbors_id[4] != -1)
        neighbors_id_ik[2] = neighbors_id[4]+1;
    // ik pp
    if (neighbors_id[1] != -1 && neighbors_id[5] != -1)
        neighbors_id_ik[3] = neighbors_id[5]+1;
    // ijk nnn
    if (neighbors_id[0] != -1 && neighbors_id[2] != -1 && neighbors_id[4] != -1)
        neighbors_id_ijk[0] = neighbors_id[4]-ndiv_i;
    // ijk nnp
    if (neighbors_id[0] != -1 && neighbors_id[2] != -1 && neighbors_id[5] != -1)
        neighbors_id_ijk[1] = neighbors_id[5]-ndiv_i;
    // ijk npn
    if (neighbors_id[0] != -1 && neighbors_id[3] != -1 && neighbors_id[4] != -1)
        neighbors_id_ijk[2] = neighbors_id[4]+ndiv_i;
    // ijk npp
    if (neighbors_id[0] != -1 && neighbors_id[3] != -1 && neighbors_id[5] != -1)
        neighbors_id_ijk[3] = neighbors_id[5]+ndiv_i;
    // ijk pnn
    if (neighbors_id[1] != -1 && neighbors_id[2] != -1 && neighbors_id[4] != -1)
        neighbors_id_ijk[4] = neighbors_id[4]-ndiv_i;
    // ijk pnp
    if (neighbors_id[1] != -1 && neighbors_id[2] != -1 && neighbors_id[5] != -1)
        neighbors_id_ijk[5] = neighbors_id[5]-ndiv_i;
    // ijk ppn
    if (neighbors_id[1] != -1 && neighbors_id[3] != -1 && neighbors_id[4] != -1)
        neighbors_id_ijk[6] = neighbors_id[4]+ndiv_i;
    // ijk ppp
    if (neighbors_id[1] != -1 && neighbors_id[3] != -1 && neighbors_id[5] != -1)
        neighbors_id_ijk[7] = neighbors_id[5]+ndiv_i;

    // debug output
    stdout_by_main("domain decomposition initialization end.");


    for (int i = 0; i < nprocs; i++) {
        synchronize_all_inter();

        if (id_sim == 0){
            if (i == myrank) {
                std::cout << "--- myrank: ---" << myrank << std::endl;

                if (i == 0) {
                    std::cout << "ndiv_i j k : "  << ndiv_i  << " " << ndiv_j  << " " << ndiv_k  << std::endl;
                    std::cout << "ngrid_i j k : " << ngrid_i << " " << ngrid_j << " " << ngrid_k << std::endl;
                }

                std::cout << "domain_i j k: "          << domain_i << " " << domain_j << " " << domain_k << std::endl;
                std::cout << "loc_I J K: "             << loc_I << " " << loc_J << " " << loc_K << std::endl;
                std::cout << "loc_I_excl_ghost J_excl_ghost K_excl_ghost: " << loc_I_excl_ghost << " " << loc_J_excl_ghost << " " << loc_K_excl_ghost << std::endl;
                std::cout << "i_start_loc i_end_loc: " << i_start_loc << " " << i_end_loc << std::endl;
                std::cout << "j_start_loc j_end_loc: " << j_start_loc << " " << j_end_loc << std::endl;
                std::cout << "k_start_loc k_end_loc: " << k_start_loc << " " << k_end_loc << std::endl;
                std::cout << "offset_nnodes offset_nelms: " << offset_nnodes << " " << offset_nelms << std::endl;
                std::cout << "n total local grids: "   << loc_I*loc_J*loc_K << "   max vector elm id:  " << I2V(loc_I-1,loc_J-1,loc_K-1) << std::endl;
                // print neighbors_id
                std::cout << "neighbors_id: ";
                for (int i = 0; i < 6; i++) {
                    std::cout << neighbors_id[i] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    synchronize_all_inter();
}



// allocate memory for arrays, called only for subdom_main
void Grid::memory_allocation() {
    // 3d arrays
    int n_total_loc_grid_points = loc_I * loc_J * loc_K;
    // skip allocation of shm arrays if not using shm
    if (sub_nprocs <= 1) {
#ifdef USE_CUDA
        if(use_gpu)
            cudaMallocHost((void**)&tau_loc, n_total_loc_grid_points * sizeof(CUSTOMREAL));
        else
            tau_loc = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 1);
#else
        tau_loc     = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 1);
#endif

        xi_loc      = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 2);
        eta_loc     = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 3);
        zeta_loc    = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 4);
        T_loc       = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 5);
        tau_old_loc = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 6);

        T0r_loc     = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 7);
        T0t_loc     = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 8);
        T0p_loc     = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 9);
        T0v_loc     = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 10);
        fac_a_loc   = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 11);
        fac_b_loc   = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 12);
        fac_c_loc   = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 13);
        fac_f_loc   = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 14);
        fun_loc     = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 15);
        is_changed  = allocateMemory<bool>(n_total_loc_grid_points, 16);

        // 1d arrays
        p_loc_1d    = allocateMemory<CUSTOMREAL>(loc_I, 17);
        t_loc_1d    = allocateMemory<CUSTOMREAL>(loc_J, 18);
        r_loc_1d    = allocateMemory<CUSTOMREAL>(loc_K, 19);
    }

    if (if_test)
        u_loc = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 20);

    // ghost layer arrays
    if (neighbors_id[0] != -1) {
        bin_s = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_i, 21);
        bin_r = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_i, 22);
    }
    if (neighbors_id[1] != -1) {
        bip_s = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_i, 23);
        bip_r = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_i, 24);
    }
    if (neighbors_id[2] != -1) {
        bjn_s = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_j, 25);
        bjn_r = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_j, 26);
    }
    if (neighbors_id[3] != -1) {
        bjp_s = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_j, 27);
        bjp_r = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_j, 28);
    }
    if (neighbors_id[4] != -1) {
        bkn_s = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_k, 29);
        bkn_r = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_k, 30);
    }
    if (neighbors_id[5] != -1) {
        bkp_s = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_k, 31);
        bkp_r = allocateMemory<CUSTOMREAL>(n_ghost_layers*n_grid_bound_k, 32);
    }
    if (neighbors_id_ij[0] != -1) {
        bij_nn_s = allocateMemory<CUSTOMREAL>(loc_K_vis, 33);
        bij_nn_r = allocateMemory<CUSTOMREAL>(loc_K_vis, 34);
    }
    if (neighbors_id_ij[1] != -1) {
        bij_np_s = allocateMemory<CUSTOMREAL>(loc_K_vis, 35);
        bij_np_r = allocateMemory<CUSTOMREAL>(loc_K_vis, 36);
    }
    if (neighbors_id_ij[2] != -1) {
        bij_pn_s = allocateMemory<CUSTOMREAL>(loc_K_vis, 37);
        bij_pn_r = allocateMemory<CUSTOMREAL>(loc_K_vis, 38);
    }
    if (neighbors_id_ij[3] != -1) {
        bij_pp_s = allocateMemory<CUSTOMREAL>(loc_K_vis, 39);
        bij_pp_r = allocateMemory<CUSTOMREAL>(loc_K_vis, 40);
    }
    if (neighbors_id_jk[0] != -1) {
        bjk_nn_s = allocateMemory<CUSTOMREAL>(loc_I_vis, 41);
        bjk_nn_r = allocateMemory<CUSTOMREAL>(loc_I_vis, 42);
    }
    if (neighbors_id_jk[1] != -1) {
        bjk_np_s = allocateMemory<CUSTOMREAL>(loc_I_vis, 43);
        bjk_np_r = allocateMemory<CUSTOMREAL>(loc_I_vis, 44);
    }
    if (neighbors_id_jk[2] != -1) {
        bjk_pn_s = allocateMemory<CUSTOMREAL>(loc_I_vis, 45);
        bjk_pn_r = allocateMemory<CUSTOMREAL>(loc_I_vis, 46);
    }
    if (neighbors_id_jk[3] != -1) {
        bjk_pp_s = allocateMemory<CUSTOMREAL>(loc_I_vis, 47);
        bjk_pp_r = allocateMemory<CUSTOMREAL>(loc_I_vis, 48);
    }
    if (neighbors_id_ik[0] != -1) {
        bik_nn_s = allocateMemory<CUSTOMREAL>(loc_J_vis, 49);
        bik_nn_r = allocateMemory<CUSTOMREAL>(loc_J_vis, 50);
    }
    if (neighbors_id_ik[1] != -1) {
        bik_np_s = allocateMemory<CUSTOMREAL>(loc_J_vis, 51);
        bik_np_r = allocateMemory<CUSTOMREAL>(loc_J_vis, 52);
    }
    if (neighbors_id_ik[2] != -1) {
        bik_pn_s = allocateMemory<CUSTOMREAL>(loc_J_vis, 53);
        bik_pn_r = allocateMemory<CUSTOMREAL>(loc_J_vis, 54);
    }
    if (neighbors_id_ik[3] != -1) {
        bik_pp_s = allocateMemory<CUSTOMREAL>(loc_J_vis, 55);
        bik_pp_r = allocateMemory<CUSTOMREAL>(loc_J_vis, 56);
    }
    if (neighbors_id_ijk[0] != -1) {
        bijk_nnn_s = allocateMemory<CUSTOMREAL>(1, 57);
        bijk_nnn_r = allocateMemory<CUSTOMREAL>(1, 58);
    }
    if (neighbors_id_ijk[1] != -1) {
        bijk_nnp_s = allocateMemory<CUSTOMREAL>(1, 59);
        bijk_nnp_r = allocateMemory<CUSTOMREAL>(1, 60);
    }
    if (neighbors_id_ijk[2] != -1) {
        bijk_npn_s = allocateMemory<CUSTOMREAL>(1, 61);
        bijk_npn_r = allocateMemory<CUSTOMREAL>(1, 62);
    }
    if (neighbors_id_ijk[3] != -1) {
        bijk_npp_s = allocateMemory<CUSTOMREAL>(1, 63);
        bijk_npp_r = allocateMemory<CUSTOMREAL>(1, 64);
    }
    if (neighbors_id_ijk[4] != -1) {
        bijk_pnn_s = allocateMemory<CUSTOMREAL>(1, 65);
        bijk_pnn_r = allocateMemory<CUSTOMREAL>(1, 66);
    }
    if (neighbors_id_ijk[5] != -1) {
        bijk_pnp_s = allocateMemory<CUSTOMREAL>(1, 67);
        bijk_pnp_r = allocateMemory<CUSTOMREAL>(1, 68);
    }
    if (neighbors_id_ijk[6] != -1) {
        bijk_ppn_s = allocateMemory<CUSTOMREAL>(1, 69);
        bijk_ppn_r = allocateMemory<CUSTOMREAL>(1, 70);
    }
    if (neighbors_id_ijk[7] != -1) {
        bijk_ppp_s = allocateMemory<CUSTOMREAL>(1, 71);
        bijk_ppp_r = allocateMemory<CUSTOMREAL>(1, 72);
    }

    // array for mpi request
    mpi_send_reqs = allocateMemory<MPI_Request>(6, 731);
    mpi_recv_reqs = allocateMemory<MPI_Request>(6, 732);
    for (int i = 0; i < 6; i++){
        mpi_send_reqs[i] = MPI_REQUEST_NULL;
        mpi_recv_reqs[i] = MPI_REQUEST_NULL;
    }

    mpi_send_reqs_kosumi = allocateMemory<MPI_Request>(20, 741);
    mpi_recv_reqs_kosumi = allocateMemory<MPI_Request>(20, 742);
    for (int i = 0; i < 20; i++){
        mpi_send_reqs_kosumi[i] = MPI_REQUEST_NULL;
        mpi_recv_reqs_kosumi[i] = MPI_REQUEST_NULL;
    }

    // arrays for data output
    if (subdom_main){   // only accessible by the main process of each subdomain, which store the field data
        int nnodes_loc_vis =  loc_I_vis    *  loc_J_vis    *  loc_K_vis;
        int nelms_loc_vis  = (loc_I_vis-1) * (loc_J_vis-1) * (loc_K_vis-1);
        x_loc_3d     = allocateMemory<CUSTOMREAL>(nnodes_loc_vis, 75);
        y_loc_3d     = allocateMemory<CUSTOMREAL>(nnodes_loc_vis, 76);
        z_loc_3d     = allocateMemory<CUSTOMREAL>(nnodes_loc_vis, 77);
        p_loc_3d     = allocateMemory<CUSTOMREAL>(nnodes_loc_vis, 78);
        t_loc_3d     = allocateMemory<CUSTOMREAL>(nnodes_loc_vis, 79);
        r_loc_3d     = allocateMemory<CUSTOMREAL>(nnodes_loc_vis, 80);
        elms_conn    = allocateMemory<int>(nelms_loc_vis*9, 81);
        my_proc_dump = allocateMemory<int>(nnodes_loc_vis, 82); // DEBUG
        vis_data     = allocateMemory<CUSTOMREAL>(nnodes_loc_vis, 83); // temp array for visuzulization
    }

    // arrays for inversion
    if (inverse_flag) {
        int n_total_loc_inv_grid = n_inv_I_loc * n_inv_J_loc * n_inv_K_loc;
        int n_total_loc_inv_grid_ani = n_inv_I_loc_ani * n_inv_J_loc_ani * n_inv_K_loc_ani;

        if (subdom_main){   // only accessible by the main process of each subdomain, which store the field data
            Ks_loc                   = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 90);
            Kxi_loc                  = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 91);
            Keta_loc                 = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 92);
            Ks_density_loc           = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 93);
            Kxi_density_loc          = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 94);
            Keta_density_loc         = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 95);
            Ks_inv_loc               = allocateMemory<CUSTOMREAL>(n_total_loc_inv_grid, 96);
            Kxi_inv_loc              = allocateMemory<CUSTOMREAL>(n_total_loc_inv_grid_ani, 97);
            Keta_inv_loc             = allocateMemory<CUSTOMREAL>(n_total_loc_inv_grid_ani, 98);

            Ks_update_loc            = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 99);
            Kxi_update_loc           = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 100);
            Keta_update_loc          = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 101);

            Ks_update_loc_previous   = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 102);
            Kxi_update_loc_previous  = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 103);
            Keta_update_loc_previous = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 104);
        }

        if (sub_nprocs <= 1){
            Tadj_loc = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 105);
            Tadj_density_loc = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 106);
        }

        if (optim_method==HALVE_STEPPING_MODE) {
            fac_b_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 107);
            fac_c_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 108);
            fac_f_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 109);
            xi_loc_back  = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 110);
            eta_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 111);
            fun_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 112);
        }

        if (optim_method==LBFGS_MODE) {
            fac_b_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 113);
            fac_c_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 114);
            fac_f_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 115);
            xi_loc_back  = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 116);
            eta_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 117);
            fun_loc_back = allocateMemory<CUSTOMREAL>(n_total_loc_grid_points, 118);

            int n_total_loc_lbfgs = n_total_loc_grid_points * Mbfgs;
            Ks_descent_dir_loc   = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 119);
            Kxi_descent_dir_loc  = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 120);
            Keta_descent_dir_loc = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 121);
            // initialize
            std::fill(Ks_descent_dir_loc,   Ks_descent_dir_loc   + n_total_loc_lbfgs, _0_CR);
            std::fill(Kxi_descent_dir_loc,  Kxi_descent_dir_loc  + n_total_loc_lbfgs, _0_CR);
            std::fill(Keta_descent_dir_loc, Keta_descent_dir_loc + n_total_loc_lbfgs, _0_CR);

            if (id_sim==0){
                Ks_grad_store_loc    = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 122);
                Kxi_grad_store_loc   = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 123);
                Keta_grad_store_loc  = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 124);
                Ks_model_store_loc   = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 125);
                Kxi_model_store_loc  = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 126);
                Keta_model_store_loc = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 127);
                fun_gradient_regularization_penalty_loc = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 128);
                xi_gradient_regularization_penalty_loc  = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 129);
                eta_gradient_regularization_penalty_loc = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 130);
                fun_regularization_penalty_loc          = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 131);
                xi_regularization_penalty_loc           = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 132);
                eta_regularization_penalty_loc          = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 133);
                fun_prior_loc                           = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 134);
                xi_prior_loc                            = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 135);
                eta_prior_loc                           = allocateMemory<CUSTOMREAL>(n_total_loc_lbfgs, 136);

                // initialize
                std::fill(Ks_grad_store_loc,                       Ks_grad_store_loc                       + n_total_loc_lbfgs, _0_CR);
                std::fill(Kxi_grad_store_loc,                      Kxi_grad_store_loc                      + n_total_loc_lbfgs, _0_CR);
                std::fill(Keta_grad_store_loc,                     Keta_grad_store_loc                     + n_total_loc_lbfgs, _0_CR);
                std::fill(Ks_model_store_loc,                      Ks_model_store_loc                      + n_total_loc_lbfgs, _0_CR);
                std::fill(Kxi_model_store_loc,                     Kxi_model_store_loc                     + n_total_loc_lbfgs, _0_CR);
                std::fill(Keta_model_store_loc,                    Keta_model_store_loc                    + n_total_loc_lbfgs, _0_CR);
                std::fill(fun_gradient_regularization_penalty_loc, fun_gradient_regularization_penalty_loc + n_total_loc_lbfgs, _0_CR);
                std::fill(xi_gradient_regularization_penalty_loc,  xi_gradient_regularization_penalty_loc  + n_total_loc_lbfgs, _0_CR);
                std::fill(eta_gradient_regularization_penalty_loc, eta_gradient_regularization_penalty_loc + n_total_loc_lbfgs, _0_CR);
                std::fill(fun_regularization_penalty_loc,          fun_regularization_penalty_loc          + n_total_loc_lbfgs, _0_CR);
                std::fill(xi_regularization_penalty_loc,           xi_regularization_penalty_loc           + n_total_loc_lbfgs, _0_CR);
                std::fill(eta_regularization_penalty_loc,          eta_regularization_penalty_loc          + n_total_loc_lbfgs, _0_CR);
                std::fill(fun_prior_loc,                           fun_prior_loc                           + n_total_loc_lbfgs, _0_CR);
                std::fill(xi_prior_loc,                            xi_prior_loc                            + n_total_loc_lbfgs, _0_CR);
                std::fill(eta_prior_loc,                           eta_prior_loc                           + n_total_loc_lbfgs, _0_CR);
            }
        }
    } // end of if inverse_flag

    stdout_by_main("Memory allocation done.");
}


void Grid::shm_memory_allocation() {

    synchronize_all_sub();

    int n_total_loc_grid_points = loc_I * loc_J * loc_K;
    if (sub_rank != 0)
        n_total_loc_grid_points = 0;

    prepare_shm_array_cr(n_total_loc_grid_points, T_loc, win_T_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, tau_old_loc, win_tau_old_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, xi_loc, win_xi_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, eta_loc, win_eta_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, zeta_loc, win_zeta_loc);

    prepare_shm_array_cr(n_total_loc_grid_points, T0r_loc, win_T0r_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, T0t_loc, win_T0t_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, T0p_loc, win_T0p_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, T0v_loc, win_T0v_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, tau_loc, win_tau_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, fac_a_loc, win_fac_a_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, fac_b_loc, win_fac_b_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, fac_c_loc, win_fac_c_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, fac_f_loc, win_fac_f_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, fun_loc, win_fun_loc);
    prepare_shm_array_bool(n_total_loc_grid_points, is_changed, win_is_changed);

    prepare_shm_array_cr(loc_I, p_loc_1d, win_p_loc_1d);
    prepare_shm_array_cr(loc_J, t_loc_1d, win_t_loc_1d);
    prepare_shm_array_cr(loc_K, r_loc_1d, win_r_loc_1d);

    // inversion
    prepare_shm_array_cr(n_total_loc_grid_points, Tadj_loc, win_Tadj_loc);
    prepare_shm_array_cr(n_total_loc_grid_points, Tadj_density_loc, win_Tadj_density_loc);
}


void Grid::shm_memory_deallocation() {
    // add if necessary
}


// function for memory allocation, called only for subdomain.
void Grid::memory_deallocation() {
   // shm array deallocation is done in shm_memory_deallocation() if shm is used
    if (sub_nprocs <= 1) {

#ifdef USE_CUDA
        if(use_gpu)
            cudaFree(tau_loc);
        else
            delete[] tau_loc;
#else
        delete[] tau_loc;
#endif
        delete[] xi_loc;
        delete[] eta_loc;
        delete[] zeta_loc;
        delete[] T_loc;
        delete[] tau_old_loc;

        delete[] T0r_loc;
        delete[] T0t_loc;
        delete[] T0p_loc;
        delete[] T0v_loc;
        delete[] fac_a_loc;
        delete[] fac_b_loc;
        delete[] fac_c_loc;
        delete[] fac_f_loc;
        delete[] fun_loc;
        delete[] is_changed;

        delete[] t_loc_1d;
        delete[] p_loc_1d;
        delete[] r_loc_1d;

    }

    if(if_test)
        delete[] u_loc;

    if (neighbors_id[0] != -1) {
        delete[] bin_s;
        delete[] bin_r;
    }
    if (neighbors_id[1] != -1) {
        delete[] bip_s;
        delete[] bip_r;
    }
    if (neighbors_id[2] != -1) {
        delete[] bjn_s;
        delete[] bjn_r;
    }
    if (neighbors_id[3] != -1) {
        delete[] bjp_s;
        delete[] bjp_r;
    }
    if (neighbors_id[4] != -1) {
        delete[] bkn_s;
        delete[] bkn_r;
    }
    if (neighbors_id[5] != -1) {
        delete[] bkp_s;
        delete[] bkp_r;
    }
    if (neighbors_id_ij[0] != -1) {
        delete[] bij_nn_s;
        delete[] bij_nn_r;
    }
    if (neighbors_id_ij[1] != -1) {
        delete[] bij_np_s;
        delete[] bij_np_r;
    }
    if (neighbors_id_ij[2] != -1) {
        delete[] bij_pn_s;
        delete[] bij_pn_r;
    }
    if (neighbors_id_ij[3] != -1) {
        delete[] bij_pp_s;
        delete[] bij_pp_r;
    }
    if (neighbors_id_jk[0] != -1) {
        delete[] bjk_nn_s;
        delete[] bjk_nn_r;
    }
    if (neighbors_id_jk[1] != -1) {
        delete[] bjk_np_s;
        delete[] bjk_np_r;
    }
    if (neighbors_id_jk[2] != -1) {
        delete[] bjk_pn_s;
        delete[] bjk_pn_r;
    }
    if (neighbors_id_jk[3] != -1) {
        delete[] bjk_pp_s;
        delete[] bjk_pp_r;
    }
    if (neighbors_id_ik[0] != -1) {
        delete[] bik_nn_s;
        delete[] bik_nn_r;
    }
    if (neighbors_id_ik[1] != -1) {
        delete[] bik_np_s;
        delete[] bik_np_r;
    }
    if (neighbors_id_ik[2] != -1) {
        delete[] bik_pn_s;
        delete[] bik_pn_r;
    }
    if (neighbors_id_ik[3] != -1) {
        delete[] bik_pp_s;
        delete[] bik_pp_r;
    }
    if (neighbors_id_ijk[0] != -1) {
        delete[] bijk_nnn_s;
        delete[] bijk_nnn_r;
    }
    if (neighbors_id_ijk[1] != -1) {
        delete[] bijk_nnp_s;
        delete[] bijk_nnp_r;
    }
    if (neighbors_id_ijk[2] != -1) {
        delete[] bijk_npn_s;
        delete[] bijk_npn_r;
    }
    if (neighbors_id_ijk[3] != -1) {
        delete[] bijk_npp_s;
        delete[] bijk_npp_r;
    }
    if (neighbors_id_ijk[4] != -1) {
        delete[] bijk_pnn_s;
        delete[] bijk_pnn_r;
    }
    if (neighbors_id_ijk[5] != -1) {
        delete[] bijk_pnp_s;
        delete[] bijk_pnp_r;
    }
    if (neighbors_id_ijk[6] != -1) {
        delete[] bijk_ppn_s;
        delete[] bijk_ppn_r;
    }
    if (neighbors_id_ijk[7] != -1) {
        delete[] bijk_ppp_s;
        delete[] bijk_ppp_r;
    }

    delete[] mpi_send_reqs;
    delete[] mpi_recv_reqs;
    delete[] mpi_send_reqs_kosumi;
    delete[] mpi_recv_reqs_kosumi;

    // delete arrays
    delete[] nnodes;
    delete[] nelms;
    delete[] nnodes_vis;
    delete[] nelms_vis;

    if (subdom_main){
        delete[] x_loc_3d;
        delete[] y_loc_3d;
        delete[] z_loc_3d;
        delete[] p_loc_3d;
        delete[] t_loc_3d;
        delete[] r_loc_3d;
        delete[] elms_conn;
        delete[] my_proc_dump;
        delete[] vis_data;
    }

    // inversion arrays
    if (inverse_flag){
        delete inv_grid;

        if (subdom_main){
            delete[] Ks_loc;
            delete[] Kxi_loc;
            delete[] Keta_loc;
            delete[] Ks_density_loc;
            delete[] Kxi_density_loc;
            delete[] Keta_density_loc;
            delete[] Ks_inv_loc;
            delete[] Kxi_inv_loc;
            delete[] Keta_inv_loc;
            // delete[] Kdensity_inv_loc;
            delete[] Ks_update_loc;
            delete[] Kxi_update_loc;
            delete[] Keta_update_loc;
            // delete[] Kdensity_update_loc;
            delete[] Ks_update_loc_previous;
            delete[] Kxi_update_loc_previous;
            delete[] Keta_update_loc_previous;
        }

       if (sub_nprocs <= 1){
            delete[] Tadj_loc;
            delete[] Tadj_density_loc;
       }

        if (optim_method==HALVE_STEPPING_MODE) {
            delete[] fac_b_loc_back;
            delete[] fac_c_loc_back;
            delete[] fac_f_loc_back;
            delete[] fun_loc_back;
            delete[] xi_loc_back;
            delete[] eta_loc_back;
        }

        if (optim_method==LBFGS_MODE) {
            delete[] fac_b_loc_back;
            delete[] fac_c_loc_back;
            delete[] fac_f_loc_back;
            delete[] fun_loc_back;
            delete[] xi_loc_back;
            delete[] eta_loc_back;

            if (id_sim==0){
                delete[] Ks_grad_store_loc;
                delete[] Kxi_grad_store_loc;
                delete[] Keta_grad_store_loc;
                delete[] Ks_model_store_loc;
                delete[] Kxi_model_store_loc;
                delete[] Keta_model_store_loc;
                delete[] fun_gradient_regularization_penalty_loc;
                delete[] xi_gradient_regularization_penalty_loc;
                delete[] eta_gradient_regularization_penalty_loc;
                delete[] fun_regularization_penalty_loc;
                delete[] xi_regularization_penalty_loc;
                delete[] eta_regularization_penalty_loc;
                delete[] fun_prior_loc;
                delete[] xi_prior_loc;
                delete[] eta_prior_loc;
            }
       }
    } // end if inverse_flag

    stdout_by_main("Memory deallocation done.");
}


// setput the grid parameters (material and source etc.)
void Grid::setup_grid_params(InputParams &IP, IO_utils& io) {

    // setup coordinates
    r_min   = depth2radius(IP.get_max_dep()); // convert from depth to radius
    r_max   = depth2radius(IP.get_min_dep()); // convert from depth to radius
    lat_min = IP.get_min_lat(); // in rad
    lat_max = IP.get_max_lat(); // in rad
    lon_min = IP.get_min_lon(); // in rad
    lon_max = IP.get_max_lon(); // in rad

    dr   = (r_max - r_min)     / (ngrid_k - 1);
    dlat = (lat_max - lat_min) / (ngrid_j - 1);
    dlon = (lon_max - lon_min) / (ngrid_i - 1);

    dt = dlat; dp = dlon;

    // starting position in global indices including ghost layers
    int tmp_offset_k = get_offset_k();
    int tmp_offset_j = get_offset_j();
    int tmp_offset_i = get_offset_i(); // offset_i - i_start_loc

    // output global grid parameters
    if (inter_sub_rank == 0 && myrank == 0 && id_sim == 0) {
        std::cout << "\n\n Global grid information: \n\n" << std::endl;
        std::cout << "  r_min r_max dr : "                << r_min   << " " << r_max << " " << dr << std::endl;
        std::cout << "  depth_min depth_max dz : "        << IP.get_min_dep() << " " << IP.get_max_dep() << std::endl;
        std::cout << "  lat_min lat_max dlat [degree] : " << lat_min*RAD2DEG << " " << lat_max*RAD2DEG << " " << dlat*RAD2DEG << std::endl;
        std::cout << "  lon_min lon_max dlon [degree] : " << lon_min*RAD2DEG << " " << lon_max*RAD2DEG << " " << dlon*RAD2DEG << std::endl;
        std::cout << "  ngrid_i ngrid_j ngrid_k : "      << ngrid_i << " " << ngrid_j << " " << ngrid_k << std::endl;
        std::cout << "\n\n" << std::endl;
    }

    // assign coordinates to 1d arrays
    // r
    for (int k = 0; k < loc_K; k++) {
        r_loc_1d[k] = r_min + (tmp_offset_k + k)*dr;
    }
    // lat
    for (int j = 0; j < loc_J; j++) {
        t_loc_1d[j] = lat_min + (tmp_offset_j + j)*dlat;
    }
    // lon
    for (int i = 0; i < loc_I; i++) {
        p_loc_1d[i] = lon_min + (tmp_offset_i + i)*dlon;
    }

    for (int i = 0; i < nprocs; i++) {
        synchronize_all_inter();
        if (i == myrank && id_sim == 0) {
            std::cout << "\n--- subdomain info ---- rank : " << i << "\n\n" << std::endl;
            std::cout << "  p_loc_1d min max in deg.: "  << p_loc_1d[0]*RAD2DEG << " " << p_loc_1d[loc_I-1]*RAD2DEG << std::endl;
            std::cout << "  t_loc_1d min max in deg.: "  << t_loc_1d[0]*RAD2DEG << " " << t_loc_1d[loc_J-1]*RAD2DEG << std::endl;
            std::cout << "  r_loc_1d min max in km  : "  << r_loc_1d[0]         << " " << r_loc_1d[loc_K-1]         << std::endl;

            std::cout << "  p_loc_1d min max in rad.: "  << p_loc_1d[0] << " " << p_loc_1d[loc_I-1] << std::endl;
            std::cout << "  t_loc_1d min max in rad.: "  << t_loc_1d[0] << " " << t_loc_1d[loc_J-1] << std::endl;

            std::cout << "  r_min, get_offset_k(), dr in km: "     << r_min           << " " << tmp_offset_k << " " << dr           << std::endl;
            std::cout << "  lat_min, get_offset_j(), dlat in deg.: " << lat_min*RAD2DEG << " " << tmp_offset_j << " " << dlat*RAD2DEG << std::endl;
            std::cout << "  lon_min, get_offset_i(), dlon in deg.: " << lon_min*RAD2DEG << " " << tmp_offset_i << " " << dlon*RAD2DEG << std::endl;
            std::cout << "\n" << std::endl;
        }
    }

    // independent read model data
    if (id_sim == 0){
        // read init model
        std::string f_model_path = IP.get_init_model_path();
        io.read_model(f_model_path,"xi",   xi_loc,    tmp_offset_i, tmp_offset_j, tmp_offset_k);
        io.read_model(f_model_path,"eta",  eta_loc,   tmp_offset_i, tmp_offset_j, tmp_offset_k);
        //io.read_model(f_model_path,"zeta", zeta_loc,  tmp_offset_i, tmp_offset_j, tmp_offset_k);
        io.read_model(f_model_path,"vel",  fun_loc,   tmp_offset_i, tmp_offset_j, tmp_offset_k); // use slowness array temprarily
        if(if_test) {
            // solver test
            io.read_model(f_model_path, "u", u_loc, tmp_offset_i, tmp_offset_j, tmp_offset_k);
        }

        // set zeta = 0 (optimization for zeta is not implemented yet)
        std::fill(zeta_loc, zeta_loc + loc_I*loc_J*loc_K, _0_CR);

        // copy initial model to prior model arrays
        if (optim_method==LBFGS_MODE){
            std::copy(xi_loc,  xi_loc + loc_I*loc_J*loc_K,  xi_prior_loc);
            std::copy(eta_loc, eta_loc + loc_I*loc_J*loc_K, eta_prior_loc);
            std::copy(fun_loc, fun_loc + loc_I*loc_J*loc_K, fun_prior_loc);
        }

    }

    // broadcast
    int n_total_loc_grid_points = loc_I * loc_J * loc_K;
    broadcast_cr_inter_sim(xi_loc,    n_total_loc_grid_points, 0);
    broadcast_cr_inter_sim(eta_loc,   n_total_loc_grid_points, 0);
    broadcast_cr_inter_sim(zeta_loc,  n_total_loc_grid_points, 0);
    broadcast_cr_inter_sim(fun_loc,   n_total_loc_grid_points, 0); // here passing velocity array
    if(if_test) {
        broadcast_cr_inter_sim(u_loc, n_total_loc_grid_points, 0);
    }

    // center of the domain
    CUSTOMREAL lon_center = (lon_min + lon_max) / 2.0;
    CUSTOMREAL lat_center = (lat_min + lat_max) / 2.0;

    // initialize other necessary arrays
    for (int k_r = 0; k_r < loc_K; k_r++) {
        for (int j_lat = 0; j_lat < loc_J; j_lat++) {
            for (int i_lon = 0; i_lon < loc_I; i_lon++) {

                // convert velocity to slowness
                fun_loc[I2V(i_lon, j_lat, k_r)] = _1_CR / fun_loc[I2V(i_lon, j_lat, k_r)];

                // calculate fac_a_loc, fac_b_loc, fac_c_loc, fac_f_loc
                fac_a_loc[I2V(i_lon, j_lat, k_r)] = _1_CR + _2_CR * zeta_loc[I2V(i_lon, j_lat, k_r)];
                fac_b_loc[I2V(i_lon, j_lat, k_r)] = _1_CR - _2_CR *   xi_loc[I2V(i_lon, j_lat, k_r)];
                fac_c_loc[I2V(i_lon, j_lat, k_r)] = _1_CR + _2_CR *   xi_loc[I2V(i_lon, j_lat, k_r)];
                fac_f_loc[I2V(i_lon, j_lat, k_r)] =       - _2_CR *  eta_loc[I2V(i_lon, j_lat, k_r)];

                // construct 3d coordinate arrays and node connectivity for visualization
                // exclude the ghost nodes
                if (i_start_vis <= i_lon && i_lon <= i_end_vis \
                 && j_start_vis <= j_lat && j_lat <= j_end_vis \
                 && k_start_vis <= k_r   && k_r   <= k_end_vis) {
                    int i_loc_tmp = i_lon - i_start_vis;
                    int j_loc_tmp = j_lat - j_start_vis;
                    int k_loc_tmp = k_r   - k_start_vis;

                    // get the coordinates of the point
                    CUSTOMREAL x, y, z;
                    //RLonLat2xyz(p_loc_1d[i_lon], t_loc_1d[j_lat], r_loc_1d[k_r], x, y, z); // complete sphere
                    WGS84ToCartesian(p_loc_1d[i_lon], t_loc_1d[j_lat], r_loc_1d[k_r], x, y, z, lon_center, lat_center); // WGS84

                    x_loc_3d[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = x;
                    y_loc_3d[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = y;
                    z_loc_3d[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = z;
                    p_loc_3d[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = p_loc_1d[i_lon]*RAD2DEG;
                    t_loc_3d[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = t_loc_1d[j_lat]*RAD2DEG;
                    // r_loc_3d[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = r_loc_1d[k_r]*M2KM;
                    r_loc_3d[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = r_loc_1d[k_r];

                    // dump proc id
                    my_proc_dump[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = myrank;

                    // node connectivity
                    // exclude the last layers
                    if (i_lon < i_end_vis \
                     && j_lat < j_end_vis \
                     && k_r   < k_end_vis) {
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9] = cell_type;
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9+1] = offset_nnodes_vis+I2V_VIS(i_loc_tmp,  j_loc_tmp,  k_loc_tmp);
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9+2] = offset_nnodes_vis+I2V_VIS(i_loc_tmp+1,j_loc_tmp,  k_loc_tmp);
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9+3] = offset_nnodes_vis+I2V_VIS(i_loc_tmp+1,j_loc_tmp+1,k_loc_tmp);
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9+4] = offset_nnodes_vis+I2V_VIS(i_loc_tmp,  j_loc_tmp+1,k_loc_tmp);
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9+5] = offset_nnodes_vis+I2V_VIS(i_loc_tmp,  j_loc_tmp,  k_loc_tmp+1);
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9+6] = offset_nnodes_vis+I2V_VIS(i_loc_tmp+1,j_loc_tmp,  k_loc_tmp+1);
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9+7] = offset_nnodes_vis+I2V_VIS(i_loc_tmp+1,j_loc_tmp+1,k_loc_tmp+1);
                        elms_conn[I2V_ELM_CONN(i_loc_tmp,j_loc_tmp,k_loc_tmp)*9+8] = offset_nnodes_vis+I2V_VIS(i_loc_tmp,  j_loc_tmp+1,k_loc_tmp+1);
                    }
                }
            }
        }
    } // end of for loop

    // setup parameters for inversion grids
//    if (inverse_flag)
//        setup_inv_grid_params(IP);

}



void Grid::initialize_kernels(){
    // initialize kernels
    if (subdom_main){
        std::fill(Ks_loc,       Ks_loc   + loc_I*loc_J*loc_K,       _0_CR);
        std::fill(Kxi_loc,      Kxi_loc  + loc_I*loc_J*loc_K,       _0_CR);
        std::fill(Keta_loc,     Keta_loc + loc_I*loc_J*loc_K,       _0_CR);
        std::fill(Ks_density_loc, Ks_density_loc + loc_I*loc_J*loc_K,   _0_CR);
        std::fill(Kxi_density_loc,Kxi_density_loc + loc_I*loc_J*loc_K,  _0_CR);
        std::fill(Keta_density_loc,Keta_density_loc + loc_I*loc_J*loc_K, _0_CR);
        //for (int k = 0; k < loc_K; k++) {
        //    for (int j = 0; j < loc_J; j++) {
        //        for (int i = 0; i < loc_I; i++) {
        //            Ks_loc[  I2V(i,j,k)] = _0_CR;
        //            Kxi_loc[ I2V(i,j,k)] = _0_CR;
        //            Keta_loc[I2V(i,j,k)] = _0_CR;
        //        }
        //    }
        //}
    }
}


// get a part of pointers from the requested array for visualization
CUSTOMREAL* Grid::get_array_for_vis(CUSTOMREAL* arr, bool inverse_value) {

    send_recev_boundary_data(arr);
    // add a routine for communication the boundary value
    // with the neighbors with line/point contact
    send_recev_boundary_data_kosumi(arr);

    for (int k_r = 0; k_r < loc_K; k_r++) {
        for (int j_lat = 0; j_lat < loc_J; j_lat++) {
            for (int i_lon = 0; i_lon < loc_I; i_lon++) {
                 if (i_start_vis <= i_lon && i_lon <= i_end_vis \
                  && j_start_vis <= j_lat && j_lat <= j_end_vis \
                  && k_start_vis <= k_r   && k_r   <= k_end_vis) {
                    int i_loc_tmp = i_lon - i_start_vis;
                    int j_loc_tmp = j_lat - j_start_vis;
                    int k_loc_tmp = k_r   - k_start_vis;

                    if(!inverse_value)
                        vis_data[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = arr[I2V(i_lon,j_lat,k_r)];
                    else
                        vis_data[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = _1_CR/arr[I2V(i_lon,j_lat,k_r)]; // used for convert from slowness to velocity
                 }
            }
        }
    }

    return vis_data;
}


// set visualization array to local array
void Grid::set_array_from_vis(CUSTOMREAL* arr) {

    for (int k_r = 0; k_r < loc_K; k_r++) {
        for (int j_lat = 0; j_lat < loc_J; j_lat++) {
            for (int i_lon = 0; i_lon < loc_I; i_lon++) {
                 if (i_start_vis <= i_lon && i_lon <= i_end_vis \
                  && j_start_vis <= j_lat && j_lat <= j_end_vis \
                  && k_start_vis <= k_r   && k_r   <= k_end_vis) {
                    int i_loc_tmp = i_lon - i_start_vis;
                    int j_loc_tmp = j_lat - j_start_vis;
                    int k_loc_tmp = k_r   - k_start_vis;

                    arr[I2V(i_lon,j_lat,k_r)] = vis_data[I2V_VIS(i_loc_tmp,j_loc_tmp,k_loc_tmp)];
                 }
            }
        }
    }

    // set values to ghost layer
    send_recev_boundary_data(arr);

}


// get a part of pointers from the requested array for visualization
void Grid::get_array_for_3d_output(const CUSTOMREAL *arr_in, CUSTOMREAL* arr_out, bool inverse_value) {

    for (int k_r = k_start_loc; k_r <= k_end_loc; k_r++) {
        for (int j_lat = j_start_loc; j_lat <= j_end_loc; j_lat++) {
            for (int i_lon = i_start_loc; i_lon <= i_end_loc; i_lon++) {
                    int i_loc_tmp = i_lon - i_start_loc;
                    int j_loc_tmp = j_lat - j_start_loc;
                    int k_loc_tmp = k_r   - k_start_loc;
                    if (!inverse_value)
                        arr_out[I2V_3D(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = arr_in[I2V(i_lon,j_lat,k_r)];
                    else
                        arr_out[I2V_3D(i_loc_tmp,j_loc_tmp,k_loc_tmp)] = _1_CR/arr_in[I2V(i_lon,j_lat,k_r)]; // used for convert from slowness to velocity
            }
        }
    }
}


void Grid::reinitialize_abcf(){
    if (subdom_main) {
        for (int k_r = 0; k_r < loc_K; k_r++) {
            for (int j_lat = 0; j_lat < loc_J; j_lat++) {
                for (int i_lon = 0; i_lon < loc_I; i_lon++) {
                    // initialize arrays
                    fac_a_loc[I2V(i_lon,j_lat,k_r)] = fac_a_loc[I2V(i_lon,j_lat,k_r)];
                    fac_b_loc[I2V(i_lon,j_lat,k_r)] = fac_b_loc[I2V(i_lon,j_lat,k_r)]/ my_square(r_loc_1d[k_r]);
                    fac_c_loc[I2V(i_lon,j_lat,k_r)] = fac_c_loc[I2V(i_lon,j_lat,k_r)]/(my_square(r_loc_1d[k_r])*my_square(std::cos(t_loc_1d[j_lat])));
                    fac_f_loc[I2V(i_lon,j_lat,k_r)] = fac_f_loc[I2V(i_lon,j_lat,k_r)]/(my_square(r_loc_1d[k_r])*          std::cos(t_loc_1d[j_lat]));
                }
            }
        }
    }
}


void Grid::rejuvenate_abcf(){
    if (subdom_main) {
        for (int k_r = 0; k_r < loc_K; k_r++) {
            for (int j_lat = 0; j_lat < loc_J; j_lat++) {
                for (int i_lon = 0; i_lon < loc_I; i_lon++) {
                    // initialize arrays
                    fac_a_loc[I2V(i_lon,j_lat,k_r)] = fac_a_loc[I2V(i_lon,j_lat,k_r)];
                    fac_b_loc[I2V(i_lon,j_lat,k_r)] = _1_CR - _2_CR * xi_loc[I2V(i_lon,j_lat,k_r)];
                    fac_c_loc[I2V(i_lon,j_lat,k_r)] = _1_CR + _2_CR * xi_loc[I2V(i_lon,j_lat,k_r)];
                    fac_f_loc[I2V(i_lon,j_lat,k_r)] =       - _2_CR * eta_loc[I2V(i_lon,j_lat,k_r)];
                }
            }
        }
    }
}


void Grid::setup_factors(Source &src){

    // calculate factors for the source
    a0   = src.get_fac_at_source(fac_a_loc);
    b0   = src.get_fac_at_source(fac_b_loc);
    c0   = src.get_fac_at_source(fac_c_loc);
    f0   = src.get_fac_at_source(fac_f_loc);
    fun0 = src.get_fac_at_source(fun_loc, false); // true for debug
}


void Grid::initialize_fields(Source& src, InputParams& IP){

    // get source position
    CUSTOMREAL src_r = src.get_src_r();
    CUSTOMREAL src_t = src.get_src_t();
    CUSTOMREAL src_p = src.get_src_p();

    // std out src positions
    CUSTOMREAL c0b0_minus_f0f0 = c0*b0 - f0*f0;

    // debug
    int n_source_node = 0;

    // std::cout << a0 << ' ' << b0 << ' ' << c0 << ' ' << f0 << ' ' << std::endl;



    for (int k_r = 0; k_r < loc_K; k_r++) {
        for (int j_lat = 0; j_lat < loc_J; j_lat++) {
            for (int i_lon = 0; i_lon < loc_I; i_lon++) {
                CUSTOMREAL dr_from_src = r_loc_1d[k_r]   - src_r;
                CUSTOMREAL dt_from_src = t_loc_1d[j_lat] - src_t;
                CUSTOMREAL dp_from_src = p_loc_1d[i_lon] - src_p;

                T0v_loc[I2V(i_lon,j_lat,k_r)] = fun0 * std::sqrt( _1_CR/a0                  *my_square(dr_from_src) \
                                                                + c0/(c0b0_minus_f0f0)      *my_square(dt_from_src) \
                                                                + b0/(c0b0_minus_f0f0)      *my_square(dp_from_src) \
                                                                + _2_CR*f0/(c0b0_minus_f0f0)*dt_from_src*dp_from_src);

                is_changed[I2V(i_lon,j_lat,k_r)] = true;

                if (isZero(T0v_loc[I2V(i_lon,j_lat,k_r)])) {
                    T0r_loc[I2V(i_lon,j_lat,k_r)] = _0_CR;
                    T0t_loc[I2V(i_lon,j_lat,k_r)] = _0_CR;
                    T0p_loc[I2V(i_lon,j_lat,k_r)] = _0_CR;
                } else {
                    T0r_loc[I2V(i_lon,j_lat,k_r)] = my_square(fun0)*(_1_CR/a0*dr_from_src)/T0v_loc[I2V(i_lon,j_lat,k_r)];
                    T0t_loc[I2V(i_lon,j_lat,k_r)] = my_square(fun0)*(c0/(c0b0_minus_f0f0)*dt_from_src+f0/(c0b0_minus_f0f0)*dp_from_src)/T0v_loc[I2V(i_lon,j_lat,k_r)];
                    T0p_loc[I2V(i_lon,j_lat,k_r)] = my_square(fun0)*(b0/(c0b0_minus_f0f0)*dp_from_src+f0/(c0b0_minus_f0f0)*dt_from_src)/T0v_loc[I2V(i_lon,j_lat,k_r)];
                }

                if (IP.get_stencil_order() == 1){
                    source_width = _1_CR-0.1;
                } else {
                    source_width = _2_CR;
                }

                if (std::abs(dr_from_src/dr) <= source_width \
                 && std::abs(dt_from_src/dt) <= source_width \
                 && std::abs(dp_from_src/dp) <= source_width) {

                    tau_loc[I2V(i_lon,j_lat,k_r)] = TAU_INITIAL_VAL;
                    is_changed[I2V(i_lon,j_lat,k_r)] = false;

                    n_source_node++;

                    // std::cout << "source def " << std::endl;
                    // std::cout << "p_loc_1d (lon): " << p_loc_1d[i_lon]*RAD2DEG << ", id_i: " << i_lon << ", src_p: " << src_p*RAD2DEG << std::endl;
                    // std::cout << "t_loc_1d (lat): " << t_loc_1d[j_lat]*RAD2DEG << ", id_j: " << j_lat << ", src_t: " << src_t*RAD2DEG << std::endl;
                    // std::cout << "r_loc_1d (r): " << r_loc_1d[k_r] << ", id_k: " << k_r << ", src_r: " << src_r << std::endl;

                    if (if_verbose) {
                        if ( (k_r == 0 && k_first()) || (k_r == loc_K-1 && k_last()) )
                            std::cout << "Warning: source is on the boundary k of the grid.\n";
                        if ( (j_lat == 0 && j_first()) || (j_lat == loc_J-1 && j_last()) )
                            std::cout << "Warning: source is on the boundary j of the grid.\n";
                        if ( (i_lon == 0 && i_first()) || (i_lon == loc_I-1 && i_last()) )
                            std::cout << "Warning: source is on the boundary i of the grid.\n";
                    }

                } else {
                    if (IP.get_stencil_type()==UPWIND)   // upwind scheme, initial tau should be large enough
                        tau_loc[I2V(i_lon,j_lat,k_r)] = TAU_INF_VAL;
                    else
                        tau_loc[I2V(i_lon,j_lat,k_r)] = TAU_INITIAL_VAL;
                    is_changed[I2V(i_lon,j_lat,k_r)] = true;

                }

                tau_old_loc[I2V(i_lon,j_lat,k_r)] = _0_CR;

            } // end loop i
        } // end loop j
    } // end loop k

    // std::cout << "p_loc_1d (lon): " << p_loc_1d[25]*RAD2DEG << ", id_i: " << 25
    //           << "t_loc_1d (lat): " << t_loc_1d[29]*RAD2DEG << ", id_j: " << 29
    //           << "r_loc_1d (r): " << r_loc_1d[41] << ", id_k: " << 41 <<  std::endl;

    // warning if source node is not found
    if( n_source_node > 0 && if_verbose )
        std::cout << "rank  n_source_node: " << myrank << "  " << n_source_node << std::endl;

}


void Grid::initialize_fields_teleseismic(){
    CUSTOMREAL inf_T = 2000.0;

    for (int k_r = 0; k_r < loc_K; k_r++) {
        for (int j_lat = 0; j_lat < loc_J; j_lat++) {
            for (int i_lon = 0; i_lon < loc_I; i_lon++) {
                is_changed[I2V(i_lon,j_lat,k_r)] = true;
                T_loc[I2V(i_lon,j_lat,k_r)]      = inf_T;
            }
        }
    }

    // setup of boundary arrival time conditions is done in iterator function
}


// copy the tau values to tau_old
void Grid::tau2tau_old() {
    std::copy(tau_loc, tau_loc+loc_I*loc_J*loc_K, tau_old_loc);
}


// copy the T values to tau_old
void Grid::T2tau_old() {
    std::copy(T_loc, T_loc+loc_I*loc_J*loc_K, tau_old_loc);
}


// copy the tau values to Tadj
void Grid::update_Tadj() {
    std::copy(tau_loc, tau_loc+loc_I*loc_J*loc_K, Tadj_loc);
}
void Grid::update_Tadj_density() {
    std::copy(tau_loc, tau_loc+loc_I*loc_J*loc_K, Tadj_density_loc);
}

void Grid::back_up_fun_xi_eta_bcf() {
    if (!subdom_main) return;

    std::copy(fun_loc, fun_loc+loc_I*loc_J*loc_K, fun_loc_back);
    std::copy(xi_loc,  xi_loc +loc_I*loc_J*loc_K, xi_loc_back);
    std::copy(eta_loc, eta_loc+loc_I*loc_J*loc_K, eta_loc_back);
    std::copy(fac_b_loc, fac_b_loc+loc_I*loc_J*loc_K, fac_b_loc_back);
    std::copy(fac_c_loc, fac_c_loc+loc_I*loc_J*loc_K, fac_c_loc_back);
    std::copy(fac_f_loc, fac_f_loc+loc_I*loc_J*loc_K, fac_f_loc_back);
}


void Grid::restore_fun_xi_eta_bcf() {
    if (!subdom_main) return;

    std::copy(fun_loc_back, fun_loc_back+loc_I*loc_J*loc_K, fun_loc);
    std::copy(xi_loc_back,  xi_loc_back +loc_I*loc_J*loc_K, xi_loc);
    std::copy(eta_loc_back, eta_loc_back+loc_I*loc_J*loc_K, eta_loc);
    std::copy(fac_b_loc_back, fac_b_loc_back+loc_I*loc_J*loc_K, fac_b_loc);
    std::copy(fac_c_loc_back, fac_c_loc_back+loc_I*loc_J*loc_K, fac_c_loc);
    std::copy(fac_f_loc_back, fac_f_loc_back+loc_I*loc_J*loc_K, fac_f_loc);
}


void Grid::calc_L1_and_Linf_diff(CUSTOMREAL& L1_diff, CUSTOMREAL& Linf_diff) {
    if (subdom_main) {
        L1_diff   = 0.0;
        Linf_diff = 0.0;
        CUSTOMREAL obj_func_glob = 0.0;

        // calculate L1 error
        for (int k_r = k_start_loc; k_r <= k_end_loc; k_r++) {
            for (int j_lat = j_start_loc; j_lat <= j_end_loc; j_lat++) {
                for (int i_lon = i_start_loc; i_lon <= i_end_loc; i_lon++) {
                    L1_diff   +=                    std::abs(tau_loc[I2V(i_lon,j_lat,k_r)] - tau_old_loc[I2V(i_lon,j_lat,k_r)]) * T0v_loc[I2V(i_lon,j_lat,k_r)];
                    Linf_diff  = std::max(Linf_diff,std::abs(tau_loc[I2V(i_lon,j_lat,k_r)] - tau_old_loc[I2V(i_lon,j_lat,k_r)]) * T0v_loc[I2V(i_lon,j_lat,k_r)]);
                    // std::cout << R_earth-r_loc_3d[k_r] << ' ' << t_loc_3d[j_lat] << ' ' << p_loc_3d[i_lon] << ' ' << T0v_loc[I2V(i_lon,j_lat,k_r)] << ' ' << std::endl;
                }
            }
        }

        // sum up the values of all processes
        obj_func_glob=0.0;
        allreduce_cr_single(L1_diff, obj_func_glob);
        L1_diff = obj_func_glob/((ngrid_i-2)*(ngrid_j-2)*(ngrid_k-2));
        obj_func_glob=0.0;
        allreduce_cr_single_max(Linf_diff, obj_func_glob);
        Linf_diff = obj_func_glob;///(ngrid_i*ngrid_j*ngrid_k);
    }
}


void Grid::calc_L1_and_Linf_diff_tele(CUSTOMREAL& L1_diff, CUSTOMREAL& Linf_diff) {
    if (subdom_main) {
        L1_diff   = 0.0;
        Linf_diff = 0.0;
        CUSTOMREAL obj_func_glob = 0.0;

        // calculate L1 error
        for (int k_r = k_start_loc; k_r <= k_end_loc; k_r++) {
            for (int j_lat = j_start_loc; j_lat <= j_end_loc; j_lat++) {
                for (int i_lon = i_start_loc; i_lon <= i_end_loc; i_lon++) {
                    L1_diff   +=                    std::abs(T_loc[I2V(i_lon,j_lat,k_r)] - tau_old_loc[I2V(i_lon,j_lat,k_r)]); // tau_old is used as T_old_loc here
                    Linf_diff  = std::max(Linf_diff,std::abs(T_loc[I2V(i_lon,j_lat,k_r)] - tau_old_loc[I2V(i_lon,j_lat,k_r)]));
                }
            }
        }

        // sum up the values of all processes
        obj_func_glob=0.0;
        allreduce_cr_single(L1_diff, obj_func_glob);
        L1_diff = obj_func_glob/((ngrid_i-2)*(ngrid_j-2)*(ngrid_k-2));
        obj_func_glob=0.0;
        allreduce_cr_single_max(Linf_diff, obj_func_glob);
        Linf_diff = obj_func_glob;///(ngrid_i*ngrid_j*ngrid_k);
    }
}


void Grid::calc_L1_and_Linf_diff_adj(CUSTOMREAL& L1_diff, CUSTOMREAL& Linf_diff) {

    if (subdom_main) {
        //L1_diff   = 0.0;
        Linf_diff = 0.0;
        CUSTOMREAL obj_func_glob = 0.0;
        CUSTOMREAL adj_factor = 1.0;

        // calculate L1 error
        for (int k_r = k_start_loc; k_r <= k_end_loc; k_r++) {
            for (int j_lat = j_start_loc; j_lat <= j_end_loc; j_lat++) {
                for (int i_lon = i_start_loc; i_lon <= i_end_loc; i_lon++) {
                    // Adjoint simulation use only Linf
                    Linf_diff  = std::max(Linf_diff,std::abs(tau_loc[I2V(i_lon,j_lat,k_r)] - Tadj_loc[I2V(i_lon,j_lat,k_r)]));
                }
            }
        }

        // sum up the values of all processes
        obj_func_glob=0.0;
        allreduce_cr_single_max(Linf_diff, obj_func_glob);
        Linf_diff = obj_func_glob*adj_factor;///(ngrid_i*ngrid_j*ngrid_k);
    }
}

void Grid::calc_L1_and_Linf_diff_adj_density(CUSTOMREAL& L1_diff, CUSTOMREAL& Linf_diff) {

    if (subdom_main) {
        //L1_diff   = 0.0;
        Linf_diff = 0.0;
        CUSTOMREAL obj_func_glob = 0.0;
        CUSTOMREAL adj_factor = 1.0;

        // calculate L1 error
        for (int k_r = k_start_loc; k_r <= k_end_loc; k_r++) {
            for (int j_lat = j_start_loc; j_lat <= j_end_loc; j_lat++) {
                for (int i_lon = i_start_loc; i_lon <= i_end_loc; i_lon++) {
                    // Adjoint simulation use only Linf
                    Linf_diff  = std::max(Linf_diff,std::abs(tau_loc[I2V(i_lon,j_lat,k_r)] - Tadj_density_loc[I2V(i_lon,j_lat,k_r)]));
                }
            }
        }

        // sum up the values of all processes
        obj_func_glob=0.0;
        allreduce_cr_single_max(Linf_diff, obj_func_glob);
        Linf_diff = obj_func_glob*adj_factor;///(ngrid_i*ngrid_j*ngrid_k);
    }
}

void Grid::calc_L1_and_Linf_error(CUSTOMREAL& L1_error, CUSTOMREAL& Linf_error) {
    L1_error   = 0.0;
    Linf_error = 0.0;
    CUSTOMREAL obj_func_glob = 0.0;

    // calculate L1 error
    for (int k_r = k_start_loc; k_r <= k_end_loc; k_r++) {
        for (int j_lat = j_start_loc; j_lat <= j_end_loc; j_lat++) {
            for (int i_lon = i_start_loc; i_lon <= i_end_loc; i_lon++) {
                L1_error   +=                     std::abs(u_loc[I2V(i_lon,j_lat,k_r)] - T0v_loc[I2V(i_lon,j_lat,k_r)] * tau_loc[I2V(i_lon,j_lat,k_r)]);
                Linf_error  = std::max(Linf_error,std::abs(u_loc[I2V(i_lon,j_lat,k_r)] - T0v_loc[I2V(i_lon,j_lat,k_r)] * tau_loc[I2V(i_lon,j_lat,k_r)]));
            }
        }
    }

    // sum up the values of all processes
    obj_func_glob=0.0;
    allreduce_cr_single(L1_error, obj_func_glob);
    L1_error       = obj_func_glob;
    obj_func_glob = 0.0;
    allreduce_cr_single_max(Linf_error, obj_func_glob);
    Linf_error     = obj_func_glob;///(ngrid_i*ngrid_j*ngrid_k);
}


void Grid::prepare_boundary_data_to_send(CUSTOMREAL* arr) {
    // store the pointers to the ghost layer's elements to be sent / to receive
    // node order should be always smaller to larger

    // j-k plane negative
    if (neighbors_id[0] != -1) {
        for (int i = 0; i < n_ghost_layers; i++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int j = j_start_loc; j <= j_end_loc; j++) {
                    bin_s[i*loc_J_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_J_excl_ghost + (j-j_start_loc)] = arr[I2V(i+n_ghost_layers,j,k)];
                }
            }
        }
    }
    // j-k plane positive
    if (neighbors_id[1] != -1) {
        for (int i = 0; i < n_ghost_layers; i++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int j = j_start_loc; j <= j_end_loc; j++) {
                    bip_s[i*loc_J_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_J_excl_ghost + (j-j_start_loc)] = arr[I2V(loc_I-n_ghost_layers*2+i,j,k)];
                }
            }
        }
    }
    // i-k plane negative
    if (neighbors_id[2] != -1) {
        for (int j = 0; j < n_ghost_layers; j++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    bjn_s[j*loc_I_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_I_excl_ghost + (i-i_start_loc)] = arr[I2V(i,j+n_ghost_layers,k)];
                }
            }
        }
    }
    // i-k plane positive
    if (neighbors_id[3] != -1) {
        for (int j = 0; j < n_ghost_layers; j++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    bjp_s[j*loc_I_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_I_excl_ghost + (i-i_start_loc)] = arr[I2V(i,loc_J-n_ghost_layers*2+j,k)];
                }
            }
        }
    }

    // i-j plane negative
    if (neighbors_id[4] != -1) {
        for (int k = 0; k < n_ghost_layers; k++) {
            for (int j = j_start_loc; j <= j_end_loc; j++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    bkn_s[k*loc_I_excl_ghost*loc_J_excl_ghost + (j-j_start_loc)*loc_I_excl_ghost + (i-i_start_loc)] = arr[I2V(i,j,k+n_ghost_layers)];
                }
            }
        }
    }
    // i-j plane positive
    if (neighbors_id[5] != -1) {
        for (int k = 0; k < n_ghost_layers; k++) {
            for (int j = j_start_loc; j <= j_end_loc; j++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    bkp_s[k*loc_I_excl_ghost*loc_J_excl_ghost + (j-j_start_loc)*loc_I_excl_ghost + (i-i_start_loc)] = arr[I2V(i,j,loc_K-n_ghost_layers*2+k)];
                }
            }
        }
    }

    //stdout_by_main("Preparation ghost layers done.");
}

void Grid::assign_received_data_to_ghost(CUSTOMREAL* arr) {
    // store the pointers to the ghost layer's elements to be sent / to receive
    // node order should be always smaller to larger

    // j-k plane negative
    if (neighbors_id[0] != -1) {
        for (int i = 0; i < n_ghost_layers; i++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int j = j_start_loc; j <= j_end_loc; j++) {
                    arr[I2V(i,j,k)] = bin_r[i*loc_J_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_J_excl_ghost + (j-j_start_loc)];
                }
            }
        }
    }
    // j-k plane positive
    if (neighbors_id[1] != -1) {
        for (int i = 0; i < n_ghost_layers; i++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int j = j_start_loc; j <= j_end_loc; j++) {
                     arr[I2V(loc_I-n_ghost_layers+i,j,k)] = bip_r[i*loc_J_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_J_excl_ghost + (j-j_start_loc)];
                }
            }
        }
    }
    // i-k plane negative
    if (neighbors_id[2] != -1) {
        for (int j = 0; j < n_ghost_layers; j++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    arr[I2V(i,j,k)] = bjn_r[j*loc_I_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_I_excl_ghost + (i-i_start_loc)];
                }
            }
        }
    }
    // i-k plane positive
    if (neighbors_id[3] != -1) {
        for (int j = 0; j < n_ghost_layers; j++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    arr[I2V(i,loc_J-n_ghost_layers+j,k)] = bjp_r[j*loc_I_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_I_excl_ghost + (i-i_start_loc)];
                }
            }
        }
    }

    // i-j plane negative
    if (neighbors_id[4] != -1) {
        for (int k = 0; k < n_ghost_layers; k++) {
            for (int j = j_start_loc; j <= j_end_loc; j++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    arr[I2V(i,j,k)] = bkn_r[k*loc_I_excl_ghost*loc_J_excl_ghost + (j-j_start_loc)*loc_I_excl_ghost + (i-i_start_loc)];
                }
            }
        }
    }
    // i-j plane positive
    if (neighbors_id[5] != -1) {
        for (int k = 0; k < n_ghost_layers; k++) {
            for (int j = j_start_loc; j <= j_end_loc; j++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    arr[I2V(i,j,loc_K-n_ghost_layers+k)] = bkp_r[k*loc_I_excl_ghost*loc_J_excl_ghost + (j-j_start_loc)*loc_I_excl_ghost + (i-i_start_loc)];
                }
            }
        }
    }

    //stdout_by_main("Preparation ghost layers done.");
}

void Grid::assign_received_data_to_ghost_av(CUSTOMREAL* arr) {
    // store the pointers to the ghost layer's elements to be sent / to receive
    // node order should be always smaller to larger

    // j-k plane negative
    if (neighbors_id[0] != -1) {
        for (int i = 0; i < n_ghost_layers; i++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int j = j_start_loc; j <= j_end_loc; j++) {
                    arr[I2V(i,j,k)] += bin_r[i*loc_J_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_J_excl_ghost + (j-j_start_loc)]/_2_CR;
                }
            }
        }
    }
    // j-k plane positive
    if (neighbors_id[1] != -1) {
        for (int i = 0; i < n_ghost_layers; i++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int j = j_start_loc; j <= j_end_loc; j++) {
                     arr[I2V(loc_I-n_ghost_layers+i,j,k)] += bip_r[i*loc_J_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_J_excl_ghost + (j-j_start_loc)]/_2_CR;
                }
            }
        }
    }
    // i-k plane negative
    if (neighbors_id[2] != -1) {
        for (int j = 0; j < n_ghost_layers; j++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    arr[I2V(i,j,k)] += bjn_r[j*loc_I_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_I_excl_ghost + (i-i_start_loc)]/_2_CR;
                }
            }
        }
    }
    // i-k plane positive
    if (neighbors_id[3] != -1) {
        for (int j = 0; j < n_ghost_layers; j++) {
            for (int k = k_start_loc; k <= k_end_loc; k++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    arr[I2V(i,loc_J-n_ghost_layers+j,k)] += bjp_r[j*loc_I_excl_ghost*loc_K_excl_ghost + (k-k_start_loc)*loc_I_excl_ghost + (i-i_start_loc)]/_2_CR;
                }
            }
        }
    }

    // i-j plane negative
    if (neighbors_id[4] != -1) {
        for (int k = 0; k < n_ghost_layers; k++) {
            for (int j = j_start_loc; j <= j_end_loc; j++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    arr[I2V(i,j,k)] += bkn_r[k*loc_I_excl_ghost*loc_J_excl_ghost + (j-j_start_loc)*loc_I_excl_ghost + (i-i_start_loc)]/_2_CR;
                }
            }
        }
    }
    // i-j plane positive
    if (neighbors_id[5] != -1) {
        for (int k = 0; k < n_ghost_layers; k++) {
            for (int j = j_start_loc; j <= j_end_loc; j++) {
                for (int i = i_start_loc; i <= i_end_loc; i++) {
                    arr[I2V(i,j,loc_K-n_ghost_layers+k)] += bkp_r[k*loc_I_excl_ghost*loc_J_excl_ghost + (j-j_start_loc)*loc_I_excl_ghost + (i-i_start_loc)]/_2_CR;
                }
            }
        }
    }

    //stdout_by_main("Preparation ghost layers done.");
}



void Grid::send_recev_boundary_data(CUSTOMREAL* arr){
    prepare_boundary_data_to_send(arr);

    // i-direction negative
    if (neighbors_id[0] != -1) {
        // send boundary layer to neighbor
        isend_cr(bin_s, n_grid_bound_i*n_ghost_layers, neighbors_id[0], mpi_send_reqs[0]);
        // receive boundary layer from neighbor
        irecv_cr(bin_r, n_grid_bound_i*n_ghost_layers, neighbors_id[0], mpi_recv_reqs[0]);
    }
    // i-direction positive
    if (neighbors_id[1] != -1) {
        // send boundary layer to neighbor
        isend_cr(bip_s, n_grid_bound_i*n_ghost_layers, neighbors_id[1], mpi_send_reqs[1]);
        // receive boundary layer from neighbor
        irecv_cr(bip_r, n_grid_bound_i*n_ghost_layers, neighbors_id[1], mpi_recv_reqs[1]);
    }
    // j-direction negative
    if (neighbors_id[2] != -1) {
        // send boundary layer to neighbor
        isend_cr(bjn_s, n_grid_bound_j*n_ghost_layers, neighbors_id[2], mpi_send_reqs[2]);
        // receive boundary layer from neighbor
        irecv_cr(bjn_r, n_grid_bound_j*n_ghost_layers, neighbors_id[2], mpi_recv_reqs[2]);
    }
    // j-direction positive
    if (neighbors_id[3] != -1) {
        // send boundary layer to neighbor
        isend_cr(bjp_s, n_grid_bound_j*n_ghost_layers, neighbors_id[3], mpi_send_reqs[3]);
        // receive boundary layer from neighbor
        irecv_cr(bjp_r, n_grid_bound_j*n_ghost_layers, neighbors_id[3], mpi_recv_reqs[3]);
    }
    // k-direction negative
    if (neighbors_id[4] != -1) {
        // send boundary layer to neighbor
        isend_cr(bkn_s, n_grid_bound_k*n_ghost_layers, neighbors_id[4], mpi_send_reqs[4]);
        // receive boundary layer from neighbor
        irecv_cr(bkn_r, n_grid_bound_k*n_ghost_layers, neighbors_id[4], mpi_recv_reqs[4]);
    }
    // k-direction positive
    if (neighbors_id[5] != -1) {
        // send boundary layer to neighbor
        isend_cr(bkp_s, n_grid_bound_k*n_ghost_layers, neighbors_id[5], mpi_send_reqs[5]);
        // receive boundary layer from neighbor
        irecv_cr(bkp_r, n_grid_bound_k*n_ghost_layers, neighbors_id[5], mpi_recv_reqs[5]);
    }

    // wait for finishing communication
    for (int i = 0; i < 6; i++) {
        if (neighbors_id[i] != -1) {
            wait_req(mpi_send_reqs[i]);
            wait_req(mpi_recv_reqs[i]);
        }
    }

    // THIS SYNCHRONIZATION IS IMPORTANT.
    synchronize_all_inter();

    assign_received_data_to_ghost(arr);

}


void Grid::send_recev_boundary_data_av(CUSTOMREAL* arr){
    prepare_boundary_data_to_send(arr);

    // i-direction negative
    if (neighbors_id[0] != -1) {
        // send boundary layer to neighbor
        isend_cr(bin_s, n_grid_bound_i*n_ghost_layers, neighbors_id[0], mpi_send_reqs[0]);
        // receive boundary layer from neighbor
        irecv_cr(bin_r, n_grid_bound_i*n_ghost_layers, neighbors_id[0], mpi_recv_reqs[0]);
    }
    // i-direction positive
    if (neighbors_id[1] != -1) {
        // send boundary layer to neighbor
        isend_cr(bip_s, n_grid_bound_i*n_ghost_layers, neighbors_id[1], mpi_send_reqs[1]);
        // receive boundary layer from neighbor
        irecv_cr(bip_r, n_grid_bound_i*n_ghost_layers, neighbors_id[1], mpi_recv_reqs[1]);
    }
    // j-direction negative
    if (neighbors_id[2] != -1) {
        // send boundary layer to neighbor
        isend_cr(bjn_s, n_grid_bound_j*n_ghost_layers, neighbors_id[2], mpi_send_reqs[2]);
        // receive boundary layer from neighbor
        irecv_cr(bjn_r, n_grid_bound_j*n_ghost_layers, neighbors_id[2], mpi_recv_reqs[2]);
    }
    // j-direction positive
    if (neighbors_id[3] != -1) {
        // send boundary layer to neighbor
        isend_cr(bjp_s, n_grid_bound_j*n_ghost_layers, neighbors_id[3], mpi_send_reqs[3]);
        // receive boundary layer from neighbor
        irecv_cr(bjp_r, n_grid_bound_j*n_ghost_layers, neighbors_id[3], mpi_recv_reqs[3]);
    }
    // k-direction negative
    if (neighbors_id[4] != -1) {
        // send boundary layer to neighbor
        isend_cr(bkn_s, n_grid_bound_k*n_ghost_layers, neighbors_id[4], mpi_send_reqs[4]);
        // receive boundary layer from neighbor
        irecv_cr(bkn_r, n_grid_bound_k*n_ghost_layers, neighbors_id[4], mpi_recv_reqs[4]);
    }
    // k-direction positive
    if (neighbors_id[5] != -1) {
        // send boundary layer to neighbor
        isend_cr(bkp_s, n_grid_bound_k*n_ghost_layers, neighbors_id[5], mpi_send_reqs[5]);
        // receive boundary layer from neighbor
        irecv_cr(bkp_r, n_grid_bound_k*n_ghost_layers, neighbors_id[5], mpi_recv_reqs[5]);
    }

    // wait for finishing communication
    for (int i = 0; i < 6; i++) {
        if (neighbors_id[i] != -1) {
            wait_req(mpi_send_reqs[i]);
            wait_req(mpi_recv_reqs[i]);
        }
    }

    assign_received_data_to_ghost_av(arr);

}


void Grid::prepare_boundary_data_to_send_kosumi(CUSTOMREAL* arr) {
    // ij nn
    if (neighbors_id_ij[0] != -1)
        for (int k = k_start_loc; k <= k_end_vis; k++)
            bij_nn_s[k-k_start_loc] = arr[I2V(i_start_loc,j_start_loc,k)];
    // ij np
    if (neighbors_id_ij[1] != -1)
        for (int k = k_start_loc; k <= k_end_vis; k++)
            bij_np_s[k-k_start_loc] = arr[I2V(i_start_loc,j_end_loc,k)];
    // ij pn
    if (neighbors_id_ij[2] != -1)
        for (int k = k_start_loc; k <= k_end_vis; k++)
            bij_pn_s[k-k_start_loc] = arr[I2V(i_end_loc,j_start_loc,k)];
    // ij pp
    if (neighbors_id_ij[3] != -1)
        for (int k = k_start_loc; k <= k_end_vis; k++)
            bij_pp_s[k-k_start_loc] = arr[I2V(i_end_loc,j_end_loc,k)];
    // jk nn
    if (neighbors_id_jk[0] != -1)
        for (int i = i_start_loc; i <= i_end_vis; i++)
            bjk_nn_s[i-i_start_loc] = arr[I2V(i,j_start_loc,k_start_loc)];
    // jk np
    if (neighbors_id_jk[1] != -1)
        for (int i = i_start_loc; i <= i_end_vis; i++)
            bjk_np_s[i-i_start_loc] = arr[I2V(i,j_start_loc,k_end_loc)];
    // jk pn
    if (neighbors_id_jk[2] != -1)
        for (int i = i_start_loc; i <= i_end_vis; i++)
            bjk_pn_s[i-i_start_loc] = arr[I2V(i,j_end_loc,k_start_loc)];
    // jk pp
    if (neighbors_id_jk[3] != -1)
        for (int i = i_start_loc; i <= i_end_vis; i++)
            bjk_pp_s[i-i_start_loc] = arr[I2V(i,j_end_loc,k_end_loc)];
    // ik nn
    if (neighbors_id_ik[0] != -1)
        for (int j = j_start_loc; j <= j_end_vis; j++)
            bik_nn_s[j-j_start_loc] = arr[I2V(i_start_loc,j,k_start_loc)];
    // ik np
    if (neighbors_id_ik[1] != -1)
        for (int j = j_start_loc; j <= j_end_vis; j++)
            bik_np_s[j-j_start_loc] = arr[I2V(i_start_loc,j,k_end_loc)];
    // ik pn
    if (neighbors_id_ik[2] != -1)
        for (int j = j_start_loc; j <= j_end_vis; j++)
            bik_pn_s[j-j_start_loc] = arr[I2V(i_end_loc,j,k_start_loc)];
    // ik pp
    if (neighbors_id_ik[3] != -1)
        for (int j = j_start_loc; j <= j_end_vis; j++)
            bik_pp_s[j-j_start_loc] = arr[I2V(i_end_loc,j,k_end_loc)];
    // ijk nnn
    if (neighbors_id_ijk[0] != -1)
        bijk_nnn_s[0] = arr[I2V(i_start_loc,j_start_loc,k_start_loc)];
    // ijk nnp
    if (neighbors_id_ijk[1] != -1)
        bijk_nnp_s[0] = arr[I2V(i_start_loc,j_start_loc,k_end_loc)];
    // ijk npn
    if (neighbors_id_ijk[2] != -1)
        bijk_npn_s[0] = arr[I2V(i_start_loc,j_end_loc,k_start_loc)];
    // ijk npp
    if (neighbors_id_ijk[3] != -1)
        bijk_npp_s[0] = arr[I2V(i_start_loc,j_end_loc,k_end_loc)];
    // ijk pnn
    if (neighbors_id_ijk[4] != -1)
        bijk_pnn_s[0] = arr[I2V(i_end_loc,j_start_loc,k_start_loc)];
    // ijk pnp
    if (neighbors_id_ijk[5] != -1)
        bijk_pnp_s[0] = arr[I2V(i_end_loc,j_start_loc,k_end_loc)];
    // ijk ppn
    if (neighbors_id_ijk[6] != -1)
        bijk_ppn_s[0] = arr[I2V(i_end_loc,j_end_loc,k_start_loc)];
    // ijk ppp
    if (neighbors_id_ijk[7] != -1)
        bijk_ppp_s[0] = arr[I2V(i_end_loc,j_end_loc,k_end_loc)];
}


void Grid::assign_received_data_to_ghost_kosumi(CUSTOMREAL* arr) {
    // ij nn
    if (neighbors_id_ij[0] != -1)
        for (int k = k_start_loc; k <= k_end_vis; k++)
            arr[I2V(i_start_loc-1,j_start_loc-1,k)] = bij_nn_r[k-k_start_loc];
    // ij np
    if (neighbors_id_ij[1] != -1)
        for (int k = k_start_loc; k <= k_end_vis; k++)
            arr[I2V(i_start_loc-1,j_end_vis,k)] = bij_np_r[k-k_start_loc];
    // ij pn
    if (neighbors_id_ij[2] != -1)
        for (int k = k_start_loc; k <= k_end_vis; k++)
            arr[I2V(i_end_vis,j_start_loc-1,k)] = bij_pn_r[k-k_start_loc];
    // ij pp
    if (neighbors_id_ij[3] != -1)
        for (int k = k_start_loc; k <= k_end_vis; k++)
            arr[I2V(i_end_vis,j_end_loc+1,k)] = bij_pp_r[k-k_start_loc];
    // jk nn
    if (neighbors_id_jk[0] != -1)
        for (int i = i_start_loc; i <= i_end_vis; i++)
            arr[I2V(i,j_start_loc-1,k_start_loc-1)] = bjk_nn_r[i-i_start_loc];
    // jk np
    if (neighbors_id_jk[1] != -1)
        for (int i = i_start_loc; i <= i_end_vis; i++)
            arr[I2V(i,j_start_loc-1,k_end_vis)] = bjk_np_r[i-i_start_loc];
    // jk pn
    if (neighbors_id_jk[2] != -1)
        for (int i = i_start_loc; i <= i_end_vis; i++)
            arr[I2V(i,j_end_vis,k_start_loc-1)] = bjk_pn_r[i-i_start_loc];
    // jk pp
    if (neighbors_id_jk[3] != -1)
        for (int i = i_start_loc; i <= i_end_vis; i++)
            arr[I2V(i,j_end_vis,k_end_vis)] = bjk_pp_r[i-i_start_loc];
    // ik nn
    if (neighbors_id_ik[0] != -1)
        for (int j = j_start_loc; j <= j_end_vis; j++)
            arr[I2V(i_start_loc-1,j,k_start_loc-1)] = bik_nn_r[j-j_start_loc];
    // ik np
    if (neighbors_id_ik[1] != -1)
        for (int j = j_start_loc; j <= j_end_vis; j++)
            arr[I2V(i_start_loc-1,j,k_end_vis)] = bik_np_r[j-j_start_loc];
    // ik pn
    if (neighbors_id_ik[2] != -1)
        for (int j = j_start_loc; j <= j_end_vis; j++)
            arr[I2V(i_end_vis,j,k_start_loc-1)] = bik_pn_r[j-j_start_loc];
    // ik pp
    if (neighbors_id_ik[3] != -1)
        for (int j = j_start_loc; j <= j_end_vis; j++)
            arr[I2V(i_end_vis,j,k_end_vis)] = bik_pp_r[j-j_start_loc];
    // ijk nnn
    if (neighbors_id_ijk[0] != -1)
        arr[I2V(i_start_loc-1,j_start_loc-1,k_start_loc-1)] = bijk_nnn_r[0];
    // ijk nnp
    if (neighbors_id_ijk[1] != -1)
        arr[I2V(i_start_loc-1,j_start_loc-1,k_end_vis)] = bijk_nnp_r[0];
    // ijk npn
    if (neighbors_id_ijk[2] != -1)
        arr[I2V(i_start_loc-1,j_end_vis,k_start_loc-1)] = bijk_npn_r[0];
    // ijk npp
    if (neighbors_id_ijk[3] != -1)
        arr[I2V(i_start_loc-1,j_end_vis,k_end_vis)] = bijk_npp_r[0];
    // ijk pnn
    if (neighbors_id_ijk[4] != -1)
        arr[I2V(i_end_vis,j_start_loc-1,k_start_loc-1)] = bijk_pnn_r[0];
    // ijk pnp
    if (neighbors_id_ijk[5] != -1)
        arr[I2V(i_end_vis,j_start_loc-1,k_end_vis)] = bijk_pnp_r[0];
    // ijk ppn
    if (neighbors_id_ijk[6] != -1)
        arr[I2V(i_end_vis,j_end_vis,k_start_loc-1)] = bijk_ppn_r[0];
    // ijk ppp
    if (neighbors_id_ijk[7] != -1)
        arr[I2V(i_end_vis,j_end_vis,k_end_vis)] = bijk_ppp_r[0];

}


void Grid::send_recev_boundary_data_kosumi(CUSTOMREAL* arr){

    prepare_boundary_data_to_send_kosumi(arr);

    // ij nn
    if (neighbors_id_ij[0] != -1){
        // send boundary layer to neighbor
        isend_cr(bij_nn_s, loc_K_vis, neighbors_id_ij[0], mpi_send_reqs_kosumi[0]);
        // receive boundary layer from neighbor
        irecv_cr(bij_nn_r, loc_K_vis, neighbors_id_ij[0], mpi_recv_reqs_kosumi[0]);
    }
    // ij np
    if (neighbors_id_ij[1] != -1){
        // send boundary layer to neighbor
        isend_cr(bij_np_s, loc_K_vis, neighbors_id_ij[1], mpi_send_reqs_kosumi[1]);
        // receive boundary layer from neighbor
        irecv_cr(bij_np_r, loc_K_vis, neighbors_id_ij[1], mpi_recv_reqs_kosumi[1]);
    }
    // ij pn
    if (neighbors_id_ij[2] != -1){
        // send boundary layer to neighbor
        isend_cr(bij_pn_s, loc_K_vis, neighbors_id_ij[2], mpi_send_reqs_kosumi[2]);
        // receive boundary layer from neighbor
        irecv_cr(bij_pn_r, loc_K_vis, neighbors_id_ij[2], mpi_recv_reqs_kosumi[2]);
    }
    // ij pp
    if (neighbors_id_ij[3] != -1){
        // send boundary layer to neighbor
        isend_cr(bij_pp_s, loc_K_vis, neighbors_id_ij[3], mpi_send_reqs_kosumi[3]);
        // receive boundary layer from neighbor
        irecv_cr(bij_pp_r, loc_K_vis, neighbors_id_ij[3], mpi_recv_reqs_kosumi[3]);
    }
    // jk nn
    if (neighbors_id_jk[0] != -1){
        // send boundary layer to neighbor
        isend_cr(bjk_nn_s, loc_I_vis, neighbors_id_jk[0], mpi_send_reqs_kosumi[4]);
        // receive boundary layer from neighbor
        irecv_cr(bjk_nn_r, loc_I_vis, neighbors_id_jk[0], mpi_recv_reqs_kosumi[4]);
    }
    // jk np
    if (neighbors_id_jk[1] != -1){
        // send boundary layer to neighbor
        isend_cr(bjk_np_s, loc_I_vis, neighbors_id_jk[1], mpi_send_reqs_kosumi[5]);
        // receive boundary layer from neighbor
        irecv_cr(bjk_np_r, loc_I_vis, neighbors_id_jk[1], mpi_recv_reqs_kosumi[5]);
    }
    // jk pn
    if (neighbors_id_jk[2] != -1){
        // send boundary layer to neighbor
        isend_cr(bjk_pn_s, loc_I_vis, neighbors_id_jk[2], mpi_send_reqs_kosumi[6]);
        // receive boundary layer from neighbor
        irecv_cr(bjk_pn_r, loc_I_vis, neighbors_id_jk[2], mpi_recv_reqs_kosumi[6]);
    }
    // jk pp
    if (neighbors_id_jk[3] != -1){
        // send boundary layer to neighbor
        isend_cr(bjk_pp_s, loc_I_vis, neighbors_id_jk[3], mpi_send_reqs_kosumi[7]);
        // receive boundary layer from neighbor
        irecv_cr(bjk_pp_r, loc_I_vis, neighbors_id_jk[3], mpi_recv_reqs_kosumi[7]);
    }
    // ik nn
    if (neighbors_id_ik[0] != -1){
        // send boundary layer to neighbor
        isend_cr(bik_nn_s, loc_J_vis, neighbors_id_ik[0], mpi_send_reqs_kosumi[8]);
        // receive boundary layer from neighbor
        irecv_cr(bik_nn_r, loc_J_vis, neighbors_id_ik[0], mpi_recv_reqs_kosumi[8]);
    }
    // ik np
    if (neighbors_id_ik[1] != -1){
        // send boundary layer to neighbor
        isend_cr(bik_np_s, loc_J_vis, neighbors_id_ik[1], mpi_send_reqs_kosumi[9]);
        // receive boundary layer from neighbor
        irecv_cr(bik_np_r, loc_J_vis, neighbors_id_ik[1], mpi_recv_reqs_kosumi[9]);
    }
    // ik pn
    if (neighbors_id_ik[2] != -1){
        // send boundary layer to neighbor
        isend_cr(bik_pn_s, loc_J_vis, neighbors_id_ik[2], mpi_send_reqs_kosumi[10]);
        // receive boundary layer from neighbor
        irecv_cr(bik_pn_r, loc_J_vis, neighbors_id_ik[2], mpi_recv_reqs_kosumi[10]);
    }
    // ik pp
    if (neighbors_id_ik[3] != -1){
        // send boundary layer to neighbor
        isend_cr(bik_pp_s, loc_J_vis, neighbors_id_ik[3], mpi_send_reqs_kosumi[11]);
        // receive boundary layer from neighbor
        irecv_cr(bik_pp_r, loc_J_vis, neighbors_id_ik[3], mpi_recv_reqs_kosumi[11]);
    }
    // ijk nnn
    if (neighbors_id_ijk[0] != -1){
        // send boundary layer to neighbor
        isend_cr(bijk_nnn_s, 1, neighbors_id_ijk[0], mpi_send_reqs_kosumi[12]);
        // receive boundary layer from neighbor
        irecv_cr(bijk_nnn_r, 1, neighbors_id_ijk[0], mpi_recv_reqs_kosumi[12]);
    }
    // ijk nnp
    if (neighbors_id_ijk[1] != -1){
        // send boundary layer to neighbor
        isend_cr(bijk_nnp_s, 1, neighbors_id_ijk[1], mpi_send_reqs_kosumi[13]);
        // receive boundary layer from neighbor
        irecv_cr(bijk_nnp_r, 1, neighbors_id_ijk[1], mpi_recv_reqs_kosumi[13]);
    }
    // ijk npn
    if (neighbors_id_ijk[2] != -1){
        // send boundary layer to neighbor
        isend_cr(bijk_npn_s, 1, neighbors_id_ijk[2], mpi_send_reqs_kosumi[14]);
        // receive boundary layer from neighbor
        irecv_cr(bijk_npn_r, 1, neighbors_id_ijk[2], mpi_recv_reqs_kosumi[14]);
    }
    // ijk npp
    if (neighbors_id_ijk[3] != -1){
        // send boundary layer to neighbor
        isend_cr(bijk_npp_s, 1, neighbors_id_ijk[3], mpi_send_reqs_kosumi[15]);
        // receive boundary layer from neighbor
        irecv_cr(bijk_npp_r, 1, neighbors_id_ijk[3], mpi_recv_reqs_kosumi[15]);
    }
    // ijk pnn
    if (neighbors_id_ijk[4] != -1){
        // send boundary layer to neighbor
        isend_cr(bijk_pnn_s, 1, neighbors_id_ijk[4], mpi_send_reqs_kosumi[16]);
        // receive boundary layer from neighbor
        irecv_cr(bijk_pnn_r, 1, neighbors_id_ijk[4], mpi_recv_reqs_kosumi[16]);
    }
    // ijk pnp
    if (neighbors_id_ijk[5] != -1){
        // send boundary layer to neighbor
        isend_cr(bijk_pnp_s, 1, neighbors_id_ijk[5], mpi_send_reqs_kosumi[17]);
        // receive boundary layer from neighbor
        irecv_cr(bijk_pnp_r, 1, neighbors_id_ijk[5], mpi_recv_reqs_kosumi[17]);
    }
    // ijk ppn
    if (neighbors_id_ijk[6] != -1){
        // send boundary layer to neighbor
        isend_cr(bijk_ppn_s, 1, neighbors_id_ijk[6], mpi_send_reqs_kosumi[18]);
        // receive boundary layer from neighbor
        irecv_cr(bijk_ppn_r, 1, neighbors_id_ijk[6], mpi_recv_reqs_kosumi[18]);
    }
    // ijk ppp
    if (neighbors_id_ijk[7] != -1){
        // send boundary layer to neighbor
        isend_cr(bijk_ppp_s, 1, neighbors_id_ijk[7], mpi_send_reqs_kosumi[19]);
        // receive boundary layer from neighbor
        irecv_cr(bijk_ppp_r, 1, neighbors_id_ijk[7], mpi_recv_reqs_kosumi[19]);
    }

    // wait for all communication to finish
    // ij nn
    if (neighbors_id_ij[0] != -1){
        wait_req(mpi_send_reqs_kosumi[0]);
        wait_req(mpi_recv_reqs_kosumi[0]);
    }
    // ij np
    if (neighbors_id_ij[1] != -1){
        wait_req(mpi_send_reqs_kosumi[1]);
        wait_req(mpi_recv_reqs_kosumi[1]);
    }
    // ij pn
    if (neighbors_id_ij[2] != -1){
        wait_req(mpi_send_reqs_kosumi[2]);
        wait_req(mpi_recv_reqs_kosumi[2]);
    }
    // ij pp
    if (neighbors_id_ij[3] != -1){
        wait_req(mpi_send_reqs_kosumi[3]);
        wait_req(mpi_recv_reqs_kosumi[3]);
    }
    // jk nn
    if (neighbors_id_jk[0] != -1){
        wait_req(mpi_send_reqs_kosumi[4]);
        wait_req(mpi_recv_reqs_kosumi[4]);
    }
    // jk np
    if (neighbors_id_jk[1] != -1){
        wait_req(mpi_send_reqs_kosumi[5]);
        wait_req(mpi_recv_reqs_kosumi[5]);
    }
    // jk pn
    if (neighbors_id_jk[2] != -1){
        wait_req(mpi_send_reqs_kosumi[6]);
        wait_req(mpi_recv_reqs_kosumi[6]);
    }
    // jk pp
    if (neighbors_id_jk[3] != -1){
        wait_req(mpi_send_reqs_kosumi[7]);
        wait_req(mpi_recv_reqs_kosumi[7]);
    }
    // ik nn
    if (neighbors_id_ik[0] != -1){
        wait_req(mpi_send_reqs_kosumi[8]);
        wait_req(mpi_recv_reqs_kosumi[8]);
    }
    // ik np
    if (neighbors_id_ik[1] != -1){
        wait_req(mpi_send_reqs_kosumi[9]);
        wait_req(mpi_recv_reqs_kosumi[9]);
    }
    // ik pn
    if (neighbors_id_ik[2] != -1){
        wait_req(mpi_send_reqs_kosumi[10]);
        wait_req(mpi_recv_reqs_kosumi[10]);
    }
    // ik pp
    if (neighbors_id_ik[3] != -1){
        wait_req(mpi_send_reqs_kosumi[11]);
        wait_req(mpi_recv_reqs_kosumi[11]);
    }
    // ijk nnn
    if (neighbors_id_ijk[0] != -1){
        wait_req(mpi_send_reqs_kosumi[12]);
        wait_req(mpi_recv_reqs_kosumi[12]);
    }
    // ijk nnp
    if (neighbors_id_ijk[1] != -1){
        wait_req(mpi_send_reqs_kosumi[13]);
        wait_req(mpi_recv_reqs_kosumi[13]);
    }
    // ijk npn
    if (neighbors_id_ijk[2] != -1){
        wait_req(mpi_send_reqs_kosumi[14]);
        wait_req(mpi_recv_reqs_kosumi[14]);
    }
    // ijk npp
    if (neighbors_id_ijk[3] != -1){
        wait_req(mpi_send_reqs_kosumi[15]);
        wait_req(mpi_recv_reqs_kosumi[15]);
    }
    // ijk pnn
    if (neighbors_id_ijk[4] != -1){
        wait_req(mpi_send_reqs_kosumi[16]);
        wait_req(mpi_recv_reqs_kosumi[16]);
    }
    // ijk pnp
    if (neighbors_id_ijk[5] != -1){
        wait_req(mpi_send_reqs_kosumi[17]);
        wait_req(mpi_recv_reqs_kosumi[17]);
    }
    // ijk ppn
    if (neighbors_id_ijk[6] != -1){
        wait_req(mpi_send_reqs_kosumi[18]);
        wait_req(mpi_recv_reqs_kosumi[18]);
    }
    // ijk ppp
    if (neighbors_id_ijk[7] != -1){
        wait_req(mpi_send_reqs_kosumi[19]);
        wait_req(mpi_recv_reqs_kosumi[19]);
    }

    assign_received_data_to_ghost_kosumi(arr);

}


void Grid::calc_T_plus_tau() {
    // calculate T_plus_tau
    for (int k_r = 0; k_r < loc_K; k_r++) {
        for (int j_lat = 0; j_lat < loc_J; j_lat++) {
            for (int i_lon = 0; i_lon < loc_I; i_lon++) {
                // T0*tau
                T_loc[I2V(i_lon,j_lat,k_r)] = T0v_loc[I2V(i_lon,j_lat,k_r)] * tau_loc[I2V(i_lon,j_lat,k_r)];

            }
        }
    }
}


void Grid::calc_residual() {
    for (int k_r = 0; k_r < loc_K; k_r++) {
        for (int j_lat = 0; j_lat < loc_J; j_lat++) {
            for (int i_lon = 0; i_lon < loc_I; i_lon++) {
                u_loc[I2V(i_lon,j_lat,k_r)] = u_loc[I2V(i_lon,j_lat,k_r)] - T_loc[I2V(i_lon,j_lat,k_r)];
            }
        }
    }
}


