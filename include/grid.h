#ifndef GRID_H
#define GRID_H

// to handle circular dependency
#pragma once
#include "grid.fwd.h"
#include "source.fwd.h"
#include "io.fwd.h"

#include <vector>
#include <array>
#include <algorithm>
#include <cstring>
#include <iomanip>

#include "utils.h"
#include "config.h"
#include "mpi_funcs.h"
#include "input_params.h"
#include "source.h"
#include "io.h"
#include "inv_grid.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

class Grid {
public:
    Grid(InputParams&, IO_utils&);
    ~Grid();
    //
    // setter
    //
    // set physical parameters
    void setup_grid_params(InputParams&, IO_utils&); // setup grid parameters
    // set factors
    void setup_factors(Source &);
    // calculate initial fields T0 T0r T0t T0p and initialize tau
    void initialize_fields(Source &, InputParams&);
    // calculate initial fields T0 T0r T0t T0p and initialize tau for teleseismic source
    void initialize_fields_teleseismic();
    // calculate L1 and Linf diff (sum of value change on the nodes)
    void calc_L1_and_Linf_diff(CUSTOMREAL&, CUSTOMREAL&);
    // calculate L1 and Linf diff for teleseismic source (sum of value change on the nodes)
    void calc_L1_and_Linf_diff_tele(CUSTOMREAL&, CUSTOMREAL&);
    // FOR TEST: calculate L1 and Linf error between analytic solution and current T
    void calc_L1_and_Linf_error(CUSTOMREAL&, CUSTOMREAL&);
    // calculate L1 and Linf diff for adjoint field
    void calc_L1_and_Linf_diff_adj(CUSTOMREAL&, CUSTOMREAL&);
    // calculate L1 and Linf diff for density of adjoint field
    void calc_L1_and_Linf_diff_adj_density(CUSTOMREAL&, CUSTOMREAL&);

    // send and receive data to/from other subdomains
    void send_recev_boundary_data(CUSTOMREAL*);
    void prepare_boundary_data_to_send(CUSTOMREAL*);
    void assign_received_data_to_ghost(CUSTOMREAL*);
    void send_recev_boundary_data_av(CUSTOMREAL*);
    void assign_received_data_to_ghost_av(CUSTOMREAL*);
    // send and receive data to/from the neighbor subdomains which contact on a line or point
    void send_recev_boundary_data_kosumi(CUSTOMREAL*);
    void prepare_boundary_data_to_send_kosumi(CUSTOMREAL*);
    void assign_received_data_to_ghost_kosumi(CUSTOMREAL*);

    // inversion methods
    InvGrid* inv_grid; // inversion grid definitions

    // check model discontinuity
    void check_velocity_discontinuity();

    void reinitialize_abcf();   // reinitialize factors
    void rejuvenate_abcf();     // reinitialize factors for earthquake relocation
    void initialize_kernels();  // fill 0 to kernels

    //
    // getters
    //

    int get_ngrid_total_excl_ghost(){return loc_nnodes;};
    int get_nelms_total_excl_ghost(){return loc_nelms;};
    int get_offset_nnodes()         {return offset_nnodes;};
    int get_offset_elms()           {return offset_nelms;};
    void get_offsets_3d(int* arr) {
        arr[0] = offset_k;
        arr[1] = offset_j;
        arr[2] = offset_i;
    };

    int get_ngrid_total_vis()  {return loc_nnodes_vis;};
    int get_nelms_total_vis()  {return loc_nelms_vis;};
    int get_offset_nnodes_vis(){return offset_nnodes_vis;};
    int get_offset_elms_vis()  {return offset_nelms_vis;};

    // return dimensions of node coordinates arra and connectivity array
    std::array<int,2> get_dim_nodes_coords() {return {offset_nnodes,3};};
    std::array<int,2> get_dim_elms_conn()    {return {offset_nelms,9};};

    // return node coordinates array and connectivity array
    int*        get_elms_conn()     {return elms_conn;};
    CUSTOMREAL* get_nodes_coords_x(){return x_loc_3d;}; // lon
    CUSTOMREAL* get_nodes_coords_y(){return y_loc_3d;}; // lat
    CUSTOMREAL* get_nodes_coords_z(){return z_loc_3d;}; // r
    CUSTOMREAL* get_nodes_coords_p(){return p_loc_3d;}; // lon
    CUSTOMREAL* get_nodes_coords_t(){return t_loc_3d;}; // lat
    CUSTOMREAL* get_nodes_coords_r(){return r_loc_3d;}; // r
    int*        get_proc_dump()     {return my_proc_dump;}; // array storing the process ids

    CUSTOMREAL* get_true_solution(){return get_array_for_vis(u_loc,     false);}; // true solution
    CUSTOMREAL* get_fun()          {return get_array_for_vis(fun_loc,   false);}; //
    CUSTOMREAL* get_xi()           {return get_array_for_vis(xi_loc,    false);}; //
    CUSTOMREAL* get_eta()          {return get_array_for_vis(eta_loc,   false);}; //
    CUSTOMREAL* get_a()            {return get_array_for_vis(fac_a_loc, false);}; //
    CUSTOMREAL* get_b()            {return get_array_for_vis(fac_b_loc, false);}; //
    CUSTOMREAL* get_c()            {return get_array_for_vis(fac_c_loc, false);}; //
    CUSTOMREAL* get_f()            {return get_array_for_vis(fac_f_loc, false);}; //
    CUSTOMREAL* get_vel()          {return get_array_for_vis(fun_loc,   true);}; // true velocity field
    CUSTOMREAL* get_T0v()          {return get_array_for_vis(T0v_loc,   false);}; // initial T0
    CUSTOMREAL* get_u()            {return get_array_for_vis(u_loc,     false);}; // current solution
    CUSTOMREAL* get_tau()          {return get_array_for_vis(tau_loc,   false);}; // current tau
    CUSTOMREAL* get_T()            {return get_array_for_vis(T_loc,     false);};
    CUSTOMREAL* get_residual()     { calc_residual();
                                     return get_array_for_vis(u_loc,   false);}; // calculate residual (T_loc over written!!)
    CUSTOMREAL* get_Tadj()         {return get_array_for_vis(Tadj_loc, false);}; // adjoint solution
    CUSTOMREAL* get_Ks()           {return get_array_for_vis(Ks_loc,   false);}; // Ks
    CUSTOMREAL* get_Kxi()          {return get_array_for_vis(Kxi_loc,         false);}; // Kxi
    CUSTOMREAL* get_Keta()         {return get_array_for_vis(Keta_loc,        false);}; // Keta
    CUSTOMREAL* get_Ks_density()   {return get_array_for_vis(Ks_density_loc,    false);}; // Ks_density
    CUSTOMREAL* get_Kxi_density()  {return get_array_for_vis(Kxi_density_loc,   false);}; // Kxi_density
    CUSTOMREAL* get_Keta_density() {return get_array_for_vis(Keta_density_loc,  false);}; // Keta_density
    CUSTOMREAL* get_Ks_update()    {return get_array_for_vis(Ks_update_loc,   false);}; // Ks_update
    CUSTOMREAL* get_Kxi_update()   {return get_array_for_vis(Kxi_update_loc,  false);}; // Kxi_update
    CUSTOMREAL* get_Keta_update()  {return get_array_for_vis(Keta_update_loc, false);}; // Keta_update
    CUSTOMREAL* get_Ks_density_update()     {return get_array_for_vis(Ks_density_update_loc,false);}; // Ks_density_update
    CUSTOMREAL* get_Kxi_density_update()    {return get_array_for_vis(Kxi_density_update_loc,false);}; // Kxi_density_update
    CUSTOMREAL* get_Keta_density_update()   {return get_array_for_vis(Keta_density_update_loc,false);}; // Keta_density_update

    CUSTOMREAL* get_Ks_descent_dir() {return get_array_for_vis(Ks_descent_dir_loc, false);}; // Ks_descent_dir
    CUSTOMREAL* get_Kxi_descent_dir(){return get_array_for_vis(Kxi_descent_dir_loc,false);}; // Kxi_descent_dir
    CUSTOMREAL* get_Keta_descent_dir(){return get_array_for_vis(Keta_descent_dir_loc,false);}; // Keta_descent_dir

    // get physical parameters
    CUSTOMREAL get_r_min()       {return r_min;};
    CUSTOMREAL get_r_max()       {return r_max;};
    CUSTOMREAL get_lat_min()     {return lat_min;};
    CUSTOMREAL get_lat_max()     {return lat_max;};
    CUSTOMREAL get_lon_min()     {return lon_min;};
    CUSTOMREAL get_lon_max()     {return lon_max;};
    CUSTOMREAL get_delta_lon()   {return dlon;};
    CUSTOMREAL get_delta_lat()   {return dlat;};
    CUSTOMREAL get_delta_r()     {return dr;};
    CUSTOMREAL get_r_min_loc_excl_gl()   {return r_loc_1d[k_start_loc];};
    CUSTOMREAL get_r_max_loc_excl_gl()   {return r_loc_1d[k_end_loc];};
    CUSTOMREAL get_lat_min_loc_excl_gl() {return t_loc_1d[j_start_loc];};
    CUSTOMREAL get_lat_max_loc_excl_gl() {return t_loc_1d[j_end_loc];};
    CUSTOMREAL get_lon_min_loc_excl_gl() {return p_loc_1d[i_start_loc];};
    CUSTOMREAL get_lon_max_loc_excl_gl() {return p_loc_1d[i_end_loc];};
    CUSTOMREAL get_r_min_loc()   {return r_loc_1d[0];};
    CUSTOMREAL get_r_max_loc()   {return r_loc_1d[loc_K-1];};
    CUSTOMREAL get_lat_min_loc() {return t_loc_1d[0];};
    CUSTOMREAL get_lat_max_loc() {return t_loc_1d[loc_J-1];};
    CUSTOMREAL get_lon_min_loc() {return p_loc_1d[0];};
    CUSTOMREAL get_lon_max_loc() {return p_loc_1d[loc_I-1];};
    CUSTOMREAL get_r_by_index(int i) {return r_loc_1d[i];};
    CUSTOMREAL get_lat_by_index(int j) {return t_loc_1d[j];};
    CUSTOMREAL get_lon_by_index(int i) {return p_loc_1d[i];};

    int get_offset_i()           {return offset_i-i_start_loc;}; // return offset index (global index of the first node)
    int get_offset_j()           {return offset_j-j_start_loc;};
    int get_offset_k()           {return offset_k-k_start_loc;};
    int get_offset_i_excl_gl()           {return offset_i;}; // return offset index
    int get_offset_j_excl_gl()           {return offset_j;};
    int get_offset_k_excl_gl()           {return offset_k;};

    // return index of the first node excluding ghost layer
    int get_k_start_loc()        {return k_start_loc;};
    int get_k_end_loc()          {return k_end_loc;};
    int get_j_start_loc()        {return j_start_loc;};
    int get_j_end_loc()          {return j_end_loc;};
    int get_i_start_loc()        {return i_start_loc;};
    int get_i_end_loc()          {return i_end_loc;};

    // copy tau to tau old
    void tau2tau_old();
    // copy T to tau old
    void T2tau_old();
    // copy tau to Tadj
    void update_Tadj();
    // copy tau to Tadj
    void update_Tadj_density();
    // back up fun xi eta
    void back_up_fun_xi_eta_bcf();
    // restore fun xi eta
    void restore_fun_xi_eta_bcf();

    // write out inversion grid file
    void write_inversion_grid_file();

private:

    //
    // member variables
    //
    // i: longitude, phi,
    // j: latitude,  theta,
    // k: radius,    r

    // starting node id for each direction expect the boundary
    int i_start_loc=0, j_start_loc=0, k_start_loc=0;
    // end node id for each direction expect the boundary
    int i_end_loc=0, j_end_loc=0, k_end_loc=0;

    // neighbors domain_id (-1 if no neighbor)
    // order of directions: -i,+i,-j,+j,-k,+k
    std::vector<int> neighbors_id{-1,-1,-1,-1,-1,-1};
    // neighbors domain_id line and point contact
    // ij plane nn, np, pn, pp
    std::vector<int> neighbors_id_ij{-1,-1,-1,-1};
    // jk plane nn, np, pn, pp
    std::vector<int> neighbors_id_jk{-1,-1,-1,-1};
    // ik plane  nn, np, pn, pp
    std::vector<int> neighbors_id_ik{-1,-1,-1,-1};
    // ijk plane nnn, nnp, npn, npp, pnn, pnp, ppn, ppp
    std::vector<int> neighbors_id_ijk{-1,-1,-1,-1,-1,-1,-1,-1};

    // layer id of this domain
    int domain_i, domain_j, domain_k;

    // number of points on each boundary surface
    int n_grid_bound_i, n_grid_bound_j, n_grid_bound_k;

    // mpi parameters
    int offset_nnodes=0, offset_nelms=0;
    int offset_i=0, offset_j=0, offset_k=0; // offset of this domain in the global grid

    // number of nodes and elements on each subdomain
    int loc_nnodes, loc_nelms;
    int *nnodes, *nelms;

    // arrays for visualization
    //int loc_I_vis, loc_J_vis, loc_K_vis;
    int i_start_vis, j_start_vis, k_start_vis;
    int i_end_vis, j_end_vis, k_end_vis;
    int loc_nnodes_vis, loc_nelms_vis;
    int *nnodes_vis, *nelms_vis;
    int offset_nnodes_vis=0, offset_nelms_vis=0;

public:
    // 3d arrays
    CUSTOMREAL *xi_loc;    // local xi
    CUSTOMREAL *eta_loc;   // local eta
    CUSTOMREAL *zeta_loc;  // local zeta
    CUSTOMREAL *fac_a_loc; // factor a
    CUSTOMREAL *fac_b_loc; // factor b
    CUSTOMREAL *fac_c_loc; // factor c
    CUSTOMREAL *fac_f_loc; // factor f
    CUSTOMREAL *fun_loc;
    CUSTOMREAL *T_loc;
    CUSTOMREAL *T0v_loc, *T0r_loc, *T0p_loc, *T0t_loc;
    CUSTOMREAL *tau_loc;
    CUSTOMREAL *tau_old_loc;
    bool       *is_changed;
    // for inversion backup
    CUSTOMREAL *fun_loc_back;
    CUSTOMREAL *xi_loc_back;
    CUSTOMREAL *eta_loc_back;
    CUSTOMREAL *fac_b_loc_back;
    CUSTOMREAL *fac_c_loc_back;
    CUSTOMREAL *fac_f_loc_back;
    // for lbfgs
    CUSTOMREAL *Ks_grad_store_loc, *Keta_grad_store_loc, *Kxi_grad_store_loc;
    CUSTOMREAL *Ks_model_store_loc, *Keta_model_store_loc, *Kxi_model_store_loc;
    CUSTOMREAL *Ks_descent_dir_loc, *Keta_descent_dir_loc, *Kxi_descent_dir_loc;
    CUSTOMREAL *fun_regularization_penalty_loc, *eta_regularization_penalty_loc, *xi_regularization_penalty_loc;
    CUSTOMREAL *fun_gradient_regularization_penalty_loc, *eta_gradient_regularization_penalty_loc, *xi_gradient_regularization_penalty_loc;
    CUSTOMREAL *fun_prior_loc, *eta_prior_loc, *xi_prior_loc; // *zeta_prior_loc; TODO
    // tmp array for file IO
    CUSTOMREAL *vis_data;

private:
    // windows for shm arrays
    MPI_Win win_fac_a_loc, win_fac_b_loc, win_fac_c_loc, win_fac_f_loc;
    MPI_Win win_T0r_loc, win_T0p_loc, win_T0t_loc, win_T0v_loc;
    MPI_Win win_tau_loc, win_fun_loc;
    MPI_Win win_is_changed;
    MPI_Win win_T_loc, win_tau_old_loc;
    MPI_Win win_xi_loc, win_eta_loc, win_zeta_loc;
    MPI_Win win_r_loc_1d, win_t_loc_1d, win_p_loc_1d;

    CUSTOMREAL *x_loc_3d;     // local (lon) x (global position)
    CUSTOMREAL *y_loc_3d;     // local (lat) y (global position)
    CUSTOMREAL *z_loc_3d;     // local (r  ) z (global position)
    CUSTOMREAL *p_loc_3d;     // local lon (x) (global position)
    CUSTOMREAL *t_loc_3d;     // local lat (y) (global position)
    CUSTOMREAL *r_loc_3d;     // local r   (z) (global position)
    int        *elms_conn;    // connectivity array
    int        *my_proc_dump; // dump process id for each node  DEBUG
public:
    // 1d arrays for coordinates, storing only subdomain's local coordinates
    CUSTOMREAL *r_loc_1d; // radius z    in kilo meter
    CUSTOMREAL *t_loc_1d; // theta lat y in radian
    CUSTOMREAL *p_loc_1d; // phi lon x   in radian
private:
    // arrays for inversion
    // TODO: check if those arrays can use shm
    bool inverse_flag = false;
    //int n_inv_grids; // in config.h
    //int n_inv_I_loc, n_inv_J_loc, n_inv_K_loc; // in config.h
public:
    CUSTOMREAL *Ks_loc;
    CUSTOMREAL *Kxi_loc;
    CUSTOMREAL *Keta_loc;
    CUSTOMREAL *Ks_density_loc;
    CUSTOMREAL *Kxi_density_loc;
    CUSTOMREAL *Keta_density_loc;
    CUSTOMREAL *Tadj_loc; // timetable for adjoint source
    CUSTOMREAL *Tadj_density_loc; // timetable for density of adjoint source
    CUSTOMREAL *Ks_inv_loc;
    CUSTOMREAL *Kxi_inv_loc;
    CUSTOMREAL *Keta_inv_loc;
    CUSTOMREAL *Ks_density_inv_loc;
    CUSTOMREAL *Kxi_density_inv_loc;
    CUSTOMREAL *Keta_density_inv_loc;
    // model update para
    CUSTOMREAL *Ks_update_loc;
    CUSTOMREAL *Kxi_update_loc;
    CUSTOMREAL *Keta_update_loc;
    CUSTOMREAL *Ks_density_update_loc;
    CUSTOMREAL *Kxi_density_update_loc;
    CUSTOMREAL *Keta_density_update_loc;
    // model update para of the previous step
    CUSTOMREAL *Ks_update_loc_previous;
    CUSTOMREAL *Kxi_update_loc_previous;
    CUSTOMREAL *Keta_update_loc_previous;
private:
    MPI_Win     win_Tadj_loc;
    MPI_Win     win_Tadj_density_loc;
    // boundary layer for mpi communication
    // storing not the value but the pointers for the elements of arrays
    // to send
    CUSTOMREAL *bin_s; // boundary i axis negative
    CUSTOMREAL *bip_s; // boundary i axis positive
    CUSTOMREAL *bjn_s; // boundary j axis negative
    CUSTOMREAL *bjp_s; // boundary j axis positive
    CUSTOMREAL *bkn_s; // boundary k axis negative
    CUSTOMREAL *bkp_s; // boundary k axis positive
    // to receive
    CUSTOMREAL *bin_r; // boundary i axis negative
    CUSTOMREAL *bip_r; // boundary i axis positive
    CUSTOMREAL *bjn_r; // boundary j axis negative
    CUSTOMREAL *bjp_r; // boundary j axis positive
    CUSTOMREAL *bkn_r; // boundary k axis negative
    CUSTOMREAL *bkp_r; // boundary k axis positive

    // kosumi boundaries (touching on lines and points)
    CUSTOMREAL *bij_pp_s, *bij_pp_r;
    CUSTOMREAL *bij_pn_s, *bij_pn_r;
    CUSTOMREAL *bij_np_s, *bij_np_r;
    CUSTOMREAL *bij_nn_s, *bij_nn_r;
    CUSTOMREAL *bjk_pp_s, *bjk_pp_r;
    CUSTOMREAL *bjk_pn_s, *bjk_pn_r;
    CUSTOMREAL *bjk_np_s, *bjk_np_r;
    CUSTOMREAL *bjk_nn_s, *bjk_nn_r;
    CUSTOMREAL *bik_pp_s, *bik_pp_r;
    CUSTOMREAL *bik_pn_s, *bik_pn_r;
    CUSTOMREAL *bik_np_s, *bik_np_r;
    CUSTOMREAL *bik_nn_s, *bik_nn_r;
    CUSTOMREAL *bijk_ppp_s,*bijk_ppp_r;
    CUSTOMREAL *bijk_ppn_s,*bijk_ppn_r;
    CUSTOMREAL *bijk_pnn_s,*bijk_pnn_r;
    CUSTOMREAL *bijk_npp_s,*bijk_npp_r;
    CUSTOMREAL *bijk_npn_s,*bijk_npn_r;
    CUSTOMREAL *bijk_nnp_s,*bijk_nnp_r;
    CUSTOMREAL *bijk_nnn_s,*bijk_nnn_r;
    CUSTOMREAL *bijk_pnp_s,*bijk_pnp_r;

    // store MPI_Request for sending and receiving
    MPI_Request *mpi_send_reqs;
    MPI_Request *mpi_recv_reqs;
    MPI_Request *mpi_send_reqs_kosumi;
    MPI_Request *mpi_recv_reqs_kosumi;
    //
    // domain definition
    //
    CUSTOMREAL r_min;   // minimum radius of global domain
    CUSTOMREAL r_max;   // maximum radius of global domain
    CUSTOMREAL lat_min; // minimum latitude of global domain
    CUSTOMREAL lat_max; // maximum latitude of global domain
    CUSTOMREAL lon_min; // minimum longitude of global domain
    CUSTOMREAL lon_max; // maximum longitude of global domain
public:
    CUSTOMREAL dr  ;
    CUSTOMREAL dlat,dt;
    CUSTOMREAL dlon,dp;
private:

    //
    // members for test
    //
    CUSTOMREAL *u_loc;    // true solution # TODO: erase for no testing
    //CUSTOMREAL *velo_loc; // velocity field, # TODO: use this for storing an intial model
    // anisotropic factors
    CUSTOMREAL a0, b0, c0, f0, fun0;

    CUSTOMREAL source_width;
    //
    // member functions
    //
    void init_decomposition(InputParams&);   // initialize domain decomposition
    void memory_allocation();                // allocate memory for arrays
    void memory_deallocation();              // deallocate memory from arrays
    void shm_memory_allocation();            // allocate memory for shared memory
    void shm_memory_deallocation();          // deallocate memory from shared memory

    CUSTOMREAL *get_array_for_vis(CUSTOMREAL *arr, bool);  // get array for visualization
public:
    void get_array_for_3d_output(const CUSTOMREAL *arr_in, CUSTOMREAL* arr_out, bool inverse_field);  // get array for 3d output
    void        set_array_from_vis(CUSTOMREAL *arr); // set vis array to local array

    // finalize the Time table
    void calc_T_plus_tau();
private:
    // check difference between true solution and computed solution
    void calc_residual();

    //
    // utilities
    //
public:
    bool i_first(){ return neighbors_id[0] == -1;}
    bool j_first(){ return neighbors_id[2] == -1;}
    bool k_first(){ return neighbors_id[4] == -1;}
    bool i_last(){  return neighbors_id[1] == -1;}
    bool j_last(){  return neighbors_id[3] == -1;}
    bool k_last(){  return neighbors_id[5] == -1;}
private:
    // number of the layers for ghost nodes
    const int n_ghost_layers = 1;
    // vtk format cell type
    const int cell_type = 9;


};

#endif // GRID_H