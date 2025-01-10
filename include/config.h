#ifndef CONFIG_H
#define CONFIG_H

#include <math.h>
#include <mpi.h>
#include <string>
#include <vector>
#include <limits>
#include <iostream>
#include "version.h"

// custom floating point accuracy
#ifdef SINGLE_PRECISION
#define CUSTOMREAL float
#define MPI_CR MPI_FLOAT
#else
#define CUSTOMREAL double
#define MPI_CR MPI_DOUBLE
#endif


#define MPI_DUMMY_TAG 1000

// version
inline const std::string TOMOATT_VERSION = std::string(PROJECT_VERSION_MAJOR) +
                                   "." + std::string(PROJECT_VERSION_MINOR) +
                                   "." + std::string(PROJECT_VERSION_PATCH);

inline int loc_I, loc_J, loc_K;
inline int loc_I_vis, loc_J_vis, loc_K_vis;
inline int loc_I_excl_ghost, loc_J_excl_ghost, loc_K_excl_ghost;
inline int n_inv_grids;
inline int n_inv_I_loc, n_inv_J_loc, n_inv_K_loc;
inline int n_inv_I_loc_ani, n_inv_J_loc_ani, n_inv_K_loc_ani;

// 3d indices to 1d index
#define I2V(A,B,C) ((C)*loc_I*loc_J + (B)*loc_I + A)
inline void V2I(const int& ijk, int& i, int& j, int& k) {
    k = ijk / (loc_I * loc_J);
    j = (ijk - k * loc_I * loc_J) / loc_I;
    i = ijk - k * loc_I * loc_J - j * loc_I;
}

#define I2V_INV_GRIDS(A,B,C,D) ((D)*n_inv_I_loc*n_inv_J_loc*n_inv_K_loc + (C)*n_inv_I_loc*n_inv_J_loc + (B)*n_inv_I_loc + A)
#define I2V_INV_KNL(A,B,C)     ((C)*n_inv_I_loc*n_inv_J_loc + (B)*n_inv_I_loc + A)
#define I2V_INV_GRIDS_1DK(A,B)    ((B)*n_inv_K_loc + (A))
#define I2V_INV_GRIDS_2DJ(A,B,C)  ((C)*n_inv_J_loc*n_inv_K_loc + (B)*n_inv_J_loc + (A))
#define I2V_INV_GRIDS_2DI(A,B,C)  ((C)*n_inv_I_loc*n_inv_K_loc + (B)*n_inv_I_loc + (A))

#define I2V_INV_ANI_GRIDS(A,B,C,D) ((D)*n_inv_I_loc_ani*n_inv_J_loc_ani*n_inv_K_loc_ani + (C)*n_inv_I_loc_ani*n_inv_J_loc_ani + (B)*n_inv_I_loc_ani + A)
#define I2V_INV_ANI_KNL(A,B,C)     ((C)*n_inv_I_loc_ani*n_inv_J_loc_ani + (B)*n_inv_I_loc_ani + A)
#define I2V_INV_ANI_GRIDS_1DK(A,B)    ((B)*n_inv_K_loc_ani + (A))
#define I2V_INV_ANI_GRIDS_1DJ(A,B,C)  ((C)*n_inv_J_loc_ani*n_inv_K_loc_ani + (B)*n_inv_J_loc_ani + (A))
#define I2V_INV_ANI_GRIDS_1DI(A,B,C)  ((C)*n_inv_I_loc_ani*n_inv_K_loc_ani + (B)*n_inv_I_loc_ani + (A))

#define I2V_INV_GRIDS_1D_GENERIC(A,B,C_ninv) ((B)*C_ninv + (A))
#define I2V_INV_GRIDS_2D_GENERIC(A,B,C,D_ninv,E_inv)  ((C)*D_ninv*E_inv + (B)*D_ninv + (A))

#define I2V_EXCL_GHOST(A,B,C)  ((C)* loc_I_excl_ghost   * loc_J_excl_ghost +    (B)* loc_I_excl_ghost    + A)
#define I2V_VIS(A,B,C)         ((C)* loc_I_vis   * loc_J_vis +    (B)* loc_I_vis    + A)
#define I2V_3D(A,B,C)          ((C)* loc_I_excl_ghost*loc_J_excl_ghost + (B)* loc_I_excl_ghost + A)
#define I2V_ELM_CONN(A,B,C)    ((C)*(loc_I_vis-1)*(loc_J_vis-1) + (B)*(loc_I_vis-1) + A)
#define I2VLBFGS(A,B,C,D)      ((A)*loc_I*loc_J*loc_K + (D)*loc_I*loc_J + (C)*loc_I + B)

// 2d indices to 1d index
#define IJ2V(A,B,C) (C*loc_I*loc_J + (B)*loc_I + A)
#define JK2V(A,B,C) (C*loc_J*loc_K + (B)*loc_J + A)
#define IK2V(A,B,C) (C*loc_I*loc_K + (B)*loc_I + A)

inline const CUSTOMREAL eps    = 1e-12;
inline const CUSTOMREAL epsAdj = 1e-6;

inline bool isZero(CUSTOMREAL x) {
    return fabs(x) < eps;
}

inline bool isZeroAdj(CUSTOMREAL x) {
    return fabs(x) < epsAdj;
}

// constants
inline const CUSTOMREAL PI        = 3.14159265358979323846264338327950288;
inline const CUSTOMREAL DEG2RAD   = PI/180.0;
inline const CUSTOMREAL RAD2DEG   = 180.0/PI;
inline const CUSTOMREAL M2KM      = 0.001;
inline const CUSTOMREAL _0_CR     = 0.0;
inline const CUSTOMREAL _0_5_CR   = 0.5;
inline const CUSTOMREAL _1_CR     = 1.0;
inline const CUSTOMREAL _1_5_CR   = 1.5;
inline const CUSTOMREAL _2_CR     = 2.0;
inline const CUSTOMREAL _3_CR     = 3.0;
inline const CUSTOMREAL _4_CR     = 4.0;
inline const CUSTOMREAL _8_CR     = 8.0;
inline const CUSTOMREAL TAU_INITIAL_VAL = 1.0;
inline const CUSTOMREAL TAU_INF_VAL = 20.0;
inline const int        SWEEPING_COEFF  = 1.0; // coefficient for calculationg sigr/sigt/sigp for cpu

// ASCII output precision
typedef std::numeric_limits< CUSTOMREAL > dbl;
inline const int        ASCII_OUTPUT_PRECISION = dbl::max_digits10;

// radious of the earth in km
//inline CUSTOMREAL R_earth = 6378.1370;
inline CUSTOMREAL       R_earth        = 6371.0; // for compatibility with fortran code
inline const CUSTOMREAL GAMMA          = 0.0;
inline const CUSTOMREAL r_kermel_mask  = 40.0;
inline CUSTOMREAL       step_length_init = 0.01; // update step size limit
inline CUSTOMREAL       step_length_init_sc = 0.001; // update step size limit (for station correction)
inline CUSTOMREAL       step_length_decay = 0.9;
inline CUSTOMREAL       step_length_down = 0.5;
inline CUSTOMREAL       step_length_up = 1.2;
inline CUSTOMREAL       step_length_lbfgs;
inline int              step_method         = 1; // 0：modulate according to obj, 1: modulate according to gradient direction.
inline CUSTOMREAL       step_length_gradient_angle = 120.0;
inline const int        OBJ_DEFINED         = 0;
inline const int        GRADIENT_DEFINED    = 1;

// halve steping params
inline const CUSTOMREAL HALVE_STEP_RATIO = 0.7;
inline const CUSTOMREAL MAX_DIFF_RATIO_VOBJ = 0.8; // maximum difference ratio between vobj_t+1 and vobj_t
inline const CUSTOMREAL HALVE_STEP_RESTORAION_RATIO = 0.7; // no restoration if == HALVE_STEP_RATIO

// RUN MODE TYPE FLAG
inline const int ONLY_FORWARD        = 0;
inline const int DO_INVERSION        = 1;
inline const int SRC_RELOCATION      = 2;
inline const int INV_RELOC           = 3;
inline const int TELESEIS_PREPROCESS = 4; // hiden function

// SWEEPING TYPE FLAG
inline const int SWEEP_TYPE_LEGACY = 0;
inline const int SWEEP_TYPE_LEVEL  = 1;
inline      bool hybrid_stencil_order = false; // if true, code at first run 1st order, then change to 3rd order (Dong Cui 2020)

// STENCIL TYPE
inline const int NON_UPWIND = 0;
inline const int UPWIND     = 1;

// convert depth <-> radius
inline CUSTOMREAL depth2radius(CUSTOMREAL depth) {
    return R_earth - depth;
}
inline CUSTOMREAL radius2depth(CUSTOMREAL radius) {
    return R_earth - radius;
}

// parallel calculation parameters
// here we use so called "inline variables" for defining global variables
inline int ngrid_i     = 0; // number of grid points in i direction
inline int ngrid_j     = 0; // number of grid points in j direction
inline int ngrid_k     = 0; // number of grid points in k direction
//inline int ngrid_i_inv = 0; // number of inversion grid points in i direction # DUPLICATED with n_inv_I_loc as all the subdomains have the same number of inversion grid points
//inline int ngrid_j_inv = 0; // number of inversion grid points in j direction # DUPLICATED with n_inv_J_loc
//inline int ngrid_k_inv = 0; // number of inversion grid points in k direction # DUPLICATED with n_inv_K_loc
//inline int ngrid_i_inv_ani = 0; // number of inversion grid points in i direction for anisotropy # DUPLICATED with n_inv_I_loc_ani
//inline int ngrid_j_inv_ani = 0; // number of inversion grid points in j direction for anisotropy # DUPLICATED with n_inv_J_loc_ani
//inline int ngrid_k_inv_ani = 0; // number of inversion grid points in k direction for anisotropy # DUPLICATED with n_inv_K_loc_ani

// inversion grid type flags
#define INV_GRID_REGULAR 0
#define INV_GRID_FLEX    1
#define INV_GRID_TRAPE   2

// mpi parameters
inline int      world_nprocs;     // total number of processes (all groups)
inline int      world_rank;       // mpi rank of this process  (all groups)
inline int      sim_nprocs;       // number of processes in this simulation group
inline int      sim_rank;         // mpi rank of this process in this simulation group
inline int      inter_sim_rank;
inline int      sub_nprocs;       // total number of processes
inline int      sub_rank;         // mpi rank of this process
inline int      inter_sub_rank;   // mpi rank of this process in the inter-subdomain communicator
inline int      inter_sub_nprocs; // number of processes in the inter-subdomain communicator
inline int      nprocs;           // = n subdomains
inline int      myrank;           // = id subdomain
inline MPI_Comm sim_comm, inter_sim_comm, sub_comm, inter_sub_comm; // mpi communicator for simulation, inter-simulation, subdomain, and inter subdomains
inline int      n_sims           = 1; // number of mpi groups for simultaneous runs
inline int      n_procs_each_sim = 1; // number of processes in each simulation group
inline int      n_subdomains     = 1; // number of subdomains
inline int      n_subprocs       = 1; // number of sub processes in each subdomain
inline int      ndiv_i           = 1; // number of divisions in x direction
inline int      ndiv_j           = 1; // number of divisions in y direction
inline int      ndiv_k           = 1; // number of divisions in z direction
inline int      id_sim           = 0; // simultaneous run id  (not equal to src id)
inline int      id_subdomain     = 0; // subdomain id
inline bool     subdom_main      = false; // true if this process is main process in subdomain

// flags for explaining the process's role
inline bool proc_read_srcrec = false;  // true if this process is reading source file
inline bool proc_store_srcrec = false; // true if this process is storing srcrec file

// MNMN stop using these global variable for avoiding misleadings during the sources' iteration loop
//inline int      id_sim_src       = 0; // id of current target source
//inline std::string name_sim_src  = "unknown";   //name of current target source

// rule to distribute source to simultaneous group
inline int select_id_sim_for_src(const int& i_src, const int& n_sims){
    return i_src % n_sims;
}

// read input yaml file
inline std::string input_file="input_params.yaml";
// output directory name
inline std::string output_dir="./OUTPUT_FILES/";
// output file format
#define OUTPUT_FORMAT_HDF5 11111
#define OUTPUT_FORMAT_ASCII 22222
//#define OUTPUT_FORMAT_BINARY 33333 //#TODO: add
inline int output_format = OUTPUT_FORMAT_HDF5; // 0 - ascii, 1 - hdf5, 2 - binary

// HDF5 io mode (independent or collective)
#ifdef USE_HDF5_IO_COLLECTIVE
#define HDF5_IO_MODE H5FD_MPIO_COLLECTIVE
#else
#define HDF5_IO_MODE H5FD_MPIO_INDEPENDENT
#endif

// smooth parameters
inline int        smooth_method = 0; // 0: multi grid parametrization, 1: laplacian smoothing
inline CUSTOMREAL smooth_lp = 1.0;
inline CUSTOMREAL smooth_lt = 1.0;
inline CUSTOMREAL smooth_lr = 1.0;
inline CUSTOMREAL regul_lp  = 1.0;
inline CUSTOMREAL regul_lt  = 1.0;
inline CUSTOMREAL regul_lr  = 1.0;
inline const int  GRADIENT_DESCENT    = 0;
inline const int  HALVE_STEPPING_MODE = 1;
inline const int  LBFGS_MODE          = 2;
inline int        optim_method        = 0; // 0: gradient descent, 1: halve_stepping, 2: LBFGS
inline const CUSTOMREAL wolfe_c1     = 1e-4;
inline const CUSTOMREAL wolfe_c2     = 0.9;
inline const int        Mbfgs        = 5;            // number of gradients/models stored in memory
inline CUSTOMREAL       regularization_weight = 0.5; // regularization weight
inline int              max_sub_iterations    = 20;  // maximum number of sub-iterations
inline const CUSTOMREAL LBFGS_RELATIVE_step_length = 3.0; // relative step size for the second and later iteration
inline CUSTOMREAL       Kdensity_coe  = 0.0;          // a preconditioner \in [0,1], kernel -> kernel / pow(kernel density, Kdensity_coe)
// variables for test
inline bool if_test = false;

// verboose mode
inline bool if_verbose = false;
//inline bool if_verbose = true;

// if use gpu
inline bool use_gpu;

// total number of sources in the srcrec file
inline int nsrc_total = 0;

// flag if common receiver double difference data is used
inline bool src_pair_exists = false;


// weight of different typs of data
inline CUSTOMREAL abs_time_local_weight    = 1.0;    // weight of absolute traveltime data for local earthquake,                        default: 1.0
inline CUSTOMREAL cr_dif_time_local_weight = 1.0;    // weight of common receiver differential traveltime data for local earthquake,    default: 1.0
inline CUSTOMREAL cs_dif_time_local_weight = 1.0;    // weight of common source differential traveltime data for local earthquake,      default: 1.0    (not ready)
inline CUSTOMREAL teleseismic_weight       = 1.0;    // weight of teleseismic data                                                      default: 1.0    (not ready)
inline CUSTOMREAL abs_time_local_weight_reloc    = 1.0;    // weight of absolute traveltime data for local earthquake,                        default: 1.0
inline CUSTOMREAL cs_dif_time_local_weight_reloc = 1.0;    // weight of common source differential traveltime data for local earthquake,      default: 1.0
inline CUSTOMREAL cr_dif_time_local_weight_reloc = 1.0;    // weight of common receiver differential traveltime data for local earthquake,    default: 1.0
inline CUSTOMREAL teleseismic_weight_reloc       = 1.0;    // weight of teleseismic data                                                      default: 1.0    (not ready)

// misfit balance
inline bool       balance_data_weight      = false;    // add the weight to normalize the initial objective function of different types of data. 1 for yes and 0 for no
inline CUSTOMREAL total_abs_local_data_weight    = 0.0;
inline CUSTOMREAL total_cr_dif_local_data_weight = 0.0;
inline CUSTOMREAL total_cs_dif_local_data_weight = 0.0;
inline CUSTOMREAL total_teleseismic_data_weight  = 0.0;
inline bool       balance_data_weight_reloc      = false;    // add the weight to normalize the initial objective function of different types of data for relocation. 1 for yes and 0 for no
inline CUSTOMREAL total_abs_local_data_weight_reloc    = 0.0;
inline CUSTOMREAL total_cr_dif_local_data_weight_reloc = 0.0;
inline CUSTOMREAL total_cs_dif_local_data_weight_reloc = 0.0;
inline CUSTOMREAL total_teleseismic_data_weight_reloc  = 0.0;

// 2d solver parameters
// use fixed domain size for all 2d simulations
inline       CUSTOMREAL rmin_2d             = 3370.5;
inline       CUSTOMREAL rmax_2d             = 6471.5;
inline       CUSTOMREAL tmin_2d             = -5.0/180.0*PI;
inline       CUSTOMREAL tmax_2d             = 120.0/180.0*PI;
inline       CUSTOMREAL dr_2d               = 2.0;
inline       CUSTOMREAL dt_2d               = PI/1000.0;
inline const CUSTOMREAL TOL_2D_SOLVER       = 1e-4;
inline const CUSTOMREAL MAX_ITER_2D_SOLVER  = 4000;
inline const CUSTOMREAL SWEEPING_COEFF_TELE = 1.05;        // coefficient for calculationg sigr/sigt/sigp
inline const int        N_LAYER_SRC_BOUND   = 1;           // number of layers for source boundary
inline       CUSTOMREAL DIST_SRC_DDT        = 2.5*DEG2RAD; // distance threshold of two stations
inline const std::string OUTPUT_DIR_2D      = "/2D_TRAVEL_TIME_FIELD/"; // output directory for 2d solver

// earthquake relocation
inline CUSTOMREAL       step_length_src_reloc       = 2.0;  // step length for source relocation
inline CUSTOMREAL       step_length_decay_src_reloc = 0.9;
inline int              N_ITER_MAX_SRC_RELOC        = 501;  // max iteration for source location
inline CUSTOMREAL       TOL_SRC_RELOC               = 1e-3; // threshold of the norm of gradient for stopping single earthquake location
inline const CUSTOMREAL TOL_step_length             = 1e-4; // threshold of the max step size for stopping single earthquake location
inline CUSTOMREAL       rescaling_dep               = 10.0;
inline CUSTOMREAL       rescaling_lat               = 1.0;
inline CUSTOMREAL       rescaling_lon               = 1.0;
inline CUSTOMREAL       rescaling_ortime            = 0.5;
inline CUSTOMREAL       max_change_dep              = 10.0;
inline CUSTOMREAL       max_change_lat              = 1.0;
inline CUSTOMREAL       max_change_lon              = 1.0;
inline CUSTOMREAL       max_change_ortime           = 0.5;
inline bool             ortime_local_search         = true;
inline int              min_Ndata_reloc             = 4;    // if an earthquake is recorded by less than <min_Ndata> times, relocation is not allowed.

// inversion strategy parameters
inline int inv_mode                     = 1;
inline int model_update_N_iter          = 1;
inline int relocation_N_iter            = 1;
inline int max_loop_mode0               = 10;
inline int max_loop_mode1               = 10;

// INV MODE TYPE FLAG
inline const int ITERATIVE        = 0;
inline const int SIMULTANEOUS     = 1;

// source receiver weight calculation
inline CUSTOMREAL ref_value = 1.0; // reference value for source receiver weight calculation
inline std::string output_file_weight = "srcrec_weight.txt"; // output file name for source receiver weight calculation

#endif // CONFIG_H