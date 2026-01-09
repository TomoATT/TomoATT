#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include <memory>
#include "mpi_funcs.h"
#include "config.h"
#include "input_params.h"
#include "grid.h"
#include "io.h"
#include "main_routines_inversion_mode.h"
#include "kernel_postprocessing.h"

class Optimizer {
public:
    Optimizer(InputParams& IP);
    virtual ~Optimizer();


    // model update function
    std::vector<CUSTOMREAL> model_update(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj, bool is_line_search); // virtual function, can be override in derived classes.

protected:

    bool need_write_model     = false;
    bool need_write_original_kernel    = false;

    int n_total_loc_grid_points;    // number of local grid points
    std::vector<CUSTOMREAL> fun_loc_backup;
    std::vector<CUSTOMREAL> xi_loc_backup;
    std::vector<CUSTOMREAL> eta_loc_backup;

    CUSTOMREAL alpha;            // step length tried in line search

    // line search bounds
    CUSTOMREAL alpha_R;                 // upper bound of step length
    CUSTOMREAL alpha_L;                 // lower bound of step length

    // ---------------------------------------------------
    // ------------------ main function ------------------
    // ---------------------------------------------------

    // write out kernels 
    void write_original_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv);

    // smooth kernels (multigrid) + kernel normalization (kernel density normalization)
    virtual void processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv);

    // write out modified kernels (descent direction)
    void write_modified_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv);

    // determine step length (original method step-size controlled)
    void determine_step_length_controlled(InputParams& IP, Grid& grid, int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj);

    // determine step length (line search method)
    std::vector<CUSTOMREAL> determine_step_length_line_search(InputParams& IP, Grid& grid, IO_utils& io, int i_inv, CUSTOMREAL& v_obj_inout);

    // set new model
    void set_new_model(InputParams& IP, Grid& grid, CUSTOMREAL step_length);

    // write new model
    void write_new_model(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv);

    // ---------------------------------------------------
    // ------------------ sub functions ------------------
    // ---------------------------------------------------

    // initialize and backup modified kernels
    void initialize_and_backup_modified_kernels(Grid& grid);

    // check kernel value range
    void check_kernel_value_range(Grid& grid);

    // check write model and kernel condition
    bool is_write_model(InputParams& IP, int& i_inv);
    bool is_write_kernel(InputParams& IP, int& i_inv);

    // vector dot product
    CUSTOMREAL grid_value_dot_product(CUSTOMREAL* vec1, CUSTOMREAL* vec2, int n);

    // evaluate line search performance
    virtual bool check_conditions_for_line_search(InputParams& IP, Grid& grid, int sub_iter, int quit_sub_iter, CUSTOMREAL v_obj_inout, CUSTOMREAL v_obj_try){return false;};

  
};

#endif // OPTIMIZER_H