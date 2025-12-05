#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include <memory>
#include "mpi_funcs.h"
#include "config.h"
#include "input_params.h"
#include "grid.h"
// #include "kernel.h"

class Optimizer {
public:
    Optimizer(InputParams& IP);
    virtual ~Optimizer();


    // model update function
    void model_update(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj, bool is_line_search); // virtual function, can be override in derived classes.

protected:

    bool need_write_model     = false;
    bool need_write_original_kernel    = false;


    // ---------------------------------------------------
    // ------------------ main function ------------------
    // ---------------------------------------------------

    // check kernel density
    void check_kernel_density(Grid&, InputParams&);

    // sum up kernels from all simulateous group (level 1)
    void sumup_kernels(Grid&);

    // write out kernels 
    void write_original_kernels(Grid&, InputParams& , IO_utils&, int&);

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    virtual void processing_kernels(Grid& grid, IO_utils& io, InputParams& IP, int& i_inv);

    // write out modified kernels (descent direction)
    void write_modified_kernels(Grid&, InputParams& , IO_utils&, int&);

    // determine step length
    void determine_step_length(Grid& grid, int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj, bool is_line_search);

    // set new model
    void set_new_model(Grid&, CUSTOMREAL);

    // write new model
    void write_new_model(Grid& grid, InputParams& IP, IO_utils& io, int& i_inv);

    // ---------------------------------------------------
    // ------------------ sub functions ------------------
    // ---------------------------------------------------

    // initialize and backup modified kernels
    void initialize_and_backup_modified_kernels(Grid& grid);

    // check kernel value range
    void check_kernel_value_range(Grid& grid);

    // calculate the angle between previous and current model update directions
    CUSTOMREAL direction_change_of_model_update(Grid& grid);

    // check write model and kernel condition
    bool is_write_model(InputParams& IP, int& i_inv);
    bool is_write_original_kernel(InputParams& IP, int& i_inv);
    bool is_write_modified_kernel(InputParams& IP, int& i_inv);

};

#endif // OPTIMIZER_H