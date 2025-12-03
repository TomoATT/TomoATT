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
    Optimizer();
    virtual ~Optimizer();

    // model update function
    virtual void model_update(InputParams&, Grid&, IO_utils&, int&){}; // virtual function, can be override in derived classes.

protected:

    // check kernel density
    void check_kernel_density(Grid&, InputParams&);

    // sum up kernels from all simulateous group (level 1)
    void sumup_kernels(Grid&);

    // write out kernels 
    void write_original_kernels(Grid&, InputParams& , IO_utils&, int&);

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    virtual void processing_kernels(Grid&, InputParams&){};

    // write out modified kernels (descent direction)
    void write_modified_kernels(Grid&, InputParams& , IO_utils&, int&);

    // determine step length
    virtual void determine_step_length(){};

    // set new model
    void set_new_model(Grid&, CUSTOMREAL);

};

#endif // OPTIMIZER_H