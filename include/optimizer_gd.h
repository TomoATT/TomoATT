#ifndef OPTIMIZER_GD_H
#define OPTIMIZER_GD_H

#include "optimizer.h"  

class Optimizer_gd : public Optimizer {
public:
    Optimizer_gd();
    ~Optimizer_gd();

    // model update function (main function)
    void model_update(InputParams&, Grid&, IO_utils&, int&) override; 


private:

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    void processing_kernels(Grid&, InputParams&) override;

    // determine step length
    void determine_step_length() override;

    // initialize and backup modified kernels
    void initialize_and_backup_modified_kernels(Grid&);

    // check kernel value range
    void check_kernel_value_range(Grid&);
};

#endif // OPTIMIZER_GD_H
