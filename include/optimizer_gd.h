#ifndef OPTIMIZER_GD_H
#define OPTIMIZER_GD_H

#include "optimizer.h"  

class Optimizer_gd : public Optimizer {
public:
    Optimizer_gd();
    ~Optimizer_gd();

    void model_update(InputParams&, Grid&, IO_utils&, int&) override; 


    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    void processing_kernels(Grid&, InputParams&) override;


    // determine step length
    void determine_step_length() override;
};

#endif // OPTIMIZER_GD_H
