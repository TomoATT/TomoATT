#ifndef OPTIMIZER_GD_H
#define OPTIMIZER_GD_H

#include "optimizer.h"  

class Optimizer_gd : public Optimizer {
public:
    Optimizer_gd(InputParams& IP);
    ~Optimizer_gd();

private:

    // ---------------------------------------------------------
    // ------------------ specified functions ------------------
    // ---------------------------------------------------------

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    void processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv) override;
    
};

#endif // OPTIMIZER_GD_H
