#ifndef OPTIMIZER_GD_H
#define OPTIMIZER_GD_H

#include "optimizer.h"  

class Optimizer_gd : public Optimizer {
public:
    Optimizer_gd(InputParams& IP);
    ~Optimizer_gd();

private:

    // for line search
    std::vector<CUSTOMREAL> alpha_sub_iter;     // store tried step lengths
    std::vector<CUSTOMREAL> v_obj_sub_iter;     // store objective function values at tried step lengths
 

    // ---------------------------------------------------------
    // ------------------ specified functions ------------------
    // ---------------------------------------------------------

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    void processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv) override;
    
    // evaluate line search performance
    bool check_conditions_for_line_search(InputParams& IP, Grid& grid, int sub_iter, int quit_sub_iter, CUSTOMREAL v_obj_inout, CUSTOMREAL v_obj_try) override;

};

#endif // OPTIMIZER_GD_H
