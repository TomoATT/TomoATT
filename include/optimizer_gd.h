#ifndef OPTIMIZER_GD_H
#define OPTIMIZER_GD_H

#include "optimizer.h"  

class Optimizer_gd : public Optimizer {
public:
    Optimizer_gd();
    ~Optimizer_gd();

private:

    // ---------------------------------------------------------
    // ------------------ specified functions ------------------
    // ---------------------------------------------------------

    // smooth kernels (multigrid or XXX (to do)) + kernel normalization (kernel density normalization, or XXX (to do))
    void processing_kernels(Grid&, InputParams&) override;

    // determine step length
    // void determine_step_length(int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj) override;
    void determine_step_length(Grid& grid, int i_inv, CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj) override;


    // ---------------------------------------------------
    // ------------------ sub functions ------------------
    // ---------------------------------------------------

    // initialize and backup modified kernels
    void initialize_and_backup_modified_kernels(Grid&);

    // check kernel value range
    void check_kernel_value_range(Grid&);

    // calculate the angle between previous and current model update directions
    CUSTOMREAL direction_change_of_model_update(Grid&);
};

#endif // OPTIMIZER_GD_H
