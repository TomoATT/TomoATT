#ifndef RECEIVER_H
#define RECEIVER_H

#include <vector>
#include "input_params.h"
#include "grid.h"

class Receiver {
public:
    Receiver();
    ~Receiver();

    void interpolate_and_store_arrival_times_at_rec_position(InputParams&, Grid&, const int);       // only for common receiver differential traveltime

    // adjoint source
    void calculate_adjoint_source(InputParams&, const int);
    // objective function and residual
    std::vector<CUSTOMREAL> calculate_obj_and_residual(InputParams&);
    // Gradient of traveltime
    void calculate_T_gradient(InputParams&, Grid&, const int);
    // initialize variables for source relocation
    void init_vars_src_reloc(InputParams&);
    // Gradient of objective function
    void calculate_grad_reloc(InputParams&, int);
    // objective function
    std::vector<CUSTOMREAL> calculate_obj_reloc(InputParams&, int);
    // update source location
    void update_source_location(InputParams&, Grid&);

private:
    CUSTOMREAL interpolate_travel_time(Grid&, InputParams&, int id_rec_att);
    std::vector<CUSTOMREAL> calculate_T_gradient_one_rec(Grid&, InputParams&, const int);
    bool check_if_receiver_is_in_this_subdomain(Grid&, const CUSTOMREAL&, const CUSTOMREAL&, const CUSTOMREAL&);
};

#endif // RECEIVER_H