#ifndef RECEIVER_H
#define RECEIVER_H

#include <vector>
#include "input_params.h"
#include "grid.h"

class Receiver {
public:
    Receiver();
    ~Receiver();

    void interpolate_and_store_arrival_times_at_rec_position(InputParams&, Grid&, const std::string&);       // only for common receiver differential traveltime

    // adjoint source
    void calculate_adjoint_source(InputParams&, const std::string&);
    // adjoint source for a specific reflected phase
    void calculate_adjoint_source_for_phase(InputParams&, const std::string& name_sim_src, const std::string& phase_name);
    // objective function and residual
    std::vector<CUSTOMREAL> calculate_obj_and_residual(InputParams&);
    // Gradient of traveltime
    void calculate_T_gradient(InputParams&, Grid&, const std::string&);
    // initialize variables for source relocation
    void init_vars_src_reloc(InputParams&);
    // Gradient of objective function
    void calculate_grad_obj_src_reloc(InputParams&, const std::string&);
    // objective function
    std::vector<CUSTOMREAL> calculate_obj_reloc(InputParams&, int);
    // update source location
    void update_source_location(InputParams&, Grid&);

private:
    CUSTOMREAL interpolate_travel_time(Grid&, InputParams&, std::string, std::string);
    void calculate_T_gradient_one_rec(Grid&, InputParams&, std::string, CUSTOMREAL*);
    bool check_if_receiver_is_in_this_subdomain(Grid&, const CUSTOMREAL&, const CUSTOMREAL&, const CUSTOMREAL&);
};

#endif // RECEIVER_H