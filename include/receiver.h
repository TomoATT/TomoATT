#ifndef RECEIVER_H
#define RECEIVER_H

#include <vector>
#include "input_params.h"
#include "grid.h"

class Receiver {
public:
    Receiver();
    ~Receiver();

    void calculate_arrival_time(InputParams&, Grid&);
    void interpolate_and_store_arrival_times_at_rec_position(InputParams&, Grid&, const std::string&);       // only for common receiver differential traveltime

    // adjoint source
    void calculate_adjoint_source(InputParams&, const std::string&);
    // objective function and residual
    std::vector<CUSTOMREAL> calculate_obj_and_residual(InputParams&);
    // teleseismic source
    // CUSTOMREAL calculate_adjoint_source_teleseismic(InputParams&, const std::string&);
    // Gradient of traveltime
    void calculate_T_gradient(InputParams&, Grid&, const std::string&);
    // initialize variables for source relocation
    void init_vars_src_reloc(InputParams&);
    // calculate gradient of objective function with respect to ortime // approximated optimal origin time
    // void calculate_grad_obj_tau_reloc(InputParams&, const std::string&); // void calculate_optimal_origin_time(InputParams&, const std::string&); // combined in calculate_grad_obj_src_reloc 
    // divide optimal origin time by summed weight
    void divide_optimal_origin_time_by_summed_weight(InputParams&);
    // Gradient of objective function
    void calculate_grad_obj_src_reloc(InputParams&, const std::string&);
    // objective function
    void calculate_obj_reloc(InputParams&, int);
    // update source location
    void update_source_location(InputParams&, Grid&);


private:
    CUSTOMREAL interpolate_travel_time(Grid&, InputParams&, std::string, std::string);
    void calculate_T_gradient_one_rec(Grid&, SrcRecInfo&, CUSTOMREAL*);
};

#endif // RECEIVER_H