#ifndef SOURCE_H
#define SOURCE_H

// to handle circular dependency
#pragma once
#include "grid.fwd.h"
#include "source.fwd.h"

#include "input_params.h"
#include "grid.h"
#include "config.h"
#include "mpi_funcs.h"

class Source {
public:
    Source(){};
    ~Source();

    // set source information
    void set_source_position(InputParams &, Grid &, bool&, const int, bool for_2d_solver=false);
    //
    // getters
    //
    CUSTOMREAL get_ds_lat(){return dis_src_lat;};
    CUSTOMREAL get_ds_lon(){return dis_src_lon;};
    CUSTOMREAL get_ds_r  (){return dis_src_r;};
    CUSTOMREAL get_src_r(){return src_r;};   // radius
    CUSTOMREAL get_src_t(){return src_lat;}; // radian
    CUSTOMREAL get_src_p(){return src_lon;}; // radian
    CUSTOMREAL get_src_dep(){return radius2depth(src_r);}; // km

    //
    // parallel getters
    //
    CUSTOMREAL get_fac_at_source(CUSTOMREAL*, bool check=false);
    CUSTOMREAL get_fac_at_point(CUSTOMREAL*, int, int, int);

private:
    // positions
    CUSTOMREAL src_lat;
    CUSTOMREAL src_lon;
    CUSTOMREAL src_r;

    // discretize source position ids (LOCAL)
    int i_src_loc;
    int j_src_loc;
    int k_src_loc;

    // discretized source position
    CUSTOMREAL dis_src_lat;
    CUSTOMREAL dis_src_lon;
    CUSTOMREAL dis_src_r;

    CUSTOMREAL dis_src_err_r;   // r1
    CUSTOMREAL dis_src_err_lat; // r2
    CUSTOMREAL dis_src_err_lon; // r3

    // if source is in this subdomain
    bool is_in_subdomain = false;
    int  src_rank;
    bool *src_flags;
    int n_dom_src = 0;

    // position error by discretization for each direction
    CUSTOMREAL error_lon, error_lat, error_r;

    // grid gaps
    CUSTOMREAL delta_lon, delta_lat, delta_r;
};

#endif // SOURCE_H