#include "inv_grid.h"


//
// utilities for inversion grid setup
//

#define ON_R 0
#define ON_T 1
#define ON_P 2


InvGrid1dBase inv_1d_grid_selector(InputParams& IP, const int id_axis, const int n_points) {
    if (id_axis == ON_R)
        return InvGrid1d(IP, n_points, IP.get_dep_inv()); // regular grid for radius
    else if (id_axis == ON_T)
        return InvGrid1d(IP, n_points, IP.get_lat_inv(), IP.get_trapezoid(), n_inv_K_loc, IP.get_dep_inv()); // regular grid for lat
    else if (id_axis == ON_P)
        return InvGrid1d(IP, n_points, IP.get_lon_inv(), IP.get_trapezoid(), n_inv_K_loc, IP.get_dep_inv()); // regular grid for lat
    else {
        std::cout << "unknown id_axis" << std::endl;
        exit(1);
    }
}


InvGrid1dBase inv_1d_grid_selector_ani(InputParams& IP, const int id_axis, const int n_points) {
    if (id_axis == ON_R)
        return InvGrid1d(IP, n_points, IP.get_dep_inv_ani()); // regular grid for radius
    else if (id_axis == ON_T)
        return InvGrid1d(IP, n_points, IP.get_lat_inv_ani(), IP.get_trapezoid_ani(), n_inv_K_loc_ani, IP.get_dep_inv_ani()); // regular grid for lat
    else if (id_axis == ON_P)
        return InvGrid1d(IP, n_points, IP.get_lon_inv_ani(), IP.get_trapezoid_ani(), n_inv_K_loc_ani, IP.get_dep_inv_ani()); // regular grid for lat
    else {
        std::cout << "unknown id_axis" << std::endl;
        exit(1);
    }
}


//
// Inversion grid class definition
//


InvGrid::InvGrid(InputParams& IP) {

    // check the inversion grid 
    IP.check_inv_grid();

    // read the parameters and store them in this class
    get_inv_grid_params(IP);

    // create 1D inversion grid for each dimension.
    // t and p grids are not 1D but 2D grids for a trapezoidal definition of the grid.

    // r (radius) grid
    r = inv_1d_grid_selector(IP, ON_R, n_inv_K_loc);
    // t (latitude) grid
    t = inv_1d_grid_selector(IP, ON_T, n_inv_J_loc);
    // p (longitude) grid
    p = inv_1d_grid_selector(IP, ON_P, n_inv_I_loc);

    // inversion grid for anisotropic parameters
    if (IP.get_invgrid_ani()){
        // r (radius) grid
        r_ani = inv_1d_grid_selector_ani(IP, ON_R, n_inv_K_loc_ani);
        // t (latitude) grid
        t_ani = inv_1d_grid_selector_ani(IP, ON_T, n_inv_J_loc_ani);
        // p (longitude) grid
        p_ani = inv_1d_grid_selector_ani(IP, ON_P, n_inv_I_loc_ani);
    } else {
        // use the same inversion grid with slowessw
        r_ani = r;
        t_ani = t;
        p_ani = p;
    }
}


InvGrid::~InvGrid() {
}

void InvGrid::get_inv_grid_params(InputParams& IP) {

    n_inv_grids = IP.get_n_inversion_grid();

    // inversion grid for velocity
    n_inv_K_loc = IP.get_n_inv_r_flex();
    n_inv_J_loc = IP.get_n_inv_t_flex();
    n_inv_I_loc = IP.get_n_inv_p_flex();

    // inversion grid for anisotropy
    if(IP.get_invgrid_ani()){
        n_inv_K_loc_ani = IP.get_n_inv_r_flex_ani();
        n_inv_J_loc_ani = IP.get_n_inv_t_flex_ani();
        n_inv_I_loc_ani = IP.get_n_inv_p_flex_ani();
    } else {
        // use the same inversion grid with slowessw
        n_inv_I_loc_ani = n_inv_I_loc;
        n_inv_J_loc_ani = n_inv_J_loc;
        n_inv_K_loc_ani = n_inv_K_loc;
    }
}


void InvGrid::write_inversion_grid_to_file(){
    if(world_rank == 0){
        std::ofstream ofs;

        std::string inversion_grid_file_out = output_dir + "/inversion_grid.txt";
        ofs.open(inversion_grid_file_out);

        // inversion grid of velocity
        for(int l = 0; l < n_inv_grids; l++){
            ofs << l << " " << n_inv_K_loc << " " << n_inv_J_loc << " " << n_inv_I_loc << std::endl;    // number of ivnersion grid

            for(int k = 0; k < n_inv_K_loc; k++){
                ofs << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ')
                    << radius2depth(r.arr[I2V_INV_GRIDS_1DK(k,l)]) << " ";
                ofs << std::endl;
                for(int j =0; j < n_inv_J_loc; j++)
                    ofs << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ')
                        << t.arr[I2V_INV_GRIDS_2DJ(j,k,l)]*RAD2DEG << " ";
                ofs << std::endl;
                for(int i =0; i < n_inv_I_loc; i++)
                    ofs << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ')
                        << p.arr[I2V_INV_GRIDS_2DI(i,k,l)]*RAD2DEG << " ";
                ofs << std::endl;
            }
        }
        // inversion grid of anisotropy
        for(int l = 0; l < n_inv_grids; l++){
            ofs << l << " " << n_inv_K_loc_ani << " " << n_inv_J_loc_ani << " " << n_inv_I_loc_ani << std::endl;    // number of ivnersion grid
            for(int k = 0; k < n_inv_K_loc_ani; k++){
                ofs << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ')
                    << radius2depth(r_ani.arr[I2V_INV_ANI_GRIDS_1DK(k,l)]) << " ";
                ofs << std::endl;
                for(int j =0; j < n_inv_J_loc_ani; j++)
                    ofs << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ')
                        << t_ani.arr[I2V_INV_ANI_GRIDS_1DJ(j,k,l)]*RAD2DEG << " ";
                ofs << std::endl;
                for(int i =0; i < n_inv_I_loc_ani; i++)
                    ofs << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ')
                        << p_ani.arr[I2V_INV_ANI_GRIDS_1DI(i,k,l)]*RAD2DEG << " ";
                ofs << std::endl;
            }
        }
    }
}

// grid position for radius
InvGrid1d::InvGrid1d(InputParams& IP, const int n_inv_loc,
                                  const CUSTOMREAL* input_pos){
    // set n
    n = n_inv_loc*n_inv_grids;

    // allocate memory for the grid
    arr = new CUSTOMREAL[n];

    // create a copy array of input_pos
    CUSTOMREAL* arr_input_pos = new CUSTOMREAL[n_inv_loc];

    // dep (km) -> r (km)
    for (int i = 0; i < n_inv_loc; i++)
        arr_input_pos[i] = depth2radius(input_pos[i]);

    for (int i = 0; i < n_inv_loc; i++){
        if (i < n_inv_loc-1)
            dinv_l = (arr_input_pos[i+1] - arr_input_pos[i]) / n_inv_grids;
        else
            dinv_l = (arr_input_pos[n_inv_loc-1] - arr_input_pos[n_inv_loc-2]) / n_inv_grids;

        for (int l = 0; l < n_inv_grids; l++)
            arr[I2V_INV_GRIDS_1D_GENERIC(i, l, n_inv_loc)] = arr_input_pos[i] + l*dinv_l;
    }

    delete[] arr_input_pos;
}

// grid position for lat and lon
InvGrid1d::InvGrid1d(InputParams& IP, const int n_inv_loc,
                                  const CUSTOMREAL* input_pos, const CUSTOMREAL* input_trapezoid, const int n_inv_loc_k, const CUSTOMREAL* input_dep) {
    // set n
    n = n_inv_loc*n_inv_loc_k*n_inv_grids;

    // allocate memory for the grid
    arr = new CUSTOMREAL[n];

    // create a copy array of input_pos
    CUSTOMREAL* arr_input_pos = new CUSTOMREAL[n_inv_loc];

    // lat,lon (deg) -> t,p (rad)
    for (int i = 0; i < n_inv_loc; i++)
        arr_input_pos[i] = input_pos[i]*DEG2RAD;

    // middle point of arr_input_pos
    CUSTOMREAL mid_value = 0;
    if (n_inv_loc%2 == 0)
        mid_value = (arr_input_pos[n_inv_loc/2-1] + arr_input_pos[n_inv_loc/2])/2;
    else
        mid_value = arr_input_pos[(n_inv_loc-1)/2];

    for (int i = 0; i < n_inv_loc; i++){
        if (i < n_inv_loc-1)
            dinv_l = (arr_input_pos[i+1] - arr_input_pos[i]) / n_inv_grids;
        elsegit 
            dinv_l = (arr_input_pos[n_inv_loc-1] - arr_input_pos[n_inv_loc-2]) / n_inv_grids;

        for (int l = 0; l < n_inv_grids; l++) {
            for (int k = 0; k < n_inv_loc_k; k++){
                CUSTOMREAL ratio = 1.0;
                if      (input_dep[k] < input_trapezoid[1])
                    ratio = 1.0;
                else if (input_dep[k] >= input_trapezoid[1] && input_dep[k] < input_trapezoid[2])
                    ratio = 1.0 + (input_dep[k] - input_trapezoid[1]) / (input_trapezoid[2] - input_trapezoid[1]) * (input_trapezoid[0] - 1.0);
                else
                    ratio = input_trapezoid[0];

                arr[I2V_INV_GRIDS_2D_GENERIC(i,k,l,n_inv_loc,n_inv_loc_k)] = mid_value + (arr_input_pos[i] + l*dinv_l - mid_value) * ratio;
            }
        }
    }

    delete[] arr_input_pos;
}



