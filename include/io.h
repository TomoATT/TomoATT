#ifndef IO_H
#define IO_H

#include <fstream>
#include <vector>
#if USE_HDF5
    #include "hdf5.h"
#endif
#include "tinyxml2.h"

// to handle circular dependency
#pragma once
#include "grid.fwd.h"
//#include "source.fwd.h"
#include "io.fwd.h"

#include "utils.h"
#include "mpi_funcs.h"
#include "grid.h"
#include "config.h"
#include "input_params.h"
#include "eikonal_solver_2d.h"


class IO_utils {

public:
    IO_utils(InputParams&);
    ~IO_utils();

    // initialize data output file
    void init_data_output_file();
    // finalize data/grid outpiut file (xdmf)
    void finalize_data_output_file();
    // change the output file name and xdmf objects
    void change_xdmf_obj(int);
    // output for updating xdmf file
    void update_xdmf_file();

    // change group name for source
    void reset_source_info(const int&, const std::string&);
    // change group name for model
    void change_group_name_for_model();
    // prepare grid for each inversion iteration
    void prepare_grid_inv_xdmf(int);

    // set id_sim
    void set_id_src(const int& id_src_){id_sim_src = id_src_;};
    // set source name
    void set_name_src(const std::string& name_src_){name_sim_src = name_src_;};


    //
    // write functions
    //

    // xdmf write out for grid data
    void write_xdmf_file_grid();

    // write grid data to output file
    void write_grid(Grid&);
    // general function for write out data in hdf5
    void write_data_h5(Grid&, std::string&, std::string&, CUSTOMREAL*, int, bool, bool for_tmp_db=false);
    // general function for write out data in hdf5 with merging subdomains
    void write_data_merged_h5(Grid&, std::string&, std::string&, std::string&, CUSTOMREAL*, bool, bool no_group=true);
    // general function for write out data in ascii
    void write_data_ascii(Grid&, std::string&, CUSTOMREAL*);
    // general function for write out data in ascii with merging subdomains
    void write_data_merged_ascii(Grid&, std::string&);
    // write true solution
    void write_true_solution(Grid&);
    // write velocity model
    void write_vel(Grid&, int);
    // write T0v
    void write_T0v(Grid&, int);
    // write u
    void write_u(Grid&);
    // write tau
    void write_tau(Grid&, int);
    // write temporal tau fields
    //void write_tmp_tau(Grid&, int);
    // write result timetable T
    void write_T(Grid&, int);
    // write temporal timetable T in a separated file
    void write_T_tmp(Grid&);
    // write residual (resudual = true_solution - result)
    void write_residual(Grid&);
    // write adjoint field
    void write_adjoint_field(Grid&, int);
    // write fun_loc
    void write_fun(Grid&, int);
    // write xi
    void write_xi(Grid&, int);
    // write eta
    void write_eta(Grid&, int);
    // write a
    void write_a(Grid&, int);
    // write b
    void write_b(Grid&, int);
    // write c
    void write_c(Grid&, int);
    // write f
    void write_f(Grid&, int);
    // Ks
    void write_Ks(Grid&, int);
    // Kxi
    void write_Kxi(Grid&, int);
    // Keta
    void write_Keta(Grid&, int);
    // Ks_density
    void write_Ks_density(Grid&, int);
    // Kxi_density
    void write_Kxi_density(Grid&, int);
    // Keta_density
    void write_Keta_density(Grid&, int);
    // Ks_over_Kden
    void write_Ks_over_Kden(Grid&, int);
    // Kxi_over_Kden
    void write_Kxi_over_Kden(Grid&, int);
    // Keta_over_Kden
    void write_Keta_over_Kden(Grid&, int);
    // Ks_update
    void write_Ks_update(Grid&, int);
    // Kxi_update
    void write_Kxi_update(Grid&, int);
    // Keta_update
    void write_Keta_update(Grid&, int);
    // Ks_density_update
    void write_Ks_density_update(Grid&, int);
    // Kxi_density_update
    void write_Kxi_density_update(Grid&, int);
    // Keta_density_update
    void write_Keta_density_update(Grid&, int);
    // Ks_descent_dir_loc
    void write_Ks_descent_dir(Grid&, int);
    // Kxi_descent_dir_loc
    void write_Kxi_descent_dir(Grid&, int);
    // Keta_descent_dir_loc
    void write_Keta_descent_dir(Grid&, int);

    // write all concerning parameters
    std::vector<CUSTOMREAL> get_grid_data(CUSTOMREAL * data);
    // void write_concerning_parameters(Grid&, int, InputParams&);

    // 2d traveltime field for teleseismic source
    void write_2d_travel_time_field(CUSTOMREAL*, CUSTOMREAL*, CUSTOMREAL*, int, int, CUSTOMREAL);
    void h5_create_and_write_dataset_2d(std::string&, int, int*, int, CUSTOMREAL*);
    void read_2d_travel_time_field(std::string&, CUSTOMREAL*, int, int);

    // 2d field for 1dinv
    void write_1dinv_field(CUSTOMREAL*, CUSTOMREAL*, CUSTOMREAL*, int, int, std::string&);

    // merged model
    void write_T_merged(Grid&, InputParams&, int);
    void write_merged_model(Grid&, InputParams&, std::string);
    bool node_of_this_subdomain(int*, const int&, const int&, const int&);

    //
    // read functions
    //

    // read model data
    void read_model(std::string&, const char*, CUSTOMREAL*, int, int, int);
    // read Travel time from file for earthquake relocation and common receiver double-difference
    void read_T(Grid&);
    // read Travel time from a temporal file for earthquake relocation and common receiver double-difference
    void read_T_tmp(Grid&);

    void read_data_ascii(Grid&, std::string&);

    //
    // delete functions
    //
    // Any deletion of data should be done on a created hdf5. Because the design of hdf5 is that
    // it is not possible to delete data from an existing hdf5 file. The only way to do so is to
    // create a new hdf5 file and copy the necessary data from the old one to the new one.
    // Therefore, the deletion of data is not implemented in this class.
    // Reference : https://stackoverflow.com/questions/1124994/removing-data-from-a-hdf5-file

private:
    // member variables
    std::string h5_output_grid_fname = "out_data_grid.h5"; // output file name
    std::string h5_output_fname      = "out_data.h5";      // output file name
    std::string h5_output_fname_tmp  = "out_data_tmp.h5";  // output file name
    std::string xdmf_output_fname    = "out_data.xmf";     // output file name
    std::string h5_group_name_grid   = "Mesh";               // group name for grid data
    std::string h5_group_name_event  = "Event";              // group name for event data
    std::string h5_group_name_data   = "Data";               // group name for event data

    std::string h5_dset_name_node_coords_x = "node_coords_x"; // dataset name for node coordinates
    std::string h5_dset_name_node_coords_y = "node_coords_y"; // dataset name for node coordinates
    std::string h5_dset_name_node_coords_z = "node_coords_z"; // dataset name for node coordinates
    std::string h5_dset_name_node_coords_p = "node_coords_p"; // dataset name for node coordinates
    std::string h5_dset_name_node_coords_t = "node_coords_t"; // dataset name for node coordinates
    std::string h5_dset_name_node_coords_r = "node_coords_r"; // dataset name for node coordinates
    std::string h5_dset_name_elem_conn     = "elem_conn";     // dataset name for element connectivity
    std::string h5_dset_name_procid        = "procid";        // dataset name for my_procs

    std::string h5_file_and_group  = h5_output_grid_fname + ":/" + h5_group_name_grid;
    std::string h5_whole_name_conn = h5_file_and_group + "/" + h5_dset_name_elem_conn;
    std::string h5_whole_name_x    = h5_file_and_group + "/" + h5_dset_name_node_coords_x;
    std::string h5_whole_name_y    = h5_file_and_group + "/" + h5_dset_name_node_coords_y;
    std::string h5_whole_name_z    = h5_file_and_group + "/" + h5_dset_name_node_coords_z;
    std::string h5_whole_name_proc = h5_file_and_group + "/" + h5_dset_name_procid;

    // index of current simulation used for naming output files and datasets
    int id_sim_src;
    // name of current simulation used for naming output files and datasets
    std::string name_sim_src;

    int nnodes_glob = 0;
    int nelms_glob  = 0;

    int array_rank_default = 2; // default array rank

    int custom_real_flag; //  custom real type

    // custom data type for real
    std::string type_name = "Float";
    std::string type_size = "4";

    // helper for xdmf/xml file io
    tinyxml2::XMLDocument* doc;
    tinyxml2::XMLNode* xdmf, *domain, *grid;
    tinyxml2::XMLNode* inversions, *inv_grid;
    void init_xdmf_file();
    void write_xdmf_file();
    void finalize_xdmf_file();
    void insert_data_xdmf(std::string&, std::string&, std::string&);

#ifdef USE_HDF5
    // h5 variables
    hid_t  file_id;
    hid_t  group_id;
    hid_t  space_id;
    hid_t  dset_id;
    hid_t  plist_id, plist_id_dset;
    hid_t  file_dspace_id, mem_dspace_id;
    herr_t status;
    hid_t  plist_id_2d;
    hid_t  file_id_2d;

    // h5 low level utilities
    void h5_create_file_by_group_main(std::string&);
    void h5_open_file_by_group_main(std::string&);
    void h5_open_file_collective(std::string&);
    void h5_open_file_collective_input(std::string&);
    void h5_close_file_by_group_main();
    void h5_close_file_collective();

    void h5_create_group_by_group_main(std::string& );
    void h5_open_group_by_group_main(std::string& );
    void h5_open_group_collective(std::string& );
    void h5_close_group_by_group_main();
    void h5_close_group_collective();

    void h5_create_dataset(std::string&, int, int*, int, bool no_group = false);
    void h5_open_dataset(std::string&);
    void h5_open_dataset_no_group(std::string&);
    void h5_close_dataset();

    template <typename T>
    void h5_write_array(std::string&, int, int*, T*, int offset_one, int offset_two=0, int offset_three=0, bool no_group=false);

    template <typename T>
    void h5_read_array(std::string&, int, int*, T*, int*, bool);

    template <typename T>
    void h5_read_array_simple(std::string&,  T*);

    void read_data_h5(Grid&, CUSTOMREAL*, std::string, std::string);
#endif // HDF_HDF5

    // utilities
    std::string create_fname_ascii(std::string&);
    std::string create_fname_ascii_model(std::string&);

    // flag for data type
    const bool model_data = true;
    const bool src_data = false;

};

#endif // IO_H