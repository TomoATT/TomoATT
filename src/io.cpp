#include "io.h"

IO_utils::IO_utils(InputParams& IP) {
    if (subdom_main) {
         stdout_by_main("--- IO object initialization ---");

        // check the custom real data type
        // if float, custom_real_flag = 2
        // if double, custom_real_flag = 3
        if (std::is_same<CUSTOMREAL, float>::value || IP.get_if_single_precision_output()) {
            custom_real_flag = 2;
        } else if (std::is_same<CUSTOMREAL, double>::value) {
            custom_real_flag = 3;
        } else {
            std::cout << "Error: custom real type is not float or double" << std::endl;
            std::cout << "Please check the definition of CUSTOMREAL in io.h" << std::endl;
            std::cout << "Exiting..." << std::endl;
            exit(1);
        }

        // initialize an output file for grid data
        if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
            // grid data is output by only simulation group 0
            if(id_sim==0)
                h5_create_file_by_group_main(h5_output_grid_fname); // file for grid+whole simulation data
            // initialize simulation data file.
            if (n_sims > 1){
                // for simultaneous run, data file is created for each simulation group
                h5_output_fname = "./out_data_sim_group_" + std::to_string(id_sim) + ".h5";
                xdmf_output_fname = "./out_data_sim_group_"+std::to_string(id_sim)+".xmf";
                h5_output_fname_tmp = "./tmp_db_sim_group_" + std::to_string(id_sim) + ".h5";
            } else {
                // otherwise, only one single data file is created for storing model params
                h5_output_fname = "./out_data_sim_group_0.h5";      // add _groud_0 to ensure consistency
                xdmf_output_fname = "./out_data_sim_group_0.xmf";
                h5_output_fname_tmp = "./tmp_db_sim_group_0.h5";
            }

            // MNMN: below is for initiliazing data file for each simulation group
            // if the file is not created in here, hdf5 will try to use preexisting file and dataset structures,
            // then error will occur if the dataset size for the later simulation group is larger than the previous one.

            // create data file
            h5_create_file_by_group_main(h5_output_fname);
            // create temporary data file
            h5_create_file_by_group_main(h5_output_fname_tmp);
#else
            std::cout << "Error: TOMOATT was not compiled with HDF5" << std::endl;
            std::cout << "Exiting..." << std::endl;
            exit(1);
#endif
        }
    }
}

IO_utils::~IO_utils() {
    stdout_by_main("--- IO object finalization ---");
}

void IO_utils::reset_source_info(const int& id_sim_src, const std::string& name_sim_src) {

    // set simulation group id and source name for output files/dataset names
    set_id_src(id_sim_src);
    set_name_src(name_sim_src);

#ifdef USE_HDF5
    // change group name for source
    // h5_group_name_data = "src_" + std::to_string(id_sim_src);
    h5_group_name_data = "src_" + name_sim_src;
#endif
}


void IO_utils::change_group_name_for_model(){
#ifdef USE_HDF5
    // change group name for model
    h5_group_name_data = "model";
#endif
}


void IO_utils::finalize_data_output_file(){
    if (output_format==OUTPUT_FORMAT_HDF5){
        // close file
        if (subdom_main && id_subdomain==0)
            finalize_xdmf_file();
    }
}


void IO_utils::init_data_output_file() {
    if (if_verbose) stdout_by_main("--- initialize data output file ---");

    // create output file
    if (output_format==OUTPUT_FORMAT_HDF5) {
#ifdef USE_HDF5
        // h5_output_fname   = "./out_data_sim_"+std::to_string(id_sim_src)+".h5";
        // xdmf_output_fname = "./out_data_sim_"+std::to_string(id_sim_src)+".xmf";
        h5_output_fname   = "./out_data_sim_"+name_sim_src+".h5";
        xdmf_output_fname = "./out_data_sim_"+name_sim_src+".xmf";

        if (subdom_main) {
            // write xdmf file
            if (id_subdomain == 0)
                write_xdmf_file_grid();

            // create output file
            h5_create_file_by_group_main(h5_output_fname); // file for field data
        }
#endif
    }
}



void IO_utils::write_grid(Grid& grid) {

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        // share grid dimensions
        int dims_ngrid[1] = {grid.get_ngrid_total_vis()};
        int dims_conn[2]  = {grid.get_nelms_total_vis(), 9};

        allreduce_i_single(dims_ngrid[0], nnodes_glob);
        allreduce_i_single(dims_conn[0], nelms_glob);

        // create xdmf file for visualization of simulation data
        if (id_subdomain == 0)
            write_xdmf_file_grid();

        if (id_sim == 0) {

            // open file
            h5_open_file_by_group_main(h5_output_grid_fname);

            // make groups Grid and Event
            h5_create_group_by_group_main(h5_group_name_grid);
            h5_open_group_by_group_main(h5_group_name_grid);

            // create datasets
            h5_create_dataset(h5_dset_name_node_coords_x, 1, dims_ngrid, custom_real_flag);
            h5_create_dataset(h5_dset_name_node_coords_y, 1, dims_ngrid, custom_real_flag);
            h5_create_dataset(h5_dset_name_node_coords_z, 1, dims_ngrid, custom_real_flag);
            h5_create_dataset(h5_dset_name_node_coords_p, 1, dims_ngrid, custom_real_flag);
            h5_create_dataset(h5_dset_name_node_coords_t, 1, dims_ngrid, custom_real_flag);
            h5_create_dataset(h5_dset_name_node_coords_r, 1, dims_ngrid, custom_real_flag);
            h5_create_dataset(h5_dset_name_elem_conn,     2, dims_conn,  1);
            h5_create_dataset(h5_dset_name_procid,        1, dims_ngrid, 1);

            // close and reopen file from all processes
            h5_close_group_by_group_main();
            h5_close_file_by_group_main();

            h5_open_file_collective(h5_output_grid_fname);
            h5_open_group_collective(h5_group_name_grid); // Open group "Grid"

            // write node coordinates
            h5_write_array(h5_dset_name_node_coords_x, 1, dims_ngrid, grid.get_nodes_coords_x(), grid.get_offset_nnodes_vis());
            h5_write_array(h5_dset_name_node_coords_y, 1, dims_ngrid, grid.get_nodes_coords_y(), grid.get_offset_nnodes_vis());
            h5_write_array(h5_dset_name_node_coords_z, 1, dims_ngrid, grid.get_nodes_coords_z(), grid.get_offset_nnodes_vis());
            h5_write_array(h5_dset_name_node_coords_p, 1, dims_ngrid, grid.get_nodes_coords_p(), grid.get_offset_nnodes_vis());
            h5_write_array(h5_dset_name_node_coords_t, 1, dims_ngrid, grid.get_nodes_coords_t(), grid.get_offset_nnodes_vis());
            h5_write_array(h5_dset_name_node_coords_r, 1, dims_ngrid, grid.get_nodes_coords_r(), grid.get_offset_nnodes_vis());
            h5_write_array(h5_dset_name_procid,        1, dims_ngrid, grid.get_proc_dump(),      grid.get_offset_nnodes_vis());

            // write element connectivity
            //(write here as 1d element to keep compatibility with other spatial discretization data)
            h5_write_array(h5_dset_name_elem_conn, 2, dims_conn, grid.get_elms_conn(), grid.get_offset_elms_vis());

            // close group "Grid"
            h5_close_group_collective();
            // close file
            h5_close_file_collective();
        }
#else
        std::cout << "Error: HDF5 is not enabled.\n";
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII) {
        if (id_sim == 0) {
            // outut grid to ASCII file

            // get data
            CUSTOMREAL* nodes_coords_x = grid.get_nodes_coords_x();
            CUSTOMREAL* nodes_coords_y = grid.get_nodes_coords_y();
            CUSTOMREAL* nodes_coords_z = grid.get_nodes_coords_z();
            CUSTOMREAL* nodes_coords_p = grid.get_nodes_coords_p();
            CUSTOMREAL* nodes_coords_t = grid.get_nodes_coords_t();
            CUSTOMREAL* nodes_coords_r = grid.get_nodes_coords_r();
            int*        connectivity   = grid.get_elms_conn();

            // ASCII data collective writing
            std::string fname_xyz  = output_dir + "/out_grid_xyz.dat";
            std::string fname_rtp  = output_dir + "/out_grid_ptr.dat";
            std::string fname_conn = output_dir + "/out_grid_conn.dat";

            for (int i_rank = 0; i_rank < n_subdomains; i_rank++){
                if (i_rank == myrank){
                    std::ofstream outfile_xyz, outfile_rtp, outfile_conn;
                    if (i_rank == 0){
                        outfile_xyz.open(fname_xyz);
                        outfile_rtp.open(fname_rtp);
                        outfile_conn.open(fname_conn);
                    }
                    else{
                        outfile_xyz.open(fname_xyz, std::ios_base::app); // append
                        outfile_rtp.open(fname_rtp, std::ios_base::app); // append
                        outfile_conn.open(fname_conn, std::ios_base::app); // append
                    }

                    // set precision
                    outfile_xyz.precision( ASCII_OUTPUT_PRECISION);
                    outfile_rtp.precision( ASCII_OUTPUT_PRECISION);
                    outfile_conn.precision(ASCII_OUTPUT_PRECISION);

                    //if (i_rank == 0) {
                    //    // write header
                    //    outfile << "# node coordinates x y z radius(km) lat.(deg.) lon.(deg.)" << std::endl;
                    //}

                    for (int k = 0; k < loc_K_vis; k++){
                        for (int j = 0; j < loc_J_vis; j++){
                            for (int i = 0; i < loc_I_vis; i++){

                                int idx = I2V_VIS(i,j,k);
                                //outfile << nodes_coords_x[idx] << " " << nodes_coords_y[idx] << " " << nodes_coords_z[idx] << " " << nodes_coords_p[idx] << " " << nodes_coords_t[idx] << " " << nodes_coords_r[idx] << "\n";
                                outfile_xyz << nodes_coords_x[idx] << " " << nodes_coords_y[idx] << " " << nodes_coords_z[idx] << "\n";
                                outfile_rtp << nodes_coords_p[idx] << " " << nodes_coords_t[idx] << " " << nodes_coords_r[idx] << "\n";

                                if (i < loc_I_vis-1 \
                                 && j < loc_J_vis-1 \
                                 && k < loc_K_vis-1) {
                                    int idx_conn = I2V_ELM_CONN(i,j,k);
                                   for (int iconn = 0; iconn < 9; iconn++){
                                        outfile_conn << connectivity[idx_conn*9 + iconn] << " ";
                                    }
                                    outfile_conn << "\n";
                                }
                            }
                        }
                    }

                    outfile_xyz.close();
                    outfile_rtp.close();
                    outfile_conn.close();
                }
                // synchronize
                synchronize_all_inter();

            } // end for i_rank
       } // end if id_sim == 0
    } // end if output_format==OUTPUT_FORMAT_ASCII

}


void IO_utils::init_xdmf_file(){
    if (if_verbose) stdout_by_main("--- init xdmf file ---");

    if (std::is_same<CUSTOMREAL, float>::value) {
        type_name = "Float";
        type_size = "4";
    } else {
        type_name = "Double";
        type_size = "8";
    }

    doc = new tinyxml2::XMLDocument(); // TODO: delete this object somewhere at the end

    // headers
    tinyxml2::XMLDeclaration* decl = doc->NewDeclaration();
    doc->InsertFirstChild(decl);
    tinyxml2::XMLUnknown* doctype = doc->NewUnknown("DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []");
    doc->InsertAfterChild(decl,doctype);

    // Xdmf node
    xdmf = doc->InsertEndChild( doc->NewElement( "Xdmf" ) );
    xdmf->ToElement()->SetAttribute("xmlns:xi","http://www.w3.org/2001/XInclude");
    xdmf->ToElement()->SetAttribute("Version", "3.0");
}


void IO_utils::write_xdmf_file_grid() {
    // write xdmf file by main process
    init_xdmf_file();

    std::string num_elm   = std::to_string(nelms_glob);
    std::string num_conn  = std::to_string(nelms_glob) + " 9";
    std::string num_nodes = std::to_string(nnodes_glob);

    // write xdmf file
    // Domain node
    domain = xdmf->InsertEndChild( doc->NewElement( "Domain" ) );
    domain->ToElement()->SetAttribute("name", "Mesh");

    // Topology node in domain
    tinyxml2::XMLNode* topology = domain->InsertEndChild( doc->NewElement( "Topology" ) );
    topology->ToElement()->SetAttribute("TopologyType", "Mixed");
    topology->ToElement()->SetAttribute("NumberOfElements", num_elm.c_str());
    // DataItem node in topology
    tinyxml2::XMLNode* dataItem = topology->InsertEndChild( doc->NewElement( "DataItem" ) );
    dataItem->ToElement()->SetAttribute("ItemType", "Uniform");
    dataItem->ToElement()->SetAttribute("Format", "HDF");
    dataItem->ToElement()->SetAttribute("NumberType", "Int");
    dataItem->ToElement()->SetAttribute("Precision", "4");
    dataItem->ToElement()->SetAttribute("Dimensions", num_conn.c_str());
    // set dataset path in dataItem
    dataItem->InsertEndChild( doc->NewText( h5_whole_name_conn.c_str() ) );

    // Geometry node in domain
    tinyxml2::XMLNode* geometry = domain->InsertEndChild( doc->NewElement( "Geometry" ) );
    geometry->ToElement()->SetAttribute("GeometryType", "X_Y_Z");
    // DataItem node in geometry
    tinyxml2::XMLNode* dataItem_coords[3] = { geometry->InsertEndChild( doc->NewElement( "DataItem" ) ),
                                              geometry->InsertEndChild( doc->NewElement( "DataItem" ) ),
                                              geometry->InsertEndChild( doc->NewElement( "DataItem" ) ) };
    for (int i = 0; i < 3; i++) {
        dataItem_coords[i]->ToElement()->SetAttribute("ItemType", "Uniform");
        dataItem_coords[i]->ToElement()->SetAttribute("Format", "HDF");
        dataItem_coords[i]->ToElement()->SetAttribute("NumberType", type_name.c_str());
        dataItem_coords[i]->ToElement()->SetAttribute("Precision", type_size.c_str());
        dataItem_coords[i]->ToElement()->SetAttribute("Dimensions", num_nodes.c_str());
    }
    // set dataset path in dataItem
    dataItem_coords[0]->InsertEndChild( doc->NewText( h5_whole_name_x.c_str() ) );
    dataItem_coords[1]->InsertEndChild( doc->NewText( h5_whole_name_y.c_str() ) );
    dataItem_coords[2]->InsertEndChild( doc->NewText( h5_whole_name_z.c_str() ) );

    // write data to file
    // Grid model
    grid = domain->InsertEndChild( doc->NewElement( "Grid" ) );
    grid->ToElement()->SetAttribute("Name", "Data");
    grid->ToElement()->SetAttribute("GridType", "Tree");

    // procid Grid
    tinyxml2::XMLNode* grid_proc = grid->InsertEndChild( doc->NewElement( "Grid" ) );
    grid_proc->ToElement()->SetAttribute("Name", "procid");
    grid_proc->ToElement()->SetAttribute("GridType", "Uniform");

    // Topology node in grid
    tinyxml2::XMLNode* topology_grid = grid_proc->InsertEndChild( doc->NewElement( "Topology" ) );
    topology_grid->ToElement()->SetAttribute("Reference", "/Xdmf/Domain/Topology");
    // Geometry node in grid
    tinyxml2::XMLNode* geometry_grid = grid_proc->InsertEndChild( doc->NewElement( "Geometry" ) );
    geometry_grid->ToElement()->SetAttribute("Reference", "/Xdmf/Domain/Geometry");
    // Attribute node in grid
    tinyxml2::XMLNode* attribute_grid = grid_proc->InsertEndChild( doc->NewElement( "Attribute" ) );
    attribute_grid->ToElement()->SetAttribute("Name", "procid");
    attribute_grid->ToElement()->SetAttribute("AttributeType", "Scalar");
    attribute_grid->ToElement()->SetAttribute("Center", "Node");
    // DataItem node in attribute
    tinyxml2::XMLNode* dataItem_procid = attribute_grid->InsertEndChild( doc->NewElement( "DataItem" ) );
    dataItem_procid->ToElement()->SetAttribute("ItemType", "Uniform");
    dataItem_procid->ToElement()->SetAttribute("Format", "HDF");
    dataItem_procid->ToElement()->SetAttribute("NumberType", "Int");
    dataItem_procid->ToElement()->SetAttribute("Precision", "4");
    dataItem_procid->ToElement()->SetAttribute("Dimensions", num_nodes.c_str());
    // set dataset path in dataItem
    dataItem_procid->InsertEndChild( doc->NewText( h5_whole_name_proc.c_str() ) );

    // prepare inversion iteration collection
    inversions = grid->InsertEndChild( doc->NewElement( "Grid" ) );
    inversions->ToElement()->SetAttribute("Name", "Inversions");
    inversions->ToElement()->SetAttribute("GridType", "Collection");
    inversions->ToElement()->SetAttribute("CollectionType", "Temporal");
    // close file
    //finalize_xdmf_file();
}


void IO_utils::update_xdmf_file() {
    if (output_format==OUTPUT_FORMAT_HDF5){
        if (subdom_main && id_subdomain==0)
            write_xdmf_file();
    }
}


void IO_utils::write_xdmf_file()
{
    if (output_format==OUTPUT_FORMAT_HDF5){
        if (subdom_main && id_subdomain==0) {
            //xdmf_output_fname = xmfname_vec[i_src];
            std::string fout = output_dir+"/"+xdmf_output_fname;
            doc->SaveFile(fout.c_str(), false); // true: compact format
        }
    }
}


void IO_utils::finalize_xdmf_file(){
    if (output_format==OUTPUT_FORMAT_HDF5){
        stdout_by_main("--- finalize xdmf file ---");

        write_xdmf_file();
    }
}

#ifdef USE_HDF5
// function to write data to HDF5 file without merging subdomains
void IO_utils::write_data_h5(Grid& grid, std::string& str_group, std::string& str_dset,
                             CUSTOMREAL* array, int i_inv, bool model_data, bool for_tmp_db) {

    std::string str_dset_h5 = "dummy";
    if (!for_tmp_db)
        str_dset_h5 = str_dset + "_inv_" + int2string_zero_fill(i_inv);
    else
        str_dset_h5 = str_dset;

    // write true solution to h5 file
    if (myrank == 0 && if_verbose)
        std::cout << "--- write data " << str_dset << " to h5 file " << h5_output_fname << " ---" << std::endl;

    // open h5 file
    h5_open_file_by_group_main(h5_output_fname);

    // make groups Grid and Event
    h5_create_group_by_group_main(str_group);
    h5_open_group_by_group_main(str_group);

    // create datasets
    int dims_ngrid[1] = {grid.get_ngrid_total_vis()};
    //int dims_conn[2]  = {grid.get_nelms_total_vis(), 9};

    allreduce_i_single(dims_ngrid[0], nnodes_glob);
    //allreduce_i_single(dims_conn[0], nelms_glob);

    h5_create_dataset(str_dset_h5, 1, dims_ngrid, custom_real_flag);

    // close and reopen file from all processes
    h5_close_group_by_group_main();
    h5_close_file_by_group_main();

    h5_open_file_collective(h5_output_fname);
    h5_open_group_collective(str_group); // Open group "Grid"

    // write datasets
    h5_write_array(str_dset_h5, 1, dims_ngrid, array, grid.get_offset_nnodes_vis());

    // close group "Grid"
    h5_close_group_collective();
    // close file
    h5_close_file_collective();

    if (!for_tmp_db) {
        std::string str_dset_xdmf;
        if (!model_data)
            str_dset_xdmf = str_dset + "_src_" + name_sim_src; // use different name for xdmf
        else
            str_dset_xdmf = str_dset; // use different name for xdmf

        // add xdmf file in
        insert_data_xdmf(str_group, str_dset_xdmf, str_dset_h5);
    }
}

// function to write data to HDF5 file with merging subdomains
void IO_utils::write_data_merged_h5(Grid& grid,std::string& str_filename, std::string& str_group, std::string& str_dset,
                                    CUSTOMREAL* array, bool inverse_field, bool no_group) {

    // write true solution to h5 file
    if (myrank == 0 && if_verbose)
        std::cout << "--- write data " << str_dset << " to h5 file " << h5_output_fname << " ---" << std::endl;

    const int ndims = 3;
    // allocate temporary array
    CUSTOMREAL* array_tmp = new CUSTOMREAL[loc_I_excl_ghost*loc_J_excl_ghost*loc_K_excl_ghost];

    // get array data except ghost nodes
    grid.get_array_for_3d_output(array, array_tmp, inverse_field);

    // open h5 file
    h5_open_file_by_group_main(str_filename);

    if (!no_group){
        h5_create_group_by_group_main(str_group);
        h5_open_group_by_group_main(str_group);
    }

    // dataset size of whole domain
    int dims_ngrid[ndims] = {ngrid_k, ngrid_j, ngrid_i};
    // create datasets
    h5_create_dataset(str_dset, ndims, dims_ngrid, custom_real_flag, no_group);

    // close and reopen file from all processes
    if(!no_group) h5_close_group_by_group_main();
    h5_close_file_by_group_main();

    h5_open_file_collective(str_filename);
    if(!no_group) h5_open_group_collective(str_group); // Open group "Grid"

    // offset info for this subdomain
    int offsets[3];
    grid.get_offsets_3d(offsets);
    // dimensions of this subdomain
    int dims_ngrid_loc[ndims] = {loc_K_excl_ghost, loc_J_excl_ghost, loc_I_excl_ghost};

    // write datasets
    h5_write_array(str_dset, ndims, dims_ngrid_loc, array_tmp, offsets[0], offsets[1], offsets[2], no_group);

    // close group "Grid"
    if(!no_group) h5_close_group_collective();
    // close file
    h5_close_file_collective();

    // add xdmf file in
    //insert_data_xdmf(str_group, str_dset_xdmf, str_dset);

    delete [] array_tmp;

}

#endif // USE_HDF5


void IO_utils::write_data_ascii(Grid& grid, std::string& fname, CUSTOMREAL* data){
    // write data in ascii file
    if (myrank == 0 && if_verbose)
        std::cout << "--- write data to ascii file " << fname << " ---" << std::endl;

    // independent write
    for (int i_rank = 0; i_rank < n_subdomains; i_rank++) {
        if (i_rank == myrank){
            std::ofstream fout;
            if (i_rank == 0)
                fout.open(fname.c_str());
            else
                fout.open(fname.c_str(), std::ios_base::app); // append
            if (!fout.is_open()) {
                std::cout << "ERROR: cannot open file " << fname << std::endl;
                exit(1);
            }

            // set output precision
            fout.precision(ASCII_OUTPUT_PRECISION);

            // iterate over nodes skipping the last layer for ignoring gap filling nodes for h5 output
            for (int k = 0; k < loc_K_vis; k++){
                for (int j = 0; j < loc_J_vis; j++){
                    for (int i = 0; i < loc_I_vis; i++){
                        int idx = I2V_VIS(i,j,k);
                        fout << data[idx] << "\n";
                    }
                }
            }

            fout.close();
        }
        // synchronize
        synchronize_all_inter();

    } // end for i_rank

}


bool IO_utils::node_of_this_subdomain(int* offsets, const int& i, const int& j, const int& k){
    // check if node is in this subdomain
    if (i >= offsets[2] && i < offsets[2] + loc_I_excl_ghost &&
        j >= offsets[1] && j < offsets[1] + loc_J_excl_ghost &&
        k >= offsets[0] && k < offsets[0] + loc_K_excl_ghost)
        return true;
    else
        return false;
}

void IO_utils::write_data_merged_ascii(Grid& grid, std::string& fname){
    // write data in ascii file
    if (myrank == 0 && if_verbose)
        std::cout << "--- write data to ascii file " << fname << " ---" << std::endl;

    // allocate temporary array
    CUSTOMREAL* array_eta  = new CUSTOMREAL[loc_I_excl_ghost*loc_J_excl_ghost*loc_K_excl_ghost];
    CUSTOMREAL* array_xi   = new CUSTOMREAL[loc_I_excl_ghost*loc_J_excl_ghost*loc_K_excl_ghost];
    //CUSTOMREAL* array_zeta = new CUSTOMREAL[loc_I_excl_ghost*loc_J_excl_ghost*loc_K_excl_ghost];
    CUSTOMREAL* array_vel  = new CUSTOMREAL[loc_I_excl_ghost*loc_J_excl_ghost*loc_K_excl_ghost];

    // offset info for this subdomain
    int offsets[3];
    grid.get_offsets_3d(offsets);

    // get array data except ghost nodes
    grid.get_array_for_3d_output(grid.eta_loc, array_eta, false);
    grid.get_array_for_3d_output(grid.xi_loc, array_xi, false);
    //grid.get_array_for_3d_output(grid.zeta_loc, array_zeta, false);
    grid.get_array_for_3d_output(grid.fun_loc, array_vel, true);

    // create output file
    std::ofstream fout;
    if (myrank == 0){
        fout.open(fname.c_str());
        fout.close();
    }

    // This way of writing is not parallelized and very slow
    // thus user is strongly encouraged to use h5 output
    for (int k = 0; k < ngrid_k; k++){
        for (int j = 0; j < ngrid_j; j++){
            for (int i = 0; i < ngrid_i; i++){
                if (node_of_this_subdomain(offsets, i,j,k)){
                    // open file
                    fout.open(fname.c_str(), std::ios_base::app); // append
                    int idx = I2V_3D(i-offsets[2],j-offsets[1],k-offsets[0]);
                    // write eta xi zeta vel
                    fout << array_eta[idx] << "   " << array_xi[idx] << "   " << 0.0 << "   " << array_vel[idx] << "\n";
                    fout.close();
                }
                // synchronize
                synchronize_all_inter();
            }
        }
    }

    delete [] array_eta;
    delete [] array_xi;
    //delete [] array_zeta;
    delete [] array_vel;

}


void IO_utils::write_1dinv_field(CUSTOMREAL* field_1dinv, CUSTOMREAL* r_1dinv, CUSTOMREAL* t_1dinv, int nr_1dinv, int nt_1dinv, std::string& field_name){
    if (myrank == 0) {
        if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
            // open h5 file
            std::string fout = output_dir + "/" + h5_output_fname;
            file_id_2d = H5Fopen(fout.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      
            // force to use CUSTOMREAL type for 2D fields
            int dtype = check_data_type(field_1dinv[0]);

            // create dataset and write
            int dims_array[2] = {nr_1dinv, nt_1dinv};
            std::string str_dset = field_name;
            
            if (H5Lexists(file_id_2d, str_dset.c_str(), H5P_DEFAULT) > 0){
                H5Ldelete(file_id_2d, str_dset.c_str(), H5P_DEFAULT);  // delete nit
            }
            h5_create_and_write_dataset_2d(str_dset, 2, dims_array, dtype, field_1dinv);

            if (H5Lexists(file_id_2d, "r_1dinv", H5P_DEFAULT) > 0){
                H5Ldelete(file_id_2d, "r_1dinv", H5P_DEFAULT);  // delete nit
            } 
            str_dset = "r_1dinv";
            h5_create_and_write_dataset_2d(str_dset, 1, &nr_1dinv, dtype, r_1dinv);
            
            if (H5Lexists(file_id_2d, "t_1dinv", H5P_DEFAULT) > 0){
                H5Ldelete(file_id_2d, "t_1dinv", H5P_DEFAULT);  // delete nit
            }
            str_dset = "t_1dinv";
            h5_create_and_write_dataset_2d(str_dset, 1, &nt_1dinv, dtype, t_1dinv);

            // close h5 file
            H5Fclose(file_id_2d);
#else
            std::cout << "ERROR: HDF5 is not enabled" << std::endl;
            exit(1);
#endif
        } else if (output_format==OUTPUT_FORMAT_ASCII){
            std::cout << "ERROR: ASCII is not supported for 1dinv field output now" << std::endl;
            exit(1);
        }
    }
}


void IO_utils::write_2d_travel_time_field(CUSTOMREAL* T, CUSTOMREAL* r, CUSTOMREAL* t, int nr, int nt, CUSTOMREAL src_dep){

    if (myrank == 0) {

        if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
            auto str = std::to_string(src_dep);
            std::string fname = output_dir + "/" + OUTPUT_DIR_2D + "/2d_travel_time_field_dep_" +str.substr(0,str.find(".")+4)+".h5";
            // create and open h5 file
            file_id_2d  = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            // force to use CUSTOMREAL type for 2D fields
            int dtype = check_data_type(T[0]);

            // create dataset and write
            int dims_T[2] = {nr, nt};
            std::string str_dset = "T";
            h5_create_and_write_dataset_2d(str_dset, 2, dims_T, dtype, T);
            str_dset = "r";
            h5_create_and_write_dataset_2d(str_dset, 1, &nr, dtype, r);
            str_dset = "t";
            h5_create_and_write_dataset_2d(str_dset, 1, &nt, dtype, t);

            // close h5 file
            H5Fclose(file_id_2d);
#else
            std::cout << "ERROR: HDF5 is not enabled" << std::endl;
            exit(1);
#endif
        } else if (output_format==OUTPUT_FORMAT_ASCII){
            // write out r t and T in ASCII
            auto str = std::to_string(src_dep);
            std::string fname = output_dir + "/" + OUTPUT_DIR_2D + "/2d_travel_time_field_dep_" +str.substr(0,str.find(".")+4)+".dat";
            std::ofstream fout(fname.c_str());

            // set precision
            fout << std::fixed << std::setprecision(ASCII_OUTPUT_PRECISION);

            for (int i=0; i<nr; i++)
                for (int j=0; j<nt; j++)
                    fout << r[i] << "   " << t[j] << "   " << T[i*nt+j] << "\n";
            fout.close();
        }
    }

}


void IO_utils::read_2d_travel_time_field(std::string& fname, CUSTOMREAL* T, int nt, int nr){
    if (myrank == 0) {
        if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
            // open h5 file
            file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            // open dataset
            std::string str_dset = "T";
            // read data
            h5_read_array_simple(str_dset,  T);
            // close h5 file
            H5Fclose(file_id);
#else
            std::cout << "ERROR: HDF5 is not enabled" << std::endl;
            exit(1);
#endif
        } else if (output_format==OUTPUT_FORMAT_ASCII){
            // read ascii file
            std::ifstream fin(fname.c_str());
            if (!fin.is_open()) {
                std::cout << "ERROR: cannot open file " << fname << std::endl;
                exit(1);
            }
            CUSTOMREAL r_dummy, t_dummy;
            for (int i=0; i<nr; i++)
                for (int j=0; j<nt; j++)
                    fin >> r_dummy >> t_dummy >> T[i*nt+j];
        }
    }
}


std::string IO_utils::create_fname_ascii(std::string& dset_name){
    // std::string fname = output_dir + "/" + dset_name
    // + "_src_" + int2string_zero_fill(id_sim_src) + ".dat";
    std::string fname = output_dir + "/" + dset_name \
    + "_src_" + name_sim_src + ".dat";
    return fname;
}


std::string IO_utils::create_fname_ascii_model(std::string& dset_name){
    std::string fname = output_dir + "/" + dset_name + ".dat";
    return fname;
}


void IO_utils::write_true_solution(Grid& grid) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "true_solution";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_true_solution(), 0, src_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "true_solution";
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_true_solution());
    }
}


void IO_utils::write_vel(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "vel";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_vel(),i_inv, model_data);
#else

        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "vel_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_vel());
    }
}


std::vector<CUSTOMREAL> IO_utils::get_grid_data(CUSTOMREAL* data) {

    std::vector<CUSTOMREAL> parameter(loc_K_vis * loc_J_vis * loc_I_vis);

    for (int k = 0; k < loc_K_vis; k++){
        for (int j = 0; j < loc_J_vis; j++){
            for (int i = 0; i < loc_I_vis; i++){
                int idx = I2V_VIS(i,j,k);
                parameter[idx] = data[idx];
            }
        }
    }
    return parameter;
}

// void IO_utils::write_concerning_parameters(Grid& grid, int i_inv, InputParams& IP) {
//     std::string dset_name = "model_parameters_inv_" + int2string_zero_fill(i_inv);
//     std::string fname = create_fname_ascii_model(dset_name);

//     if (myrank == 0 )
//         std::cout << "--- write data to ascii file " << fname << " ---" << std::endl;

//     // collective write
//     if (subdom_main){

//         for (int i_rank = 0; i_rank < n_subdomains; i_rank++) {
//             if (i_rank == myrank){
//                 std::ofstream fout;
//                 if (i_rank == 0){
//                     fout.open(fname.c_str());
//                     fout<< std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << "depth" << " "
//                         << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << "latitude" << " "
//                         << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << "longitude" << " "
//                         << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << "slowness" << " "
//                         << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << "xi" << " "
//                         << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << "eta" << " "
//                         << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << "velocity" << " "
//                         << std::fixed << std::setprecision(5) << std::setw(11) << std::right << std::setfill(' ') << "Traveltime" << " ";
//                         if (IP.get_run_mode() == DO_INVERSION || IP.get_run_mode() == INV_RELOC) {
//                             fout << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << "Ks" << " "
//                             // << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << "Tadj" << " "
//                             << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << "Ks_update" << " ";
//                         }
//                         fout << std::endl;
//                 } else
//                     fout.open(fname.c_str(), std::ios_base::app); // append
//                 if (!fout.is_open()) {
//                     std::cout << "ERROR: cannot open file " << fname << std::endl;
//                     exit(1);
//                 }

//                 // iterate over nodes skipping the last layer for ignoring gap filling nodes for h5 output

//                 CUSTOMREAL* nodes_coords_p = grid.get_nodes_coords_p();     // dep,lat,lon
//                 CUSTOMREAL* nodes_coords_t = grid.get_nodes_coords_t();
//                 CUSTOMREAL* nodes_coords_r = grid.get_nodes_coords_r();
//                 std::vector<CUSTOMREAL> slowness = get_grid_data(grid.get_fun());
//                 std::vector<CUSTOMREAL> xi = get_grid_data(grid.get_xi());
//                 std::vector<CUSTOMREAL> eta = get_grid_data(grid.get_eta());
//                 std::vector<CUSTOMREAL> Ks, Ks_update;
//                 if (IP.get_run_mode() == DO_INVERSION || IP.get_run_mode() == INV_RELOC) {
//                     //std::vector<CUSTOMREAL> Tadj = get_grid_data(grid.get_Tadj());
//                     Ks = get_grid_data(grid.get_Ks());
//                     Ks_update = get_grid_data(grid.get_Ks_update());
//                 }
//                 std::vector<CUSTOMREAL> T = get_grid_data(grid.get_T());

//                 for (int k = 0; k < loc_K_vis; k++){
//                     for (int j = 0; j < loc_J_vis; j++){
//                         for (int i = 0; i < loc_I_vis; i++){
//                             int idx = I2V_VIS(i,j,k);
//                             fout<< std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << radius2depth(nodes_coords_r[idx]) << " "
//                                 << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << nodes_coords_t[idx] << " "
//                                 << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << nodes_coords_p[idx] << " "
//                                 << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << slowness[idx] << " "
//                                 << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << xi[idx] << " "
//                                 << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << eta[idx] << " "
//                                 << std::fixed << std::setprecision(7) << std::setw(9) << std::right << std::setfill(' ') << _1_CR/slowness[idx] << " "
//                                 << std::fixed << std::setprecision(5) << std::setw(11) << std::right << std::setfill(' ') << T[idx] << " ";

//                                 if (IP.get_run_mode() == DO_INVERSION || IP.get_run_mode() == INV_RELOC) {
//                                     fout << std::fixed << std::setprecision(7) << std::setw(12) << std::right << std::setfill(' ') << Ks[idx] << " "
//                                     // << std::fixed << std::setprecision(7) << std::setw(12) << std::right << std::setfill(' ') << Tadj[idx] << " "
//                                     << std::fixed << std::setprecision(7) << std::setw(12) << std::right << std::setfill(' ') << Ks_update[idx] << " ";
//                                 }
//                                 fout << std::endl;
//                         }
//                     }
//                 }
//                 fout.close();
//             }


//         } // end for i_rank
//     } // end if subdom_main

//     // synchronize
//     synchronize_all_inter();
// }


void IO_utils::write_T0v(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "T0v";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_T0v(), i_inv, src_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "T0v_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_T0v());
    }
}


void IO_utils::write_u(Grid& grid) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "u";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_u(), 0, src_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "u";
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_u());
    }
}


void IO_utils::write_tau(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "tau";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_tau(), i_inv, src_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "tau_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_tau());
    }
}

//
//void IO_utils::write_tmp_tau(Grid& grid, int i_iter) {
//    if (output_format==OUTPUT_FORMAT_HDF5){
//#ifdef USE_HDF5
//        std::string h5_dset_name = "tau_iter_" + std::to_string(i_iter);
//        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_tau(), i_iter);
//#else
//        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
//        exit(1);
//#endif
//    } else if (output_format==OUTPUT_FORMAT_ASCII){
//        std::string dset_name = "tau_tmp_iter_" + int2string_zero_fill(i_iter);
//        std::string fname = create_fname_ascii(dset_name);
//        write_data_ascii(grid, fname, grid.get_tau());
//    }
//}
//

void IO_utils::write_T(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "time_field";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_T(), i_inv, src_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if(output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "time_field_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_T());
    }
}


void IO_utils::write_T_tmp(Grid& grid) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        // temporally replice a filename for T_tmp
        std::string h5_fname_back = h5_output_fname;
        h5_output_fname = h5_output_fname_tmp;

        // create file if not exist
        std::string out_file_full = output_dir+"/"+h5_output_fname;
        if (!is_file_exist(out_file_full.c_str()))
            h5_create_file_by_group_main(h5_output_fname);

        // dataset name
        std::string h5_dset_name = "T_tmp";
        bool for_tmp_db = true;
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_T(), 0, src_data, for_tmp_db);

        // restore filename
        h5_output_fname = h5_fname_back;
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if(output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "T_tmp";
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_T());
    }
}


void IO_utils::write_residual(Grid& grid) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "residual";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_residual(), 0, src_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    }else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "residual";
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_residual());
    }

}


void IO_utils::write_adjoint_field(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "adjoint_field";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Tadj(), i_inv, src_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "adjoint_field_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_Tadj());
    }
}


void IO_utils::write_fun(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "fun";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_fun(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "fun_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_fun());
    }
}


void IO_utils::write_xi(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "xi";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_xi(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "xi_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_xi());
    }
}


void IO_utils::write_eta(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "eta";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_eta(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "eta_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_eta());
    }
}


void IO_utils::write_a(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "fac_a";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_a(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "fac_a_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_a());
    }
}


void IO_utils::write_b(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "fac_b";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_b(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "fac_b_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_b());
    }

}


void IO_utils::write_c(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "fac_c";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_c(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "fac_c_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_c());
    }
}


void IO_utils::write_f(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "fac_f";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_f(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "fac_f_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_f());
    }
}


void IO_utils::write_Ks(Grid& grid, int i_inv) {
    if (!subdom_main) return;

   if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Ks";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Ks(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Ks_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Ks());
    }
}


void IO_utils::write_Kxi(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Kxi";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Kxi(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Kxi_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Kxi());
    }
}


void IO_utils::write_Keta(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Keta";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Keta(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Keta_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Keta());
    }
}


void IO_utils::write_Ks_density(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Ks_density";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Ks_density(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Ks_density_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Ks_density());
    }
}

void IO_utils::write_Kxi_density(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Kxi_density";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Kxi_density(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Kxi_density_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Kxi_density());
    }
}

void IO_utils::write_Keta_density(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Keta_density";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Keta_density(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Keta_density_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Keta_density());
    }
}

void IO_utils::write_Ks_over_Kden(Grid& grid, int i_inv) {
    if (!subdom_main) return;

   if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Ks_over_Kden";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Ks(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Ks_over_Kden_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Ks());
    }
}

void IO_utils::write_Kxi_over_Kden(Grid& grid, int i_inv) {
    if (!subdom_main) return;

   if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Kxi_over_Kden";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Kxi(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Ks_over_Kden_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Ks());
    }
}

void IO_utils::write_Keta_over_Kden(Grid& grid, int i_inv) {
    if (!subdom_main) return;

   if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Keta_over_Kden";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Keta(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Ks_over_Kden_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Ks());
    }
}

void IO_utils::write_Ks_update(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Ks_update";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Ks_update(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Ks_update_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Ks_update());
    }
}


void IO_utils::write_Kxi_update(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Kxi_update";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Kxi_update(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Kxi_update_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Kxi_update());
    }
}


void IO_utils::write_Keta_update(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Keta_update";
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Keta_update(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if (output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "Keta_update_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii_model(dset_name);
        write_data_ascii(grid, fname, grid.get_Keta_update());
    }
}

// void IO_utils::write_Kdensity_update(Grid& grid, int i_inv) {
//     if (!subdom_main) return;

//     if (output_format==OUTPUT_FORMAT_HDF5){
// #ifdef USE_HDF5
//         std::string h5_dset_name = "Kdensity_update";
//         write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Kdensity_update(), i_inv, model_data);
// #else
//         std::cout << "ERROR: HDF5 is not enabled" << std::endl;
//         exit(1);
// #endif
//     } else if (output_format==OUTPUT_FORMAT_ASCII){
//         std::string dset_name = "Kdensity_update_inv_" + int2string_zero_fill(i_inv);
//         std::string fname = create_fname_ascii_model(dset_name);
//         write_data_ascii(grid, fname, grid.get_Kdensity_update());
//     }
// }

void IO_utils::write_Ks_descent_dir(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Ks_descent_dir_local_inv_" + int2string_zero_fill(i_inv);
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Ks_descent_dir(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
        } else if (output_format==OUTPUT_FORMAT_ASCII){
            std::string dset_name = "Ks_descent_dir_local_inv_" + int2string_zero_fill(i_inv);
            std::string fname = create_fname_ascii_model(dset_name);
            write_data_ascii(grid, fname, grid.get_Ks_descent_dir());
        }
    }


void IO_utils::write_Kxi_descent_dir(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Kxi_descent_dir_local_inv_" + int2string_zero_fill(i_inv);
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Kxi_descent_dir(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
        } else if (output_format==OUTPUT_FORMAT_ASCII){
            std::string dset_name = "Kxi_descent_dir_local_inv_" + int2string_zero_fill(i_inv);
            std::string fname = create_fname_ascii_model(dset_name);
            write_data_ascii(grid, fname, grid.get_Kxi_descent_dir());
        }
    }


void IO_utils::write_Keta_descent_dir(Grid& grid, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "Keta_descent_dir_local_inv_" + int2string_zero_fill(i_inv);
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_Keta_descent_dir(), i_inv, model_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
        } else if (output_format==OUTPUT_FORMAT_ASCII){
            std::string dset_name = "Keta_descent_dir_local_inv_" + int2string_zero_fill(i_inv);
            std::string fname = create_fname_ascii_model(dset_name);
            write_data_ascii(grid, fname, grid.get_Keta_descent_dir());
        }
    }


void IO_utils::write_T_merged(Grid& grid, InputParams& IP, int i_inv) {
    if (!subdom_main) return;

    if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
        std::string h5_dset_name = "time_field";
        std::string h5_dset_name_merged = "time_field_merged_inv_" + int2string_zero_fill(i_inv);
        bool inverse_field = false;
        write_data_merged_h5(grid, h5_output_fname, h5_group_name_data, h5_dset_name_merged, grid.get_T(), i_inv, inverse_field);
        write_data_h5(grid, h5_group_name_data, h5_dset_name, grid.get_T(), i_inv, src_data);
#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif
    } else if(output_format==OUTPUT_FORMAT_ASCII){
        std::string dset_name = "time_field_inv_" + int2string_zero_fill(i_inv);
        std::string fname = create_fname_ascii(dset_name);
        write_data_ascii(grid, fname, grid.get_T());
    }
}


void IO_utils::write_merged_model(Grid& grid, InputParams& IP, std::string fname) {
    // this function is called only from simulation group == 0
    if (id_sim == 0 && subdom_main) {

        if (output_format==OUTPUT_FORMAT_HDF5){
#ifdef USE_HDF5
            // create file
            h5_create_file_by_group_main(fname);

            std::string gname_dummy = "dummy";
            std::string dset_name = "vel";
            bool inverse_field = true;
            write_data_merged_h5(grid, fname, gname_dummy, dset_name, grid.fun_loc, inverse_field);
            dset_name = "eta";
            inverse_field = false;
            write_data_merged_h5(grid, fname, gname_dummy, dset_name, grid.eta_loc, inverse_field);
            dset_name = "xi";
            inverse_field = false;
            write_data_merged_h5(grid, fname, gname_dummy, dset_name, grid.xi_loc, inverse_field);
            //dset_name = "zeta";
            //inverse_field = false;
            //write_data_merged_h5(grid, fname, gname_dummy, dset_name, grid.zeta_loc, inverse_field);


#else
            std::cout << "ERROR: HDF5 is not enabled" << std::endl;
            exit(1);
#endif
        } else if (output_format==OUTPUT_FORMAT_ASCII){
            // check if the fname finish with .h5
            if (fname.find(".h5") != std::string::npos) {
                // remove .h5
                fname.erase(fname.end()-3, fname.end());
            }

            fname = create_fname_ascii_model(fname);
            write_data_merged_ascii(grid, fname);
        }


    } // end id_sim == 0 && subdom_main
}


void IO_utils::prepare_grid_inv_xdmf(int i_inv) {
    if ((output_format==OUTPUT_FORMAT_HDF5) && id_subdomain==0 && subdom_main){
        std::string str_inv = "inv_" + int2string_zero_fill(i_inv);

        inv_grid = inversions->InsertEndChild(doc->NewElement("Grid"));
        inv_grid->ToElement()->SetAttribute("Name", str_inv.c_str());
        inv_grid->ToElement()->SetAttribute("GridType", "Uniform");

        // Topology node in grid
        tinyxml2::XMLNode* topology_grid = inv_grid->InsertEndChild( doc->NewElement( "Topology" ) );
        topology_grid->ToElement()->SetAttribute("Reference", "/Xdmf/Domain/Topology");
        // Geometry node in grid
        tinyxml2::XMLNode* geometry_grid = inv_grid->InsertEndChild( doc->NewElement( "Geometry" ) );
        geometry_grid->ToElement()->SetAttribute("Reference", "/Xdmf/Domain/Geometry");
        // insert time
        tinyxml2::XMLNode* time = inv_grid->InsertEndChild( doc->NewElement( "Time" ) );
        time->ToElement()->SetAttribute("Value", std::to_string(i_inv).c_str());

    }
}


void IO_utils::insert_data_xdmf(std::string& group_name, std::string& dset_name_xdmf, std::string& dset_name_h5) {

    if (id_subdomain==0 && subdom_main){ // only the main of each subdomain write
        std::string num_nodes = std::to_string(nnodes_glob);

        // Attribute node in grid
        tinyxml2::XMLNode* attribute = inv_grid->InsertEndChild( doc->NewElement( "Attribute" ) );
        attribute->ToElement()->SetAttribute("Name", dset_name_xdmf.c_str());
        attribute->ToElement()->SetAttribute("AttributeType", "Scalar");
        attribute->ToElement()->SetAttribute("Center", "Node");
        // DataItem node in attribute
        tinyxml2::XMLNode* dataItem = attribute->InsertEndChild( doc->NewElement( "DataItem" ) );
        dataItem->ToElement()->SetAttribute("ItemType", "Uniform");
        dataItem->ToElement()->SetAttribute("Format", "HDF");
        dataItem->ToElement()->SetAttribute("NumberType", type_name.c_str());
        dataItem->ToElement()->SetAttribute("Precision", type_size.c_str());
        dataItem->ToElement()->SetAttribute("Dimensions", num_nodes.c_str());
        // set dataset path in dataItem
        std::string h5_dset_path = h5_output_fname + ":/" + group_name + "/" + dset_name_h5;
        dataItem->InsertEndChild( doc->NewText( h5_dset_path.c_str() ) );
    }
}


void IO_utils::read_model(std::string& model_fname, const char* dset_name_in, CUSTOMREAL* darr, \
                          int offset_i, int offset_j, int offset_k) {

    std::string dset_name(dset_name_in);

    if (output_format==OUTPUT_FORMAT_HDF5){

#ifdef USE_HDF5
        // open h5 file
        h5_open_file_collective_input(model_fname);

        if (inter_sub_rank == 0) {
            std::cout << "--- read model data " << dset_name << " from h5 file ---" << std::endl;
        }

        // prepare temporary buffer in float precision for reading model data
        // #TODO: need to be modify here for directly read data in double precision
        float* f_darr = new float[loc_I*loc_J*loc_K];

        // read data
        int dims_model[3] = {loc_I, loc_J, loc_K};
        int offset_model[3] = {offset_i, offset_j, offset_k};
        h5_read_array(dset_name, 3, dims_model, f_darr, offset_model, false);

        // copy values to darr
        for (int i=0; i<loc_I; i++) {
            for (int j=0; j<loc_J; j++) {
                for (int k=0; k<loc_K; k++) {
                    darr[I2V(i,j,k)] = static_cast<CUSTOMREAL> (f_darr[I2V(i,j,k)]);
                }
            }
        }
        // close file
        h5_close_file_collective();

        // close temporary buffer
        delete[] f_darr;

#else
        std::cout << "ERROR: HDF5 is not enabled" << std::endl;
        exit(1);
#endif

        } else if (output_format==OUTPUT_FORMAT_ASCII){

            if (inter_sub_rank == 0) {
                std::cout << "--- read model data " << dset_name << " from ASCII file ---" << std::endl;
            }

            CUSTOMREAL tmp_eta, tmp_xi, tmp_zeta, tmp_vel;

            // read data

            std::ifstream model_file(model_fname.c_str());
            if (!model_file.is_open()) {
                std::cout << "Error: cannot open model file " << model_fname << std::endl;
                exit(1);
            }

            int line_count = 0;
            int i,j,k;

            if (if_test) {
                while(model_file >> tmp_eta >> tmp_xi >> tmp_zeta >> tmp_vel) {
                    k =  line_count /   (ngrid_i*ngrid_j);
                    j = (line_count  - k*ngrid_i*ngrid_j) /  ngrid_i;
                    i =  line_count  - k*ngrid_i*ngrid_j - j*ngrid_i;

                    if (i>=offset_i && i<offset_i+loc_I && j>=offset_j && j<offset_j+loc_J && k>=offset_k && k<offset_k+loc_K) {
                        if (!dset_name.compare(std::string("xi")))
                            darr[I2V(i-offset_i,j-offset_j,k-offset_k)] = static_cast<CUSTOMREAL> (tmp_xi);
                        else if (!dset_name.compare(std::string("eta")))
                            darr[I2V(i-offset_i,j-offset_j,k-offset_k)] = static_cast<CUSTOMREAL> (tmp_eta);
                        else if (!dset_name.compare(std::string("zeta")))
                            darr[I2V(i-offset_i,j-offset_j,k-offset_k)] = static_cast<CUSTOMREAL> (tmp_zeta);
                        else if (!dset_name.compare(std::string("vel")))
                            darr[I2V(i-offset_i,j-offset_j,k-offset_k)] = static_cast<CUSTOMREAL> (tmp_vel);
                    }

                    line_count++;
                }
            } else {
                while(model_file >> tmp_eta >> tmp_xi >> tmp_zeta >> tmp_vel) {
                    k = line_count /(ngrid_i*ngrid_j);
                    j = (line_count - k*ngrid_i*ngrid_j) /  ngrid_i;
                    i = line_count  - k*ngrid_i*ngrid_j - j*ngrid_i;

                    if (i>=offset_i && i<offset_i+loc_I && j>=offset_j && j<offset_j+loc_J && k>=offset_k && k<offset_k+loc_K) {
                        if (!dset_name.compare(std::string("xi")))
                            darr[I2V(i-offset_i,j-offset_j,k-offset_k)] = static_cast<CUSTOMREAL> (tmp_xi);
                        else if (!dset_name.compare(std::string("eta")))
                            darr[I2V(i-offset_i,j-offset_j,k-offset_k)] = static_cast<CUSTOMREAL> (tmp_eta);
                        else if (!dset_name.compare(std::string("zeta")))
                            darr[I2V(i-offset_i,j-offset_j,k-offset_k)] = static_cast<CUSTOMREAL> (tmp_zeta);
                        else if (!dset_name.compare(std::string("vel")))
                            darr[I2V(i-offset_i,j-offset_j,k-offset_k)] = static_cast<CUSTOMREAL> (tmp_vel);
                    }

                    line_count++;
                }
            }
        }

}


// read travel time data from file
void IO_utils::read_T(Grid& grid) {
    if (!subdom_main) return;

    if (output_format == OUTPUT_FORMAT_HDF5) {
        // read traveltime field from HDF5 file
#ifdef USE_HDF5
        // h5_group_name_data = "src_" + std::to_string(id_sim_src);
        h5_group_name_data = "src_" + name_sim_src;
        std::string h5_dset_name = "time_field_inv_" + int2string_zero_fill(0);
        read_data_h5(grid, grid.vis_data, h5_group_name_data, h5_dset_name);
#else
        std::cerr << "Error: HDF5 is not enabled." << std::endl;
        exit(1);
#endif
    } else if (output_format == OUTPUT_FORMAT_ASCII) {
        // read traveltime field from ASCII file
        std::string dset_name = "time_field_inv_" + int2string_zero_fill(0);
        std::string filename = create_fname_ascii(dset_name);

        read_data_ascii(grid, filename);
    }

    // set to T_loc array from grid.vis_data
    grid.set_array_from_vis(grid.T_loc);
}


void IO_utils::read_T_tmp(Grid& grid) {
    if (!subdom_main) return;

    if (output_format == OUTPUT_FORMAT_HDF5) {
        // read traveltime field from HDF5 file
#ifdef USE_HDF5
        // temporally replice a filename for T_tmp
        std::string h5_fname_back = h5_output_fname;
        h5_output_fname = h5_output_fname_tmp;

        // h5_group_name_data = "src_" + std::to_string(id_sim_src);
        h5_group_name_data = "src_" + name_sim_src;
        std::string h5_dset_name = "T_tmp";

        read_data_h5(grid, grid.vis_data, h5_group_name_data, h5_dset_name);

        // recover h5_output_fname
        h5_output_fname = h5_fname_back;
#else
        std::cerr << "Error: HDF5 is not enabled." << std::endl;
        exit(1);
#endif
    } else if (output_format == OUTPUT_FORMAT_ASCII) {
        // read traveltime field from ASCII file
        std::string dset_name = "T_tmp";
        std::string filename = create_fname_ascii(dset_name);

        read_data_ascii(grid, filename);
    }

    // set to T_loc array from grid.vis_data
    grid.set_array_from_vis(grid.T_loc);
}





void IO_utils::read_data_ascii(Grid& grid, std::string& fname){
    // read data in ascii file
    if (myrank == 0 && if_verbose)
        std::cout << "--- read data from ascii file " << fname << " ---" << std::endl;

    // get offset
    int offset_this_rank = grid.get_offset_nnodes_vis();
    int ngrid_this_rank  = grid.get_ngrid_total_vis();

    // collective read
    for (int i_rank = 0; i_rank < n_subdomains; i_rank++) {
        if (i_rank == myrank) {
            std::string line;
            std::ifstream in_file(fname);

            int line_count = 0;

            while (std::getline(in_file, line)) {
                std::istringstream iss(line);
                CUSTOMREAL v_tmp;
                if (!(iss >> v_tmp)) { break; } // error

                // set to grid.vis_data
                if (line_count >= offset_this_rank && line_count < offset_this_rank + ngrid_this_rank)
                    grid.vis_data[line_count - offset_this_rank] = v_tmp;

                line_count++;
            }

            in_file.close();
        }

        // synchronize
        synchronize_all_inter();
    }

}


//
// hdf5 low level utilities
//

#ifdef USE_HDF5


// read data from hdf5 file, which is in output format of TomoATT
void IO_utils::read_data_h5(Grid& grid, CUSTOMREAL* arr, std::string h5_group_name, std::string h5_dset_name) {
    // write true solution to h5 file
    if (myrank == 0 && if_verbose)
        std::cout << "--- read data " << h5_dset_name << " from h5 file " << h5_output_fname << " ---" << std::endl;

    // open file collective
    h5_open_file_collective(h5_output_fname);

    // open group collective
    h5_open_group_collective(h5_group_name);

    // get offset
    int dims_ngrid[1] = {grid.get_ngrid_total_vis()};
    allreduce_i_single(dims_ngrid[0], nnodes_glob);
    int offset_this[1] = {grid.get_offset_nnodes_vis()};

    // read data from h5 file
    h5_read_array(h5_dset_name, 1, dims_ngrid, arr, offset_this, true);

    // close group
    h5_close_group_collective();
    // close file
    h5_close_file_collective();
}


void IO_utils::h5_create_file_by_group_main(std::string& fname) {
    // initialize hdf5

    // create file by the first rank of this mpi group
    if (myrank == 0) {
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        //H5Pset_fapl_mpio(plist_id, inter_sub_comm, MPI_INFO_NULL);

        std::string fout = output_dir + "/" + fname;
        file_id = H5Fcreate(fout.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        //file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        H5Pclose(plist_id);
        H5Fclose(file_id);
    }
    synchronize_all_inter();
}

void IO_utils::h5_open_file_by_group_main(std::string& fname){
    // open file by the first rank of first mpi group
    if (myrank == 0) {
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        std::string fout = output_dir + "/" + fname;
        file_id = H5Fopen(fout.c_str(), H5F_ACC_RDWR, plist_id);
    }
}

// this function need to be read by only subdomain rank = 0
void IO_utils::h5_open_file_collective(std::string& fname){
    // open file by all ranks
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, inter_sub_comm, MPI_INFO_NULL);
    std::string fout = output_dir + "/" + fname;
    file_id = H5Fopen(fout.c_str(), H5F_ACC_RDWR, plist_id);
    H5Pclose(plist_id);
}


void IO_utils::h5_open_file_collective_input(std::string& fname){
    // open file by all ranks
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, inter_sub_comm, MPI_INFO_NULL);
    std::string fout = fname;
    file_id = H5Fopen(fout.c_str(), H5F_ACC_RDWR, plist_id);
}


void IO_utils::h5_close_file_by_group_main(){
    if(myrank == 0) {
        H5Pclose(plist_id);
        H5Fclose(file_id);
    }
}

void IO_utils::h5_close_file_collective(){
    //H5Pclose(plist_id);
    H5Fclose(file_id);
}


void IO_utils::h5_create_group_by_group_main(std::string& group_name){
    // create group by only main rank
    if (myrank == 0) {
        // check if group exists
        if (!H5Lexists(file_id, group_name.c_str(), H5P_DEFAULT)) {
            group_id = H5Gcreate(file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            // close group
            H5Gclose(group_id);
        }
    }
}

void IO_utils::h5_open_group_by_group_main(std::string& group_name){
    // open group by mpi group main
    if(myrank==0)
        group_id = H5Gopen(file_id, group_name.c_str(), H5P_DEFAULT);
}

void IO_utils::h5_open_group_collective(std::string& group_name){
    // open group by all ranks
    group_id = H5Gopen(file_id, group_name.c_str(), H5P_DEFAULT);
}


void IO_utils::h5_close_group_by_group_main(){
    // close group by all ranks
    if(myrank==0)
        H5Gclose(group_id);
}

void IO_utils::h5_close_group_collective(){
    // close group by all ranks
    H5Gclose(group_id);
}


void IO_utils::h5_create_dataset(std::string& dset_name, int ndims, int* dims, int dtype, bool no_group){
    /*
    create dataset in a group by all ranks

    dset_name: name of the dataset
    ndims: number of dimensions
    dims: number of elements in each dimension
    dtype: data type 0: bool, 1: int, 2: float, 3: double
    no_group: if true, create dataset in the file, not in a group
    */

    if (no_group)
        group_id = file_id; // create dataset in the file

    // gather data size from all rank in this simulation group
    // subdomain dependent size need to be in the first element of dims
    int *dims_all = new int[ndims];

    // for 1d array
    if (ndims != 3) {
        for (int i = 0; i < ndims; i++) {
            if (i==0)
                allreduce_i_single(dims[i], dims_all[i]);
            else
                dims_all[i] = dims[i];
        }
    } else {
        for (int i = 0; i < ndims; i++) {
            dims_all[i] = dims[i];
        }
    }
    // dataset is created only by the main rank
    // thus the file need to be opened by the main rank
    if (myrank == 0) {
        // create a dataspace with the fixed size
        hsize_t *dims_h5 = new hsize_t[ndims];
        for (int i = 0; i < ndims; i++) {
            dims_h5[i] = dims_all[i];
        }

        // check if the dataset exists
        if (!H5Lexists(group_id, dset_name.c_str(), H5P_DEFAULT) ) {
            space_id = H5Screate_simple(ndims, dims_h5, NULL);

            // create dataset
            switch (dtype)
            {
            case 0:
                dset_id = H5Dcreate(group_id, dset_name.c_str(), H5T_NATIVE_HBOOL, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                break;
            case 1:
                dset_id = H5Dcreate(group_id, dset_name.c_str(), H5T_NATIVE_INT, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                break;
            case 2:
                dset_id = H5Dcreate(group_id, dset_name.c_str(), H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                break;
            case 3:
                dset_id = H5Dcreate(group_id, dset_name.c_str(), H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                break;
            }

            // error handle
            if (dset_id < 0) {
                std::cout << "Error: H5Dcreate for " << dset_name << " failed" << std::endl;
                exit(1);
            }

            // close dataset
            H5Dclose(dset_id);
            // close dataspace
            H5Sclose(space_id);
        }

        delete[] dims_h5;
    }

    delete[] dims_all;
}


void IO_utils::h5_create_and_write_dataset_2d(std::string& dset_name, int ndims, int* dims, int dtype, CUSTOMREAL* data){
    /*
    create dataset in a group by all ranks

    dset_name: name of the dataset
    ndims: number of dimensions
    dims: number of elements in each dimension
    dtype: data type 0: bool, 1: int, 2: float, 3: double
    */

    hid_t space_id_2d=0, dset_id_2d=0;

    // dataset is created only by the main rank
    // thus the file need to be opened by the main rank
    if (myrank == 0) {
        // create a dataspace with the fixed size
        hsize_t *dims_h5 = new hsize_t[ndims];
        for (int i = 0; i < ndims; i++) {
            dims_h5[i] = dims[i];
        }
        space_id_2d = H5Screate_simple(ndims, dims_h5, NULL);

        // create dataset
        switch (dtype)
        {
        case 0:
            dset_id_2d = H5Dcreate(file_id_2d, dset_name.c_str(), H5T_NATIVE_HBOOL, space_id_2d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status     = H5Dwrite(dset_id_2d, H5T_NATIVE_HBOOL, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            break;
        case 1:
            dset_id_2d = H5Dcreate(file_id_2d, dset_name.c_str(), H5T_NATIVE_INT, space_id_2d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status     = H5Dwrite(dset_id_2d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            break;
        case 2:
            dset_id_2d = H5Dcreate(file_id_2d, dset_name.c_str(), H5T_NATIVE_FLOAT, space_id_2d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status  = H5Dwrite(dset_id_2d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            break;
        case 3:
            dset_id_2d = H5Dcreate(file_id_2d, dset_name.c_str(), H5T_NATIVE_DOUBLE, space_id_2d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status     = H5Dwrite(dset_id_2d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            break;
        }

        // error handle
        if (dset_id_2d < 0) {
            std::cout << "Error: H5Dcreate for " << dset_name << " failed" << std::endl;
            exit(1);
        }

        // close dataset
        H5Dclose(dset_id_2d);
        // close dataspace
        H5Sclose(space_id_2d);

        delete[] dims_h5;
    }

}


void IO_utils::h5_open_dataset(std::string& dset_name){
    // open dataset in a group by all ranks
    dset_id = H5Dopen(group_id, dset_name.c_str(), H5P_DEFAULT);

    if (dset_id < 0) {
        std::cout << "Error: H5Dopen for " << dset_name << " failed" << std::endl;
        exit(1);
    }
}

void IO_utils::h5_open_dataset_no_group(std::string& dset_name){
    // open dataset in a group by all ranks
    dset_id = H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT);

    if (dset_id < 0) {
        std::cout << "Error: H5Dopen for " << dset_name << " failed" << std::endl;
        exit(1);
    }
}

void IO_utils::h5_close_dataset(){
    // close dataset in a group by all ranks
    H5Dclose(dset_id);
}

template <typename T>
void IO_utils::h5_write_array(std::string& dset_name, int rank, int* dims_in, T* data, int offset_in, int offset_in2, int offset_in3, bool no_group) {
    // write a data array to a dataset by all ranks

    // use group_id as file_id if no_group is true
    if (no_group)
        group_id = file_id;

    hsize_t* offset = new hsize_t[rank];
    hsize_t* count  = new hsize_t[rank];
    hsize_t* stride = new hsize_t[rank];
    hsize_t* block  = new hsize_t[rank];

    if (rank != 3){
        for (int i_rank = 0; i_rank < rank; i_rank++) {
            if (i_rank == 0)
                offset[i_rank] = offset_in;
            else if (i_rank == 1)
                offset[i_rank] = offset_in2;
            else if (i_rank == 2)
                offset[i_rank] = offset_in3;
            count[i_rank]  = dims_in[i_rank];
            block[i_rank]  = 1;
            stride[i_rank] = 1;
        }
    } else {
        for (int i_rank = 0; i_rank < rank; i_rank++) {
            if (i_rank == 0){
                offset[i_rank] = offset_in;
                count[i_rank]  = dims_in[0];
            }
            else if (i_rank == 1){
                offset[i_rank] = offset_in2;
                count[i_rank]  = dims_in[1];
            }
            else if (i_rank == 2){
                offset[i_rank] = offset_in3;
                count[i_rank]  = dims_in[2];
            }

            block[i_rank]  = 1;
            stride[i_rank] = 1;
        }
    }

    // check data type of array
    int dtype = check_data_type(data[0]);

    // open dataset
    h5_open_dataset(dset_name);

    // select hyperslab
    mem_dspace_id  = H5Screate_simple(rank, count, NULL);
    file_dspace_id = H5Dget_space(dset_id);

    H5Sselect_hyperslab(file_dspace_id, H5S_SELECT_SET, offset, stride, count, block);

    // create dataset prop list
    plist_id_dset = H5Pcreate(H5P_DATASET_XFER);
    //H5Pset_buffer(plist_id_dset, BUF_SIZE, NULL, NULL); // this will be important for machine dependent tuning
    H5Pset_dxpl_mpio(plist_id_dset, HDF5_IO_MODE);

    // check if this writing is done collectively
    H5D_mpio_actual_io_mode_t actualIOMode;
    H5Pget_mpio_actual_io_mode(plist_id_dset, &actualIOMode);

    if (actualIOMode == H5D_MPIO_NO_COLLECTIVE) {
        if (rank == 0) {
            std::cout << "Warning: Collective I/O is not enabled." << std::endl;
        }
    }

    // write array
    switch (dtype)
    {
        case 0: // write bool
            status = H5Dwrite(dset_id, H5T_NATIVE_HBOOL, mem_dspace_id, file_dspace_id, plist_id_dset, data);
            break;
        case 1: // write int
            status = H5Dwrite(dset_id, H5T_NATIVE_INT, mem_dspace_id, file_dspace_id, plist_id_dset, data);
            break;
        case 2: // write float
            status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, mem_dspace_id, file_dspace_id, plist_id_dset, data);
            break;
        case 3: // write double
            status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, mem_dspace_id, file_dspace_id, plist_id_dset, data);
            break;
    }

    // close dataset prop
    H5Pclose(plist_id_dset);
    // close dataspace
    H5Sclose(mem_dspace_id);
    H5Sclose(file_dspace_id);
    // close dataset
    h5_close_dataset();

    delete[] offset;
    delete[] count;
    delete[] stride;
    delete[] block;
}

template <typename T>
void IO_utils::h5_read_array(std::string& dset_name, int rank, int* dims_in, T* data, int* offset_in, bool in_group) {

    hsize_t* offset = new hsize_t[rank];
    hsize_t* count  = new hsize_t[rank];
    hsize_t* stride = new hsize_t[rank];
    hsize_t* block  = new hsize_t[rank];

    if (rank == 3){ // used for reading input model
        offset[0] = offset_in[2];
        count[0]  = dims_in[2];
        offset[1] = offset_in[1];
        count[1]  = dims_in[1];
        offset[2] = offset_in[0];
        count[2]  = dims_in[0];
    } else if (rank == 1) { // used for reading output data from TomoATT
        offset[0] = offset_in[0];
        count[0]  = dims_in[0];
    }

    for (int i_rank = 0; i_rank < rank; i_rank++) {
        stride[i_rank] = 1;
        block[i_rank]  = 1;
    }

    // check data type of array
    int dtype = check_data_type(data[0]);

    // open dataset
    if (!in_group)
        h5_open_dataset_no_group(dset_name);
    else
        h5_open_dataset(dset_name);

    // select hyperslab
    mem_dspace_id  = H5Screate_simple(rank, count, NULL);
    file_dspace_id = H5Dget_space(dset_id);
    H5Sselect_hyperslab(file_dspace_id, H5S_SELECT_SET, offset, stride, count, block);

    // create dataset prop list
    plist_id_dset = H5Pcreate(H5P_DATASET_XFER);
    //H5Pset_buffer(plist_id_dset, BUF_SIZE, NULL, NULL); // this will be important for machine dependent tuning
    H5Pset_dxpl_mpio(plist_id_dset, H5FD_MPIO_COLLECTIVE);

    // read array
    switch (dtype)
    {
        case 0: // read bool
            status = H5Dread(dset_id, H5T_NATIVE_HBOOL,  mem_dspace_id, file_dspace_id, plist_id_dset, data);
            break;
        case 1: // read int
            status = H5Dread(dset_id, H5T_NATIVE_INT,    mem_dspace_id, file_dspace_id, plist_id_dset, data);
            break;
        case 2: // read float
            status = H5Dread(dset_id, H5T_NATIVE_FLOAT,  mem_dspace_id, file_dspace_id, plist_id_dset, data);
            break;
        case 3: // read double
            status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, mem_dspace_id, file_dspace_id, plist_id_dset, data);
            break;
    }

    // close dataset prop
    H5Pclose(plist_id_dset);
    // close dataspace
    H5Sclose(mem_dspace_id);
    H5Sclose(file_dspace_id);
    // close dataset
    h5_close_dataset();

    delete[] offset;
    delete[] count;
    delete[] stride;
    delete[] block;
}

template <typename T>
void IO_utils::h5_read_array_simple(std::string& dset_name, T* data) {

    // check data type of array
    int dtype = check_data_type(data[0]);

    // open dataset
    h5_open_dataset_no_group(dset_name);

    switch (dtype)
    {
        case 0: // read bool
            status = H5Dread(dset_id, H5T_NATIVE_HBOOL, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            break;
        case 1: // read int
            status = H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            break;
        case 2: // read float
            status = H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            break;
        case 3: // read double
            status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            break;
    }

    // close dataset
    h5_close_dataset();

}

#endif // USE_HDF5