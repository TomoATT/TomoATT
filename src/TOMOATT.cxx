#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>
#include "mpi_funcs.h"
#include "config.h"
#include "utils.h"
#include "input_params.h"
#include "grid.h"
#include "io.h"
#include "source.h"
#include "receiver.h"
#include "kernel.h"
#include "model_update.h"
#include "main_routines_calling.h"
#include "eikonal_solver_2d.h"
#include "version.h"

#ifdef USE_CUDA
#include "cuda_initialize.cuh"
#endif

//#ifdef USE_BLAS
//#include "cblas.h"
//#endif

// TOMOATT main function
int main(int argc, char *argv[])
{
    // parse options
    parse_options(argc, argv);

    // initialize mpi
    initialize_mpi();

    stdout_by_rank_zero("------------------------------------------------------");
    stdout_by_rank_zero("TOMOATT calculation starting...");
    std::string version = "TOMOATT version: " + std::string(TOMOATT_VERSION);
    stdout_by_rank_zero(version.c_str());
    stdout_by_rank_zero("------------------------------------------------------");

    // read input file
    InputParams IP(input_file);

    // create output directory
    create_output_dir(output_dir);

#ifdef USE_CUDA
    // initialize cuda
    if(use_gpu) initialize_cuda();
#endif

#ifdef USE_SIMD
    // check SIMD type
    if (myrank==0)
        print_simd_type();
#endif

    // check the number of mpi processes and ndiv setting is consistent
    check_total_nprocs_and_ndiv();

    // split mpi communicator for simultaneous run, subdomain division and sweep parallelization
    split_mpi_comm();

    // assign source for each simultaneous run group
    IP.prepare_src_map();

    // initialize file IO object
    IO_utils io(IP); // create IO object for main and non-main process as well

    // initialize compute grids
    Grid grid(IP, io); // member objects are created in only the main process of subdomain groups

    // output inversion grid file (by main process)
    grid.inv_grid->write_inversion_grid_to_file();

    if (subdom_main) {
        // output grid data (grid data is only output in the main simulation)
        io.write_grid(grid);
    }

    // preapre teleseismic boundary conditions (do nothinng if no teleseismic source is defined)
    prepare_teleseismic_boundary_conditions(IP, grid, io);      // not ready for new version of src rec data

    synchronize_all_world();

    //
    // run main calculation routines depending on specified mode
    //
    if (IP.get_run_mode() == ONLY_FORWARD || IP.get_run_mode() == DO_INVERSION){
        run_forward_only_or_inversion(IP, grid, io);
    } else if (IP.get_run_mode() == TELESEIS_PREPROCESS) {
        // terminate the program here if the run mode is TELESEIS_PREPROCESS
        // because prepare_teleseismic_boundary_conditions is already called.
    } else if (IP.get_run_mode() == SRC_RELOCATION) {
        run_earthquake_relocation(IP, grid, io);
    } else if (IP.get_run_mode() == INV_RELOC) {
        run_inversion_and_relocation(IP,grid,io);
    } else {
        std::cerr << "Error: invalid run mode is specified." << std::endl;
        exit(1);
    }

    // output final state of the model
    if (IP.get_if_output_final_model()) {
        io.write_merged_model(grid, IP, "final_model.h5");
    }


    // finalize cuda
#ifdef USE_CUDA
    if (use_gpu) finalize_cuda();
#endif

    // finalize mpi
    finalize_mpi();

    stdout_by_rank_zero("------------------------------------------------------");
    stdout_by_rank_zero("TOMOATT calculation end.");
    stdout_by_rank_zero("------------------------------------------------------");

    return 0;
}