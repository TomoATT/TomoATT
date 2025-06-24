#ifndef MPI_FUNCS_H
#define MPI_FUNCS_H

#include <stdlib.h>
#include <mpi.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <numeric>
#include "utils.h"
#include "config.h"

#ifdef USE_OMP
#include <omp.h>
#endif

inline void initialize_mpi();
inline void finalize_mpi();

inline void mpi_debug(char const* str);

template<typename T>
inline std::vector<size_t> node_reordering(const std::vector<T>&);
inline int  check_total_nprocs_and_ndiv();
inline void synchronize_all();
inline void synchronize_all_inter_sim();
inline void synchronize_all_sim();
inline void synchronize_all_sub();
inline void synchronize_all_inter();
inline void synchronize_all_world();
inline void send_bool_single_sim(bool*, int);
inline void recv_bool_single_sim(bool*, int);
inline void send_i_single_sim(int*, int);
inline void recv_i_single_sim(int*, int);
inline void send_cr(CUSTOMREAL*, int, int);
inline void recv_cr(CUSTOMREAL*, int, int);
inline void isend_cr(CUSTOMREAL*, int, int, MPI_Request&);
inline void irecv_cr(CUSTOMREAL*, int, int, MPI_Request&);
inline void send_cr_single_sim(CUSTOMREAL *, int);
inline void recv_cr_single_sim(CUSTOMREAL *, int);
inline void send_str_sim(std::string,  int);
inline void recv_str_sim(std::string&, int);
inline void send_str(std::string,  int);
inline void recv_str(std::string&, int);
inline void allreduce_i_single(int&, int&);
inline void allreduce_cr_single(CUSTOMREAL&, CUSTOMREAL&);
inline void allreduce_i_inplace(int*, int);
inline void allreduce_i_sim_single_inplace(int&);
inline void allreduce_bool_inplace_inter_sim(bool*, int);
inline void allreduce_bool_inplace(bool*, int);
inline void allreduce_bool_inplace_sub(bool*, int);
inline void allreduce_bool_single_inplace(bool&);
inline void allreduce_bool_single_inplace_world(bool&);
inline void allreduce_cr_inplace(CUSTOMREAL*, int);
inline void allreduce_cr_sim(CUSTOMREAL*, int, CUSTOMREAL*);
inline void allreduce_cr_sim_inplace(CUSTOMREAL*, int);
inline void allreduce_cr_sim_single_inplace(CUSTOMREAL&);
inline void allgather_i_single(int*, int*);
inline void allgather_cr_single(CUSTOMREAL*, CUSTOMREAL*);
inline void allgather_str(const std::string&, std::vector<std::string>&);
inline void broadcast_bool_single(bool&, int);
inline void broadcast_bool_single_sub(bool&, int);
inline void broadcast_bool_single_inter_sim(bool&, int);
inline void broadcast_bool_inter_and_intra_sim(bool&, int);
inline void broadcast_i_single(int&, int);
inline void broadcast_i_single_inter_sim(int&, int);
inline void broadcast_i_single_sub(int&, int);
inline void broadcast_i_single_intra_sim(int&, int);
inline void broadcast_i_single_inter_and_intra_sim(int&, int);
inline void broadcast_f_single(float&, int);
inline void broadcast_cr(CUSTOMREAL* , int, int);
inline void broadcast_cr_single(CUSTOMREAL&, int);
inline void broadcast_cr_single_inplace(CUSTOMREAL&);
inline void broadcast_cr_inter_sim(CUSTOMREAL*, int, int);
inline void broadcast_str(std::string&, int);
inline void broadcast_str_inter_sim(std::string&, int);
inline void broadcast_str_sub(std::string&, int);
inline void broadcast_str_inter_and_intra_sim(std::string&, int);
inline void broadcast_cr_single_sub(CUSTOMREAL&, int);
inline void broadcast_cr_sub(CUSTOMREAL*, int, int);
inline void broadcast_cr_single_inter_and_intra_sim(CUSTOMREAL&, int);
inline void prepare_shm_array_cr(int, CUSTOMREAL*&, MPI_Win&);
inline void prepare_shm_array_bool(int, bool*&, MPI_Win&);
inline void cleanup_mpi_win(MPI_Win&);
inline void cleanup_mpi_wins(std::initializer_list<MPI_Win*>);
inline void init_mpi_wins(std::initializer_list<MPI_Win*>);

inline void wait_req(MPI_Request&);
inline void shm_fence(MPI_Win&);

inline void initialize_mpi(){
    // Initialize the MPI environment
#ifndef USE_OMP
    MPI_Init(NULL, NULL);
#else
    int provided;
    MPI_Init_thread(NULL,NULL,MPI_THREAD_FUNNELED,&provided);
    if(provided != MPI_THREAD_FUNNELED){
        std::cerr << "MPI_THREAD_FUNNELED is not supported" << std::endl;
        exit(1);
    }

    // currently no routine except src/rec weight calculation is parallelized by openmp
    // thus put 1 thread per process
    omp_set_num_threads(1);

    // show the number of threads
    int nthreads = omp_get_max_threads();

    // error check
    if (nthreads != 1){
        std::cerr << "number of threads is not 1" << std::endl;
        exit(1);
    }

    //if (world_rank == 0)
    //    std::cout << "Number of threads = " << nthreads << std::endl;
#endif
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // temporary use those parameters for reading input file
    nprocs = world_nprocs;
    myrank = world_rank;
    inter_sub_comm = MPI_COMM_WORLD;

    stdout_by_rank_zero("mpi initialized.");
}


inline void finalize_mpi(){

    synchronize_all_world();

    // free communicators if necessary
    //if (sub_comm != MPI_COMM_NULL)
    //    MPI_Comm_free(&sub_comm);
    //if (inter_sub_comm != MPI_COMM_WORLD)
    //    MPI_Comm_free(&inter_sub_comm);
    //if (sim_comm != MPI_COMM_NULL)
    //    MPI_Comm_free(&sim_comm);

    // Finalize the MPI environment.
    MPI_Finalize();

    stdout_by_rank_zero("mpi finalized.");
}


inline void mpi_debug(char const* str){
    std::cout << "rank: " << world_rank << ", " << str << std::endl;
    synchronize_all_world();
}


template<typename T>
inline std::vector<size_t> node_reordering(const std::vector<T>& vec){
    // reordering the nodes in the order of the number of particles
    // the first node has the most particles
    std::vector<size_t> idx(vec.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&vec](size_t i1, size_t i2) {return vec[i1] > vec[i2];});
    return idx;
}


inline std::vector<int> define_node_ids(std::vector<std::string>& mpi_node_names_pre){
    std::vector<int> mpi_node_ids(world_nprocs);
    std::vector<std::string> mpi_node_names_unique = mpi_node_names_pre;
    std::sort(mpi_node_names_unique.begin(), mpi_node_names_unique.end());
    mpi_node_names_unique.erase(std::unique(mpi_node_names_unique.begin(), mpi_node_names_unique.end()), mpi_node_names_unique.end());
    for (int irank = 0; irank < world_nprocs; irank++){
        for (long unsigned int i = 0; i < mpi_node_names_unique.size(); i++){
            if (mpi_node_names_pre[irank] == mpi_node_names_unique[i]){
                mpi_node_ids[irank] = i;
                break;
            }
        }
    }
    return mpi_node_ids;
}


inline int count_number_compute_nodes(std::vector<std::string> mpi_node_names) {
    std::vector<std::string> mpi_node_names_unique = mpi_node_names;
    std::sort(mpi_node_names_unique.begin(), mpi_node_names_unique.end());
    mpi_node_names_unique.erase(std::unique(mpi_node_names_unique.begin(), mpi_node_names_unique.end()), mpi_node_names_unique.end());
    return mpi_node_names_unique.size();
}


inline void split_mpi_comm(){
    /*
    This function splits MPI_COMM_WORLD into 3 layers of mpi communication groups
    - simulation groups
    - subdomain groups
    - group of only the sub domain's leader

    subdomain leader is selected from the first processes of each simulation group.
    Then the later processes will be assigned to the subdomain group.

    # memo
    h5 grid file is written only by the first group of simultaneous runs
    h5 data is written by the subdomain main processes of all the simultaneous runs.
    xdmf file is written by only the first subdomain's subdomain main process of all the simultaneous runs.
    */

    // make a list of mpi node names and redefine the world_rank
    std::vector<std::string> mpi_node_names(world_nprocs, "not_initialized"), mpi_node_names_pre(world_nprocs, "not_initialized");
    std::vector<int> mpi_node_ids(world_nprocs, -1);  // we put id for each node

    // get processor name of this process
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    mpi_node_names_pre[world_rank] = std::string(processor_name);

//
// DEBUG ONLY rename mpi node names into 4 different names
//
//    //if (world_nprocs != 32){
//    //    std::cout << "ERROR: world_nprocs is not 64. (for processor assignment test)" << std::endl;
//    //    exit(1);
//    //}
//    if (world_nprocs%2 != 0){
//        std::cout << "ERROR: world_nprocs is not even. (for processor assignment test)" << std::endl;
//        exit(1);
//    }
//
//    if (world_rank %2 == 0) {
//        mpi_node_names_pre[world_rank] = "node_inu";
//    //} else if (world_rank % 4 == 1) {
//    //    mpi_node_names_pre[world_rank] = "node_neko";
//    //} else if (world_rank % 4 == 2) {
//    //    mpi_node_names_pre[world_rank] = "node_kani";
//    } else {
//        mpi_node_names_pre[world_rank] = "node_usagi";
//    }
//
//    // debug out
//    if (world_rank == 0){
//        for (int irank = 0; irank < world_nprocs; irank++){
//                std::cout << "rank: " << irank << ", node name: " << mpi_node_names_pre[irank] << std::endl;
//        }
//    }
//
//    synchronize_all_world();
//
// DEBUG node fake name assignment done
//

    // gather all the node names in std::vector
    allgather_str(mpi_node_names_pre[world_rank], mpi_node_names_pre);

    // define node id for each node
    mpi_node_ids = define_node_ids(mpi_node_names_pre);

    // total number of nodes (unique)
    int n_compute_nodes = count_number_compute_nodes(mpi_node_names_pre);

    // reorder if the number of unique mpi_node_ids != 1
    if (n_compute_nodes > 1){

        // sort mpi_node_names and change world rank accordingly
        std::vector<size_t>      node_reorder = node_reordering(mpi_node_names_pre);
        std::vector<std::string> mpi_node_names_sorted(world_nprocs);
        std::vector<int>         mpi_node_ids_sorted(world_nprocs);
        for (int irank = 0; irank < world_nprocs; irank++){
            mpi_node_names_sorted[irank] = mpi_node_names_pre[node_reorder[irank]];
            mpi_node_ids_sorted[irank]   = mpi_node_ids[node_reorder[irank]];
        }
        mpi_node_names = mpi_node_names_sorted;
        mpi_node_ids   = mpi_node_ids_sorted;

        // renumbering this process's rank
        world_rank = node_reorder[world_rank];

    } else {
        mpi_node_names = mpi_node_names_pre;
        //mpi_node_ids   = mpi_node_ids;
    }

    // debug out
//    if (world_rank == 0){
//        for (int irank = 0; irank < world_nprocs; irank++){
//                std::cout << "apres rank: " << irank << ", node name: " << mpi_node_names[irank] << ", node id : " << mpi_node_ids[irank] << std::endl;
//        }
//    }
//    synchronize_all_world();

    // show node name and number summary
    if (world_rank == 0){
        std::cout << "\n\n Node name summary\n" << std::endl;

        std::cout << "Total number of compute nodes: " << n_compute_nodes << std::endl;

        for (long unsigned int i = 0; i < mpi_node_names.size(); i++){
            // only one show for  each node name
            if (i == 0 || mpi_node_names[i] != mpi_node_names[i-1]){
                std::cout << "node name: " << mpi_node_names[i] << ", number: " << std::count(mpi_node_names.begin(), mpi_node_names.end(), mpi_node_names[i]) << std::endl;
            }
        }
        std::cout << "\n" << std::endl;
    }

    // create communicator for this group if simultaneous run mode
    if(n_sims > 1) {
        // set a simultaneous run id
        if (world_nprocs%n_sims == 0) {
            n_procs_each_sim = static_cast<int>(world_nprocs/n_sims);
            id_sim           = std::floor(world_rank/n_procs_each_sim);
            //id_sim           = static_cast<int>(world_rank/n_procs_each_sim);
        } else {
            stdout_by_main("Error: requested nproc is not divisible by n_sims.");
            finalize_mpi();
            exit(1);
        }

        // create communicator for simulation group
        MPI_Comm_split(MPI_COMM_WORLD, id_sim, world_rank, &sim_comm);
        MPI_Comm_rank(sim_comm, &sim_rank);   // rank in simulation group
        MPI_Comm_size(sim_comm, &sim_nprocs); // number of processes in simulation group

        // recreate node_names and node_ids for this simulation group
        std::vector<std::string> mpi_node_names_tmp(sim_nprocs);
        std::vector<int> mpi_node_ids_tmp(sim_nprocs);

        for (int irank = 0; irank < sim_nprocs; irank++){
            mpi_node_names_tmp[irank] = mpi_node_names[n_procs_each_sim*id_sim + irank];
            mpi_node_ids_tmp[irank]   = mpi_node_ids[  n_procs_each_sim*id_sim + irank];
        }

        // re-assign node_names and node_ids
        mpi_node_names = mpi_node_names_tmp;
        mpi_node_ids   = mpi_node_ids_tmp;

        // count the number of compute nodes (unique) again
        n_compute_nodes = count_number_compute_nodes(mpi_node_names);

    } else {
        // this is not a simultaneous run
        n_procs_each_sim = world_nprocs;
        id_sim           = 0;

        sim_comm   = MPI_COMM_WORLD;
        sim_rank   = world_rank;
        sim_nprocs = world_nprocs;
    }

    // inter-communicator (between simulation groups)
    MPI_Comm_split(MPI_COMM_WORLD, sim_rank, id_sim, &inter_sim_comm);
    MPI_Comm_rank(inter_sim_comm, &inter_sim_rank);

    // number of subdomains
    n_subdomains = ndiv_i*ndiv_j*ndiv_k;

    // check if the number of compute node is multiple of n_subdomains
    if (n_subdomains % n_compute_nodes != 0){
        std::cout << "ERROR: n_compute_nodes is not multiple of n_subdomains" << std::endl;
        exit(1);
    }

    // number of subdomains per node
    //int n_subdomains_per_node = static_cast<int>(n_subdomains/n_compute_nodes);
    int n_procs_per_subdomain = static_cast<int>(sim_nprocs/n_subdomains);
    id_subdomain              = std::floor(sim_rank/n_procs_per_subdomain);
    int id_proc_in_subdomain  = sim_rank - id_subdomain*n_procs_per_subdomain;

    // assign first ranks to the main node of subdomain
    // and create a commmunicator for main processes of subdomain
    if (id_proc_in_subdomain == 0) {
        //std::cout << "my sim_rank: " << sim_rank << std::endl;
        subdom_main = true;
        MPI_Comm_split(sim_comm, 0, sim_rank, &inter_sub_comm);
        MPI_Comm_rank(inter_sub_comm, &inter_sub_rank);
        MPI_Comm_size(inter_sub_comm, &inter_sub_nprocs);

        // then the calculation in the sub domain will be done by only the processes
        // with subdomom_main == true and with inter_sub_comm commmunicator

        // use the rank number of inter_sub_comm as the myrank
        myrank = inter_sub_rank;
        nprocs = inter_sub_nprocs;

    } else {
        //std::cout << "my sim_rank to sub : " << sim_rank << std::endl;
        MPI_Comm_split(sim_comm, 1, sim_rank, &inter_sub_comm);

        // assign those ranks as subprocesses of each sub domain.
        subdom_main = false;
        myrank = -9999;
        nprocs = -9999;
    }

    synchronize_all_sim();

    MPI_Comm_split(sim_comm, id_subdomain, sim_rank, &sub_comm);
    MPI_Comm_rank(sub_comm, &sub_rank);
    MPI_Comm_size(sub_comm, &sub_nprocs);

    // convert the sub_comm to shared memory available communicator
    MPI_Comm tmp_comm;
    MPI_Comm_split_type(sub_comm, MPI_COMM_TYPE_SHARED, id_subdomain, MPI_INFO_NULL, &tmp_comm);
    sub_comm = tmp_comm;
    MPI_Comm_rank(sub_comm, &sub_rank);
    MPI_Comm_size(sub_comm, &sub_nprocs);

    synchronize_all_world();

    // check processors
    for (int irank = 0; irank < world_nprocs; irank++){
        synchronize_all_world();
        if (irank == world_rank)
            std::cout << "global rank: " << world_rank << ", node name: " << mpi_node_names[world_rank] \
                                                       << " | i_simul/total: " << id_sim       << "/" << n_sims       << ", n_procs_each_sim: " << n_procs_each_sim \
                                                       << " | i_subdom/total: " << id_subdomain << "/" << n_subdomains << ", subdom_main: "      << subdom_main \
                                                       << ", sub_rank/total: "  << sub_rank     << "/" << sub_nprocs \
            << std::endl;
    }

    synchronize_all_world();

    // // check processors
    // for (int irank = 0; irank < world_nprocs; irank++){
    //     synchronize_all_world();
    //     if (irank == world_rank) {
    //         std::cout   << "global rank: " << world_rank << ", node name: " << mpi_node_names[world_rank]
    //                     << ", world_rank: " << world_rank << ", world_nprocs: " << world_nprocs
    //                     << ", sim_rank: " << sim_rank << ", sim_nprocs: " << sim_nprocs
    //                     << ", sub_rank: " << sub_rank << ", sub_nprocs: " << sub_nprocs
    //                     << ", inter_sim_rank: " << inter_sim_rank
    //                     << ", inter_sub_rank: " << inter_sub_rank << ", inter_sub_nprocs: " << inter_sub_nprocs
    //                     << ", id_sim: " << id_sim << ", id_subdomain: " << id_subdomain
    //                     << ", subdom_main: " << subdom_main
    //                     << ", id_proc_in_subdomain: " << id_proc_in_subdomain
    //                     << ", id_subdomain: " << id_subdomain
    //                     << std::endl;
    //         std::cout << std::endl;
    //     }
    // }
    // synchronize_all_world();

}


inline int check_total_nprocs_and_ndiv(){
    if (world_nprocs != (n_sims*ndiv_i*ndiv_j*ndiv_k*n_subprocs)){
        stdout_by_main("ERROR: the number of requested processors and n_sims*ndiv_rtp[0]*ndiv_rtp[1]*ndiv_rtp[2]*nproc_sub in input_params.yml need to be the same.");
        if (world_rank==0){
            // print all params
            std::cout << "n_sims: " << n_sims << std::endl;
            std::cout << "ndiv_rtp: " << ndiv_k << " " << ndiv_j << " " << ndiv_i << std::endl;
            std::cout << "n_subprocs: " << n_subprocs << std::endl;
            std::cout << "nprocs should be n_sims*ndiv_p*ndiv_t*ndiv_r*n_subprocs = " << n_sims*ndiv_i*ndiv_j*ndiv_k*n_subprocs << std::endl;
            std::cout << "but the actual nprocs = " << nprocs << std::endl;
        }
        finalize_mpi();
        exit(EXIT_FAILURE);
    }
    return 0;
}


inline void synchronize_all(){
    MPI_Barrier(sim_comm);
}


inline void synchronize_all_sim(){
    MPI_Barrier(sim_comm);
}


inline void synchronize_all_sub(){
    MPI_Barrier(sub_comm);
}


inline void synchronize_all_inter(){
    MPI_Barrier(inter_sub_comm);
}


inline void synchronize_all_inter_sim(){
    MPI_Barrier(inter_sim_comm);
}


inline void synchronize_all_world(){
    MPI_Barrier(MPI_COMM_WORLD);
}


inline void send_bool_single_sim(bool* b, int dest){
    const int n = 1;
    MPI_Send(b, n, MPI_C_BOOL, dest, MPI_DUMMY_TAG, inter_sim_comm);
}


inline void recv_bool_single_sim(bool* b, int src){
    const int n = 1;
    MPI_Recv(b, n, MPI_C_BOOL, src, MPI_DUMMY_TAG, inter_sim_comm, MPI_STATUS_IGNORE);
}


inline void send_i_single_sim(int* i, int dest){
    const int n = 1;
    MPI_Send(i, n, MPI_INT, dest, MPI_DUMMY_TAG, inter_sim_comm);
}


inline void recv_i_single_sim(int* i, int src){
    const int n = 1;
    MPI_Recv(i, n, MPI_INT, src, MPI_DUMMY_TAG, inter_sim_comm, MPI_STATUS_IGNORE);
}


inline void send_cr_single_sim(CUSTOMREAL *cr, int dest){
    const int n = 1;
    MPI_Send(cr, n, MPI_CR, dest, MPI_DUMMY_TAG, inter_sim_comm);
}


inline void recv_cr_single_sim(CUSTOMREAL *cr, int src){
    const int n = 1;
    MPI_Recv(cr, n, MPI_CR, src, MPI_DUMMY_TAG, inter_sim_comm, MPI_STATUS_IGNORE);
}


inline void send_str_sim(std::string str, int dest){
    const int n = str.size();
    char* cstr = new char[n+1];
    strcpy(cstr, str.c_str());
    MPI_Send(cstr, n, MPI_CHAR, dest, MPI_DUMMY_TAG, inter_sim_comm);
    delete[] cstr;
}


inline void recv_str_sim(std::string& str, int src){
    MPI_Status status;
    int n;
    MPI_Probe(src, MPI_DUMMY_TAG, inter_sim_comm, &status);
    MPI_Get_count(&status, MPI_CHAR, &n);
    char* cstr = new char[n+1];
    MPI_Recv(cstr, n, MPI_CHAR, src, MPI_DUMMY_TAG, inter_sim_comm, MPI_STATUS_IGNORE);
    cstr[n] = '\0';
    str = std::string(cstr);
    delete[] cstr;
}


inline void send_str(std::string str, int dest){
    const int n = str.size();
    char* cstr = new char[n+1];
    strcpy(cstr, str.c_str());
    MPI_Send(cstr, n, MPI_CHAR, dest, MPI_DUMMY_TAG, inter_sub_comm);
    delete[] cstr;
}


inline void recv_str(std::string& str, int src){
    MPI_Status status;
    int n;
    MPI_Probe(src, MPI_DUMMY_TAG, inter_sub_comm, &status);
    MPI_Get_count(&status, MPI_CHAR, &n);
    char* cstr = new char[n+1];
    MPI_Recv(cstr, n, MPI_CHAR, src, MPI_DUMMY_TAG, inter_sub_comm, MPI_STATUS_IGNORE);
    cstr[n] = '\0';
    str = std::string(cstr);
    delete[] cstr;
}


inline void send_cr(CUSTOMREAL *cr, int n, int dest){
    MPI_Send(cr, n, MPI_CR, dest, MPI_DUMMY_TAG, inter_sub_comm);
}


inline void recv_cr(CUSTOMREAL *cr, int n, int src){
    MPI_Recv(cr, n, MPI_CR, src, MPI_DUMMY_TAG, inter_sub_comm, MPI_STATUS_IGNORE);
}

inline void isend_cr(CUSTOMREAL* buf, int count, int dest, MPI_Request& request){
    //MPI_Request request = MPI_REQUEST_NULL;
    //std::cout << "sending from : " << inter_sub_rank << ", to : " << dest <<", size : " << count << std::endl;
    MPI_Isend(buf, count, MPI_CR, dest, MPI_DUMMY_TAG, inter_sub_comm, &request);
}

inline void irecv_cr(CUSTOMREAL* buf, int count, int source, MPI_Request& request){
    //MPI_Request request = MPI_REQUEST_NULL;
    //std::cout << "receiving by : " << inter_sub_rank << ", from : " << source << ", size : " << count << std::endl;
    MPI_Irecv(buf, count, MPI_CR, source, MPI_DUMMY_TAG, inter_sub_comm, &request);
}

inline void allreduce_i_single(int& value, int& result){
    int count = 1;
    MPI_Allreduce(&value, &result, count, MPI_INT, MPI_SUM, inter_sub_comm);
}

inline void allreduce_i_sim_single_inplace(int& value){
    int count = 1;
    MPI_Allreduce(MPI_IN_PLACE, &value, count, MPI_INT, MPI_SUM, inter_sim_comm);
}

inline void allreduce_cr_single(CUSTOMREAL& loc_buf, CUSTOMREAL& all_buf){
    int count = 1;
    MPI_Allreduce(&loc_buf, &all_buf, count, MPI_CR, MPI_SUM, inter_sub_comm);
}


inline void allreduce_bool_inplace_inter_sim(bool* loc_buf, int count){
    // return true if any of the processes return true
    MPI_Allreduce(MPI_IN_PLACE, loc_buf, count, MPI_CXX_BOOL, MPI_LOR, inter_sim_comm);
}


inline void allreduce_bool_inplace(bool* loc_buf, int count){
    // return true if any of the processes return true
    MPI_Allreduce(MPI_IN_PLACE, loc_buf, count, MPI_CXX_BOOL, MPI_LOR, inter_sub_comm);
}


inline void allreduce_bool_inplace_sub(bool* loc_buf, int count){
    // return true if any of the processes return true
    MPI_Allreduce(MPI_IN_PLACE, loc_buf, count, MPI_CXX_BOOL, MPI_LOR, sub_comm);
}


inline void allreduce_bool_single_inplace(bool& loc_buf){
    MPI_Allreduce(MPI_IN_PLACE, &loc_buf, 1, MPI_CXX_BOOL, MPI_LAND, inter_sub_comm);
}

// true if all processes return true else false
inline void allreduce_bool_single_inplace_sim(bool& loc_buf){
    MPI_Allreduce(MPI_IN_PLACE, &loc_buf, 1, MPI_CXX_BOOL, MPI_LAND, inter_sim_comm);
}

inline void allreduce_i_inplace(int* loc_buf, int count){
    MPI_Allreduce(MPI_IN_PLACE, loc_buf, count, MPI_INT, MPI_SUM, inter_sub_comm);
}

inline void allreduce_cr_inplace(CUSTOMREAL* loc_buf, int count){
    MPI_Allreduce(MPI_IN_PLACE, loc_buf, count, MPI_CR, MPI_SUM, inter_sub_comm);
}

inline void allreduce_cr_sim(CUSTOMREAL* loc_buf, int count, CUSTOMREAL* all_buf){
    MPI_Allreduce(loc_buf, all_buf, count, MPI_CR, MPI_SUM, inter_sim_comm);
}

inline void allreduce_cr_sim_single(CUSTOMREAL& loc_buf, CUSTOMREAL& all_buf){
    int count = 1;
    MPI_Allreduce(&loc_buf, &all_buf, count, MPI_CR, MPI_SUM, inter_sim_comm);
}

inline void allreduce_cr_sim_single_inplace(CUSTOMREAL& loc_buf){
    int count = 1;
    MPI_Allreduce(MPI_IN_PLACE, &loc_buf, count, MPI_CR, MPI_SUM, inter_sim_comm);
}

inline void allreduce_cr_sim_inplace(CUSTOMREAL* loc_buf, int count){
    MPI_Allreduce(MPI_IN_PLACE, loc_buf, count, MPI_CR, MPI_SUM, inter_sim_comm);
}

inline void alleduce_cr_sim_single_inplace(CUSTOMREAL& loc_buf){
    int count = 1;
    MPI_Allreduce(MPI_IN_PLACE, &loc_buf, count, MPI_CR, MPI_SUM, inter_sim_comm);
}

inline void allreduce_cr_single_max(CUSTOMREAL& loc_buf, CUSTOMREAL& all_buf){
    int count = 1;
    MPI_Allreduce(&loc_buf, &all_buf, count, MPI_CR, MPI_MAX, inter_sub_comm);
}

inline void allgather_i_single(int* loc_buf, int* all_buf){
    int count = 1;
    MPI_Allgather(loc_buf, count, MPI_INT, all_buf, count, MPI_INT, inter_sub_comm);
}

inline void allgather_cr_single(CUSTOMREAL* loc_buf, CUSTOMREAL* all_buf){
    int count = 1;
    MPI_Allgather(loc_buf, count, MPI_CR, all_buf, count, MPI_CR, inter_sub_comm);
}

inline void allgather_bool_single(bool* loc_buf, bool* all_buf){
    int count = 1;
    MPI_Allgather(loc_buf, count, MPI_CXX_BOOL, all_buf, count, MPI_CXX_BOOL, inter_sub_comm);
}

inline void broadcast_bool_single(bool& value, int root){
    int count = 1;
    MPI_Bcast(&value, count, MPI_CXX_BOOL, root, inter_sub_comm);
}

inline void broadcast_bool_single_sub(bool& value, int root){
    int count = 1;
    MPI_Bcast(&value, count, MPI_CXX_BOOL, root, sub_comm);
}

inline void broadcast_bool_single_inter_sim(bool& value, int root){
    int count = 1;
    MPI_Bcast(&value, count, MPI_CXX_BOOL, root, inter_sim_comm);
}

inline void broadcast_bool_inter_and_intra_sim(bool& value, int root){
    broadcast_bool_single_inter_sim(value, root); // broadcast among simultaneous run group
    broadcast_bool_single(value, root);           // broadcast among subdomain group
    broadcast_bool_single_sub(value, root);       // broadcast within subdomain group
}

inline void broadcast_i_single(int& value, int root){
    int count = 1;
    MPI_Bcast(&value, count, MPI_INT, root, inter_sub_comm);
}


inline void broadcast_i_single_inter_sim(int& value, int root){
    int count = 1;
    MPI_Bcast(&value, count, MPI_INT, root, inter_sim_comm);
}

inline void broadcast_i_single_intra_sim(int& value, int root){
    broadcast_i_single(value, root);           // broadcast among subdomain group
    broadcast_i_single_sub(value, root);       // broadcast within subdomain group
}

inline void broadcast_i_single_inter_and_intra_sim(int& value, int root){
    broadcast_i_single_inter_sim(value, root); // broadcast among simultaneous run group
    broadcast_i_single(value, root);           // broadcast among subdomain group
    broadcast_i_single_sub(value, root);       // broadcast within subdomain group
}

inline void broadcast_f_single(float& value, int root){ // !!!! FOR ONLY READ PARAMETER !!!!!
    int count = 1;
    MPI_Bcast(&value, count, MPI_FLOAT, root, inter_sub_comm);
}

inline void broadcast_cr_single(CUSTOMREAL& buf, int root){
    MPI_Bcast(&buf, 1, MPI_CR, root, inter_sub_comm);
}

inline void broadcast_cr(CUSTOMREAL* buf, int count, int root){
    MPI_Bcast(buf, count, MPI_CR, root, inter_sub_comm);
}

inline void broadcast_cr_single_inplace(CUSTOMREAL& buf, int root){
    MPI_Bcast(&buf, 1, MPI_CR, root, inter_sub_comm);
}

inline void broadcast_cr_inter_sim(CUSTOMREAL* buf, int count, int root){
    MPI_Bcast(buf, count, MPI_CR, root, inter_sim_comm);
}


inline void broadcast_cr_single_inter_sim(CUSTOMREAL& buf, int root){
    int count = 1;
    MPI_Bcast(&buf, count, MPI_CR, root, inter_sim_comm);
}


inline void broadcast_cr_single_inter_and_intra_sim(CUSTOMREAL& buf, int root){
    broadcast_cr_single_inter_sim(buf, root); // broadcast among simultaneous run group
    broadcast_cr_single(buf, root);           // broadcast among subdomain group
    broadcast_cr_single_sub(buf, root);       // broadcast within subdomain group
}


inline void broadcast_i_single_sub(int& value, int root){
    int count = 1;
    MPI_Bcast(&value, count, MPI_INT, root, sub_comm);
}

//inline void broadcast_f_single_sub(float& value, int root){ // !!!! FOR ONLY READ PARAMETER !!!!!
//    int count = 1;
//    MPI_Bcast(&value, count, MPI_FLOAT, root, sub_comm);
//}

inline void broadcast_cr_single_sub(CUSTOMREAL& buf, int root){
    int count = 1;
    MPI_Bcast(&buf, count, MPI_CR, root, sub_comm);
}

inline void broadcast_cr_sub(CUSTOMREAL* buf, int count, int root){
    MPI_Bcast(buf, count, MPI_CR, root, sub_comm);
}

inline void broadcast_str(std::string& str, int root) {
    int count = str.size();
    MPI_Bcast(&count, 1, MPI_INT, root, inter_sub_comm);
    char* buf = new char[count+1];
    if (root == inter_sub_rank) {
        std::strcpy(buf, str.c_str());
    }
    MPI_Bcast(buf, count+1, MPI_CHAR, root, inter_sub_comm);
    str = buf;
    delete[] buf;
}

inline void broadcast_str_sub(std::string& str, int root) {
    int count = str.size();
    MPI_Bcast(&count, 1, MPI_INT, root, sub_comm);
    char* buf = new char[count+1];
    if (root == sub_rank) {
        std::strcpy(buf, str.c_str());
    }
    MPI_Bcast(buf, count+1, MPI_CHAR, root, sub_comm);
    str = buf;
    delete[] buf;
}

inline void broadcast_str_inter_sim(std::string& str, int root) {
    int count = str.size();
    MPI_Bcast(&count, 1, MPI_INT, root, inter_sim_comm);
    char* buf = new char[count+1];
    if (root == inter_sim_rank) {
        std::strcpy(buf, str.c_str());
    }
    MPI_Bcast(buf, count+1, MPI_CHAR, root, inter_sim_comm);
    str = buf;
    delete[] buf;
}

inline void broadcast_str_inter_and_intra_sim(std::string& str, int root) {
    broadcast_str_inter_sim(str, root);
    broadcast_str(str, root);
    broadcast_str_sub(str, root);
}


inline void allgather_str(const std::string &str, std::vector<std::string> &result) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int size = world_nprocs;

    int str_size = str.size();
    std::vector<int> str_sizes(size);
    MPI_Allgather(&str_size, 1, MPI_INT, str_sizes.data(), 1, MPI_INT, comm);

    int total_size = 0;
    std::vector<int> displs(size);
    for (int i = 0; i < size; i++) {
      total_size += str_sizes[i];
      displs[i] = (i == 0) ? 0 : (displs[i - 1] + str_sizes[i - 1]);
    }

    std::vector<char> data(total_size);
    MPI_Allgatherv(str.data(), str_size, MPI_CHAR, data.data(), str_sizes.data(), displs.data(), MPI_CHAR, comm);

    result.reserve(size);
    int start = 0;
    for (int i = 0; i < size; i++) {
      result[i] = std::string(&data[start], str_sizes[i]);
      start += str_sizes[i];
    }

}


inline void prepare_shm_array_cr(int n_elms, CUSTOMREAL* &buf, MPI_Win& win){
    int* model;
    int flag;
    MPI_Aint winsize_dummy;
    int windisp_dummy;

    // Allocate shared memory for the array on only the subdomain's main process (n_elms == 0 for the other processes)
    MPI_Win_allocate_shared(n_elms*sizeof(CUSTOMREAL), sizeof(CUSTOMREAL), MPI_INFO_NULL, sub_comm, &buf, &win);
    // get attribute
    MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
    // shared query
    if (sub_rank != 0){
        MPI_Win_shared_query(win, 0, &winsize_dummy, &windisp_dummy, &buf);
    }

    shm_fence(win);
}

/*
inline void prepare_shm_single_i(int*& value, MPI_Win& win){
    int* model;
    int flag;
    MPI_Aint winsize_dummy;
    int windisp_dummy;
    int mpi_size;
    if (sub_rank == 0){
        mpi_size = sizeof(int)*1;
    } else {
        mpi_size = 0;
    }

    // Allocate shared memory for the array on only the subdomain's main process (n_elms == 0 for the other processes)
    MPI_Win_allocate_shared(mpi_size, sizeof(int), MPI_INFO_NULL, sub_comm, &value, &win);
    // get attribute
    MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
    // shared query
    if (sub_rank != 0){
        MPI_Win_shared_query(win, 0, &winsize_dummy, &windisp_dummy, &value);
    }

    shm_fence(win);
}
*/
/*
inline void prepare_shm_single_cr(CUSTOMREAL*& value, MPI_Win& win){
    int* model;
    int flag;
    MPI_Aint winsize_dummy;
    int windisp_dummy;
    int mpi_size;
    if (sub_rank == 0){
        mpi_size = sizeof(CUSTOMREAL)*1;
    } else {
        mpi_size = 0;
    }

    // Allocate shared memory for the array on only the subdomain's main process (n_elms == 0 for the other processes)
    MPI_Win_allocate_shared(mpi_size, sizeof(CUSTOMREAL), MPI_INFO_NULL, sub_comm, &value, &win);
    // get attribute
    MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
    // shared query
    if (sub_rank != 0){
        MPI_Win_shared_query(win, 0, &winsize_dummy, &windisp_dummy, &value);
    }

    shm_fence(win);
}
*/

inline void prepare_shm_array_bool(int n_elms, bool* &buf, MPI_Win& win){
    int* model;
    int flag;
    MPI_Aint winsize_dummy;
    int windisp_dummy;

    // Allocate shared memory for the array on only the subdomain's main process (n_elms == 0 for the other processes)
    MPI_Win_allocate_shared(n_elms*sizeof(bool), sizeof(bool), MPI_INFO_NULL, sub_comm, &buf, &win);
    // get attribute
    MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
    // shared query
    if (sub_rank != 0){
        MPI_Win_shared_query(win, 0, &winsize_dummy, &windisp_dummy, &buf);
    }

    shm_fence(win);
}


inline void wait_req(MPI_Request& req){
    MPI_Status status;
    MPI_Wait(&req, &status);
}


inline void shm_fence(MPI_Win& win){
    MPI_Win_fence(0, win);
}

// Function to safely free a single MPI window
inline void cleanup_mpi_win(MPI_Win& win) {
    if (win != MPI_WIN_NULL) {
        int mpi_error = MPI_Win_free(&win);
        if (mpi_error != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length_of_error_string;
            MPI_Error_string(mpi_error, error_string, &length_of_error_string);
            std::cerr << "ERROR: MPI_Win_free failed with error code " << mpi_error
                      << ": " << error_string << std::endl;
            std::cerr << "This may indicate issues with Intel OneAPI MPI shared memory cleanup." << std::endl;
            // Don't call MPI_Abort here as we're likely in cleanup phase
        } else {
            win = MPI_WIN_NULL;
        }
    }
}

// Function to safely free multiple MPI windows
inline void cleanup_mpi_wins(std::initializer_list<MPI_Win*> wins) {
    int error_count = 0;
    int total_windows = 0;

    for (MPI_Win* win : wins) {
        if (win != nullptr) {
            total_windows++;
            if (*win != MPI_WIN_NULL) {
                int mpi_error = MPI_Win_free(win);
                if (mpi_error != MPI_SUCCESS) {
                    error_count++;
                    char error_string[MPI_MAX_ERROR_STRING];
                    int length_of_error_string;
                    MPI_Error_string(mpi_error, error_string, &length_of_error_string);
                    std::cerr << "ERROR: MPI_Win_free failed for window " << total_windows
                              << " with error code " << mpi_error
                              << ": " << error_string << std::endl;
                } else {
                    *win = MPI_WIN_NULL;
                }
            }
        }
    }

    if (error_count > 0) {
        std::cerr << "WARNING: " << error_count << " out of " << total_windows
                  << " MPI windows failed to free properly." << std::endl;
        std::cerr << "This may indicate Intel OneAPI MPI shared memory cleanup issues." << std::endl;
    } else if (total_windows > 0) {
        // Only show success message in verbose mode or for debugging
        if (if_verbose && world_rank == 0) {
            std::cout << "Successfully freed " << total_windows << " MPI shared memory windows." << std::endl;
        }
    }
}

// Function to initialize multiple MPI windows to NULL
inline void init_mpi_wins(std::initializer_list<MPI_Win*> wins) {
    int initialized_count = 0;

    for (MPI_Win* win : wins) {
        if (win != nullptr) {
            *win = MPI_WIN_NULL;
            initialized_count++;
        } else {
            std::cerr << "WARNING: Null pointer passed to init_mpi_wins, skipping." << std::endl;
        }
    }

    if (if_verbose && world_rank == 0 && initialized_count > 0) {
        std::cout << "Initialized " << initialized_count << " MPI windows to MPI_WIN_NULL." << std::endl;
    }
}




#endif // MPI_FUNCS_H