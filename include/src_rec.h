#ifndef SRC_REC_H
#define SRC_REC_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

#include "config.h"
#include "utils.h"
#include "timer.h"
#include "mpi_funcs.h"


// information of source or receiver
class SrcRecInfo {
public:
    int id = -1;                        // id in the file
    std::string name = "unknown";       // name in the file
    CUSTOMREAL dep; // stored as depth (km), so need to depth2radious function when using it in other part of the code
    CUSTOMREAL lat; // stored in degree, but convarted to radian when passed through get_src_* function
    CUSTOMREAL lon; // stored in degree, but convarted to radian when passed through get_src_* function

    int year             = -1;
    int month            = -1;
    int day              = -1;
    int hour             = -1;
    int min              = -1;
    CUSTOMREAL sec       = -1.0;
    CUSTOMREAL mag       = -1.0;

    CUSTOMREAL adjoint_source = 0.0;
    CUSTOMREAL adjoint_source_density = 0.0;

    int n_data = 0;

    // unique id for TomoATT, used to identify the source or receiver in TomoATT. 
    int id_att = -1;    // the index of sources in TomoATT, used to identify this source. Each source has a unique id_src_att.

    // arrays for storing arrival times on boundary surfaces, calculated by 2D Eikonal solver
    bool        is_out_of_region    = false;   // is the source or receiver in the region; false: in the refion; true: teleseismic
    //CUSTOMREAL* arr_times_bound_N   = nullptr; // arrival time of the receiver at the north boundary of the subdomain
    //CUSTOMREAL* arr_times_bound_E   = nullptr; // arrival time of the receiver at the east boundary of the subdomain
    //CUSTOMREAL* arr_times_bound_W   = nullptr; // arrival time of the receiver at the west boundary of the subdomain
    //CUSTOMREAL* arr_times_bound_S   = nullptr; // arrival time of the receiver at the south boundary of the subdomain
    //CUSTOMREAL* arr_times_bound_Bot = nullptr; // arrival time of the receiver at the bottom boundary of the subdomain
    //bool*       is_bound_src;                  // true if the source is on the boundary surface

    // kernel
    CUSTOMREAL sta_correct = 0.0;
    CUSTOMREAL sta_correct_kernel = 0.0;

    // relocation
    bool       is_stop      = false;
    CUSTOMREAL tau_opt      = 0.0;
    CUSTOMREAL grad_chi_i   = 0.0;
    CUSTOMREAL grad_chi_j   = 0.0;
    CUSTOMREAL grad_chi_k   = 0.0;
    CUSTOMREAL grad_tau     = 0.0;
    CUSTOMREAL sum_weight   = 0.0;
    CUSTOMREAL vobj_src_reloc_old       = 99999999.9;
    CUSTOMREAL vobj_src_reloc           = 0.0;
    CUSTOMREAL vobj_grad_norm_src_reloc = 0.0;

    CUSTOMREAL step_length_max  = step_length_src_reloc;  // 2 km default, step length for relocation
    CUSTOMREAL change_dep = 0.0;
    CUSTOMREAL change_lat = 0.0;
    CUSTOMREAL change_lon = 0.0;
    CUSTOMREAL change_tau = 0.0;
    int        Ndata      = 0.0;

    bool       is_T_written_into_file = false; 
};

class DataInfo {
public:

    // weight
    CUSTOMREAL data_weight  = 1.0;   // the weight in the src_rec file
    CUSTOMREAL weight       = 1.0;   // the actual weight in the inversion, equal   data_weight * weight about the data type;
    CUSTOMREAL weight_reloc = 1.0;   // the actual weight for relocation,   equal   data_weight * weight about the data type;

    // phase
    std::string phase = "unknown";
    // int phase_id = -1;
    
    // if true, this data is a dual data, used for generating kernel, but not for obj estimation (if true, data type = 2 or 3)
    bool dual_data   = false;   

    // three types of data, infomation is conbined together to reduce RAM:

    // data_type: replace 
    // 0: absolute traveltime, single source - receiver
    // 1: common source differential traveltime
    // 2: common receiver differential traveltime
    // replaces
    // bool is_src_rec  (for case 1)
    // bool is_src_pair (for case 2)
    // bool is_rec_pair (for case 3)
    int data_type = -1; 

    int id_src_att = -1;    // the index of sources in TomoATT, used to identify this source. Each source has a unique id_src_att.
    int id_rec_att = -1;    // the index of receivers in TomoATT, used to identify this receiver. Each receiver has a unique id_rec_att.
    int id_pair_att = -1; // the index of the second source or receiver in TomoATT. valid only for data_type = 1 or 2. For data_type = 0, id_pari_att = -1.

    // source information
    // for data_type = 0, id_srcs[0] is the source id; 
    // for data_type = 1, id_srcs[0] is the source id;
    // for data_type = 2, id_srcs[0] and id_srcs[1] are the source ids
    // std::vector<int>            id_src_pair = {-1, -1};
    // std::vector<std::string>    name_src_pair = {"unknown", "unknown"};

    // receiver information
    // for data_type = 0, id_recs[0] is the receiver id;
    // for data_type = 1, id_recs[0] and id_recs[1] are the receiver ids;
    // for data_type = 2, id_recs[0] is the receiver id;
    // std::vector<int>            id_rec_pair = {-1, -1};
    // std::vector<std::string>    name_rec_pair = {"unknown", "unknown"};

    // traveltime
    CUSTOMREAL travel_time     = -999.0;        // synthetic travel time from id_srcs[0] to id_recs[0]
    CUSTOMREAL dif_travel_time = -999.0;        // synthetic differential travel time for specific data tpye. (type_type = 1 or 2)
    // observed time
    CUSTOMREAL time_observation = -999.0;        // observed travel time for data_type = 0; observed differential travel time for data_type = 1 or 2 


    // bool is_src_rec = false; // absolute traveltime, single source - receiver

    // // source information
    // int         id_src   = -1;
    // std::string name_src = "unknown";

    // // receiver information
    // int         id_rec   = -1;
    // std::string name_rec = "unknown";

    // // traveltime
    // CUSTOMREAL travel_time     = -999.0;
    // CUSTOMREAL travel_time_obs = -999.0;

    // // receiver pair infomation
    // bool is_rec_pair                       = false;   // common source differential traveltime
    // std::vector<int>         id_rec_pair   = {-1,-1};
    // std::vector<std::string> name_rec_pair = {"unknown","unknown"};

    // // common source differential travel time
    // CUSTOMREAL cs_dif_travel_time     = -999.0;
    // CUSTOMREAL cs_dif_travel_time_obs = -999.0;

    // // source pair infomation
    // bool                     is_src_pair   = false;   // common receiver differential traveltime (for future)
    // std::vector<int>         id_src_pair   = {-1,-1};
    // std::vector<std::string> name_src_pair = {"unknown","unknown"};

    // // common receiver differential travel time
    // CUSTOMREAL cr_dif_travel_time     = -999.0;
    // CUSTOMREAL cr_dif_travel_time_obs = -999.0;

    // source relocation
    // CUSTOMREAL DTi          = 0.0;
    // CUSTOMREAL DTj          = 0.0;
    // CUSTOMREAL DTk          = 0.0;

    // DTi is stored in DTi_pair[0]
    std::vector<CUSTOMREAL> DTi_pair = {0.0, 0.0};
    std::vector<CUSTOMREAL> DTj_pair = {0.0, 0.0};
    std::vector<CUSTOMREAL> DTk_pair = {0.0, 0.0};

};



// methods for managing SrcRec objects/lists

// parse src_rec_file
void parse_src_rec_file(std::string&                      , \
                        std::map<std::string, SrcRecInfo>&, \
                        std::map<std::string, SrcRecInfo>&, \
                        std::map<std::string, std::map<std::string,std::vector<DataInfo>>>&, \
                        std::vector<std::string>&, \
                        std::vector<std::vector<std::vector<std::string>>>&);

// parse sta_correctoion_file
void parse_sta_correction_file(std::string&, \
                               std::map<std::string, SrcRecInfo>&);

// swap the sources and receivers
void do_swap_src_rec(std::map<std::string, SrcRecInfo> &, \
                     std::map<std::string, SrcRecInfo> &, \
                     std::map<std::string, std::map<std::string, std::vector<DataInfo>>> &, \
                     std::vector<std::string>&);

// do not swap the sources and receivers
void do_not_swap_src_rec(std::map<std::string, SrcRecInfo> &, \
                     std::map<std::string, SrcRecInfo> &, \
                     std::map<std::string, std::map<std::string, std::vector<DataInfo>>> &, \
                     std::vector<std::string>&);

// tele seismic source management
void separate_region_and_tele_src_rec_data(std::map<std::string, SrcRecInfo> &,
                                           std::map<std::string, SrcRecInfo> &,
                                           std::map<std::string, std::map<std::string, std::vector<DataInfo>>>&,
                                           std::map<std::string, SrcRecInfo> &,
                                           std::map<std::string, SrcRecInfo> &,
                                           std::map<std::string, std::map<std::string, std::vector<DataInfo>>>&,
                                           std::map<std::string, SrcRecInfo> &,
                                           std::map<std::string, SrcRecInfo> &,
                                           std::map<std::string, std::map<std::string, std::vector<DataInfo>>>&,
                                           std::map<std::string, int>        &,
                                           int                               &,
                                           int                               &,
                                           int                               &,
                                           int                               &,
                                           int                               &,
                                           const CUSTOMREAL, const CUSTOMREAL,
                                           const CUSTOMREAL, const CUSTOMREAL,
                                           const CUSTOMREAL, const CUSTOMREAL,
                                           bool);

void merge_region_and_tele_src(std::map<std::string, SrcRecInfo> &,
                               std::map<std::string, SrcRecInfo> &,
                               std::map<std::string, std::map<std::string,std::vector<DataInfo>>>&,
                               std::vector<std::string>&,
                               std::map<std::string, SrcRecInfo> &,
                               std::map<std::string, SrcRecInfo> &,
                               std::map<std::string, std::map<std::string,std::vector<DataInfo>>>&);

// distribute srcrec data to all simulation groups
void distribute_src_rec_data(std::map<std::string, SrcRecInfo>&, \
                             std::map<std::string, SrcRecInfo>&, \
                             std::map<std::string, std::map<std::string,std::vector<DataInfo>>>&, \
                             std::vector<std::string>&, \
                             std::map<std::string, SrcRecInfo>&, \
                             std::map<std::string, SrcRecInfo>&, \
                             std::map<std::string, std::map<std::string,std::vector<DataInfo>>>&, \
                             std::vector<std::string>&, \
                             std::vector<std::string>&);

void prepare_src_map_for_2d_solver(std::map<std::string, SrcRecInfo>&, \
                                   std::map<std::string, SrcRecInfo>&, \
                                   std::vector<std::string>&, \
                                   std::map<std::string, SrcRecInfo>&);

void send_src_info_inter_sim(SrcRecInfo&, int);
void recv_src_info_inter_sim(SrcRecInfo&, int);
void broadcast_src_info_intra_sim(SrcRecInfo&, int);
void send_rec_info_inter_sim(SrcRecInfo&, int);
void recv_rec_info_inter_sim(SrcRecInfo&, int);
void broadcast_rec_info_intra_sim(SrcRecInfo&, int);
void send_data_info_inter_sim(DataInfo&, int);
void recv_data_info_inter_sim(DataInfo&, int);

#endif //SRC_REC_H