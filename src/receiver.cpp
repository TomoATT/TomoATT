#include "receiver.h"


Receiver::Receiver() {
}


Receiver::~Receiver() {
}

void Receiver::interpolate_and_store_arrival_times_at_rec_position(InputParams& IP, Grid& grid, const int id_src_att) {
    if(subdom_main){    // do it across level 2, (main of level 3).

        // int mykey_send=9999;
        // int mykey_end=9998;
        // int key = 0;

        // share the traveltime values on the corner points of the subdomains for interpolation
        // this is not necessary for sweeping (as the stencil is closs shape)
        grid.send_recev_boundary_data(grid.T_loc);
        grid.send_recev_boundary_data_kosumi(grid.T_loc);

        if (proc_store_srcrec){     // main of level 2
            
            // 1. estimate the number of traveltime to be calculated for this source
            int n_time = 0;
            int data_begin = IP.src_map[id_src_att].data_begin;
            int data_end   = IP.src_map[id_src_att].data_end;
            for (int i_data = data_begin; i_data < data_end; i_data++){
                auto& data = IP.data_vec[i_data];    
                if (data.data_type == DATA_TYPE_ABS){
                    n_time += 1;
                } else if (data.data_type == DATA_TYPE_CSDIF) {
                    n_time += 2;
                } else if (data.data_type == DATA_TYPE_CRDIF) {
                    n_time += 1;
                } else {
                    std::cout << "error type of data" << std::endl;
                    exit(1);
                }
            }
            broadcast_i_single(n_time, 0);   // (level 2) broadcast the number of traveltime to be calculated for this source

            // 2. loop all data, to calculate traveltime
            for (int i_data = data_begin; i_data < data_end; i_data++){
                auto& data = IP.data_vec[i_data];

                if (data.data_type == DATA_TYPE_ABS){   // absolute traveltime
                    // send id_rec_att
                    int id_rec_att = data.id_rec_att;
                    broadcast_i_single(id_rec_att, 0);
                    // calculate travel time
                    data.travel_time = interpolate_travel_time(grid, IP, id_rec_att);

                } else if (data.data_type == DATA_TYPE_CSDIF) {     // common source differential traveltime
                    // send id_rec_att for the first receiver
                    int id_rec1_att = data.id_rec_att;
                    broadcast_i_single(id_rec1_att, 0);
                    // calculate travel time for the first receiver
                    CUSTOMREAL travel_time_1 = interpolate_travel_time(grid, IP, id_rec1_att);

                    // send id_rec_att for the second receiver
                    int id_rec2_att = data.id_pair_att;
                    broadcast_i_single(id_rec2_att, 0);
                    // calculate travel time for the second receiver
                    CUSTOMREAL travel_time_2 = interpolate_travel_time(grid, IP, id_rec2_att);

                    // store travel time and differential traveltime
                    data.travel_time   = travel_time_1;
                    data.dif_travel_time = travel_time_1 - travel_time_2;

                } else if (data.data_type == DATA_TYPE_CRDIF) {     // common receiver differential traveltime
                    // send id_rec_att
                    int id_rec_att = data.id_rec_att;
                    broadcast_i_single(id_rec_att, 0);
                    // calculate travel time
                    data.travel_time = interpolate_travel_time(grid, IP, id_rec_att);
                } else {
                    std::cout << "error type of data" << std::endl;
                    exit(1);
                }
            }

        } else {    // other ranks which do not have src_map, rec_map, data_vec

            // 1. receive the number of traveltime to be calculated for this source
            int n_time = 0;
            broadcast_i_single(n_time, 0);   // (level 2) receive the number of traveltime to be calculated for this source

            // 2. loop all data, to calculate traveltime
            for (int i_time = 0; i_time < n_time; i_time++){
                // receive id_rec_att
                int id_rec_att = 0;
                broadcast_i_single(id_rec_att, 0);
                // calculate travel time
                CUSTOMREAL dummy_time = interpolate_travel_time(grid, IP, id_rec_att);
                (void) dummy_time; // avoid compiler warning
            }
        }
    } // end subdomain

    synchronize_all();

        // <OLD VERSION> JC: 20260618

        // // using this way, because the number of receivers is not known for other processors.
        // // the main processor will send a request to other processors if it (main processor) requires a traveltime.
        // // the request is "broadcast_i_single(mykey_send, 0)".
        // // Once it ends, the main processor will send a ending signal to other processors, "broadcast_i_single(mykey_end, 0)". 
        // if (proc_store_srcrec){     // main of level 2
        //     // routine for processes which have the source and receiver data

        //     // calculate the travel time of the receiver by interpolation
        //     for (auto it_rec = IP.data_map[name_sim_src].begin(); it_rec != IP.data_map[name_sim_src].end(); ++it_rec) {
        //         for (auto& data: it_rec->second){

        //             if (data.data_type == DATA_TYPE_ABS){   // absolute traveltime
        //                 // send receivr name as a starting signal for interpolation

        //                 // send dummy integer with key
        //                 broadcast_i_single(mykey_send, 0);
        //                 broadcast_str(data.name_rec_pair[0], 0);

        //                 // store travel time on single receiver and double receivers (what is double receivers? by CHEN Jing)
        //                 // store travel time from name_sim_src(src_name) to it_rec->first(rec_name)
        //                 data.travel_time = interpolate_travel_time(grid, IP, name_sim_src, it_rec->first);
        //             } else if (data.data_type == DATA_TYPE_CSDIF) {     // common source differential traveltime
        //                 // store travel time from name_sim_src(src_name) to rec1_name and rec2_name
        //                 // calculate travel times for two receivers
        //                 broadcast_i_single(mykey_send, 0);
        //                 broadcast_str(data.name_rec_pair[0], 0);
        //                 CUSTOMREAL travel_time   = interpolate_travel_time(grid, IP, name_sim_src, data.name_rec_pair[0]);

        //                 broadcast_i_single(mykey_send, 0);
        //                 broadcast_str(data.name_rec_pair[1], 0);
        //                 CUSTOMREAL travel_time_2 = interpolate_travel_time(grid, IP, name_sim_src, data.name_rec_pair[1]);

        //                 // Because name_sim_src = data.name_src; it_rec->first = name_rec = name_rec_pair[0]
        //                 // Thus data.travel_time is travel_time
        //                 data.travel_time = travel_time;

        //                 // calculate and store travel time difference
        //                 data.dif_travel_time = travel_time - travel_time_2;
        //             } else if (data.data_type == DATA_TYPE_CRDIF) {     // common receiver differential traveltime
        //                 // store travel time from name_sim_src(src1_name) to it_rec->first(rec_name)
        //                 broadcast_i_single(mykey_send, 0);
        //                 broadcast_str(data.name_rec_pair[0], 0);
        //                 data.travel_time = interpolate_travel_time(grid, IP, name_sim_src, it_rec->first);

        //             } else {
        //                 std::cout << "error type of data" << std::endl;
        //             }
        //         }
        //     }

        //     // send a endind signal to the processes which do not have the source receiver data
        //     broadcast_i_single(mykey_end, 0);

        // } else {    // other ranks in level 2
        //     // routine for processes which do not have the source and receiver data

        //     // waiting the communication from the proc_store_srcrec
        //     while (true) {

        //         // receive dummy integer
        //         broadcast_i_single(key, 0);

        //         // check the tag
        //         if (key == mykey_send) {
        //             std::string name_rec;
        //             broadcast_str(name_rec, 0);

        //             CUSTOMREAL dummy_time = interpolate_travel_time(grid, IP, name_sim_src, name_rec);
        //             (void) dummy_time; // avoid compiler warning

        //         } else if (key == mykey_end) {
        //             // receive dummy integer
        //             break;
        //         } else {
        //             std::cout << "error in the tag" << std::endl;
        //         }

        //     }
        // }
    // } // end subdomain

    // synchronize_all();
}


void Receiver::calculate_adjoint_source(InputParams& IP, const int id_src_att) {

    // #TODO: run this function only by proc_store_srcrec
    if (proc_store_srcrec) {  // maid of level 2 and 3

        // rec.adjoint_source = 0 && rec.adjoint_source_density = 0
        IP.initialize_adjoint_source();

        int data_begin = IP.src_map[id_src_att].data_begin;
        int data_end   = IP.src_map[id_src_att].data_end;

        // loop all data related to this source MNMN: use reference(auto&) to avoid copy
        for (int i_data = data_begin; i_data < data_end; i_data++){
            auto& data = IP.data_vec[i_data];

            bool is_tele = IP.src_map[id_src_att].is_out_of_region;
            //
            // absolute traveltime
            //
            if (data.data_type == DATA_TYPE_ABS) {
                if (!IP.get_use_abs()){ // if we do not use abs data, ignore to consider the total obj and adjoint source
                    continue;
                }
                
                int id_src_att      = data.id_src_att;
                int id_rec_att      = data.id_rec_att;
                CUSTOMREAL syn_time       = data.travel_time;
                CUSTOMREAL obs_time       = data.time_observation;

                // assign local weight
                CUSTOMREAL  local_weight = _1_CR;

                // evaluate residual_weight_abs （If run_mode == DO_INVERSION, tau_opt always equal 0. But when run_mode == INV_RELOC, we need to consider the change of ortime of earthquakes (swapped receiver)）
                CUSTOMREAL  local_residual = abs(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt);
                CUSTOMREAL* res_weight = IP.get_residual_weight_abs();

                if      (local_residual < res_weight[0])    local_weight *= res_weight[2];
                else if (local_residual > res_weight[1])    local_weight *= res_weight[3];
                else                                        local_weight *= ((local_residual - res_weight[0])/(res_weight[1] - res_weight[0]) * (res_weight[3] - res_weight[2]) + res_weight[2]);


                // evaluate distance_weight_abs
                CUSTOMREAL  local_dis    =   _0_CR;
                Epicentral_distance_sphere(IP.get_rec_point(id_rec_att).lat*DEG2RAD, IP.get_rec_point(id_rec_att).lon*DEG2RAD, IP.get_src_point(id_src_att).lat*DEG2RAD, IP.get_src_point(id_src_att).lon*DEG2RAD, local_dis);
                local_dis *= R_earth;       // rad to km
                CUSTOMREAL* dis_weight = IP.get_distance_weight_abs();

                if      (local_dis < dis_weight[0])         local_weight *= dis_weight[2];
                else if (local_dis > dis_weight[1])         local_weight *= dis_weight[3];
                else                                        local_weight *= ((local_dis - dis_weight[0])/(dis_weight[1] - dis_weight[0]) * (dis_weight[3] - dis_weight[2]) + dis_weight[2]);

                // assign adjoint source
                CUSTOMREAL adjoint_source = IP.get_rec_point(id_rec_att).adjoint_source + (syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt) * data.weight * local_weight;
                IP.set_adjoint_source(id_rec_att, adjoint_source); // set adjoint source to rec_map[id_rec_att]

                // assign adjoint source density
                CUSTOMREAL adjoint_source_density = IP.get_rec_point(id_rec_att).adjoint_source_density + _1_CR;
                IP.set_adjoint_source_density(id_rec_att, adjoint_source_density);

            //
            // common receiver differential traveltime && we use this data
            //
            } else if (data.data_type == DATA_TYPE_CRDIF) {
                if (!((IP.get_use_cr() && !IP.get_is_srcrec_swap()) ||
                        (IP.get_use_cs() &&  IP.get_is_srcrec_swap())))
                    continue;   // if we do not use this data (cr + not swap) or (cs + swap) or (cs + tele), ignore to consider the adjoint source

                int id_src1_att = data.id_src_att;
                int id_src2_att = data.id_pair_att;
                int id_rec_att  = data.id_rec_att;

                CUSTOMREAL syn_dif_time   = data.dif_travel_time;
                CUSTOMREAL obs_dif_time   = data.time_observation;

                // assign local weight
                CUSTOMREAL  local_weight = _1_CR;

                // evaluate residual_weight_abs
                CUSTOMREAL  local_residual = abs(syn_dif_time - obs_dif_time);
                CUSTOMREAL* res_weight;
                if (IP.get_is_srcrec_swap())    res_weight = IP.get_residual_weight_cs();
                else                            res_weight = IP.get_residual_weight_cr();

                if      (local_residual < res_weight[0])    local_weight *= res_weight[2];
                else if (local_residual > res_weight[1])    local_weight *= res_weight[3];
                else                                        local_weight *= ((local_residual - res_weight[0])/(res_weight[1] - res_weight[0]) * (res_weight[3] - res_weight[2]) + res_weight[2]);


                // evaluate distance_weight_abs
                CUSTOMREAL  local_azi1    =   _0_CR;
                Azimuth_sphere(IP.get_rec_point(id_rec_att).lat*DEG2RAD, IP.get_rec_point(id_rec_att).lon*DEG2RAD, IP.get_src_point(id_src1_att).lat*DEG2RAD, IP.get_src_point(id_src1_att).lon*DEG2RAD, local_azi1);
                CUSTOMREAL  local_azi2    =   _0_CR;
                Azimuth_sphere(IP.get_rec_point(id_rec_att).lat*DEG2RAD, IP.get_rec_point(id_rec_att).lon*DEG2RAD, IP.get_src_point(id_src2_att).lat*DEG2RAD, IP.get_src_point(id_src2_att).lon*DEG2RAD, local_azi2);
                CUSTOMREAL  local_azi   = abs(local_azi1 - local_azi2)*RAD2DEG;
                if(local_azi > 180.0)   local_azi = 360.0 - local_azi;


                CUSTOMREAL* azi_weight;
                if (IP.get_is_srcrec_swap())    azi_weight = IP.get_azimuthal_weight_cs();
                else                            azi_weight = IP.get_azimuthal_weight_cr();

                if      (local_azi < azi_weight[0])         local_weight *= azi_weight[2];
                else if (local_azi > azi_weight[1])         local_weight *= azi_weight[3];
                else                                        local_weight *= ((local_azi - azi_weight[0])/(azi_weight[1] - azi_weight[0]) * (azi_weight[3] - azi_weight[2]) + azi_weight[2]);


                // assign adjoint source
                CUSTOMREAL adjoint_source = IP.get_rec_point(id_rec_att).adjoint_source + (syn_dif_time - obs_dif_time) * data.weight * local_weight;
                IP.set_adjoint_source(id_rec_att, adjoint_source);

                // assign adjoint source density
                CUSTOMREAL adjoint_source_density = IP.get_rec_point(id_rec_att).adjoint_source_density + _1_CR;
                IP.set_adjoint_source_density(id_rec_att, adjoint_source_density);

            //
            // common source differential traveltime
            //
            } else if (data.data_type == DATA_TYPE_CSDIF) {
                if (!((IP.get_use_cs() && !IP.get_is_srcrec_swap()) ||
                        (IP.get_use_cr() &&  IP.get_is_srcrec_swap()) ||
                        (IP.get_use_cs() &&  is_tele                )))
                    continue; // if we do not use this data (cs + not swap) or (cr + swap), ignore to consider the total obj and adjoint source

                int id_src_att  = data.id_src_att;
                int id_rec1_att = data.id_rec_att;
                int id_rec2_att = data.id_pair_att;

                CUSTOMREAL syn_dif_time = data.dif_travel_time;
                CUSTOMREAL obs_dif_time = data.time_observation;

                if(is_tele){    // station correction for teleseismic data
                    syn_dif_time = syn_dif_time + IP.get_rec_point(id_rec1_att).sta_correct - IP.get_rec_point(id_rec2_att).sta_correct;
                }

                // assign local weight
                CUSTOMREAL  local_weight = _1_CR;

                // evaluate residual_weight_abs (see the remark in absolute traveltime data for considering tau_opt here)
                CUSTOMREAL  local_residual = abs(syn_dif_time - obs_dif_time + IP.get_rec_point(id_rec1_att).tau_opt - IP.get_rec_point(id_rec2_att).tau_opt);
                CUSTOMREAL* res_weight;
                if (IP.get_is_srcrec_swap() && !is_tele)    res_weight = IP.get_residual_weight_cr();
                else                                        res_weight = IP.get_residual_weight_cs();

                if      (local_residual < res_weight[0])    local_weight *= res_weight[2];
                else if (local_residual > res_weight[1])    local_weight *= res_weight[3];
                else                                        local_weight *= ((local_residual - res_weight[0])/(res_weight[1] - res_weight[0]) * (res_weight[3] - res_weight[2]) + res_weight[2]);


                // evaluate distance_weight_abs
                CUSTOMREAL  local_azi1    =   _0_CR;
                Azimuth_sphere(IP.get_rec_point(id_rec1_att).lat*DEG2RAD, IP.get_rec_point(id_rec1_att).lon*DEG2RAD, IP.get_src_point(id_src_att).lat*DEG2RAD, IP.get_src_point(id_src_att).lon*DEG2RAD, local_azi1);
                CUSTOMREAL  local_azi2    =   _0_CR;
                Azimuth_sphere(IP.get_rec_point(id_rec2_att).lat*DEG2RAD, IP.get_rec_point(id_rec2_att).lon*DEG2RAD, IP.get_src_point(id_src_att).lat*DEG2RAD, IP.get_src_point(id_src_att).lon*DEG2RAD, local_azi2);
                CUSTOMREAL  local_azi   = abs(local_azi1 - local_azi2)*RAD2DEG;
                if(local_azi > 180.0)   local_azi = 360.0 - local_azi;


                CUSTOMREAL* azi_weight;
                if (IP.get_is_srcrec_swap() && !is_tele)    azi_weight = IP.get_azimuthal_weight_cr();
                else                                        azi_weight = IP.get_azimuthal_weight_cs();

                if      (local_azi < azi_weight[0])         local_weight *= azi_weight[2];
                else if (local_azi > azi_weight[1])         local_weight *= azi_weight[3];
                else                                        local_weight *= ((local_azi - azi_weight[0])/(azi_weight[1] - azi_weight[0]) * (azi_weight[3] - azi_weight[2]) + azi_weight[2]);


                // assign adjoint source
                CUSTOMREAL adjoint_source;
                adjoint_source = IP.get_rec_point(id_rec1_att).adjoint_source + (syn_dif_time - obs_dif_time + IP.get_rec_point(id_rec1_att).tau_opt - IP.get_rec_point(id_rec2_att).tau_opt) * data.weight * local_weight;
                IP.set_adjoint_source(id_rec1_att, adjoint_source);

                adjoint_source = IP.get_rec_point(id_rec2_att).adjoint_source - (syn_dif_time - obs_dif_time + IP.get_rec_point(id_rec1_att).tau_opt - IP.get_rec_point(id_rec2_att).tau_opt) * data.weight * local_weight;
                IP.set_adjoint_source(id_rec2_att, adjoint_source);

                // assign adjoint source density
                CUSTOMREAL adjoint_source_density;
                adjoint_source_density = IP.get_rec_point(id_rec1_att).adjoint_source_density + _1_CR;
                IP.set_adjoint_source_density(id_rec1_att, adjoint_source_density);

                adjoint_source_density = IP.get_rec_point(id_rec2_att).adjoint_source_density + _1_CR;
                IP.set_adjoint_source_density(id_rec2_att, adjoint_source_density);
            }

        }

    } // end proc_store_srcrec

    synchronize_all();
}


std::vector<CUSTOMREAL> Receiver:: calculate_obj_and_residual(InputParams& IP) {

    CUSTOMREAL obj           = 0.0;
    CUSTOMREAL obj_abs       = 0.0;
    CUSTOMREAL obj_cs_dif    = 0.0;
    CUSTOMREAL obj_cr_dif    = 0.0;
    CUSTOMREAL obj_tele      = 0.0;
    CUSTOMREAL res           = 0.0;
    CUSTOMREAL res_sq        = 0.0;
    CUSTOMREAL res_abs       = 0.0;
    CUSTOMREAL res_abs_sq    = 0.0;
    CUSTOMREAL res_cs_dif    = 0.0;
    CUSTOMREAL res_cs_dif_sq = 0.0;
    CUSTOMREAL res_cr_dif    = 0.0;
    CUSTOMREAL res_cr_dif_sq = 0.0;
    CUSTOMREAL res_tele      = 0.0;
    CUSTOMREAL res_tele_sq   = 0.0;

    std::vector<CUSTOMREAL> obj_residual;

    if (proc_store_srcrec) {

        for(auto& data : IP.data_vec){
            if (data.dual_data) continue; // dual data is not used for calculating obj and residual

            //
            // absolute traveltime
            //
            if (data.data_type == DATA_TYPE_ABS) {

                int id_src_att      = data.id_src_att;
                int id_rec_att      = data.id_rec_att;
                CUSTOMREAL syn_time       = data.travel_time;
                CUSTOMREAL obs_time       = data.time_observation;
                bool is_tele = (IP.get_src_point(id_src_att).is_out_of_region || IP.get_rec_point(id_rec_att).is_out_of_region);
                if (is_tele){
                    syn_time = syn_time + IP.rec_map[id_rec_att].sta_correct; // station correction for teleseismic data
                }
                // contribute misfit of specific type of data
                res     += 1.0 *          (syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt);
                res_sq  += 1.0 * my_square(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt);

                if (is_tele){
                    obj_tele        +=  1.0 * my_square(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt) * data.weight;
                    res_tele        +=  1.0 *          (syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt);
                    res_tele_sq     +=  1.0 * my_square(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt);

                    if(!IP.get_use_abs())
                        continue;   // if we do not use abs data, ignore to consider the total obj
                    obj     += 1.0 * my_square(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt) * data.weight;
                } else{
                    obj_abs         +=  1.0 * my_square(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt) * data.weight;
                    res_abs         +=  1.0 *          (syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt);
                    res_abs_sq      +=  1.0 * my_square(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt);

                    if (!IP.get_use_abs())
                        continue;   // if we do not use abs data, ignore to consider the total obj
                    obj     += 1.0 * my_square(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt) * data.weight;
                }



            } else if (data.data_type == DATA_TYPE_CRDIF) {  // common receiver differential traveltime

                int id_src1_att = data.id_src_att;
                // int id_src2_att = data.id_pair_att;
                int id_rec_att  = data.id_rec_att;

                CUSTOMREAL syn_dif_time   = data.dif_travel_time;
                CUSTOMREAL obs_dif_time   = data.time_observation;

                bool is_tele = (IP.get_src_point(id_src1_att).is_out_of_region ||
                                IP.get_rec_point(id_rec_att).is_out_of_region);     
                //  no need to check the second src. 1. this src may not at the processor, 
                //  2. src1 is out of the region, then, so does src2. one tele and one local is not allowed in "separate_region_and_tele_src_rec_data"

                // contribute misfit of specific type of data
                res     += 1.0 *          (syn_dif_time - obs_dif_time);
                res_sq  += 1.0 * my_square(syn_dif_time - obs_dif_time);

                if (is_tele){
                    obj_tele        += 1.0 * my_square(syn_dif_time - obs_dif_time)*data.weight;
                    res_tele        += 1.0 *          (syn_dif_time - obs_dif_time);
                    res_tele_sq     += 1.0 * my_square(syn_dif_time - obs_dif_time);

                    if(!IP.get_use_cr())
                        continue;   // if we do not use cr data, ignore to consider the total obj
                    obj     += 1.0 * my_square(syn_dif_time - obs_dif_time)*data.weight;
                } else{
                    obj_cr_dif      += 1.0 * my_square(syn_dif_time - obs_dif_time)*data.weight;
                    res_cr_dif      += 1.0 *          (syn_dif_time - obs_dif_time);
                    res_cr_dif_sq   += 1.0 * my_square(syn_dif_time - obs_dif_time);

                    if (!((IP.get_use_cr() && !IP.get_is_srcrec_swap()) || (IP.get_use_cs() && IP.get_is_srcrec_swap())))
                        continue;   // if we do not use this data (cr + not swap) or (cs + swap), ignore to consider the total obj and adjoint source
                    obj     += 1.0 * my_square(syn_dif_time - obs_dif_time)*data.weight;
                }

            } else if (data.data_type == DATA_TYPE_CSDIF) {   // common source differential traveltime

                int id_src_att  = data.id_src_att;
                int id_rec1_att = data.id_rec_att;
                int id_rec2_att = data.id_pair_att;

                CUSTOMREAL syn_dif_time = data.dif_travel_time;
                CUSTOMREAL obs_dif_time = data.time_observation;

                bool is_tele = (IP.get_src_point(id_src_att).is_out_of_region || 
                                IP.get_rec_point(id_rec1_att).is_out_of_region || 
                                IP.get_rec_point(id_rec2_att).is_out_of_region);
                if(is_tele){
                    syn_dif_time = syn_dif_time + IP.rec_map[id_rec1_att].sta_correct - IP.rec_map[id_rec2_att].sta_correct; // station correction for teleseismic data
                }

                // contribute misfit of specific type of data
                res     += 1.0 *          (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                res_sq  += 1.0 * my_square(syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);

                if (is_tele){
                    obj_tele        += 1.0 * my_square(syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt)*data.weight;
                    res_tele        += 1.0 *          (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                    res_tele_sq     += 1.0 * my_square(syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);

                    if(!IP.get_use_cs())
                        continue;   // if we do not use cs data, ignore to consider the total obj
                    obj     += 1.0 * my_square(syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt) * data.weight;

                } else{
                    obj_cs_dif      += 1.0 * my_square(syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt)*data.weight;
                    res_cs_dif      += 1.0 *          (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                    res_cs_dif_sq   += 1.0 * my_square(syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);

                    if (!((IP.get_use_cs() && !IP.get_is_srcrec_swap()) || (IP.get_use_cr() && IP.get_is_srcrec_swap())))
                        continue; // if we do not use this data (cs + not swap) or (cr + swap), ignore to consider the total obj and adjoint source
                    obj     += 1.0 * my_square(syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt) * data.weight;
                }
            }

        } // end of loop over data

    } // end proc_store_srcrec


    broadcast_cr_single_sub(obj,0);
    broadcast_cr_single_sub(obj_abs,0);
    broadcast_cr_single_sub(obj_cs_dif,0);
    broadcast_cr_single_sub(obj_cr_dif,0);
    broadcast_cr_single_sub(obj_tele,0);
    broadcast_cr_single_sub(res,0);
    broadcast_cr_single_sub(res_sq,0);
    broadcast_cr_single_sub(res_abs,0);
    broadcast_cr_single_sub(res_abs_sq,0);
    broadcast_cr_single_sub(res_cs_dif,0);
    broadcast_cr_single_sub(res_cs_dif_sq,0);
    broadcast_cr_single_sub(res_cr_dif,0);
    broadcast_cr_single_sub(res_cr_dif_sq,0);
    broadcast_cr_single_sub(res_tele,0);
    broadcast_cr_single_sub(res_tele_sq,0);

    broadcast_cr_single(obj,0);
    broadcast_cr_single(obj_abs,0);
    broadcast_cr_single(obj_cs_dif,0);
    broadcast_cr_single(obj_cr_dif,0);
    broadcast_cr_single(obj_tele,0);
    broadcast_cr_single(res,0);
    broadcast_cr_single(res_sq,0);
    broadcast_cr_single(res_abs,0);
    broadcast_cr_single(res_abs_sq,0);
    broadcast_cr_single(res_cs_dif,0);
    broadcast_cr_single(res_cs_dif_sq,0);
    broadcast_cr_single(res_cr_dif,0);
    broadcast_cr_single(res_cr_dif_sq,0);
    broadcast_cr_single(res_tele,0);
    broadcast_cr_single(res_tele_sq,0);

    obj_residual = {obj, obj_abs, obj_cs_dif, obj_cr_dif, obj_tele, res, res_sq, res_abs, res_abs_sq, res_cs_dif, res_cs_dif_sq, res_cr_dif, res_cr_dif_sq, res_tele, res_tele_sq};

    for(int i = 0; i < (int)obj_residual.size(); i++){
        allreduce_cr_sim_single_inplace(obj_residual[i]);
    }

    return obj_residual;
}


bool Receiver::check_if_receiver_is_in_this_subdomain(Grid& grid, const CUSTOMREAL& rec_lon, const CUSTOMREAL& rec_lat, const CUSTOMREAL& rec_r) {

    bool is_in_subdomain = false;

    if (grid.get_lon_min_loc() <= rec_lon && rec_lon < grid.get_lon_max_loc() && \
        grid.get_lat_min_loc() <= rec_lat && rec_lat < grid.get_lat_max_loc() && \
        grid.get_r_min_loc()   <= rec_r   && rec_r   < grid.get_r_max_loc()   ) {

        // check if the receiver is on the upper boundary of the subdomain
        // if so, the interpolation will be failed because *_rec_p1 cannot be defined.
        if (isZero(rec_lon - grid.get_lon_max_loc())
         || isZero(rec_lat - grid.get_lat_max_loc())
         || isZero(rec_r - grid.get_r_max_loc())) {
            is_in_subdomain = false;
        } else {
            is_in_subdomain = true;
        }
    }

    return is_in_subdomain;
}


CUSTOMREAL Receiver::interpolate_travel_time(Grid& grid, InputParams& IP, const int id_rec_att) {
    // calculate the travel time of the receiver by 3d linear interpolation

    // get the reference for a receiver
    const SrcRecInfo rec = IP.get_rec_point_bcast(id_rec_att);  // get by main of level 2 and broadcast to all processes in level 2

    // copy some parameters
    CUSTOMREAL delta_lon = grid.get_delta_lon();
    CUSTOMREAL delta_lat = grid.get_delta_lat();
    CUSTOMREAL delta_r   = grid.get_delta_r();

    // store receiver position in radian
    CUSTOMREAL rec_lon = rec.lon*DEG2RAD;
    CUSTOMREAL rec_lat = rec.lat*DEG2RAD;
    CUSTOMREAL rec_r = depth2radius(rec.dep); // r in km

    // check if the receiver is in this subdomain
    bool is_in_subdomain = check_if_receiver_is_in_this_subdomain(grid, rec_lon, rec_lat, rec_r);

    // check the rank where the source is located
    int rec_rank = -1;
    int n_subdom_rec = 0;
    bool* rec_flags = new bool[nprocs];
    allgather_bool_single(&is_in_subdomain, rec_flags);
    for (int i = 0; i < nprocs; i++) {
        if (rec_flags[i]) {
            rec_rank = i;
            //break; // this break means that the first subdomain is used if the receiver is in multiple subdomains (ghost layer)
            n_subdom_rec++;
        }
    }
    delete[] rec_flags;
     // check if the receiver is in the global domain
    if (rec_rank == -1) {
        std::cout << "Error: the receiver is not in the global domain" << std::endl;
        // print rec
        std::cout << " name_rec: " << rec.name << " depth: " << rec.dep << " lat: " << rec.lat << " lon: " << rec.lon << std::endl;
        // print boundary
        //std::cout << "lon min max rec: " << grid.get_lon_min_loc() << " " << grid.get_lon_max_loc() << " " << rec_lon << std::endl;
        //std::cout << "lat min max rec: " << grid.get_lat_min_loc() << " " << grid.get_lat_max_loc() << " " << rec_lat << std::endl;
        //std::cout << "r min max rec: " << grid.get_r_min_loc() << " " << grid.get_r_max_loc() << " " << rec_r << std::endl;

        std::cout << "lon+bound min max rec: " << (grid.get_lon_min_loc() - delta_lon)*RAD2DEG     << " " << (grid.get_lon_max_loc() + delta_lon)*RAD2DEG     << " " << rec_lon*RAD2DEG    << std::endl;
        std::cout << "lat+bound min max rec: " << (grid.get_lat_min_loc() - delta_lat)*RAD2DEG     << " " << (grid.get_lat_max_loc() + delta_lat)*RAD2DEG     << " " << rec_lat*RAD2DEG    << std::endl;
        std::cout << "r+bound min max rec: "   << radius2depth(grid.get_r_min_loc()   - delta_r  ) << " " << radius2depth(grid.get_r_max_loc()   + delta_r  ) << " " << radius2depth(rec_r)<< std::endl;
        exit(1);
    }

    CUSTOMREAL vinterp = 0.0;

    if (is_in_subdomain) {
        // calculate the interpolated travel time and broadcast it

        // descretize source position (LOCAL) ID)
        int i_rec = std::floor((rec_lon - grid.get_lon_min_loc()) / delta_lon);
        int j_rec = std::floor((rec_lat - grid.get_lat_min_loc()) / delta_lat);
        int k_rec = std::floor((rec_r   - grid.get_r_min_loc())   / delta_r);

        // discretized receiver position
        CUSTOMREAL dis_rec_lon = grid.p_loc_1d[i_rec];
        CUSTOMREAL dis_rec_lat = grid.t_loc_1d[j_rec];
        CUSTOMREAL dis_rec_r   = grid.r_loc_1d[k_rec];

        // relative position errors
        CUSTOMREAL e_lon = std::min({_1_CR,(rec_lon - dis_rec_lon)/delta_lon});
        CUSTOMREAL e_lat = std::min({_1_CR,(rec_lat - dis_rec_lat)/delta_lat});
        CUSTOMREAL e_r   = std::min({_1_CR,(rec_r   - dis_rec_r)  /delta_r});

        // numerical precision error of std::floor
        if (e_lon >= _1_CR) {
            e_lon = e_lon - _1_CR;
            i_rec++;
        } else if (e_lon < 0) {
            e_lon = e_lon + _1_CR;
            i_rec--;
        }

        if (e_lat == _1_CR) {
            e_lat = 0.0;
            j_rec++;
        } else if (e_lat < 0) {
            e_lat = e_lat + _1_CR;
            j_rec--;
        }

        if (e_r == _1_CR) {
            e_r = 0.0;
            k_rec++;
        } else if (e_r < 0) {
            e_r = e_r + _1_CR;
            k_rec--;
        }

//        if(if_verbose){
//            std::cout << "(rec_lon - dis_rec_lon)/dlon: " << (rec_lon - dis_rec_lon)/delta_lon << std::endl;
//            std::cout << "(rec_lat - dis_rec_lat)/dlat: " << (rec_lat - dis_rec_lat)/delta_lat << std::endl;
//            std::cout << "(rec_r   - dis_rec_r  )/dr: "   << (rec_r   - dis_rec_r  )/delta_r << std::endl;
//            std::cout << "rec_lon, dis_rec_lon, delta_lon: " << rec_lon << " " << dis_rec_lon << " " << delta_lon << std::endl;
//            std::cout << "rec_lat, dis_rec_lat, delta_lat: " << rec_lat << " " << dis_rec_lat << " " << delta_lat << std::endl;
//            std::cout << "rec_r, dis_rec_r, delta_r  : " << rec_r << ", " << dis_rec_r << ", " <<  delta_r  << std::endl;
//            std::cout << "loc_K. k_rec: " << loc_K << "," << k_rec << std::endl;
//            std::cout << "r_loc_1d[loc_K-1]: " << grid.r_loc_1d[loc_K-1] << std::endl;
//        }

        int i_rec_p1 = i_rec + 1;
        int j_rec_p1 = j_rec + 1;
        int k_rec_p1 = k_rec + 1;

        // exclude the points if they are out of the domain
        if (i_rec_p1 > loc_I-1 \
         || j_rec_p1 > loc_J-1 \
         || k_rec_p1 > loc_K-1) {
            // exit(1) as the source is out of the domain
            std::cout << "Error: the receiver is out of the domain" << std::endl;
            std::cout << " name_rec: " << rec.name << " depth: " << rec.dep << " lat: " << rec.lat << " lon: " << rec.lon << std::endl;
            std::cout << "lon min max rec: " << grid.get_lon_min_loc()*RAD2DEG << " " << grid.get_lon_max_loc()*RAD2DEG << " " << rec_lon*RAD2DEG << std::endl;
            std::cout << "lat min max rec: " << grid.get_lat_min_loc()*RAD2DEG << " " << grid.get_lat_max_loc()*RAD2DEG << " " << rec_lat*RAD2DEG << std::endl;
            std::cout << "r min max rec: " << radius2depth(grid.get_r_min_loc()) << " " << radius2depth(grid.get_r_max_loc()) << " " << radius2depth(rec_r) << std::endl;
            std::cout << "i_rec: " << i_rec << " j_rec: " << j_rec << " k_rec: " << k_rec << std::endl;
            std::cout << "i_rec_p1: " << i_rec_p1 << " j_rec_p1: " << j_rec_p1 << " k_rec_p1: " << k_rec_p1 << std::endl;
            std::cout << "loc_I-1: " << loc_I-1 << " loc_J-1: " << loc_J-1 << " loc_K-1: " << loc_K-1 << std::endl;
            //finalize_mpi();
            exit(1);
         }


        vinterp = (_1_CR - e_lon) * (_1_CR - e_lat) * (_1_CR - e_r) * grid.T_loc[I2V(i_rec,   j_rec,   k_rec)]   \
                +          e_lon  * (_1_CR - e_lat) * (_1_CR - e_r) * grid.T_loc[I2V(i_rec_p1,j_rec,   k_rec)]   \
                + (_1_CR - e_lon) *          e_lat  * (_1_CR - e_r) * grid.T_loc[I2V(i_rec,   j_rec_p1,k_rec)]   \
                + (_1_CR - e_lon) * (_1_CR - e_lat) *          e_r  * grid.T_loc[I2V(i_rec,   j_rec,   k_rec_p1)] \
                +          e_lon  *          e_lat  * (_1_CR - e_r) * grid.T_loc[I2V(i_rec_p1,j_rec_p1,k_rec)]   \
                +          e_lon  * (_1_CR - e_lat) *          e_r  * grid.T_loc[I2V(i_rec_p1,j_rec,   k_rec_p1)] \
                + (_1_CR - e_lon) *          e_lat  *          e_r  * grid.T_loc[I2V(i_rec,   j_rec_p1,k_rec_p1)] \
                +          e_lon  *          e_lat  *          e_r  * grid.T_loc[I2V(i_rec_p1,j_rec_p1,k_rec_p1)];

        //std::cout << "DEBUG near and vinterp : " << grid.T_loc[I2V(i_rec,j_rec,k_rec)] << ", " << vinterp << std::endl;
        // std::cout << "times: " << grid.T_loc[I2V(i_rec,   j_rec,   k_rec)] << ", "
        //           << grid.T_loc[I2V(i_rec_p1,j_rec,   k_rec)] << ", "
        //           << grid.T_loc[I2V(i_rec,   j_rec_p1,k_rec)] << ", "
        //           << grid.T_loc[I2V(i_rec,   j_rec,   k_rec_p1)] << ", "
        //           << grid.T_loc[I2V(i_rec_p1,j_rec_p1,k_rec)] << ", "
        //           << grid.T_loc[I2V(i_rec_p1,j_rec,   k_rec_p1)] << ", "
        //           << grid.T_loc[I2V(i_rec,   j_rec_p1,k_rec_p1)] << ", "
        //           << grid.T_loc[I2V(i_rec_p1,j_rec_p1,k_rec_p1)] << ", "
        //           << std::endl;
        // std::cout << "rec: " << rec.name << ", lat: " << rec_lat
        //           << ", lon: " << rec_lon << ", dep: " << rec.dep
        //           << ", time: " << vinterp
        //           << std::endl;

        // broadcast interpolated travel time
        //broadcast_cr_single(vinterp, rec_rank);
        allreduce_cr_inplace(&vinterp, 1);

    } else {
        // receive the calculated traveltime
        //broadcast_cr_single(vinterp, rec_rank);
        allreduce_cr_inplace(&vinterp, 1);
    }

    // use an averaged value if the receiver is in multiple subdomains
    vinterp /= n_subdom_rec;

    // return the calculated travel time
    return vinterp;
}



void Receiver::init_vars_src_reloc(InputParams& IP){
    if (proc_store_srcrec) {

        // calculate gradient of travel time at each receiver (swapped source)
        for (auto iter = IP.rec_map.begin(); iter != IP.rec_map.end(); iter++) {
            if (!iter->second.is_stop){
                iter->second.grad_tau                   = _0_CR;
            }
            iter->second.grad_chi_i                 = _0_CR;
            iter->second.grad_chi_j                 = _0_CR;
            iter->second.grad_chi_k                 = _0_CR;
            iter->second.Ndata                      = 0;
            iter->second.sum_weight                 = _0_CR;    // what does it mean?
            iter->second.vobj_src_reloc_old         = iter->second.vobj_src_reloc;
            iter->second.vobj_src_reloc             = _0_CR;

            iter->second.vobj_grad_norm_src_reloc   = _0_CR;
        }

    }
}

void Receiver::calculate_T_gradient(InputParams& IP, Grid& grid, const int id_src_att){

    if (subdom_main){

        // share the traveltime values on the corner points of the subdomains for interpolation
        // this is not necessary for sweeping (as the stencil is closs shape)
        grid.send_recev_boundary_data(grid.T_loc);
        grid.send_recev_boundary_data_kosumi(grid.T_loc);

        if (proc_store_srcrec){     // main of level 2
            
            // 1. estimate the number of traveltime to be calculated for this source
            int n_time = 0;
            int data_begin = IP.src_map[id_src_att].data_begin;
            int data_end   = IP.src_map[id_src_att].data_end;
            for (int i_data = data_begin; i_data < data_end; i_data++){
                auto& data = IP.data_vec[i_data];    
                // case 1: absolute traveltime for reloc
                if (data.data_type == DATA_TYPE_ABS && IP.get_use_abs_reloc()){   // abs data && we use it
                    n_time += 1;
                // case 2: common receiver (swapped source) double difference (double source, or double swapped receiver) for reloc
                // in reloc, must swapped. thus, use cr mean sc here
                } else if (data.data_type == DATA_TYPE_CSDIF && IP.get_use_cr_reloc()) {  // common receiver data (swapped common source) and we use it.
                    n_time += 2;
                } else if (data.data_type == DATA_TYPE_CRDIF && IP.get_use_cs_reloc()) {    // common source data (swapped common receiver) and we use it.
                    n_time += 1;
                } else {
                    std::cout << "error type of data in reloc" << std::endl;
                    exit(1);
                }
            }
            broadcast_i_single(n_time, 0);   // (level 2) broadcast the number of traveltime to be calculated for this source

            // 2. loop all data, to calculate traveltime
            for (int i_data = data_begin; i_data < data_end; i_data++){
                auto& data = IP.data_vec[i_data];

                // case 1: absolute traveltime for reloc
                if (data.data_type == DATA_TYPE_ABS && IP.get_use_abs_reloc()){   // abs data && we use it
                    // send id_rec_att
                    int id_rec_att = data.id_rec_att;
                    broadcast_i_single(id_rec_att, 0);

                    // calculate travel time gradient
                    std::vector<CUSTOMREAL> DTijk = calculate_T_gradient_one_rec(grid, IP, id_rec_att);
                    data.DTi_pair[0] = DTijk[0];
                    data.DTj_pair[0] = DTijk[1];
                    data.DTk_pair[0] = DTijk[2];

                } else if (data.data_type == DATA_TYPE_CSDIF && IP.get_use_cr_reloc()) {  // common receiver data (swapped common source) and we use it.
                    // send id_rec_att for the first receiver
                    int id_rec1_att = data.id_rec_att;
                    broadcast_i_single(id_rec1_att, 0);
                    // calculate travel time gradient for the first receiver
                    std::vector<CUSTOMREAL> DTijk = calculate_T_gradient_one_rec(grid, IP, id_rec1_att);
                    data.DTi_pair[0]  = DTijk[0];
                    data.DTj_pair[0]  = DTijk[1];
                    data.DTk_pair[0]  = DTijk[2];

                    // send id_rec_att for the second receiver
                    int id_rec2_att = data.id_pair_att;
                    broadcast_i_single(id_rec2_att, 0);
                    // calculate travel time gradient for the second receiver
                    std::vector<CUSTOMREAL> DTijk_2 = calculate_T_gradient_one_rec(grid, IP, id_rec2_att);
                    data.DTi_pair[1]  = DTijk_2[0];
                    data.DTj_pair[1]  = DTijk_2[1];
                    data.DTk_pair[1]  = DTijk_2[2];
                    

                } else if (data.data_type == DATA_TYPE_CRDIF && IP.get_use_cs_reloc()) {    // common source data (swapped common receiver) and we use it.
                    // send id_rec_att
                    int id_rec_att = data.id_rec_att;
                    broadcast_i_single(id_rec_att, 0);
                    // calculate travel time gradient
                    std::vector<CUSTOMREAL> DTijk = calculate_T_gradient_one_rec(grid, IP, id_rec_att);
                    data.DTi_pair[0]  = DTijk[0];
                    data.DTj_pair[0]  = DTijk[1];
                    data.DTk_pair[0]  = DTijk[2];

                } else {
                    std::cout << "error type of data in reloc" << std::endl;
                    exit(1);
                }
            }

        } else {    // other ranks which do not have src_map, rec_map, data_vec

            // 1. receive the number of traveltime to be calculated for this source
            int n_time = 0;
            broadcast_i_single(n_time, 0);   // (level 2) receive the number of traveltime to be calculated for this source

            // 2. loop all data, to calculate traveltime
            for (int i_time = 0; i_time < n_time; i_time++){
                // receive id_rec_att
                int id_rec_att = 0;
                broadcast_i_single(id_rec_att, 0);
                // calculate travel time gradient
                std::vector<CUSTOMREAL> DTijk = calculate_T_gradient_one_rec(grid, IP, id_rec_att);
                (void) DTijk; // avoid compiler warning
            }
        }
    } // end subdom_main

}


std::vector<CUSTOMREAL> Receiver::calculate_T_gradient_one_rec(Grid& grid, InputParams& IP, const int id_rec_att){

    // calculate the travel time of the receiver by 3d linear interpolation

    // get the reference for a receiver
    const SrcRecInfo rec = IP.get_rec_point_bcast(id_rec_att);

    // copy some parameters
    CUSTOMREAL delta_lon = grid.get_delta_lon();
    CUSTOMREAL delta_lat = grid.get_delta_lat();
    CUSTOMREAL delta_r   = grid.get_delta_r();

    // store receiver position in radian
    CUSTOMREAL rec_lon = rec.lon*DEG2RAD;
    CUSTOMREAL rec_lat = rec.lat*DEG2RAD;
    CUSTOMREAL rec_r = depth2radius(rec.dep); // r in km

    // check if the receiver is in this subdomain
    bool is_in_subdomain = check_if_receiver_is_in_this_subdomain(grid, rec_lon, rec_lat, rec_r);

    // check the rank where the source is located
    int rec_rank = -1;
    bool* rec_flags = new bool[nprocs];
    allgather_bool_single(&is_in_subdomain, rec_flags);
    for (int i = 0; i < nprocs; i++) {
        if (rec_flags[i]) {
            rec_rank = i;
            //break; // this break means that the first subdomain is used if the receiver is in multiple subdomains (ghost layer)
        }
    }

    // count the number of subdomains where the receiver is located
    int ndoms_rec = 0;
    for (int i = 0; i < nprocs; i++) {
        if (rec_flags[i]) {
            ndoms_rec++;
        }
    }

    // check if the receiver is in the global domain
    if (rec_rank == -1) {
        std::cout << "Error: the receiver is not in the global domain" << std::endl;
        // print rec
        std::cout << " id_rec: " << rec.id << " name: " << rec.name << " depth: " << rec.dep << " lat: " << rec.lat << " lon: " << rec.lon << std::endl;
        std::cout << "lon+bound min max rec: " << (grid.get_lon_min_loc() - delta_lon)*RAD2DEG     << " " << (grid.get_lon_max_loc() + delta_lon)*RAD2DEG     << " " << rec_lon*RAD2DEG    << std::endl;
        std::cout << "lat+bound min max rec: " << (grid.get_lat_min_loc() - delta_lat)*RAD2DEG     << " " << (grid.get_lat_max_loc() + delta_lat)*RAD2DEG     << " " << rec_lat*RAD2DEG    << std::endl;
        std::cout << "r+bound min max rec: "   << radius2depth(grid.get_r_min_loc()   - delta_r  ) << " " << radius2depth(grid.get_r_max_loc()   + delta_r  ) << " " << radius2depth(rec_r)<< std::endl;
        exit(1);
    }

    CUSTOMREAL DTi = 0.0, DTj = 0.0, DTk = 0.0;

    if (is_in_subdomain) {

        // descretized source position (LOCAL) ID)
        int i_rec = std::floor((rec_lon - grid.get_lon_min_loc()) / delta_lon);
        int j_rec = std::floor((rec_lat - grid.get_lat_min_loc()) / delta_lat);
        int k_rec = std::floor((rec_r   - grid.get_r_min_loc())   / delta_r);

        // std::cout << "lon: " << rec.lon << ", lat: " << rec.lat << ", dep: " << rec.dep << std::endl;
        // std::cout << "i_rec: " << i_rec << ", j_rec: " << j_rec << ", k_rec: " << k_rec << std::endl;

        // discretized receiver position
        CUSTOMREAL dis_rec_lon = grid.p_loc_1d[i_rec];
        CUSTOMREAL dis_rec_lat = grid.t_loc_1d[j_rec];
        CUSTOMREAL dis_rec_r   = grid.r_loc_1d[k_rec];

        // relative position errors
        CUSTOMREAL e_lon = std::min({_1_CR,(rec_lon - dis_rec_lon)/delta_lon});
        CUSTOMREAL e_lat = std::min({_1_CR,(rec_lat - dis_rec_lat)/delta_lat});
        CUSTOMREAL e_r   = std::min({_1_CR,(rec_r   - dis_rec_r)  /delta_r});

        // numerical precision errors on std::floor
        if (e_lon >= _1_CR) {
            e_lon = e_lon - _1_CR;
            i_rec++;
        } else if (e_lon < 0) {
            e_lon = e_lon + _1_CR;
            i_rec--;
        }

        if (e_lat == _1_CR) {
            e_lat = 0.0;
            j_rec++;
        } else if (e_lat < 0) {
            e_lat = e_lat + _1_CR;
            j_rec--;
        }

        if (e_r == _1_CR) {
            e_r = 0.0;
            k_rec++;
        } else if (e_r < 0) {
            e_r = e_r + _1_CR;
            k_rec--;
        }

        int i_rec_p1 = i_rec + 1;
        int j_rec_p1 = j_rec + 1;
        int k_rec_p1 = k_rec + 1;
        int i_rec_m1 = i_rec - 1;
        int j_rec_m1 = j_rec - 1;
        int k_rec_m1 = k_rec - 1;
        int i_rec_p1p1 = i_rec + 2;
        int j_rec_p1p1 = j_rec + 2;
        int k_rec_p1p1 = k_rec + 2;

        // treatment for the nodes on the bounary
        // if not, sources cannot go over the subdomain boundary
        if (i_rec == 0)
            i_rec_m1 += 1;
        if (j_rec == 0)
            j_rec_m1 += 1;
        if (k_rec == 0)
            k_rec_m1 += 1;
        if (i_rec_p1 == loc_I-1)
            i_rec_p1p1 -= 1;
        if (j_rec_p1 == loc_J-1)
            j_rec_p1p1 -= 1;
        if (k_rec_p1 == loc_K-1)
            k_rec_p1p1 -= 1;

        CUSTOMREAL DT1, DT2, DT3, DT4, DT5, DT6, DT7, DT8;
        DT1 = (grid.T_loc[I2V(i_rec,     j_rec,     k_rec_p1)]   - grid.T_loc[I2V(i_rec,     j_rec,     k_rec_m1)]) / _2_CR / delta_r;
        DT2 = (grid.T_loc[I2V(i_rec_p1,  j_rec,     k_rec_p1)]   - grid.T_loc[I2V(i_rec_p1,  j_rec,     k_rec_m1)]) / _2_CR / delta_r;
        DT3 = (grid.T_loc[I2V(i_rec,     j_rec_p1,  k_rec_p1)]   - grid.T_loc[I2V(i_rec,     j_rec_p1,  k_rec_m1)]) / _2_CR / delta_r;
        DT4 = (grid.T_loc[I2V(i_rec_p1,  j_rec_p1,  k_rec_p1)]   - grid.T_loc[I2V(i_rec_p1,  j_rec_p1,  k_rec_m1)]) / _2_CR / delta_r;
        DT5 = (grid.T_loc[I2V(i_rec,     j_rec,     k_rec_p1p1)] - grid.T_loc[I2V(i_rec,     j_rec,     k_rec)])    / _2_CR / delta_r;
        DT6 = (grid.T_loc[I2V(i_rec_p1,  j_rec,     k_rec_p1p1)] - grid.T_loc[I2V(i_rec_p1,  j_rec,     k_rec)])    / _2_CR / delta_r;
        DT7 = (grid.T_loc[I2V(i_rec,     j_rec_p1,  k_rec_p1p1)] - grid.T_loc[I2V(i_rec,     j_rec_p1,  k_rec)])    / _2_CR / delta_r;
        DT8 = (grid.T_loc[I2V(i_rec_p1,  j_rec_p1,  k_rec_p1p1)] - grid.T_loc[I2V(i_rec_p1,  j_rec_p1,  k_rec)])    / _2_CR / delta_r;

        DTk =   (_1_CR - e_lon) * (_1_CR - e_lat) * (_1_CR - e_r) * DT1
              +          e_lon  * (_1_CR - e_lat) * (_1_CR - e_r) * DT2
              + (_1_CR - e_lon) *          e_lat  * (_1_CR - e_r) * DT3
              +          e_lon  *          e_lat  * (_1_CR - e_r) * DT4
              + (_1_CR - e_lon) * (_1_CR - e_lat) *          e_r  * DT5
              +          e_lon  * (_1_CR - e_lat) *          e_r  * DT6
              + (_1_CR - e_lon) *          e_lat  *          e_r  * DT7
              +          e_lon  *          e_lat  *          e_r  * DT8;

        DT1 = (grid.T_loc[I2V(i_rec,     j_rec_p1,    k_rec   )] - grid.T_loc[I2V(i_rec,     j_rec_m1,  k_rec   )]) / _2_CR / delta_lat;
        DT2 = (grid.T_loc[I2V(i_rec_p1,  j_rec_p1,    k_rec   )] - grid.T_loc[I2V(i_rec_p1,  j_rec_m1,  k_rec   )]) / _2_CR / delta_lat;
        DT3 = (grid.T_loc[I2V(i_rec,     j_rec_p1p1,  k_rec   )] - grid.T_loc[I2V(i_rec,     j_rec,     k_rec   )]) / _2_CR / delta_lat;
        DT4 = (grid.T_loc[I2V(i_rec_p1,  j_rec_p1p1,  k_rec   )] - grid.T_loc[I2V(i_rec_p1,  j_rec,     k_rec   )]) / _2_CR / delta_lat;
        DT5 = (grid.T_loc[I2V(i_rec,     j_rec_p1,    k_rec_p1)] - grid.T_loc[I2V(i_rec,     j_rec_m1,  k_rec_p1)]) / _2_CR / delta_lat;
        DT6 = (grid.T_loc[I2V(i_rec_p1,  j_rec_p1,    k_rec_p1)] - grid.T_loc[I2V(i_rec_p1,  j_rec_m1,  k_rec_p1)]) / _2_CR / delta_lat;
        DT7 = (grid.T_loc[I2V(i_rec,     j_rec_p1p1,  k_rec_p1)] - grid.T_loc[I2V(i_rec,     j_rec,     k_rec_p1)]) / _2_CR / delta_lat;
        DT8 = (grid.T_loc[I2V(i_rec_p1,  j_rec_p1p1,  k_rec_p1)] - grid.T_loc[I2V(i_rec_p1,  j_rec,     k_rec_p1)]) / _2_CR / delta_lat;

        DTj =   (_1_CR - e_lon) * (_1_CR - e_lat) * (_1_CR - e_r) * DT1
              +          e_lon  * (_1_CR - e_lat) * (_1_CR - e_r) * DT2
              + (_1_CR - e_lon) *          e_lat  * (_1_CR - e_r) * DT3
              +          e_lon  *          e_lat  * (_1_CR - e_r) * DT4
              + (_1_CR - e_lon) * (_1_CR - e_lat) *          e_r  * DT5
              +          e_lon  * (_1_CR - e_lat) *          e_r  * DT6
              + (_1_CR - e_lon) *          e_lat  *          e_r  * DT7
              +          e_lon  *          e_lat  *          e_r  * DT8;

        DT1 = (grid.T_loc[I2V(i_rec_p1,    j_rec   ,  k_rec   )] - grid.T_loc[I2V(i_rec_m1,  j_rec   ,  k_rec   )]) / _2_CR / delta_lon;
        DT2 = (grid.T_loc[I2V(i_rec_p1p1,  j_rec   ,  k_rec   )] - grid.T_loc[I2V(i_rec,     j_rec   ,  k_rec   )]) / _2_CR / delta_lon;
        DT3 = (grid.T_loc[I2V(i_rec_p1,    j_rec_p1,  k_rec   )] - grid.T_loc[I2V(i_rec_m1,  j_rec_p1,  k_rec   )]) / _2_CR / delta_lon;
        DT4 = (grid.T_loc[I2V(i_rec_p1p1,  j_rec_p1,  k_rec   )] - grid.T_loc[I2V(i_rec,     j_rec_p1,  k_rec   )]) / _2_CR / delta_lon;
        DT5 = (grid.T_loc[I2V(i_rec_p1,    j_rec   ,  k_rec_p1)] - grid.T_loc[I2V(i_rec_m1,  j_rec   ,  k_rec_p1)]) / _2_CR / delta_lon;
        DT6 = (grid.T_loc[I2V(i_rec_p1p1,  j_rec   ,  k_rec_p1)] - grid.T_loc[I2V(i_rec,     j_rec   ,  k_rec_p1)]) / _2_CR / delta_lon;
        DT7 = (grid.T_loc[I2V(i_rec_p1,    j_rec_p1,  k_rec_p1)] - grid.T_loc[I2V(i_rec_m1,  j_rec_p1,  k_rec_p1)]) / _2_CR / delta_lon;
        DT8 = (grid.T_loc[I2V(i_rec_p1p1,  j_rec_p1,  k_rec_p1)] - grid.T_loc[I2V(i_rec,     j_rec_p1,  k_rec_p1)]) / _2_CR / delta_lon;

        DTi =   (_1_CR - e_lon) * (_1_CR - e_lat) * (_1_CR - e_r) * DT1
              +          e_lon  * (_1_CR - e_lat) * (_1_CR - e_r) * DT2
              + (_1_CR - e_lon) *          e_lat  * (_1_CR - e_r) * DT3
              +          e_lon  *          e_lat  * (_1_CR - e_r) * DT4
              + (_1_CR - e_lon) * (_1_CR - e_lat) *          e_r  * DT5
              +          e_lon  * (_1_CR - e_lat) *          e_r  * DT6
              + (_1_CR - e_lon) *          e_lat  *          e_r  * DT7
              +          e_lon  *          e_lat  *          e_r  * DT8;
        // std::cout << grid.T_loc[I2V(i_rec    + 1,  j_rec   ,  k_rec   )] << ", " << grid.T_loc[I2V(i_rec    - 1,  j_rec   ,  k_rec   )]
        //           << ", " << _2_CR * delta_lon << std::endl;
        // std::cout << DT1 << "," << DT2 << "," << DT3 << "," << DT4 << "," << DT5 << "," << DT6 << "," << DT7 << "," << DT8 <<std::endl;
        // std::cout << "DTi: " << DTi << std::endl;

        // sum the DTi, DTj, DTk over all subdomains
        allreduce_cr_inplace(&DTi, 1);
        allreduce_cr_inplace(&DTj, 1);
        allreduce_cr_inplace(&DTk, 1);

    } else {
        // sum the DTi, DTj, DTk over all subdomains
        allreduce_cr_inplace(&DTi, 1);
        allreduce_cr_inplace(&DTj, 1);
        allreduce_cr_inplace(&DTk, 1);
    }

    // average the DTi, DTj, DTk by the number of subdomains where the receiver is located
    DTi /= ndoms_rec;
    DTj /= ndoms_rec;
    DTk /= ndoms_rec;

    // store the calculated travel time TODO: should be stored in data as DT is dependent with src-rec pair
    std::vector<CUSTOMREAL> DTijk(3);
    DTijk[2] = DTk;
    DTijk[1] = DTj;
    DTijk[0] = DTi;


    delete[] rec_flags;

    return DTijk;
}


// void Receiver::divide_optimal_origin_time_by_summed_weight(InputParams& IP) {
//     if (proc_store_srcrec) {

//         for (auto iter = IP.rec_map.begin(); iter != IP.rec_map.end();  iter++) {
//             if (IP.rec_map[iter->first].is_stop) continue; // keep the completed tau_opt

//             iter->second.tau_opt /= iter->second.sum_weight;

//             //std::cout << "DEBUG1: id_sim" << id_sim << ", name: " << iter->first << ", ortime: " << iter->second.tau_opt <<std::endl;
//         }
//     }
//     //synchronize_all_world(); // not necessary because allreduce is already synchronizing communication
// }

std::vector<CUSTOMREAL> Receiver::calculate_obj_reloc(InputParams& IP, int i_iter){

    CUSTOMREAL obj           = 0.0;
    CUSTOMREAL obj_abs       = 0.0;
    CUSTOMREAL obj_cs_dif    = 0.0;
    CUSTOMREAL obj_cr_dif    = 0.0;
    CUSTOMREAL obj_tele      = 0.0;
    CUSTOMREAL res           = 0.0;
    CUSTOMREAL res_sq        = 0.0;
    CUSTOMREAL res_abs       = 0.0;
    CUSTOMREAL res_abs_sq    = 0.0;
    CUSTOMREAL res_cs_dif    = 0.0;
    CUSTOMREAL res_cs_dif_sq = 0.0;
    CUSTOMREAL res_cr_dif    = 0.0;
    CUSTOMREAL res_cr_dif_sq = 0.0;
    CUSTOMREAL res_tele      = 0.0;
    CUSTOMREAL res_tele_sq   = 0.0;

    // sum obj_residual from all sources
    std::vector<CUSTOMREAL> obj_residual;

    if (proc_store_srcrec) {    // main of level 2 and 3

        for(auto& data : IP.data_vec){

            if (data.dual_data) continue; // dual data is not used for calculating obj and residual

            // case 1: absolute traveltime for reloc
            if (data.data_type == DATA_TYPE_ABS){     // abs data && we use it
                int id_rec_att = data.id_rec_att;

                CUSTOMREAL travel_time     = data.travel_time;
                CUSTOMREAL travel_time_obs = data.time_observation;

                // assign obj
                if (IP.get_use_abs_reloc()){
                    IP.rec_map[id_rec_att].vobj_src_reloc     += data.weight_reloc * my_square(travel_time - travel_time_obs + IP.rec_map[id_rec_att].tau_opt);
                    obj                                     += data.weight_reloc * my_square(travel_time - travel_time_obs + IP.rec_map[id_rec_att].tau_opt);
                }
                // assign obj
                obj_abs                                     += data.weight_reloc * my_square(travel_time - travel_time_obs + IP.rec_map[id_rec_att].tau_opt);

                // assign residual
                res                                         +=          (travel_time - travel_time_obs + IP.rec_map[id_rec_att].tau_opt);
                res_sq                                      += my_square(travel_time - travel_time_obs + IP.rec_map[id_rec_att].tau_opt);

                res_abs                                     +=          (travel_time - travel_time_obs + IP.rec_map[id_rec_att].tau_opt);
                res_abs_sq                                  += my_square(travel_time - travel_time_obs + IP.rec_map[id_rec_att].tau_opt);

            // case 2: common receiver (swapped source) double difference (double source, or double swapped receiver) for reloc
            } else if (data.data_type == DATA_TYPE_CSDIF ) {  // common receiver data (swapped common source)

                int id_rec1_att = data.id_rec_att;
                int id_rec2_att = data.id_pair_att;

                CUSTOMREAL cs_dif_travel_time     = data.dif_travel_time;
                CUSTOMREAL cs_dif_travel_time_obs = data.time_observation;

                // assign obj (for EQ1, obj+1, for EQ2, obj+1, and for total obj also +1)
                if (IP.get_use_cr_reloc()){
                    IP.rec_map[id_rec1_att].vobj_src_reloc+= 1.0 * data.weight_reloc * my_square(cs_dif_travel_time - cs_dif_travel_time_obs + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                    IP.rec_map[id_rec2_att].vobj_src_reloc+= 1.0 * data.weight_reloc * my_square(cs_dif_travel_time - cs_dif_travel_time_obs + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                    obj                                   += 1.0 * data.weight_reloc * my_square(cs_dif_travel_time - cs_dif_travel_time_obs + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                }

                // assign obj
                obj_cs_dif                              += 1.0 * data.weight_reloc * my_square(cs_dif_travel_time - cs_dif_travel_time_obs + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);

                // assign residual
                res                                     += 1.0 *          (cs_dif_travel_time - cs_dif_travel_time_obs + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                res_sq                                  += 1.0 * my_square(cs_dif_travel_time - cs_dif_travel_time_obs + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);

                res_cs_dif                              += 1.0 *          (cs_dif_travel_time - cs_dif_travel_time_obs + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                res_cs_dif_sq                           += 1.0 * my_square(cs_dif_travel_time - cs_dif_travel_time_obs + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);

                // }

            } else if (data.data_type == DATA_TYPE_CRDIF) {  // we only record the obj of this kind of data
                int id_rec_att = data.id_rec_att;

                // if(IP.rec_map[name_rec].is_stop) continue;

                CUSTOMREAL cr_dif_travel_time     = data.dif_travel_time;
                CUSTOMREAL cr_dif_travel_time_obs = data.time_observation;


                // assign obj
                if (IP.get_use_cs_reloc()){
                    IP.rec_map[id_rec_att].vobj_src_reloc     += 1.0 * data.weight_reloc * my_square(cr_dif_travel_time - cr_dif_travel_time_obs);
                    obj                                       += 1.0 * data.weight_reloc * my_square(cr_dif_travel_time - cr_dif_travel_time_obs);
                }

                // assign obj
                obj_cr_dif                                  += 1.0 * data.weight_reloc * my_square(cr_dif_travel_time - cr_dif_travel_time_obs);

                // assign residual
                res                                         += 1.0 *          (cr_dif_travel_time - cr_dif_travel_time_obs);
                res_sq                                      += 1.0 * my_square(cr_dif_travel_time - cr_dif_travel_time_obs);

                res_cr_dif                                  += 1.0 *          (cr_dif_travel_time - cr_dif_travel_time_obs);
                res_cr_dif_sq                               += 1.0 * my_square(cr_dif_travel_time - cr_dif_travel_time_obs);

            } else {    // unsupported data (swapped common receiver, or others)
                continue;
            }
      
        } // end of loop over all data

        // sum the obj from all sources (swapped receivers)
        IP.allreduce_rec_map_vobj_src_reloc();

    } // end of if (proc_store_srcrec)

    synchronize_all_world();

    broadcast_cr_single(obj,0);
    broadcast_cr_single(obj_abs,0);
    broadcast_cr_single(obj_cs_dif,0);
    broadcast_cr_single(obj_cr_dif,0);
    broadcast_cr_single(obj_tele,0);
    broadcast_cr_single(res,0);
    broadcast_cr_single(res_sq,0);
    broadcast_cr_single(res_abs,0);
    broadcast_cr_single(res_abs_sq,0);
    broadcast_cr_single(res_cs_dif,0);
    broadcast_cr_single(res_cs_dif_sq,0);
    broadcast_cr_single(res_cr_dif,0);
    broadcast_cr_single(res_cr_dif_sq,0);
    broadcast_cr_single(res_tele,0);
    broadcast_cr_single(res_tele_sq,0);

    broadcast_cr_single_sub(obj,0);
    broadcast_cr_single_sub(obj_abs,0);
    broadcast_cr_single_sub(obj_cs_dif,0);
    broadcast_cr_single_sub(obj_cr_dif,0);
    broadcast_cr_single_sub(obj_tele,0);
    broadcast_cr_single_sub(res,0);
    broadcast_cr_single_sub(res_sq,0);
    broadcast_cr_single_sub(res_abs,0);
    broadcast_cr_single_sub(res_abs_sq,0);
    broadcast_cr_single_sub(res_cs_dif,0);
    broadcast_cr_single_sub(res_cs_dif_sq,0);
    broadcast_cr_single_sub(res_cr_dif,0);
    broadcast_cr_single_sub(res_cr_dif_sq,0);
    broadcast_cr_single_sub(res_tele,0);
    broadcast_cr_single_sub(res_tele_sq,0);


    obj_residual = {obj, obj_abs, obj_cs_dif, obj_cr_dif, obj_tele, res, res_sq, res_abs, res_abs_sq, res_cs_dif, res_cs_dif_sq, res_cr_dif, res_cr_dif_sq, res_tele, res_tele_sq};

    for(int i = 0; i < (int)obj_residual.size(); i++){
        allreduce_cr_sim_single_inplace(obj_residual[i]);
    }

    // adjust the step length for each source.
    if (proc_store_srcrec) {
        for (auto iter = IP.rec_map.begin(); iter != IP.rec_map.end(); iter++){
            CUSTOMREAL obj = iter->second.vobj_src_reloc;
            CUSTOMREAL old_obj = iter->second.vobj_src_reloc_old;
            if (i_iter != 0 && old_obj < obj){    // if obj increase, decrease the step length of this (swapped) source
                // std::cout << "before, step_length_max: " << iter->second.step_length_max << "step_length_decay: " << step_length_decay << std::endl;
                iter->second.step_length_max *= step_length_decay_src_reloc;
                // std::cout << "after, step_length_max: " << iter->second.step_length_max << "step_length_decay: " << step_length_decay << std::endl;
            }
            // std::cout << "id_sim: " << id_sim << ", name: " << iter->first << ", obj: " << obj << ", old obj: " << old_obj << ", step_length_max: " << iter->second.step_length_max
            //           << ", step_length_decay: " << step_length_decay
            //           << std::endl;
        }
    }

    return obj_residual;
}


// calculate the gradient of the objective function
void Receiver::calculate_grad_reloc(InputParams& IP, int id_src_att) {

    if(proc_store_srcrec){
        // calculate gradient of travel time at each receiver (swapped source)
        int data_begin = IP.src_map[id_src_att].data_begin;
        int data_end   = IP.src_map[id_src_att].data_end;

        for (int i = data_begin; i < data_end; i++) {
            const DataInfo data = IP.data_vec[i];

            // case 1: absolute traveltime for reloc
            if (data.data_type == DATA_TYPE_ABS && IP.get_use_abs_reloc()){   // abs data && we use it
                int id_rec_att = data.id_rec_att;

                if(IP.rec_map[id_rec_att].is_stop) continue;  // if this receiver (swapped source) is already located

                CUSTOMREAL syn_time       = data.travel_time;
                CUSTOMREAL obs_time       = data.time_observation;

                // local weight
                CUSTOMREAL local_weight = 1.0;

                // evaluate residual_weight_abs_reloc
                CUSTOMREAL  local_residual = abs(syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt);
                CUSTOMREAL* res_weight = IP.get_residual_weight_abs_reloc();

                if      (local_residual < res_weight[0])    local_weight *= res_weight[2];
                else if (local_residual > res_weight[1])    local_weight *= res_weight[3];
                else                                        local_weight *= ((local_residual - res_weight[0])/(res_weight[1] - res_weight[0]) * (res_weight[3] - res_weight[2]) + res_weight[2]);

                // evaluate distance_weight_abs_reloc
                CUSTOMREAL  local_dis    =   0.0;
                Epicentral_distance_sphere(IP.get_rec_point(id_rec_att).lat*DEG2RAD, IP.get_rec_point(id_rec_att).lon*DEG2RAD, IP.get_src_point(id_src_att).lat*DEG2RAD, IP.get_src_point(id_src_att).lon*DEG2RAD, local_dis);
                local_dis *= R_earth;       // rad to km
                CUSTOMREAL* dis_weight = IP.get_distance_weight_abs_reloc();

                if      (local_dis < dis_weight[0])         local_weight *= dis_weight[2];
                else if (local_dis > dis_weight[1])         local_weight *= dis_weight[3];
                else                                        local_weight *= ((local_dis - dis_weight[0])/(dis_weight[1] - dis_weight[0]) * (dis_weight[3] - dis_weight[2]) + dis_weight[2]);

                // assign kernel
                IP.rec_map[id_rec_att].grad_chi_k += (syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt) * data.DTk_pair[0] * data.weight_reloc * local_weight;
                IP.rec_map[id_rec_att].grad_chi_j += (syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt) * data.DTj_pair[0] * data.weight_reloc * local_weight;
                IP.rec_map[id_rec_att].grad_chi_i += (syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt) * data.DTi_pair[0] * data.weight_reloc * local_weight;
                IP.rec_map[id_rec_att].grad_tau   += (syn_time - obs_time + IP.rec_map[id_rec_att].tau_opt)                    * data.weight_reloc * local_weight;

                // count the data
                IP.rec_map[id_rec_att].Ndata      += 1;
            // case 2: common receiver (swapped source) double difference (double source, or double swapped receiver) for reloc
            } else if (data.data_type == DATA_TYPE_CSDIF && IP.get_use_cr_reloc()) {  // common receiver data (swapped common source) and we use it.
                int id_rec1_att = data.id_rec_att;
                int id_rec2_att = data.id_pair_att;

                if(IP.rec_map[id_rec1_att].is_stop && IP.rec_map[id_rec2_att].is_stop) continue;  // if both receivers (swapped sources) are already located

                CUSTOMREAL syn_dif_time       = data.dif_travel_time;
                CUSTOMREAL obs_dif_time       = data.time_observation;

                // assign local weight
                CUSTOMREAL  local_weight = 1.0;

                // evaluate residual_weight_abs
                CUSTOMREAL  local_residual = abs(syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt);
                CUSTOMREAL* res_weight = IP.get_residual_weight_cr_reloc();       // common receiver when not swapped

                if      (local_residual < res_weight[0])    local_weight *= res_weight[2];
                else if (local_residual > res_weight[1])    local_weight *= res_weight[3];
                else                                        local_weight *= ((local_residual - res_weight[0])/(res_weight[1] - res_weight[0]) * (res_weight[3] - res_weight[2]) + res_weight[2]);

                // evaluate distance_weight_abs
                CUSTOMREAL  local_azi1    =   0.0;
                Azimuth_sphere(IP.get_rec_point(id_rec1_att).lat*DEG2RAD, IP.get_rec_point(id_rec1_att).lon*DEG2RAD, IP.get_src_point(id_src_att).lat*DEG2RAD, IP.get_src_point(id_src_att).lon*DEG2RAD, local_azi1);
                CUSTOMREAL  local_azi2    =   0.0;
                Azimuth_sphere(IP.get_rec_point(id_rec2_att).lat*DEG2RAD, IP.get_rec_point(id_rec2_att).lon*DEG2RAD, IP.get_src_point(id_src_att).lat*DEG2RAD, IP.get_src_point(id_src_att).lon*DEG2RAD, local_azi2);
                CUSTOMREAL  local_azi   = abs(local_azi1 - local_azi2)*RAD2DEG;
                if(local_azi > 180.0)   local_azi = 360.0 - local_azi;

                CUSTOMREAL* azi_weight = IP.get_azimuthal_weight_cr_reloc();

                if      (local_azi < azi_weight[0])         local_weight *= azi_weight[2];
                else if (local_azi > azi_weight[1])         local_weight *= azi_weight[3];
                else                                        local_weight *= ((local_azi - azi_weight[0])/(azi_weight[1] - azi_weight[0]) * (azi_weight[3] - azi_weight[2]) + azi_weight[2]);

                // assign kernel
                if(!IP.rec_map[id_rec1_att].is_stop){
                    IP.rec_map[id_rec1_att].grad_chi_k += (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt) * data.DTk_pair[0] * data.weight_reloc * local_weight;
                    IP.rec_map[id_rec1_att].grad_chi_j += (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt) * data.DTj_pair[0] * data.weight_reloc * local_weight;
                    IP.rec_map[id_rec1_att].grad_chi_i += (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt) * data.DTi_pair[0] * data.weight_reloc * local_weight;
                    IP.rec_map[id_rec1_att].grad_tau   += (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt)                    * data.weight_reloc * local_weight;
                    IP.rec_map[id_rec1_att].Ndata      += 1;
                }
                if(!IP.rec_map[id_rec2_att].is_stop){
                    IP.rec_map[id_rec2_att].grad_chi_k -= (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt) * data.DTk_pair[1] * data.weight_reloc * local_weight;
                    IP.rec_map[id_rec2_att].grad_chi_j -= (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt) * data.DTj_pair[1] * data.weight_reloc * local_weight;
                    IP.rec_map[id_rec2_att].grad_chi_i -= (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt) * data.DTi_pair[1] * data.weight_reloc * local_weight;
                    IP.rec_map[id_rec2_att].grad_tau   -= (syn_dif_time - obs_dif_time + IP.rec_map[id_rec1_att].tau_opt - IP.rec_map[id_rec2_att].tau_opt)                    * data.weight_reloc * local_weight;
                    IP.rec_map[id_rec2_att].Ndata      += 1;
                }

            // case 3: common source (swapped receiver) double difference (double receiver, or double swapped source) for reloc
            } else if (data.data_type == DATA_TYPE_CRDIF && IP.get_use_cs_reloc()) {  // common receiver data (swapped common source) and we use it.
                int id_src1_att = data.id_src_att;
                int id_src2_att = data.id_pair_att;
                int id_rec_att  = data.id_rec_att;

                if(IP.rec_map[id_rec_att].is_stop) continue;  // if both receivers (swapped sources) are already located

                CUSTOMREAL syn_dif_time       = data.dif_travel_time;
                CUSTOMREAL obs_dif_time       = data.time_observation;

                // assign local weight
                CUSTOMREAL  local_weight = 1.0;

                // evaluate residual_weight_abs
                CUSTOMREAL  local_residual = abs(syn_dif_time - obs_dif_time);  // common swapped source, so ortime is cancelled.
                CUSTOMREAL* res_weight = IP.get_residual_weight_cs_reloc();       // common receiver when not swapped

                if      (local_residual < res_weight[0])    local_weight *= res_weight[2];
                else if (local_residual > res_weight[1])    local_weight *= res_weight[3];
                else                                        local_weight *= ((local_residual - res_weight[0])/(res_weight[1] - res_weight[0]) * (res_weight[3] - res_weight[2]) + res_weight[2]);

                // evaluate distance_weight_abs
                CUSTOMREAL  local_azi1    =   0.0;
                Azimuth_sphere(IP.get_rec_point(id_rec_att).lat*DEG2RAD, IP.get_rec_point(id_rec_att).lon*DEG2RAD, IP.get_src_point(id_src1_att).lat*DEG2RAD, IP.get_src_point(id_src2_att).lon*DEG2RAD, local_azi1);
                CUSTOMREAL  local_azi2    =   0.0;
                Azimuth_sphere(IP.get_rec_point(id_rec_att).lat*DEG2RAD, IP.get_rec_point(id_rec_att).lon*DEG2RAD, IP.get_src_point(id_src1_att).lat*DEG2RAD, IP.get_src_point(id_src2_att).lon*DEG2RAD, local_azi2);
                CUSTOMREAL  local_azi   = abs(local_azi1 - local_azi2)*RAD2DEG;
                if(local_azi > 180.0)   local_azi = 360.0 - local_azi;

                CUSTOMREAL* azi_weight = IP.get_azimuthal_weight_cs_reloc();

                if      (local_azi < azi_weight[0])         local_weight *= azi_weight[2];
                else if (local_azi > azi_weight[1])         local_weight *= azi_weight[3];
                else                                        local_weight *= ((local_azi - azi_weight[0])/(azi_weight[1] - azi_weight[0]) * (azi_weight[3] - azi_weight[2]) + azi_weight[2]);

                // assign kernel (we only consider the DTijk of first source (swapped receiver), because only the DTijk of the first source is calculated)
                IP.rec_map[id_rec_att].grad_chi_k += (syn_dif_time - obs_dif_time) * data.DTk_pair[0] * data.weight_reloc * local_weight;
                IP.rec_map[id_rec_att].grad_chi_j += (syn_dif_time - obs_dif_time) * data.DTj_pair[0] * data.weight_reloc * local_weight;
                IP.rec_map[id_rec_att].grad_chi_i += (syn_dif_time - obs_dif_time) * data.DTi_pair[0] * data.weight_reloc * local_weight;
                IP.rec_map[id_rec_att].grad_tau   += 0;       // common swapped source, so ortime is cancelled.
                IP.rec_map[id_rec_att].Ndata      += 1;

                // The DTijk of the second source (swapped receiver) will be considered when the loop goes to the other common receiver data with the other source as the key value

            } else {    // unsupported data (swapped common receiver, or others)
                continue;
            }
            
        } // end of loop over all data_map for this src
    } // end of if(proc_store_srcrec)
}

void Receiver::update_source_location(InputParams& IP, Grid& grid) {

    if (proc_store_srcrec) {
        // get list of receivers from input parameters
        for(auto iter = IP.rec_map.begin(); iter != IP.rec_map.end(); iter++){

            int id_rec_att = iter->first;

            if (IP.rec_map[id_rec_att].is_stop){      // do not relocation
                // do nothing
            } else if (IP.rec_map[id_rec_att].Ndata < min_Ndata_reloc) {
                IP.rec_map[id_rec_att].is_stop = true;
            } else {
                // Here grad_dep_km is the kernel of obj with respect to the depth (unit is km) * rescaling_dep. The same for lat,lon,ortime
                CUSTOMREAL grad_dep_km = 0.0;
                if (abs(max_change_dep - abs(IP.rec_map[id_rec_att].change_dep)) < 0.001)
                    grad_dep_km = 0.0;
                else
                    grad_dep_km = - IP.rec_map[id_rec_att].grad_chi_k * rescaling_dep; // over rescaling_dep is rescaling

                CUSTOMREAL grad_lat_km = 0.0;
                if (abs(max_change_lat - abs(IP.rec_map[id_rec_att].change_lat)) < 0.001)
                    grad_lat_km = 0.0;
                else
                    grad_lat_km = IP.rec_map[id_rec_att].grad_chi_j/(R_earth) * rescaling_lat;

                CUSTOMREAL grad_lon_km = 0.0;
                if (abs(max_change_lon - abs(IP.rec_map[id_rec_att].change_lon)) < 0.001)
                    grad_lon_km = 0.0;
                else
                    grad_lon_km = IP.rec_map[id_rec_att].grad_chi_i/(R_earth * cos(IP.rec_map[id_rec_att].lat * DEG2RAD)) * rescaling_lon;

                CUSTOMREAL grad_ortime = 0.0;
                if (abs(max_change_ortime - abs(IP.rec_map[id_rec_att].change_tau)) < 0.001)
                    grad_ortime = 0.0;
                else
                    grad_ortime = IP.rec_map[id_rec_att].grad_tau * rescaling_ortime;

                CUSTOMREAL norm_grad;
                norm_grad = std::sqrt(my_square(grad_dep_km) + my_square(grad_lat_km) + my_square(grad_lon_km) + my_square(grad_ortime));

                // if norm is smaller than a threshold, stop update
                if (norm_grad < TOL_SRC_RELOC){
                    IP.rec_map[id_rec_att].is_stop = true;
                    continue;
                }

                CUSTOMREAL step_length;
                step_length = 0.5 * IP.rec_map[id_rec_att].vobj_src_reloc/my_square(norm_grad);

                // rescale update value for perturbation of dep/rescaling_dep, lat/rescaling_lat, lon/rescaling_lon, time/rescaling_ortime
                CUSTOMREAL update_dep       = step_length * grad_dep_km;
                CUSTOMREAL update_lat       = step_length * grad_lat_km;
                CUSTOMREAL update_lon       = step_length * grad_lon_km;
                CUSTOMREAL update_ortime    = step_length * grad_ortime;

                CUSTOMREAL update_max = -1.0;
                CUSTOMREAL downscale = 1.0;
                update_max = std::max(update_max,abs(update_dep));
                update_max = std::max(update_max,abs(update_lat));
                update_max = std::max(update_max,abs(update_lon));
                update_max = std::max(update_max,abs(update_ortime));
                if (update_max > IP.rec_map[id_rec_att].step_length_max){     // make sure update_max * downscale = min(step_length_max, update_max)
                    downscale = IP.rec_map[id_rec_att].step_length_max / update_max;
                }

                // update value for dep (km), lat (km), lon (km), ortime (s)
                CUSTOMREAL update_dep_km   = - update_dep    * downscale * rescaling_dep;
                CUSTOMREAL update_lat_km   = - update_lat    * downscale * rescaling_lat; // /  R_earth * RAD2DEG;
                CUSTOMREAL update_lon_km   = - update_lon    * downscale * rescaling_lon; // / (R_earth * cos(IP.rec_map[id_rec_att].lat * DEG2RAD)) * RAD2DEG;
                CUSTOMREAL update_ortime_s = - update_ortime * downscale * rescaling_ortime;


                // limit the update for dep, lat, lon
                if (abs(IP.rec_map[id_rec_att].change_dep + update_dep_km) > max_change_dep){
                    if (IP.rec_map[id_rec_att].change_dep + update_dep_km > 0)
                        update_dep_km =   max_change_dep - IP.rec_map[id_rec_att].change_dep;
                    else
                        update_dep_km = - max_change_dep - IP.rec_map[id_rec_att].change_dep;

                }
                if (abs(IP.rec_map[id_rec_att].change_lat + update_lat_km) > max_change_lat){
                    if (IP.rec_map[id_rec_att].change_lat + update_lat_km > 0)
                        update_lat_km =   max_change_lat - IP.rec_map[id_rec_att].change_lat;
                    else
                        update_lat_km = - max_change_lat - IP.rec_map[id_rec_att].change_lat;
                }
                if (abs(IP.rec_map[id_rec_att].change_lon + update_lon_km) > max_change_lon){
                    if (IP.rec_map[id_rec_att].change_lon + update_lon_km > 0)
                        update_lon_km =   max_change_lon - IP.rec_map[id_rec_att].change_lon;
                    else
                        update_lon_km = - max_change_lon - IP.rec_map[id_rec_att].change_lon;
                }
                if (abs(IP.rec_map[id_rec_att].change_tau + update_ortime_s) > max_change_ortime){
                    if (IP.rec_map[id_rec_att].change_tau + update_ortime_s > 0)
                        update_ortime_s =   max_change_ortime - IP.rec_map[id_rec_att].change_tau;
                    else
                        update_ortime_s = - max_change_ortime - IP.rec_map[id_rec_att].change_tau;
                }
                // remark:  in the case of  local search for ortime, change of ortime need to be checked as above
                //          in the case of global search for ortime, update_ortime is always zero (because grad_ortime = 0), thus always satisfied

                // earthquake should be below the surface (0 km)
                if (IP.rec_map[id_rec_att].dep + update_dep_km < 0){
                    // update_dep_km = - IP.rec_map[id_rec_att].dep;
                    update_dep_km = - 2.0 * IP.rec_map[id_rec_att].dep - update_dep_km;
                }


                // update value for dep (km), lat (degree), lon (degree)
                IP.rec_map[id_rec_att].vobj_grad_norm_src_reloc = norm_grad;

                IP.rec_map[id_rec_att].dep        += update_dep_km;
                IP.rec_map[id_rec_att].lat        += update_lat_km / R_earth * RAD2DEG;
                IP.rec_map[id_rec_att].lon        += update_lon_km /(R_earth * cos(IP.rec_map[id_rec_att].lat * DEG2RAD)) * RAD2DEG;
                IP.rec_map[id_rec_att].tau_opt    += update_ortime_s;

                IP.rec_map[id_rec_att].change_dep += update_dep_km;
                IP.rec_map[id_rec_att].change_lat += update_lat_km;
                IP.rec_map[id_rec_att].change_lon += update_lon_km;
                IP.rec_map[id_rec_att].change_tau += update_ortime_s;


                // if (IP.rec_map[id_rec_att].step_length_max < TOL_step_length){
                //     IP.rec_map[id_rec_att].is_stop = true;
                // }

                


                // detect nan and inf then exit the program
                if (std::isnan(IP.rec_map[id_rec_att].dep) || std::isinf(IP.rec_map[id_rec_att].dep) ||
                    std::isnan(IP.rec_map[id_rec_att].lat) || std::isinf(IP.rec_map[id_rec_att].lat) ||
                    std::isnan(IP.rec_map[id_rec_att].lon) || std::isinf(IP.rec_map[id_rec_att].lon)){
                    std::cout << "Error: nan or inf detected in source relocation!" << std::endl;
                    std::cout << "id_sim: " << id_sim
                            << ", src name: " << IP.rec_map[id_rec_att].name
                            << ", obj: " << IP.rec_map[id_rec_att].vobj_src_reloc
                            << ", lat: " << IP.rec_map[id_rec_att].lat
                            << ", lon: " << IP.rec_map[id_rec_att].lon
                            << ", dep: " << IP.rec_map[id_rec_att].dep
                            << ", ortime: " << IP.rec_map[id_rec_att].tau_opt
                            << ", is_stop: " << IP.rec_map[id_rec_att].is_stop
                            << ", grad_dep_km(pert): " << grad_dep_km
                            << ", grad_lat_km(pert): " << grad_lat_km
                            << ", grad_lon_km(pert): " << grad_lon_km
                            << ", vobj_src_reloc: " << IP.rec_map[id_rec_att].vobj_src_reloc
                            << std::endl;

                    exit(1);
                }

                // check if the new receiver position is within the domain
                // if not then set the receiver position to the closest point on the domain

                // gap to boundary
                CUSTOMREAL boundary_gap_lon = _0_5_CR * grid.get_delta_lon() * RAD2DEG;
                CUSTOMREAL boundary_gap_lat = _0_5_CR * grid.get_delta_lat() * RAD2DEG;
                CUSTOMREAL boundary_gap_r   = _0_5_CR * grid.get_delta_r();

                if (IP.rec_map[id_rec_att].lon < IP.get_min_lon()*RAD2DEG){
                    IP.rec_map[id_rec_att].lon = IP.get_min_lon()*RAD2DEG + boundary_gap_lon;
                    // report to user
                    std::cout << "Warning: source/receiver " << IP.rec_map[id_rec_att].name << " is out of domain in longitude, set the location near min_lon boundary: " << IP.get_min_lon()*RAD2DEG + boundary_gap_lon << std::endl;
                }
                if (IP.rec_map[id_rec_att].lon > IP.get_max_lon()*RAD2DEG){
                    IP.rec_map[id_rec_att].lon = IP.get_max_lon()*RAD2DEG - boundary_gap_lon;
                    // report to user
                    std::cout << "Warning: source/receiver " << IP.rec_map[id_rec_att].name << " is out of domain in longitude, set the location near max_lon boundary: " << IP.get_max_lon()*RAD2DEG - boundary_gap_lon << std::endl;
                }
                if (IP.rec_map[id_rec_att].lat < IP.get_min_lat()*RAD2DEG){
                    IP.rec_map[id_rec_att].lat = IP.get_min_lat()*RAD2DEG + boundary_gap_lat;
                    // report to user
                    std::cout << "Warning: source/receiver " << IP.rec_map[id_rec_att].name << " is out of domain in latitude, set the location near min_lat boundary: " << IP.get_min_lat()*RAD2DEG + boundary_gap_lat << std::endl;
                }
                if (IP.rec_map[id_rec_att].lat > IP.get_max_lat()*RAD2DEG){
                    IP.rec_map[id_rec_att].lat = IP.get_max_lat()*RAD2DEG - boundary_gap_lat;
                    // report to user
                    std::cout << "Warning: source/receiver " << IP.rec_map[id_rec_att].name << " is out of domain in latitude, set the location near max_lat boundary: " << IP.get_max_lat()*RAD2DEG - boundary_gap_lat << std::endl;
                }
                if (IP.rec_map[id_rec_att].dep < IP.get_min_dep()){
                    IP.rec_map[id_rec_att].dep = IP.get_min_dep() + boundary_gap_r;
                    // report to user
                    std::cout << "Warning: source/receiver " << IP.rec_map[id_rec_att].name << " is out of domain in depth, set the location near min_dep boundary: " << IP.get_min_dep() + boundary_gap_r << std::endl;
                }
                if (IP.rec_map[id_rec_att].dep > IP.get_max_dep()){
                    IP.rec_map[id_rec_att].dep = IP.get_max_dep() - boundary_gap_r;
                    // report to user
                    std::cout << "Warning: source/receiver " << IP.rec_map[id_rec_att].name << " is out of domain in depth, set the location near max_dep boundary: " << IP.get_max_dep() - boundary_gap_r << std::endl;
                }   
            }

            // share the flag of stop within the same simultanoue run group
            //allreduce_bool_single_inplace(IP.rec_map[name_rec].is_stop); // this is done in src_rec->broadcast_rec_info_intra_sim

        } // end iter loopvobj_grad_norm_src_reloc
    } // end if(proc_store_srcrec)

    //IP.allreduce_rec_map_vobj_grad_norm_src_reloc();

}
