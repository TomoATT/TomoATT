#include <iostream>
#include <vector>
#include <map>

#include "utils.h"
#include "input_params.h"
#include "src_rec.h"
#include "mpi_funcs.h"

#ifdef USE_OMP
#include <omp.h>
#endif

/*
    caluclate geographical weight for sources and receivers.
    At first, we calculate inversed weight for each source from summation of epicentral distances to all sources.
    Then, we calculate inversed weight for each receiver from summation of epicentral distances to all receivers.

    ref: Youyi Ruan(2019) https://academic.oup.com/gji/article/219/2/1225/5542709
*/

void calc_dist_min_max(std::map<std::string,SrcRecInfo>& src_or_rec, \
                       std::vector<std::string>& id2name,\
                       CUSTOMREAL& d_min, CUSTOMREAL& d_max){
    // calculate the min and max epicentral distance of all src_or_rec

    // check the min and max epicentral distance of all src_or_rec
    CUSTOMREAL tmp_min = 1.0e+10;
    CUSTOMREAL tmp_max = 0.0;

    int n_elm = src_or_rec.size();

    #pragma omp parallel for default(none) shared(src_or_rec, id2name, n_elm) reduction(min:tmp_min) reduction(max:tmp_max)
    for (int i = 0; i < n_elm-1; i++) {
        for (int j = i+1; j < n_elm; j++) {
            CUSTOMREAL d_ij = 0.0;

            SrcRecInfo& sr_i = src_or_rec[id2name[i]];
            SrcRecInfo& sr_j = src_or_rec[id2name[j]];

            Epicentral_distance_sphere(sr_i.lat*DEG2RAD, \
                                       sr_i.lon*DEG2RAD, \
                                       sr_j.lat*DEG2RAD, \
                                       sr_j.lon*DEG2RAD, \
                                       d_ij);

            if (d_ij < tmp_min) tmp_min = d_ij;
            if (d_ij > tmp_max) tmp_max = d_ij;
        }
    }

    // set result
    d_min = tmp_min;
    d_max = tmp_max;

    // print the min and max epicentral distance
    //std::cout << "min epicentral distance = " << d_min << std::endl;
    //std::cout << "max epicentral distance = " << d_max << std::endl;

}

void init_weight(std::map<std::string, SrcRecInfo>& src_or_rec, \
                 std::vector<std::string>& id2name){
    // initialize all the source and receiver weights to be zero
    #pragma omp parallel for default(none) shared(src_or_rec, id2name)
    for (int i = 0; i < (int)id2name.size(); i++) {
        src_or_rec[id2name[i]].sum_weight = 0.0;
    }
}


CUSTOMREAL _calc_weight(std::map<std::string, SrcRecInfo>& src_or_rec, \
                        std::vector<std::string>& id2name, \
                        CUSTOMREAL& d_zero){

    int n_elm = id2name.size();

    // parallelize loop over sources and receivers
    // sharing src_or_rec.weight
    #pragma omp parallel for default(none) shared(src_or_rec, id2name, n_elm, d_zero, std::cout)
    for (int i = 0; i < n_elm-1; i++) {
        for (int j = i+1; j < n_elm; j++) {
            CUSTOMREAL d_ij = 0.0;

            SrcRecInfo& sr_i = src_or_rec[id2name[i]];
            SrcRecInfo& sr_j = src_or_rec[id2name[j]];

            Epicentral_distance_sphere(sr_i.lat*DEG2RAD, \
                                       sr_i.lon*DEG2RAD, \
                                       sr_j.lat*DEG2RAD, \
                                       sr_j.lon*DEG2RAD, \
                                       d_ij);

            d_ij *= RAD2DEG;

            CUSTOMREAL w_inv_tmp = std::exp(-std::pow(d_ij/d_zero, 2.0));
            //CUSTOMREAL w_inv_tmp = std::pow(-(d_ij/d_zero)*(d_ij/d_zero),4.0);

            // check if w_inv_tmp is nan or inf
            if (std::isnan(w_inv_tmp) || std::isinf(w_inv_tmp)) {
                std::cout << "w_inv_tmp is nan or inf " << std::endl;
                std::cout << "d_ij = " << d_ij << std::endl;
                std::cout << "d_zero = " << d_zero << std::endl;
                std::cout << "i = " << i << std::endl;
                std::cout << "j = " << j << std::endl;
                std::cout << "name_i = " << id2name[i] << std::endl;
                std::cout << "name_j = " << id2name[j] << std::endl;
                std::cout << "sr_i.id = " << sr_i.id << std::endl;
                std::cout << "sr_j.id = " << sr_j.id << std::endl;
                std::cout << "sr_i.lat = " << sr_i.lat << std::endl;
                std::cout << "sr_i.lon = " << sr_i.lon << std::endl;
                std::cout << "sr_j.lat = " << sr_j.lat << std::endl;
                std::cout << "sr_j.lon = " << sr_j.lon << std::endl;
                std::cout << "w_inv_tmp = " << w_inv_tmp << std::endl;
                exit(1);
            }

            sr_i.sum_weight += w_inv_tmp;
            sr_j.sum_weight += w_inv_tmp;
        }
    }


    // here the weight is inversed weight, so we need to invert it
    #pragma omp parallel for default(none) shared(src_or_rec, id2name, n_elm, d_zero, std::cout)
    for (int i = 0; i < n_elm; i++) {
        CUSTOMREAL w_tmp = src_or_rec[id2name[i]].sum_weight;
        // check if w_tmp is nan or inf
        if (std::isnan(w_tmp) || std::isinf(w_tmp)){
            std::cout << "w_tmp is nan or inf" << std::endl;
            std::cout << "i = " << i << std::endl;
            std::cout << "name_i = " << id2name[i] << std::endl;
            std::cout << "w_tmp = " << w_tmp << std::endl;
            exit(1);
        }


        src_or_rec[id2name[i]].sum_weight = 1.0/src_or_rec[id2name[i]].sum_weight;
    }

    // calculate condition number (min/max of weight)
    CUSTOMREAL w_min = 1.0e+10;
    CUSTOMREAL w_max = 0.0;

    // parallelize loop for finding min and max with reduction
    #pragma omp parallel for default(none) shared(src_or_rec, id2name, n_elm) reduction(min:w_min) reduction(max:w_max)
    for (int i = 0; i < n_elm; i++) {
        if (src_or_rec[id2name[i]].sum_weight < w_min) w_min = src_or_rec[id2name[i]].sum_weight;
        if (src_or_rec[id2name[i]].sum_weight > w_max) w_max = src_or_rec[id2name[i]].sum_weight;
    }

    std::cout << "weight min = " << w_min << std::endl;
    std::cout << "weight max = " << w_max << std::endl;
    std::cout << "condition number = " << w_max/w_min << std::endl;

    return w_max/w_min;

}


int find_good_ncond(std::vector<CUSTOMREAL>& condition_numbers){
    // select the best condition number

    int n_try = condition_numbers.size();

    CUSTOMREAL ncond_min = 1.0e+10;
    CUSTOMREAL ncond_max = 0.0;

    for (int i = 0; i < n_try; i++) {
        // ignore abnormal value
        if (std::isnan(condition_numbers[i])) continue;
        if (std::isinf(condition_numbers[i])) continue;

        if (condition_numbers[i] < ncond_min) {
            ncond_min = condition_numbers[i];
        }
        if (condition_numbers[i] > ncond_max) {
            ncond_max = condition_numbers[i];
        }
    }

    // find the first one-third point between min and max condition number
    CUSTOMREAL ncond_1_3 = ncond_min + (ncond_max - ncond_min) / 3.0;
    int ncond_1_3_idx = 0;

    for (int i = 0; i < n_try; i++) {
        // ignore abnormal value
        if (std::isnan(condition_numbers[i])) continue;
        if (std::isinf(condition_numbers[i])) continue;

        if (condition_numbers[i] > ncond_1_3) {
            ncond_1_3_idx = i;
            break;
        }
    }

    return ncond_1_3_idx;
}


void normalize_weight(std::map<std::string, SrcRecInfo>& src_or_rec, std::vector<std::string>& id2name){

    // scale weight values to be \sum_{i}^{S} w_i = S

    int n_elm = src_or_rec.size();
    CUSTOMREAL w_sum = 0.0;
    #pragma omp parallel for default(none) reduction(+:w_sum) shared(src_or_rec, id2name, n_elm)
    for(int i = 0; i < n_elm; i++){
        w_sum += src_or_rec[id2name[i]].sum_weight;
    }

    #pragma omp parallel for default(none) shared(src_or_rec, id2name, n_elm, w_sum)
    for (int i=0; i<n_elm; i++){
        src_or_rec[id2name[i]].sum_weight = (src_or_rec[id2name[i]].sum_weight / w_sum) * n_elm;
    }

}


void normalize_weight_data(std::map<std::string, std::map<std::string, std::vector<DataInfo>>>& data_map, \
                           std::map<std::string, SrcRecInfo>& src_map, \
                           std::map<std::string, SrcRecInfo>& rec_map, \
                           std::vector<std::string>&          src_id2name, \
                           std::vector<std::string>&          rec_id2name){

    // normalize weight in each event (sum of weight*n_data becomes 1.0*number of data tracks in the event)
    int n_src = src_id2name.size();
    int n_rec = rec_id2name.size();

//
// this way cancel the geographical weight on the receiver side
//
    #pragma omp parallel for default(none) shared(data_map, src_map, rec_map, src_id2name, rec_id2name, n_src, n_rec)
    for (int i = 0; i < n_src; i++) {
        CUSTOMREAL w_sum_tmp = 0.0; // sum of receiver weight
        for (int j = 0; j < n_rec; j++) {
            int v_data_size = data_map[src_id2name[i]][rec_id2name[j]].size();
            if (v_data_size > 0) {
                CUSTOMREAL& weight_rec = rec_map[rec_id2name[j]].sum_weight;
                w_sum_tmp += weight_rec*v_data_size;
                //n_data    +=v_data_size;
            }
        }

        // normalize for receivers
        for (int j = 0; j < n_rec; j++) {
            for(auto &data : data_map[src_id2name[i]][rec_id2name[j]]){
                data.data_weight = src_map[src_id2name[i]].sum_weight*rec_map[rec_id2name[j]].sum_weight/w_sum_tmp;
            }
        }
   }

//    #pragma omp parallel for default(none) shared(data_map, src_map, rec_map, src_id2name, rec_id2name, n_src, n_rec)
//    for (int i = 0; i < n_src; i++) {
//        for (int j = 0; j < n_rec; j++) {
//            int v_data_size = data_map[src_id2name[i]][rec_id2name[j]].size();
//            for (int k = 0; k < v_data_size; k++) {
//                data_map[src_id2name[i]][rec_id2name[j]][k].data_weight = src_map[src_id2name[i]].sum_weight*rec_map[rec_id2name[j]].sum_weight;
//            }
//        }
//    }
}




void calc_weight(std::map<std::string, SrcRecInfo>& src_or_rec, \
                 std::vector<std::string>&          id2name){

    CUSTOMREAL d_zero_fin;

    if (ref_value < 0) {

        // calculate the weights of sources or receivers
        int n_try = 30; // number of tries

        // get the min and max epicentral distance of all src_or_rec
        CUSTOMREAL d_min, d_max;
    //    calc_dist_min_max(src_or_rec, id2name, d_min, d_max);

        // convert rad to deg
        //d_min *= RAD2DEG;
        //d_max *= RAD2DEG;

        d_max = 5.0; // in deg
        d_min = 0.01; // shouldn't be 0.0
    //
        std::cout << "ref distance min = " << d_min << std::endl;
        std::cout << "ref distance max = " << d_max << std::endl;

        std::vector<CUSTOMREAL> condition_numbers; // store the condition numbers
        std::vector<CUSTOMREAL> test_d_zero; // store the test d_zero

        // prepare the test d_zero (d_min ~ d_max by n_try times)
        CUSTOMREAL d_zero_step = (d_max - d_min) / (n_try - 1);
        CUSTOMREAL d_zero_tmp = d_min; // void d_zero = 0.0
        for (int i = 0; i < n_try; i++) {
            test_d_zero.push_back(d_zero_tmp);
            d_zero_tmp += d_zero_step;
        }

        // calculate the weights iteratively
        int c = 0;
        for (auto& d_zero_try : test_d_zero) {
            // initialize all the source or receiver weights to be zero
            init_weight(src_or_rec, id2name);
            CUSTOMREAL ncond = _calc_weight(src_or_rec, id2name, d_zero_try);

            // store the condition number
            condition_numbers.push_back(ncond);

            // output d_zero and condition number
            std::cout << "i-iter/total: " << c << "/" << n_try << ", d_zero: " << d_zero_try << ", condition number: " << ncond << std::endl;

            c++;
        }

        // find the one-third point between min and max condition number
        d_zero_fin = test_d_zero[find_good_ncond(condition_numbers)];

    } else {
        // use largest distance as d_zero_fin
        d_zero_fin = ref_value;
    }

    // calculate the weights with the final d_zero
    init_weight(src_or_rec, id2name);
    _calc_weight(src_or_rec, id2name, d_zero_fin);

}

void calculate_src_rec_weight(std::map<std::string, SrcRecInfo>                                   &src_map, \
                              std::map<std::string, SrcRecInfo>                                   &rec_map, \
                              std::map<std::string, std::map<std::string, std::vector<DataInfo>>> &data_map, \
                              std::vector<std::string>                                            &src_id2name, \
                              std::vector<std::string>                                            &rec_id2name) {


    // calculate source weights
    std::cout << "calculating source weights..." << std::endl;
    calc_weight(src_map, src_id2name);
    // normalize the source weight
    normalize_weight(src_map, src_id2name);

    // calculate receiver weights
    std::cout << "calculating receiver weights..." << std::endl;
    calc_weight(rec_map, rec_id2name);

    // normalize the receiver weight
    // (the receiver weight is normalized for each source)
    normalize_weight_data(data_map, src_map, rec_map, src_id2name, rec_id2name);

    // end
}


void write_src_rec_file_with_weight(std::string src_rec_file_out, \
                                    std::map<std::string, SrcRecInfo> &src_map, \
                                    std::map<std::string, SrcRecInfo> &rec_map, \
                                    std::map<std::string, std::map<std::string, std::vector<DataInfo>>> &data_map, \
                                    std::vector<std::string>          &src_id2name, \
                                    std::vector<std::string>          &rec_id2name){

    std::ofstream ofs;

    // open file
    ofs.open(src_rec_file_out);

    for (int i_src = 0; i_src < (int)src_id2name.size(); i_src++){

        std::string name_src = src_id2name[i_src];
        SrcRecInfo  src      = src_map[name_src];

        // format should be the same as input src_rec_file
        // source line :  id_src year month day hour min sec lat lon dep_km mag num_recs id_event
        ofs << std::setw(7) << std::right << std::setfill(' ') <<  src.id << " "
            << src.year << " " << src.month << " " << src.day << " "
            << src.hour << " " << src.min   << " "
            << std::fixed << std::setprecision(2) << std::setw(5) << std::right << std::setfill(' ') << src.sec << " "
            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.lat << " "
            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.lon << " "
            << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << src.dep << " "
            << std::fixed << std::setprecision(2) << std::setw(5) << std::right << std::setfill(' ') << src.mag << " "
            << std::setw(5) << std::right << std::setfill(' ') << src.n_data << " "
            << src.name << " "
            << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << 1.0 //src.sum_weight
            << std::endl;

        // data line
        int i_rec = 0;
        for (auto iter = data_map[name_src].begin(); iter != data_map[name_src].end(); iter++){

            const std::string name_rec = iter->first;
            std::vector<DataInfo> v_data = data_map[name_src][name_rec];

            for (const auto& data : v_data){

                // absolute traveltime data
                if (data.data_type == DATA_TYPE_ABS) {
                    SrcRecInfo  rec      = rec_map[name_rec];

                    CUSTOMREAL  travel_time = data.time_observation; // write the observed travel time

                    // receiver line : id_src id_rec name_rec lat lon elevation_m phase epicentral_distance_km arival_time
                    ofs << std::setw(7) << std::right << std::setfill(' ') << src.id << " "
                        << std::setw(5) << std::right << std::setfill(' ') << i_rec << " "
                        << rec.name << " "
                        << std::fixed << std::setprecision(4) << std::setw(9) << rec.lat << " "
                        << std::fixed << std::setprecision(4) << std::setw(9) << rec.lon << " "
                        << std::fixed << std::setprecision(4) << std::setw(9) << -1.0*rec.dep*1000.0 << " "
                        << data.phase << " "
                        << std::fixed << std::setprecision(4) << std::setw(9) << std::right << std::setfill(' ') << travel_time << " "
                        << std::fixed << std::setprecision(4) << std::setw(6) << std::right << std::setfill(' ') << data.data_weight
                        << std::endl;

                        i_rec++;
                }else {
                    std::cout << "Error: data type is not defined." << std::endl;
                    exit(1);
                }

            } // end of for (const auto& data : v_data)
        } // end of for (auto iter = data_map_back[name_src].begin(); iter != data_map_back[name_src].end(); iter++)
    } // end of for (int i_src = 0; i_src < (int)src_name_list.size(); i_src++)

    // close file
    ofs.close();


}

//
// function for calculating the source and receiver weight
// MNMN: receiver pair data is not supported yet.
int main(int argc, char *argv[])
{
    // parse options
    parse_options_srcrec_weight(argc, argv);

    // init mpi
    initialize_mpi();

    // exit if number of processes is not 1
    if (nprocs > 1) {
        std::cout << "number of processes should be 1." << std::endl;
        exit(1);
    }

#ifdef USE_OMP
    // srcrec weight calculation uses opnemp parallelization
    // set 12 threads for openmp
    omp_set_num_threads(12);

    // n threads
    int n_threads = omp_get_max_threads();
#endif

    stdout_by_main("------------------------------------------------------");
    stdout_by_main("start Src Rec weight calculation.");
    std::cout << "number of openmp threads: " << n_threads << std::endl;
    stdout_by_main("------------------------------------------------------");

    std::map<std::string, SrcRecInfo>                                  src_map;
    std::map<std::string, SrcRecInfo>                                  rec_map;
    std::map<std::string, std::map<std::string,std::vector<DataInfo>>> data_map;
    std::vector<std::string> src_id2name, rec_id2name;
    std::vector<std::vector<std::vector<std::string>>> rec_id2name_back_dummy;

    std::cout << "parsing src_rec file: " << input_file << std::endl;

    // read src_rec file
    parse_src_rec_file(input_file, \
                       src_map, \
                       rec_map, \
                       data_map, \
                       src_id2name, \
                       rec_id2name_back_dummy);

    // create rec_id2name
    for (auto& rec: rec_map) {
        rec_id2name.push_back(rec.first);
    }

    std::cout << "calculating source and receiver weight..." << std::endl;

    // calculate source and receiver weight
    calculate_src_rec_weight(src_map, rec_map, data_map, src_id2name, rec_id2name);

    std::cout << "writing src_rec file with weight: " << output_file_weight << std::endl;

    // write src_rec file with calculated weights
    write_src_rec_file_with_weight(output_file_weight, \
                                   src_map, \
                                   rec_map, \
                                   data_map, \
                                   src_id2name, \
                                   rec_id2name);

    stdout_by_main("------------------------------------------------------");
    stdout_by_main("end Src Rec weight calculation.");
    stdout_by_main(("output file: " + output_file_weight).c_str());
    stdout_by_main("------------------------------------------------------");
    return 0;

}