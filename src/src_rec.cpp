#include "src_rec.h"


//
// functions for processing src_rec_file
//

void parse_src_rec_file(std::string& src_rec_file,
                        std::map<int, SrcRecInfo>& src_map,
                        std::map<int, SrcRecInfo>& rec_map,
                        std::vector<DataInfo>& data_vec,
                        std::vector<int>& src_id_in_file,
                        std::vector<std::vector<std::vector<int>>>& rec_id_in_file){
                        
    // start timer
    std::string timer_name = "parse_src_rec_file";
    Timer timer(timer_name);

    std::ifstream ifs;          // dummy for all processes except world_rank 0
    std::stringstream ss_whole; // for parsing the whole file
    std::stringstream ss;       // for parsing each line

    // only world_rank 0 reads the file
    //if (sim_rank == 0){
        ifs.open(src_rec_file);
        // abort if file does not exist
        if (!ifs.is_open()){
            std::cerr << "Error: src_rec_file " << src_rec_file << " does not exist!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ss_whole << ifs.rdbuf();
        ifs.close();
    //}

    std::string line;
    int cc = 0;        // count the number of lines
    int i_src_now = 0; // count the number of srcs
    int ndata_tmp = 0; // count the number of receivers or differential traveltime data for each source
    src_map.clear();            // src name -> SrcRecInfo
    rec_map.clear();            // rec name -> SrcRecInfo
    data_vec.clear();           // data vector
    src_id_in_file.clear();     // store the order of sources in the file
    rec_id_in_file.clear();     // store the order of receivers for each source in the file


    std::map<std::string, int> src_name2id_att; // map from src name to id_att, used for checking duplicated source name in the file
    std::map<std::string, int> rec_name2id_att;  


    std::string src_name;
    CUSTOMREAL src_weight = 1.0;
    CUSTOMREAL rec_weight = 1.0;
    int src_id = -1;    // src id in the file
    int src_id_att = -1; // the index of sources in TomoATT, used to identify this source. Each source has a unique id_src_att. --- IGNORE ---

    // temporary receiver name list for each source
    // this stores station id and the data type (DATA_TYPE_ABS, DATA_TYPE_CSDIF or DATA_TYPE_CRDIF) for each data line.
    std::vector<std::vector<int>> rec_id_list;

    while (true) {

        bool end_of_file = false;
        bool skip_this_line = false;

        line.clear(); // clear the line before use

        // read a line
        //if (sim_rank == 0){
            if (!std::getline(ss_whole, line))
                end_of_file = true;
        //}

        // broadcast end_of_file
        //broadcast_bool_single(end_of_file, 0);

        if (end_of_file)
            break;

        // skip comment and empty lines
        if (sim_rank == 0){
            if (line[0] == '#' || line.empty())
                skip_this_line = true;
        }

        //broadcast_bool_single(skip_this_line, 0);

        if (skip_this_line)
            continue;

        // parse the line
        //int ntokens = 0;
        std::string token;
        std::vector<std::string> tokens;

        //if (sim_rank==0){
            // erase the trailing space
            line.erase(line.find_last_not_of(" \n\r\t")+1);

            // parse the line with arbitrary number of spaces
            ss.clear(); // clear the stringstream before use
            ss << line;

            while (std::getline(ss, token, ' ')) {
                if (token.size() > 0) // skip the first spaces and multiple spaces
                    tokens.push_back(token);
            }

            // length of tokens
            //ntokens = tokens.size();
        //}

        // broadcast ntokens
        //broadcast_i_single(ntokens, 0);
        // broadcast tokens
        //for (int i=0; i<ntokens; i++){
        //    if (sim_rank == 0)
        //        token = tokens[i];
        //    broadcast_str(token, 0);
        //    if (sim_rank != 0)
        //        tokens.push_back(token);
        //}

        try { // check failure of parsing line by line

            // store values into structure
            if (cc == 0){ // read source info
                SrcRecInfo src;
                src.id     = std::stoi(tokens[0]);
                src.year   = std::stoi(tokens[1]);
                src.month  = std::stoi(tokens[2]);
                src.day    = std::stoi(tokens[3]);
                src.hour   = std::stoi(tokens[4]);
                src.min    = std::stoi(tokens[5]);
                src.sec    = static_cast<CUSTOMREAL>(std::stod(tokens[6]));
                src.lat    = static_cast<CUSTOMREAL>(std::stod(tokens[7])); // in degree
                src.lon    = static_cast<CUSTOMREAL>(std::stod(tokens[8])); // in degree
                src.dep    = static_cast<CUSTOMREAL>(std::stod(tokens[9])); // source in km
                src.mag    = static_cast<CUSTOMREAL>(std::stod(tokens[10]));
                src.n_data = std::stoi(tokens[11]);
                src.name   = tokens[12];

                if (src_name2id_att.find(src.name) == src_name2id_att.end()){   // if this is a new source, assign id.
                    src.id_att = 1+static_cast<int>(src_name2id_att.size()); // positive id for source
                    src_name2id_att[src.name] = src.id_att; // assign id_att based on the current size of src_name2id_att,
                } else {
                    src.id_att = src_name2id_att[src.name]; // if the source already exists, use the existing id_att
                }

                // whether the src.name exists or not, overwrite it. (it can overwrite the src_info in the cr_dif data, whose source infomation (e.g., ortime, Ndata) is incomplete.)
                src_map[src.id_att] = src; // store the source info in src_map with id_att as key

                cc++;

                // check if tokens[13] exists, then read weight
                if (tokens.size() > 13)
                    src_weight = static_cast<CUSTOMREAL>(std::stod(tokens[13]));
                else
                    src_weight = 1.0; // default weight

                // new source detected by its name
                // TODO: add error check for duplicated source name (but different event info)
                // if (src_map.find(src.name) == src_map.end())
                //     src_map[src.name] = src;

                // src_map[src.name] = src;

                src_id      = src.id;
                src_id_att  = src.id_att;
                src_name = src.name;

                ndata_tmp = src.n_data;
                src_id_in_file.push_back(src_id); // store order of sources in the file

                // source with no receiver is allowed (cc = 1, ndata_tmp = 0)
                if (cc > ndata_tmp) {
                    // go to the next source
                    cc = 0;
                    i_src_now++;

                    // store the receiver name list for the source
                    rec_id_in_file.push_back(rec_id_list);
                    // clear the temporary receiver name list
                    rec_id_list.clear();

                    // timer
                    if (i_src_now % 1000 == 0 && world_rank == 0) {
                        std::cout << "reading source " << i_src_now << " finished in " << timer.get_t() << " seconds. dt = " << timer.get_t_delta() << " seconds. \n";
                    }
                }

            } else { // read receiver(s) and travel time info

                // read single receiver or differential traveltime data
                if (tokens.size() < 11) {
                    // store receiver name of onle receiver line in src rec file
                    std::vector<int> rec_id_list_one_line;

                    SrcRecInfo rec;

                    rec.id   = std::stoi(tokens[1]);
                    rec.name = tokens[2];
                    rec.lat  = static_cast<CUSTOMREAL>(std::stod(tokens[3])); // in degree
                    rec.lon  = static_cast<CUSTOMREAL>(std::stod(tokens[4])); // in degree
                    rec.dep  = static_cast<CUSTOMREAL>(-1.0*std::stod(tokens[5])/1000.0); // convert elevation in meter to depth in km
                    
                    // new receiver detected by its name
                    if(rec_name2id_att.find(rec.name) == rec_name2id_att.end()){    // new receiver
                        rec.id_att = -1-static_cast<int>(rec_name2id_att.size()); // assign id_rec_att based on the current size of rec_name2id_att
                        rec_name2id_att[rec.name] = rec.id_att; // assign id_att based on the current size of rec_name2id_att, 
                        rec_map[rec.id_att] = rec; // store the receiver info in rec_map with id_rec_att as key
                    } else {
                        rec.id_att = rec_name2id_att[rec.name]; // if the receiver already exists, use the existing id_rec_att
                    }

                    // store temporary receiver id list for each source
                    rec_id_list_one_line.push_back(rec.id_att);
                    rec_id_list_one_line.push_back(DATA_TYPE_ABS);

                    // traveltime data
                    DataInfo data;
                    if (tokens.size() > 8)
                        rec_weight = static_cast<CUSTOMREAL>(std::stod(tokens[8]));
                    else
                        rec_weight = 1.0; // default weight


                    data.data_weight = src_weight * rec_weight;
                    data.weight      = data.data_weight * abs_time_local_weight;
                    data.weight_reloc= data.data_weight * abs_time_local_weight_reloc;
                    data.phase       = tokens[6];

                    data.data_type          = DATA_TYPE_ABS;
                    data.id_src_att          = src_id_att;
                    data.id_rec_att          = rec.id_att;
                    data.time_observation   = static_cast<CUSTOMREAL>(std::stod(tokens[7])); // store read data

                    data_vec.push_back(data);

                    // store receiver id of onle receiver line in src rec file
                    rec_id_list.push_back(rec_id_list_one_line);

                    cc++;

                } else {
                    // read common source differential traveltime (cs_dif) or common receiver differential traveltime (cr_dif)

                    std::vector<int> rec_id_list_one_line;

                    // read differential traveltime
                    SrcRecInfo rec;
                    rec.id   = std::stoi(tokens[1]);
                    rec.name = tokens[2];
                    rec.lat  = static_cast<CUSTOMREAL>(std::stod(tokens[3])); // in degree
                    rec.lon  = static_cast<CUSTOMREAL>(std::stod(tokens[4])); // in degree
                    rec.dep  = static_cast<CUSTOMREAL>(-1.0*std::stod(tokens[5])/1000.0); // convert elevation in meter to depth in km

                    // new receiver detected by its name
                    if(rec_name2id_att.find(rec.name) == rec_name2id_att.end()){
                        rec.id_att = -1-static_cast<int>(rec_name2id_att.size()); // assign id_rec_att based on the current size of rec_name2id_att
                        rec_name2id_att[rec.name] = rec.id_att; // assign id_att based on the current size of rec_name2id_att, 
                        rec_map[rec.id_att] = rec; // store the receiver info in rec_map with id_rec_att as key
                    } else {
                        rec.id_att = rec_name2id_att[rec.name]; // if the receiver already exists, use the existing id_rec_att

                    }
                    // store temporary receiver id list for each source
                    rec_id_list_one_line.push_back(rec.id_att);


                    // differential traveltime data
                    DataInfo data;
                    if (tokens.size() > 13)
                        rec_weight = static_cast<CUSTOMREAL>(std::stod(tokens[13]));
                    else
                        rec_weight = 1.0; // default weight

                    data.data_weight = src_weight * rec_weight;
                    data.phase       = tokens[11];

                    //data.id_src_single          = src_id;
                    //data.name_src_single        = src_name;
                    // use common variables with src-rec data
                    // data.id_src          = src_id;
                    // data.name_src        = src_name;

                    // store the id and name of the first receiver (just used for key of the data_map)
                    // data.id_rec          = rec.id;
                    // data.name_rec        = rec.name;

                    // determine this data is cr_dif or cs_dif
                    bool is_cr_dif = tokens[11].find("cr")!=std::string::npos;

                    if (is_cr_dif) {
                        // cr_dif data
                        SrcRecInfo src2;
                        src2.id   = std::stoi(tokens[6]);
                        src2.name = tokens[7];
                        src2.lat  = static_cast<CUSTOMREAL>(std::stod(tokens[8])); // in degree
                        src2.lon  = static_cast<CUSTOMREAL>(std::stod(tokens[9])); // in degree
                        src2.dep  = static_cast<CUSTOMREAL>(std::stod(tokens[10])); // convert elevation in meter to depth in km

                        // new source detected by its name
                        if(src_name2id_att.find(src2.name) == src_name2id_att.end()){ // new source
                            src2.id_att = 1+static_cast<int>(src_name2id_att.size()); // assign id_src_att based on the current size of src_name2id_att
                            src_name2id_att[src2.name] = src2.id_att; // assign id_att based on the current size of src_name2id_att, 
                            src_map[src2.id_att] = src2; // store the source info in src_map with id_src_att as key
                        } else {
                            src2.id_att = src_name2id_att[src2.name]; // if the source already exists, use the existing id_src_att
                        }


                        
                        // store temporary receiver(source) id list for each source
                        rec_id_list_one_line.push_back(src2.id_att);
                        rec_id_list_one_line.push_back(DATA_TYPE_CRDIF);
                        // common receiver differential traveltime data
                        data.data_type              = DATA_TYPE_CRDIF;
                        data.id_src_att             = src_id_att;
                        data.id_rec_att             = rec.id_att;
                        data.id_pair_att            = src2.id_att;
                        data.time_observation       = static_cast<CUSTOMREAL>(std::stod(tokens[12])); // store read data

                        data.weight         = data.data_weight * cr_dif_time_local_weight;
                        data.weight_reloc   = data.data_weight * cr_dif_time_local_weight_reloc;
                        data_vec.push_back(data); // USE ONE-DATAMAP-FOR-ONE-SRCREC-LINE
                    } else {
                        // cs_dif data
                        SrcRecInfo rec2;
                        rec2.id   = std::stoi(tokens[6]);
                        rec2.name = tokens[7];
                        rec2.lat  = static_cast<CUSTOMREAL>(std::stod(tokens[8])); // in degree
                        rec2.lon  = static_cast<CUSTOMREAL>(std::stod(tokens[9])); // in degree
                        rec2.dep  = static_cast<CUSTOMREAL>(-1.0*std::stod(tokens[10])/1000.0); // convert elevation in meter to depth in km

                        // new receiver detected by its name
                        if(rec_name2id_att.find(rec2.name) == rec_name2id_att.end()){
                            rec2.id_att = -1-static_cast<int>(rec_name2id_att.size()); // assign id_rec_att based on the current size of rec_name2id_att
                            rec_name2id_att[rec2.name] = rec2.id_att; // assign id_att based on the current size of rec_name2id_att, 
                            rec_map[rec2.id_att] = rec2; // store the receiver info in rec_map with id_rec_att as key
                        } else {
                            rec2.id_att = rec_name2id_att[rec2.name]; // if the receiver already exists, use the existing id_rec_att
                        }

                        // store temporary receiver id list for each source
                        rec_id_list_one_line.push_back(rec2.id_att);
                        rec_id_list_one_line.push_back(DATA_TYPE_CSDIF);

                        // common source differential traveltime data
                        data.data_type              = DATA_TYPE_CSDIF;
                        data.id_src_att             = src_id_att;
                        data.id_rec_att             = rec.id_att;
                        data.id_pair_att            = rec2.id_att;
                        data.time_observation       = static_cast<CUSTOMREAL>(std::stod(tokens[12]));
                        
                        data.weight       = data.data_weight * cs_dif_time_local_weight;
                        data.weight_reloc = data.data_weight * cs_dif_time_local_weight_reloc;
                        data_vec.push_back(data); // USE ONE-DATAMAP-FOR-ONE-SRCREC-LINE
                    }

                    // store receiver id of one receiver line in src rec file
                    rec_id_list.push_back(rec_id_list_one_line);

                    cc++;
                }

                if (cc > ndata_tmp) {
                    // go to the next source
                    cc = 0;
                    i_src_now++;

                    // store the receiver id list for the source
                    rec_id_in_file.push_back(rec_id_list);
                    // clear the temporary receiver id list
                    rec_id_list.clear();

                    // timer
                    if (i_src_now % 1000 == 0 && world_rank == 0) {
                        std::cout << "reading source " << i_src_now << " finished in " << timer.get_t() << " seconds. dt = " << timer.get_t_delta() << " seconds. \n";
                    }
                }
            }

        } catch (std::invalid_argument& e) {
                std::cout << "Error: invalid argument in src_rec_file. Abort." << std::endl;
                std::cout << "problematic line: \n\n" << line << std::endl;
                std::cout << "Please carefully check whether n_data is consistent with the number of data of this earthquake." << std::endl << std::endl;
                exit(1);
                //MPI_Abort(MPI_COMM_WORLD, 1);
        }

    /*
        // print for DEBUG
        for (auto& t : tokens) {
            std::cout << t << "---";
        }
        std::cout << std::endl;
    */

    } // end of while loop

    // indicate the number of sources and receivers, data_info
    if (world_rank == 0){
        std::cout << "\nReading src_rec_file finished." << std::endl;
        std::cout << "number of sources: "   << src_map.size()  << std::endl;
        std::cout << "number of receivers: " << rec_map.size()  << std::endl;
        // std::cout << "number of data: "      << data_map.size() << "\n" << std::endl;
    }

    // indicate elapsed time
    std::cout << "Total elapsed time for reading src_rec_file: " << timer.get_t() << " seconds.\n";

    // check new version of src rec data
    if (if_verbose){
        for(auto iter = src_map.begin(); iter != src_map.end(); iter++){
            std::cout   << "source id: "     << iter->second.id
                        << ", source name: " << iter->second.name
                        << std::endl;
        }

        for(auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
            std::cout   << "receiver id: "     << iter->second.id
                        << ", receiver name: " << iter->second.name
                        << std::endl;
        }

        for (auto data : data_vec) {
            if (data.data_type == DATA_TYPE_ABS) {
                std::cout   << "source id_att: "     << data.id_src_att
                            << ", receiver id_att: " << data.id_rec_att
                            << ", traveltime: "    << data.time_observation
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CSDIF) {
                std::cout   << "source id_att: "          << data.id_src_att
                            << ", receiver pair id_att: " << data.id_rec_att
                            << ", "                     << data.id_pair_att
                            << ", traveltime: "         << data.time_observation
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CRDIF) {
                std::cout   << "source pair id_att: "     << data.id_src_att
                            << ", "                     << data.id_pair_att
                            << ", receiver id_att: "      << data.id_rec_att
                            << ", traveltime: "         << data.time_observation
                            << std::endl;
            } else {
                std::cout   << "error type of data" << std::endl;
            }
        }

    }
}


void parse_sta_correction_file(std::string& sta_correction_file,
                               std::map<int, SrcRecInfo>& rec_map){

    // read station correction file
    std::ifstream ifs;
    std::stringstream ss, ss_whole;

    if (sim_rank == 0){
        ifs.open(sta_correction_file);
        if (!ifs.is_open()){
            std::cout << "Error: cannot open sta_correction_file. Abort." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ss_whole << ifs.rdbuf();
        ifs.close();
    }

    std::string line;

    while(true){
        bool end_of_file = false;
        bool skip_this_line = false;

        line.clear();

        // read a line
        //if (sim_rank == 0){
            if (!std::getline(ss_whole,line)){
                end_of_file = true;
            }
        //}

        //broadcast_bool_single(end_of_file, 0);

        if (end_of_file){
            break;
        }

        // skip this line if it is a comment line
        //if (sim_rank == 0){
            if (line[0] == '#' || line.empty()){
                skip_this_line = true;
            }
        //}

        //broadcast_bool_single(skip_this_line, 0);

        if (skip_this_line) {
            continue;
        }

        // parse the line
        //int ntokens = 0;
        std::string token;
        std::vector<std::string> tokens;

        //if (sim_rank == 0){
            // erase the last space
            line.erase(line.find_last_not_of(" \n\r\t")+1);

            // parse the line with arbitrary number of spaces
            ss.clear();
            ss << line;

            while (std::getline(ss,token,' ')) {
                if (token.size() > 0)
                    tokens.push_back(token);
            }
            // number of tokens
            //ntokens = tokens.size();
        //}

        // broadcast ntokens
        //broadcast_i_single(ntokens, 0);
        // broadcast tokens
        //for (int i=0; i<ntokens; i++){
        //    if (sim_rank == 0)
        //        token = tokens[i];
        //    broadcast_str(token, 0);
        //    if (sim_rank != 0)
        //        tokens.push_back(token);
        //}

        try { // check failure of parsion line by line

            // store station corrections into rec_map
            std::string tmp_sta_name = tokens[0];

            for (auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
                if (iter->second.name == tmp_sta_name){
                    iter->second.sta_correct = static_cast<CUSTOMREAL>(std::stod(tokens[4]));
                    iter->second.sta_correct_kernel = 0.0;
                    break;
                }

                std::cout << "Did not find station " << tmp_sta_name << " in the src_rec file. Omit this station correction." << std::endl;
            } 
            
        } catch (std::invalid_argument& e) {
            std::cout << "Error: invalid argument in sta_correction_file. Abort." << std::endl;
            std::cout << "problematic line: \n\n" << line << std::endl;

            exit(1);
            //MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}


void separate_region_and_tele_src_rec_data(std::map<int, SrcRecInfo>            &src_map_back,
                                           std::map<int, SrcRecInfo>            &rec_map_back,
                                           std::vector<DataInfo>                &data_vec_back,
                                           std::map<int, SrcRecInfo>            &src_map,
                                           std::map<int, SrcRecInfo>            &rec_map,
                                           std::vector<DataInfo>                &data_vec,
                                           std::map<int, SrcRecInfo>            &src_map_tele,
                                           std::map<int, SrcRecInfo>            &rec_map_tele,
                                           std::vector<DataInfo>                &data_vec_tele,
                                           std::map<std::string, int> &data_type,
                                           int                        &N_abs_local_data,
                                           int                        &N_cr_dif_local_data,
                                           int                        &N_cs_dif_local_data,
                                           int                        &N_teleseismic_data,
                                           int                        &N_data,
                                           const CUSTOMREAL min_lat, const CUSTOMREAL max_lat,
                                           const CUSTOMREAL min_lon, const CUSTOMREAL max_lon,
                                           const CUSTOMREAL min_dep, const CUSTOMREAL max_dep,
                                           bool have_tele_data){
    // check if the source is inside the simulation boundary

    // initialize vectors for teleseismic events
    rec_map_tele.clear();
    src_map_tele.clear();
    data_vec_tele.clear();

    // clear original src, rec, data
    src_map.clear();
    rec_map.clear();
    data_vec.clear();

    // divide source list
    //
    // src_map_back ____> src_map
    //              \___> src_map_tele (teleseismic events)
    //
    for(auto iter = src_map_back.begin(); iter != src_map_back.end(); iter++){
        SrcRecInfo src = iter->second;
        if (src.lat < min_lat || src.lat > max_lat \
         || src.lon < min_lon || src.lon > max_lon \
         || src.dep < min_dep || src.dep > max_dep){

            // out of region (teleseismic events)
            src.is_out_of_region      = true;
            src_map_tele[iter->first] = src;

            // set a flag on backup data (for output)
            src_map_back[iter->first].is_out_of_region = true;

        } else {
            // within region (local events)
            src.is_out_of_region  = false;
            src_map[iter->first] = src;

        }
    }

    // check receiver location
    // receiver must be within the study.
    for(auto iter = rec_map_back.begin(); iter != rec_map_back.end(); iter++){
        SrcRecInfo rec = iter->second;
        if (rec.lat < min_lat || rec.lat > max_lat \
         || rec.lon < min_lon || rec.lon > max_lon \
         || rec.dep < min_dep || rec.dep > max_dep){

            std::cout   << "ERROR: receiver " << rec.name
                        << ", lat: " << rec.lat
                        << ", lon: " << rec.lon
                        << ", dep: " << rec.dep
                        << " is out of region. Please check the src_rec file." << std::endl;
            exit(1);
        }
    }

    // divide receiver and data list
    //
    // rec_map_back  ____> rec_map
    //               \___> rec_map_tele
    // data_vec_back____> data_vec
    //              \___> data_vec_tele
    //

    // loop over all data
    for (DataInfo& data : data_vec_back){

        // absolute traveltime
        if(data.data_type == DATA_TYPE_ABS){
            int id_src_att = data.id_src_att;
            int id_rec_att = data.id_rec_att;

            // if src is out of region, this is teleseismic data
            if(src_map_tele.find(id_src_att) != src_map_tele.end()){
                total_teleseismic_data_weight       += data.data_weight;
                data.weight                          = data.data_weight * teleseismic_weight;
                total_teleseismic_data_weight_reloc += data.data_weight;
                data.weight_reloc                    = data.data_weight * teleseismic_weight_reloc;
                data_vec_tele.push_back(data);
                rec_map_tele[id_rec_att]             = rec_map_back[id_rec_att];
                data_type["tele"]                    = 1;
            
                // if src is in the region
            } else {
                total_abs_local_data_weight         += data.data_weight;
                total_abs_local_data_weight_reloc   += data.data_weight;

                data_vec.push_back(data);
                rec_map[id_rec_att]                  = rec_map_back[id_rec_att];
                data_type["abs"]                     = 1;
            }
        
        // common receiver differential traveltime
        } else if (data.data_type == DATA_TYPE_CRDIF){
            int id_src_att1 = data.id_src_att;
            int id_src_att2 = data.id_pair_att;
            int id_rec_att  = data.id_rec_att;

            // if both sources are out of region
            if(src_map_tele.find(id_src_att1) != src_map_tele.end() \
            && src_map_tele.find(id_src_att2) != src_map_tele.end()){
                total_teleseismic_data_weight           += data.data_weight;
                data.weight                              = data.data_weight * teleseismic_weight;
                total_teleseismic_data_weight_reloc     += data.data_weight;
                data.weight_reloc                        = data.data_weight * teleseismic_weight_reloc;

                data_vec_tele.push_back(data);
                rec_map_tele[id_rec_att]                 = rec_map_back[id_rec_att];
                data_type["tele"]                        = 1;

            // if both sources is in the region
            } else if (src_map.find(id_src_att1) != src_map.end() \
                    && src_map.find(id_src_att2) != src_map.end() ) {
                total_cr_dif_local_data_weight          += data.data_weight;
                total_cr_dif_local_data_weight_reloc    += data.data_weight;
                data_vec.push_back(data);
                rec_map[id_rec_att]                      = rec_map_back[id_rec_att];
                data_type["cr_dif"]                      = 1;

            } else {
                std::cout << "ERROR data: common receiver differential time, but one teleseismic source, one local source";
                exit(1);
            }

        // common source differential traveltime
        } else if (data.data_type == DATA_TYPE_CSDIF){
            int id_src_att  = data.id_src_att;
            int id_rec_att1 = data.id_rec_att;
            int id_rec_att2 = data.id_pair_att;

            // if id_src_att is out of region
            if(src_map_tele.find(id_src_att) != src_map_tele.end() ){
                total_teleseismic_data_weight           += data.data_weight;
                data.weight                              = data.data_weight * teleseismic_weight;
                total_teleseismic_data_weight_reloc     += data.data_weight;
                data.weight_reloc                        = data.data_weight * teleseismic_weight_reloc;
                data_vec_tele.push_back(data);
                rec_map_tele[id_rec_att1]                = rec_map_back[id_rec_att1];
                rec_map_tele[id_rec_att2]                = rec_map_back[id_rec_att2];
                data_type["tele"]                        = 1;

            // if name_src is in the region
            } else {
                total_cs_dif_local_data_weight          += data.data_weight;
                total_cs_dif_local_data_weight_reloc    += data.data_weight;
                data_vec.push_back(data);
                rec_map[id_rec_att1]                     = rec_map_back[id_rec_att1];
                rec_map[id_rec_att2]                     = rec_map_back[id_rec_att2];
                data_type["cs_dif"]                      = 1;
            }
        }
    }



    if((!have_tele_data) && (src_map_tele.size() > 0)){
        std::cout << "ERROR: have_tele_data is false, but there are earthquakes out of study region:" << std::endl;
        for(auto iter = src_map_tele.begin(); iter != src_map_tele.end(); iter++){
            std::cout   << "source name: " << iter->second.name 
                        <<  ", lat: " << iter->second.lat
                        <<  ", lon: " << iter->second.lon
                        <<  ", dep: " << iter->second.dep
                        <<  std::endl;
        }
        std::cout << "Please set have_tele_data in InputParams.yaml to be TRUE, or remove above earthquakes" << std::endl;
        exit(1);
    }

    if((!have_tele_data) && (rec_map_tele.size() > 0)){
        std::cout << "ERROR: have_tele_data is false, but there are stations out of study region:" << std::endl;
        for(auto iter = rec_map_tele.begin(); iter != rec_map_tele.end(); iter++){
            std::cout   << "receiver name: " << iter->second.name
                        <<  ", lat: " << iter->second.lat
                        <<  ", lon: " << iter->second.lon
                        <<  ", dep: " << iter->second.dep
                        <<  std::endl;
        }
        std::cout << "Please set have_tele_data in InputParams.yaml to be TRUE, or remove above stations" << std::endl;
        exit(1);
    }


    //
    // balance the data weight
    //
    if (balance_data_weight){

        // local data
        for(DataInfo& data : data_vec){
            if(data.data_type == DATA_TYPE_ABS){
                data.weight = data.weight / total_abs_local_data_weight * (total_abs_local_data_weight + total_cr_dif_local_data_weight + total_cs_dif_local_data_weight);
            // common receiver differential traveltime
            } else if (data.data_type == DATA_TYPE_CRDIF){
                data.weight = data.weight / total_cr_dif_local_data_weight * (total_abs_local_data_weight + total_cr_dif_local_data_weight + total_cs_dif_local_data_weight);
            // common source differential traveltime
            } else if (data.data_type == DATA_TYPE_CSDIF){
                data.weight = data.weight / total_cs_dif_local_data_weight * (total_abs_local_data_weight + total_cr_dif_local_data_weight + total_cs_dif_local_data_weight);
            }
        }

        // teleseismic data
        for(DataInfo& data : data_vec_tele){
            data.weight = data.weight / total_teleseismic_data_weight;
        }
    }

    // balance the data weight for relocation
    if (balance_data_weight_reloc){
        for(DataInfo& data : data_vec){
            // absolute traveltime
            if(data.data_type == DATA_TYPE_ABS){
                data.weight_reloc = data.weight_reloc / total_abs_local_data_weight_reloc * (total_abs_local_data_weight_reloc + total_cr_dif_local_data_weight_reloc + total_cs_dif_local_data_weight_reloc);

            // common receiver differential traveltime
            } else if (data.data_type == DATA_TYPE_CRDIF){
                data.weight_reloc = data.weight_reloc / total_cr_dif_local_data_weight_reloc * (total_abs_local_data_weight_reloc + total_cr_dif_local_data_weight_reloc + total_cs_dif_local_data_weight_reloc);

            // common source differential traveltime
            } else if (data.data_type == DATA_TYPE_CSDIF){
                data.weight_reloc = data.weight_reloc / total_cs_dif_local_data_weight_reloc * (total_abs_local_data_weight_reloc + total_cr_dif_local_data_weight_reloc + total_cs_dif_local_data_weight_reloc);
            }
        }
    }

    //
    // count the number of data
    //
    // local data
    for(DataInfo& data : data_vec){
        // absolute traveltime
        if(data.data_type == DATA_TYPE_ABS){
            N_abs_local_data += 1;

        // common receiver differential traveltime
        } else if (data.data_type == DATA_TYPE_CRDIF){
            N_cr_dif_local_data += 1;

        // common source differential traveltime
        } else if (data.data_type == DATA_TYPE_CSDIF){
            N_cs_dif_local_data += 1;
        }
    }

    // teleseismic data
    N_teleseismic_data = data_vec_tele.size();

    // N_data is the total number of data
    N_data = N_abs_local_data + N_cr_dif_local_data + N_cs_dif_local_data + N_teleseismic_data;


    // check new version of src rec data
    if (if_verbose){
        std::cout << "local data: " << std::endl;

        for(auto iter = src_map.begin(); iter != src_map.end(); iter++){
            std::cout   << "source id: "     << iter->second.id
                        << ", source name: " << iter->second.name
                        << std::endl;
        }

        for(auto iter = rec_map.begin(); iter != rec_map.end(); iter++){
            std::cout   << "receiver id: "     << iter->second.id
                        << ", receiver name: " << iter->second.name
                        << std::endl;
        }


        for (DataInfo& data : data_vec) {
            if (data.data_type == DATA_TYPE_ABS) {
                std::cout   << "source name: "     << src_map[data.id_src_att].name
                            << ", receiver name: " << rec_map[data.id_rec_att].name
                            << ", traveltime: "    << data.time_observation
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CSDIF) {
                std::cout   << "source name: "          << src_map[data.id_src_att].name
                            << ", receiver pair name: " << rec_map[data.id_rec_att].name
                            << ", "                     << rec_map[data.id_pair_att].name
                            << ", traveltime: "         << data.time_observation
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CRDIF) {
                std::cout   << "source pair name: "     << src_map[data.id_src_att].name
                            << ", "                     << src_map[data.id_pair_att].name
                            << ", receiver name: "      << rec_map[data.id_rec_att].name
                            << ", traveltime: "         << data.time_observation
                            << std::endl;
            } else {
                std::cout   << "error type of data" << std::endl;
            }
        }

        std::cout << std::endl << "tele data: " << std::endl;

        for(auto iter = src_map_tele.begin(); iter != src_map_tele.end(); iter++){
            std::cout   << "source id: "     << iter->second.id
                        << ", source name: " << iter->second.name
                        << std::endl;
        }

        for(auto iter = rec_map_tele.begin(); iter != rec_map_tele.end(); iter++){
            std::cout   << "receiver id: "     << iter->second.id
                        << ", receiver name: " << iter->second.name
                        << std::endl;
        }

        for (DataInfo& data : data_vec_tele) {
            if (data.data_type == DATA_TYPE_ABS) {
                std::cout   << "source name: "     << src_map[data.id_src_att].name
                            << ", receiver name: " << rec_map[data.id_rec_att].name
                            << ", traveltime: "    << data.time_observation
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CSDIF) {
                std::cout   << "source name: "          << src_map[data.id_src_att].name
                            << ", receiver pair name: " << rec_map[data.id_rec_att].name
                            << ", "                     << rec_map[data.id_pair_att].name
                            << ", traveltime: "         << data.time_observation
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CRDIF) {
                std::cout   << "source pair name: "     << src_map[data.id_src_att].name
                            << ", "                     << src_map[data.id_pair_att].name
                            << ", receiver name: "      << rec_map[data.id_rec_att].name
                            << ", traveltime: "         << data.time_observation
                            << std::endl;
            } else {
                std::cout   << "error type of data" << std::endl;
            }
        }
    }

}


void do_swap_src_rec(std::map<int, SrcRecInfo> &src_map_all,
                     std::map<int, SrcRecInfo> &rec_map_all,
                     std::vector<DataInfo> &data_vec_all) {

    // swap src/rec points
    // at this moment, all the sources are divided into src_points (regional) and tele_src_points (teleseismic)

    // Start timer
    std::string timer_name = "swap_src_rec";
    Timer timer(timer_name);

    // swap src_map_all and rec_map_all
    std::map<int, SrcRecInfo> tmp_src_rec_map = src_map_all;
    src_map_all = rec_map_all;
    rec_map_all = tmp_src_rec_map;


    std::vector<DataInfo> tmp_data_vec; // -> data_vec_all
    // swap data_vec_all. Only need to modofy 
    // 1. data_type: DATA_TYPE_ABS -> DATA_TYPE_ABS, DATA_TYPE_CSDIF -> DATA_TYPE_CRDIF, DATA_TYPE_CRDIF -> DATA_TYPE_CSDIF
    // 2. id_src_att <-> id_rec_att, id_pair_att <-> id_pair_att
    for(const auto& data : data_vec_all){
        DataInfo tmp_data = data;

        if (tmp_data.data_type == DATA_TYPE_ABS){
            // absolute traveltime  ->  absolute traveltime
            // |    abs     |               |    abs     |
            // |  s0 - r0   |       ->      |  r0 - s0   |
            // |            |               |            |
            // |            |               |            |

            tmp_data.id_src_att = data.id_rec_att;
            tmp_data.id_rec_att = data.id_src_att;
            tmp_data.dual_data = false; // is not a dual data. contribute both objective function and kernel
            tmp_data_vec.push_back(tmp_data);
        
        } else if (tmp_data.data_type == DATA_TYPE_CSDIF){
            // common source differential traveltime  ->  common receiver differential traveltime
            // |    cs_dif     |            |          cr_dif           |
            // |   s0 - r1     |    ->      |   r1 - s0     r2 - s0     |
            // |   |           |            |        |           |      |
            // |   r2          |            |        r2          r1     |    

            tmp_data.data_type              = DATA_TYPE_CRDIF;

            // one data
            tmp_data.id_src_att              = data.id_rec_att;
            tmp_data.id_rec_att              = data.id_src_att;
            // tmp_data.id_pair_att             = data.id_pair_att;
            tmp_data.dual_data = false; // one is not a dual data. contribute both objective function and kernel
            tmp_data_vec.push_back(tmp_data);

            // the other data
            tmp_data.id_src_att              = data.id_pair_att;
            tmp_data.id_rec_att              = data.id_src_att;
            tmp_data.id_pair_att             = data.id_rec_att;
            tmp_data.dual_data = true; // the other is a dual data. contribute kernel only
            tmp_data.time_observation = -1.0 * data.time_observation;
            tmp_data_vec.push_back(tmp_data);
        
        } else if (tmp_data.data_type == DATA_TYPE_CRDIF){
            // common receiver differential traveltime  ->  common source differential traveltime
            // |    cr_dif     |        |    cs_dif     |
            // |   s0 - r3     |    ->  |   r3 - s0     |
            // |        |      |        |   |           |
            // |        s1     |        |   s1          |

            tmp_data.data_type              = DATA_TYPE_CSDIF;

            tmp_data.id_src_att              = data.id_rec_att;
            tmp_data.id_rec_att              = data.id_src_att;
            // tmp_data.id_pair_att             = data.id_pair_att;
            tmp_data.dual_data = false; // only need one data for common source dif, contribute both objective function and kernel
            tmp_data_vec.push_back(tmp_data);
        }
    }

    // replace data_vec_all with swapped data_vec_all
    data_vec_all = tmp_data_vec;

    // swap total_data_weight
    CUSTOMREAL tmp_wt = total_cr_dif_local_data_weight;
    total_cr_dif_local_data_weight = total_cs_dif_local_data_weight;
    total_cs_dif_local_data_weight = tmp_wt;

    // TO DO
    // // set n_data (number of receivers for each source)
    // for (auto it_src = src_map.begin(); it_src != src_map.end(); it_src++){
    //     it_src->second.n_data = data_map[it_src->second.name].size();
    // }

    

    // check new version of src rec data
    if (if_verbose){
        std::cout << "do swap sources and receivers" << std::endl;

        for(auto iter = src_map_all.begin(); iter != src_map_all.end(); iter++){
            std::cout   << "source id: " << iter->second.id
                        << ", source name: " << iter->second.name
                        << std::endl;
        }

        for(auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
            std::cout   << "receiver id: " << iter->second.id
                        << ", receiver name: " << iter->second.name
                        << std::endl;
        }

        for(auto data : data_vec_all){
            if (data.data_type == DATA_TYPE_ABS){
                std::cout   << ", absolute traveltime: " << data.time_observation
                            << ", source name: "       << src_map_all[data.id_src_att].name
                            << ", receiver name: "     << rec_map_all[data.id_rec_att].name
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CSDIF){
                std::cout   << ", common source differential traveltime: " << data.time_observation
                            << ", source name: "                         << src_map_all[data.id_src_att].name
                            << ", receiver pair name: "                  << rec_map_all[data.id_rec_att].name
                            << ", "                                      << rec_map_all[data.id_pair_att].name
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CRDIF){
                std::cout   << ", common receiver differential traveltime: " << data.time_observation
                            << ", source pair name: "                      << src_map_all[data.id_src_att].name
                            << ", "                                        << src_map_all[data.id_pair_att].name
                            << ", receiver name: "                         << rec_map_all[data.id_rec_att].name
                            << std::endl;
            }
        }
        std::cout << "data vec size: " << data_vec_all.size() << std::endl;
    }

    // indicate elapsed time
    std::cout << "Total elapsed time for swapping src rec: " << timer.get_t() << " seconds.\n";
}

// do not swap source and receiver, process common receiver differential traveltime data
void do_not_swap_src_rec(std::map<int, SrcRecInfo> &src_map_all,
                         std::map<int, SrcRecInfo> &rec_map_all,
                         std::vector<DataInfo>     &data_vec_all) {

    // do not swap src/rec points

    // Start timer
    std::string timer_name = "do_not_swap_src_rec";
    Timer timer(timer_name);

    std::vector<DataInfo> tmp_data_vec;// = data_vec_all;

    // for each element of src_map, count the number of rec_map with the same value of
    for (auto data : data_vec_all){
        DataInfo tmp_data = data;

        // common receiver differential traveltime
        if (tmp_data.data_type == DATA_TYPE_CRDIF){
            // keep the original data, meanwhile, add cr_dif data with the other source
            // |    cr_dif     |            |          cr_dif           |
            // |   s0 - r3     |    ->      |   s0 - r3     s1 - r3     |
            // |        |      |            |        |           |      |
            // |        s1     |            |        s1          s0     |

            // one data
            tmp_data.dual_data = false;  // contribute both objective function and kernel
            tmp_data_vec.push_back(tmp_data);     // original data

            // the other data with the other source.
            // Note: name_src = cr_dif time always represent time(src1, rec) - time(src2,rec). Thus, -1 is necessary
            tmp_data.id_src_att = data.id_pair_att;
            tmp_data.id_pair_att = data.id_src_att;
            tmp_data.time_observation = -1.0 * data.time_observation;
            tmp_data.dual_data = true; // contribute kernel only
            tmp_data_vec.push_back(tmp_data);
        } else {
            // abs data and cs_dif data remain unchanged
            // |    abs     |    cs_dif     |
            // |  s0 - r0   |   s0 - r1     |
            // |            |   |           |
            // |            |   r2          |

            tmp_data.dual_data = false; // contribute both objective function and kernel
            tmp_data_vec.push_back(tmp_data);
        }
    }

    // replace data_map with swapped data map
    data_vec_all = tmp_data_vec;


    // check new version of src rec data
    if (if_verbose){
        std::cout << "do not swap sources and receivers" << std::endl;

        for(auto iter = src_map_all.begin(); iter != src_map_all.end(); iter++){
            std::cout   << "source id: " << iter->second.id
                        << ", source name: " << iter->second.name
                        << std::endl;
        }

        for(auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
            std::cout   << "receiver id: " << iter->second.id
                        << ", receiver name: " << iter->second.name
                        << std::endl;
        }

        for(auto data : data_vec_all){
            if (data.data_type == DATA_TYPE_ABS){
                std::cout   << ", absolute traveltime: " << data.time_observation
                            << ", source name: "       << src_map_all[data.id_src_att].name
                            << ", receiver name: "     << rec_map_all[data.id_rec_att].name
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CSDIF){
                std::cout   << ", common source differential traveltime: " << data.time_observation
                            << ", source name: "                         << src_map_all[data.id_src_att].name
                            << ", receiver pair name: "                  << rec_map_all[data.id_rec_att].name
                            << ", "                                      << rec_map_all[data.id_pair_att].name
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CRDIF){
                std::cout   << ", common receiver differential traveltime: " << data.time_observation
                            << ", source pair name: "                      << src_map_all[data.id_src_att].name
                            << ", "                                        << src_map_all[data.id_pair_att].name
                            << ", receiver name: "                         << rec_map_all[data.id_rec_att].name
                            << std::endl;
            }
        }
        std::cout << "data vec size: " << data_vec_all.size() << std::endl;
    }

    // indicate elapsed time
    std::cout << "Total elapsed time for not swapping src rec: " << timer.get_t() << " seconds.\n";
}



// merge the teleseismic data lsit into the local data list
void merge_region_and_tele_src(std::map<int, SrcRecInfo> &src_map_all,
                               std::map<int, SrcRecInfo> &rec_map_all,
                               std::vector<DataInfo>     &data_vec_all,
                               std::map<int, SrcRecInfo> &src_map_tele,
                               std::map<int, SrcRecInfo> &rec_map_tele,
                               std::vector<DataInfo>     &data_vec_tele){

    if(src_map_tele.size() > 0) {
        for (auto iter = src_map_tele.cbegin(); iter != src_map_tele.cend();){
            src_map_all[iter->first] = iter->second;
            // erase pushed data
            src_map_tele.erase(iter++);
        }

        for (auto iter = rec_map_tele.cbegin(); iter != rec_map_tele.cend();){
            rec_map_all[iter->first] = iter->second;
            // erase pushed data
            rec_map_tele.erase(iter++);
        }

        for (auto data : data_vec_tele){
            data_vec_all.push_back(data);
        }   
    }

    // delete after merging
    rec_map_tele.clear();
    src_map_tele.clear();
    data_vec_tele.clear();


    if (if_verbose){
        std::cout << "merge region and tele src" << std::endl;

        for(auto iter = src_map_all.begin(); iter != src_map_all.end(); iter++){
            std::cout   << "source id: " << iter->second.id
                        << ", source name: " << iter->second.name
                        << std::endl;
        }

        for(auto iter = rec_map_all.begin(); iter != rec_map_all.end(); iter++){
            std::cout   << "receiver id: " << iter->second.id
                        << ", receiver name: " << iter->second.name
                        << std::endl;
        }

        for(auto data : data_vec_all){

            if (data.data_type == DATA_TYPE_ABS){
                std::cout   << ", absolute traveltime: " << data.time_observation
                            << ", source name: "       << src_map_all[data.id_src_att].name
                            << ", receiver name: "     << rec_map_all[data.id_rec_att].name
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CSDIF){
                std::cout   << ", common source differential traveltime: " << data.time_observation
                            << ", source name: "                         << src_map_all[data.id_src_att].name
                            << ", receiver pair name: "                  << rec_map_all[data.id_rec_att].name
                            << ", "                                      << rec_map_all[data.id_pair_att].name
                            << std::endl;
            } else if (data.data_type == DATA_TYPE_CRDIF){
                std::cout   << ", common receiver differential traveltime: " << data.time_observation
                            << ", source pair name: "                      << src_map_all[data.id_src_att].name
                            << ", "                                        << src_map_all[data.id_pair_att].name
                            << ", receiver name: "                         << rec_map_all[data.id_rec_att].name
                            << std::endl;
            }
        }
        std::cout << "data vec size: " << data_vec_all.size() << std::endl;
    }
}

// reorder data vector in the order of id_src_att, id_rec_att, id_pair_att.        
void reorder_data_vector(std::map<int, SrcRecInfo> &src_map_all,
                         std::vector<DataInfo>     &data_vec_all){

    // sort data_vec_all by id_src_att, id_rec_att, and id_pair_att
    std::sort(
        data_vec_all.begin(),
        data_vec_all.end(),
        // sort by id_src_att, id_rec_att, and id_pair_att
        [](const DataInfo& a, const DataInfo& b) {
            return std::tie(a.id_src_att, a.id_rec_att, a.id_pair_att)
                 < std::tie(b.id_src_att, b.id_rec_att, b.id_pair_att);
        }
    );

    // set data_begin, data_end, and n_data for each source
    // for(auto iter = src_map_all.begin(); iter != src_map_all.end(); iter++){
    //     iter->second.data_begin = -1;
    // }

    size_t begin = 0;  // begin index

    while (begin < data_vec_all.size()) {
        int id_src_att = data_vec_all[begin].id_src_att;    // source id

        // find the end index of the current source
        size_t end = begin + 1;
        while (end < data_vec_all.size() && data_vec_all[end].id_src_att == id_src_att) {
            end++;  // if src id is the same, move to the next data
        }
        // now, data_vec_all[end] has a different src id, or end is out of range


        // set data_begin, data_end, and n_data for the current source
        src_map_all[id_src_att].data_begin = begin;
        src_map_all[id_src_att].data_end = end;
        src_map_all[id_src_att].n_data = end - begin;

        // move to the next source
        begin = end;
    }
}


// distribute the source/receiver list and data list to all the processors
void distribute_src_rec_data(std::map<int, SrcRecInfo>&     src_map_all,
                             std::map<int, SrcRecInfo>&     rec_map_all,
                             std::vector<DataInfo>&         data_vec_all,
                             std::map<int, SrcRecInfo>&     src_map_this_sim,
                             std::map<int, SrcRecInfo>&     rec_map_this_sim,
                             std::vector<DataInfo>&         data_vec_this_sim){


    // this process is done by only the processes which stores the data.
    if (proc_store_srcrec) {

        // initialize the source/receiver list and data list for this simulutaneous run group
        src_map_this_sim.clear();
        rec_map_this_sim.clear();
        data_vec_this_sim.clear();

        // number of total sources
        int n_src = 0;

        // number of total sources
        if (proc_read_srcrec)
            n_src = src_map_all.size();

        // broadcast the number of sources to all the processors (level 1)
        broadcast_i_single_inter_sim(n_src, 0); // inter simulutaneous run group

        // store the total number of sources
        nsrc_total = n_src;

        

        // detemine id_src corresponding to which id_src_att in src_map_all 
        std::vector<int> id_src_att_vector;
        if (id_sim == 0) {  // for rank 0, src_map_all -> id_src_att_vector
            id_src_att_vector = srcrec_id_2_id_att(src_map_all);
        }

        // assign sources to each simulutaneous run group
        for (int i_src = 0; i_src < n_src; i_src++) {
            // id of simulutaneous run group to which the i_src-th source belongs
            int dst_id_sim = select_id_sim_for_src(i_src, n_sims);

            // broadcast the source name
            int id_src_att;
            if (id_sim == 0 && subdom_main){            // for rank 0, get the key of src_map_all
                id_src_att = id_src_att_vector[i_src];
            }

            broadcast_i_single_inter_sim(id_src_att, 0); // (level 1) broadcast the source id to all the simulutaneous run groups
            if (id_sim==0){ // sender

                if (dst_id_sim == id_sim){ // if the destination is itself (rank 0), directly get the info from src_map_all, rec_map_all, and data_map_all.
                    int data_begin = src_map_all[id_src_att].data_begin;
                    int data_end   = src_map_all[id_src_att].data_end;

                    // src
                    src_map_this_sim[id_src_att] = src_map_all[id_src_att];
                    src_map_this_sim[id_src_att].data_begin = data_vec_this_sim.size(); // data begin index for this source
                    src_map_this_sim[id_src_att].data_end   = src_map_this_sim[id_src_att].data_begin + (data_end - data_begin); // data end index for this source
                    src_map_this_sim[id_src_att].n_data     = data_end - data_begin; // number of data for this source

                    // data
                    for (int i_data = data_begin; i_data < data_end; i_data++){
                        DataInfo& data = data_vec_all[i_data];
                        data_vec_this_sim.push_back(data);

                        // rec by data
                        rec_map_this_sim[data.id_rec_att] = rec_map_all[data.id_rec_att];
                        
                        // store the second receiver for rec_pair
                        if (data.data_type == DATA_TYPE_CSDIF){
                            rec_map_this_sim[data.id_pair_att] = rec_map_all[data.id_pair_att];
                        }
                    }


                } else { // (level 1) if the destination is not itself (rank 0), send info to other ranks
                    // 1. send src
                    send_src_info_inter_sim(src_map_all[id_src_att], dst_id_sim);  // n_data has been included in src_map

                    // 2. send data first
                    int n_data = src_map_all[id_src_att].n_data;
                    std::map<int, bool> rec_id_to_be_sent; // store the receiver id to be sent
                    
                    int data_begin = src_map_all[id_src_att].data_begin;
                    int data_end   = src_map_all[id_src_att].data_end;
                    if (n_data > 0){
                        for (int idata = data_begin; idata < data_end; idata++){
                            DataInfo& data = data_vec_all[idata];
                            send_data_info_inter_sim(data, dst_id_sim);

                            rec_id_to_be_sent[data.id_rec_att] = true; // mark the receiver id to be sent
                            // send the second receiver for rec_pair
                            if (data.data_type == DATA_TYPE_CSDIF){
                                rec_id_to_be_sent[data.id_pair_att] = true; // mark the second receiver id to be sent
                            }
                        }
                    }

                    // 3. send receiver next
                    int n_rec = rec_id_to_be_sent.size();
                    send_i_single_sim(&n_rec, dst_id_sim);  // send the number of receivers (level 1, inter sim)

                    for (auto iter = rec_id_to_be_sent.begin(); iter != rec_id_to_be_sent.end(); iter++){
                        send_rec_info_inter_sim(rec_map_all[iter->first], dst_id_sim);
                    }
                }

            } else { // receive (other ranks)

                if (dst_id_sim == id_sim){      // this rank is the destination. Try to receive the info

                    // 1. receive src
                    SrcRecInfo tmp_SrcInfo;
                    recv_src_info_inter_sim(tmp_SrcInfo, 0);
                    int n_data = tmp_SrcInfo.data_end - tmp_SrcInfo.data_begin; // number of data for this source

                    // add the received src_map to the src_map (id_src_att == tmp_SrcInfo.id_att)
                    tmp_SrcInfo.data_begin = data_vec_this_sim.size(); // data begin index for this source
                    tmp_SrcInfo.data_end   = tmp_SrcInfo.data_begin + n_data;
                    src_map_this_sim[tmp_SrcInfo.id_att] = tmp_SrcInfo;

                    // 2. receiver data
                    for (int i_data = 0; i_data < n_data; i_data++){
                        // receive data_info from the main process of dst_id_sim
                        DataInfo tmp_DataInfo;
                        recv_data_info_inter_sim(tmp_DataInfo, 0);
                        data_vec_this_sim.push_back(tmp_DataInfo);
                    }

                    // 3. receive receivers
                    int n_rec = 0;
                    recv_i_single_sim(&n_rec, 0);  // receive the number of receivers (level 1, inter sim)

                    for (int i_rec = 0; i_rec < n_rec; i_rec++){
                        // receive rec_info from the main process of dst_id_sim
                        SrcRecInfo tmp_RecInfo;
                        recv_rec_info_inter_sim(tmp_RecInfo, 0);
                        rec_map_this_sim[tmp_RecInfo.id_att] = tmp_RecInfo;
                    }

                } else {
                    // do nothing
                }

            } // end of if (id_sim==0)

        } // end of for i_src

    } // end of if (proc_store_srcrec)

    // check IP.src_ids_this_sim for this rank
    if (myrank==0 && if_verbose) {
        std::cout << id_sim << "assigned src id(name) : ";
        for (auto iter = src_map_this_sim.begin(); iter != src_map_this_sim.end(); iter++){
            std::cout << iter->second.id << "(" << iter->second.name << ") ";
        }
        std::cout << std::endl;
    }

}


// generate a list of events which involve common receiver double difference traveltime
void generate_src_map_with_common_receiver(std::vector<DataInfo>&       data_map,
                                           std::map<int, SrcRecInfo>&   src_map,
                                           std::map<int, SrcRecInfo>&   src_map_comm_recp){

    if (proc_store_srcrec) {

        for(auto data : data_map){
            if (data.data_type == DATA_TYPE_CRDIF) {
                // add this source and turn to the next source
                src_map_comm_recp[data.id_src_att] = src_map[data.id_src_att];
            }
        }

        // check if this sim group has common source double difference traveltime
        if (src_map_comm_recp.size() > 0){
            src_pair_exists = true;
        }

    } // end of if (proc_store_srcrec)

    // flag if any src_pair exists
    allreduce_bool_inplace_inter_sim(&src_pair_exists, 1); // inter-sim (level 1)
    allreduce_bool_inplace(&src_pair_exists, 1); // intra-sim / inter subdom (level 2)
    allreduce_bool_inplace_sub(&src_pair_exists, 1); // intra-subdom (level 3)
}


void prepare_src_map_for_2d_solver(std::map<int, SrcRecInfo>& src_map_all,
                                   std::map<int, SrcRecInfo>& src_map,
                                   std::map<int, SrcRecInfo>& src_map_2d) {
    //
    // src_map_2d: src map assigned to this simultaneous run group. source with the same depth, will be regarded as the same source

    if (proc_store_srcrec) {

        std::map<int, SrcRecInfo> tmp_src_map_unique;
        std::vector<int> tmp_src_id_list_unique;

        // at first, make a depth-unique source list in the main process from src_map_tele
        if (proc_read_srcrec) {

            for (auto iter = src_map_all.begin(); iter != src_map_all.end(); iter++){

                // skip if this is not a teleseismic source
                if (!iter->second.is_out_of_region)
                    continue;

                int tmp_src_id = iter->second.id_att;

                // check if there is no element in tmp_src_map_unique with the same iter->second.depth
                bool if_unique = true;
                for (auto iter2 = tmp_src_map_unique.begin(); iter2 != tmp_src_map_unique.end(); iter2++){
                    if (iter2->second.dep == iter->second.dep){
                        if_unique = false;
                        break;
                    }
                }

                if (if_unique) {
                    tmp_src_map_unique[tmp_src_id] = iter->second;
                }
            }
        }

        // broadcast the number of unique sources to all processes
        int n_src_unique = 0;
        if (proc_read_srcrec){  // rank 0
            n_src_unique = tmp_src_map_unique.size();
        } 
        broadcast_i_single_inter_sim(n_src_unique, 0); // inter simulutaneous run group

        std::vector<int> id_src_att_vector;
        if (id_sim == 0) {  // for rank 0, src_map_all -> id_src_att_vector
            id_src_att_vector = srcrec_id_2_id_att(tmp_src_map_unique);
        }

        // iterate over all the unique sources
        for (int i_src_unique = 0; i_src_unique < n_src_unique; i_src_unique++){
            int dst_id_sim = select_id_sim_for_src(i_src_unique, n_sims);

            if (id_sim==0){   // sender
                int id_src_att = id_src_att_vector[i_src_unique];
                if (dst_id_sim==id_sim){    // if the destination is itself (rank 0), directly get the info from src_map_all, rec_map_all, and data_map_all.
                    // store
                    src_map_2d[id_src_att] = tmp_src_map_unique[id_src_att];
                } else {    // (level 1) if the destination is not itself (rank 0), send info to other ranks
                    // send to dst_id_sim 
                    send_src_info_inter_sim(tmp_src_map_unique[id_src_att], dst_id_sim);
                }
            } else {  // receiver
                if (dst_id_sim==id_sim){    // this rank is the destination. Try to receive the info
                    // receive from 0
                    SrcRecInfo tmp_src;
                    recv_src_info_inter_sim(tmp_src, 0);
                    src_map_2d[tmp_src.id_att] = tmp_src;
                } else {
                    // do nothing
                }
            }
        }
    } // end of if (proc_store_srcrec)

    // print the number of sources ssigned to this simultaneous run group
    for (int i_sim=0; i_sim < n_sims; i_sim++){
        if (id_sim==i_sim && subdom_main && id_subdomain==0){
            std::cout << "id_sim = " << id_sim << " : " << src_map_2d.size() << " 2d-sources assigned." << std::endl;
        }
        synchronize_all_world();
    }

}

// srcrec_id_2_id_att is only called by main of level 2 and level 3.
std::vector<int> srcrec_id_2_id_att(std::map<int, SrcRecInfo>& srcrec_map){
    std::vector<int> id_srcrec_att_vector;
    if (proc_store_srcrec) {
        for (auto iter = srcrec_map.begin(); iter != srcrec_map.end(); iter++) {
            id_srcrec_att_vector.push_back(iter->first); // store the source id
        }
    }
    return id_srcrec_att_vector;
}

//
// Belows are the function for send/receiving SrcRecInfo, DataInfo object
// so when you add the member in those class, it will be necessary to modify
// the functions below for sharing the new member.
//

void send_src_info_inter_sim(SrcRecInfo &src, int dest){

    send_i_single_sim(&src.id, dest);
    send_i_single_sim(&src.year , dest);
    send_i_single_sim(&src.month, dest);
    send_i_single_sim(&src.day  , dest);
    send_i_single_sim(&src.hour , dest);
    send_i_single_sim(&src.min  , dest);
    send_cr_single_sim(&src.sec, dest);
    send_cr_single_sim(&src.lat, dest);
    send_cr_single_sim(&src.lon, dest);
    send_cr_single_sim(&src.dep, dest);
    send_cr_single_sim(&src.mag, dest);
    send_i_single_sim(&src.n_data, dest);
    send_str_sim(src.name, dest);
    send_bool_single_sim(&src.is_out_of_region, dest);
    send_i_single_sim(&src.data_begin, dest);
    send_i_single_sim(&src.data_end, dest);

}

void recv_src_info_inter_sim(SrcRecInfo &src, int orig){

    recv_i_single_sim(&src.id, orig);
    recv_i_single_sim(&src.year , orig);
    recv_i_single_sim(&src.month, orig);
    recv_i_single_sim(&src.day  , orig);
    recv_i_single_sim(&src.hour , orig);
    recv_i_single_sim(&src.min  , orig);
    recv_cr_single_sim(&src.sec, orig);
    recv_cr_single_sim(&src.lat, orig);
    recv_cr_single_sim(&src.lon, orig);
    recv_cr_single_sim(&src.dep, orig);
    recv_cr_single_sim(&src.mag, orig);
    recv_i_single_sim(&src.n_data, orig);
    recv_str_sim(src.name, orig);
    recv_bool_single_sim(&src.is_out_of_region, orig);
}


void broadcast_src_info(SrcRecInfo& src, int orig){
        broadcast_i_single(src.id, orig);
        //broadcast_i_single(src.year , orig);
        //broadcast_i_single(src.month, orig);
        //broadcast_i_single(src.day  , orig);
        //broadcast_i_single(src.hour , orig);
        //broadcast_i_single(src.min  , orig);
        //broadcast_cr_single(src.sec, orig);
        broadcast_cr_single(src.lat, orig);
        broadcast_cr_single(src.lon, orig);
        broadcast_cr_single(src.dep, orig);
        //broadcast_cr_single(src.mag, orig);
        broadcast_i_single(src.n_data, orig);
        broadcast_str(src.name, orig);
        broadcast_bool_single(src.is_out_of_region, orig);
}


void send_rec_info_inter_sim(SrcRecInfo &rec, int dest){

    send_i_single_sim(&rec.id, dest);
    send_str_sim(rec.name, dest);
    send_cr_single_sim(&rec.lon, dest);
    send_cr_single_sim(&rec.lat, dest);
    send_cr_single_sim(&rec.dep, dest);
}


void recv_rec_info_inter_sim(SrcRecInfo &rec, int orig){

    recv_i_single_sim(&rec.id, orig);
    recv_str_sim(rec.name, orig);
    recv_cr_single_sim(&rec.lon, orig);
    recv_cr_single_sim(&rec.lat, orig);
    recv_cr_single_sim(&rec.dep, orig);
}


void broadcast_rec_info(SrcRecInfo& rec, int orig){
        broadcast_i_single(rec.id, orig);
        broadcast_str(rec.name, orig);
        broadcast_cr_single(rec.lon, orig);
        broadcast_cr_single(rec.lat, orig);
        broadcast_cr_single(rec.dep, orig);
        broadcast_cr_single(rec.adjoint_source, orig);
        broadcast_cr_single(rec.adjoint_source_density, orig);
        broadcast_bool_single(rec.is_stop, orig);
}

void send_data_info_inter_sim(DataInfo &data, int dest){

    send_cr_single_sim(&data.data_weight, dest);
    send_cr_single_sim(&data.weight, dest);
    send_cr_single_sim(&data.weight_reloc, dest);
    send_str_sim(data.phase, dest);

    send_i_single_sim(&data.data_type, dest);

    send_bool_single_sim(&data.dual_data, dest);

    send_cr_single_sim(&data.time_observation, dest);

    send_i_single_sim(&data.id_src_att, dest);
    send_i_single_sim(&data.id_rec_att, dest);
    send_i_single_sim(&data.id_pair_att, dest);

    // for (int ipair=0; ipair<2; ipair++){
    //     send_i_single_sim(&data.id_rec_pair[ipair], dest);
    //     send_str_sim(data.name_rec_pair[ipair], dest);

    //     send_i_single_sim(&data.id_src_pair[ipair], dest);
    //     send_str_sim(data.name_src_pair[ipair], dest);
    // }
}


void recv_data_info_inter_sim(DataInfo &data, int orig){

    recv_cr_single_sim(&data.data_weight, orig);
    recv_cr_single_sim(&data.weight, orig);
    recv_cr_single_sim(&data.weight_reloc, orig);
    recv_str_sim(data.phase, orig);

    recv_i_single_sim(&data.data_type, orig);

    recv_bool_single_sim(&data.dual_data, orig);

    recv_cr_single_sim(&data.time_observation, orig);

    recv_i_single_sim(&data.id_src_att, orig);
    recv_i_single_sim(&data.id_rec_att, orig);
    recv_i_single_sim(&data.id_pair_att, orig);
    
    // for (int ipair=0; ipair<2; ipair++){
    //     recv_i_single_sim(&data.id_rec_pair[ipair], orig);
    //     recv_str_sim(data.name_rec_pair[ipair], orig);

    //     recv_i_single_sim(&data.id_src_pair[ipair], orig);
    //     recv_str_sim(data.name_src_pair[ipair], orig);
    // }
}


