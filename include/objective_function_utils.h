#ifndef OBJECTIVE_FUNCTION_UTILS_H
#define OBJECTIVE_FUNCTION_UTILS_H

#include <iostream>
#include <vector>

#include "config.h"
#include "input_params.h"

// prepare header line of objective_funciton.txt
inline void prepare_header_line(InputParams &IP, std::ofstream &out_main) {
    // prepare output for iteration status
    if(myrank == 0 && id_sim ==0){
        out_main.open(output_dir + "/objective_function.txt");
        if (optim_method == GRADIENT_DESCENT || optim_method == HALVE_STEPPING_MODE){

            out_main << std::setw(8) << std::right << "# iter,";
            out_main << std::setw(13) << std::right << " type,";

            // if (optim_method == HALVE_STEPPING_MODE)
            //     out_main << std::setw(8) << std::right << "subiter,";        (TODO in the future)
            std::string tmp = "obj(";
            if(IP.get_run_mode() == ONED_INVERSION){
                tmp.append(std::to_string(IP.N_abs_local_data));  // for 1d inversion, only absolute time residual is considered in obj
            } else {
                tmp.append(std::to_string(IP.N_data));
            }
            tmp.append("),");
            out_main << std::setw(20) << tmp;

            tmp = "obj_abs(";
            tmp.append(std::to_string(IP.N_abs_local_data));
            tmp.append("),");
            out_main << std::setw(20) << tmp;

            tmp = "obj_cs_dif(";
            if (IP.get_is_srcrec_swap())
                tmp.append(std::to_string(IP.N_cr_dif_local_data));
            else
                tmp.append(std::to_string(IP.N_cs_dif_local_data));
            tmp.append("),");
            out_main << std::setw(20) << tmp;

            tmp = "obj_cr_dif(";
            if (IP.get_is_srcrec_swap())
                tmp.append(std::to_string(IP.N_cs_dif_local_data));
            else
                tmp.append(std::to_string(IP.N_cr_dif_local_data));
            tmp.append("),");
            out_main << std::setw(20) << tmp;

            tmp = "obj_tele(";
            tmp.append(std::to_string(IP.N_teleseismic_data));
            tmp.append("),");
            out_main << std::setw(20) << tmp;

            out_main << std::setw(25) << "res(mean/std),";

            out_main << std::setw(25) << "res_abs(mean/std),";

            out_main << std::setw(25) << "res_cs_dif(mean/std),";

            out_main << std::setw(25) << "res_cr_dif(mean/std),";

            out_main << std::setw(25) << "res_tele(mean/std),";

            out_main << std::setw(20) << "step_length," << std::endl;

        } else if (optim_method == LBFGS_MODE)
            out_main << std::setw(6)  << "it,"        << std::setw(6)  << "subit,"  << std::setw(16) << "step_length," << std::setw(16) << "q_0," << std::setw(16) << "q_t," \
                     << std::setw(16) << "v_obj_reg," << std::setw(16) << "qp_0,"  << std::setw(16) << "qp_t,"       << std::setw(16) << "td,"  << std::setw(16) << "tg," \
                     << std::setw(16) << "c1*qp_0,"    << std::setw(16) << "c2*qp_0," << std::setw(6)  << "step ok,"    << std::endl;

    }
}


inline void write_objective_function(InputParams& IP, int i_inv, std::vector<CUSTOMREAL>& v_misfit_inout, std::ofstream& out_main, std::string type) {
    // output objective function
    if (myrank==0 && id_sim==0) {
        out_main << std::setw(8) << std::to_string(i_inv)+",";
        out_main << std::setw(13) << type;
        out_main << "," << std::setw(19) << v_misfit_inout[0];
        out_main << "," << std::setw(19) << v_misfit_inout[1];
        if (IP.get_is_srcrec_swap())
            out_main << "," << std::setw(19) << v_misfit_inout[3];
        else
            out_main << "," << std::setw(19) << v_misfit_inout[2];
        if (IP.get_is_srcrec_swap())
            out_main << "," << std::setw(19) << v_misfit_inout[2];
        else
            out_main << "," << std::setw(19) << v_misfit_inout[3];
        out_main << "," << std::setw(19) << v_misfit_inout[4];
        // res
        CUSTOMREAL mean;
        CUSTOMREAL std;
        std::string tmp;
        if (IP.N_data > 0) {
            mean = v_misfit_inout[5]/IP.N_data;
            std  = sqrt(v_misfit_inout[6]/IP.N_data - my_square(mean));
            tmp = std::to_string(mean);
            tmp.append("/");
            tmp.append(std::to_string(std));
            out_main << "," << std::setw(24) << tmp;
        } else {
            out_main << "," << std::setw(24) << "0.0/0.0";
        }
        // res_abs
        if (IP.N_abs_local_data > 0) {
            mean = v_misfit_inout[7]/IP.N_abs_local_data;
            std  = sqrt(v_misfit_inout[8]/IP.N_abs_local_data - my_square(mean));
            tmp = std::to_string(mean);
            tmp.append("/");
            tmp.append(std::to_string(std));
            out_main << "," << std::setw(24) << tmp;
        } else {
            out_main << "," << std::setw(24) << "0.0/0.0";
        }
        if (IP.get_is_srcrec_swap()){
            if (IP.N_cr_dif_local_data > 0) {
                mean = v_misfit_inout[11]/IP.N_cr_dif_local_data;
                std  = sqrt(v_misfit_inout[12]/IP.N_cr_dif_local_data - my_square(mean));
                tmp = std::to_string(mean);
                tmp.append("/");
                tmp.append(std::to_string(std));
                out_main << "," << std::setw(24) << tmp;
            } else {
                out_main << "," << std::setw(24) << "0.0/0.0";
            }
            if (IP.N_cs_dif_local_data > 0) {
                mean = v_misfit_inout[9]/IP.N_cs_dif_local_data;
                std  = sqrt(v_misfit_inout[10]/IP.N_cs_dif_local_data - my_square(mean));
                tmp = std::to_string(mean);
                tmp.append("/");
                tmp.append(std::to_string(std));
                out_main << "," << std::setw(24) << tmp;
            } else {
                out_main << "," << std::setw(24) << "0.0/0.0";
            }
        } else {
            if (IP.N_cs_dif_local_data > 0) {
                mean = v_misfit_inout[9]/IP.N_cs_dif_local_data;
                std  = sqrt(v_misfit_inout[10]/IP.N_cs_dif_local_data - my_square(mean));
                tmp = std::to_string(mean);
                tmp.append("/");
                tmp.append(std::to_string(std));
                out_main << "," << std::setw(24) << tmp;
            } else {
                out_main << "," << std::setw(24) << "0.0/0.0";
            }
            if (IP.N_cr_dif_local_data > 0) {
                mean = v_misfit_inout[11]/IP.N_cr_dif_local_data;
                std  = sqrt(v_misfit_inout[12]/IP.N_cr_dif_local_data - my_square(mean));
                tmp = std::to_string(mean);
                tmp.append("/");
                tmp.append(std::to_string(std));
                out_main << "," << std::setw(24) << tmp;
            } else {
                out_main << "," << std::setw(24) << "0.0/0.0";
            }
        }

        if (IP.N_teleseismic_data > 0) {
            mean = v_misfit_inout[13]/IP.N_teleseismic_data;
            std  = sqrt(v_misfit_inout[14]/IP.N_teleseismic_data - my_square(mean));
            tmp = std::to_string(mean);
            tmp.append("/");
            tmp.append(std::to_string(std));
            out_main << "," << std::setw(24) << tmp;
        } else {
            out_main << "," << std::setw(24) << "0.0/0.0";
        }
        if(type == "model update" || type == "1d inversion")
            out_main << "," << std::setw(19) << step_length_init << ",";

        out_main << std::endl;
    }
}


#endif // OBJECTIVE_FUNCTION_UTILS_H