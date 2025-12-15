#include "optimizer_gd.h"
#include <iostream>
#include "kernel_postprocessing.h"

Optimizer_gd::Optimizer_gd(InputParams& IP) : Optimizer(IP) {

    // for gradient descent method, 
    need_write_model = false;
    need_write_original_kernel = false;

    if(line_search_mode){
        alpha_sub_iter.resize(3);
        v_obj_sub_iter.resize(3);
    }
}

Optimizer_gd::~Optimizer_gd() {
}

// ---------------------------------------------------------
// ------------------ specified functions ------------------
// ---------------------------------------------------------


// smooth kernels (multigrid) + kernel normalization (kernel density normalization)
void Optimizer_gd::processing_kernels(InputParams& IP, Grid& grid, IO_utils& io, int& i_inv) {
    
    // initialize and backup modified kernels
    initialize_and_backup_modified_kernels(grid);

    // check kernel value range
    check_kernel_value_range(grid);

    // multigrid parameterization + kernel density normalization
    // Ks_loc, Keta_loc, Kxi_loc
    // --> 
    // Ks_processing_loc, Keta_processing_loc, Kxi_processing_loc
    Kernel_postprocessing::process_kernels(IP, grid);

    // 2. normalize kernels to -1 ~ 1
    Kernel_postprocessing::normalize_kernels(grid);

    // assign to modified kernels
    // Ks_processing_loc, Keta_processing_loc, Kxi_processing_loc
    // -->
    // Ks_update_loc, Keta_update_loc, Kxi_update_loc
    Kernel_postprocessing::assign_to_modified_kernels(grid);
}


// evaluate line search performance
bool Optimizer_gd::check_conditions_for_line_search(InputParams& IP, Grid& grid, int sub_iter, int quit_sub_iter, CUSTOMREAL v_obj_inout, CUSTOMREAL v_obj_try){
    // There are 8 ways to adjust step:
    // The current model: (step, obj) = (0, v_obj_inout)
    // The first try: (alpha, v1)
    // If v1 < v_obj_inout. Then, the second try: (2*alpha, v2), else (alpha/2, v2). So, the 6 cases are: 
    // v1 > v_obj_inout, (0, alpha, 1/2 alpha), quadratic: y = a*x^2 + b*x + v_obj_inout, optimal step alpha_quad = -b/2a
    // 1. a < 0, third try: 1/4 alpha                       (a new step, need to do forward modeling)
    // 2. a > 0, alpha_quad < 0, third try: 1/4 alpha       (a new step)
    // 3. a > 0, alpha_quad > 0, third try, alpha_quad      (a new step) if alpha_quad < 1/4 alpha, using 1/4 alpha to avoid too small step
    // v1 < v_obj_inout, (0, alpha, 2*alpha), 
    // 4. a < 0, third try: 2*alpha                         (have done, no need to do forward modeling)
    // 5. a > 0, alpha_quad > 2*alpha, third try: 2*alpha   (have done) Now, alpha_quad mush be greater than 0, so, just select 2*alpha
    // 6. a > 0, alpha_quad < 2*alpha, third try: alpha_quad (a new step) if alpha_quad < alpha/2, using alpha/2 to avoid too small step

    bool exit_flag = false;   // whether to break the line search loop

    if(sub_iter == 0){      // first try
        alpha = step_length_init;
        alpha_sub_iter[0] = alpha;
        v_obj_sub_iter[0] = v_obj_try;
        if(myrank == 0 && id_sim == 0){
            std::cout << std::endl; 
            std::cout << "Evaluate line search at sub-iteration 1: " << std::endl;
            std::cout << "    Tried step length alpha = " << alpha << std::endl;
            std::cout << "    Objective function value at current model: " << v_obj_inout << std::endl;
            std::cout << "    Objective function value at tried model: " << v_obj_try << std::endl;
        }
        if (v_obj_try > v_obj_inout){        // objective function increases
            alpha = 0.5 * alpha;
            if(myrank == 0 && id_sim == 0){
                std::cout << "    Objective function increases, so step length decreases to: " << alpha << std::endl;
            }
        } else {                            // objective function decreases
            alpha = 2.0 * alpha;    
            if(myrank == 0 && id_sim == 0){
                std::cout << "    Objective function decreases, so step length increases to: " << alpha << std::endl;
            }
        }
        if(myrank == 0 && id_sim == 0){
            std::cout << std::endl;
        }
    } else if (sub_iter == 1){      // second try
        alpha_sub_iter[1] = alpha;
        v_obj_sub_iter[1] = v_obj_try;
        CUSTOMREAL x1 = alpha_sub_iter[0];
        CUSTOMREAL x2 = alpha_sub_iter[1];
        CUSTOMREAL y0 = v_obj_inout;
        CUSTOMREAL y1 = v_obj_sub_iter[0];
        CUSTOMREAL y2 = v_obj_sub_iter[1];
        CUSTOMREAL quad_a = ((y2 - y0)*x1 - (y1 - y0)*x2) / (x1*x2*(x2-x1));
        CUSTOMREAL quad_b = ((y2 - y0)*x1*x1 - (y1 - y0)*x2*x2) / (x1*x2*(x1-x2));
        CUSTOMREAL alpha_quad = - quad_b / (2.0 * quad_a);
        if(myrank == 0 && id_sim == 0){
            std::cout << std::endl; 
            std::cout << "Evaluate line search at sub-iteration 2: " << std::endl;
            std::cout << "    Info of three models: " << std::endl;
            std::cout << "    Step lengths   (x): " << 0 << ", " << x1 << ", " << x2 << std::endl;
            std::cout << "    Obj fun values (y): " << y0 << ", " << y1 << ", " << y2 << std::endl;
            std::cout << "    Quadratic regression: y = " << quad_a << " * x^2 + " << quad_b << " * x + " << y0 << std::endl;
        }
        if(y1 > y0){       // path: 0, alpha, 1/2 alpha
            if(quad_a <= 0){                // case 1, a < 0
                alpha = 0.5 * alpha;
                if(myrank == 0 && id_sim == 0){
                    std::cout << "    Quadratic regression failed (a <= 0), a = " << quad_a << ", reduce step length to: " << alpha << std::endl;
                }
            } else if (alpha_quad <= 0){    // case 2, a > 0, alpha_quad < 0
                alpha = 0.5 * alpha;
                if(myrank == 0 && id_sim == 0){
                    std::cout << "    Quadratic regression gives negative step length, alpha_quad = " << alpha_quad << ", reduce step length to: " << alpha << std::endl;
                }
            } else {    // case 3, a > 0, alpha_quad > 0
                if (alpha_quad > 0.5 * alpha) {
                    alpha = alpha_quad;
                    if(myrank == 0 && id_sim == 0){
                        std::cout << "    Quadratic regression gives positive step length, alpha_quad = " << alpha_quad << ", accept this step length." << std::endl;
                    }
                } else {
                    alpha = 0.5 * alpha;
                    if(myrank == 0 && id_sim == 0){
                        std::cout << "    Quadratic regression gives too small step length, alpha_quad = " << alpha_quad << ", reduce step length by half to: " << alpha << std::endl;
                    }
                }
            }
        } else {        // path: 0, alpha, 2*alpha
            if(quad_a <= 0){        // case 4, a < 0
                exit_flag = true;
                if(myrank == 0 && id_sim == 0){
                    std::cout << "    Quadratic regression failed (a <= 0), a = " << quad_a << ", choose the second try, the step length is: " << alpha << std::endl;
                }
            } else if (alpha_quad >= alpha){    // case 5, a > 0, alpha_quad > 2*alpha
                exit_flag = true;
                if(myrank == 0 && id_sim == 0){
                    std::cout << "    Quadratic regression gives too large step length, alpha_quad = " << alpha_quad << ", choose the second try, the step length is: " << alpha << std::endl;
                }
            } else {    // case 6, a > 0, alpha_quad < 2*alpha
                if (alpha_quad > 0.25 * alpha) {
                    alpha = alpha_quad;
                    if(myrank == 0 && id_sim == 0){
                        std::cout << "    Quadratic regression gives moderate step length, alpha_quad = " << alpha_quad << ", accept this step length." << std::endl;
                    }
                } else {
                    alpha = 0.25 * alpha;
                    if(myrank == 0 && id_sim == 0){
                        std::cout << "    Quadratic regression gives too small step length, alpha_quad = " << alpha_quad << ", reduce step length by quarter to: " << alpha << std::endl;
                    }
                }
            }
        }
        if(myrank == 0 && id_sim == 0){
            std::cout << std::endl;
        }
    } else {        // final try
        exit_flag = true;
        alpha_sub_iter[2] = alpha;
        v_obj_sub_iter[2] = v_obj_try;
        if(myrank == 0 && id_sim == 0){
            std::cout << std::endl; 
            std::cout << "    Tried step length alpha = " << alpha << std::endl;
            std::cout << "    Objective function value at current model: " << v_obj_inout << std::endl;
            std::cout << "    Objective function value at tried model: " << v_obj_try << std::endl;
            std::cout << std::endl;
        }
    }

    // --------------- evaluate exit flag ---------------
    if(exit_flag){
        if(myrank == 0 && id_sim == 0){
            std::cout << std::endl; 
            std::cout << "Line search ends. The search process contains  " << sub_iter+1 << " sub-iterations." << std::endl;
            std::cout << "The information is summarized as follows: " << std::endl;
            std::cout   << std::setw(25) << "    Current,";
            std::cout   << std::setw(25) << " step length = " << 0;
            std::cout   << std::setw(25) << ", obj value = " << v_obj_inout << std::endl;
            for(int tmp_i = 0; tmp_i <= sub_iter; tmp_i++){
                std::cout   << std::setw(25) << "    Sub-iter " << tmp_i+1 << ",";
                std::cout   << std::setw(25) << " step length = " << alpha_sub_iter[tmp_i];
                std::cout   << std::setw(25) << ", obj value = " << v_obj_sub_iter[tmp_i] << std::endl;
            }
            // (to do) the output step length (in obj) is not the actual tried step length (used for model update), instead, the initial step length of the next iteration.
            // fix this bug later
            std::cout << std::endl;
            std::cout << "The step length is " << alpha << ". This step length may exceed the range [" << step_length_min << ", " << step_length_max << "], due to line search." << std::endl;
            alpha = std::max(step_length_min, std::min(step_length_max,  alpha));
            std::cout << "The initial step of next iteration is set to: " << alpha << std::endl;
            std::cout << std::endl;
        }
        
    }

    return exit_flag;
}





