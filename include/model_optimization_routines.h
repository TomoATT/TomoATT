#ifndef MODEL_OPTIMIZATION_ROUTINES_H
#define MODEL_OPTIMIZATION_ROUTINES_H

#include <iostream>
#include <memory>
#include "mpi_funcs.h"
#include "config.h"
#include "utils.h"
#include "input_params.h"
#include "grid.h"
#include "io.h"
#include "main_routines_calling.h"
#include "iterator_selector.h"
#include "iterator.h"
#include "iterator_legacy.h"
#include "iterator_level.h"
#include "source.h"
#include "receiver.h"
#include "kernel.h"
#include "model_update.h"
#include "lbfgs.h"
#include "objective_function_utils.h"


// do model update by gradient descent
inline void model_optimize(InputParams& IP, Grid& grid, IO_utils& io, int i_inv, \
                    CUSTOMREAL& v_obj_inout, CUSTOMREAL& old_v_obj, bool& first_src, std::ofstream& out_main) {

    // check kernel density
    check_kernel_density(grid, IP);

    // sum kernels among all simultaneous runs
    sumup_kernels(grid);

    // write out original kernels
    if (id_sim==0 && subdom_main && IP.get_if_output_kernel() && (IP.get_if_output_in_process() || i_inv >= IP.get_max_iter_inv() - 2 || i_inv == 0)) {
        // store kernel only in the first src datafile
        io.change_group_name_for_model();

        // write kernel
        io.write_Ks(grid, i_inv);
        io.write_Keta(grid, i_inv);
        io.write_Kxi(grid, i_inv);

        // write kernel density
        io.write_Ks_density(grid, i_inv);
        io.write_Kxi_density(grid, i_inv);
        io.write_Keta_density(grid, i_inv);
    }

    // smooth kernels (multigrid, and kdensity normalization)
    smooth_kernels(grid, IP);

    // write out modified kernels
    if (id_sim==0 && subdom_main && IP.get_if_output_kernel() && (IP.get_if_output_in_process() || i_inv >= IP.get_max_iter_inv() - 2 || i_inv == 0)) {
        // store kernel only in the first src datafile
        io.change_group_name_for_model();

        // kernel over density
        io.write_Ks_over_Kden(grid, i_inv);
        io.write_Keta_over_Kden(grid, i_inv);
        io.write_Kxi_over_Kden(grid, i_inv);

        // kernel over density with smoothing
        io.write_Ks_update(grid, i_inv);
        io.write_Keta_update(grid, i_inv);
        io.write_Kxi_update(grid, i_inv);
    }

    // change stepsize
    // Option 1: the step length is modulated when obj changes.
    if (step_method == OBJ_DEFINED){
        if(i_inv != 0){
            if (v_obj_inout < old_v_obj) {
                step_length_init    = std::min((CUSTOMREAL)0.02, step_length_init);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The obj keeps decreasing, from " << old_v_obj << " to " << v_obj_inout
                            << ", the step length is " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            } else if (v_obj_inout >= old_v_obj) {
                step_length_init    = std::max((CUSTOMREAL)0.0001, step_length_init*step_length_decay);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The obj keep increases, from " << old_v_obj << " to " << v_obj_inout
                            << ", the step length decreases from " << step_length_init/step_length_decay
                            << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            }
        } else {
            if(myrank == 0 && id_sim == 0){
                std::cout << std::endl;
                std::cout << "At the first iteration, the step length is " << step_length_init << std::endl;
                std::cout << std::endl;
            }
        }
    } else if (step_method == GRADIENT_DEFINED){
        // Option 2: we modulate the step length according to the angle between the previous and current gradient directions.
        // If the angle is less than XX degree, which means the model update direction is successive, we should enlarge the step size
        // Otherwise, the step length should decrease
        CUSTOMREAL angle = direction_change_of_model_update(grid);
        if(i_inv != 0){
            if (angle > step_length_gradient_angle){
                CUSTOMREAL old_step_length = step_length_init;
                step_length_init    = std::max((CUSTOMREAL)0.0001, step_length_init * step_length_down);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The angle between two update darections is " << angle
                            << ". Because the angle is greater than " << step_length_gradient_angle << " degree, the step length decreases from "
                            << old_step_length << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            } else if (angle <= step_length_gradient_angle) {
                CUSTOMREAL old_step_length = step_length_init;
                step_length_init    = std::min((CUSTOMREAL)0.02, step_length_init * step_length_up);
                if(myrank == 0 && id_sim == 0){
                    std::cout << std::endl;
                    std::cout << "The angle between two update darections is " << angle
                            << ". Because the angle is less than " << step_length_gradient_angle << " degree, the step length increases from "
                            << old_step_length << " to " << step_length_init << std::endl;
                    std::cout << std::endl;
                }
            }
        } else {
            if(myrank == 0 && id_sim == 0){
                std::cout << std::endl;
                std::cout << "At the first iteration, the step length is " << step_length_init << std::endl;
                std::cout << std::endl;
            }
        }
    } else {
        std::cout << std::endl;
        std::cout << "No supported method for step size change, step keep the same: " << step_length_init << std::endl;
        std::cout << std::endl;
    }

    // broadcast the step_length
    broadcast_cr_single(step_length_init,0);

    // update the model with the initial step size
    set_new_model(grid, step_length_init);

    // make station correction
    IP.station_correction_update(step_length_init_sc);

    // writeout temporary xdmf file
    if (IP.get_verbose_output_level())
        io.update_xdmf_file();

    synchronize_all_world();
}


inline std::vector<CUSTOMREAL> model_optimize_halve_stepping(InputParams& IP, Grid& grid, IO_utils& io, int i_inv, CUSTOMREAL& v_obj_inout, bool& first_src, std::ofstream& out_main) {

    CUSTOMREAL step_length = step_length_init; // step size init is global variable
    CUSTOMREAL diff_obj  = - 9999999999;
    CUSTOMREAL v_obj_old = v_obj_inout;
    CUSTOMREAL v_obj_new = 0.0;
    std::vector<CUSTOMREAL> v_obj_misfit_new = std::vector<CUSTOMREAL>(10, 0.0);
    int sub_iter_count   = 0;

    // sum kernels among all simultaneous runs
    sumup_kernels(grid);

    // smooth kernels
    smooth_kernels(grid, IP);

    // backup the initial model
    grid.back_up_fun_xi_eta_bcf();

    // update the model with the initial step size
    set_new_model(grid, step_length);


    if (IP.get_verbose_output_level()) {
        // store kernel only in the first src datafile
        io.change_group_name_for_model();

        // output updated velocity models
        io.write_Ks(grid, i_inv);
        io.write_Keta(grid, i_inv);
        io.write_Kxi(grid, i_inv);

        // output descent direction
        io.write_Ks_update(grid, i_inv);
        io.write_Keta_update(grid, i_inv);
        io.write_Kxi_update(grid, i_inv);
    }

    // writeout temporary xdmf file
    if (IP.get_verbose_output_level())
        io.update_xdmf_file();

    synchronize_all_world();

    // loop till
    while (sub_iter_count < max_sub_iterations) {
        // check the new objective function value
        // v_obj_new = run_simulation_one_step(IP, grid, io, i_inv, first_src, true); // run simulations with line search mode
        v_obj_misfit_new = run_simulation_one_step(IP, grid, io, i_inv, first_src, true, false); // run simulations with line search mode
        v_obj_new = v_obj_misfit_new[0];
        // if the new objective function value is larger than the old one, make the step width to be half of the previous one
        diff_obj = v_obj_new - v_obj_old;

        if ( diff_obj > _0_CR // if the objective function value is larger than the old one
          || std::abs(diff_obj/v_obj_old) > MAX_DIFF_RATIO_VOBJ) { // if the objective function reduced too much  ) {

            // set step length just for writing out
            step_length_init = step_length;
            // print status
            write_objective_function(IP, i_inv, v_obj_misfit_new, out_main, "sub iter");

            grid.restore_fun_xi_eta_bcf();
            step_length *= HALVE_STEP_RATIO;
            set_new_model(grid, step_length);

            sub_iter_count++;
        } else {
            // if the new objective function value is smaller than the old one, make the step width to be twice of the previous one
            goto end_of_sub_iteration;
        }

    }

end_of_sub_iteration:

    v_obj_inout = v_obj_new;
    step_length_init = step_length/(HALVE_STEP_RATIO)*HALVE_STEP_RESTORAION_RATIO; // use this step size for the next inversion

    // write adjoint field
    int next_i_inv = i_inv + 1;
    if (subdom_main && IP.get_if_output_source_field())
        io.write_adjoint_field(grid,next_i_inv);

    return v_obj_misfit_new;

}


// do model update
inline bool model_optimize_lbfgs(InputParams& IP, Grid& grid, IO_utils& io, int i_inv, CUSTOMREAL& v_obj_inout, bool& first_src, std::ofstream& out_main) {

    int        subiter_count = 0;              // subiteration count
    //CUSTOMREAL qp_0          = _0_CR;          // store initial p_k * grad(f_k) (descent_direction * gradient)
    CUSTOMREAL qp_t          = _0_CR;          // store p_k * grad(f_k)
    CUSTOMREAL td            = _0_CR;          // wolfes right step
    CUSTOMREAL tg            = _0_CR;          // wolfes left step
    CUSTOMREAL step_length   = step_length_init; // step size init is global variable
    CUSTOMREAL v_obj_reg     = _0_CR;          // regularization term == q_t
    //CUSTOMREAL q_0           = _0_CR;          // initial cost function
    CUSTOMREAL q_t           = _0_CR;          // current cost function
    std::vector<CUSTOMREAL> v_obj_misfit_new = std::vector<CUSTOMREAL>(10, 0.0);

    //
    // initialize
    //
    if (i_inv == 0) {
        // initialize regularization
        // regularization_penalty[:] = 1.0
        init_regulaization_penalty_with_one(grid);

        // calculate volume_domain = SquaredL2Norm(regularization_penalty)
        volume_domain = compute_volume_domain(grid);

        // volume_domain /= N_params
        volume_domain /= N_params;

        // weight_Tikonov = penalty_weight / volume_domain
        weight_Tikonov = regularization_weight / volume_domain;
        weight_Tikonov_ani = regularization_weight;
        std::cout << "DEBUG: weight_Tikonov: " << weight_Tikonov << std::endl;

        // calculate damp_weights
        compute_damp_weights(grid);

        // calculate gradient (run onestep forward+adjoint) not necessary to do here. already done
        //v_obj_misfit_new = run_simulation_one_step(IP, grid, io, i_inv, first_src, true);

        // Q0 = initial obj
        q_0 = v_obj_inout;

        // sum kernels among all simultaneous runs
        sumup_kernels(grid);

        // smooth gradient
        smooth_kernels(grid, IP);

        // calc regularization penalties
        calculate_regularization_penalty(grid);

        //// smooth calculated grad regularization penalty
        //smooth_gradient_regularization(grid);

        // calc (update) obj_regl = squaredL2Norm(regularization_penalty)
        v_obj_reg = calculate_regularization_obj(grid);

        // Q0 += 0.5*weight_Tiknov*obj_regl
        q_0 += v_obj_reg;

        // init_grad += weight_Tikonov * gadient_regularization_penalty
        add_regularization_grad(grid);

        // store model and gradient
        store_model_and_gradient(grid, i_inv);

        // log out
        if(myrank == 0 && id_sim ==0) {
            out_main \
                     << std::setw(5)  << i_inv \
              << "," << std::setw(5)  << subiter_count \
              << "," << std::setw(15) << step_length \
              << "," << std::setw(15) << q_0 \
              << "," << std::setw(15) << q_t \
              << "," << std::setw(15) << v_obj_reg \
              << "," << std::setw(15) << qp_0 \
              << "," << std::setw(15) << qp_t \
              << "," << std::setw(15) << td \
              << "," << std::setw(15) << tg \
              << "," << std::setw(15) << wolfe_c1*qp_0 \
              << "," << std::setw(15) << wolfe_c2*qp_0 \
              << "," << std::setw(15) << "init" << std::endl;
        }
    }

    // backup the initial model
    grid.back_up_fun_xi_eta_bcf();

    // update the model with the initial step size
    if (IP.get_verbose_output_level() && id_sim==0) {
        // store kernel only in the first src datafile
        io.change_group_name_for_model();

        // output updated velocity models
        io.write_Ks(grid, i_inv);
        io.write_Keta(grid, i_inv);
        io.write_Kxi(grid, i_inv);

        // output descent direction
        io.write_Ks_descent_dir(grid, i_inv);
        io.write_Keta_descent_dir(grid, i_inv);
        io.write_Kxi_descent_dir(grid, i_inv);
    }

    // writeout temporary xdmf file
    if (IP.get_verbose_output_level()&& id_sim==0)
        io.update_xdmf_file();

    //
    // iteration
    //
    if (i_inv > 0) {
        // sum kernels among all simultaneous runs
        sumup_kernels(grid);

        // smooth kernels
        smooth_kernels(grid, IP);
    }

    // compute descent direction
    calc_descent_direction(grid, i_inv, IP);

    // smooth descent direction
    smooth_descent_direction(grid);

    // calc qp_0 = inital grad * descent direction
    qp_0 = compute_q_k(grid);

    // loop wolfes subiterations
    bool wolfe_cond_ok = false;
    while (wolfe_cond_ok != true) {

        // initial guess for isub = 0
        if (i_inv == 0 && subiter_count == 0)
            initial_guess_step(grid, step_length, 1.0);
        else if (i_inv == 1 && subiter_count == 0)
            initial_guess_step(grid, step_length, LBFGS_RELATIVE_step_length);

        //// Update model
        set_new_model(grid, step_length);

        //// calculate gradient (run onestep forward+adjoint)
        v_obj_misfit_new = run_simulation_one_step(IP, grid, io, i_inv, first_src, true, false);

        //// Qt update (=current obj)
        q_t = v_obj_misfit_new[0];

        // sum kernels among all simultaneous runs
        sumup_kernels(grid);

        //// smooth gradient
        smooth_kernels(grid, IP);

        //// calculate regularization to grad
        calculate_regularization_penalty(grid);

        //// smooth calculated grad regularization penalty
        //smooth_gradient_regularization(grid);

        //// add regularization term to Qt (Qt += 0.5 * obj_regl )
        v_obj_reg = calculate_regularization_obj(grid);
        q_t += v_obj_reg;

        //// current grad += weight_Tikonov * gadient_regularization_penalty
        add_regularization_grad(grid);

        //// Qpt = Inner product(current_grad * descent_direction)
        qp_t = compute_q_k(grid);

        //// check wolfe conditions
        wolfe_cond_ok = check_wolfe_cond(grid, q_0, q_t, qp_0, qp_t, td, tg, step_length);

        // update the model with the initial step size
        if (IP.get_verbose_output_level() && id_sim==0) {
            // store kernel only in the first src datafile
            io.change_group_name_for_model();

            // gradient
            io.write_Ks_update(grid, i_inv*100+subiter_count);
            io.write_Keta_update(grid, i_inv*100+subiter_count);
            io.write_Kxi_update(grid, i_inv*100+subiter_count);

            // output descent direction
            io.write_Ks_descent_dir(grid, i_inv*100+subiter_count);
            io.write_Keta_descent_dir(grid, i_inv*100+subiter_count);
            io.write_Kxi_descent_dir(grid, i_inv*100+subiter_count);

        }

        // writeout temporary xdmf file
        if (IP.get_verbose_output_level()&& id_sim==0)
            io.update_xdmf_file();

        // log out
        if(myrank == 0 && id_sim ==0)
            out_main \
                     << std::setw(5)  << i_inv \
              << "," << std::setw(5)  << subiter_count \
              << "," << std::setw(15) << step_length \
              << "," << std::setw(15) << q_0 \
              << "," << std::setw(15) << q_t \
              << "," << std::setw(15) << v_obj_reg \
              << "," << std::setw(15) << qp_0 \
              << "," << std::setw(15) << qp_t \
              << "," << std::setw(15) << td \
              << "," << std::setw(15) << tg \
              << "," << std::setw(15) << wolfe_c1*qp_0 \
              << "," << std::setw(15) << wolfe_c2*qp_0 \
              << "," << std::setw(15) << wolfe_cond_ok << std::endl;

        if (wolfe_cond_ok) {
            goto end_of_subiteration;
        } else if (subiter_count > max_sub_iterations){
            // reached max subiter
            // exit
            std::cout << "reached max subiterations" << std::endl;

            return false;

        } else {
            // wolfe conditions not satisfied
            subiter_count++;

            grid.restore_fun_xi_eta_bcf();
        }

    }

end_of_subiteration:
    //// store model and gradient,
    store_model_and_gradient(grid, i_inv+1);

    //// Q0=Qt
    q_0 = q_t;
    qp_0 = qp_t;

    step_length_init = step_length; // use current step size for the next iteration


    if (myrank == 0)
        std::cout << "Wolfe conditions satisfied at iteration " << subiter_count << std::endl;

    return true;

}



#endif // MODEL_OPTIMIZATION_ROUTINES_H