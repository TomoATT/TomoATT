#ifndef ITERATOR_WRAPPER_CUH
#define ITERATOR_WRAPPER_CUH

#include <memory>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include "grid_wrapper.cuh"
#include "cuda_utils.cuh"


//void cuda_do_sweep_level_kernel_3rd();
//void cuda_do_sweep_level_kernel_1st();

void run_kernel(Grid_on_device*, int const&, int const&, int const&, dim3&, dim3&, int const&);

void initialize_sweep_params(Grid_on_device*);
void finalize_sweep_params(Grid_on_device*);
void cuda_run_iteration_forward(Grid_on_device*, int const&);
void cuda_run_iteration_upwind(Grid_on_device*, int const&);


#endif // ITERATOR_WRAPPER_CUH