#ifndef KERNEL_SMOOTH_H
#define KERNEL_SMOOTH_H

#include "config.h"
#include "grid.h"
#include "input_params.h"

namespace Kernel_smooth {

    void smooth_kernels(Grid& grid, InputParams& IP);
    
}


#endif // KERNEL_SMOOTH_H