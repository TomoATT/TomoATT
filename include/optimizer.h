#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include <memory>
#include "config.h"

namespace LBFGS_vtest {
    extern CUSTOMREAL alpha_R, alpha_L;     // extern is necessary to avoid redefinition, because alpha_R is declared in cpp. This line can be ignored.
}

#endif // OPTIMIZER_H