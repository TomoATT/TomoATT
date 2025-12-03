#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include <memory>
#include "config.h"

class Optimizer {
public:
    Optimizer();
    ~Optimizer();

    CUSTOMREAL alpha_R, alpha_L;
};

#endif // OPTIMIZER_H