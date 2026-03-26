#ifndef ITERATOR_LEVEL_H
#define ITERATOR_LEVEL_H

#include "iterator.h"


class Iterator_level : public Iterator {
public:
    Iterator_level(InputParams&, Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
protected:
    void do_sweep_adj(int, Grid&, InputParams&) override ; // do sweeping for adjoint routine
    void do_sweep(int, Grid&, InputParams&) override {}; // do sweeping
};

class Iterator_level_tele : public Iterator {
public:
    Iterator_level_tele(InputParams& , Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
protected:
    void do_sweep_adj(int, Grid&, InputParams&) override ; // do sweeping for adjoint routine
    void do_sweep(int, Grid&, InputParams&) override {}; // do sweeping
};

class Iterator_level_1st_order : public Iterator_level {
public:
    Iterator_level_1st_order(InputParams&, Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping
};

class Iterator_level_3rd_order : public Iterator_level {
public:
    Iterator_level_3rd_order(InputParams&, Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping
};

class Iterator_level_1st_order_upwind : public Iterator_level {
public:
    Iterator_level_1st_order_upwind(InputParams&, Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping
};

class Iterator_level_1st_order_tele : public Iterator_level_tele {
public:
    Iterator_level_1st_order_tele(InputParams&, Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping
};

class Iterator_level_3rd_order_tele : public Iterator_level_tele {
public:
    Iterator_level_3rd_order_tele(InputParams&, Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping
};

class Iterator_level_1st_order_upwind_tele : public Iterator_level_tele {
public:
    Iterator_level_1st_order_upwind_tele(InputParams&, Grid&, Source&, IO_utils&, const std::string&, bool, bool, bool);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping
};

#endif // ITERATOR_LEVEL_H