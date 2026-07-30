#ifndef ITERATOR_legacy_H
#define ITERATOR_legacy_H

#include "iterator.h"


class Iterator_legacy : public Iterator {
public:
    Iterator_legacy(InputParams&, Grid&, Source&, IO_utils&, bool, bool, bool);
protected:
    void do_sweep_adj(int, Grid&, InputParams&) override ; // do sweeping for adjoint routine
    virtual void do_sweep(int, Grid&, InputParams&) {}; // do sweeping
};


class Iterator_legacy_tele : public Iterator {
public:
    Iterator_legacy_tele(InputParams& , Grid&, Source&, IO_utils&, bool, bool, bool);
protected:
    void do_sweep_adj(int, Grid&, InputParams&) override ; // do sweeping for adjoint routine
    virtual void do_sweep(int, Grid&, InputParams&) {}; // do sweeping
};


class Iterator_legacy_1st_order : public Iterator_legacy {

public:
    Iterator_legacy_1st_order(InputParams&, Grid&, Source&, IO_utils&, bool, bool, bool);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping

};

class Iterator_legacy_3rd_order : public Iterator_legacy {

public:
    Iterator_legacy_3rd_order(InputParams&, Grid&, Source&, IO_utils&, bool, bool, bool);
    //void run_iteration_forward(InputParams&, Grid&, IO_utils&, bool&);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping

};

class Iterator_legacy_1st_order_upwind : public Iterator_legacy {

public:
    Iterator_legacy_1st_order_upwind(InputParams&, Grid&, Source&, IO_utils&, bool, bool, bool);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping
};

class Iterator_legacy_1st_order_tele : public Iterator_legacy_tele {

public:
    Iterator_legacy_1st_order_tele(InputParams&, Grid&, Source&, IO_utils&, bool, bool, bool);
    //void run_iteration_forward(InputParams&, Grid&, IO_utils&, bool&);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping

};

class Iterator_legacy_3rd_order_tele : public Iterator_legacy_tele {

public:
    Iterator_legacy_3rd_order_tele(InputParams&, Grid&, Source&, IO_utils&, bool, bool, bool);
    //void run_iteration_forward(InputParams&, Grid&, IO_utils&, bool&);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping

};

class Iterator_legacy_1st_order_upwind_tele : public Iterator_legacy_tele {

public:
    Iterator_legacy_1st_order_upwind_tele(InputParams&, Grid&, Source&, IO_utils&, bool, bool, bool);
    //void run_iteration_forward(InputParams&, Grid&, IO_utils&, bool&);
private:
    void do_sweep(int, Grid&, InputParams&) override ; // do sweeping

};

#endif // ITERATOR_LEGACY_H