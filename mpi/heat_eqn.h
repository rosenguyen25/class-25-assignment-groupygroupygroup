
#ifndef HEAT_EQN_H
#define HEAT_EQN_H

#include "fld1d.h"

void fill_ghosts_periodic(struct fld1d *f);

void heat_eqn_calc_rhs(struct fld1d *f, struct fld1d *fprime, double dx, double kappa);

#endif
