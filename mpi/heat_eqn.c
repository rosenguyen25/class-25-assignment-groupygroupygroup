
#include "heat_eqn.h"

#include <math.h>
#include <assert.h>

void
heat_eqn_calc_rhs(struct fld1d *f, struct fld1d *fprime, double dx, double kappa)
{
  fld1d_fill_ghosts_periodic(f);

  for (int i = fprime->ib; i < fprime->ie; i++) {
    F1(fprime, i) = kappa * (F1(f, i+1) - 2. * F1(f, i) + F1(f, i-1)) / (dx*dx);
  }
}
