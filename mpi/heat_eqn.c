
#include "heat_eqn.h"

#include <math.h>
#include <assert.h>

void
fill_ghosts_periodic(struct fld1d *f, int ib, int ie)
{
  // get global number of cells in the domain
  int N = ie - ib;

  F1(f, -1) = F1(f, N-1);
  F1(f,  N) = F1(f, 0  );
}

void
heat_eqn_calc_rhs(struct fld1d *f, struct fld1d *fprime, double dx, double kappa,
		  int ib, int ie)
{
  fill_ghosts_periodic(f, ib, ie);

  for (int i = ib; i < ie; i++) {
    F1(fprime, i) = kappa * (F1(f, i+1) - 2. * F1(f, i) + F1(f, i-1)) / (dx*dx);
  }
}
