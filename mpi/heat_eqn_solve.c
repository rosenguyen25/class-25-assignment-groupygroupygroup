
#include "fld1d.h"
#include "heat_eqn.h"

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <mpi.h>

#define sqr(a) ((a) * (a))

int
main(int argc, char **argv)
{
  const int N = 100;
  const double L = 2. * M_PI;
  const int n_timesteps = 200;
  const int out_every = 10;
  const double kappa = .01; 

  MPI_Init(&argc, &argv);

  fld1d_option_write_single_file = true;

  double dx = L / N;
  double dt = .5 * dx * dx / kappa; // pick dt to satisfy CFL condition

  struct fld1d *x = fld1d_create(N, 1);
  struct fld1d *rhs = fld1d_create(N, 0);

  // set up initial condition
  for (int i = x->ib; i < x->ie; i++) {
    double xx = (i + .5) * dx;
    F1(x, i) = exp(-sqr(xx - M_PI) / sqr(.5));
  }

  for (int n = 0; n < n_timesteps; n++) {
    // write out current solution every so many steps
    if (n % out_every == 0) {
      char fname[10];
      sprintf(fname, "x%d", n);
      fld1d_write(x, fname, dx);
    }

    // A simple forward Euler step x^{n+1} = x^{n} + dt * rhs(x^n)
    // works fine for integrating this equation:

    // calculate rhs first
    heat_eqn_calc_rhs(x, rhs, dx, kappa);

    // then update solution: x += dt * rhs
    fld1d_axpy(x, dt, rhs);
  }

  fld1d_destroy(x);
  fld1d_destroy(rhs);

  MPI_Finalize();
  return 0;
}
