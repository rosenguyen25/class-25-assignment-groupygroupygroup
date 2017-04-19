
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi_debug.h"

#include "fld1d.h"

// ----------------------------------------------------------------------
// set_sine
//
// initializes the array x with the sine function

static void
set_sine(struct fld1d *x, double dx)
{
  for (int i = x->ib; i < x->ie; i++) {
    double xx = (i + .5) * dx;
    F1(x, i) = sin(xx+1);
  }
}

// ----------------------------------------------------------------------
// calc_derivative
//
// calculates a 2nd order centered difference approximation to the derivative

static void
calc_derivative(struct fld1d *d, struct fld1d *x, double dx)
{
  fld1d_fill_ghosts_periodic(x);

  for (int i = d->ib; i < d->ie; i++) {
    F1(d, i) = (F1(x, i+1) - F1(x, i-1)) / (2. * dx);
  }
}

// ----------------------------------------------------------------------
// main

int
main(int argc, char **argv)
{
  const int N = 50;
  const double L = 2. * M_PI;
  
  MPI_Init(&argc, &argv);

  double dx = L / N;
  
  struct fld1d *x = fld1d_create(N, 1);
  struct fld1d *d = fld1d_create(N, 0);

  set_sine(x, dx);

  calc_derivative(d, x, dx);
  fld1d_write(x, "x", dx);
  fld1d_write(d, "d", dx);

  fld1d_destroy(d);
  fld1d_destroy(x);

  MPI_Finalize();
  return 0;
}
