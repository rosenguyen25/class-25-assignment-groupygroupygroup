
#include "fld1d.h"
#include "heat_eqn.h"

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

// ----------------------------------------------------------------------
// main
//
// tests that the r.h.s (2nd derivative) for a cosine is -cosine.

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  const int N = 100;
  const double L = 2. * M_PI;

  double dx = L / N;

  struct fld1d *x = fld1d_create(0, N, 1);
  struct fld1d *rhs = fld1d_create(0, N, 0);
  struct fld1d *neg_cos = fld1d_create(0, N, 0);

  // put a discretized cosine into x,
  // and negative cosine into neg_cos, to be used to check the result.
  for (int i = 0; i < N; i++) {
    double xx = (i + .5) * dx;
    F1(x, i) = cos(xx);
    F1(neg_cos, i) = - cos(xx);
  }
  fld1d_write(x, "x", dx);

  // calculate an example r.h.s., which is pretty much just the 2nd spatial derivative
  heat_eqn_calc_rhs(x, rhs, dx, 1.);
  fld1d_write(rhs, "rhs", dx);

  // verify that the derivative we calculated is negative sine
  assert(fld1d_is_almost_equal(rhs, neg_cos, 1e-3));

  fld1d_destroy(x);
  fld1d_destroy(rhs);
  fld1d_destroy(neg_cos);

  MPI_Finalize();

  return 0;
}
