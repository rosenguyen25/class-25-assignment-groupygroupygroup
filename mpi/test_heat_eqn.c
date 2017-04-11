
#include "fld1d.h"
#include "heat_eqn.h"

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

// ----------------------------------------------------------------------
// fl1d_write
//
// writes the array to disk

static void
fld1d_write(struct fld1d *x, int N, const char *filename)
{
  double dx = 2. * M_PI / N;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char s[100];
  snprintf(s, 100, "%s-%d.asc", filename, rank);
  FILE *f = fopen(s, "w");

  for (int i = x->ib; i < x->ie; i++) {
    double xx = i * dx;
    fprintf(f, "%g %g\n", xx, F1(x, i));
  }

  fclose(f);
}

// ----------------------------------------------------------------------
// main
//
// tests that the r.h.s (2nd derivative) for a cosine is -cosine.

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  const int N = 100;
  double dx = 2. * M_PI / N;

  struct fld1d *x = fld1d_create(-1, N+1);
  struct fld1d *rhs = fld1d_create(0, N);
  struct fld1d *neg_cos = fld1d_create(0, N);

  // put a discretized cosine into x,
  // and negative cosine into neg_cos, to be used to check the result.
  for (int i = 0; i < N; i++) {
    double xx = (i + .5) * dx;
    F1(x, i) = cos(xx);
    F1(neg_cos, i) = - cos(xx);
  }
  fld1d_write(x, N, "x");

  // calculate an example r.h.s., which is pretty much just the 2nd spatial derivative
  heat_eqn_calc_rhs(x, rhs, dx, 1., 0, N);
  fld1d_write(rhs, N, "rhs");

  // verify that the derivative we calculated is negative sine
  assert(fld1d_is_almost_equal(rhs, neg_cos, 1e-3));

  fld1d_destroy(x);
  fld1d_destroy(rhs);
  fld1d_destroy(neg_cos);

  MPI_Finalize();

  return 0;
}
