
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
set_sine(struct fld1d *x, int N, int ib, int ie)
{
  double dx = 2. * M_PI / N;

  for (int i = ib; i < ie; i++) {
    double xx = i * dx;
    F1(x, i) = sin(xx+1);
  }
}

// ----------------------------------------------------------------------
// write
//
// writes the array to disk

static void
write(struct fld1d *x, int N, const char *filename)
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
// fill_ghosts
//
// fills the ghost cells at either end of x

static void
fill_ghosts(struct fld1d *x, int ib, int ie, int N)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    MPI_Send(&F1(x, ib  ), 1, MPI_DOUBLE, 1, 111, MPI_COMM_WORLD);
    MPI_Send(&F1(x, ie-1), 1, MPI_DOUBLE, 1, 111, MPI_COMM_WORLD);
  } else if (rank == 1) {
    MPI_Send(&F1(x, ie-1), 1, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD);
    MPI_Send(&F1(x, ib  ), 1, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    MPI_Recv(&F1(x, ib-1), 1, MPI_DOUBLE, 1, 111, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    MPI_Recv(&F1(x, ie  ), 1, MPI_DOUBLE, 1, 111, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Recv(&F1(x, ie  ), 1, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    MPI_Recv(&F1(x, ib-1), 1, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
  }
}

// ----------------------------------------------------------------------
// calc_derivative
//
// calculates a 2nd order centered difference approximation to the derivative

static void
calc_derivative(struct fld1d *d, struct fld1d *x, int N)
{
  fill_ghosts(x, d->ib, d->ie, N);

  double dx = 2. * M_PI / N;

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
  
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(size == 2);
  // we only handle the case where the number of points is evenly divisible by
  // the number of procs
  assert(N % size == 0);
  int n = N / size; // number of points on each proc
  int ib = rank * n, ie = (rank + 1) * n;
  
  struct fld1d *x = fld1d_create(ib-1, ie+1);
  struct fld1d *d = fld1d_create(ib  , ie  );

  set_sine(x, N, ib, ie);

  calc_derivative(d, x, N);
  write(x, N, "x");
  write(d, N, "d");

  fld1d_destroy(d);
  fld1d_destroy(x);

  MPI_Finalize();
  return 0;
}
