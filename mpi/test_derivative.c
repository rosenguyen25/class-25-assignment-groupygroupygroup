
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
// fill_ghosts
//
// fills the ghost cells at either end of x

static void
fill_ghosts(struct fld1d *x)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int ib = x->ib, ie = x->ie;
  // the MPI ranks of our right and left neighbors
  int rank_right = (rank + 1) % size, rank_left = (rank + size - 1) % size;

  MPI_Send(&F1(x, ib  ), 1, MPI_DOUBLE, rank_left , 111, MPI_COMM_WORLD);
  MPI_Send(&F1(x, ie-1), 1, MPI_DOUBLE, rank_right, 111, MPI_COMM_WORLD);

  MPI_Recv(&F1(x, ie  ), 1, MPI_DOUBLE, rank_right, 111, MPI_COMM_WORLD,
	   MPI_STATUS_IGNORE);
  MPI_Recv(&F1(x, ib-1), 1, MPI_DOUBLE, rank_left , 111, MPI_COMM_WORLD,
	   MPI_STATUS_IGNORE);
}

// ----------------------------------------------------------------------
// calc_derivative
//
// calculates a 2nd order centered difference approximation to the derivative

static void
calc_derivative(struct fld1d *d, struct fld1d *x, double dx)
{
  fill_ghosts(x);

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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // we only handle the case where the number of points is evenly divisible by
  // the number of procs
  assert(N % size == 0);
  int n = N / size; // number of points on each proc
  int ib = rank * n, ie = (rank + 1) * n;

  double dx = L / N;
  
  struct fld1d *x = fld1d_create(ib, ie, 1);
  struct fld1d *d = fld1d_create(ib, ie, 0);

  set_sine(x, dx);

  calc_derivative(d, x, dx);
  fld1d_write(x, "x", dx);
  fld1d_write(d, "d", dx);

  fld1d_destroy(d);
  fld1d_destroy(x);

  MPI_Finalize();
  return 0;
}
