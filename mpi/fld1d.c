
#include "fld1d.h"

#include <math.h>
#include <stdio.h>
#include <mpi.h>

// ----------------------------------------------------------------------
// fld1d_create
//
// allocates and initializes a fld1d, setting all elements to zero

struct fld1d *
fld1d_create(int N, int n_ghosts)
{
  // allocate fld1d struct
  struct fld1d *v = calloc(1, sizeof(*v));

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // we only handle the case where the number of points is evenly divisible by
  // the number of procs
  assert(N % size == 0);
  int n = N / size; // number of points on each proc
  v->ib = rank * n;
  v->ie = (rank + 1) * n;
  v->n_ghosts = n_ghosts;
  v->vals = calloc(n + 2 * n_ghosts, sizeof(v->vals[0]));

  // our MPI rank
  v->rank = rank;
  // the MPI ranks of our right and left neighbors
  v->rank_right = (rank + 1) % size;
  v->rank_left  = (rank + size - 1) % size;
  
  return v;
}

// ----------------------------------------------------------------------
// fld1d_destroy
//
// this function is called when we're done using a fld1d

void
fld1d_destroy(struct fld1d *v)
{
  free(v->vals);
  // not strictly needed, but will make sure that we crash if we
  // access the fld1d after we called fld1d_destroy()
  v->vals = NULL;
  free(v);
}

// ----------------------------------------------------------------------
// fld1d_is_almost_equal
//
// this function is checks whether a and b are equal up to a given threshold

bool
fld1d_is_almost_equal(struct fld1d *a, struct fld1d *b, double eps)
{
  // make sure the two fields cover the same index range
  assert(a->ib == b->ib && a->ie == b->ie);

  for (int i = a->ib; i < a->ie; i++) {
    if (fabs(F1(a, i) - F1(b, i)) > eps) {
      return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------
// fld1d_write
//
// writes the array to disk

void
fld1d_write(struct fld1d *x, const char *filename, double dx)
{
  char s[100];
  snprintf(s, 100, "%s-%d.asc", filename, x->rank);
  FILE *f = fopen(s, "w");

  for (int i = x->ib - x->n_ghosts; i < x->ie + x->n_ghosts; i++) {
    double xx = (i + .5) * dx;
    fprintf(f, "%g %g\n", xx, F1(x, i));
  }

  fclose(f);
}

// ----------------------------------------------------------------------
// fld1d_axpy
//
// calculate y = a*x + y

void
fld1d_axpy(struct fld1d *y, double alpha, struct fld1d *x)
{
  // make sure the two fields cover the same index range
  assert(x->ib == y->ib && x->ie == y->ie);
  
  for (int i = y->ib; i < y->ie; i++) {
    F1(y, i) += alpha * F1(x, i);
  }
}

// ----------------------------------------------------------------------
// fld1d_fill_ghosts_periodic
//
// fills the ghost cells at either end of x

void
fld1d_fill_ghosts_periodic(struct fld1d *x)
{
  int ib = x->ib, ie = x->ie;

  MPI_Send(&F1(x, ib  ), 1, MPI_DOUBLE, x->rank_left , 111, MPI_COMM_WORLD);
  MPI_Send(&F1(x, ie-1), 1, MPI_DOUBLE, x->rank_right, 111, MPI_COMM_WORLD);

  MPI_Recv(&F1(x, ie  ), 1, MPI_DOUBLE, x->rank_right, 111, MPI_COMM_WORLD,
	   MPI_STATUS_IGNORE);
  MPI_Recv(&F1(x, ib-1), 1, MPI_DOUBLE, x->rank_left , 111, MPI_COMM_WORLD,
	   MPI_STATUS_IGNORE);
}

