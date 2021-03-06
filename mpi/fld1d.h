
#ifndef FLD1D_H
#define FLD1D_H

#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

#define BOUNDSCHECK

// ----------------------------------------------------------------------
// struct fld1d

struct fld1d {
  double *vals;
  int ib; // starting index of local part
  int ie; // end index (+1) of local part
  int n_ghosts; // number of ghost points (outside of [ib,ie[ )
  int rank; // this process's rank, which determines our local range
  int rank_left, rank_right; // our neighbors (for ghost point communication)
};

#ifdef BOUNDSCHECK
#define F1(v, i) (*({							\
	assert((i) >= (v)->ib - (v)->n_ghosts);				\
	assert((i) < (v)->ie + (v)->n_ghosts);				\
	&((v)->vals[(i) - ((v)->ib - (v)->n_ghosts)]);			\
      })) 
#else
#define F1(v, i) ((v)->vals[(i) - ((v)->ib - (v)->n_ghosts)])
#endif

struct fld1d *fld1d_create(int N, int n_ghosts);
void fld1d_destroy(struct fld1d *v);
bool fld1d_is_almost_equal(struct fld1d *a, struct fld1d *b, double eps);
void fld1d_write(struct fld1d *x, const char *filename, double dx);
void fld1d_axpy(struct fld1d *y, double alpha, struct fld1d *x);
void fld1d_fill_ghosts_periodic(struct fld1d *x);

extern bool fld1d_option_write_single_file;

#endif
