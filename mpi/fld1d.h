
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
};

#ifdef BOUNDSCHECK
#define F1(v, i) (*({							\
	assert((i) >= (v)->ib && (i) < (v)->ie);			\
	&((v)->vals[(i) - (v)->ib]);					\
      })) 
#else
#define F1(v, i) ((v)->vals[(i) - (v)->ib])
#endif

struct fld1d *fld1d_create(int ib, int ie, int n_ghosts);
void fld1d_destroy(struct fld1d *v);
bool fld1d_is_almost_equal(struct fld1d *a, struct fld1d *b, double eps);
void fld1d_write(struct fld1d *x, int N, const char *filename);
void fld1d_axpy(struct fld1d *y, double alpha, struct fld1d *x, int ib, int ie);

#endif
