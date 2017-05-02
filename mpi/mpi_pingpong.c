
#include "params.h"

#include <mpi.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define BUF_SIZE ((1 << 24) * sizeof(double))
#define NR_REPEAT (10)

// let's use a fixed buffer
static char buf[BUF_SIZE];

static inline void
ping_pong_one(int cnt, MPI_Datatype type, int rank)
{
  if (rank == 0) {
    MPI_Send(buf, cnt, type, 1, 0x1000, MPI_COMM_WORLD);
    MPI_Recv(buf, cnt, type, 1, 0x1001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(buf, cnt, type, 0, 0x1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(buf, cnt, type, 0, 0x1001, MPI_COMM_WORLD);
  }
}

static size_t
get_type_size(MPI_Datatype type)
{
  if (type == MPI_DOUBLE) {
    return sizeof(double);
  } else if (type == MPI_FLOAT) {
    return sizeof(float);
  } else {
    fprintf(stderr, "unknown MPI type!\n");
    abort();
  }
}

// ----------------------------------------------------------------------
// blocking ping pong test

void
ping_pong(int cnt, MPI_Datatype type, int repeat, FILE *file)
{
  size_t type_size = get_type_size(type);

  assert(cnt * type_size <= BUF_SIZE);
  memset(buf, 0, cnt * type_size);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // try to start synchronized, warm up
  MPI_Barrier(MPI_COMM_WORLD);
  ping_pong_one(cnt, type, rank);
  MPI_Barrier(MPI_COMM_WORLD);

  double t_min = 1e10;
  double t_avg = 0.;
  for (int i = 0; i < repeat / NR_REPEAT; i++) {
    double t_begin = MPI_Wtime();
    for (int ii = 0; ii < NR_REPEAT; ii++) {
      ping_pong_one(cnt, type, rank);
    }
    double t_elapsed = MPI_Wtime() - t_begin;
    t_avg += t_elapsed;
    t_min = fmin(t_min, t_elapsed);
  }
  // factor two is because we're measuring both ping and pong
  t_min /= (2*NR_REPEAT);
  t_avg /= (2*repeat);

  if (rank == 0) {
    fprintf(file, "%d %ld %g %g %g\n",
	    cnt, cnt * type_size, t_avg, t_min, cnt * type_size / t_min);
  }
}

// ----------------------------------------------------------------------
// non-blocking ping pong test

void
ping_pong_nb(int cnt, MPI_Datatype type, int repeat, FILE *file)
{
  size_t type_size = get_type_size(type);

  assert(cnt * type_size <= BUF_SIZE);
  memset(buf, 0, cnt * type_size);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double t_min = 1e10;
  double t_avg = 0.;
  for (int i = 0; i < repeat; i++) {
    MPI_Request request;
    MPI_Irecv(buf, cnt, type, 1-rank, 0x1000, MPI_COMM_WORLD, &request);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_begin = MPI_Wtime();
    if (rank == 0) {
      MPI_Send(buf, cnt, type, 1, 0x1000, MPI_COMM_WORLD);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    } else {
      MPI_Wait(&request, MPI_STATUS_IGNORE);
      MPI_Send(buf, cnt, type, 0, 0x1000, MPI_COMM_WORLD);
    }
    double t_elapsed = MPI_Wtime() - t_begin;
    t_avg += t_elapsed;
    t_min = fmin(t_min, t_elapsed);
  }
  // factor two is because we're measuring both ping and pong
  t_min /= 2;
  t_avg /= 2*repeat;

  if (rank == 0) {
    fprintf(file, "%d %ld %g %g %g\n",
	    cnt, cnt * type_size, t_avg, t_min, cnt * type_size / t_min);
  }
}

struct pingpong_param {
  const char *file;
  const char *type; // float or double
  const char *test; // ping_pong or ping_pong_nb
  int min_msgsize;        
  int max_msgsize;
  bool linear;      // linear: increase message size by adding step repeatedly
                    // logarithmic (linear == false): increase message size
                    // by multiplying with "factor"
  int repeat;       // number of repetitions
  int step;
  double factor;
};

#define VAR(x) (void *)offsetof(struct pingpong_param, x)

static struct param pingpong_param_descr[] = {
  { "file"       , VAR(file)        , PARAM_STRING(NULL)           },
  { "type"       , VAR(type)        , PARAM_STRING("double")       },
  { "test"       , VAR(test)        , PARAM_STRING("ping_pong")    },
  { "min_msgsize", VAR(min_msgsize) , PARAM_INT(1)                 },
  { "max_msgsize", VAR(max_msgsize) , PARAM_INT(1 << 24)           },
  { "linear"     , VAR(linear)      , PARAM_BOOL(false)            },
  { "repeat"     , VAR(repeat)      , PARAM_INT(10)                },
  { "step"       , VAR(step)        , PARAM_INT(1)                 },
  { "factor"     , VAR(factor)      , PARAM_DOUBLE(2.)             },
  {},
};

#undef VAR

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  // need to run on exactly two procs
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  assert(size == 2);

  params_init(argc, argv);

  struct pingpong_param par;
  params_parse_cmdline(&par, pingpong_param_descr, "pingpong", MPI_COMM_WORLD);
  params_print(&par, pingpong_param_descr, "pingpong", MPI_COMM_WORLD);

  // ensure par.repeat is divisible by NR_REPEAT
  par.repeat = (par.repeat + NR_REPEAT - 1) / NR_REPEAT * NR_REPEAT;

  MPI_Datatype type;
  if (strcasecmp(par.type, "double") == 0) {
    type = MPI_DOUBLE;
  } else if (strcasecmp(par.type, "float") == 0) {
    type = MPI_FLOAT;
  } else {
    fprintf(stderr, "unknown data type '%s'\n", par.type);
    abort();
  }

  void (*test)(int cnt, MPI_Datatype type, int repeat, FILE *file) = 0;
  if (strcasecmp(par.test, "ping_pong") == 0) {
    test = ping_pong;
  } else if (strcasecmp(par.test, "ping_pong_nb") == 0) {
    test = ping_pong_nb;
  } else {
    fprintf(stderr, "unknown test '%s'\n", par.test);
    abort();
  }

  FILE *file = NULL;
  if (rank == 0) {
    if (par.file) {
      file = fopen(par.file, "w");
    } else {
      file = stdout;
    }
    fprintf(file, "#cnt bytes time_avg time_min bandwidth\n");
  }

  if (par.linear) {
    // loop linearly
    for (int cnt = par.min_msgsize; cnt <= par.max_msgsize; cnt += par.step) {
      test(cnt, type, par.repeat, file);
    }
  } else {
    // loop on logarithmic scale
    for (int cnt = par.min_msgsize; cnt <= par.max_msgsize; cnt *= par.factor) {
      test(cnt, type, par.repeat, file);
    }
  }

  MPI_Finalize();
}
