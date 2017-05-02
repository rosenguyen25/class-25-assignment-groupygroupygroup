
#include <stdio.h>
#include <../linear_algebra/linear_algebra.h>
#include "params.h"

#define N_ITER (1L << 30)
#define NPAD 0

struct l {
  struct l *next;
  unsigned long pad[NPAD];
};

// ----------------------------------------------------------------------
// create_working_set_sequential
//
// create a working set that will be traversed simply in sequential order

static struct l *
create_working_set_sequential(int power)
{
  int working_set_size = 1 << power;
  int n_elems = working_set_size / sizeof(struct l);

  struct l *mem = calloc(n_elems, sizeof(struct l));

  for (int i = 0; i < n_elems; i++) {
    mem[i].next = &mem[(i+1) % n_elems];
  }

  return mem;
}

// ----------------------------------------------------------------------
// create_working_set_random
//
// create a working set that will be traversed in random order
//
// this is actually a difficult problem to do efficiently, the way we're doing it
// is not that great

static struct l *
create_working_set_random(int power)
{
  int working_set_size = 1 << power;
  int n_elems = working_set_size / sizeof(struct l);

  struct l *mem = calloc(n_elems, sizeof(struct l));

  // current element we're working on
  int cur = 0;
  for (int i = 0; i < n_elems  - 1; i++) {
    // look for a random next element that's not in the list yet
    int next;
    do {
      next = random() % n_elems;
    } while (mem[next].next);
    mem[cur].next = &mem[next];
    cur = next;
  }
  // finally, point back to the first element
  mem[cur].next = mem;

  return mem;
}

// need to return final p in order to avoid the loop getting optimized out

struct l *
traverse_working_set(struct l *mem)
{
  struct l *p = mem;
  for (int i = 0; i < N_ITER; i++) {
    p = p->next;
  }
  return p;
}

// ----------------------------------------------------------------------
// parameters

struct memory_param {
  bool random;
  const char *file;
  int min_power;        
  int max_power;
};

#define VAR(x) (void *)offsetof(struct memory_param, x)

static struct param pingpong_param_descr[] = {
  { "random"     , VAR(random)      , PARAM_BOOL(false)            },
  { "file"       , VAR(file)        , PARAM_STRING(NULL)           },
  { "min_power"  , VAR(min_power)   , PARAM_INT(10)                },
  { "max_power"  , VAR(max_power)   , PARAM_INT(20)                },
  {},
};

#undef VAR

// ----------------------------------------------------------------------
// main
//
// parse parameters, run test for working sets of increasing size

int
main(int argc, char **argv)
{
  // we don't really need MPI here at all, but the parameter parsing
  // wants it
  MPI_Init(&argc, &argv);

  params_init(argc, argv);

  struct memory_param prm;
  params_parse_cmdline(&prm, pingpong_param_descr, "memory", MPI_COMM_WORLD);
  params_print(&prm, pingpong_param_descr, "memory", MPI_COMM_WORLD);

  while ((1 << prm.min_power) < sizeof(struct l)) {
    prm.min_power++;
  }

  FILE *f = NULL;
  if (prm.file) {
    f = fopen(prm.file, "w");
    fprintf(f, "#power time\n");
    assert(f);
  }

  for (int power = prm.min_power; power <= prm.max_power; power++) {
    struct l *mem;
    if (prm.random) {
      mem = create_working_set_random(power);
    } else {
      mem = create_working_set_sequential(power);
    }
    
    double t_beg = WTime();
    struct l *p = traverse_working_set(mem);
    double t_end = WTime();
    printf("2^%d: %g ns / elem (total %g) final ptr %p\n",
	   power, (t_end - t_beg) / N_ITER, t_end - t_beg, p);
    if (f) {
      fprintf(f, "%d %g\n", power, (t_end - t_beg) / N_ITER);
      fflush(f);
    }
    
    free(mem);
  }

  fclose(f);
  MPI_Finalize();
  return 0;
}
