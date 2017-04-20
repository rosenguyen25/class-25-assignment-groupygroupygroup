
#include "fld1d.h"
#include "heat_eqn.h"

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#define sqr(a) ((a) * (a))

int
main(int argc, char **argv)
{
  const double L = 2. * M_PI;
  int N = 100;
  double max_time = 50.;
  double out_every = 1.;
  double kappa = .01;

  MPI_Init(&argc, &argv);

  // parse options
  opterr = 0;
  int c;
  while ((c = getopt (argc, argv, "sm:N:o:k:")) != -1) {
    switch (c) {
    case 's': fld1d_option_write_single_file = true; break;
    case 'N': N = atoi(optarg); break;
    case 'm': max_time = atof(optarg); break;
    case 'o': out_every = atof(optarg); break;
    case 'k': kappa = atof(optarg); break;
    case '?':
      if (optopt == 'N' || optopt == 'm' || optopt == 'o' || optopt == 'k') {
        fprintf (stderr, "Option -%c requires an argument.\n", optopt);
      } else {
        fprintf (stderr, "Unknown option '-%c'.\n", optopt);
      }
      return 1;
    default:
      abort();
    }
  }

  double dx = L / N;
  double dt = .5 * dx * dx / kappa; // pick dt to satisfy CFL condition

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    printf("Parameters:\n");
    printf("  -s                use single file output (%s)\n",
	   fld1d_option_write_single_file ? "true" : "false");
    printf("  -N <N>            number of grid points (%d)\n", N);
    printf("  -m <max_time>     final time to run the simulation to (%g)\n", max_time);
    printf("  -o <out_every>    output every so often; 0 for no output (%g)\n", out_every);
    printf("  -k <kappa>        heat conductivity (%g)\n", kappa);
    
    printf("\n");
    printf("Grid spacing dx  = %g\n", dx);
    printf("Timestep     dt  = %g\n", dt);
    printf("CFL number   CFL = %g\n", kappa * dt / sqr(dx));
  }
  
  struct fld1d *x = fld1d_create(N, 1);
  struct fld1d *rhs = fld1d_create(N, 0);

  // set up initial condition
  for (int i = x->ib; i < x->ie; i++) {
    double xx = (i + .5) * dx;
    F1(x, i) = exp(-sqr(xx - M_PI) / sqr(.5));
  }

  // time integration loop
  double tbeg = MPI_Wtime();
  double t_rhs = 0.;
  double t_axpy = 0.;
  double t_out = 0.;
  int c_rhs = 0;
  int c_axpy = 0;
  int c_out = 0;
  
  int n = 0; // step counter
  int out_cnt = 0; // counter for output
  double time_next_out = 0.; // time for next output
  double time = 0.; // current time
  while (time <= max_time) {
    // write out current solution every so often
    if (out_every && time >= time_next_out) {
      t_out -= MPI_Wtime();
      char fname[100];
      snprintf(fname, 100, "x%d", out_cnt++);
      //printf("step %d time %g: Writing output \"%s\"\n", n, time, fname);
      fld1d_write(x, fname, dx);
      while (time_next_out <= time) {
        time_next_out += out_every;
      }
      t_out += MPI_Wtime();
      c_out++;
    }

    // A simple forward Euler step x^{n+1} = x^{n} + dt * rhs(x^n)
    // works fine for integrating this equation:

    // calculate rhs first
    t_rhs -= MPI_Wtime();
    heat_eqn_calc_rhs(x, rhs, dx, kappa);
    t_rhs += MPI_Wtime();
    c_rhs++;

    // then update solution: x += dt * rhs
    t_axpy -= MPI_Wtime();
    fld1d_axpy(x, dt, rhs);
    t_axpy += MPI_Wtime();
    c_axpy++;
    
    time += dt;
    n++;
  }

  double tend = MPI_Wtime();

  if (rank == 0) {
    printf("Integrated %d steps and wrote %d output files. Wall time = %g s\n",
	   n, out_cnt, tend - tbeg);
    printf("           TOTAL (s)      COUNT    AVG (us)\n");
    printf("rhs     %12.6f %10d %12.6g\n", t_rhs, c_rhs, t_rhs * 1e6 / c_rhs);
    printf("axpy    %12.6f %10d %12.6g\n", t_axpy, c_axpy, t_axpy * 1e6 / c_axpy);
    printf("out     %12.6f %10d %12.6g\n", t_out, c_out, t_out * 1e6 / c_out);
  }
  
  fld1d_destroy(x);
  fld1d_destroy(rhs);

  MPI_Finalize();
  return 0;
}
