
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  if (argc != 2) {
    printf("Usage: %s <N>\nwhere <N> is the size of the message to exchange.\n",
	   argv[0]);
    exit(1);
  }
  int N = atoi(argv[1]);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(size == 2);

  double *buf_send = calloc(N, sizeof(double));
  double *buf_recv = calloc(N, sizeof(double));

  if (rank == 0) {
    MPI_Request req, req2;
    MPI_Isend(buf_send, N, MPI_DOUBLE, 1, 1234, MPI_COMM_WORLD, &req);
    MPI_Irecv(buf_recv, N, MPI_DOUBLE, 1, 1234, MPI_COMM_WORLD, &req2);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
  } else { // rank == 1
    MPI_Send(buf_send, N, MPI_DOUBLE, 0, 1234, MPI_COMM_WORLD);
    MPI_Recv(buf_recv, N, MPI_DOUBLE, 0, 1234, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
  }

  free(buf_send);
  free(buf_recv);

  MPI_Finalize();
  return 0;
}
