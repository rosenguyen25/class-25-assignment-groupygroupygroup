
ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS = \
	hello_mpi \
	hello_mpi_2 \
	hello_mpi_3 \
	hello_mpi_4 \
	message_mpi \
	message_mpi_2 \
	test_integrate \
	test_derivative \
	test_heat_eqn \
	heat_eqn_solve \
	mpi_dl_test \
	mpi_pingpong \
	memory

test_heat_eqn_SOURCES = \
	test_heat_eqn.c \
	heat_eqn.c

heat_eqn_solve_SOURCES = \
	heat_eqn_solve.c \
	heat_eqn.c

mpi_pingpong_SOURCES = \
	mpi_pingpong.c \
	params.c

memory_SOURCES = \
	memory.c \
	params.c

LDADD = libfld1d.a

lib_LIBRARIES = libfld1d.a

libfld1d_a_SOURCES = \
	fld1d.c \
	fld1d.h

TESTS = test_heat_eqn

