#ifndef __STARSH_FUGAKU_GC_H__
#define __STARSH_FUGAKU_GC_H__

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"

typedef struct starsh_molecules
{
  STARSH_particles particles;
  STARSH_int N;
  STARSH_int ndim;
} STARSH_molecules;

int starsh_file_grid_read_kmeans(const char* file_name,
                                 STARSH_particles *particles,
                                 STARSH_int N,
                                 STARSH_int ndim);

int starsh_file_block_kernel(int nrows, int cols,
                             STARSH_int *irow,
                             STARSH_int *icol,
                             void* row_data,
                             void* col_data,
                             void *result, int ld);

double starsh_file_point_kernel(STARSH_int *irow,
                                STARSH_int *icol,
                                void *row_data,
                                void *col_data);

typedef struct starsh_laplace
{
    STARSH_particles particles;
    STARSH_int N;
    STARSH_int ndim;
    double PV;
    //!< Value to add to each diagonal element (for positive definiteness).
} STARSH_laplace;

void starsh_print_nice_things();

double starsh_laplace_point_kernel(STARSH_int *irow,
                                   STARSH_int *icol,
                                   void *row_data,
                                   void *col_data);

int starsh_normal_grid_generate(STARSH_particles** data, STARSH_int N,
    STARSH_int ndim);

void starsh_laplace_block_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);

int starsh_laplace_grid_free(STARSH_laplace **data);

int starsh_laplace_grid_generate(STARSH_laplace **data,
                                 STARSH_int N,
                                 STARSH_int ndim,
                                 double PV,
                                 enum STARSH_PARTICLES_PLACEMENT place);
#endif
