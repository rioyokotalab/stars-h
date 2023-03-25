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



typedef struct starsh_laplace
{
    STARSH_particles particles;
    STARSH_int N;
    STARSH_int ndim;
    double PV;
    //!< Value to add to each diagonal element (for positive definiteness).
} STARSH_laplace;

typedef struct STARSH_matern {
  STARSH_particles particles;
  STARSH_int N;
  STARSH_int ndim;
  double sigma, nu, smoothness;
} STARSH_matern;

typedef struct STARSH_yukawa {
  STARSH_particles particles;
  STARSH_int N;
  STARSH_int ndim;
  double alpha, singularity;
} STARSH_yukawa;

/* yukawa */
int
starsh_yukawa_block_kernel(int nrows, int cols,
                           STARSH_int *irow,
                           STARSH_int *icol,
                           void* row_data,
                           void* col_data,
                           void *result, int ld);

double
starsh_yukawa_point_kernel(STARSH_int *irow,
                           STARSH_int *icol,
                           void *row_data,
                           void *col_data);

int
starsh_yukawa_grid_generate(STARSH_yukawa **data, STARSH_int N,
                            STARSH_int ndim, double alpha, double singularity,
                            enum STARSH_PARTICLES_PLACEMENT place);

void starsh_print_nice_things();

/* laplace */

double starsh_laplace_point_kernel(STARSH_int *irow,
                                   STARSH_int *icol,
                                   void *row_data,
                                   void *col_data);

void starsh_laplace_block_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);

int starsh_laplace_grid_generate(STARSH_laplace **data,
                                 STARSH_int N,
                                 STARSH_int ndim,
                                 double PV,
                                 enum STARSH_PARTICLES_PLACEMENT place);
int starsh_laplace_grid_free(STARSH_laplace **data);

/* matern */
int
starsh_matern_grid_generate(STARSH_matern **data, STARSH_int N,
                            STARSH_int ndim, double sigma, double nu,
                            double smoothness, enum STARSH_PARTICLES_PLACEMENT place);

double
starsh_matern_point_kernel(STARSH_int *irow,
                           STARSH_int *icol,
                           void *row_data,
                           void *col_data);

void
starsh_matern_block_kernel(int nrows, int ncols, STARSH_int *irow,
                           STARSH_int *icol, void *row_data, void *col_data, void *result,
                           int ld);


#endif
