#include <stdio.h>

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-fugaku_gc.h"

int starsh_file_grid_read_kmeans(const char* file_name,
                                 STARSH_particles *particles,
                                 STARSH_int N,
                                 STARSH_int ndim) {
  FILE *fp;
  int info;
  size_t nelem = N * ndim;

  particles->point = (double*)malloc(nelem * sizeof(double));
  particles->ndim = ndim;
  particles->count = N;

  STARSH_int *kmeans_index = (STARSH_int*)malloc(N * sizeof(STARSH_int));

  fp = fopen(file_name, "r");
  if (fp == NULL) {
    fprintf(stderr, "could not read file.\n");
    abort();
  }

  STARSH_int one_pcent = N / 100;
  /* printf("starsh-> reading file.\n"); */

  for (STARSH_int i = 0; i < N; ++i) {
    double x, y, z;
    fscanf(fp, "%lf %lf %lf %lld", &x, &y, &z, &kmeans_index[i]);
    if (feof(fp) && i < N) {
      fprintf(stderr, "reached end of file after %ld.\n", i);
      exit(2);
    }

    /* if (i % one_pcent == 0) { */
    /*   printf("file read %lf %% done.\n", ((double)i / N) * 100); */
    /* } */

    particles->point[i] = x;
    particles->point[i + N] = y;
    particles->point[i + 2 * N] = z;
  }

  fclose(fp);

  return STARSH_SUCCESS;
}

int starsh_yukawa_block_kernel(int nrows, int ncols, STARSH_int *irow,
                             STARSH_int *icol, void* row_data, void* col_data,
                             void *result, int ld) {
  STARSH_laplace *data1 = row_data;
  STARSH_laplace *data2 = col_data;

  STARSH_int N = data1->N;
  double *buffer = result;
  STARSH_int ndim = data1->ndim;

  double *x1[ndim], *x2[ndim];

  x1[0] = data1->particles.point;
  x2[0] = data2->particles.point;
  for (STARSH_int k = 1; k < ndim; ++k) {
    x1[k] = x1[0] + k * data1->particles.count;
    x2[k] = x2[0] + k * data2->particles.count;
  }


  for (STARSH_int i = 0; i < nrows; ++i) {
    for (STARSH_int j = 0; j < ncols; ++j) {
      double rij = 0;
      for (STARSH_int k = 0; k < ndim; ++k) {
        rij += pow(x1[k][irow[i]] - x2[k][icol[j]], 2);
      }
      rij = sqrt(rij);
      double out = exp(-rij) / (1e-8 + rij);
      buffer[i + j * ld] = out;
    }
  }

  return STARSH_SUCCESS;
}

double starsh_yukawa_point_kernel(STARSH_int *irow,
                                STARSH_int *icol,
                                void *row_data,
                                void *col_data) {
  STARSH_molecules *data1 = row_data;
  STARSH_molecules *data2 = col_data;

  STARSH_int N = data1->N;
  STARSH_int ndim = data1->ndim;

  double *x1[ndim], *x2[ndim];

  x1[0] = data1->particles.point;
  x2[0] = data2->particles.point;
  for (STARSH_int k = 1; k < ndim; ++k) {
    x1[k] = x1[0] + k * data1->particles.count;
    x2[k] = x2[0] + k * data2->particles.count;
  }

  double r = 0;
  for (STARSH_int k = 0; k < ndim; ++k) {
    r += pow(x1[k][irow[0]] - x2[k][icol[0]], 2);
  }
  r = sqrt(r);

  /* v = exp(-r) / (1.e-8 + r) */
  return exp(-r) / (1e-8 + r);
}

void starsh_print_nice_things() {
  printf("nice things.\n");
}

double starsh_laplace_point_kernel(STARSH_int *irow,
                                   STARSH_int *icol,
                                   void *row_data,
                                   void *col_data)
{
  STARSH_laplace *data1 = row_data;
  STARSH_laplace *data2 = col_data;

  STARSH_int N = data1->N;
  double PV = data1->PV;
  STARSH_int ndim = data1->ndim;

  double *x1[ndim], *x2[ndim];

  x1[0] = data1->particles.point;
  x2[0] = data2->particles.point;
  for (STARSH_int k = 1; k < ndim; ++k) {
    x1[k] = x1[0] + k * data1->particles.count;
    x2[k] = x2[0] + k * data2->particles.count;
  }

  double rij = 0;
  for (STARSH_int k = 0; k < ndim; ++k) {
    rij += pow(x1[k][irow[0]] - x2[k][icol[0]], 2);
  }
  double out = 1 / (sqrt(rij) + PV);

  return out;
}

void starsh_laplace_block_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld)
//! The only kernel for @ref STARSH_laplace object.
/*! @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (@ref STARSH_laplace object).
 * @param[in] col_data: Pointer to physical data (@ref STARSH_laplace object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * @ingroup app-laplace
 * */
{
    STARSH_laplace *data1 = row_data;
    STARSH_laplace *data2 = col_data;

    STARSH_int N = data1->N;
    double PV = data1->PV;
    double *buffer = result;
    STARSH_int ndim = data1->ndim;

    double *x1[ndim], *x2[ndim];

    x1[0] = data1->particles.point;
    x2[0] = data2->particles.point;
    for (STARSH_int k = 1; k < ndim; ++k) {
        x1[k] = x1[0] + k * data1->particles.count;
        x2[k] = x2[0] + k * data2->particles.count;
    }

    for (STARSH_int i = 0; i < nrows; ++i) {
      for (STARSH_int j = 0; j < ncols; ++j) {
        double rij = 0;
        for (STARSH_int k = 0; k < ndim; ++k) {
          rij += pow(x1[k][irow[i]] - x2[k][icol[j]], 2);
        }
        double out = 1 / (sqrt(rij) + PV);
        buffer[i + j * ld] = out;
      }
    }

}

int starsh_normal_grid_generate(STARSH_particles** data, STARSH_int N,
    STARSH_int ndim) {
    STARSH_MALLOC(*data, 1);
    (*data)->count= N;
    (*data)->ndim = ndim;

    double *point;
    assert(ndim == 3);
    STARSH_MALLOC(point, (*data)->count * ndim);
    srand(1);

    for (STARSH_int i = 0; i < N; ++i) {
        point[i] = (double)i / N;
        point[i + N] = (double)i / N;
        point[i + 2 * N] = (double)i / N;
    }

    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return STARSH_SUCCESS;
}

int starsh_laplace_grid_free(STARSH_laplace **data) {
    starsh_particles_free(&(*data)->particles);
}

int starsh_laplace_grid_generate(STARSH_laplace **data, STARSH_int N,
                                 STARSH_int ndim,
                                 double PV,
                                 enum STARSH_PARTICLES_PLACEMENT place) {

    if (data == NULL) {
        STARSH_ERROR("Invalid value of data.");
        return STARSH_WRONG_PARAMETER;
    }
    if (N <= 0) {
        STARSH_ERROR("Invalid value of N.\n");
        return STARSH_WRONG_PARAMETER;
    }
    if (PV > 1) {
        STARSH_ERROR("Invalid value of PV.\n");
        return STARSH_WRONG_PARAMETER;
    }

    int info;
    STARSH_particles *particles;
    info = starsh_particles_generate(&particles, N, ndim, place, 0);
    /* info = starsh_normal_grid_generate(&particles, N, ndim); */
    if(info != STARSH_SUCCESS)
    {
        fprintf(stderr, "INFO=%d\n", info);
        return info;
    }
    STARSH_MALLOC(*data, 1);
    (*data)->particles = *particles;
    free(particles);
    (*data)->N = N;
    (*data)->PV = PV;
    (*data)->ndim = ndim;

    return STARSH_SUCCESS;
}
