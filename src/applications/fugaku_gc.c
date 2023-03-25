#include <stdio.h>

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-fugaku_gc.h"

#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>

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

int starsh_generate_2d_grid(STARSH_particles**data, STARSH_int count) {
  double *point;
  STARSH_MALLOC(point, ndim*count);

  int side = sqrt(count);
  double space = 1.0 / side;
  for (int i = 0; i < side; ++i) {
    for (int j = 0; j < side; ++j) {
      point[(i + j * side)] = i * space; /* x co-ordinate */
      point[count + (i + j * side)] = j * space; /* y co-ordinate */
    }
  }

  STARSH_MALLOC(*data, 1);
  (*data)->count = count;
  (*data)->ndim = 2;
  (*data)->point = point;

  return STARSH_SUCCESS;
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
    info = starsh_generate_2d_grid(&particles, N);
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

void
starsh_matern_block_kernel(int nrows, int ncols, STARSH_int *irow,
                           STARSH_int *icol, void *row_data, void *col_data, void *result,
                           int ld) {
  double* buffer = (double*)result;
  STARSH_matern *data1 = row_data;
  STARSH_matern *data2 = col_data;

  STARSH_int N = data1->N;
  double sigma = data1->sigma;
  double nu = data1->nu;
  double smoothness = data1->smoothness;
  STARSH_int ndim = data1->ndim;
  double out;
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
      double dist = sqrt(rij);

      double expr = 0.0;
      double con = 0.0;
      double sigma_square = sigma*sigma;

      con = pow(2, (smoothness - 1)) * gsl_sf_gamma(smoothness);
      con = 1.0 / con;
      con = sigma_square * con;

      if (dist != 0) {
        expr = dist / nu;
        out =  con * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
      }
      else {
        out = sigma_square;
      }

      buffer[i + j * ld] = out;
    }
  }
}

double starsh_matern_point_kernel(STARSH_int *irow,
                                  STARSH_int *icol,
                                  void *row_data,
                                  void *col_data) {
  STARSH_matern *data1 = row_data;
  STARSH_matern *data2 = col_data;

  STARSH_int N = data1->N;
  double sigma = data1->sigma;
  double nu = data1->nu;
  double smoothness = data1->smoothness;
  STARSH_int ndim = data1->ndim;
  double out;
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
  double dist = sqrt(rij);

  double expr = 0.0;
  double con = 0.0;
  double sigma_square = sigma*sigma;

  con = pow(2, (smoothness - 1)) * gsl_sf_gamma(smoothness);
  con = 1.0 / con;
  con = sigma_square * con;

  if (dist != 0) {
    expr = dist / nu;
    out =  con * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
  }
  else {
    out = sigma_square;
  }

  return out;
}

int
starsh_matern_grid_generate(STARSH_matern **data, STARSH_int N,
                            STARSH_int ndim, double sigma, double nu,
                            double smoothness, enum STARSH_PARTICLES_PLACEMENT place) {
  if (data == NULL) {
    STARSH_ERROR("Invalid value of data.");
    return STARSH_WRONG_PARAMETER;
  }
  if (N <= 0) {
    STARSH_ERROR("Invalid value of N.\n");
    return STARSH_WRONG_PARAMETER;
  }

  STARSH_particles *particles;
  int info = starsh_generate_2d_grid(&particles, N);
  if(info != STARSH_SUCCESS)
    {
      fprintf(stderr, "INFO=%d\n", info);
      return info;
    }
  STARSH_MALLOC(*data, 1);
  (*data)->particles = *particles;
  free(particles);
  (*data)->N = N;
  (*data)->ndim = ndim;
  (*data)->sigma = sigma;
  (*data)->nu = nu;
  (*data)->smoothness = smoothness;

  return STARSH_SUCCESS;
}

int starsh_yukawa_block_kernel(int nrows, int ncols, STARSH_int *irow,
                             STARSH_int *icol, void* row_data, void* col_data,
                             void *result, int ld) {
  STARSH_yukawa *data1 = row_data;
  STARSH_yukawa *data2 = col_data;

  double alpha = data1->alpha;
  double singularity = data1->singularity;

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
      double out = exp(-rij * alpha) / (singularity + rij);
      buffer[i + j * ld] = out;
    }
  }

  return STARSH_SUCCESS;
}

double starsh_yukawa_point_kernel(STARSH_int *irow,
                                STARSH_int *icol,
                                void *row_data,
                                void *col_data) {
  STARSH_yukawa *data1 = row_data;
  STARSH_yukawa *data2 = col_data;

  double alpha = data1->alpha;
  double singularity = data1->singularity;

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
  r = sqrt(r);                  /* distance */

  return exp(-r * alpha) / (singularity + r);
}

int
starsh_yukawa_grid_generate(STARSH_yukawa **data, STARSH_int N,
                            STARSH_int ndim, double alpha, double singularity,
                            enum STARSH_PARTICLES_PLACEMENT place) {
  if (data == NULL) {
    STARSH_ERROR("Invalid value of data.");
    return STARSH_WRONG_PARAMETER;
  }
  if (N <= 0) {
    STARSH_ERROR("Invalid value of N.\n");
    return STARSH_WRONG_PARAMETER;
  }

  STARSH_particles *particles;
  int info = starsh_generate_2d_grid(&particles, N);
  if(info != STARSH_SUCCESS)
    {
      fprintf(stderr, "INFO=%d\n", info);
      return info;
    }
  STARSH_MALLOC(*data, 1);
  (*data)->particles = *particles;
  free(particles);
  (*data)->N = N;
  (*data)->ndim = ndim;
  (*data)->alpha = alpha;
  (*data)->singularity = singularity;

  return STARSH_SUCCESS;
}
