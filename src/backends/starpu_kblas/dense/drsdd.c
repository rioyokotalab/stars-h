/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/dense/drsdd.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu-kblas.h"

void starsh_dense_dlrrsdd_starpu_kblas_cpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    STARSH_blrf *F;
    int maxrank;
    int oversample;
    double tol;
    starpu_codelet_unpack_args(cl_arg, &maxrank, &oversample, &tol);
    double *D = (double *)STARPU_MATRIX_GET_PTR(buffer[0]);
    int nrows = STARPU_MATRIX_GET_NX(buffer[0]);
    int ncols = STARPU_MATRIX_GET_NY(buffer[0]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    int *rank = (int *)STARPU_VARIABLE_GET_PTR(buffer[3]);
    double *work = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int lwork = STARPU_VECTOR_GET_NX(buffer[4]);
    int *iwork = (int *)STARPU_VECTOR_GET_PTR(buffer[5]);
    starsh_dense_dlrrsdd(nrows, ncols, D, nrows, U, nrows, V, ncols, rank,
            maxrank, oversample, tol, work, lwork, iwork);
}
