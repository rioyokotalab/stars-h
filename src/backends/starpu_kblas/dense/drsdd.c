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
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <kblas.h>
#include "batch_rand.h"

void starsh_dense_dlrrsdd_starpu_kblas_cpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int maxrank;
    int oversample;
    double tol;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    double *singular_values;
    starpu_codelet_unpack_args(cl_arg, &maxrank, &oversample, &tol,
            &cublas_handles, &kblas_handles, &kblas_states, &singular_values);
    //printf("CODELET: %p, %p, %p\n", cublas_handles, kblas_handles,
    //        singular_values);
    double *D = (double *)STARPU_MATRIX_GET_PTR(buffer[0]);
    int nrows = STARPU_MATRIX_GET_NX(buffer[0]);
    int ncols = STARPU_MATRIX_GET_NY(buffer[0]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[3]);
    double *work = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int lwork = STARPU_VECTOR_GET_NX(buffer[4]);
    int *iwork = (int *)STARPU_VECTOR_GET_PTR(buffer[5]);
    starsh_dense_dlrrsdd(nrows, ncols, D, nrows, U, nrows, V, ncols, rank,
            maxrank, oversample, tol, work, lwork, iwork);
}

void starsh_dense_dlrrsdd_starpu_kblas_gpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int batch_size;
    int nb;
    int maxrank;
    int oversample;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    starpu_codelet_unpack_args(cl_arg, &batch_size, &nb, &maxrank, &oversample,
            &cublas_handles, &kblas_handles, &kblas_states);
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffer[0]);
    double *Dcopy = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *S = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    int mn = maxrank+oversample;
    if(mn > nb)
        mn = nb;
    int id = starpu_worker_get_id();
    kblasHandle_t khandle = kblas_handles[id];
    cublasHandle_t cuhandle = cublas_handles[id];
    kblasRandState_t state = kblas_states[id];
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Create copy of D, since kblas_rsvd spoils it
    cublasDcopy(cuhandle, batch_size*nb*nb, D, 1, Dcopy, 1);
    // Run randomized SVD, get left singular vectors and singular values
    //printf("BATCH SIZE ON GPU=%d\n", batch_size);
    kblasDrsvd_batch_strided(khandle, nb, nb, mn, D, nb, nb*nb, S, mn, state,
            batch_size);
}

void starsh_dense_dlrrsdd_starpu_kblas_cpu_S(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int size, mn, maxrank;
    double tol;
    starpu_codelet_unpack_args(cl_arg, &size, &mn, &maxrank, &tol);
    double *S = (double *)STARPU_VECTOR_GET_PTR(buffer[0]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[1]);
    *rank = starsh_dense_dsvfr(size, S, tol);
    if(*rank >= mn/2 || *rank > maxrank)
        *rank = -1;
}

void starsh_dense_dlrrsdd_starpu_kblas_gpu_dgemm(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int maxrank;
    int oversample;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    starpu_codelet_unpack_args(cl_arg, &maxrank, &oversample,
            &cublas_handles, &kblas_handles, &kblas_states);
    double *D = (double *)STARPU_MATRIX_GET_PTR(buffer[0]);
    int nrows = STARPU_MATRIX_GET_NX(buffer[0]);
    int ncols = STARPU_MATRIX_GET_NY(buffer[0]);
    double *Dcopy = (double *)STARPU_MATRIX_GET_PTR(buffer[1]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[2]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[3]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int mn = nrows < ncols ? nrows : ncols;
    int mn2 = maxrank+oversample;
    if(mn2 > mn)
        mn2 = mn;
    int id = starpu_worker_get_id();
    kblasHandle_t khandle = kblas_handles[id];
    cublasHandle_t cuhandle = cublas_handles[id];
    cudaStream_t stream = starpu_cuda_get_local_stream();
    int local_rank;
    cudaMemcpyAsync(&local_rank, rank, sizeof(int), cudaMemcpyDeviceToHost,
            stream);
    if(local_rank == -1)
        return;
    // Compute right factor of low-rank approximation, using given left
    // singular vectors
    double one = 1.0;
    double zero = 0.0;
    cublasDgemm(cuhandle, CUBLAS_OP_T, CUBLAS_OP_N, ncols, local_rank,
            nrows, &one, Dcopy, nrows, D, nrows, &zero, V, ncols);
    cublasDcopy(cuhandle, nrows*local_rank, D, 1, U, 1);
}

