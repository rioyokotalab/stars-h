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

void starsh_dense_dlrrsdd_starpu_kblas_cpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int maxrank;
    int oversample;
    double tol;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    double *singular_values;
    starpu_codelet_unpack_args(cl_arg, &maxrank, &oversample, &tol,
            &cublas_handles, &kblas_handles, &singular_values);
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
    int maxrank;
    int oversample;
    double tol;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    double *singular_values;
    starpu_codelet_unpack_args(cl_arg, &maxrank, &oversample, &tol,
            &cublas_handles, &kblas_handles, &singular_values);
    double *D = (double *)STARPU_MATRIX_GET_PTR(buffer[0]);
    int nrows = STARPU_MATRIX_GET_NX(buffer[0]);
    int ncols = STARPU_MATRIX_GET_NY(buffer[0]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[3]);
    double *work = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int mn = nrows < ncols ? nrows : ncols;
    int mn2 = maxrank+oversample;
    if(mn2 > mn)
        mn2 = mn;
    int id = starpu_worker_get_id();
    kblasHandle_t khandle = kblas_handles[id];
    cublasHandle_t cuhandle = cublas_handles[id];
    double *host_S = singular_values+id*(maxrank+oversample);
    double *device_S = work+nrows*ncols;
    // Create copy of D, since kblas_rsvd spoils it
    cublasDcopy(cuhandle, nrows*ncols, D, 1, work, 1);
    // Run randomized SVD, get left singular vectors and singular values
    kblasDrsvd_batch_strided(khandle, nrows, ncols, mn2, work, nrows,
            nrows*ncols, device_S, nrows, 1);
    cudaMemcpy(host_S, device_S, mn2*sizeof(*host_S), cudaMemcpyDeviceToHost);
    //printf("SV:");
    //for(int i = 0; i < mn2; i++)
    //    printf(" %f", host_S[i]);
    //printf("\n");
    // Get rank, corresponding to given error tolerance
    int local_rank = starsh_dense_dsvfr(mn2, host_S, tol);
    if(local_rank < mn/2 && local_rank <= maxrank)
    {
        // Compute right factor of low-rank approximation, using given left
        // singular vectors
        double one = 1.0;
        double zero = 0.0;
        cublasDgemm(cuhandle, CUBLAS_OP_T, CUBLAS_OP_N, ncols, local_rank,
                nrows, &one, D, nrows, work, nrows, &zero, V, ncols);
        cublasDcopy(cuhandle, nrows*local_rank, work, 1, U, 1);
    }
    else
        local_rank = -1;
    cudaError_t err;
    // Write new rank back into device memory
    err = cudaMemcpy(rank, &local_rank, sizeof(local_rank),
            cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        printf("ERROR IN CUDAMEMCPY\n");
}

