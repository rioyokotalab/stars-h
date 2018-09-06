/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/dense/kernel.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu-kblas.h"
#include <omp.h>

void starsh_dense_kernel_starpu_kblas_cpu(void *buffers[], void *cl_arg)
//! STARPU kernel for matrix kernel.
{
    //printf("START\n");
    double time0 = omp_get_wtime();
    STARSH_blrf *F;
    STARSH_int batch_size;
    starpu_codelet_unpack_args(cl_arg, &F, &batch_size);
    //printf("F=%p bs=%d\n", F, batch_size);
    //printf("START2\n");
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster, *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    STARSH_int *ind = (STARSH_int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    //printf("START3\n");
    // This works only for equal square tiles
    STARSH_int N = RC->size[0];
    //printf("N=%d\n", N);
    STARSH_int stride = N*N;
    int pool_size = starpu_combined_worker_get_size();
    int pool_rank = starpu_combined_worker_get_rank();
    STARSH_int job_size = (batch_size-1)/pool_size + 1;
    STARSH_int job_start = job_size * pool_rank;
    STARSH_int job_end = job_start + job_size;
    if(job_end > batch_size)
        job_end = batch_size;
    for(STARSH_int ibatch = job_start; ibatch < job_end; ++ibatch)
    {
        int i = ind[ibatch*2];
        int j = ind[ibatch*2+1];
        kernel(N, N, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
                RD, CD, D + ibatch*stride, N);
    }
    //printf("END\n");
    //printf("FINISH BATCH IN: %f seconds\n", omp_get_wtime()-time0);
}

