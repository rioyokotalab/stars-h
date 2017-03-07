#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-plasma.h"
#include <plasma.h>

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        printf("%d\n", argc);
        printf("spatial n block_size beta\n");
        exit(1);
    }
    int n = atoi(argv[1]), block_size = atoi(argv[2]);
    double beta = atof(argv[3]);
    int maxrank = 100, oversample = 10, onfly = 0;
    double tol = 1e-9;
    char *scheme = "omp_rsdd";
    srand(100);
    // Generate data for spatial statistics problem
    STARSH_ssdata *data;
    STARSH_kernel kernel;
    starsh_gen_ssdata(&data, &kernel, n, beta);
    int ndim = 2, shape[2] = {data->count, data->count};
    char symm = 'S', dtype = 'd';
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Spatial Statistics example");
    starsh_problem_info(P);
    // Get array from problem
    Array *A;
    double time0 = omp_get_wtime();
    array_from_buffer(&A, 2, shape, 'd', 'F', NULL);
    starsh_problem_to_array(P, &A);
    time0 = omp_get_wtime()-time0;
    printf("Compute entire matrix: %e secs\n", time0);
    int N = A->shape[0];
    int nrhs = 10;
    double *b, *x, *x_CG, *CG_work;
    STARSH_MALLOC(b, N*nrhs);
    STARSH_MALLOC(x, N*nrhs);
    STARSH_MALLOC(x_CG, N*nrhs);
    int iseed[4] = {0, 0, 0, 1};
    LAPACKE_dlarnv_work(3, iseed, N*nrhs, b);
    //LAPACKE_dlarnv_work(3, iseed, N, x);
    STARSH_MALLOC(CG_work, 3*(N+1)*nrhs);
    plasma_init();
    time0 = omp_get_wtime();
    solve(N, A->data, N, nrhs, b, N, x, N);
    time0 = omp_get_wtime()-time0;
    printf("Time to solve SPD problem: %e secs\n", time0);
    plasma_finalize();
    // Init tiled cluster for tiled low-rank approximation and print info
    time0 = omp_get_wtime();
    STARSH_cluster *C;
    starsh_cluster_new_tiled(&C, data, data->count, block_size);
    //starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    starsh_blrf_new_tiled(&F, P, C, C, symm);
    //starsh_blrf_info(F);
    // Approximate each admissible block
    STARSH_blrm *M;
    double time1 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, oversample, tol, onfly, scheme);
    time1 = omp_get_wtime()-time1;
    starsh_blrf_info(F);
    starsh_blrm_info(M);
    printf("TIME TO APPROXIMATE: %e secs\n", time1);
    // Solve with CG, approximate solution is in x, initial guess is zero
    //memset(x, 0, data->count*sizeof(double));
    cblas_dcopy(N, b, 1, x_CG, 1);
    time1 = omp_get_wtime();
    int info = starsh_itersolvers__dcg(M, nrhs, b, N, x_CG, N, tol, CG_work);
    printf("CG INFO: %d\n", info);
    cblas_daxpy(N*nrhs, -1.0, x, 1, x_CG, 1);
    //for(int i = 0; i < 1000; i++)
    //    starsh_blrm__dmml_omp(M, 1, 1.0, b, N, 0.0, x, N);
    double time2 = omp_get_wtime();
    printf("TIME TO SOLVE: %e secs\n", time2-time1);
    printf("TOTAL TIME FOR STARSH: %e secs\n", time2-time0);
    printf("ACCURACY OF SOLUTION: %e\n", cblas_dnrm2(N*nrhs, x_CG, 1)/cblas_dnrm2(N*nrhs, x, 1));
    return 0;
}
