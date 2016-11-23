#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "stars.h"
#include "stars-spatial.h"

int main(int argc, char **argv)
// Example of how to use STARS library for spatial statistics.
// For more information on STARS structures look inside of header files.
{
    if(argc < 7)
    {
        printf("%d\n", argc);
        printf("spatial.out row_blocks col_blocks block_size maxrank "
                "tol beta\n");
        exit(0);
    }
    size_t row_blocks = atoi(argv[1]), col_blocks = atoi(argv[2]);
    size_t block_size = atoi(argv[3]), maxrank = atoi(argv[4]);
    double tol = atof(argv[5]), beta = atof(argv[6]);
    printf("\nrb=%zu, cb=%zu, bs=%zu, mr=%zu, tol=%e, beta=%f\n",
            row_blocks, col_blocks, block_size, maxrank, tol, beta);
    // Setting random seed
    srand(time(NULL));
    // Generate data for spatial statistics problem
    STARS_ssdata *data = STARS_gen_ssdata(row_blocks, col_blocks, block_size,
            beta);
    size_t ndim = 2, shape[2] = {data->count, data->count};
    char symm = 'S', dtype = 'd';
    // Init problem with given data and kernel
    STARS_Problem *problem = STARS_Problem_init(ndim, shape, symm, dtype,
            data, data, STARS_ssdata_block_exp_kernel, "Spatial Statistics "
            "example");
    STARS_Problem_info(problem);
    // Init tiled cluster for tiled low-rank approximation
    STARS_Cluster *cluster = STARS_Cluster_init_tiled(data, data->count,
            block_size);
    STARS_Cluster_info(cluster);
    // Init tiled division into admissible blocks
    STARS_BLRF *blrf = STARS_BLRF_init_tiled(problem, cluster, cluster, 'S');
    STARS_BLRF_info(blrf);
    // Approximate each admissible block
    STARS_BLRM *blrm = STARS_blrf_tiled_compress_algebraic_svd_ompfor(blrf,
            maxrank, tol, 1); // 0 for onfly=0
    STARS_BLRM_info(blrm);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(blrm);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(blrm);
    // Approximate each admissible block
    blrm = STARS_blrf_tiled_compress_algebraic_svd_batched(blrf, maxrank, tol,
            0, 1000000000);
    STARS_BLRM_info(blrm);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(blrm);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(blrm);
    // Free memory, used by block low-rank format
    STARS_BLRF_free(blrf);
    // Free memory, used by clusterization info
    STARS_Cluster_free(cluster);
    // Free memory, used by STARS_Problem instance
    STARS_Problem_free(problem);
    // Free memory, used by spatial statistics data
    STARS_ssdata_free(data);
    return 0;
}
