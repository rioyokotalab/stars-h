#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-fugaku_gc.h"

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
    STARSH_laplace *data = row_data;

    STARSH_int N = data->N;
    STARSH_int nblocks = data->nblocks;
    STARSH_int block_size = data->block_size;
    double PV = data->PV;
    double *buffer = result;

    STARSH_laplace *data1 = row_data;
    STARSH_laplace *data2 = col_data;
    int ndim = data->ndim;

    double *x1[ndim], *x2[ndim];

    double rij = 0;
    for (int k = 0; k < ndim; ++k) {
        double* coord_x = (double*)row_data;
//        rij += pow()
    }
}

int starsh_normal_grid_generate(STARSH_particles** data, STARSH_int N,
    STARSH_int ndim) {
    STARSH_MALLOC(*data, 1);
    (*data)->count = N * ndim;
    (*data)->ndim = ndim;

    double *point;
    STARSH_MALLOC(point, (*data)->count);
    srand(1);

    for (int i = 0; i < N; ++i) {
        point[i] = rand() / N;
        point[i + N] = rand() / N;
        point[i + 2 * N] = rand() / N;
    }

    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return 1;
}

int starsh_laplace_grid_generate(STARSH_laplace **data, STARSH_int N,
        STARSH_int ndim,
        STARSH_int block_size, STARSH_int nblocks, double PV) {

    if (data == NULL) {
        STARSH_ERROR("Invalid value of data.");
        return STARSH_WRONG_PARAMETER;
    }
    if (N <= 0) {
        STARSH_ERROR("Invalid value of N.\n");
        return STARSH_WRONG_PARAMETER;
    }
    if (block_size <= 0) {
        STARSH_ERROR("Invalid value of block_size.\n");
        return STARSH_WRONG_PARAMETER;
    }
    if (PV > 1) {
        STARSH_ERROR("Invalid value of PV.\n");
        return STARSH_WRONG_PARAMETER;
    }

    int info;
    STARSH_particles *particles;
    info = starsh_normal_grid_generate(&particles, N, ndim);
    if(info != STARSH_SUCCESS)
    {
        fprintf(stderr, "INFO=%d\n", info);
        return info;
    }
    STARSH_MALLOC(*data, 1);
    (*data)->particles = *particles;
    free(particles);
    (*data)->N = N;
    (*data)->nblocks = nblocks;
    (*data)->block_size = block_size;
    (*data)->PV = PV;
    (*data)->ndim = ndim;

    return STARSH_SUCCESS;
}