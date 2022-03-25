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
    STARSH_laplace *data1 = row_data;
    STARSH_laplace *data2 = col_data;

    STARSH_int N = data1->N;
    STARSH_int nblocks = data1->nblocks;
    STARSH_int block_size = data1->block_size;
    double PV = data1->PV;
    double *buffer = result;
    int ndim = data1->ndim;

    double *x1[ndim], *x2[ndim];

    x1[0] = data1->particles.point;
    x2[0] = data2->particles.point;
    for (int k = 1; k < ndim; ++k) {
        x1[k] = x1[0] + k * data1->particles.count;
        x2[k] = x2[0] + k * data2->particles.count;
    }



    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            double rij = 0;
            for (int k = 0; k < ndim; ++k) {
                rij += pow(x1[k][irow[i]] - x2[k][icol[j]], 2);
            }
            double out = 1 / (sqrt(rij) + PV);

            //printf("PV: %f points: %f %f %f irow: %d out: %d.\n", PV, x1[0][irow[i]], x1[1][irow[i]], x1[2][irow[i]], irow[i], out);
            buffer[i + j * ld] = out;
        }
    }

}

int starsh_normal_grid_generate(STARSH_particles** data, STARSH_int N,
    STARSH_int ndim) {
    STARSH_MALLOC(*data, 1);
    (*data)->count = N;
    (*data)->ndim = ndim;

    double *point;
    STARSH_MALLOC(point, (*data)->count * ndim);
    srand(1);

    for (int i = 0; i < N; ++i) {
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