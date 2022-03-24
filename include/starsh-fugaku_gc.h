#ifndef __STARSH_FUGAKU_GC_H__
#define __STARSH_FUGAKU_GC_H__

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"

typedef struct starsh_laplace
{
    STARSH_particles particles;

    STARSH_int N;
    //!< Number of rows/columns of synthetic matrix.
    STARSH_int nblocks;
    //!< Number of tiles in one dimension.
    STARSH_int block_size;
    //!< Size of each tile.
    STARSH_int ndim;
    double PV;
    //!< Value to add to each diagonal element (for positive definiteness).
} STARSH_laplace;

#endif