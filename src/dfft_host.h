/*
 * Distributed FFT on the host
 */

#ifndef __DFFT_HOST_H__
#define __DFFT_HOST_H__

#include <mpi.h>
#include <dfft_lib_config.h>

#if (LOCAL_FFT_LIB == MKL) /* MKL, single precision */
#include "mkl_single_interface.h"
#endif

#ifdef __cplusplus
#define EXTERN_DFFT extern "C"
#else
#define EXTERN_DFFT
#endif

/*
 * Data structure for a distributed FFT
 */
typedef struct
    {
    int ndim;            /* dimensionality */
    int *gdim;           /* global input array dimensions */
    int *inembed;        /* embedding, per dimension, of input array */
    int *oembed;         /* embedding, per dimension, of output array */
 
    plan_t *plans_short_forward;/* short distance butterflies, forward dir */
    plan_t *plans_long_forward;  /* long distance butterflies, inverse dir */
    plan_t *plans_short_inverse; /* short distance butterflies, inverse dir */
    plan_t *plans_long_inverse;  /* long distance butterflies, inverse dir */

    int **rho_L;        /* bit reversal lookup length L, per dimension */   
    int **rho_pk0;      /* bit reversal lookup length p/k0, per dimension */
    int **rho_Lk0;      /* bit reversal lookup length L/k0, per dimension */   

    int *pdim;            /* Dimensions of processor grid */
    int *pidx;            /* Processor index, per dimension */
    MPI_Comm comm;        /* MPI communicator */

    int *offset_recv; /* temporary arrays */
    int *offset_send;
    int *nrecv;
    int *nsend;

    cpx_t *scratch;       /* Scratch array */
    cpx_t *scratch_2;     /* Scratch array */
    int scratch_size;     /* Size of scratch array */

    int np;               /* size of problem (number of elements per proc) */
    int size_in;          /* size including embedding */
    int size_out;         /* size including embedding */
    int *k0;              /* Last stage of butterflies (per dimension */
    
    int input_cyclic;     /* ==1 if input for the forward transform is cyclic */
    int output_cyclic;    /* ==1 if output for the backward transform is cyclic */
    } dfft_plan;

/*
 * Create a plan for distributed FFT
 */
EXTERN_DFFT int dfft_create_plan(dfft_plan *plan,
    int ndim, int *gdim,
    int *inembed, int *ombed, 
    int *pdim, int input_cyclic, int output_cyclic,
    MPI_Comm comm);

/*
 * Destroy a plan
 */
EXTERN_DFFT void dfft_destroy_plan(dfft_plan plan);

/*
 * Execute the parallel FFT
 */
EXTERN_DFFT int dfft_execute(cpx_t *in, cpx_t *out, int dir, dfft_plan p);
#endif
