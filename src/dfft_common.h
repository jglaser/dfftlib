/*
 * Distributed FFT global defines
 */

#ifndef __DFFT_COMMON_H__
#define __DFFT_COMMON_H__

#include <dfft_lib_config.h>
#include "dfft_local_fft_config.h"

#include <mpi.h>

/*
 * Data structure for a distributed FFT
 */
typedef struct
    {
    int ndim;            /* dimensionality */
    int *gdim;           /* global input array dimensions */
    int *inembed;        /* embedding, per dimension, of input array */
    int *oembed;         /* embedding, per dimension, of output array */

    #ifdef ENABLE_HOST
    plan_t *plans_short_forward;/* short distance butterflies, forward dir */
    plan_t *plans_long_forward;  /* long distance butterflies, inverse dir */
    plan_t *plans_short_inverse; /* short distance butterflies, inverse dir */
    plan_t *plans_long_inverse;  /* long distance butterflies, inverse dir */
    #endif

    #ifdef ENABLE_CUDA
    cuda_plan_t *cuda_plans_short_forward; /* Cuda plans */
    cuda_plan_t *cuda_plans_long_forward; 
    cuda_plan_t *cuda_plans_short_inverse;
    cuda_plan_t *cuda_plans_long_inverse; 
    #endif

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

    #ifdef ENABLE_HOST
    cpx_t *scratch;       /* Scratch array */
    cpx_t *scratch_2;     /* Scratch array */
    #endif

    #ifdef ENABLE_CUDA
    cuda_cpx_t *d_scratch;   /* Scratch array */
    cuda_cpx_t *d_scratch_2; /* Scratch array */
    #endif 

    int scratch_size;     /* Size of scratch array */

    int np;               /* size of problem (number of elements per proc) */
    int size_in;          /* size including embedding */
    int size_out;         /* size including embedding */
    int *k0;              /* Last stage of butterflies (per dimension */
    
    int input_cyclic;     /* ==1 if input for the forward transform is cyclic */
    int output_cyclic;    /* ==1 if output for the backward transform is cyclic */

    int device;           /* ==1 if this is a device plan */
    } dfft_plan;

/*
 * Create a plan for distributed FFT (internal interface)
 */
int dfft_create_plan_common(dfft_plan *plan,
    int ndim, int *gdim,
    int *inembed, int *ombed, 
    int *pdim, int input_cyclic, int output_cyclic,
    MPI_Comm comm,
    int device);

/*
 * Destroy a plan (internal interface)
 */
void dfft_destroy_plan_common(dfft_plan plan, int device);

#endif
