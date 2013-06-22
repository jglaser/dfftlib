/*
 * Distributed FFT using CUDA
 */

#ifdef CUFFT /* CUFFT, single precision */
#include "cufft_interface.h"
typedef cufftComplex cuda_cpx_t;
typedef cufftHandle cuda_plan_t;
#endif

/* Data structures needed for a distributed FFT
 */
struct 
    {
    int ndim;            /* dimensionality */
    int *gdim;           /* global input array dimensions */
    int *inembed;        /* embedding, per dimension, of input array */
    int istride;         /* stride of input array */
    int idist;           /* number of elements between batches (input array) */
    int *oembed;         /* embedding, per dimension, of output array */
    int ostride;         /* stride of input array */
    int odist;           /* number of elements between batches (output array) */
 
    cuda_plan_t **plans_short_forward; /* short distance butterflies, forward dir */
    cuda_plan_t **plans_long_forward;  /* long distance butterflies, inverse dir */
    cuda_plan_t **plans_short_inverse; /* short distance butterflies, inverse dir */
    cuda_plan_t **plans_long_inverse;  /* long distance butterflies, inverse dir */

    int **rho_L;          /* bit reversal lookup, length L, per dimension */   
    int **rho_k0;         /* bit reversal lookup, length L, per dimension */   
    int **rho_pk0;        /* bit reversal lookup, length L, per dimension */   

    } dfft_cuda_plan;
