/*
 * CUFFT (single precision) backend for distributed FFT, implementation
 */

#include "cufft_single_interface.h"

/* Initialize the library
 */
int dfft_cuda_init_local_fft()
    {
    return 0;
    }

/* De-initialize the library
 */
void dfft_cuda_teardown_local_fft()
    {
    }

/* Create a FFTW plan
 *
 * sign = 0 (forward) or 1 (inverse)
 */
int dfft_cuda_create_1d_plan(
    cuda_plan_t *plan,
    int dim,
    int howmany,
    int istride,
    int idist,
    int ostride,
    int odist,
    int dir)
    {
    int dims[1];
    dims[0] = dim;

    cufftPlanMany(plan, 1, dims, dims, istride, idist, dims, ostride, odist,
        CUFFT_C2C, howmany);
    return 0;
    }

int dfft_cuda_allocate_aligned_memory(cuda_cpx_t **ptr, size_t size)
    {
    cudaMalloc((void **) ptr,size);
    return 0;
    }

void dfft_cuda_free_aligned_memory(cuda_cpx_t *ptr)
    {
    cudaFree(ptr);
    }

/* Destroy a 1d plan */
void dfft_cuda_destroy_1d_plan(cuda_plan_t *p)
    {
    cufftDestroy(*p);
    }

/*
 * Excecute a local 1D FFT
 */
void dfft_cuda_local_1dfft(
    cuda_cpx_t *in,
    cuda_cpx_t *out,
    cuda_plan_t p,
    int dir)
    {
    cufftExecC2C(p, in, out, dir ? CUFFT_INVERSE : CUFFT_FORWARD);
    }

