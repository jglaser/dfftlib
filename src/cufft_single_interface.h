/*
 * CUFFT (single precision) backend for distributed FFT
 */

#ifndef __DFFT_CUFFT_SINGLE_INTERFACE_H__
#define __DFFT_CUFFT_SINGLE_INTERFACE_H__

#include <cufft.h>
#include <cuda_runtime.h>

typedef cufftComplex cuda_cpx_t;
typedef cufftHandle cuda_plan_t;

#define CUDA_RE(X) X.x
#define CUDA_IM(X) X.y

#ifndef NVCC

/* Initialize the library
 */
int dfft_cuda_init_local_fft();

/* De-initialize the library
 */
void dfft_cuda_teardown_local_fft();

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
    int dir);

int dfft_cuda_allocate_aligned_memory(cuda_cpx_t **ptr, size_t size);

void dfft_cuda_free_aligned_memory(cuda_cpx_t *ptr);

/* Destroy a 1d plan */
void dfft_cuda_destroy_1d_plan(cuda_plan_t *p);

/*
 * Excecute a local 1D FFT
 */
void dfft_cuda_local_1dfft(
    cuda_cpx_t *in,
    cuda_cpx_t *out,
    cuda_plan_t p,
    int dir);

#endif /* NVCC */
#endif
