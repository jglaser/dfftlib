/* MKL (single precision) backend for distributed FFT, implementation
 */

#include "acml_single_interface.h"

/* Initialize the library
 */
int dfft_init_local_fft()
    {
    return 0;
    }

/* De-initialize the library
 */
void dfft_teardown_local_fft()
    {
    }

/* Create a FFTW plan
 *
 * sign = 0 (forward) or 1 (inverse)
 */
int dfft_create_1d_plan(
    plan_t *plan,
    int dim,
    int howmany,
    int istride,
    int idist,
    int ostride,
    int odist,
    int dir)
    {
    plan->dim = dim;
    plan->howmany = howmany;
    plan->istride = istride;
    plan->idist = idist;
    plan->ostride = ostride;
    plan->odist = odist;
    plan->comm = (complex *) malloc(sizeof(complex)*(5*dim+100));
    /* Initialize communication buffer */
    int info;
    cfft1mx(100, 1.0, 0, howmany, dim, NULL, istride, idist, NULL,
             ostride, odist, plan->comm, &info);
    return 0;
    }

int dfft_allocate_aligned_memory(cpx_t **ptr, size_t size)
    {
    *ptr = (cpx_t *) malloc(size);
    return 0;
    }

void dfft_free_aligned_memory(cpx_t *ptr)
    {
    free(ptr);
    }

/* Destroy a 1d plan */
void dfft_destroy_1d_plan(plan_t *p)
    {
    free(p->comm);
    }

/* Excecute a local 1D FFT
 */
void dfft_local_1dfft(
    cpx_t *in,
    cpx_t *out,
    plan_t p,
    int dir)
    {
    int info;
    if (!dir)
        cfft1mx(-1, 1.0, (in==out) ? 1 : 0,
             p.howmany, p.dim, in, p.istride, p.idist,
             out, p.ostride, p.odist, p.comm, &info);
    else
        cfft1mx(1, 1.0, (in == out) ? 1 : 0,
            p.howmany, p.dim, in, p.istride, p.idist,
            out,p.ostride, p.odist, p.comm, &info);
    }
