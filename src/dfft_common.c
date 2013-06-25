#include <stdlib.h>
#include <string.h>

#include "dfft_common.h"
#include "dfft_host.h"
#include "dfft_cuda.h"

/*****************************************************************************
 * Implementation of common functions for device and host distributed FFT
 *****************************************************************************/
void bitrev_init(int n, int *rho)
    {
    int j;
    int n1, rem, val, k, lastbit, one=1;

    if (n==1)
        {
        rho[0]= 0;
        return;
        }
    n1= n;
    for(j=0; j<n; j++)
        {
        rem= j; 
        val= 0;
        for (k=1; k<n1; k <<= 1)
            {
            lastbit= rem & one;
            rem >>= 1;
            val <<= 1;
            val |= lastbit;
            }
        rho[j]= (int)val;
        }
   }

/*****************************************************************************
 * Plan management
 *****************************************************************************/
int dfft_create_plan_common(dfft_plan *p,
    int ndim, int *gdim,
    int *inembed, int *oembed, 
    int *pdim, int *pidx, int row_m,
    int input_cyclic, int output_cyclic,
    MPI_Comm comm,
    int device)
    {
    int s,nump;

    p->comm = comm;

    MPI_Comm_size(comm,&nump);
  
    /* number of processor must be power of two */
    if (nump & (nump-1)) return 4; 

    p->pdim = malloc(ndim*sizeof(int));
    p->gdim = malloc(ndim*sizeof(int));
    p->pidx = malloc(ndim*sizeof(int));

    p->inembed = malloc(ndim*sizeof(int));
    p->oembed = malloc(ndim*sizeof(int));

    p->ndim = ndim;

    int i;
    for (i = 0; i < ndim; i++)
        {
        p->gdim[i] = gdim[i];
       
        /* Every dimension must be a power of two */
        if (gdim[i] & (gdim[i]-1)) return 5; 

        p->pdim[i] = pdim[i];
        }

    if (inembed != NULL)
        {
        for (i = 0; i < ndim; i++)
            p->inembed[i] = inembed[i];
        }
    else
        {
        for (i = 0; i < ndim; i++)
            p->inembed[i] = p->gdim[i]/p->pdim[i];
        }

    if (oembed != NULL)
        {
        for (i = 0; i < ndim; i++)
            p->oembed[i] = oembed[i];
        }
    else
        {
        for (i = 0; i < ndim; i++)
            p->oembed[i] = p->gdim[i]/p->pdim[i];
        }

    /* since we expect column-major input, the leading dimension 
      has no embedding */
    p->inembed[0] = gdim[0]/pdim[0];
    p->oembed[0] = gdim[0]/pdim[0];

    p->offset_send = (int *)malloc(sizeof(int)*nump);
    p->offset_recv = (int *)malloc(sizeof(int)*nump);
    p->nsend = (int *)malloc(sizeof(int)*nump);
    p->nrecv = (int *)malloc(sizeof(int)*nump);

    if (!device)
        {
        #ifdef ENABLE_HOST
        p->plans_short_forward = malloc(sizeof(plan_t)*ndim);
        p->plans_long_forward = malloc(sizeof(plan_t)*ndim);
        p->plans_short_inverse = malloc(sizeof(plan_t)*ndim);
        p->plans_long_inverse = malloc(sizeof(plan_t)*ndim);
        #else
        return 3;
        #endif
        }
    else
        {
        #ifdef ENABLE_CUDA
        p->cuda_plans_short_forward = malloc(sizeof(cuda_plan_t)*ndim);
        p->cuda_plans_long_forward = malloc(sizeof(cuda_plan_t)*ndim);
        p->cuda_plans_short_inverse = malloc(sizeof(cuda_plan_t)*ndim);
        p->cuda_plans_long_inverse = malloc(sizeof(cuda_plan_t)*ndim);
        #else
        return 2;
        #endif
        }

    /* local problem size */
    int size_in = 1;
    int size_out = 1;

    for (i = 0; i < ndim ; ++i)
        {
        size_in *= p->inembed[i];
        size_out *= p->oembed[i];
        }

    p->size_in = size_in;
    p->size_out = size_out;

    int delta_in = 0;
    int delta_out = 0;
    for (i = 0; i < ndim; ++i)
        {
        delta_in *= p->inembed[i];
        delta_in += (p->inembed[i]- gdim[i]/pdim[i]);
        delta_out *= p->oembed[i];
        delta_out += (p->oembed[i]- gdim[i]/pdim[i]);
        }
    p->delta_in = delta_in;
    p->delta_out = delta_out;

    /* find length k0 of last stage of butterflies */
    p->k0 = malloc(sizeof(int)*ndim);

    for (i = 0; i< ndim; ++i)
        {
        int length = gdim[i]/pdim[i];
        if (length > 1)
            {
            int c;
            for (c=gdim[i]; c>length; c /= length)
                ;
            p->k0[i] = c; 
            }
        else
            {
            p->k0[i] = 1;
            }
        }

    p->rho_L = (int **)malloc(ndim*sizeof(int *)); 
    p->rho_pk0= (int **)malloc(ndim*sizeof(int *)); 
    p->rho_Lk0 = (int **)malloc(ndim*sizeof(int *));

    for (i = 0; i < ndim; ++i)
        {
        int length = gdim[i]/pdim[i];
        p->rho_L[i] = (int *) malloc(sizeof(int)*length);
        p->rho_pk0[i] = (int *) malloc(sizeof(int)*pdim[i]/(p->k0[i]));
        p->rho_Lk0[i] = (int *) malloc(sizeof(int)*length/(p->k0[i]));
        bitrev_init(length, p->rho_L[i]);
        bitrev_init(pdim[i]/(p->k0[i]),p->rho_pk0[i]);
        bitrev_init(length/(p->k0[i]),p->rho_Lk0[i]);
        }

    /* processor coordinates */
    for (i = 0; i < ndim; ++i)
        {
        p->pidx[i] = pidx[i];
        }

    /* init local FFT library */
    int res;
    if (!device)
        {
        #ifdef ENABLE_HOST
        res = dfft_init_local_fft();
        #else
        return 3;
        #endif
        }
    else
        {
        #ifdef ENABLE_CUDA
        res = dfft_cuda_init_local_fft();
        #else
        return 2;
        #endif
        }

    if (res) return 1;

    int size = size_in;
    for (i = 0; i < ndim; ++i)
        {
        /* plan for short-distance butterflies */
        int st = size/p->inembed[i]*(gdim[i]/pdim[i]);
        if (!device)
            {
            #ifdef ENABLE_HOST
            dfft_create_1d_plan(&(p->plans_short_forward[i]),p->k0[i],
                st/(p->k0[i]), st/(p->k0[i]), 1, st/(p->k0[i]), 1, 0);
            dfft_create_1d_plan(&(p->plans_short_inverse[i]),p->k0[i],
                st/(p->k0[i]), st/(p->k0[i]), 1, st/(p->k0[i]), 1, 1);

            /* plan for long-distance butterflies */
            int length = gdim[i]/pdim[i];
            dfft_create_1d_plan(&(p->plans_long_forward[i]), length,
                st/length, st/length,1, st/length,1, 0);
            dfft_create_1d_plan(&(p->plans_long_inverse[i]), length,
                st/length, st/length,1, st/length,1, 1);
            #else
            return 3;
            #endif
            }
        else
            {
            #ifdef ENABLE_CUDA
            dfft_cuda_create_1d_plan(&(p->cuda_plans_short_forward[i]),p->k0[i],
                st/(p->k0[i]), st/(p->k0[i]), 1, st/(p->k0[i]), 1, 0);
            dfft_cuda_create_1d_plan(&(p->cuda_plans_short_inverse[i]),p->k0[i],
                st/(p->k0[i]), st/(p->k0[i]), 1, st/(p->k0[i]), 1, 1);

            /* plan for long-distance butterflies */
            int length = gdim[i]/pdim[i];
            dfft_cuda_create_1d_plan(&(p->cuda_plans_long_forward[i]), length,
                st/length, st/length,1, st/length,1, 0);
            dfft_cuda_create_1d_plan(&(p->cuda_plans_long_inverse[i]), length,
                st/length, st/length,1, st/length,1, 1);
            #else
            return 2;
            #endif
            }

        size /= p->inembed[i];
        size *= p->oembed[i];
        }

    /* Allocate scratch space */
    int scratch_size = 1;
    for (i = 0; i < ndim; ++i)
        scratch_size *= ((p->inembed[i] > p->oembed[i]) ? p->inembed[i]  : p->oembed[i]);
    p->scratch_size = scratch_size;

    if (!device)
        {
        #ifdef ENABLE_HOST
        dfft_allocate_aligned_memory(&(p->scratch),sizeof(cpx_t)*scratch_size);
        dfft_allocate_aligned_memory(&(p->scratch_2),sizeof(cpx_t)*scratch_size);
        #else
        return 3;
        #endif
        }
    else
        {
        #ifdef ENABLE_CUDA
        dfft_cuda_allocate_aligned_memory(&(p->d_scratch),sizeof(cuda_cpx_t)*scratch_size);
        dfft_cuda_allocate_aligned_memory(&(p->d_scratch_2),sizeof(cuda_cpx_t)*scratch_size);
        #else
        return 2;
        #endif
        }

    p->input_cyclic = input_cyclic;
    p->output_cyclic = output_cyclic;

    p->device = device;
    #ifdef ENABLE_CUDA
    p->check_cuda_errors = 0;
    #endif

    p->row_m = row_m;
    return 0;
    } 

void dfft_destroy_plan_common(dfft_plan p, int device)
    {
    if (device != p.device) return;

    /* Clean-up */
    int i;
    int ndim = p.ndim;
    for (i = 0; i < ndim; ++i)
        {
        if (!device)
            {
            #ifdef ENABLE_HOST
            dfft_destroy_1d_plan(&p.plans_short_forward[i]);
            dfft_destroy_1d_plan(&p.plans_short_inverse[i]);
            dfft_destroy_1d_plan(&p.plans_long_forward[i]);
            dfft_destroy_1d_plan(&p.plans_long_inverse[i]);
            #endif
            }
        else
            {
            #ifdef ENABLE_CUDA
            dfft_cuda_destroy_1d_plan(&p.cuda_plans_short_forward[i]);
            dfft_cuda_destroy_1d_plan(&p.cuda_plans_short_inverse[i]);
            dfft_cuda_destroy_1d_plan(&p.cuda_plans_long_forward[i]);
            dfft_cuda_destroy_1d_plan(&p.cuda_plans_long_inverse[i]);
            #endif
            } 
        }

    for (i = 0; i < ndim; ++i)
        {
        free(p.rho_L[i]);
        free(p.rho_pk0[i]);
        free(p.rho_Lk0[i]);
        }
    
    free(p.rho_L);
    free(p.rho_pk0);
    free(p.rho_Lk0);

    free(p.offset_recv);
    free(p.offset_send);
    free(p.nrecv);
    free(p.nsend);

    free(p.pidx);
    free(p.pdim);
    free(p.gdim);

    if (!device)
        {
        #ifdef ENABLE_HOST
        dfft_free_aligned_memory(p.scratch);
        dfft_free_aligned_memory(p.scratch_2);
        dfft_teardown_local_fft();
        #endif
        }
    else
        {
        #ifdef ENABLE_CUDA
        dfft_cuda_free_aligned_memory(p.d_scratch);
        dfft_cuda_free_aligned_memory(p.d_scratch_2);
        dfft_cuda_teardown_local_fft();
        #endif
        }

    }
