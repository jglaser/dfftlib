#include "dfft_cuda.h"

#include <cuda_runtime.h>

#include <mpi.h>

#include <stdlib.h>
#include <unistd.h>

#include "dfft_common.h"
#include "dfft_cuda.h"
#include "dfft_cuda.cuh"

#define CHECK_CUDA() \
    {                                                                       \
    cudaDeviceSynchronize();                                                \
    cudaError_t err = cudaGetLastError();                                   \
    if (err != cudaSuccess)                                                 \
        {                                                                   \
        printf("CUDA Error in file %s, line %d: %s\n", __FILE__,__LINE__,   \
            cudaGetErrorString(err));                                       \
        exit(1);                                                            \
        }                                                                   \
    }                                                                       \

/*****************************************************************************
 * Implementation of the distributed FFT
 *****************************************************************************/

/*
 * Redistribute from group-cyclic with cycle c0 to cycle c1>=c0
 */
void dfft_cuda_redistribute_block_to_cyclic_1d(
                  int *dim,
                  int *pdim,
                  int ndim,
                  int current_dim,
                  int c0,
                  int c1,
                  int* pidx,
                  int size_in,
                  int *embed,
                  cuda_cpx_t *d_work,
                  cuda_cpx_t *d_scratch,
                  cuda_cpx_t *h_stage_in,
                  cuda_cpx_t *h_stage_out,
                  int *dfft_nsend,
                  int *dfft_nrecv,
                  int *dfft_offset_send,
                  int *dfft_offset_recv,
                  MPI_Comm comm,
                  int check_err)
    {
    /* exit early if nothing needs to be done */
    if (c0 == c1) return;

    int length = dim[current_dim]/pdim[current_dim];

    /* compute stride for column major matrix storage */
    int stride = size_in/embed[current_dim];

    /* processor index along current dimension */
    int s = pidx[current_dim];
    int p = pdim[current_dim];

    int ratio = c1/c0;
    int size = ((length/ratio > 1) ? (length/ratio) : 1);
    int npackets = length/size;
    size *= stride;

    int pdim_tot=1;
    int k;
    for (k = 0; k < ndim; ++k)
        pdim_tot *= pdim[k];

    int t;
    for (t = 0; t<pdim_tot; ++t)
        {
        dfft_nsend[t] = 0;
        dfft_nrecv[t] = 0;
        dfft_offset_send[t] = 0;
        dfft_offset_recv[t] = 0;
        }

    int j0;
    int j2;

    j0 = s % c0;
    j2 = s / c0;

    /* initialize send offsets and pack data */
    int j;
    for (j = 0; j < npackets; ++j)
        {
        int offset = j*size;
        int jglob = j2*c0*length + j * c0 + j0;
        int desti = (jglob/(c1*length))*c1+ jglob%c1;
        int destproc = 0;
        for (k = 0; k < ndim; ++k)
            {
            destproc *= pdim[k];
            destproc += ((current_dim == k) ? desti : pidx[k]);
            }
        dfft_nsend[destproc] = size*sizeof(cuda_cpx_t);
        dfft_offset_send[destproc] = offset*sizeof(cuda_cpx_t);
        }

    /* pack data */
    gpu_b2c_pack(npackets*size, ratio, size, npackets, stride, d_work, d_scratch);
    if (check_err) CHECK_CUDA();

    /* initialize recv offsets */
    int offset = 0;
    j0 = s % c1;
    j2 = s/c1;

    int r;
    for (r = 0; r < npackets; ++r)
        {
        offset = r*size;
        j = r*size/stride;
        int jglob = j2*c1*length+ j * c1 + j0;
        int srci = (jglob/(c0*length))*c0+jglob%c0;
        int srcproc = 0;
        int k;
        for (k = 0; k < ndim; ++k)
            {
            srcproc *= pdim[k];
            srcproc += ((current_dim == k) ? srci : pidx[k]);
            }
 
        dfft_nrecv[srcproc] = size*sizeof(cuda_cpx_t);
        dfft_offset_recv[srcproc] = offset*sizeof(cuda_cpx_t);
        }

    /* synchronize */
    MPI_Barrier(comm);

    /* communicate */
    #ifdef ENABLE_MPI_CUDA
    MPI_Alltoallv(d_scratch,dfft_nsend, dfft_offset_send, MPI_BYTE,
                  d_work, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                  comm);
    #else
    // stage into host buf
    cudaMemcpy(h_stage_in, d_scratch, sizeof(cuda_cpx_t)*npackets*size,cudaMemcpyDefault); 
    if (check_err) CHECK_CUDA();

    MPI_Alltoallv(h_stage_in,dfft_nsend, dfft_offset_send, MPI_BYTE,
                  h_stage_out, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                  comm);

    // copy back received data
    cudaMemcpy(d_work,h_stage_out, sizeof(cuda_cpx_t)*size_in,cudaMemcpyDefault); 
    if (check_err) CHECK_CUDA();
    #endif
    }

/* Redistribute from group-cyclic with cycle c0 to cycle c0>=c1
 * rev=1 if local order is reversed
 *
 * if rev = 1 and np >= c0 (last stage) it really transforms
 * into a hybrid-distribution, which after the last local ordered
 * DFT becomes the cyclic distribution
 */
void dfft_cuda_redistribute_cyclic_to_block_1d(int *dim,
                     int *pdim,
                     int ndim,
                     int current_dim,
                     int c0,
                     int c1,
                     int* pidx,
                     int rev,
                     int size_in,
                     int *embed,
                     cuda_cpx_t *d_work,
                     cuda_cpx_t *d_scratch,
                     cuda_cpx_t *h_stage_in,
                     cuda_cpx_t *h_stage_out,
                     int *rho_L,
                     int *rho_pk0,
                     int *dfft_nsend,
                     int *dfft_nrecv,
                     int *dfft_offset_send,
                     int *dfft_offset_recv,
                     MPI_Comm comm,
                     int check_err
                     )
    {
    if (c1 == c0) return;

    /* length along current dimension */
    int length = dim[current_dim]/pdim[current_dim];
    int size = length*c1/c0;
    size = (size ? size : 1);
    int npackets = length/size; 

    int stride = size_in/embed[current_dim];

    /* processor index along current dimension */
    int s=pidx[current_dim];
    /* number of procs along current dimension */
    int p=pdim[current_dim];

    size *= stride;

    int offset = 0;
    int recv_size,send_size;
    int j0_local = s%c0;
    int j2_local = s/c0;
    int j0_new_local = s%c1;
    int j2_new_local = s/c1;

    int pdim_tot=1;
    int k;
    for (k = 0; k < ndim; ++k)
        pdim_tot *= pdim[k];

    int i;
    for (i = 0; i < pdim_tot; ++i)
        {
        dfft_nsend[i] = 0;
        dfft_nrecv[i] = 0;
        dfft_offset_send[i] = 0;
        dfft_offset_recv[i] = 0;
        }

    for (i = 0; i < p; ++i)
        {
        int j0_remote = i%c0;
        int j2_remote = i/c0;

        int j0_new_remote = i % c1;
        int j2_new_remote = i/c1;
    
        /* decision to send and/or receive */
        int send = 0;
        int recv = 0;
        if (rev && (length >= c0))
            {
            /* redistribute into block with reversed processor id
               and swapped-partially reversed local order (the c0 LSB
               of the local index are MSB, and the n/p/c0 MSB
               are LSB and are reversed */
            send = (((j2_new_remote % (p/c0)) == (rho_pk0[j2_local])) ? 1 : 0);
            recv = (((j2_new_local % (p/c0)) == (rho_pk0[j2_remote])) ? 1 : 0);
            }
        else
            {
            send = (((j2_new_remote / (c0/c1)) == j2_local) && ((j0_local % c1)==j0_new_remote) ? 1 : 0); 
            recv = (((j2_new_local / (c0/c1)) == j2_remote) &&  ((j0_remote % c1)==j0_new_local) ? 1 : 0);

            if (length*c1 < c0)
                {
                send &= (j0_local/(length*c1) == j2_new_remote % (c0/(length*c1)));
                recv &= (j0_remote/(length*c1) == j2_new_local % (c0/(length*c1)));
                }
            }

        /* offset of first element sent */
        int j1;
        if (length*c1 >= c0)
            {
            j1 = (j2_new_remote % (c0/c1))*length*c1/c0;
            }
        else
            {
            j1 = (j2_new_remote / (c0/(length*c1))) % length;
            }

        if (rev)
            {
            if (length >= c0)
                {
                j1 = j2_new_remote/(p/c0);
                }
            else
                j1 = rho_L[j1];
            }
        
        /* mirror remote decision to send */
        send_size = (send ? size : 0);
        recv_size = (recv ? size : 0);

        int destproc = 0;
        int k;
        for (k = 0; k < ndim; ++k)
            {
            destproc *= pdim[k];
            destproc += ((current_dim == k) ? i : pidx[k]);
            }

        dfft_offset_send[destproc] = (send ? (stride*j1*sizeof(cuda_cpx_t)) : 0);
        if (rev && (length > c0/c1))
            {
            /* we are directly receving into the work buf */
            dfft_offset_recv[destproc] = stride*j0_remote*length/c0*sizeof(cuda_cpx_t);
            }
        else
            {
            dfft_offset_recv[destproc] = offset*sizeof(cuda_cpx_t);
            }

        dfft_nsend[destproc] = send_size*sizeof(cuda_cpx_t);
        dfft_nrecv[destproc] = recv_size*sizeof(cuda_cpx_t);
        offset+=(recv ? size : 0);
        }

    /* we need to pack data if the local input buffer is reversed
       and we are sending more than one element */
    if (rev && (size > stride))
        {
        offset = 0;
        int i;
        for (i = 0; i <p; ++i)
            {
            int destproc = 0;
            int k;
            for (k = 0; k < ndim; ++k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? i : pidx[k]);
                }

            int j1_offset = dfft_offset_send[destproc]/sizeof(cuda_cpx_t)/stride;

            /* we are sending from a tmp buffer/stride */
            dfft_offset_send[destproc] = offset*sizeof(cuda_cpx_t)*stride;
            int n = dfft_nsend[destproc]/stride/sizeof(cuda_cpx_t);
            int j;
            offset += n;
            }

        /* pack data */
        gpu_b2c_pack(size_in, c0, size, c0, stride, d_work, d_scratch);
        if (check_err) CHECK_CUDA();
       
        /* perform communication */
        MPI_Barrier(comm);
        #ifdef ENABLE_MPI_CUDA
        MPI_Alltoallv(d_scratch,dfft_nsend, dfft_offset_send, MPI_BYTE,
                      d_work, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                      comm);
        #else
        // stage into host buf
        cudaMemcpy(h_stage_in, d_scratch, sizeof(cuda_cpx_t)*length*stride,cudaMemcpyDefault); 
        if (check_err) CHECK_CUDA();

        MPI_Alltoallv(h_stage_in,dfft_nsend, dfft_offset_send, MPI_BYTE,
                      h_stage_out, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                      comm);

        // copy back received data
        cudaMemcpy(d_work,h_stage_out, sizeof(cuda_cpx_t)*npackets*size,cudaMemcpyDefault); 
        if (check_err) CHECK_CUDA();
        #endif
        }
    else
        {
        /* perform communication */
        MPI_Barrier(comm);
        #ifdef ENABLE_MPI_CUDA
        MPI_Alltoallv(d_work,dfft_nsend, dfft_offset_send, MPI_BYTE,
                      d_scratch, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                      comm);
        #else
        // stage into host buf
        cudaMemcpy(h_stage_in, d_work, sizeof(cuda_cpx_t)*size_in,cudaMemcpyDefault); 
        if (check_err) CHECK_CUDA();

        MPI_Alltoallv(h_stage_in,dfft_nsend, dfft_offset_send, MPI_BYTE,
                      h_stage_out, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                      comm);

        // copy back received data
        cudaMemcpy(d_scratch,h_stage_out, sizeof(cuda_cpx_t)*npackets*size,cudaMemcpyDefault); 
        if (check_err) CHECK_CUDA();
        #endif

        /* unpack */
        gpu_c2b_unpack(npackets*size, length, c0, c1, size, j0_new_local, stride, rev, d_work, d_scratch);
        if (check_err) CHECK_CUDA();
        }
    }

/* plan_long: complete local FFT
   plan_short: partial local FFT
   input and output are M-cyclic (M=pdim[current_dim])
   (out-of-place version, overwrites input)
   */
void cuda_mpifft1d_dif(int *dim,
            int *pdim,
            int ndim,
            int current_dim,
            int* pidx,
            int inverse,
            int size,
            int *embed,
            cuda_cpx_t *d_in,
            cuda_cpx_t *d_out,
            cuda_cpx_t *h_stage_in,
            cuda_cpx_t *h_stage_out,
            cuda_plan_t plan_short,
            cuda_plan_t plan_long,
            int *rho_L,
            int *rho_pk0,
            int *rho_Lk0,
            int *dfft_nsend,
            int *dfft_nrecv,
            int *dfft_offset_send,
            int *dfft_offset_recv,
            MPI_Comm comm,
            int check_err)
    {
    int p = pdim[current_dim];
    int length = dim[current_dim]/pdim[current_dim];

    /* compute stride for column major matrix storage */
    int stride = size/embed[current_dim];

    int c;
    for (c = p; c >1; c /= length)
        {
        /* do local out-of-place place FFT (long-distance butterflies) */
        dfft_cuda_local_1dfft(d_in, d_out, plan_long, inverse);

        /* apply twiddle factors */
        double alpha = ((double)(pidx[current_dim] %c))/(double)c;

        gpu_twiddle(size, length, stride, alpha, d_out, d_in, inverse);
        if (check_err) CHECK_CUDA();

        /* in-place redistribute from group-cyclic c -> c1 */
        int rev = 1;
        int c1 = ((c > length) ? (c/length) : 1);
        dfft_cuda_redistribute_cyclic_to_block_1d(dim,pdim,ndim,current_dim,
            c, c1, pidx, rev, size, embed, d_in,d_out,h_stage_in, h_stage_out,
            rho_L,rho_pk0, dfft_nsend,dfft_nrecv,dfft_offset_send,
            dfft_offset_recv, comm, check_err);
        }

    /* perform remaining short-distance butterflies,
     * out-of-place 1d FFT */
    dfft_cuda_local_1dfft(d_in, d_out, plan_short,inverse);
    } 

/*
 * n-dimensional fft routine (in-place)
 */
void cuda_mpifftnd_dif(int *dim,
            int *pdim,
            int ndim,
            int* pidx,
            int inv,
            int size_in,
            int *inembed,
            int *oembed,
            cuda_cpx_t *d_work,
            cuda_cpx_t *d_scratch,
            cuda_cpx_t *h_stage_in,
            cuda_cpx_t *h_stage_out,
            cuda_plan_t *plans_short,
            cuda_plan_t *plans_long,
            int **rho_L,
            int **rho_pk0,
            int **rho_Lk0,
            int *dfft_nsend,
            int *dfft_nrecv,
            int *dfft_offset_send,
            int *dfft_offset_recv,
            MPI_Comm comm,
            int check_err)
    {
    int size = size_in;
    int current_dim;
    for (current_dim = 0; current_dim < ndim; ++current_dim)
        {
        /* assume input in local column major */
        cuda_mpifft1d_dif(dim, pdim,ndim,current_dim,pidx, inv,
            size, inembed, d_work, d_scratch,h_stage_in, h_stage_out,
            plans_short[current_dim],
            plans_long[current_dim], rho_L[current_dim],
            rho_pk0[current_dim],rho_Lk0[current_dim],
            dfft_nsend,dfft_nrecv,dfft_offset_send,dfft_offset_recv,
            comm,check_err);

        int l = dim[current_dim]/pdim[current_dim];
        int stride = size/inembed[current_dim];

        /* transpose local matrix */
        gpu_transpose(size,l,stride, oembed[current_dim],d_scratch, d_work);
        if (check_err) CHECK_CUDA();

        /* update size */
        size *= oembed[current_dim];
        size /= inembed[current_dim];
        }
    }

void cuda_redistribute_nd(int *dim,
            int *pdim,
            int ndim,
            int* pidx,
            int size,
            int *embed,
            cuda_cpx_t *d_work,
            cuda_cpx_t *d_scratch,
            cuda_cpx_t *h_stage_in,
            cuda_cpx_t *h_stage_out,
            int *dfft_nsend,
            int *dfft_nrecv,
            int *dfft_offset_send,
            int *dfft_offset_recv,
            int c2b,
            MPI_Comm comm,
            int check_err)
    {
    cuda_cpx_t *cur_work =d_work;
    cuda_cpx_t *cur_scratch =d_scratch;

    int current_dim;
    for (current_dim = 0; current_dim < ndim; ++current_dim)
        {
        /* redistribute along one dimension (in-place) */
        if (!c2b)
            dfft_cuda_redistribute_block_to_cyclic_1d(dim, pdim, ndim,
                current_dim, 1, pdim[current_dim], pidx, size, embed,
                cur_work, cur_scratch, h_stage_in, h_stage_out,
                dfft_nsend,dfft_nrecv,
                dfft_offset_send, dfft_offset_recv, comm,
                check_err);
        else
            dfft_cuda_redistribute_cyclic_to_block_1d(dim, pdim, ndim,
                current_dim, pdim[current_dim], 1, pidx, 0, size, embed,
                cur_work, cur_scratch, h_stage_in, h_stage_out,
                NULL, NULL, dfft_nsend, dfft_nrecv, dfft_offset_send,
                dfft_offset_recv, comm,
                check_err);
        
        int l = dim[current_dim]/pdim[current_dim];
        int stride = size/embed[current_dim];

        /* transpose local matrix */
        gpu_transpose(size,l,stride, embed[current_dim],cur_work, cur_scratch);
        if (check_err) CHECK_CUDA();

        /* swap buffers */
        cuda_cpx_t *tmp;        
        tmp = cur_scratch;
        cur_scratch = cur_work;
        cur_work = tmp;
        }

    if (ndim % 2)
        {
        cudaMemcpy(d_work, d_scratch, sizeof(cuda_cpx_t)*size,cudaMemcpyDefault);
        if (check_err) CHECK_CUDA();
        }
    }


/*****************************************************************************
 * Distributed FFT interface
 *****************************************************************************/
int dfft_cuda_execute(cuda_cpx_t *d_in, cuda_cpx_t *d_out, int dir, dfft_plan p)
    {
    int out_of_place = (d_in == d_out) ? 0 : 1;

    int check_err = p.check_cuda_errors;
    cuda_cpx_t *d_scratch, *d_work;

    if (out_of_place)
        {
        d_work = p.d_scratch;
        d_scratch = p.d_scratch_2; 
        int size = p.size_in - p.delta_in;
        cudaMemcpy(d_work, d_in, size*sizeof(cuda_cpx_t),cudaMemcpyDefault);
        if (check_err) CHECK_CUDA();
        }
    else
        {
        d_scratch = p.d_scratch;
        /*! FIXME need to ensure in buf size >= scratch_size */
        d_work = d_in;
        }

    if ((!dir && !p.input_cyclic) || (dir && !p.output_cyclic))
        {
        /* redistribution of input */
        cuda_redistribute_nd(p.gdim, p.pdim, p.ndim, p.pidx,
            p.size_in, p.inembed, d_work, d_scratch, p.h_stage_in,
            p.h_stage_out, p.nsend,p.nrecv, p.offset_send,
            p.offset_recv, 0, p.comm,check_err); 
        }

    /* multi-dimensional FFT */
    cuda_mpifftnd_dif(p.gdim, p.pdim, p.ndim, p.pidx, dir,
        p.size_in,p.inembed,p.oembed, d_work, d_scratch,
        p.h_stage_in, p.h_stage_out,
        dir ? p.cuda_plans_short_inverse : p.cuda_plans_short_forward,
        dir ? p.cuda_plans_long_inverse : p.cuda_plans_long_forward,
        p.rho_L, p.rho_pk0, p.rho_Lk0, p.nsend,p.nrecv,
        p.offset_send,p.offset_recv, p.comm,check_err);

    if ((dir && !p.input_cyclic) || (!dir && !p.output_cyclic))
        {
        /* redistribution of output */
        cuda_redistribute_nd(p.gdim, p.pdim, p.ndim, p.pidx,
            p.size_out,p.oembed, d_work, d_scratch, p.h_stage_in, p.h_stage_out,
            p.nsend,p.nrecv, p.offset_send,p.offset_recv, 1, p.comm,check_err); 
        }

    if (out_of_place)
        {
        int size = p.size_out - p.delta_out;
        cudaMemcpy(d_out, d_work, sizeof(cuda_cpx_t)*size,cudaMemcpyDefault);
        if (check_err) CHECK_CUDA();
        }

    return 0;
    }

int dfft_cuda_create_plan(dfft_plan *p,
    int ndim, int *gdim,
    int *inembed, int *oembed, 
    int *pdim, int *pidx,
    int input_cyclic, int output_cyclic,
    MPI_Comm comm)
    {
    int res = dfft_create_plan_common(p, ndim, gdim, inembed, oembed,
        pdim, pidx, input_cyclic, output_cyclic, comm, 1);

    #ifndef ENABLE_MPI_CUDA
    /* allocate staging bufs */
    /* we need to use posix_memalign/cudaHostRegister instead
     * of cudaHostAlloc, because cudaHostAlloc doesn't have hooks
     * in the MPI library, and using it would lead to data corruption
     */
    int size = p->scratch_size*sizeof(cuda_cpx_t);
    int page_size = getpagesize();
    size = ((size + page_size - 1) / page_size) * page_size;
    posix_memalign((void **)&(p->h_stage_in),page_size,size);
    posix_memalign((void **)&(p->h_stage_out),page_size,size);
    cudaHostRegister(p->h_stage_in, size, cudaHostAllocDefault);
    cudaHostRegister(p->h_stage_out, size, cudaHostAllocDefault);
    #endif
    } 

void dfft_cuda_destroy_plan(dfft_plan plan)
    {
    dfft_destroy_plan_common(plan, 1);
    #ifndef ENABLE_MPI_CUDA
    cudaHostUnregister(plan.h_stage_in);
    cudaHostUnregister(plan.h_stage_out);
    free(plan.h_stage_in);
    free(plan.h_stage_out);
    #endif
    }

void dfft_cuda_check_errors(dfft_plan *plan, int check_err)
    {
    plan->check_cuda_errors = check_err;
    }
