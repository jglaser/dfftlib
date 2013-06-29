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
 * n-dimensional redistribute from group-cyclic with cycle c0 to cycle c1>=c0
 */
void dfft_cuda_redistribute_block_to_cyclic_nd(
                  int *dim,
                  int *pdim,
                  int ndim,
                  int *c0,
                  int *c1,
                  int* pidx,
                  int size_in,
                  int *embed,
                  int *d_c0,
                  int *d_c1,
                  int *d_pidx,
                  int *d_pdim,
                  int *d_embed,
                  int *d_length,
                  cuda_cpx_t *d_work,
                  cuda_cpx_t *d_scratch,
                  cuda_cpx_t *d_scratch_2,
                  cuda_cpx_t *h_stage_in,
                  cuda_cpx_t *h_stage_out,
                  int *nsend,
                  int *nrecv,
                  int *offset_send,
                  int *offset_recv,
                  MPI_Comm comm,
                  int check_err,
                  int row_m)
    {
    /* exit early if nothing needs to be done */
    int res = 1;
    int i;
    for (i = 0; i < ndim; ++i)
        if (!(c0[i] == c1[i])) res = 0;
    if (res) return;

    int pdim_tot=1;
    int k,t;
    for (k = 0; k < ndim; ++k)
        pdim_tot *= pdim[k];
    for (t = 0; t<pdim_tot; ++t)
        {
        nsend[t] = 0;
        nrecv[t] = 0;
        offset_send[t] = 0;
        offset_recv[t] = 0;
        }

    int roffs = 0;
    int soffs = 0;
    for (t = 0; t < pdim_tot; ++t)
        {
        int send_size = 1;
        int recv_size = 1;

        /* send and recv flags */
        int send =1;
        int recv=1;
        int current_dim;
        int tmp = t;
        int tmp_pidx = pdim_tot;
        for (current_dim = 0; current_dim < ndim; ++current_dim)
            {
            /* find coordinate of remote procesor along current dimension */
            if (!row_m)
                {
                tmp_pidx /= pdim[current_dim];
                i = tmp / tmp_pidx;
                tmp %= tmp_pidx;
                }
            else
                {
                i = (tmp % pdim[k]);
                tmp/=pdim[current_dim];
                }
            int length = dim[current_dim]/pdim[current_dim];

            /* processor index along current dimension */
            int s = pidx[current_dim];
            int j0_local = s % c0[current_dim];
            int j2_local = s / c0[current_dim];
            int j0_new_local = s % c1[current_dim];
            int j2_new_local = s / c1[current_dim];
            int p = pdim[current_dim];

            int ratio = c1[current_dim]/c0[current_dim];
            int size = ((length/ratio > 1) ? (length/ratio) : 1);

            /* initialize send offsets */
            int j0_remote = i % c0[current_dim];
            int j2_remote = i / c0[current_dim];
            int j0_new_remote = i % c1[current_dim];
            int j2_new_remote = i / c1[current_dim];

            send &= ((j0_local == (j0_new_remote % c0[current_dim]))
                && (j2_new_remote == j2_local / ratio));
            recv &= ((j0_remote == (j0_new_local % c0[current_dim]))
                && (j2_new_local == j2_remote / ratio));
            if (c1[current_dim] >= length*c0[current_dim])
                {
                send &= ((j0_new_remote / (length*c0[current_dim])) == (j2_local % (ratio/length)));
                recv &= ((j0_new_local / (length*c0[current_dim])) == (j2_remote % (ratio/length)));
                }
          
            /* determine packet length for current dimension */
            if (c1[current_dim] >= length*c0[current_dim])
                size = 1;
            else
                size = length*c0[current_dim]/c1[current_dim];

            recv_size *= (recv ? size : 0);
            send_size *= (send ? size : 0);
            } /* end loop over dimensions */

        nsend[t] = send_size*sizeof(cuda_cpx_t);
        nrecv[t] = recv_size*sizeof(cuda_cpx_t);
        offset_send[t] = soffs*sizeof(cuda_cpx_t);
        offset_recv[t] = roffs*sizeof(cuda_cpx_t);
        roffs += recv_size;
        soffs += send_size;
        } /* end loop over processors */

    cudaMemcpy(d_c0, c0, sizeof(int)*ndim,cudaMemcpyDefault);
    cudaMemcpy(d_c1, c1, sizeof(int)*ndim,cudaMemcpyDefault);

    /* pack data */
    gpu_b2c_pack_nd(size_in, d_c0, d_c1, ndim, d_embed,
        d_length, row_m, d_work, d_scratch);
    if (check_err) CHECK_CUDA();

    /* synchronize */
    MPI_Barrier(comm);

    /* communicate */
    #ifdef ENABLE_MPI_CUDA
    MPI_Alltoallv(d_scratch,nsend, offset_send, MPI_BYTE,
                  d_scratch_2, nrecv, offset_recv, MPI_BYTE,
                  comm);
    #else
    // stage into host buf
    cudaMemcpy(h_stage_in, d_scratch, sizeof(cuda_cpx_t)*npackets*size,cudaMemcpyDefault); 
    if (check_err) CHECK_CUDA();

    MPI_Alltoallv(h_stage_in,nsend, offset_send, MPI_BYTE,
                  h_stage_out,nrecv, offset_recv, MPI_BYTE,
                  comm);

    // copy back received data
    cudaMemcpy(d_scratch_2,h_stage_out, sizeof(cuda_cpx_t)*size_in,cudaMemcpyDefault); 
    if (check_err) CHECK_CUDA();
    #endif

    /* unpack data */
    gpu_b2c_unpack_nd(size_in, d_c0, d_c1, ndim, d_embed, d_length,
        row_m, d_scratch_2, d_work);
    if (check_err) CHECK_CUDA();
    }


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
                  int check_err,
                  int row_m)
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
        if (row_m)
            {
            for (k = ndim-1; k >=0 ;--k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? desti : pidx[k]);
                }
            }
        else
            {
            for (k = 0; k < ndim; ++k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? desti : pidx[k]);
                }
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
        if (row_m)
            {
            for (k = ndim-1; k >= 0; --k)
                {
                srcproc *= pdim[k];
                srcproc += ((current_dim == k) ? srci : pidx[k]);
                } 
            }
        else
            {
            for (k = 0; k < ndim; ++k)
                {
                srcproc *= pdim[k];
                srcproc += ((current_dim == k) ? srci : pidx[k]);
                }
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
                     int check_err,
                     int row_m
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
        if (row_m)
            {
            for (k = ndim-1; k >=0 ;--k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? i : pidx[k]);
                }
            }
        else
            {
            for (k = 0; k < ndim; ++k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? i : pidx[k]);
                }
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
            if (row_m)
                {
                for (k = ndim-1; k >=0 ;--k)
                    {
                    destproc *= pdim[k];
                    destproc += ((current_dim == k) ? i : pidx[k]);
                    }
                }
            else
                {
                for (k = 0; k < ndim; ++k)
                    {
                    destproc *= pdim[k];
                    destproc += ((current_dim == k) ? i : pidx[k]);
                    }
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
            int check_err,
            int row_m)
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
            dfft_offset_recv, comm, check_err, row_m);
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
            int check_err,
            int row_m)
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
            comm,check_err,row_m);

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
            int *c0,
            int *c1,
            int *d_c0,
            int *d_c1,
            int *d_pidx,
            int *d_pdim,
            int *d_embed,
            int *d_length,
            cuda_cpx_t *d_work,
            cuda_cpx_t *d_scratch,
            cuda_cpx_t *d_scratch_2,
            cuda_cpx_t *h_stage_in,
            cuda_cpx_t *h_stage_out,
            int *nsend,
            int *nrecv,
            int *offset_send,
            int *offset_recv,
            int c2b,
            MPI_Comm comm,
            int check_err,int row_m)
    {
    cuda_cpx_t *cur_work =d_work;
    cuda_cpx_t *cur_scratch =d_scratch;

    if (!c2b)
        {
        int i;
        for (i = 0; i < ndim; ++i)
            {
            /* block to cyclic */
            c0[i] = 1;
            c1[i] = pdim[i];
            }

        dfft_cuda_redistribute_block_to_cyclic_nd(dim,pdim,ndim, c0,
                  c1, pidx, size, embed, d_c0, d_c1, d_pidx,
                  d_pdim, d_embed, d_length, d_work, d_scratch,
                  d_scratch_2, h_stage_in, h_stage_out, nsend,
                  nrecv, offset_send, offset_recv, comm,
                  check_err, row_m);
        return;
        }
 
    int current_dim;
    for (current_dim = 0; current_dim < ndim; ++current_dim)
        {
        /* redistribute along one dimension (in-place) */
        if (!c2b)
            {
            dfft_cuda_redistribute_block_to_cyclic_1d(dim, pdim, ndim,
                current_dim, 1, pdim[current_dim], pidx, size, embed,
                cur_work, cur_scratch, h_stage_in, h_stage_out,
                nsend,nrecv, offset_send, offset_recv, comm,
                check_err,row_m);
            }
        else
            dfft_cuda_redistribute_cyclic_to_block_1d(dim, pdim, ndim,
                current_dim, pdim[current_dim], 1, pidx, 0, size, embed,
                cur_work, cur_scratch, h_stage_in, h_stage_out,
                NULL, NULL, nsend, nrecv, offset_send,
                offset_recv, comm, check_err,row_m);
        
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
            p.size_in, p.inembed, p.c0, p.c1, p.d_c0, p.d_c1,
            p.d_pidx, p.d_pdim, p.d_iembed, p.d_length,
            d_work, d_scratch, p.d_scratch_3, p.h_stage_in,
            p.h_stage_out, p.nsend,p.nrecv, p.offset_send,
            p.offset_recv, 0, p.comm,check_err,p.row_m); 
        }

    /* multi-dimensional FFT */
    cuda_mpifftnd_dif(p.gdim, p.pdim, p.ndim, p.pidx, dir,
        p.size_in,p.inembed,p.oembed, d_work, d_scratch,
        p.h_stage_in, p.h_stage_out,
        dir ? p.cuda_plans_short_inverse : p.cuda_plans_short_forward,
        dir ? p.cuda_plans_long_inverse : p.cuda_plans_long_forward,
        p.rho_L, p.rho_pk0, p.rho_Lk0, p.nsend,p.nrecv,
        p.offset_send,p.offset_recv, p.comm,check_err,p.row_m);

    if ((dir && !p.input_cyclic) || (!dir && !p.output_cyclic))
        {
        /* redistribution of output */
        cuda_redistribute_nd(p.gdim, p.pdim, p.ndim, p.pidx,
            p.size_out, p.oembed, p.c0, p.c1, p.d_c0, p.d_c1,
            p.d_pidx, p.d_pdim, p.d_oembed, p.d_length,
            d_work, d_scratch, p.d_scratch_3, p.h_stage_in,
            p.h_stage_out, p.nsend,p.nrecv, p.offset_send,
            p.offset_recv, 1, p.comm,check_err,p.row_m); 
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
    int *pdim, int *pidx, int row_m,
    int input_cyclic, int output_cyclic,
    MPI_Comm comm)
    {
    int res = dfft_create_plan_common(p, ndim, gdim, inembed, oembed,
        pdim, pidx, row_m, input_cyclic, output_cyclic, comm, 1);

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
    CHECK_CUDA();
    cudaHostRegister(p->h_stage_out, size, cudaHostAllocDefault);
    CHECK_CUDA();
    #endif

    /* allocate memory for passing variables */
    cudaMalloc((void **)&(p->d_c0), sizeof(int)*ndim);
    CHECK_CUDA();
    cudaMalloc((void **)&(p->d_c1), sizeof(int)*ndim);
    CHECK_CUDA();
    cudaMalloc((void **)&(p->d_pidx), sizeof(int)*ndim);
    CHECK_CUDA();
    cudaMalloc((void **)&(p->d_pdim), sizeof(int)*ndim);
    CHECK_CUDA();
    cudaMalloc((void **)&(p->d_iembed), sizeof(int)*ndim);
    CHECK_CUDA();
    cudaMalloc((void **)&(p->d_oembed), sizeof(int)*ndim);
    CHECK_CUDA();
    cudaMalloc((void **)&(p->d_length), sizeof(int)*ndim);
    CHECK_CUDA();

    /* initialize cuda buffers */
    int *h_length = (int *)malloc(sizeof(int)*ndim);
    int i;
    for (i = 0; i < ndim; ++i)
        h_length[i] = gdim[i]/pdim[i];
    cudaMemcpy(p->d_pidx, pidx, sizeof(int)*ndim, cudaMemcpyDefault); 
    CHECK_CUDA();
    cudaMemcpy(p->d_pdim, pdim, sizeof(int)*ndim, cudaMemcpyDefault); 
    CHECK_CUDA();
    cudaMemcpy(p->d_iembed, p->inembed, sizeof(int)*ndim, cudaMemcpyDefault); 
    CHECK_CUDA();
    cudaMemcpy(p->d_oembed, p->oembed, sizeof(int)*ndim, cudaMemcpyDefault); 
    CHECK_CUDA();
    cudaMemcpy(p->d_length, h_length, sizeof(int)*ndim, cudaMemcpyDefault); 
    CHECK_CUDA();
    free(h_length);

    return res;
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

    cudaFree(plan.d_c0);
    cudaFree(plan.d_c1);
    cudaFree(plan.d_pidx);
    cudaFree(plan.d_pdim);
    cudaFree(plan.d_iembed);
    cudaFree(plan.d_oembed);
    cudaFree(plan.d_length);
    }

void dfft_cuda_check_errors(dfft_plan *plan, int check_err)
    {
    plan->check_cuda_errors = check_err;
    }
