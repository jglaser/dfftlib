#include <cuda.h>
#include "dfft_cuda.cuh"

// redistribute between group-cyclic distributions with different cycles
// (direction from block to cyclic)
__global__ void gpu_b2c_pack_kernel(unsigned int local_size,
                                    unsigned int ratio,
                                    unsigned int size,
                                    unsigned int npackets,
                                    unsigned int stride,
                                    cuda_cpx_t *local_data,
                                    cuda_cpx_t *send_data
                                    )
    {
    // index of local component
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // do not read beyond end of array
    if (idx >= local_size) return;

    unsigned int j = (idx/stride) % npackets; // packet number
    unsigned int r = (idx/stride - j)/ratio; // index in packet

    unsigned int offset = j*size;
    send_data[offset + r*stride + (idx%stride)] = local_data[idx];
    }

void gpu_b2c_pack(unsigned int local_size,
                  unsigned int ratio,
                  unsigned int size,
                  unsigned int npackets,
                  unsigned int stride,
                  cuda_cpx_t *local_data,
                  cuda_cpx_t *send_data)
    {
    unsigned int block_size =512;
    unsigned int n_blocks = local_size/block_size;
    if (local_size % block_size) n_blocks++;

    gpu_b2c_pack_kernel<<<n_blocks, block_size>>>(local_size,
                                                  ratio,
                                                  size,
                                                  npackets,
                                                  stride,
                                                  local_data,
                                                  send_data);
    }

// apply twiddle factors
__global__ void gpu_twiddle_kernel(unsigned int local_size,
                                   const unsigned int length,
                                   const unsigned int stride,
                                   float alpha,
                                   cuda_cpx_t *d_in,
                                   cuda_cpx_t *d_out,
                                   int inv)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= local_size) return;

    int j = idx/stride;
    if (j >= length) return;
    float theta = -2.0f * float(M_PI) * alpha/(float) length;
    cuda_cpx_t w;
    CUDA_RE(w) = cosf((float)j*theta);
    CUDA_IM(w) = sinf((float)j*theta);

    cuda_cpx_t in = d_in[idx];
    cuda_cpx_t out;
    float sign = inv ? -1.0f : 1.0f;

    w.y *= sign;

    CUDA_RE(out) = CUDA_RE(in) * CUDA_RE(w) - CUDA_IM(in) * CUDA_IM(w);
    CUDA_IM(out) = CUDA_RE(in) * CUDA_IM(w) + CUDA_IM(in) * CUDA_RE(w); 

    d_out[idx] = out;
    }

void gpu_twiddle(unsigned int local_size,
                 const unsigned int length,
                 const unsigned int stride,
                 float alpha,
                 cuda_cpx_t *d_in,
                 cuda_cpx_t *d_out,
                 int inv)
    {
    unsigned int block_size =512;
    unsigned int n_block = local_size/block_size;
    if (local_size % block_size ) n_block++;

    gpu_twiddle_kernel<<<n_block, block_size>>>(local_size,
                                                length,
                                                stride,
                                                alpha,
                                                d_in,
                                                d_out,
                                                inv);
}

__global__ void gpu_c2b_unpack_kernel(const unsigned int local_size,
                                      const unsigned int length,
                                      const unsigned int c0,
                                      const unsigned int c1, 
                                      const unsigned int size,
                                      const unsigned int j0,
                                      const unsigned int stride,
                                      int rev,
                                      cuda_cpx_t *d_local_data,
                                      const cuda_cpx_t *d_scratch)
    {
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;

    if (idx >= local_size) return;

    // source processor
    int r = idx/size; // packet index
    int j1, j1_offset, del;
    int j0_remote = j0 + r*c1;
    if (rev && (length >= c0))
        {
        j1_offset = j0_remote*length/c0;
        del = 1;
        }
    else
        {
        j1_offset = j0_remote/c1;
        del = c0/c1;
        }

    // local index
    j1 = j1_offset + ((idx%size)/stride)*del;
    
    d_local_data[j1*stride+idx%stride] = d_scratch[idx];
    }

void gpu_c2b_unpack(const unsigned int local_size,
                    const unsigned int length,
                    const unsigned int c0,
                    const unsigned int c1, 
                    const unsigned int size,
                    const unsigned int j0,
                    const unsigned int stride,
                    const int rev,
                    cuda_cpx_t *d_local_data,
                    const cuda_cpx_t *d_scratch)
    {
    unsigned int block_size =512;
    unsigned int n_block = local_size/block_size;
    if (local_size % block_size ) n_block++;

    gpu_c2b_unpack_kernel<<<n_block, block_size>>>(local_size,
                                                   length,
                                                   c0,
                                                   c1,
                                                   size,
                                                   j0,
                                                   stride,
                                                   rev,
                                                   d_local_data,
                                                   d_scratch);
    }

__global__ void gpu_transpose_kernel(const unsigned int size,
                                     const unsigned int length,
                                     const unsigned int stride,
                                     const unsigned int embed,
                                     const cuda_cpx_t *in,
                                     cuda_cpx_t *out)
    {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= size) return;

    int i = idx / stride;
    if (i >= length) return;

    int j = idx % stride;

    out[j*embed + i] = in[idx];
    }

void gpu_transpose(const unsigned int size,
                   const unsigned int length,
                   const unsigned int stride,
                   const unsigned int embed,
                   const cuda_cpx_t *in,
                   cuda_cpx_t *out)
    {
    unsigned int block_size =512;
    unsigned int n_block = size/block_size;
    if (size % block_size ) n_block++;
    
    gpu_transpose_kernel<<<n_block, block_size>>>(size, length, stride, embed, in, out);
    }
