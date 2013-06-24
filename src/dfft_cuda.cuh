#include <dfft_lib_config.h>
#include "dfft_local_fft_config.h"

#ifdef __cplusplus
#define EXTERN_DFFT extern "C"
#else
#define EXTERN_DFFT
#endif

EXTERN_DFFT void gpu_b2c_pack(unsigned int local_size,
                  unsigned int ratio,
                  unsigned int size,
                  unsigned int npackets,
                  unsigned int stride,
                  cuda_cpx_t *local_data,
                  cuda_cpx_t *send_data);

EXTERN_DFFT void gpu_twiddle(unsigned int np,
                 const unsigned int length,
                 const unsigned int stride,
                 float alpha,
                 cuda_cpx_t *d_in,
                 cuda_cpx_t *d_out,
                 int inv);
 
EXTERN_DFFT void gpu_c2b_unpack(const unsigned int local_size,
                    const unsigned int length,
                    const unsigned int c0,
                    const unsigned int c1, 
                    const unsigned int size,
                    const unsigned int j0,
                    const unsigned int stride,
                    const int rev,
                    cuda_cpx_t *d_local_data,
                    const cuda_cpx_t *d_scratch);

EXTERN_DFFT void gpu_transpose(const unsigned int local_size,
                   const unsigned int length,
                   const unsigned int stride,
                   const unsigned int embed,
                   const cuda_cpx_t *in,
                   cuda_cpx_t *out);

#undef EXTERN_DFFT
