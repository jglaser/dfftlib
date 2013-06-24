/*
 * Include interface definitions for local FFT libraries
 */

#ifdef ENABLE_HOST
/* Local FFT library for host DFFT */
/* MKL, single precision is the default library*/
#include "mkl_single_interface.h"
#endif

#ifdef ENABLE_CUDA
/* CUFFT is the default library */
#include "cufft_single_interface.h"
#endif

