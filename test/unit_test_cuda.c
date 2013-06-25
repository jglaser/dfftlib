#include <math.h>
#include <mpi.h>
#include "dfft_cuda.h"
#include "kiss_fftnd.h"
#include <cuda_runtime.h>

#include "test_common.h"

void test_distributed_fft_nd(int nd);
void test_distributed_fft_1d_compare(int n);
void test_distributed_fft_3d_compare();

char enable_cuda[] = "MV2_USE_CUDA=1";
char enable_threads[] = "MV2_ENABLE_AFFINITY=0";

int main(int argc, char **argv)
    {
    int gpu = 0;
    char *str = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (!str) str =getenv("OMPI_COMM_WORLD_LOCAL_RANK");

    if (str) gpu=atoi(str);

    str =getenv("MV2_COMM_WORLD_RANK");
    if (!str) str =getenv("OMPI_COMM_WORLD_RANK");

    int s_env = 0;
    if (str) s_env = atoi(str);
    printf("Rank %d running on GPU %d\n",s_env, gpu);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaSetDevice(gpu);
    /* initialize context */
    cudaFree(0);

    putenv(enable_cuda);
    putenv(enable_threads);
    MPI_Init(&argc, &argv);

    int s,p;
    MPI_Comm_rank(MPI_COMM_WORLD, &s);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    /* basic test in n = 1 .. 7 dimensions */
    int nd;
    for (nd = 1; nd <= 7; nd++)
        {
        if (!s) printf("Testing distributed FFT in %d dimensions ...\n",nd);
        test_distributed_fft_nd(nd);
        }

    if (!s) printf("Compare against KISS FFT (d=1)...\n");
    int i;
    for (i = 1; i < 24; ++i)
        {
        int n = (1 << i);
        if (n <= p) continue;
        if (!s) printf("N=%d\n",n);
        test_distributed_fft_1d_compare(n);
        }

    if (!s) printf("Compare against KISS FFT (d=3)... \n");
    test_distributed_fft_3d_compare();
    MPI_Finalize();
    }

/* Basic functionality test for distributed 3D FFT
   This tests the integrity of a forward transform,
   followed by an inverse transform
 */
void test_distributed_fft_nd(int nd)
    {
    int p,s;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &s);

    int *pdim;
    pdim = (int *) malloc(sizeof(int)*nd);

    /* choose a decomposition */
    int r = powf(p,1.0/(double)nd)+0.5;
    int root = 1;
    while (root < r) root*=2;
    int ptot = 1;
    if (!s) printf("Processor grid: ");
    int i;
    for (i = 0; i < nd-1; ++i)
        {
        pdim[i] = (((ptot*root) > p) ? 1 : root);
        ptot *= pdim[i];
        if (!s) printf("%d x ",pdim[i]);
        }
    pdim[nd-1] = p/ptot;
    if (!s) printf("%d\n", pdim[nd-1]);

    /* determine processor index */
    int *pidx;
    pidx = (int*)malloc(nd*sizeof(int));
    int idx = s;
    for (i = nd-1; i >= 0; --i)
        {
        pidx[i] = idx % pdim[i];
        idx /= pdim[i];
        }

    double tol = 0.001;
    double abs_tol = 0.2;
    int *dim_glob;
    dim_glob = (int *) malloc(sizeof(int)*nd);

    /* test with a 1x1 matrix, but only if we are executing on a single rank */
    if (p == 1)
        {
        cuda_cpx_t *in_1_d;
        cuda_cpx_t *in_1_h;
        in_1_h = (cuda_cpx_t *) malloc(sizeof(cuda_cpx_t)*1);
        cudaMalloc((void **)&in_1_d,sizeof(cuda_cpx_t)*1);

        for (i = 0; i < nd; ++i)
            dim_glob[i] = 1*pdim[i];
     
        CUDA_RE(in_1_h[0]) = (double) s;
        CUDA_IM(in_1_h[0]) = 0.0f;
       
        /* copy to device */
        cudaMemcpy(in_1_d, in_1_h, sizeof(cuda_cpx_t)*1,cudaMemcpyDefault);

        dfft_plan plan_1;

        int pidx[1];
        pidx[0] =s;
        dfft_cuda_create_plan(&plan_1, nd, dim_glob, NULL, NULL, pdim, pidx,
            0, 0, 0, MPI_COMM_WORLD);

        /* forward transform */
        dfft_cuda_execute(in_1_d, in_1_d, 0, plan_1);

        /* backward transform */
        dfft_cuda_execute(in_1_d, in_1_d, 1, plan_1);

        /* copy back to host */
        cudaMemcpy(in_1_h, in_1_d, sizeof(cuda_cpx_t)*1,cudaMemcpyDefault);

        if (s == 0)
            {
            CHECK_SMALL(CUDA_RE(in_1_h[0]),abs_tol);
            CHECK_SMALL(CUDA_IM(in_1_h[0]),abs_tol);
            }
        else
            { 
            CHECK_CLOSE(CUDA_RE(in_1_h[0]),(double)s/(double)p,tol);
            CHECK_SMALL(CUDA_IM(in_1_h[0]),abs_tol);
            }

        cudaFree(in_1_d);
        free(in_1_h);
        dfft_cuda_destroy_plan(plan_1);
        }

    /* now do a test with a 2^n local matrix */
    cuda_cpx_t *in_2_d;
    cuda_cpx_t *in_2_h;
    int size_2n = 1;
    for (i = 0; i < nd; ++i)
        size_2n *= 2; 

    cudaMalloc((void **)&in_2_d,sizeof(cuda_cpx_t)*size_2n);
    in_2_h = (cuda_cpx_t *) malloc(sizeof(cuda_cpx_t)*size_2n);

    for (i = 0; i < size_2n; ++i)
        {
        CUDA_RE(in_2_h[i]) =(i+s*(double)size_2n);
        CUDA_IM(in_2_h[i]) =0.0f;
        }

    /* copy data to device */
    cudaMemcpy(in_2_d, in_2_h, sizeof(cuda_cpx_t)*size_2n,cudaMemcpyDefault);

    for (i = 0; i < nd; ++i)
        dim_glob[i] = 2*pdim[i];

    for (i = 0; i < nd-1; ++i)
        if (!s) printf("%d x ",dim_glob[i]);
    if (!s) printf("%d matrix\n", dim_glob[nd-1]);
  
    dfft_plan plan_2;
    dfft_cuda_create_plan(&plan_2, nd, dim_glob, NULL, NULL, pdim, pidx, 0, 0, 0, MPI_COMM_WORLD);

    /* forward transfom */
    dfft_cuda_execute(in_2_d, in_2_d, 0, plan_2);

    /* inverse transform */
    dfft_cuda_execute(in_2_d, in_2_d, 1, plan_2);

    /* copy back to host */
    cudaMemcpy(in_2_h, in_2_d, sizeof(cuda_cpx_t)*size_2n, cudaMemcpyDefault);

    for (i = 0; i < size_2n; ++i)
        {
        if (s == 0 && i == 0)
            {
            CHECK_SMALL(CUDA_RE(in_2_h[i]), abs_tol);
            CHECK_SMALL(CUDA_IM(in_2_h[i]), abs_tol);
            }
        else
            {
            CHECK_CLOSE(CUDA_RE(in_2_h[i]), size_2n*p*(i+(double)size_2n*(double)s),tol);
            CHECK_SMALL(CUDA_IM(in_2_h[i]), abs_tol);
            }
        }

    cudaFree(in_2_d);
    free(in_2_h);
    dfft_cuda_destroy_plan(plan_2);

    // test with a padded 4^n local matrix (two ghost cells per axis)
    cuda_cpx_t *in_3_d;
    cuda_cpx_t *out_3_d;
    cuda_cpx_t *in_3_h;

    int *inembed;
    inembed = (int *) malloc(sizeof(int)*nd);

    int size_4n_embed = 1;
    int size_4n = 1;
    for (i = 0; i < nd; ++i)
        {
        inembed[i] = 4;
        size_4n_embed *= 4;
        size_4n *= 2;
        dim_glob[i]= 2*pdim[i];
        }

    for (i = 0; i < nd-1; ++i)
        if (!s) printf("%d x ",dim_glob[i]);
    if (!s) printf("%d matrix with padding\n", dim_glob[nd-1]);
 
    cudaMalloc((void **)&in_3_d,sizeof(cuda_cpx_t)*size_4n_embed);
    cudaMalloc((void **)&out_3_d,sizeof(cuda_cpx_t)*size_4n);
    in_3_h = (cuda_cpx_t *) malloc(sizeof(cuda_cpx_t)*size_4n_embed);

    int *lidx;
    int *gidx;
    lidx = (int *) malloc(nd*sizeof(int));
    gidx = (int *) malloc(nd*sizeof(int));

    for (i = 0; i < size_4n_embed; ++i)
        {
        //processor grid in column major
        int idx = i;

        // find local coordinate tuple
        int j;
        for (j = nd-1; j >= 0; --j)
            {
            int length = inembed[j];
            lidx[j] = idx % length;
            idx /= length;
            }

        // global coordinates in block distribution
        for (j = 0; j<nd; ++j)
            gidx[j] = lidx[j] + inembed[j]*pidx[j];

        int val=0;
        for (j = 0; j<nd; ++j)
            {
            val *= inembed[j];
            val += gidx[j];
            }

        int ghost_cell = 0;
        for (j = 0; j < nd; ++j)
            if (lidx[j] == 0 || lidx[j] == 3) ghost_cell = 1;

        if (ghost_cell)
            {
            CUDA_RE(in_3_h[i]) = 99.0;
            CUDA_IM(in_3_h[i]) = 123.0;
            }
        else
            {
            CUDA_RE(in_3_h[i]) = (double) val;
            CUDA_IM(in_3_h[i]) = 0.0;               
            }
        }

    /* copy data to device */
    cudaMemcpy(in_3_d, in_3_h, sizeof(cuda_cpx_t)*size_4n_embed,cudaMemcpyDefault);

    dfft_plan plan_3_fw,plan_3_bw;
    dfft_cuda_create_plan(&plan_3_fw, nd, dim_glob, inembed, NULL, pdim, pidx, 0, 0, 0, MPI_COMM_WORLD);
    dfft_cuda_create_plan(&plan_3_bw, nd, dim_glob, NULL, inembed, pdim, pidx, 0, 0, 0, MPI_COMM_WORLD);
   
    int offset = 0;
    int n_ghost = 2;
    for (i = 0; i < nd; i++)
        {
        offset *= 4;
        offset += n_ghost/2;
        }

    /* forward transform */
    dfft_cuda_execute(in_3_d+offset, out_3_d, 0, plan_3_fw);
        
    /* inverse transform */
    dfft_cuda_execute(out_3_d,in_3_d+offset, 1, plan_3_bw);

    /* copy data back to host */
    cudaMemcpy(in_3_h, in_3_d, sizeof(cuda_cpx_t)*size_4n_embed,cudaMemcpyDefault);

    /* check results */
    for (i = 0; i < size_4n_embed; ++i)
        {
        //processor grid in column major
        int idx = i;

        // find local coordinate tuple
        int j;
        for (j = nd-1; j >= 0; --j)
            {
            int length = inembed[j];
            lidx[j] = idx % length;
            idx /= length;
            }

        // global coordinates in block distribution
        for (j = 0; j<nd; ++j)
            gidx[j] = lidx[j] + inembed[j]*pidx[j];

        int val=0;
        for (j = 0; j<nd; ++j)
            {
            val *= inembed[j];
            val += gidx[j];
            }

        int ghost_cell = 0;
        for (j = 0; j < nd; ++j)
            if (lidx[j] == 0 || lidx[j] == 3) ghost_cell = 1;

        if (ghost_cell)
            {
            //CHECK_CLOSE(CUDA_RE(in_3[i]),99.0,tol);
            //CHECK_CLOSE(CUDA_IM(in_3[i]), 123.0,tol);
            }
        else
            {
            CHECK_CLOSE(CUDA_RE(in_3_h[i]), (double) val*(double)p*(double)size_4n,tol);
            CHECK_SMALL(CUDA_IM(in_3_h[i]), abs_tol);
            }
        }

    cudaFree(in_3_d);
    cudaFree(out_3_d);
    free(in_3_h);
    free(lidx);
    free(gidx);

    dfft_cuda_destroy_plan(plan_3_fw);
    dfft_cuda_destroy_plan(plan_3_bw);

    free(pidx);
    free(pdim);
    free(dim_glob);
    }

/* Compare a 1d FFT against a reference FFT */
void test_distributed_fft_1d_compare(int n)
    {
    int s,p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &s);

    int pdim[1];
    pdim[0]=p;

    /* determine processor index */
    int pidx[1];
    pidx[0]=s;

    double tol = 0.01;
    double abs_tol = .1;
    int dim_glob[1];
    dim_glob[0] = n;

    // Do a size n FFT (n = power of two)
    kiss_fft_cpx *in_kiss;
    in_kiss = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx)*n);

    srand(45678);

    // fill vector with complex random numbers in row major order
    int i;
    for (i = 0; i < n; ++i)
        {
        in_kiss[i].r = (float)rand()/(float)RAND_MAX;
        in_kiss[i].i =(float)rand()/(float)RAND_MAX;
        }

    kiss_fft_cpx *out_kiss;
    out_kiss = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx)*n);

    // construct forward transform
    kiss_fft_cfg cfg = kiss_fft_alloc(n,0,NULL,NULL);

    // carry out conventional FFT
    kiss_fft(cfg, in_kiss, out_kiss);

    // compare to distributed FFT
    cuda_cpx_t * in_d, *in_h;
    cudaMalloc((void **)&in_d,sizeof(cuda_cpx_t)*n/p);
    in_h = (cuda_cpx_t *) malloc(sizeof(cuda_cpx_t)*n/p);

    for (i = 0; i < n/p; ++i)
        {
        CUDA_RE(in_h[i]) = in_kiss[s*n/p+i].r;
        CUDA_IM(in_h[i]) = in_kiss[s*n/p+i].i;
        }

    cuda_cpx_t *out_d,*out_h;
    cudaMalloc((void **)&out_d,sizeof(cuda_cpx_t)*n/p);
    out_h = (cuda_cpx_t *)malloc(sizeof(cuda_cpx_t)*n/p);
    
    dfft_plan plan;
    dfft_cuda_create_plan(&plan,1, dim_glob, NULL, NULL, pdim, pidx,0, 0, 0, MPI_COMM_WORLD);

    /* copy data to device */
    cudaMemcpy(in_d, in_h, sizeof(cuda_cpx_t)*n/p, cudaMemcpyDefault);

    /* forward transform */
    dfft_cuda_execute(in_d, out_d, 0, plan);

    /* copy data back to host */
    cudaMemcpy(out_h, out_d, sizeof(cuda_cpx_t)*n/p, cudaMemcpyDefault);

    /* do comparison */
    for (i = 0; i < n/p; ++i)
        {

        int j = s*n/p + i;

        double re = CUDA_RE(out_h[i]);
        double im = CUDA_IM(out_h[i]);
        double re_kiss = out_kiss[j].r;
        double im_kiss = out_kiss[j].i;  

        if (fabs(re_kiss) < abs_tol)
            {
            CHECK_SMALL(re,2*abs_tol);
            }
        else
            {
            CHECK_CLOSE(re,re_kiss, tol);
            }

        if (fabs(im_kiss) < abs_tol)
            {
            CHECK_SMALL(im,2*abs_tol);
            }
        else
            {
            CHECK_CLOSE(im, im_kiss, tol);
            }
        }
    free(in_kiss);
    free(out_kiss);
    cudaFree(out_d);
    cudaFree(in_d);
    free(in_h);
    free(out_h);
    dfft_cuda_destroy_plan(plan);
    }
 
/* Compare a 3d FFT against a reference FFT */
void test_distributed_fft_3d_compare()
    {
    int s,p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &s);

    int nd = 3;
    int *pdim;
    pdim = (int *) malloc(sizeof(int)*nd);
    /* choose a decomposition */
    int r = powf(p,1.0/(double)nd)+0.5;
    int root = 1;
    while (root < r) root*=2;
    int ptot = 1;
    if (!s) printf("Processor grid: ");
    int i;
    for (i = 0; i < nd-1; ++i)
        {
        pdim[i] = (((ptot*root) > p) ? 1 : root);
        ptot *= pdim[i];
        if (!s) printf("%d x ",pdim[i]);
        }
    pdim[nd-1] = p/ptot;
    if (!s) printf("%d\n", pdim[nd-1]);

    /* determine processor index */
    int *pidx;
    pidx = (int*)malloc(nd*sizeof(int));
    int idx = s;
    for (i = nd-1; i >= 0; --i)
        {
        pidx[i] = idx % pdim[i];
        idx /= pdim[i];
        }

    double tol = 0.001;
    double abs_tol = 0.2;
    int *dim_glob;
    dim_glob = (int *) malloc(sizeof(int)*nd);

    // Do a pdim[0]*4 x pdim[1]* 8 x pdim[2] * 16 FFT (powers of two)
    int local_nx = 4;
    int local_ny = 8;
    int local_nz = 16;
    dim_glob[0] = pdim[0]*local_nx;
    dim_glob[1] = pdim[1]*local_ny;
    dim_glob[2] = pdim[2]*local_nz;

    for (i = 0; i < nd-1; ++i)
        if (!s) printf("%d x ",dim_glob[i]);
    if (!s) printf("%d matrix\n", dim_glob[nd-1]);
 
    kiss_fft_cpx *in_kiss;
    in_kiss = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx)*dim_glob[0]*dim_glob[1]*dim_glob[2]);

    srand(12345);

    // fill table with complex random numbers in row major order
    int x,y,z;
    int nx = dim_glob[0];
    int ny = dim_glob[1];
    int nz = dim_glob[2];
    for (x = 0; x < dim_glob[0]; ++x)
        for (y = 0; y < dim_glob[1]; ++y)
            for (z = 0; z < dim_glob[2]; ++z)
                {
                // KISS has column-major storage
                in_kiss[z+nz*(y+ny*x)].r = (float)rand()/(float)RAND_MAX;
                in_kiss[z+nz*(y+ny*x)].i =(float)rand()/(float)RAND_MAX;
                }

    kiss_fft_cpx *out_kiss;
    out_kiss = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx)*dim_glob[0]*dim_glob[1]*dim_glob[2]);

    // construct forward transform
    kiss_fftnd_cfg cfg = kiss_fftnd_alloc(dim_glob,3,0,NULL,NULL);

    // carry out conventional FFT
    kiss_fftnd(cfg, in_kiss, out_kiss);

    // compare to distributed FFT
    cuda_cpx_t * in_d, *in_h;
    cudaMalloc((void **)&in_d,sizeof(cuda_cpx_t)*local_nx*local_ny*local_nz);
    in_h = (cuda_cpx_t *) malloc(sizeof(cuda_cpx_t)*local_nx*local_ny*local_nz);

    int x_local, y_local, z_local;
    for (x = 0; x < nx; ++x)
        for (y = 0; y < ny; ++y)
            for (z = 0; z < nz; ++z)
                {
                if (x>=pidx[0]*local_nx && x < (pidx[0]+1)*local_nx && 
                    y>=pidx[1]*local_ny && y < (pidx[1]+1)*local_ny &&
                    z>=pidx[2]*local_nz && z < (pidx[2]+1)*local_nz)
                    {
                    x_local = x - pidx[0]*local_nx;
                    y_local = y - pidx[1]*local_ny;
                    z_local = z - pidx[2]*local_nz;
                   
                    CUDA_RE(in_h[z_local+local_nz*(y_local+local_ny*x_local)]) =
                        in_kiss[z+nz*(y+ny*x)].r;
                    CUDA_IM(in_h[z_local+local_nz*(y_local+local_ny*x_local)]) =
                        in_kiss[z+nz*(y+ny*x)].i;
                    }
                }

    cuda_cpx_t *out_d, *out_h;
    cudaMalloc((void **)&out_d,sizeof(cuda_cpx_t)*local_nx*local_ny*local_nz);
    out_h = (cuda_cpx_t *) malloc(sizeof(cuda_cpx_t)*local_nx*local_ny*local_nz);

    dfft_plan plan;
    dfft_cuda_create_plan(&plan,3, dim_glob, NULL, NULL, pdim, pidx,0, 0, 0, MPI_COMM_WORLD);

    /* copy data to device */
    cudaMemcpy(in_d, in_h, sizeof(cuda_cpx_t)*local_nx*local_ny*local_nz, cudaMemcpyDefault);

    // forward transform
    dfft_cuda_execute(in_d, out_d, 0, plan);

    /* copy data back to host */
    cudaMemcpy(out_h, out_d, sizeof(cuda_cpx_t)*local_nx*local_ny*local_nz, cudaMemcpyDefault);

    // do comparison
    int n_wave_local = local_nx * local_ny * local_nz;
    for (i = 0; i < n_wave_local; ++i)
        {

        x_local = i / local_ny / local_nz;
        y_local = (i - x_local*local_ny*local_nz)/local_nz;
        z_local = i % local_nz;

        x = pidx[0]*local_nx + x_local;
        y = pidx[1]*local_ny + y_local;
        z = pidx[2]*local_nz + z_local;

        double re = CUDA_RE(out_h[i]);
        double im = CUDA_IM(out_h[i]);
        double re_kiss = out_kiss[z+nz*(y+ny*x)].r;
        double im_kiss = out_kiss[z+nz*(y+ny*x)].i;  

        if (fabs(re_kiss) < abs_tol)
            {
            CHECK_SMALL(re,2*abs_tol);
            }
        else
            {
            CHECK_CLOSE(re,re_kiss, tol);
            }

        if (fabs(im_kiss) < abs_tol)
            {
            CHECK_SMALL(im,2*abs_tol);
            }
        else
            {
            CHECK_CLOSE(im, im_kiss, tol);
            }
        }
    free(in_kiss);
    free(out_kiss);
    cudaFree(out_d);
    cudaFree(in_d);
    free(in_h);
    free(out_h);
    free(pidx);
    free(dim_glob);
    dfft_cuda_destroy_plan(plan);
    }
