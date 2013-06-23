#include <math.h>
#include <mpi.h>
#include "dfft_host.h"
#include "kiss_fftnd.h"

#define CHECK_CLOSE(x,y,rel_tol) \
    {if (!(((x>=0.0) && (x*(1.0+rel_tol) >= y) && (x*(1.0-rel_tol) <= y)) || \
     ((x <0.0) && (x*(1.0+rel_tol) <= y) && (x*(1.0-rel_tol) >= y))))        \
        {                                                                    \
        printf("Test failed at line %d! %f != %f (rel tol: %f)\n",           \
            __LINE__, x, y, rel_tol);                                        \
        exit(1);                                                             \
        }                                                                    \
    }

#define CHECK_SMALL(x,abs_tol) \
    { if (!(copysign(x,1.0) <= abs_tol))                                     \
        {                                                                    \
        printf("Test failed at line %d! abs(%f) >= %f \n",                   \
            __LINE__, x, abs_tol);                                           \
        exit(1);                                                             \
        }                                                                    \
    }
        
void test_distributed_fft_nd(int nd);

int main(int argc, char **argv)
    {
    MPI_Init(&argc, &argv);

    int s;
    MPI_Comm_rank(MPI_COMM_WORLD, &s);
    /* basic test in n = 1 .. 7 dimensions */
    int nd;
    for (nd = 1; nd <= 7; nd++)
        {
        if (!s) printf("Testing distributed FFT in %d dimensions ...\n",nd);
        test_distributed_fft_nd(nd);
        }

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
        cpx_t *in_1;
        in_1 = (cpx_t *) malloc(sizeof(cpx_t)*1);

        for (i = 0; i < nd; ++i)
            dim_glob[i] = 1*pdim[i];
     
        RE(in_1[0]) = (double) s;
        IM(in_1[0]) = 0.0f;

        dfft_plan plan_1;

        dfft_create_plan(&plan_1, nd, dim_glob, NULL, NULL, pdim, 0, 0,
            MPI_COMM_WORLD);

        /* forward transform */
        dfft_execute(in_1, in_1, 0, plan_1);

        /* backward transform */
        dfft_execute(in_1, in_1, 1, plan_1);

        if (s == 0)
            {
            CHECK_SMALL(RE(in_1[0]),abs_tol);
            CHECK_SMALL(IM(in_1[0]),abs_tol);
            }
        else
            { 
            CHECK_CLOSE(RE(in_1[0]),(double)s/(double)p,tol);
            CHECK_SMALL(IM(in_1[0]),abs_tol);
            }

        free(in_1);
        dfft_destroy_plan(plan_1);
        }

    /* now do a test with a 2^n local matrix */
    cpx_t *in_2;
    int size_2n = 1;
    for (i = 0; i < nd; ++i)
        size_2n *= 2; 

    in_2 = (cpx_t *) malloc(sizeof(cpx_t)*size_2n);

    for (i = 0; i < size_2n; ++i)
        {
        RE(in_2[i]) =(i+s*(double)size_2n);
        IM(in_2[i]) =0.0f;
        }

    for (i = 0; i < nd; ++i)
        dim_glob[i] = 2*pdim[i];

    for (i = 0; i < nd-1; ++i)
        if (!s) printf("%d x ",dim_glob[i]);
    if (!s) printf("%d matrix\n", dim_glob[nd-1]);
   
    dfft_plan plan_2;
    dfft_create_plan(&plan_2, nd, dim_glob, NULL, NULL, pdim, 0, 0, MPI_COMM_WORLD);

    /* forward transfom */
    dfft_execute(in_2, in_2, 0, plan_2);

    /* inverse transform */
    dfft_execute(in_2, in_2, 1, plan_2);

    for (i = 0; i < size_2n; ++i)
        {
        if (s == 0 && i == 0)
            {
            CHECK_SMALL(RE(in_2[i]), abs_tol);
            }
        else
            {
            CHECK_CLOSE(RE(in_2[i]), size_2n*p*(i+(double)size_2n*(double)s),tol);
            CHECK_SMALL(IM(in_2[i]), abs_tol);
            }
        }

    free(in_2);
    dfft_destroy_plan(plan_2);

    // test with a padded 4^n local matrix (two ghost cells per axis)
    cpx_t *in_3;
    cpx_t *out_3;

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
 
    in_3 = (cpx_t *) malloc(size_4n_embed*sizeof(cpx_t));
    out_3 = (cpx_t *) malloc(size_4n*sizeof(cpx_t));

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
            RE(in_3[i]) = 99.0;
            IM(in_3[i]) = 123.0;
            }
        else
            {
            RE(in_3[i]) = (double) val;
            IM(in_3[i]) = 0.0;               
            }
        }

    dfft_plan plan_3_fw,plan_3_bw;
    dfft_create_plan(&plan_3_fw, nd, dim_glob, inembed, NULL, pdim, 0, 0, MPI_COMM_WORLD);
    dfft_create_plan(&plan_3_bw, nd, dim_glob, NULL, inembed, pdim, 0, 0, MPI_COMM_WORLD);
   
    int offset = 0;
    int n_ghost = 2;
    for (i = 0; i < nd; i++)
        {
        offset *= 4;
        offset += n_ghost/2;
        }

    /* forward transform */
    dfft_execute(in_3+offset, out_3, 0, plan_3_fw);
        
    /* inverse transform */
    dfft_execute(out_3,in_3+offset, 1, plan_3_bw);

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
            //CHECK_CLOSE(RE(in_3[i]),99.0,tol);
            //CHECK_CLOSE(IM(in_3[i]), 123.0,tol);
            }
        else
            {
            CHECK_CLOSE(RE(in_3[i]), (double) val*(double)p*(double)size_4n,tol);
            CHECK_SMALL(IM(in_3[i]), abs_tol);
            }
        }

    free(in_3);
    free(out_3);

    dfft_destroy_plan(plan_3_fw);
    dfft_destroy_plan(plan_3_bw);

    free(pidx);
    free(pdim);
    }

#if 0
//! Compares a distributed FFT against a single-processor FFT
template< class DFFT, class cpx_type >
void test_distributed_fft_compare(shared_ptr<ExecutionConfiguration> exec_conf)
    {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // initialize the domain decomposition
    boost::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf, make_scalar3(1.0,1.0,1.0)));

    // in FFT w/single prec, round-off errors can sum up quickly
    Scalar tol_rough = 0.15;

    // Do a size.x*4 x size.y * 6 x size.z * 8 FFT (total dimensions)
    Index3D domain_idx = decomposition->getDomainIndexer();
    int dims[3];
    unsigned int local_nx = 4;
    unsigned int local_ny = 6;
    unsigned int local_nz = 8;
    unsigned int nx = domain_idx.getW()*local_nx;
    unsigned int ny = domain_idx.getH()*local_ny;
    unsigned int nz = domain_idx.getD()*local_nz;
    dims[0] = nz; dims[1] = ny; dims[2] = nx;

    kiss_fft_cpx *in_kiss = new kiss_fft_cpx[nx*ny*nz];

    Saru saru(12345);

    // fill table with complex random numbers in row major order
    for (unsigned int x = 0; x < nx; ++x)
        for (unsigned int y = 0; y < ny; ++y)
            for (unsigned int z = 0; z < nz; ++z)
                {
                in_kiss[x+nx*(y+ny*z)].r =  saru.f(-100.0,100.0);
                in_kiss[x+nx*(y+ny*z)].i =  saru.f(-100.0,100.0);
                }

    kiss_fft_cpx *out_kiss = new kiss_fft_cpx[nx*ny*nz];

    // construct forward transform
    kiss_fftnd_cfg cfg = kiss_fftnd_alloc(dims,3,0,NULL,NULL);

    // carry out conventional FFT
    kiss_fftnd(cfg, in_kiss, out_kiss);

    // compare to distributed FFT
    GPUArray<cpx_type> in(local_nx*local_ny*local_nz,exec_conf);
    uint3 grid_pos = domain_idx.getTriple(exec_conf->getRank());

        {
        // fill in array identically to reference FFT
        ArrayHandle<cpx_type> h_in(in, access_location::host, access_mode::overwrite);
        for (unsigned int x = 0; x < nx; ++x)
            for (unsigned int y = 0; y < ny; ++y)
                for (unsigned int z = 0; z < nz; ++z)
                    {
                    if (x>= grid_pos.x*local_nx && x < (grid_pos.x+1)*local_nx && 
                        y>= grid_pos.y*local_ny && y < (grid_pos.y+1)*local_ny &&
                        z>= grid_pos.z*local_nz && z < (grid_pos.z+1)*local_nz)
                        {
                        unsigned int x_local = x - grid_pos.x*local_nx;
                        unsigned int y_local = y - grid_pos.y*local_ny;
                        unsigned int z_local = z - grid_pos.z*local_nz;
                        
                        h_in.data[z_local+local_nz*(y_local+local_ny*x_local)] =
                            make_complex<cpx_type>(in_kiss[x+nx*(y+ny*z)].r,
                                                   in_kiss[x+nx*(y+ny*z)].i);
                        }
                    }
        }

    GPUArray<cpx_type> out(local_nx*local_ny*local_nz,exec_conf);

    DFFT dfft(exec_conf,decomposition, make_uint3(local_nx,local_ny,local_nz), make_uint3(0,0,0));

    // forward transform
    dfft.FFT3D(in, out, false);

    DFFTIndex index = dfft.getIndexer();

    // do comparison
        {
        ArrayHandle<cpx_type> h_out(out, access_location::host, access_mode::read);

        unsigned int n_wave_local = local_nx * local_ny * local_nz;
        for (i = 0; i < n_wave_local; ++i)
            {
            uint3 kidx = index(i);
            BOOST_CHECK(kidx.x < nx);
            BOOST_CHECK(kidx.y < ny);
            BOOST_CHECK(kidx.z < nz);

            Scalar re = getRe(h_out.data[i]);
            Scalar im = getIm(h_out.data[i]);
            Scalar re_kiss = out_kiss[kidx.x+nx*(kidx.y+ny*kidx.z)].r;
            Scalar im_kiss = out_kiss[kidx.x+nx*(kidx.y+ny*kidx.z)].i;  

            if (fabs(re_kiss) < tol_rough)
                BOOST_CHECK_SMALL(re,tol_rough);
            else
                BOOST_CHECK_CLOSE(re,re_kiss, tol_rough);

            if (fabs(im_kiss) < tol_rough)
                BOOST_CHECK_SMALL(im,tol_rough);
            else
                BOOST_CHECK_CLOSE(im, im_kiss, tol_rough);
            }
        }
    delete[] in_kiss;
    delete[] out_kiss;
    }
#endif

