/* A simple radix-2 FFT */
typedef struct { double x,y;} cpxdbl;

typedef struct
    {
    int n;
    int istride;
    int ostride;
    int idist;
    int odist;
    int howmany;
    } bare_fft_plan;

void radix2_fft(cpxdbl *in, cpxdbl *out, const int n, const int isign, bare_fft_plan plan);
