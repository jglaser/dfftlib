/* A simple radix-2 FFT */
typedef struct { double x,y;} double2;

typedef struct
    {
    int n;
    int istride;
    int ostride;
    int idist;
    int odist;
    int howmany;
    } bare_fft_plan;

void radix2_fft(double2 *in, double2 *out, const int n, const int isign, bare_fft_plan plan);
