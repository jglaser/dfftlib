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
