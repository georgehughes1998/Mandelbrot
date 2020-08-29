#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define PYOPENCL_DEFINE_CDOUBLE

#include <pyopencl-complex.h>

    kernel void znplus1(
    global long *out_buf,
            double xmax,
            double xmin,
            double ymax,
            double ymin,
            int width,
            int height,
            double xoffset,
            double yoffset,
            double xscale,
            double yscale,
            int depth,
            double z_power,
            double cutoff)
    {{
        int gid = get_global_id(0);

        int x_int = (gid + 1)/height;
        int y_int = (gid + 1)%height;

        double x_unscaled = (xmax - xmin)*(x_int/(double)width) + xmin;
        double y_unscaled = (ymax - ymin)*(y_int/(double)height) + ymin;

        double x = xscale * x_unscaled + xoffset;
        double y = yscale * y_unscaled + yoffset;

        cdouble_t c;
        c.real = x;
        c.imag = y;

        cdouble_t z;
        z.real = 0;
        z.imag = 0;

        int count = 0;
        for (count = depth; count > 0; count--)
        {{
            z = cdouble_add(cdouble_powr(z, z_power), c);
            if (z.real > cutoff)
            {{
                break;
            }}
        }}

        out_buf[gid] = count;
    }}