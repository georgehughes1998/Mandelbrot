#include <pyopencl-complex.h>

    kernel void znplus1(
    global float *out_buf,
            float xmax,
            float xmin,
            float ymax,
            float ymin,
            int width,
            int height,
            float xoffset,
            float yoffset,
            float xscale,
            float yscale,
            int depth,
            float z_power,
            float cutoff)
    {{
        int gid = get_global_id(0);

        int x_int = (gid + 1)/height;
        int y_int = (gid + 1)%height;

        float x_unscaled = (xmax - xmin)*(x_int/(float)width) + xmin;
        float y_unscaled = (ymax - ymin)*(y_int/(float)height) + ymin;

        float x = xscale * x_unscaled + xoffset;
        float y = yscale * y_unscaled + yoffset;

        cfloat_t c;
        c.real = x;
        c.imag = y;

        cfloat_t z;
        z.real = 0;
        z.imag = 0;

        int count;
        for (count = depth; count > 0; count--)
        {{
            z = cfloat_add(cfloat_powr(z, z_power), c);
            if (z.real > cutoff)
            {{
                break;
            }}
        }}

        out_buf[gid] = count;
    }}