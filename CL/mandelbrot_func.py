import time

import numpy as np
import pyopencl as cl

import matplotlib.pyplot as plt


_WRITE_ONLY = mf = cl.mem_flags.WRITE_ONLY


def create_cl_context_and_queue(use_gpu=True):
    platforms = cl.get_platforms()
    if platforms:
        if use_gpu:
            device_type = cl.device_type.GPU
        else:
            device_type = cl.device_type.CPU

        devices = platforms[0].get_devices(device_type=device_type)
        device = devices[0]

    print(f"Using device: {device.name}")
    context = cl.Context(devices=[device])
    # context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    return context, queue, device


def create_and_build_program(context, kernel_filename):
    with open(kernel_filename) as f:
        kernel = f.read()

    program = cl.Program(context, kernel).build(options=[])
    return program


def create_out_array_and_buffer(context, array_shape, dtype=np.float32):
    out_np = np.zeros(shape=array_shape, dtype=dtype)
    out_buf = cl.Buffer(context, _WRITE_ONLY, out_np.nbytes)

    return out_np, out_buf


def calculate_mandelbrot_opencl(queue, program, out_np, out_buf, scalar_args, scalar_arg_types):
    kernel = program.znplus1
    kernel.set_scalar_arg_dtypes(scalar_arg_types)

    kernel(queue, out_np.flatten().shape, None, out_buf, *scalar_args)

    cl.enqueue_copy(queue, out_np, out_buf)

    return 0


def _run_test():
    SHAPE = WIDTH, HEIGHT = 1323, 761
    XMIN, XMAX = -2, 2
    YMIN, YMAX = -2, 2

    XOFFSET, YOFFSET = 0, 0
    XSCALE = YSCALE = 1
    DEPTH = 750
    Z_POWER = 2
    CUTOFF = 2

    SCALAR_ARG_TYPES = [None,
                            np.float32,
                            np.float32,
                            np.float32,
                            np.float32,
                            np.int32,
                            np.int32,
                            np.float32,
                            np.float32,
                            np.float32,
                            np.float32,
                            np.int32,
                            np.float32,
                            np.float32]

    context, queue, _ = create_cl_context_and_queue(use_gpu=True)
    out_np, out_buf = create_out_array_and_buffer(context, SHAPE)
    program = create_and_build_program(context, "kernel.c")

    scalar_args = (np.float32(XMAX),
                   np.float32(XMIN),
                   np.float32(YMAX),
                   np.float32(YMIN),
                   np.int32(WIDTH),
                   np.int32(HEIGHT),
                   np.float32(XOFFSET),
                   np.float32(YOFFSET),
                   np.float32(XSCALE),
                   np.float32(YSCALE),
                   np.int32(DEPTH),
                   np.float32(Z_POWER),
                   np.float32(CUTOFF))

    start_time = time.time()
    calculate_mandelbrot_opencl(queue, program, out_np, out_buf, scalar_args, SCALAR_ARG_TYPES)
    end_time = time.time()
    print(f"Completed in {end_time - start_time}")

    fig = plt.figure(figsize=(25.6, 19.2))
    extent = (XMIN * XSCALE + XOFFSET, XMAX * XSCALE + XOFFSET, YMIN * YSCALE + YOFFSET, YMAX * YSCALE + YOFFSET)
    plt.imshow(out_np, extent=extent, cmap='Reds')
    plt.grid(True, which='both')
    plt.show()


def _run_test_cpu():
    SHAPE = WIDTH, HEIGHT = 1024, 1024
    XMIN, XMAX = -2, 2
    YMIN, YMAX = -2, 2

    XOFFSET, YOFFSET = 0, 0
    XSCALE = YSCALE = 1
    DEPTH = 750
    Z_POWER = 2
    CUTOFF = 2

    SCALAR_ARG_TYPES = [None,
         np.double,
         np.double,
         np.double,
         np.double,
         np.int32,
         np.int32,
         np.double,
         np.double,
         np.double,
         np.double,
         np.int32,
         np.double,
         np.double]

    context, queue, _ = create_cl_context_and_queue(use_gpu=False)
    out_np, out_buf = create_out_array_and_buffer(context, SHAPE, dtype=np.int64)
    program = create_and_build_program(context, "kernel_double.c")

    scalar_args = (np.double(XMAX),
                   np.double(XMIN),
                   np.double(YMAX),
                   np.double(YMIN),
                   np.int32(WIDTH),
                   np.int32(HEIGHT),
                   np.double(XOFFSET),
                   np.double(YOFFSET),
                   np.double(XSCALE),
                   np.double(YSCALE),
                   np.int32(DEPTH),
                   np.double(Z_POWER),
                   np.double(CUTOFF))

    start_time = time.time()
    calculate_mandelbrot_opencl(queue, program, out_np, out_buf, scalar_args, SCALAR_ARG_TYPES)
    end_time = time.time()
    print(f"Completed in {end_time - start_time}")

    fig = plt.figure(figsize=(25.6, 19.2))
    extent = (XMIN * XSCALE + XOFFSET, XMAX * XSCALE + XOFFSET, YMIN * YSCALE + YOFFSET, YMAX * YSCALE + YOFFSET)
    plt.imshow(out_np, extent=extent, cmap='Reds')
    plt.grid(True, which='both')
    plt.show()


# _run_test_cpu()
