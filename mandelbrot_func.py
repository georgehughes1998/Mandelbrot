import multiprocessing as mp
import time

from project_constants import *

# Iterative Function
def znplus1(c, z=0, i=DEPTH):
    if i <= 0 or z > 2:
        return z
    else:
        return znplus1(c, z=(z ** Z_POWER) + (c/SCALE) + XOFFSET + (YOFFSET*1j), i=i - 1)


# Set surfarray file value for x,y pos
def set_mandelbrot(x, y):
    surf_array = np.memmap(SURFARRAY_FILENAME, dtype=SURFARRAY_DTYPE, mode='r+', shape=SURFARRAY_SHAPE)

    num = round(XRANGE[x], PRECISION_ROUND) + (round(YRANGE[y], PRECISION_ROUND) * 1j)

    if znplus1(num) > 2:
        surf_array[x, y] = 0
    else:
        surf_array[x, y] = 255

    return 0


# Calculate members of the mandelbrot set and write to file
def calculate_mandelbrot(pipe):
    surf_array = np.memmap(SURFARRAY_FILENAME, dtype=SURFARRAY_DTYPE, mode='w+', shape=SURFARRAY_SHAPE)

    # Inform file is created
    pipe.send(1)

    start_time = time.time()
    with mp.Pool(processes=NUM_PROC) as pool:
        pool.starmap(set_mandelbrot, XY_INDEX_RANGE, chunksize=CHUNK_SIZE)

    end_time = time.time()
    total_time = round(end_time - start_time, TIME_PRECISION)

    pipe.send(total_time)

    return 0