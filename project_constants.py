import numpy as np
import itertools

SCREEN_WIDTH, SCREEN_HEIGHT = SCREEN_RESOLUTION = (1024, 480)

# MP Constants
CHUNK_SIZE = 192
NUM_PROC = 4

# Mandelbrot Constants
PRECISION = 0.001
PRECISION_ROUND = min([(n if round(PRECISION, n) == PRECISION else 10) for n in range(8)])

XOFFSET = 0
YOFFSET = 0
SCALE = 1

DEPTH = 100

MINX, MAXX = -2.1, 1
MINY, MAXY = -1, 1

XRANGE = np.arange(MINX, MAXX, PRECISION)
YRANGE = np.arange(MINY, MAXY, PRECISION)
XRANGE_INDEX = range(len(XRANGE))
YRANGE_INDEX = range(len(YRANGE))
XY_INDEX_RANGE = list(itertools.product(XRANGE_INDEX, YRANGE_INDEX))
SHAPE = len(XRANGE), len(YRANGE)

Z_POWER = 2

# Surface Array File Constants
SURFARRAY_FILENAME = "surf_array.dat"
SURFARRAY_DTYPE = 'int'
SURFARRAY_SHAPE = SHAPE + (3,)

# Text Display Constants
TIME_PRECISION = 2
TIME_FONT_NAME = 'Courier New'
TIME_FONT_SIZE = 16
TIME_BOLD = True

# Misc
AUTOSAVE = False
