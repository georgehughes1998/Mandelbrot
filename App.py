import numpy as np
import multiprocessing as mp
import math, random, sys, itertools, time

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


if __name__ == '__main__':
    from pygame.locals import *
    import pygame.freetype

    BLACK = pygame.Color(0,0,0)
    WHITE = pygame.Color(255,255,255)
    GREY = pygame.Color(145,145,145)
    RED = pygame.Color(255,0,0)
    GREEN = pygame.Color(0,128,0)
    BLUE = pygame.Color(0,0,128)

    def draw_text(surface, text, pos, color):
        text_surface, text_rect = font.render(text, fgcolor=color)
        text_rect.topleft = pos
        surface.blit(text_surface, text_rect)


    pygame.init()
    surface = pygame.display.set_mode(SCREEN_RESOLUTION)
    fps_clock = pygame.time.Clock()
    font = pygame.freetype.SysFont(pygame.freetype.get_default_font(), FONT_SIZE)

    calc_parent_conn, calc_child_conn = mp.Pipe()
    p = mp.Process(target=calculate_mandelbrot, args=(calc_child_conn,))
    p.start()

    # Ensure mandelbrot file is created
    calc_parent_conn.recv()

    surf_array = np.memmap(SURFARRAY_FILENAME, dtype=SURFARRAY_DTYPE, mode='r+', shape=SURFARRAY_SHAPE)

    running = True
    saved = False
    x = 0

    total_time = 0
    recv_time = False
    start_time = time.time()

    while running:
        # Background
        surface.fill(GREY)

        surf_array_surface = pygame.surfarray.make_surface(surf_array)
        surf_array_surface_trans = pygame.transform.scale(surf_array_surface, SCREEN_RESOLUTION)
        surface.blit(surf_array_surface_trans, (0, 0))

        draw_text(surface, f"{total_time}s", (16, 16), GREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: # The user closed the window!
                if not saved and AUTOSAVE:
                    t = time.asctime().replace(':', '')
                    pygame.image.save(surf_array_surface, f"capture/mandelbrot_image {t}.png")
                running = False # Stop running
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    t = time.asctime().replace(':', '')
                    pygame.image.save(surf_array_surface, f"capture/mandelbrot_image {t}.png")
                    saved = True
        pygame.display.update()
        fps_clock.tick(30)

        if not recv_time:
            total_time = round(time.time() - start_time, TIME_PRECISION)
            if calc_parent_conn.poll():
                total_time = calc_parent_conn.recv()
                recv_time = True


    # Merge subprocess
    p.join(timeout=1)
    p.terminate()

    # surf_p.join(timeout=0.5)
    # surf_p.terminate()

    pygame.quit() # Close the window
