import numpy as np

from pygame.locals import *
import pygame.freetype

from CL.mandelbrot_func import create_and_build_program, create_out_array_and_buffer, create_cl_context_and_queue, calculate_mandelbrot_opencl


# Mandelbrot dimensions
SHAPE = WIDTH, HEIGHT = 1024, 1024
XMIN, XMAX = -2, 2
YMIN, YMAX = -2, 2

# Mandelbrot scalars
XOFFSET, YOFFSET = 0, 0
XSCALE = YSCALE = 1
DEPTH = 750
Z_POWER = 2
CUTOFF = 2

# Interaction & display constants
ZOOM_FACTOR = 1.3
PAN_STEP_DIVIDER = 5
USE_MOUSE = False
DO_FULLSCREEN = False

# Pygame colours definition
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
GREY = pygame.Color(145, 145, 145)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 128, 0)
BLUE = pygame.Color(0, 0, 128)

context, queue, device = create_cl_context_and_queue(use_gpu=True)
out_np, out_buf = create_out_array_and_buffer(context, SHAPE)
program = create_and_build_program(context, "CL/kernel.c")

device_name = device.name


def draw_text(surface, text, font, pos, color):
    text_surface, text_rect = font.render(text, fgcolor=color)
    text_rect.topleft = pos
    surface.blit(text_surface, text_rect)


pygame.init()

screen_width, screen_height = screen_res = (1024, 960)
if DO_FULLSCREEN:
    surface = pygame.display.set_mode(screen_res, FULLSCREEN)
else:
    surface = pygame.display.set_mode(screen_res)

fps_clock = pygame.time.Clock()
font = pygame.freetype.SysFont(pygame.freetype.get_default_font(), 16)

joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
if joysticks:
    joystick = joysticks[0]
    print(f"Found joystick to use: {joystick.get_name()}")
    joystick.init()

step = 0

running = True
while running:
    # Background
    surface.fill(GREY)
    out_surface = pygame.surfarray.make_surface(out_np)
    out_surface_scale = pygame.transform.scale(out_surface, screen_res)
    surface.blit(out_surface_scale, (0, 0))

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

    calculate_mandelbrot_opencl(queue, program, out_np, out_buf, scalar_args)

    step = (step + 1)

    locktext = ["[Mouse Locked]", ""][USE_MOUSE]
    pos_text = f"x {round(XOFFSET, 5)}, y {round(YOFFSET, 5)}, zoom {round(1 / XSCALE)}"
    draw_text(surface, f"{locktext} Rendering on: {device_name} | Frame {step} | {pos_text}", font, (16, 16), GREEN)

    for event in pygame.event.get():
        #         if not event.type == MOUSEMOTION: print(event)
        if event.type == pygame.QUIT:  # The user closed the window!
            running = False  # Stop running
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            if event.key == K_r:
                step = 0
            if event.key == K_SPACE:
                pygame.image.save(out_surface, f"capture/Mandelbrot {pos_text}.png")

            if event.key == K_UP:
                YOFFSET -= YSCALE / PAN_STEP_DIVIDER
            if event.key == K_DOWN:
                YOFFSET += YSCALE / PAN_STEP_DIVIDER
            if event.key == K_LEFT:
                XOFFSET -= XSCALE / PAN_STEP_DIVIDER
            if event.key == K_RIGHT:
                XOFFSET += XSCALE / PAN_STEP_DIVIDER

        elif event.type == MOUSEMOTION:
            if USE_MOUSE:
                mousex, mousey = pygame.mouse.get_pos()
                XOFFSET = (XMAX - XMIN) * (mousex / screen_width) + XMIN;
                YOFFSET = (YMAX - YMIN) * (mousey / screen_height) + YMIN;
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 4:
                XSCALE /= ZOOM_FACTOR
                YSCALE /= ZOOM_FACTOR
            if event.button == 5:
                XSCALE *= ZOOM_FACTOR
                YSCALE *= ZOOM_FACTOR
            if event.button == 2:
                USE_MOUSE = not USE_MOUSE

        elif event.type == JOYBUTTONDOWN:
            if event.button == 0:
                pygame.image.save(out_surface, f"capture/Mandelbrot {pos_text}.png")
    #             print(event)

    if joysticks:
        zoom_amount = round(joystick.get_axis(2), 4)
        XSCALE *= 1 + (zoom_amount) / 8
        YSCALE *= 1 + (zoom_amount) / 8

        xmove_amount = round(joystick.get_axis(0), 4)
        if xmove_amount > 0.2 or xmove_amount < -0.2:
            XOFFSET += xmove_amount * XSCALE / PAN_STEP_DIVIDER

        ymove_amount = round(joystick.get_axis(1), 4)
        if ymove_amount > 0.2 or ymove_amount < -0.2:
            YOFFSET += ymove_amount * YSCALE / PAN_STEP_DIVIDER

    pygame.display.update()
    fps_clock.tick(60)

pygame.quit()  # Close the window