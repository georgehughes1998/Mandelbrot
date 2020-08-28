import numpy as np

from pygame.locals import *
import pygame.freetype

from CL.mandelbrot_func import create_and_build_program, create_out_array_and_buffer, create_cl_context_and_queue, calculate_mandelbrot_opencl

# Mandelbrot dimensions
SHAPE = WIDTH, HEIGHT = 1024, 1024
CAPTURE_SHAPE = CAPTURE_WIDTH, CAPTURE_HEIGHT = 1024*4, 1024*4
XMIN, XMAX = -2, 2
YMIN, YMAX = -2, 2

# Mandelbrot scalars
XOFFSET, YOFFSET = 0, 0
XSCALE = YSCALE = 1
DEPTH = 250
Z_POWER = 2
CUTOFF = 2

# Interaction & display constants
ZOOM_FACTOR = 1.3
PAN_STEP_DIVIDER = 5
USE_MOUSE = False
DO_FULLSCREEN = True

# Pygame colours definition
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
GREY = pygame.Color(145, 145, 145)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 196, 0)
BLUE = pygame.Color(0, 0, 128)

context, queue, device = create_cl_context_and_queue(use_gpu=True)
out_np, out_buf = create_out_array_and_buffer(context, SHAPE)
program = create_and_build_program(context, "CL/kernel.c")

device_name = device.name


def draw_text(surface, text, font, pos, color):
    text_surface, text_rect = font.render(text, fgcolor=color)
    text_rect.topleft = pos
    surface.blit(text_surface, text_rect)

    return(text_rect)


pygame.init()

# pygame.display.Info()["current_h"]
# pygame.display.Info()["current_w"]
display_info = pygame.display.Info()

if DO_FULLSCREEN:
    screen_width, screen_height = screen_res = (display_info.current_w, display_info.current_h)
    surface = pygame.display.set_mode((screen_res), FULLSCREEN)
else:
    screen_width, screen_height = screen_res = (display_info.current_w//2, display_info.current_h//2)
    surface = pygame.display.set_mode(screen_res, RESIZABLE)

pygame.display.set_caption("Mandelbrot Explorer")
icon = pygame.image.load("assets/favicon.ico")
pygame.display.set_icon(icon)
cursor = pygame.cursors.load_xbm("assets/crosshair.xbm", mask="assets/crosshair_mask.xbm")
pygame.mouse.set_cursor(*cursor)
fps_clock = pygame.time.Clock()
font = pygame.freetype.SysFont(pygame.freetype.get_default_font(), 16)

joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
if joysticks:
    joystick = joysticks[0]
    print(f"Found joystick to use: {joystick.get_name()}")
    joystick.init()

step = 0
do_save = False
text_rect = (16,16,128,32)

running = True
while running:

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

    # Save a high res version
    if do_save:
        capture_out_np, capture_out_buf = create_out_array_and_buffer(context, CAPTURE_SHAPE)
        scalar_args = (np.float32(XMAX),
                       np.float32(XMIN),
                       np.float32(YMAX),
                       np.float32(YMIN),
                       np.int32(CAPTURE_WIDTH),
                       np.int32(CAPTURE_HEIGHT),
                       np.float32(XOFFSET),
                       np.float32(YOFFSET),
                       np.float32(XSCALE),
                       np.float32(YSCALE),
                       np.int32(DEPTH),
                       np.float32(Z_POWER),
                       np.float32(CUTOFF))
        calculate_mandelbrot_opencl(queue, program, capture_out_np, capture_out_buf, scalar_args)
        capture_surface = pygame.surfarray.make_surface(capture_out_np)
        pygame.image.save(capture_surface, f"capture/Mandelbrot {pos_text}.png")
        do_save = False

    # Pygame events loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == VIDEORESIZE:
            SHAPE = WIDTH, HEIGHT = event.size
            print(f"Resizing to {SHAPE}")
            screen_res = SHAPE
            out_np, out_buf = create_out_array_and_buffer(context, SHAPE)

            if DO_FULLSCREEN:
                surface = pygame.display.set_mode(screen_res, FULLSCREEN)
            else:
                surface = pygame.display.set_mode(screen_res, RESIZABLE)

        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_r:
                step = 0
            elif event.key == K_SPACE:
                do_save = True

            if event.key == K_UP:
                YOFFSET -= YSCALE / PAN_STEP_DIVIDER
            elif event.key == K_DOWN:
                YOFFSET += YSCALE / PAN_STEP_DIVIDER
            elif event.key == K_LEFT:
                XOFFSET -= XSCALE / PAN_STEP_DIVIDER
            elif event.key == K_RIGHT:
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
            elif event.button == 5:
                XSCALE *= ZOOM_FACTOR
                YSCALE *= ZOOM_FACTOR
            elif event.button == 2:
                USE_MOUSE = not USE_MOUSE

        elif event.type == JOYBUTTONDOWN:
            if event.button == 0:
                do_save = True
            elif event.button == 8:
                XSCALE, YSCALE = 1, 1
            elif event.button == 9:
                Z_POWER = 2
            elif event.button == 6:
                running = False
            else:
                print(f"Unmapped button {event.button}")

        elif event.type == JOYHATMOTION:
            x_pan, y_pan = event.value
            XOFFSET += x_pan * XSCALE / PAN_STEP_DIVIDER
            YOFFSET -= y_pan * YSCALE / PAN_STEP_DIVIDER

    # Receive any movement from the joystick
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

        # zchange_amount = round(joystick.get_axis(3), 4)
        # if zchange_amount > 0.2 or zchange_amount < -0.2:
        #     Z_POWER += zchange_amount/100

        zchange_amount = round(joystick.get_axis(4), 4)
        if zchange_amount > 0.2 or zchange_amount < -0.2:
            DEPTH += round(zchange_amount)

    # Drawing to screen
    surface.fill(GREY)
    out_surface = pygame.surfarray.make_surface(out_np)
    # out_surface_scale = pygame.transform.scale(out_surface, screen_res)
    surface.blit(out_surface, (0, 0))

    # locktext = ["[Mouse Locked]", ""][USE_MOUSE]
    save_text = ["", "[Saving...]"][do_save]
    pos_text = f"x {round(XOFFSET, 5)}, y {round(YOFFSET, 5)}, zoom {round(1 / XSCALE)}"
    pygame.draw.rect(surface, BLACK, text_rect)
    text_rect = draw_text(surface, f"{save_text} Rendering on: {device_name} | Frame {step} | {pos_text} | Z^{round(Z_POWER, 2)} | Depth {DEPTH}", font, pygame.mouse.get_pos(), GREEN)

    pygame.display.update()
    fps_clock.tick(60)

pygame.quit()