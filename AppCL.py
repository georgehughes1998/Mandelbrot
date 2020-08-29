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
FONT_SIZE = 12
DO_FULLSCREEN = True
FLOAT_CUTOFF = 0.000001

SCALAR_ARG_TYPES_GPU = [None,
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

SCALAR_ARG_TYPES_CPU = [None,
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

# Pygame colours definition
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
GREY = pygame.Color(145, 145, 145)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 196, 0)
BLUE = pygame.Color(0, 0, 128)

context_gpu, queue_gpu, device_gpu = create_cl_context_and_queue(use_gpu=True)
# out_np, out_buf_gpu = create_out_array_and_buffer(context_gpu, SHAPE)
program_gpu = create_and_build_program(context_gpu, "CL/kernel.c")

context_cpu, queue_cpu, device_cpu = create_cl_context_and_queue(use_gpu=False)
# _, out_buf_cpu = create_out_array_and_buffer(context_cpu, SHAPE, )
program_cpu = create_and_build_program(context_cpu, "CL/kernel_double.c")

context = None
queue = None
program = None
out_np = None
out_buf = None
device = None
scalar_arg_types = None
device_using = None

def set_device(mode):
    global context, queue, program, out_buf, out_np, device, scalar_arg_types, device_using

    if mode == 0:
        context = context_gpu
        queue = queue_gpu
        program = program_gpu
        out_np, out_buf = create_out_array_and_buffer(context_gpu, SHAPE, dtype=np.int64)
        device = device_gpu
        scalar_arg_types = SCALAR_ARG_TYPES_GPU
        device_using = 0
    else:
        context = context_cpu
        queue = queue_cpu
        program = program_cpu
        out_np, out_buf = create_out_array_and_buffer(context_cpu, SHAPE, dtype=np.int64)
        device = device_cpu
        scalar_arg_types = SCALAR_ARG_TYPES_CPU
        device_using = 1

def get_scalar_args(do_capture=False):
    global device_using
    floattype = [np.float32, np.double][device_using]

    scalar_args = (floattype(XMAX),
                   floattype(XMIN),
                   floattype(YMAX),
                   floattype(YMIN),
                   np.int32([WIDTH, CAPTURE_WIDTH][do_capture]),
                   np.int32([HEIGHT, CAPTURE_HEIGHT][do_capture]),
                   floattype(XOFFSET),
                   floattype(YOFFSET),
                   floattype(XSCALE),
                   floattype(YSCALE),
                   np.int32(DEPTH),
                   floattype(Z_POWER),
                   floattype(CUTOFF))
    return scalar_args


def draw_text(surface, text, font, pos, color):
    text_surface, text_rect = font.render(text, fgcolor=color)
    text_rect.topleft = pos
    surface.blit(text_surface, text_rect)

    return(text_rect)


set_device(0)

pygame.init()

# pygame.display.Info()["current_h"]
# pygame.display.Info()["current_w"]
display_info = pygame.display.Info()
max_w, max_h = (display_info.current_w, display_info.current_h)

if DO_FULLSCREEN:
    screen_width, screen_height = screen_res = (max_w, max_h)
    surface = pygame.display.set_mode((screen_res), FULLSCREEN)
else:
    screen_width, screen_height = screen_res = (max_w//2, max_h//2)
    surface = pygame.display.set_mode(screen_res, RESIZABLE)

pygame.display.set_caption("Mandelbrot Explorer")
icon = pygame.image.load("assets/favicon.ico")
pygame.display.set_icon(icon)
cursor = pygame.cursors.load_xbm("assets/crosshair.xbm", mask="assets/crosshair_mask.xbm")
pygame.mouse.set_cursor(*cursor)
fps_clock = pygame.time.Clock()
font = pygame.freetype.SysFont(pygame.freetype.get_default_font(), FONT_SIZE)

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

    scalar_args = get_scalar_args()
    calculate_mandelbrot_opencl(queue, program, out_np, out_buf, scalar_args, scalar_arg_types)

    step = (step + 1)

    # Save a high res version
    if do_save:
        capture_out_np, capture_out_buf = create_out_array_and_buffer(context, CAPTURE_SHAPE, dtype=np.int64)
        scalar_args = get_scalar_args(do_capture=True)
        calculate_mandelbrot_opencl(queue, program, capture_out_np, capture_out_buf, scalar_args, scalar_arg_types)
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
            out_np, out_buf = create_out_array_and_buffer(context, SHAPE, dtype=np.int64)

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
                set_device(0)
            elif event.button == 9:
                set_device(not device_using)
            elif event.button == 6:
                running = False
            elif event.button == 7:
                DO_FULLSCREEN = not DO_FULLSCREEN

                if DO_FULLSCREEN:
                    surface = pygame.display.set_mode((max_w, max_h), FULLSCREEN)
                else:
                    surface = pygame.display.set_mode((max_w//4, max_h//4), RESIZABLE)
            else:
                print(f"Unmapped button {event.button}")

        elif event.type == JOYHATMOTION:
            x_pan, y_pan = event.value
            XOFFSET += x_pan * XSCALE / PAN_STEP_DIVIDER
            YOFFSET -= y_pan * YSCALE / PAN_STEP_DIVIDER

    # Receive any movement from the joystick
    if joysticks:
        zoom_amount = round(joystick.get_axis(2), 16)
        old_scale = XSCALE

        XSCALE *= 1 + (zoom_amount) / 8
        YSCALE *= 1 + (zoom_amount) / 8

        if XSCALE < FLOAT_CUTOFF and old_scale > FLOAT_CUTOFF:
            set_device(1)
        elif XSCALE > FLOAT_CUTOFF and old_scale < FLOAT_CUTOFF:
            set_device(0)

        xmove_amount = round(joystick.get_axis(0), 16)
        if xmove_amount > 0.2 or xmove_amount < -0.2:
            XOFFSET += xmove_amount * XSCALE / PAN_STEP_DIVIDER

        ymove_amount = round(joystick.get_axis(1), 16)
        if ymove_amount > 0.2 or ymove_amount < -0.2:
            YOFFSET += ymove_amount * YSCALE / PAN_STEP_DIVIDER

        # zchange_amount = round(joystick.get_axis(3), 4)
        # if zchange_amount > 0.2 or zchange_amount < -0.2:
        #     Z_POWER += zchange_amount/100

        depth_change = round(joystick.get_axis(4), 4)
        if depth_change > 0.2 or depth_change < -0.2:
            DEPTH += round(depth_change)

    # Drawing to screen
    surface.fill(GREY)
    out_surface = pygame.surfarray.make_surface(out_np)
    # out_surface_scale = pygame.transform.scale(out_surface, screen_res)
    surface.blit(out_surface, (0, 0))

    # locktext = ["[Mouse Locked]", ""][USE_MOUSE]
    save_text = ["", "[Saving...]"][do_save]
    pos_text = f"x {round(XOFFSET, 5)}, y {round(YOFFSET, 5)}, zoom {round(1 / XSCALE)}"
    pygame.draw.rect(surface, BLACK, text_rect)
    text_rect = draw_text(surface, f"{save_text} Rendering on: {device.name} | Frame {step} | {pos_text} | Z^{round(Z_POWER, 2)} | Depth {DEPTH}", font, pygame.mouse.get_pos(), GREEN)

    pygame.display.update()
    fps_clock.tick(60)

pygame.quit()