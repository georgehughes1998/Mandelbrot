from mandelbrot_func import *


if __name__ == '__main__':
    from pygame.locals import *
    import pygame.freetype

    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    GREY = pygame.Color(145, 145, 145)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 128, 0)
    BLUE = pygame.Color(0, 0, 128)

    # Function to draw text
    def draw_text(surface, text_string, text_font, text_pos, text_colour):
        text_surface, text_rect = text_font.render(text_string, fgcolor=text_colour)
        text_rect.topleft = text_pos
        surface.blit(text_surface, text_rect)

    def save_mandelbrot_surface(save_surface):
        asctime = time.asctime().replace(':', '')
        pygame.image.save(save_surface, f"capture/mandelbrot_image {asctime}.png")


    # Initialise pygame
    pygame.init()
    s = pygame.display.set_mode(SCREEN_RESOLUTION)
    fps_clock = pygame.time.Clock()

    # Font
    time_font = pygame.freetype.SysFont(TIME_FONT_NAME, TIME_FONT_SIZE, bold=TIME_BOLD)

    # Create sub process for calculating mandelbrot set
    calc_parent_conn, calc_child_conn = mp.Pipe()
    progress_value = mp.Value('d', 0.0)
    p = mp.Process(target=calculate_mandelbrot, args=(calc_child_conn,progress_value))
    p.start()

    # Ensure mandelbrot file is created
    calc_parent_conn.recv()

    # Map surface array file
    surf_array = np.memmap(SURFARRAY_FILENAME, dtype=SURFARRAY_DTYPE, mode='r+', shape=SURFARRAY_SHAPE)

    # Initialise variables for main loop
    do_loop = True
    is_saved = False

    total_time = 0
    recv_time = False
    start_time = time.time()

    # Main loop
    while do_loop:
        # Fill background
        s.fill(GREY)

        # Draw surf array
        surf_array_surface = pygame.surfarray.make_surface(surf_array)
        surf_array_surface_trans = pygame.transform.scale(surf_array_surface, SCREEN_RESOLUTION)
        s.blit(surf_array_surface_trans, (0, 0))

        # Draw time
        time_text = f"{total_time}s"
        time_text += ['[Rendering]', '[Complete]'][recv_time]
        time_text += ['[Unsaved]', '[Saved]'][is_saved]
        draw_text(s, time_text, time_font, (16, 16), GREEN)

        draw_text(s, f"{progress_value.value}%", time_font, (16, 32), GREEN)

        # Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # The user closed the window!
                if not is_saved and AUTOSAVE:
                    save_mandelbrot_surface(surf_array_surface)
                do_loop = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    save_mandelbrot_surface(surf_array_surface)
                    is_saved = True

        # Redraw screen
        pygame.display.update()
        fps_clock.tick(30)

        # Display time
        if not recv_time:
            # Current elapsed time
            total_time = round(time.time() - start_time, TIME_PRECISION)
            if calc_parent_conn.poll():
                # Receive time from subprocess
                total_time = calc_parent_conn.recv()
                recv_time = True
                is_saved = False

    # Merge subprocess
    p.join(timeout=1)
    p.terminate()

    pygame.quit()
