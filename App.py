from mandelbrot_func import *


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

    pygame.quit()
