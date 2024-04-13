import pygame
from pygame.locals import *
# from pgu import gui
import numpy as np
import sys

from constants import *
from grid import Grid

# Initialize Pygame
pygame.init()

# Initialize the screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("A-star Algorithm")

def start_search():
    pass

def main():
    running = True

    grid = Grid()
    grid.init_grid()

    # app = gui.App()
    # container = gui.Container(width=600, height=100)
    # app.init(container)

    # btn_search = gui.Button('Start A* Search')
    # btn_search.connect(locals.CLICK, start_search)
    # container.add(btn_search, 10, 10)

    # btn_search = gui.Button('Random walls')
    # btn_search.connect(locals.CLICK, set_obstacles_randomly)
    # container.add(btn_search, 210, 10)

    # btn_search = gui.Button('Reset')
    # btn_search.connect(locals.CLICK, reset)
    # container.add(btn_search, 410, 10)

    while running:
        screen.fill(Color.WHITE)
        grid.draw_grid(screen)
        grid.draw_cells(screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    grid.toggle_cell(pygame.mouse.get_pos())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    grid.set_obstacles_randomly()
                if event.key == pygame.K_RETURN:
                    start_search()
                if event.key == pygame.K_r:
                    grid.init_grid()
                if event.key == pygame.K_q:
                    running = False
        # app.paint(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()