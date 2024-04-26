from pprint import pprint

import pygame
from pygame.locals import *
from grid import Grid
from utils import Location
from constants import Cell, Color

WINDOW_WIDTH, WINDOW_HEIGHT = 700, 700
GRID_WIDTH, GRID_HEIGHT = 600, 600

class Gui:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.cell_width = GRID_WIDTH // grid.n
        self.cell_height = GRID_HEIGHT // grid.m

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    def run(self):
        self.draw_grid()
        self.draw_cells()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.grid.toggle_cell(self._pos_to_loc(pygame.mouse.get_pos()))
                        self.draw_cells()

                if event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)

                    if key == 'space':
                        self.grid.set_obstacles_randomly()
                        self.draw_cells()
                    if key == 'return':
                        self.start_search()
                        self.draw_cells()
                        pass
                    if key == 'r':
                        self.grid.init_grid()
                        self.draw_cells()
                    elif key == 'q':
                        running = False
                    elif key == 'm':
                        self.grid.clear_path()
                        self.grid.toggle_dist_metric()
                        self.draw_cells()
        
        pygame.quit()

    def _pos_to_loc(self, pos):
        x, y = pos
        r = y // self.cell_height
        c = x // self.cell_width

        return Location(r, c)

    def draw_grid(self):
        self.screen.fill(Color.WHITE)
        for x in range(0, GRID_WIDTH+1, self.cell_width):
            pygame.draw.line(self.screen, Color.BLACK, (x, 0), (x, GRID_HEIGHT))
        for y in range(0, GRID_HEIGHT+1, self.cell_height):
            pygame.draw.line(self.screen, Color.BLACK, (0, y), (GRID_WIDTH, y))

        pygame.display.update()

    _cell_to_color = {
        Cell.BLANK: Color.WHITE,
        Cell.OBSTACLE: Color.GRAY,
        Cell.START: Color.RED,
        Cell.GOAL: Color.GREEN,
        Cell.PATH: Color.YELLOW,
    }

    def draw_cells(self):
        for i in range(self.grid.m):
            for j in range(self.grid.n):
                color = self._cell_to_color[self.grid.cell[i][j]]
                pygame.draw.rect(self.screen, color, (j * self.cell_width+1, i * self.cell_height+1, self.cell_width-1, self.cell_height-1))
        
        pygame.display.update()

    
    def draw_text(self):
        pass

    def start_search(self):
        ret, came_from, target, n_explored_nodes = self.grid.a_star()

        if ret == True:
            print('Number of explored nodes: ' + str(n_explored_nodes))
        elif ret == False:
            print('Cannot find path')

        cur = came_from[target.r][target.c]
        while cur != self.grid.start:
            self.grid.cell[cur.r][cur.c] = Cell.PATH
            cur = came_from[cur.r][cur.c]