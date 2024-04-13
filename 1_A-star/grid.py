import pygame

from constants import *
from utils import rand_bool

class Location2D:
    def __init__(self, r=0, c=0):
        self.r = r
        self.c = c

class Grid:
    def __init__(self):
        self.init_grid()

    def init_grid(self):
        self.cell = [ [Cell.BLANK]*N for _ in range(M)]
        self.start, self.goal = Location2D(0, 0), Location2D(M-1, N-1)
        self.cell[self.start.r][self.start.c] = Cell.START
        self.cell[self.goal.r][self.goal.c] = Cell.GOAL

    def draw_grid(self, screen):
        for x in range(0, GRID_WIDTH+1, CELL_WIDTH):
            pygame.draw.line(screen, Color.BLACK, (x, 0), (x, GRID_HEIGHT))
        for y in range(0, GRID_HEIGHT+1, CELL_HEIGHT):
            pygame.draw.line(screen, Color.BLACK, (0, y), (GRID_WIDTH, y))

    def draw_cells(self, screen):
        for i in range(M):
            for j in range(N):
                color = cell_to_color[self.cell[i][j]]
                pygame.draw.rect(screen, color, (j * CELL_WIDTH+1, i * CELL_HEIGHT+1, CELL_WIDTH-1, CELL_HEIGHT-1))

    def draw_text(self, ):
        pass

    def is_valid(self, r, c):
        return not (r<0 or c<0 or r>=M or c>=N)

    def toggle_cell(self, pos):
        x, y = pos
        i = y // CELL_HEIGHT
        j = x // CELL_WIDTH

        if not self.is_valid(i, j):
            return

        self.cell[i][j] = not self.cell[i][j]

    def set_obstacles_randomly(self):
        num_cells = M*N
        # -2: start, goal
        num_obstacles = num_cells*INC_OBSTACLE_RATIO-2

        obstacles = rand_bool(num_cells, num_obstacles)

        idx = 0
        for i in range(M):
            for j in range(N):
                if self.cell[i][j] in [Cell.START, Cell.GOAL]:
                    continue
                self.cell[i][j] = int(obstacles[idx])
                idx += 1