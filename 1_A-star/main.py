import pygame
from pygame.locals import *
from pgu import gui
import sys
import numpy as np

from constant import *

# Initialize Pygame
pygame.init()

# Initialize the screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("A-star Algorithm")

cell = [ [0]*N for _ in range(M)]
start_pos, goal_pos = (0, 0), (M-1, N-1)
cell[start_pos[0]][start_pos[1]] = 's'
cell[goal_pos[0]][goal_pos[1]] = 'g'

def draw_grid():
    for x in range(0, GRID_WIDTH+1, CELL_WIDTH):
        pygame.draw.line(screen, Color.BLACK, (x, 0), (x, GRID_HEIGHT))
    for y in range(0, GRID_HEIGHT+1, CELL_HEIGHT):
        pygame.draw.line(screen, Color.BLACK, (0, y), (GRID_WIDTH, y))

def draw_cells():
    for i in range(M):
        for j in range(N):
            if cell[i][j] == 0:
                color = Color.WHITE
            elif cell[i][j] == 1:
                color = Color.GRAY
            elif cell[i][j] == 's':
                color = Color.RED
            elif cell[i][j] == 'g':
                color = Color.GREEN
            pygame.draw.rect(screen, color, (j * CELL_WIDTH+1, i * CELL_HEIGHT+1, CELL_WIDTH-1, CELL_HEIGHT-1))

def draw_text():
    pass

def is_valid(r, c):
    return not (r<0 or c<0 or r>=M or c>=N)

def toggle_cell(pos):
    x, y = pos
    i = y // CELL_HEIGHT
    j = x // CELL_WIDTH

    if not is_valid(i, j):
        return

    cell[i][j] = not cell[i][j]

def randbool(l, n):
    return np.random.permutation(l) < n

def set_obstacles_randomly():
    global cell

    num_cells = M*N
    # -2: start, goal
    num_obstacles = num_cells*INC_OBSTACLE_RATIO - 2

    obstacles = randbool(num_cells, num_obstacles)

    idx = 0
    for i in range(M):
        for j in range(N):
            if cell[i][j] in ['s', 'g']:
                continue
            cell[i][j] = int(obstacles[idx])
            idx += 1




def start_search():
    pass

def init_cell():
    global cell
    cell = [ [0]*N for _ in range(M)]
    start_pos, goal_pos = (0, 0), (M-1, N-1)
    cell[start_pos[0]][start_pos[1]] = 's'
    cell[goal_pos[0]][goal_pos[1]] = 'g'

def main():
    running = True

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
        draw_grid()
        draw_cells()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    toggle_cell(pygame.mouse.get_pos())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    set_obstacles_randomly()
                if event.key == pygame.K_RETURN:
                    start_search()
                if event.key == pygame.K_r:
                    init_cell()
        # app.paint(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()