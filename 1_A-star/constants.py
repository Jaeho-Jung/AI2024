import argparse
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description="A-star Algorithm")
parser.add_argument("--m", type=int, default=30, help="Number of rows")
parser.add_argument("--n", type=int, default=30, help="Number of columns")
parser.add_argument("--obs", type=float, default=0.2, help="Obstacle ratio")
args = parser.parse_args()

# Set up grid dimensions and cell size
M, N = args.m, args.n
INC_OBSTACLE_RATIO = args.obs
WINDOW_WIDTH, WINDOW_HEIGHT = 700, 700
GRID_WIDTH, GRID_HEIGHT = 600, 600
CELL_WIDTH, CELL_HEIGHT = GRID_WIDTH // N, GRID_HEIGHT // M

INF = np.inf

class Cell():
    BLANK = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3

class Color():
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    RED = (255, 0, 0)
    GREEN = (0, 0, 255)

cell_to_color = {
    Cell.BLANK: Color.WHITE,
    Cell.OBSTACLE: Color.GRAY,
    Cell.START: Color.RED,
    Cell.GOAL: Color.GREEN,
}