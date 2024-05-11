import numpy as np
from enum import IntEnum

INF = np.inf

class Cell(IntEnum):
    BLANK = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4

class Color():
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    RED = (255, 0, 0)
    GREEN = (0, 0, 255)
    YELLOW = (255, 255, 0)

class Distance(IntEnum):
    MANHATTAN = 0,
    EUCLIDEAN = 1

