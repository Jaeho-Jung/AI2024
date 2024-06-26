from pprint import pprint
import time

import pygame
from pygame.locals import *

from cube import Cube
from move import Move

HEIGHT = 1024
WIDTH = 1440
CUBIE_SIZE = 125
HORIZONTAL_START = 100

class Gui:
    def __init__(self, cube: Cube):
        self.cube = cube
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

    def run(self):
        self.draw_cube()

        running = True
        while running:
            for event in pygame.event.get():
                prime = pygame.key.get_pressed()[pygame.K_LSHIFT]
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)

                    if key in {"u", "f", "l", "r", "d", "b"}:
                        self.cube._rotate(Move(key.upper(), prime))
                        self.draw_cube()
                        print(self.cube.get_cubies())
                    if key == 'q':
                        running = False
            
    def draw_cube(self):
        for face_num, face in enumerate(["U", "F", "D", "B", "L", "R"]):
            for row_num, row in enumerate(self.cube.faces[face]):
                for cubie_num, cubie in enumerate(row):
                    if face == "L":
                        face_num = 1
                        horizontal_adjust = - self.cube.size * CUBIE_SIZE
                    elif face == "R":
                        face_num = 1
                        horizontal_adjust = self.cube.size * CUBIE_SIZE
                    elif face == "B":
                        face_num = 1
                        horizontal_adjust = 2 * self.cube.size * CUBIE_SIZE
                    else:
                        horizontal_adjust = 0
                        
                    x = WIDTH / 3 + cubie_num * CUBIE_SIZE + horizontal_adjust
                    y = self.cube.size * face_num * CUBIE_SIZE + row_num * CUBIE_SIZE + HORIZONTAL_START
                    
                    pygame.draw.rect(self.screen, cubie, (x, y, CUBIE_SIZE, CUBIE_SIZE), 0)
                    pygame.draw.rect(self.screen, (0, 0, 0), (x, y, CUBIE_SIZE, CUBIE_SIZE), 5)

        pygame.display.update()
