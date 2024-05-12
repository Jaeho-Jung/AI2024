from typing import List, TypeVar, Union

import numpy as np

from move import Move
from cubie import Cubie
from color import Color
from constants import INIT_FACE_COLOR, INIT_CUBIE_STATE

'''
    size = 2
    faces =  {
        'U': [[W,W],[W,W]],
        'R': [[R,R],[R,R]],
        'F': [[G,G],[G,G]],
        'D': [[Y,Y],[Y,Y]],
        'L': [[O,O],[O,O]],
        'B': [[B,B],[B,B]],
    }
    cubies = {
        'ULB': Cubie('ULB', 0),
        'URB': Cubie('URB', 0),
        'ULF': Cubie('ULF', 0),
        'URF': Cubie('URF', 0),
        'DLB': Cubie('DLB', 0),
        'DRB': Cubie('DRB', 0),
        'DLF': Cubie('DLF', 0),
        'DRF': Cubie('DRF', 0),
    }
'''

class Cube:
    def __init__(self, size: int):
        self.size = size
        self.faces = {
            face: self._generate_face(color, self.size) for face, color in INIT_FACE_COLOR
        }
        self.cubies = {
            position: Cubie(position, orientation) for position, orientation in INIT_CUBIE_STATE
        }
    
    def set_cubies(self, cubies):
        self.cubies = cubies

    def get_state(self) -> str:
        return str(self.cubies)

    def get_cubies(self):
        return self.cubies

    def do_moves(self, moves: Union[str, List[Move]]):
        for move in moves:
            self._rotate(move)

    def is_solved(self) -> bool:
        cubies = {
            position: Cubie(position, orientation) for position, orientation in INIT_CUBIE_STATE
        }
        return self.cubies == cubies
    
    def _generate_face(self, color: Color, size: int) -> np.ndarray:
        return np.array([[color] * size for _ in range(size)])

    # 면 회전
    # clockwise, 90°
    def _face_rotate(self, face: str):
        self.faces[face] = np.array([list(row) for row in zip(*self.faces[face][::-1])])

    # 면 인접 조각 회전
    # clockwise, 90°
    def _adjacent_face_swap(self, face: str):
        if face == 'U':
            tmp = [self.faces[face][0].copy() for face in ['F', 'L', 'B', 'R']]

            self.faces['F'][0], self.faces['L'][0], self.faces['B'][0], self.faces['R'][0] = tmp[-1:] + tmp[:-1]

        elif face == 'D':
            tmp = [self.faces[face][-1].copy() for face in ['F', 'R', 'B', 'L']]

            self.faces['F'][-1], self.faces['R'][-1], self.faces['B'][-1], self.faces['L'][-1] = tmp[-1:] + tmp[:-1]

        elif face == 'F':
            tmp = [self.faces['U'][-1].copy(), self.faces['R'][:,0][::-1].copy(), self.faces['D'][0].copy(), self.faces['L'][:,-1][::-1].copy()]

            self.faces['U'][-1], self.faces['R'][:,0], self.faces['D'][0], self.faces['L'][:,-1] = tmp[-1:] + tmp[:-1]

        elif face == 'B':
            tmp = [self.faces['U'][0].copy(), self.faces['L'][:,0].copy(), self.faces['D'][-1].copy(), self.faces['R'][:,-1].copy()]

            self.faces['U'][0], self.faces['L'][:,0][::-1], self.faces['D'][-1], self.faces['R'][:,-1][::-1] = tmp[-1:] + tmp[:-1]

        elif face == 'R':
            tmp = [self.faces['U'][:,-1].copy(), self.faces['B'][:,0].copy(), self.faces['D'][:,-1].copy(), self.faces['F'][:,-1].copy()]

            self.faces['U'][:,-1], self.faces['B'][:,0][::-1], self.faces['D'][:,-1][::-1], self.faces['F'][:,-1] = tmp[-1:] + tmp[:-1]

        elif face == 'L':
            tmp = [self.faces['U'][:,0].copy(), self.faces['F'][:,0].copy(), self.faces['D'][:,0].copy(), self.faces['B'][:,-1].copy()]

            self.faces['U'][:,0][::-1], self.faces['F'][:,0], self.faces['D'][:,0], self.faces['B'][:,-1][::-1] = tmp[-1:] + tmp[:-1]

    def _cubie_rotate(self, face: str):
        if face == 'U':
            tmp = [self.cubies[cubie] for cubie in ['URF', 'ULF', 'ULB', 'URB']]

            self.cubies['URF'], self.cubies['ULF'], self.cubies['ULB'], self.cubies['URB'] = tmp[-1:] + tmp[:-1]

        elif face == 'D':
            tmp = [self.cubies[cubie] for cubie in ['DRF', 'DRB', 'DLB', 'DLF']]

            self.cubies['DRF'], self.cubies['DRB'], self.cubies['DLB'], self.cubies['DLF'] = tmp[-1:] + tmp[:-1]

        elif face == 'F':
            tmp = [self.cubies['ULF'], self.cubies['URF'], self.cubies['DRF'], self.cubies['DLF']]

            tmp[0].orientation = (tmp[0].orientation + 1) % 3
            tmp[1].orientation = (tmp[1].orientation + 2) % 3
            tmp[2].orientation = (tmp[2].orientation + 1) % 3
            tmp[3].orientation = (tmp[3].orientation + 2) % 3

            self.cubies['ULF'], self.cubies['URF'], self.cubies['DRF'], self.cubies['DLF'] = tmp[-1:] + tmp[:-1]

        elif face == 'B':
            tmp = [self.cubies['URB'], self.cubies['ULB'], self.cubies['DLB'], self.cubies['DRB']]

            tmp[0].orientation = (tmp[0].orientation + 1) % 3
            tmp[1].orientation = (tmp[1].orientation + 2) % 3
            tmp[2].orientation = (tmp[2].orientation + 1) % 3
            tmp[3].orientation = (tmp[3].orientation + 2) % 3

            self.cubies['URB'], self.cubies['ULB'], self.cubies['DLB'], self.cubies['DRB'] = tmp[-1:] + tmp[:-1]

        elif face == 'R':
            tmp = [self.cubies['URF'], self.cubies['URB'], self.cubies['DRB'], self.cubies['DRF']]

            tmp[0].orientation = (tmp[0].orientation + 1) % 3
            tmp[1].orientation = (tmp[1].orientation + 2) % 3
            tmp[2].orientation = (tmp[2].orientation + 1) % 3
            tmp[3].orientation = (tmp[3].orientation + 2) % 3

            self.cubies['URF'], self.cubies['URB'], self.cubies['DRB'], self.cubies['DRF'] = tmp[-1:] + tmp[:-1]

        elif face == 'L':
            tmp = [self.cubies['ULB'], self.cubies['ULF'], self.cubies['DLF'], self.cubies['DLB']]

            tmp[0].orientation = (tmp[0].orientation + 1) % 3
            tmp[1].orientation = (tmp[1].orientation + 2) % 3
            tmp[2].orientation = (tmp[2].orientation + 1) % 3
            tmp[3].orientation = (tmp[3].orientation + 2) % 3

            self.cubies['ULB'], self.cubies['ULF'], self.cubies['DLF'], self.cubies['DLB'] = tmp[-1:] + tmp[:-1]

    def _rotate(self, move: Move):
        for _ in range(3 if move.invert else 1):
            # rotate faces
            self._face_rotate(move.face)
            self._adjacent_face_swap(move.face)
            # rotate cubies
            self._cubie_rotate(move.face)