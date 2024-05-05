from typing import List, TypeVar, Union

import numpy as np

from move import Move
from color import Color, color_to_char
from piece import Corner, CORNER_TO_URF
from scramble import parser
from constants import INIT_FACE_COLOR

'''
    size = 2
    faces =  {
        'U': [[W,W],[W,W]]
        'R': [[R,R],[R,R]]
        'F': [[G,G],[G,G]]
        'D': [[Y,Y],[Y,Y]]
        'L': [[O,O],[O,O]]
        'B': [[B,B],[B,B]]
    }
'''

class Cube:
    def __init__(self, size: int):
        self.size = size
        self.faces = {
            face: self._generate_face(color, self.size) for face, color in INIT_FACE_COLOR
        }
    
    def get_state(self) -> str:
        return ''.join(color_to_char[color] for face in self.faces.values() for row in face for color in row)

    def get_color(self, position: str, orientation: int) -> Color:
        return self.get_corner(position)[orientation]

    def get_corner(self, position: str) -> Corner:
        moves = parser.scramble_to_moves(CORNER_TO_URF[position])

        self.do_moves(moves)
        corner = Corner({
            position[0]: Color(self.faces['U'][-1][-1]), 
            position[1]: Color(self.faces['R'][0][0]),
            position[2]: Color(self.faces['F'][0][-1])
        })
        inverted_moves = parser.invert_moves(moves)
        self.do_moves(inverted_moves)

        return corner

    def do_moves(self, moves: Union[str, List[Move]]):
        if isinstance(moves, str):
            moves = parser.scramble_to_moves(moves)
            
        for move in moves:
            self._rotate(move)

    def is_solved(self) -> bool:
        return not any(piece_colour != face[0][0] for face in self.faces.values() for row in face for piece_colour in row)
    
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

    def _rotate(self, move: Move):
        for _ in range(3 if move.invert else 1):
            self._face_rotate(move.face)
            self._adjacent_face_swap(move.face)


T = TypeVar('T')
def _transpose(l: List[List[T]]) -> List[List[T]]:
    return [list(i) for i in zip(*l)]

cube = Cube(2)