from color import Color
from piece import Corner
from constants import INIT_FACE_COLOR


class Cube:
    def __init__(self, size: int):
        self.size = size
        self .faces = {
            face: self._generate_face(color) for face, color in INIT_FACE_COLOR
        }

    def _generate_face(self, color: Color):
        return [[color] * self.size for _ in range(self.size)]