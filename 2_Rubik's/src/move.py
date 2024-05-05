from dataclasses import dataclass

'''
    Move
    if invert = True:
        counterclockwise move
'''

@dataclass
class Move:
    face: str
    invert: bool