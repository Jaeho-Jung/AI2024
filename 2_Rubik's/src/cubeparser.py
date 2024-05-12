from typing import List

from move import Move


def invert_moves(moves: List[Move]):
    inverted_moves = []

    for move in reversed(moves):
        inverted_move = Move(move.face, not move.invert)
        inverted_moves.append(inverted_move)

    return inverted_moves