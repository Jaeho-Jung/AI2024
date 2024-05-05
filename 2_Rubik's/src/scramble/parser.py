from typing import List, Tuple

from move import Move


def scramble_to_moves(scramble: str) -> List[Move]:
    moves = []

    for move in scramble.split():
        is_invert = "'" in move
        moves.append(Move(move[0], is_invert))

    return moves


def moves_to_scramble(moves: List[Move]) -> str:
    scramble = []

    for move in moves:
        cur_move = move.face
        
        if move.invert:
            cur_move += "'"

        scramble.append(cur_move)
    
    return " ".join(scramble)


def invert_moves(moves: List[Move]):
    inverted_moves = []

    for move in reversed(moves):
        inverted_move = Move(move.face, not move.invert)
        inverted_moves.append(inverted_move)

    return inverted_moves


if __name__ == "__main__":
    scramble = "L U2 D B' R2 U2 F R B2 U2 R2 U R2 U2 F2 D R2 D F2"

    print(scramble_to_moves(scramble))