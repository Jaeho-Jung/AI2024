import argparse
import itertools
import random

from move import Move
from cube import Cube
from cubie import Cubie

def save_state(file_name: str, state: str):
    with open(file_name, 'w+') as f:
        f.write(state)

def load_state(file_name: str):
    with open(file_name, 'r') as f:
        line = f.readline()
        return eval(line)

def scramble(file_name: str, n: int):
    moves = [ Move(move, invert) for move, invert in list(itertools.product(['F', 'B', 'R', 'L', 'U', 'D'], [True, False]))]

    cube = Cube(2)

    random_moves = random.choices(moves, k=n)

    cube.do_moves(random_moves)

    save_state(file_name, cube.get_state())

def main():
    parser = argparse.ArgumentParser(description="scramble")
    parser.add_argument("--n", type=int, default=20, help="Number of moves")
    args = parser.parse_args()

    scramble('scramble_state.txt', args.n)
    # load_state('scramble_state.txt')

if __name__ == "__main__":
    main()