from collections import defaultdict
import copy
import itertools

from cube import Cube, INIT_CUBIE_STATE
from cubie import Cubie
from move import Move
from cubeparser import invert_moves
from scramble import load_state

class Solver:
    def __init__(self):
        self.goal = {
            position: Cubie(position, orientation) for position, orientation in INIT_CUBIE_STATE
        }
        self.moves = [ Move(move, invert) for move, invert in list(itertools.product(['F', 'B', 'R', 'L', 'U', 'D'], [True, False]))]
        self.positions = ['ULB', 'URB', 'ULF', 'URF', 'DLB', 'DRB', 'DLF', 'DRF']
        self.orientations = [0, 1, 2]

        self.min_move_table = defaultdict(lambda: defaultdict(int))
        self.solution = []
        self._init_table()

    def _init_table(self):
        for pos, ori in list(itertools.product(self.positions, self.orientations)):
            for pos2, ori2 in list(itertools.product(self.positions, self.orientations)):
                self.min_move_table[(pos, ori)][(pos2, ori2)] = self.min_move(Cubie(pos, ori), Cubie(pos2, ori2))


    def heuristic_summation(self, cube: Cube):
        sum = 0
        for pos in self.positions:
            sum += self.min_move_table[(cube.cubies[pos].position,cube.cubies[pos].orientation)][(self.goal[pos].position, self.goal[pos].orientation)]
        
        return sum

    def heuristic_maximum(self, cube: Cube):
        maximum = 0
        for pos in self.positions:
            maximum = max(maximum, self.min_move_table[(cube.cubies[pos].position,cube.cubies[pos].orientation)][(self.goal[pos].position, self.goal[pos].orientation)])

        return maximum

    def min_move(self, start: Cubie, goal: Cubie):
        cube = Cube(2)
        start.orientation = (start.orientation - goal.orientation) % 3
        goal.orientation = 0
        
        cubies = {position: Cubie('', 0) for position in cube.cubies}
        cubies[start.position] = Cubie(goal.position, start.orientation)
        cube.set_cubies(cubies)

        # BFS
        queue = list()
        visited = list()

        queue.append((cube, 0))
        visited.append(start)


        while queue:
            (cube, cnt) = queue.pop(0)

            cur_cubie = [Cubie(position, cubie.orientation) for position, cubie in cube.cubies.items() if cubie.position == goal.position][0]
            if cur_cubie == goal:
                return cnt

            for move in self.moves:
                cur_cube = copy.deepcopy(cube)
                cur_cube.do_moves([move])
                nxt_cubie = [Cubie(position, cubie.orientation) for position, cubie in cur_cube.cubies.items() if cubie.position == goal.position][0]
                if nxt_cubie not in visited:
                    visited.append(nxt_cubie)
                    queue.append((cur_cube, cnt+1))
                    # print(move.face + "'" if move.invert else move.face)

        return -1

    def ida_star(self, cube: Cube, heuristic):
        self.solution = []
        threshold = heuristic(cube)
        while True:
            result = self.search(cube, 0, threshold, heuristic)
            if result == "found":
                return self.solution
            if result == float('inf'):
                return "No solution exists"
            threshold = result
            # print(threshold)

    def search(self, cube: Cube, g: int, threshold: int, heuristic):
        # Check if the cube is solved
        if cube.is_solved():
            return "found"
        
        # Estimate the number of moves to solve
        f = g + heuristic(cube)
        # Return if the estimate is too high
        if f > threshold:
            return f
        
        min_cost = float('inf')
        
        # Try every move
        for move in self.moves:
            # Try the move and add itto our solution stack
            cube.do_moves([move])
            self.solution.append(move)
            # Search the subtree from this new state
            cost = self.search(cube, g + 1, threshold, heuristic)
            if cost == "found":
                return "found"
            min_cost = min(min_cost, cost)
            cube.do_moves(invert_moves([move]))
            self.solution.pop()
        return min_cost

    def solve(self, cube: Cube):
        with open('solution.txt', 'w+') as f:
            f.write(str(self.ida_star(cube, self.heuristic_summation)))
            f.write('\n')
            f.write(str(self.ida_star(cube, self.heuristic_maximum)))
    

def main():
    solver = Solver()
    
    cubies = load_state('scramble_state.txt')
    cube = Cube(2)
    cube.set_cubies(cubies)

    solver.solve(cube)

if __name__ == "__main__":
    main()
