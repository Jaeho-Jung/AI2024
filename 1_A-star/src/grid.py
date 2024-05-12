from queue import PriorityQueue

from constants import *
from utils import rand_bool, Location, directions, h

class Grid:
    def __init__(self, m: int, n: int, obs_ratio: float):
        self.m = m
        self.n = n
        self.obs_ratio = obs_ratio
        self.init_grid()
        self.dist_metric = Distance.EUCLIDEAN

    def init_grid(self):
        self.cell = [ [Cell.BLANK]*self.n for _ in range(self.m)]
        self.start, self.goal = Location(0, 0), Location(self.m-1, self.n-1)
        self.cell[self.start.r][self.start.c] = Cell.START
        self.cell[self.goal.r][self.goal.c] = Cell.GOAL

    def clear_path(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.cell[i][j] == Cell.PATH:
                    self.cell[i][j] = Cell.BLANK

    def toggle_cell(self, loc: Location):
        if not self.is_valid(loc):
            return
        
        if self.cell[loc.r][loc.c] in {Cell.BLANK, Cell.PATH}:
            self.cell[loc.r][loc.c] = Cell.OBSTACLE
        elif self.cell[loc.r][loc.c] == Cell.OBSTACLE:
            self.cell[loc.r][loc.c] = Cell.BLANK

    def set_obstacles_randomly(self):
        num_cells = self.m*self.n
        # -2: start, goal
        num_obstacles = num_cells*self.obs_ratio-2

        obstacles = rand_bool(num_cells, num_obstacles)

        idx = 0
        for i in range(self.m):
            for j in range(self.n):
                if self.cell[i][j] in [Cell.START, Cell.GOAL]:
                    continue
                self.cell[i][j] = int(obstacles[idx])
                idx += 1

    def get_dist_metric(self):
        return self.dist_metric

    def toggle_dist_metric(self):
        if self.dist_metric == Distance.EUCLIDEAN:
            self.dist_metric = Distance.MANHATTAN
        elif self.dist_metric == Distance.MANHATTAN:
            self.dist_metric = Distance.EUCLIDEAN

    def is_valid(self, loc):
        return not (loc.r<0 or loc.c<0 or loc.r>=self.m or loc.c>=self.n)

    def a_star(self):
        class Item:
            def __init__(self, priority, loc):
                self.priority = priority
                self.loc = loc
            
            def __lt__(self, other):
                return self.priority < other.priority
            
        frontier = PriorityQueue()
        frontier.put(Item(0, self.start))

        # For node n, came_from[n] is the node immediately preceding it on the cheapest path from the start to n currently known.
        came_from = [[Location()]*self.n for _ in range(self.m)]

        # For node n, g_score[n] is the cost of the cheapest path from start to n currently known.
        g_score = [[INF]*self.n for _ in range(self.m)]
        g_score[self.start.r][self.start.c] = 0
        # print(f'{len(g_score)}, {len(g_score[0])}')

        # For node n, f_score[n] := g_score[n] + h(n). f_score[n] represents our current best guess as to how cheap a path could be from start to finish if it goes through n.
        f_score = [[INF]*self.n for _ in range(self.m)]
        f_score[self.start.r][self.start.c] = h[self.dist_metric](self.start, self.goal)

        target = Location()
        min_f = INF
        n_explored_nodes = 0
        while not frontier.empty():
            cur = frontier.get().loc

            n_explored_nodes += 1

            if cur == self.goal:
                return True, came_from, self.goal, n_explored_nodes
            
            for d in directions:
                nxt = Location(cur.r+d.r, cur.c+d.c)
                
                if not self.is_valid(nxt):
                    continue
                if self.cell[nxt.r][nxt.c] == Cell.OBSTACLE:
                    continue

                # print(f'{nxt.r}, {nxt.c}')
                tentative_g_score = g_score[cur.r][cur.c] + 1
                if tentative_g_score < g_score[nxt.r][nxt.c]:
                    came_from[nxt.r][nxt.c] = cur
                    g_score[nxt.r][nxt.c] = tentative_g_score
                    f_score[nxt.r][nxt.c] = tentative_g_score + h[self.dist_metric](nxt, self.goal)
                    
                    if min_f > f_score[nxt.r][nxt.c]:
                        target = nxt

                    frontier.put(Item(f_score[nxt.r][nxt.c], nxt))

        return False, came_from, target, None