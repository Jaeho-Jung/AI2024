import pygame
from queue import PriorityQueue

from constants import *
from utils import rand_bool, Location, directions, h

class Grid:
    def __init__(self):
        self.init_grid()
        self.dist_metric = Distance.EUCLIDEAN

    def init_grid(self):
        self.cell = [ [Cell.BLANK]*N for _ in range(M)]
        self.start, self.goal = Location(0, 0), Location(M-1, N-1)
        self.cell[self.start.r][self.start.c] = Cell.START
        self.cell[self.goal.r][self.goal.c] = Cell.GOAL

    def draw_grid(self, screen):
        for x in range(0, GRID_WIDTH+1, CELL_WIDTH):
            pygame.draw.line(screen, Color.BLACK, (x, 0), (x, GRID_HEIGHT))
        for y in range(0, GRID_HEIGHT+1, CELL_HEIGHT):
            pygame.draw.line(screen, Color.BLACK, (0, y), (GRID_WIDTH, y))

    def draw_cells(self, screen):
        for i in range(M):
            for j in range(N):
                color = cell_to_color[self.cell[i][j]]
                pygame.draw.rect(screen, color, (j * CELL_WIDTH+1, i * CELL_HEIGHT+1, CELL_WIDTH-1, CELL_HEIGHT-1))

    def draw_text(self):
        pass

    def toggle_cell(self, pos):
        x, y = pos
        i = y // CELL_HEIGHT
        j = x // CELL_WIDTH

        if not self.is_valid(Location(i, j)):
            return

        self.cell[i][j] = not self.cell[i][j]

    def set_obstacles_randomly(self):
        num_cells = M*N
        # -2: start, goal
        num_obstacles = num_cells*INC_OBSTACLE_RATIO-2

        obstacles = rand_bool(num_cells, num_obstacles)

        idx = 0
        for i in range(M):
            for j in range(N):
                if self.cell[i][j] in [Cell.START, Cell.GOAL]:
                    continue
                self.cell[i][j] = int(obstacles[idx])
                idx += 1

    def toggle_dist_metric(self):
        if self.dist_metric == Distance.EUCLIDEAN:
            self.dist_metric = Distance.MANHATTAN
        elif self.dist_metric == Distance.MANHATTAN:
            self.dist_metric = Distance.EUCLIDEAN

    def is_valid(self, loc):
        return not (loc.r<0 or loc.c<0 or loc.r>=M or loc.c>=N) and self.cell[loc.r][loc.c] != Cell.OBSTACLE

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
        came_from = [[0]*M for _ in range(N)]

        # For node n, g_score[n] is the cost of the cheapest path from start to n currently known.
        g_score = [[INF]*M for _ in range(N)]
        g_score[self.start.r][self.start.c] = 0

        # For node n, f_score[n] := g_score[n] + h(n). f_score[n] represents our current best guess as to how cheap a path could be from start to finish if it goes through n.
        f_score = [[INF]*M for _ in range(N)]
        f_score[self.start.r][self.start.c] = h[self.dist_metric](self.start, self.goal)

        while not frontier.empty():
            cur = frontier.get().loc
            print(cur.r, cur.c)
            if cur == self.goal:
                return True, came_from
            
            for d in directions:
                nxt = Location(cur.r+d.r, cur.c+d.c)
                
                if not self.is_valid(nxt):
                    continue

                tentative_g_score = g_score[cur.r][cur.c] + 1
                if tentative_g_score < g_score[nxt.r][nxt.c]:
                    came_from[nxt.r][nxt.c] = cur
                    g_score[nxt.r][nxt.c] = tentative_g_score
                    f_score[nxt.r][nxt.c] = tentative_g_score + h[self.dist_metric](nxt, self.goal)
                    # frontier.put((f_score[nxt.r][nxt.c], nxt))
                    frontier.put(Item(f_score[nxt.r][nxt.c], nxt))

        return False, None