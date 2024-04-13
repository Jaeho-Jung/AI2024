from queue import PriorityQueue
import numpy as np
from constants import INF, M, N

class Location2D:
    def __init__(self, r=0, c=0):
        self.r = r
        self.c = c

directions = [
    Location2D(-1, 0),
    Location2D(1, 0),
    Location2D(0, -1),
    Location2D(0, 1),
]

def rand_bool(l, n):
    return np.random.permutation(l) < n

# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
def h(node):
    pass

def A_Star(start, goal):

    pq = PriorityQueue()
    pq.put(start)

    '''
    pq.put(1)
    pq.get()
    '''
    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from the start to n currently known.
    cameFrom = [[0]*M for _ in range(N)]

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = np.array([[INF]*M for _ in range(N)])
    gScore[start.r][start.c] = 0

    # For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current best guess as to how cheap a path could be from start to finish if it goes through n.
    fScore = np.array([[INF]*M for _ in range(N)])
    fScore[start.r][start.c] = h(start)

    while not pq.empty():
        cur = pq.get()
        if cur == goal:
            return
        
        for d in directions:
            nxt = Location2D(cur.r+d.r, cur.c+d.c)
            tentative_gScore = gScore[cur.r][cur.c] + 1
            if tentative_gScore < gScore[nxt.r][nxt.c]:
                cameFrom[nxt.r][nxt.c] = cur
                gScore[nxt.r][nxt.c] = tentative_gScore
                fScore[nxt.r][nxt.c] = tentative_gScore + h(nxt)
                if nxt not in pq:
                    pq.put(nxt)
    return False

# A_Star()