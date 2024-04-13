from queue import PriorityQueue
import numpy as np
from constants import INF, M, N

def rand_bool(l, n):
    return np.random.permutation(l) < n

def A_Star(start, goal):

    pq = PriorityQueue()

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


# A_Star()