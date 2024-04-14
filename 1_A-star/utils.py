import math
import numpy as np
from constants import INF, M, N, Distance

class Location:
    def __init__(self, r=0, c=0):
        self.r = r
        self.c = c
    def __eq__(self, other):
        return self.r == other.r and self.c == other.c
directions = [
    Location(-1, 0),
    Location(1, 0),
    Location(0, -1),
    Location(0, 1),
]

def rand_bool(l, n):
    return np.random.permutation(l) < n

# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
# TODO: choose dist metric
def dist_euclidean(loc1, loc2):
    return math.sqrt((loc1.r-loc2.r)**2 + (loc1.c-loc2.c)**2)

def dist_manhattan(loc1, loc2):
    return abs(loc1.r-loc2.r) + abs(loc1.c-loc2.c)

h = {
    Distance.EUCLIDEAN: dist_euclidean,
    Distance.MANHATTAN: dist_manhattan
}