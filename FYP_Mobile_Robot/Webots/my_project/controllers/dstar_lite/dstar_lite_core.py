import heapq
import math

# Priority queue for D* Lite
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

    def top_key(self):
        return self.elements[0][0] if self.elements else (float('inf'), float('inf'))

    def empty(self):
        return not self.elements

# Heuristic function (Euclidean Distance)
def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Movement cost (includes diagonal distance)
def movement_cost(a, b, grid):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return grid[b[0]][b[1]] * math.hypot(dx, dy)

# Valid neighbors with corner-cut prevention
def neighbors(pos, grid):
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    x, y = pos
    result = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != 9:
            # prevent cutting corners through obstacles
            if dx != 0 and dy != 0:
                if grid[x+dx][y] == 9 or grid[x][y+dy] == 9:
                    continue
            result.append((nx, ny))
    return result

# D* Lite Algorithm (records parent pointers, returns optimal path)
def dstar_lite(grid, start, goal):
    # Initialize priority queue and cost maps
    U = PriorityQueue()
    g   = { (i, j): float('inf') for i in range(len(grid)) for j in range(len(grid[0])) }
    rhs = g.copy()
    parent = {}

    rhs[goal]   = 0
    parent[goal] = None
    U.put(goal, (heuristic(start, goal), 0))

    # Calculate keys with Euclidean heuristic
    def calculate_key(u):
        val = min(g[u], rhs[u])
        return (val + heuristic(start, u), val)

    # Update or insert a vertex
    def update_vertex(u):
        if u != goal:
            best_val = float('inf')
            best_s = None
            for s in neighbors(u, grid):
                cost = g[s] + movement_cost(s, u, grid)
                if cost < best_val:
                    best_val, best_s = cost, s
            rhs[u] = best_val
            parent[u] = best_s
        # Remove old queue entries
        U.elements = [(p,n) for p,n in U.elements if n != u]
        if g[u] != rhs[u]:
            U.put(u, calculate_key(u))

    # Main D* Lite loop
    iterations = 0
    while not U.empty() and (U.top_key() < calculate_key(start) or rhs[start] != g[start]):
        _, u = heapq.heappop(U.elements)
        if g[u] > rhs[u]:
            g[u] = rhs[u]
        else:
            g[u] = float('inf')
            update_vertex(u)
        for s in neighbors(u, grid):
            update_vertex(s)
        iterations += 1
        if iterations > 10000:
            return [], iterations

    # Reconstruct path from start→goal via parent pointers
    if rhs[start] == float('inf'):
        return [], iterations
    path = []
    node = start
    while node is not None:
        path.append(node)
        if node == goal:
            break
        node = parent.get(node)
    return path, iterations