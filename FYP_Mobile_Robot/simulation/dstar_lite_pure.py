import heapq
import math

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.replan_count = 0
        self.entry_count = 0  # For tie-breaking
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        # Add entry count for stable sorting
        self.entry_count += 1
        heapq.heappush(self.elements, (priority, self.entry_count, item))
    
    def get(self):
        return heapq.heappop(self.elements)[2]
    
    def top_key(self):
        if not self.elements:
            return (float('inf'), float('inf'))
        return self.elements[0][0]
    
    def remove(self, item):
        self.elements = [(p, c, n) for p, c, n in self.elements if n != item]
        heapq.heapify(self.elements)
        
    def increment_replans(self):
        self.replan_count += 1

def dstar_lite(grid, start, goal):
    """
    Improved D* Lite implementation for pathfinding in a grid with weighted costs.
    Args:
        grid: 2D list where each cell contains cost (1-8) or obstacle (9)
        start: Tuple (x, y) of start position
        goal: Tuple (x, y) of goal position
    Returns:
        (path, visited_nodes, total_cost, replan_count)
    """
    rows, cols = len(grid), len(grid[0])
    
    def heuristic(a, b):
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        directions = [(-1,0), (1,0), (0,-1), (0,1), 
                     (-1,-1), (-1,1), (1,-1), (1,1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == 9:
                continue
            # Prevent diagonal movement through obstacles
            if dx != 0 and dy != 0:
                if grid[x + dx][y] == 9 or grid[x][y + dy] == 9:
                    continue
            neighbors.append((nx, ny))
        return neighbors

    def get_cost(current, next_node):
        dx = abs(next_node[0] - current[0])
        dy = abs(next_node[1] - current[1])
        movement_cost = math.sqrt(2) if (dx != 0 and dy != 0) else 1
        return movement_cost * grid[next_node[0]][next_node[1]]

    # Initialize
    g = {}
    rhs = {}
    visited = set()
    U = PriorityQueue()
    
    for i in range(rows):
        for j in range(cols):
            g[(i, j)] = float('inf')
            rhs[(i, j)] = float('inf')
    
    rhs[goal] = 0
    U.put(goal, (heuristic(start, goal), 0))

    def calculate_key(s):
        return (
            min(g[s], rhs[s]) + heuristic(start, s),
            min(g[s], rhs[s])
        )

    def update_vertex(u):
        if u != goal:
            min_rhs = float('inf')
            for s in get_neighbors(u):
                new_rhs = g[s] + get_cost(u, s)
                if new_rhs < min_rhs:
                    min_rhs = new_rhs
            rhs[u] = min_rhs
        
        U.remove(u)
        if g[u] != rhs[u]:
            U.put(u, calculate_key(u))

    def compute_shortest_path():
        while not U.empty():
            k_old = U.top_key()
            u = U.get()
            visited.add(u)
            
            if u == start and g[u] == rhs[u]:
                break
                
            k_new = calculate_key(u)
            if k_old < k_new:
                U.put(u, k_new)
            elif g[u] > rhs[u]:
                g[u] = rhs[u]
                for s in get_neighbors(u):
                    update_vertex(s)
            else:
                g[u] = float('inf')
                update_vertex(u)
                for s in get_neighbors(u):
                    update_vertex(s)

    # Main search
    compute_shortest_path()
    U.increment_replans()  # Count this as a replan

    if g[start] == float('inf'):
        return [], visited, float('inf')

    # Extract path
    path = []
    current = start
    total_cost = 0

    while current != goal:
        path.append(current)
        min_cost = float('inf')
        next_node = None
        
        for neighbor in get_neighbors(current):
            cost = get_cost(current, neighbor) + g[neighbor]
            if cost < min_cost:
                min_cost = cost
                next_node = neighbor
                
        if next_node is None:
            return [], visited, float('inf')
            
        total_cost += get_cost(current, next_node)
        current = next_node

    path.append(goal)
    
    replan_count = U.replan_count  # D* Lite tracks replanning
    return path, visited, total_cost, replan_count
