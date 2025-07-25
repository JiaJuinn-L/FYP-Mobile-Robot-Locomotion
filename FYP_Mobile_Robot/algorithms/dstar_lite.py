# ----------------- Re-import essentials after code reset -------------------
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy
from statistics import mean
import time
import random
import math
import heapq
import numpy as np
import pandas as pd
from IPython.display import display

# ----------------- Grid Generator -------------------
def generate_weighted_grid(size, obstacle_prob=0.2, max_cost=5, seed=None, start=(0, 0), goal=None):
    if goal is None:
        goal = (size - 1, size - 1)
    rnd = random.Random(seed)
    while True:
        grid = []
        for i in range(size):
            row = []
            for j in range(size):
                if rnd.random() < obstacle_prob:
                    row.append(9)
                else:
                    row.append(rnd.randint(1, max_cost))
            grid.append(row)
        if grid[start[0]][start[1]] != 9 and grid[goal[0]][goal[1]] != 9:
            return grid

# ----------------- Strict Neighbors (No Corner Cutting) -------------------
def get_neighbors(pos, grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    result = []
    rows, cols = len(grid), len(grid[0])
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < rows and 0 <= ny < cols):
            continue
        if grid[nx][ny] == 9:
            continue
        if dx != 0 and dy != 0:
            if grid[pos[0] + dx][pos[1]] == 9 or grid[pos[0]][pos[1] + dy] == 9:
                continue
        result.append((nx, ny))
    return result

# ----------------- D* Lite (Weighted) -------------------
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def put(self, item, priority):
        heapq.heappush(self.queue, (priority, item))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def top_key(self):
        return self.queue[0][0] if self.queue else (math.inf, math.inf)

    def empty(self):
        return not self.queue

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def cost(a, b, grid):
    return grid[b[0]][b[1]] * math.hypot(a[0] - b[0], a[1] - b[1])

class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rhs = {}
        self.g = {}
        self.U = PriorityQueue()
        self.km = 0
        self.last = start
        self.visited = set()
        self.replan_count = 0

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                self.rhs[(x, y)] = math.inf
                self.g[(x, y)] = math.inf
        self.rhs[goal] = 0
        self.U.put(goal, self.calculate_key(goal))

    def calculate_key(self, s):
        return (min(self.g[s], self.rhs[s]) + heuristic(self.start, s) + self.km,
                min(self.g[s], self.rhs[s]))

    def update_vertex(self, u):
        if u != self.goal:
            neighbors = get_neighbors(u, self.grid)
            if neighbors:
                self.rhs[u] = min([self.g.get(s, math.inf) + cost(u, s, self.grid)
                                   for s in neighbors])
            else:
                self.rhs[u] = math.inf
        self.U.queue = [(k, n) for k, n in self.U.queue if n != u]
        if self.g[u] != self.rhs[u]:
            self.U.put(u, self.calculate_key(u))

    def compute_shortest_path(self):
        self.replan_count += 1
        while (self.U.top_key() < self.calculate_key(self.start)) or \
              (self.rhs[self.start] != self.g[self.start]):
            k_old = self.U.top_key()
            u = self.U.get()
            self.visited.add(u)
            k_new = self.calculate_key(u)
            if k_old < k_new:
                self.U.put(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in get_neighbors(u, self.grid):
                    self.update_vertex(s)
            else:
                self.g[u] = math.inf
                self.update_vertex(u)
                for s in get_neighbors(u, self.grid):
                    self.update_vertex(s)

    def get_path(self):
        path = [self.start]
        current = self.start
        while current != self.goal:
            neighbors = get_neighbors(current, self.grid)
            min_cost = math.inf
            next_cell = None
            for n in neighbors:
                c = cost(current, n, self.grid) + self.g.get(n, math.inf)
                if c < min_cost:
                    min_cost = c
                    next_cell = n
            if not next_cell or next_cell == current:
                return None
            current = next_cell
            path.append(current)
        return path

# ----------------- Plotting -------------------
def plot_grid(grid, path, visited_nodes, start, goal, title="D* Lite Pathfinding"):
    rows, cols = len(grid), len(grid[0])
    display_grid = deepcopy(grid)
    max_cost = max(max(row) for row in grid if row) if grid else 5
    terrain_cmap = plt.cm.YlGn
    norm = mcolors.Normalize(vmin=1, vmax=max_cost)
    fig, ax = plt.subplots(figsize=(cols / 5, rows / 5))

    for r in range(rows):
        for c in range(cols):
            if display_grid[r][c] == 9:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='black'))
            else:
                color = terrain_cmap(norm(display_grid[r][c]))
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color=color))

    for r, c in visited_nodes:
        if (r, c) != start and (r, c) != goal:
            ax.plot(c + 0.5, r + 0.5, 'o', color='gray', markersize=3, alpha=0.4)

    if path:
        path_x = [p[1] + 0.5 for p in path]
        path_y = [p[0] + 0.5 for p in path]
        ax.plot(path_x, path_y, color='blue', linewidth=2, zorder=10)

    ax.add_patch(plt.Circle((start[1] + 0.5, start[0] + 0.5), 0.3, color='green', zorder=15, label='Start'))
    ax.add_patch(plt.Circle((goal[1] + 0.5, goal[0] + 0.5), 0.3, color='red', zorder=15, label='Goal'))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.3)
    ax.invert_yaxis()
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()

# ----------------- Test D* Lite -------------------
def dstar_lite(grid_size=10, obstacle_prob=0.2, seed=42):
    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)
    grid = generate_weighted_grid(size=grid_size, obstacle_prob=obstacle_prob, seed=seed, start=start, goal=goal)
    start_time = time.time()
    planner = DStarLite(grid, start, goal)
    planner.compute_shortest_path()
    path = planner.get_path()
    exec_time = round((time.time() - start_time) * 1000, 2)
    plot_grid(grid, path, planner.visited, start, goal, title=f"D* Lite on {grid_size}x{grid_size} grid")
    return {
        "grid_size": f"{grid_size}x{grid_size}",
        "path_length": len(path) if path else 0,
        "visited_count": len(planner.visited),
        "execution_time_ms": exec_time,
        "replan_count": planner.replan_count
    }

# ----------------- Run D* Lite for Configurations -------------------
configurations = [
    (10, 0.2),
    (30, 0.2),
    (50, 0.2),
    (70, 0.2),
    (100, 0.2),
    (200, 0.2),
]

dstar_results = []
for size, prob in configurations:
    result = dstar_lite(grid_size=size, obstacle_prob=prob, seed=1)
    dstar_results.append(result)

df = pd.DataFrame(dstar_results)
display(df)
