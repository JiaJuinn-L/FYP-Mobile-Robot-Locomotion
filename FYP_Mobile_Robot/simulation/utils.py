import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy

configurations = [
    (10, 0.2),
    (30, 0.2),
    (50, 0.2),
    (70, 0.2),
    (100, 0.2),
    (200, 0.2),
    (500,0.2)
]

# ----------------- Grid Generator -------------------
def generate_weighted_grid(size, obstacle_prob=0.2, max_cost=10, seed=None, start=(0,0), goal=None):
    if goal is None:
        goal = (size - 1, size - 1)
    rnd = random.Random(seed)
    while True:
        grid = []
        for i in range(size):
            row = []
            for j in range(size):
                if (i,j) == start or (i,j) == goal:
                    row.append(1)  # Ensure start and goal are passable
                elif rnd.random() < obstacle_prob:
                    row.append(float('inf'))  # True obstacles are impassable
                else:
                    # More diverse costs: exponential distribution
                    row.append(int(rnd.uniform(1, max_cost)))
            grid.append(row)
        if grid[start[0]][start[1]] != 9 and grid[goal[0]][goal[1]] != 9:
            return grid

# ----------------- Wrapper -------------------
def wrap_with_cost(fn):
    def wrapped(grid, start, goal):
        res = fn(grid, start, goal)
        if len(res) == 5:
            path, visited, cost, replan_count, avg_operation_time = res
        elif len(res) == 4:
            path, visited, cost, replan_count = res
            avg_operation_time = 0
        elif len(res) == 3:
            path, visited, cost = res
            replan_count = 0
            avg_operation_time = 0
        else:
            path, visited = res
            cost = 0
            replan_count = 0
            avg_operation_time = 0
        return path, visited, cost, replan_count, avg_operation_time
    return wrapped

# ----------------- Plotting -------------------
def plot_grid(grid, path, visited_nodes, start, goal, title="Grid Pathfinding"):
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
