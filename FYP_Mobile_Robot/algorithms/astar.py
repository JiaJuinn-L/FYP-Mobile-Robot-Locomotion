import heapq
import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy

# ----------------- A* Weighted -------------------
def astar_weighted(grid, start, goal):
    rows, cols = len(grid), len(grid[0])

    def h(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    frontier = [(0 + h(start, goal), start)]
    came_from = {}
    cost_so_far = {start: 0}
    visited_nodes = set()
    replan_count = 0  # Static case, no dynamic changes for now

    while frontier:
        f_score, current = heapq.heappop(frontier)

        if current in cost_so_far and f_score > cost_so_far[current] + h(current, goal):
            continue
        if current == goal:
            break
        if current in visited_nodes:
            continue
        visited_nodes.add(current)

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == 9:
                continue
            if dx != 0 and dy != 0:
                if grid[x + dx][y] == 9 or grid[x][y + dy] == 9:
                    continue
            step_cost = grid[nx][ny] * math.hypot(dx, dy)
            new_cost = cost_so_far[current] + step_cost
            neighbor = (nx, ny)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + h(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    if goal not in came_from and start != goal:
        return [], visited_nodes, float('inf'), replan_count

    path = []
    current_node = goal
    while current_node != start:
        if current_node not in came_from:
            return [], visited_nodes, float('inf'), replan_count
        path.append(current_node)
        current_node = came_from[current_node]
    path.append(start)
    path.reverse()

    total_path_cost = 0
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            dx = abs(u[0] - v[0])
            dy = abs(u[1] - v[1])
            total_path_cost += grid[v[0]][v[1]] * math.hypot(dx, dy)
    elif path and start == goal:
        total_path_cost = 0

    return path, visited_nodes, total_path_cost, replan_count

# ----------------- Grid Generator -------------------
def generate_weighted_grid(size, obstacle_prob=0.2, max_cost=5, seed=None, start=(0,0), goal=None):
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

# ----------------- Wrapper -------------------
def wrap_with_cost(fn):
    def wrapped(grid, start, goal):
        res = fn(grid, start, goal)
        if len(res) == 4:
            path, visited, cost, replan_count = res
        elif len(res) == 3:
            path, visited, cost = res
            replan_count = 0
        else:
            path, visited = res
            cost = 0
            replan_count = 0
        return path, visited, cost, replan_count
    return wrapped

astar_cost = wrap_with_cost(astar_weighted)

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

# ----------------- Run Configurations -------------------
configurations = [
    (10, 0.2),
    (30, 0.2),
    (50, 0.2),
    (70, 0.2),
    (100, 0.2),
    (200, 0.2),
    (500,0.2)
]


print("Running A* visualization for different grid sizes and obstacle probabilities...")
results_a_star = []

for size, prob in configurations:
    start_node = (0, 0)
    goal_node = (size - 1, size - 1)
    grid = generate_weighted_grid(size, prob, seed=1, start=start_node, goal=goal_node)

    print(f"\n--- A* on {size}x{size} Grid (Obstacle Prob: {prob}) ---")
    path, visited_nodes, total_cost, replan_count = astar_cost(grid, start_node, goal_node)

    if path:
        print(f"Path found! Length: {len(path)}, Visited Nodes: {len(visited_nodes)}, Total Cost: {total_cost:.2f}, Replans: {replan_count}")
    else:
        print(f"No path found. Visited Nodes: {len(visited_nodes)}, Total Cost: {total_cost:.2f}, Replans: {replan_count}")

    results_a_star.append({
        'grid_size': f"{size}x{size}",
        'obstacle_prob': prob,
        'path_found': bool(path),
        'path_length': len(path) if path else 0,
        'visited_nodes': len(visited_nodes),
        'total_cost': total_cost,
        'replan_count': replan_count
    })

    if size <= 100:
        plot_grid(grid, path, visited_nodes, start_node, goal_node,
                  title=f"A* Path on {size}x{size} Grid (Obstacle Prob: {prob})")
    else:
        print(f"Skipping visualization for {size}x{size} grid due to size.")

# ----------------- Print Summary -------------------
print("\n--- A* Visualization Results ---")
for res in results_a_star:
    print(res)
