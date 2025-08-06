import heapq
import math

# A* pathfinding algorithm with terrain-weight-aware heuristic
def astar(grid, start, goal):
    def heuristic(a, b):
        base = math.hypot(a[0] - b[0], a[1] - b[1])  # Euclidean distance
        return base * average_cost_estimate  # Incorporate expected terrain weight

    # Estimate average terrain cost (excluding obstacles)
    flat_cells = [cell for row in grid for cell in row if cell != 9]
    average_cost_estimate = sum(flat_cells) / len(flat_cells) if flat_cells else 1.0

    rows, cols = len(grid), len(grid[0])
    visited = set()
    came_from = {}
    cost_so_far = {start: 0}
    heap = [(heuristic(start, goal), start)]

    # Allow 8 directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap:
        _, current = heapq.heappop(heap)
        if current == goal:
            break
        if current in visited:
            continue
        visited.add(current)

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            # Bounds and obstacle check
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 9:
                # Prevent diagonal corner-cutting
                if dx != 0 and dy != 0:
                    if grid[x + dx][y] == 9 or grid[x][y + dy] == 9:
                        continue

                terrain_cost = grid[nx][ny]
                move_cost = terrain_cost * math.hypot(dx, dy)
                new_cost = cost_so_far[current] + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(heap, (priority, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return [], 0  # No path found
    path.append(start)
    path.reverse()
    return path, len(visited)
