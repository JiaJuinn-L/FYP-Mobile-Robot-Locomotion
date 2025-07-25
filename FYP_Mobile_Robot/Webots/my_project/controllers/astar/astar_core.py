import heapq
import math

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    # Estimate average terrain cost (excluding obstacles)
    flat_cells = [cell for row in grid for cell in row if cell != 9]
    avg_cost = sum(flat_cells) / len(flat_cells) if flat_cells else 1.0

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) * avg_cost

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    frontier = [(heuristic(start, goal), start)]
    came_from = {}
    cost_so_far = {start: 0}
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 9:
                # Prevent diagonal corner-cutting
                if dx != 0 and dy != 0:
                    if grid[x + dx][y] == 9 or grid[x][y + dy] == 9:
                        continue

                move_cost = grid[nx][ny] * math.hypot(dx, dy)
                new_cost = cost_so_far[current] + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return [], visited  # No path
    path.append(start)
    path.reverse()
    return path, visited
