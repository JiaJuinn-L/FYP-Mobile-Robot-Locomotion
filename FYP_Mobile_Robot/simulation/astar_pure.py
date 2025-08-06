import heapq
import math

def astar(grid, start, goal):
    """
    A* implementation matching the original astar_weighted functionality.
    Args:
        grid: 2D list where each cell contains cost (1-8) or obstacle (9)
        start: Tuple (x, y) of start position
        goal: Tuple (x, y) of goal position
    Returns:
        (path, visited_nodes, total_cost, replan_count)
    """
    rows, cols = len(grid), len(grid[0])
    
    def heuristic(a, b):
        # Diagonal distance heuristic
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    # Priority queue with (f_score, position)
    frontier = [(0 + heuristic(start, goal), start)]
    came_from = {}
    cost_so_far = {start: 0}
    visited = []  # Changed to list to maintain order of exploration
    frontier_set = {start}  # Track current frontier nodes

    # All possible movements (including diagonals)
    directions = [(-1,0), (1,0), (0,-1), (0,1), 
                 (-1,-1), (-1,1), (1,-1), (1,1)]

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        if current in set(visited):  # Convert to set for faster lookup
            continue
            
        visited.append(current)  # Add to list to maintain order
        frontier_set.remove(current)  # Remove from frontier set

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            # Check bounds and obstacles
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == 9:
                continue

            # Prevent diagonal movement through obstacles
            if dx != 0 and dy != 0:
                if grid[x + dx][y] == 9 or grid[x][y + dy] == 9:
                    continue

            # Calculate new cost (use diagonal distance for diagonal moves)
            movement_cost = math.sqrt(2) if (dx != 0 and dy != 0) else 1
            new_cost = cost_so_far[current] + grid[nx][ny] * movement_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                f_score = new_cost + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_score, neighbor))
                frontier_set.add(neighbor)  # Add to frontier set
                came_from[neighbor] = current

    # Reconstruct path
    if goal not in came_from and start != goal:
        return [], visited, float('inf')

    path = []
    current = goal
    total_cost = 0

    while current != start:
        path.append(current)
        prev = came_from[current]
        dx = abs(current[0] - prev[0])
        dy = abs(current[1] - prev[1])
        movement_cost = math.sqrt(2) if (dx != 0 and dy != 0) else 1
        total_cost += grid[current[0]][current[1]] * movement_cost
        current = prev
    path.append(start)
    path.reverse()

    replan_count = 0  # Static case, no replanning
    return path, visited, total_cost, replan_count
