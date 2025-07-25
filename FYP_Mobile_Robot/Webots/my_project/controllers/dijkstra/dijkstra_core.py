import heapq
import math

def dijkstra(grid, start, goal):
    visited = set()
    came_from = {}
    cost_so_far = {start: 0}
    heap = [(0,start)]
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while heap:
        cost, cur = heapq.heappop(heap)
        if cur == goal: break
        if cur in visited: continue
        visited.add(cur)
        x,y = cur
        for dx,dy in dirs:
            nx,ny = x+dx, y+dy
            nb = (nx,ny)
            if not (0<=nx<len(grid) and 0<=ny<len(grid[0])) or grid[nx][ny]==9:
                continue
            if dx and dy and (grid[x+dx][y]==9 or grid[x][y+dy]==9):
                continue
            new_cost = cost + grid[nx][ny]*math.hypot(dx,dy)
            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                came_from[nb] = cur
                heapq.heappush(heap, (new_cost, nb))

    path=[]
    node=goal
    while node and node!=start:
        path.append(node)
        node=came_from.get(node)
    path.append(start)
    path.reverse()
    return path, visited