import numpy as np
import time
import heapq
from statistics import mean

def random_grid(size, obstacle_prob):
    grid = np.random.choice([1, 9], size=(size, size), p=[1 - obstacle_prob, obstacle_prob])
    grid[0, 0] = grid[-1, -1] = 1  # Ensure start and goal are accessible
    return grid.tolist()

def dynamic_obstacles(grid, change_prob=0.05):
    size = len(grid)
    for i in range(size):
        for j in range(size):
            if (i, j) in [(0, 0), (size - 1, size - 1)]:
                continue
            if np.random.rand() < change_prob:
                grid[i][j] = 9 if grid[i][j] != 9 else 1
    return grid

def run_dynamic_simulation(algorithms, configurations, trials=100):
    results = []

    for grid_size, obstacle_prob in configurations:
        for algorithm in algorithms:
            successes = 0
            stats = {'times_ms': [], 'lengths': [], 'visited': [], 'costs': [], 'replans': []}

            for _ in range(trials):
                grid = random_grid(grid_size, obstacle_prob)
                start, goal = (0, 0), (grid_size - 1, grid_size - 1)

                # Simulate dynamic changes
                grid = dynamic_obstacles(grid)

                start_time = time.perf_counter()
                res = algorithm(grid, start, goal)
                end_time = time.perf_counter()

                exec_time_ms = (end_time - start_time) * 1000

                # Support both 3-value and 4-value returns
                if len(res) == 4:
                    path, visited, cost, replan_count = res
                else:
                    path, visited, cost = res
                    replan_count = 0

                if path:
                    successes += 1
                    stats['times_ms'].append(exec_time_ms)
                    stats['lengths'].append(len(path))
                    stats['visited'].append(len(visited) if isinstance(visited, set) else visited)
                    stats['costs'].append(cost)
                    stats['replans'].append(replan_count)

            result = {
                'grid_size': f"{grid_size}×{grid_size}",
                'algorithm': algorithm.__name__,
                'trials': trials,
                'success_rate': round(successes / trials * 100, 2),
                'max_time_ms': round(max(stats['times_ms']), 2) if stats['times_ms'] else None,
                'min_time_ms': round(min(stats['times_ms']), 2) if stats['times_ms'] else None,
                'avg_time_ms': round(mean(stats['times_ms']), 2) if stats['times_ms'] else None,
                'avg_path_length': round(mean(stats['lengths']), 2) if stats['lengths'] else None,
                'max_path_length': max(stats['lengths']) if stats['lengths'] else None,
                'min_path_length': min(stats['lengths']) if stats['lengths'] else None,
                'avg_nodes_visited': round(mean(stats['visited']), 2) if stats['visited'] else None,
                'max_nodes_visited': max(stats['visited']) if stats['visited'] else None,
                'min_nodes_visited': min(stats['visited']) if stats['visited'] else None,
                'avg_path_cost': round(mean(stats['costs']), 2) if stats['costs'] else None,
                'max_path_cost': max(stats['costs']) if stats['costs'] else None,
                'min_path_cost': min(stats['costs']) if stats['costs'] else None,
                'avg_replans': round(mean(stats['replans']), 2) if stats['replans'] else None,
                'max_replans': max(stats['replans']) if stats['replans'] else None,
                'min_replans': min(stats['replans']) if stats['replans'] else None,
            }

            results.append(result)

    return results

# Usage example:
algorithms = [astar_weighted, dijkstra_weighted]
configurations = [(10, 0.2), (30, 0.2), (50, 0.2), (70, 0.2), (100, 0.2), (200, 0.2),(500,0.2)]

simulation_results = run_dynamic_simulation(algorithms, configurations, trials=100)

# Display results
for res in simulation_results:
    print(res)
