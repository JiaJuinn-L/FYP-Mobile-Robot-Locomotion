import numpy as np
import time
import heapq
from statistics import mean
import random
import logging
from datetime import datetime
from astar_pure import astar
from dijkstra_pure import dijkstra
from dstar_lite_pure import dstar_lite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'simulation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'),
        logging.StreamHandler()
    ]
)

def generate_static_grid(size, obstacle_prob, max_cost=5, seed=None):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = []
    for i in range(size):
        row = []
        for j in range(size):
            if (i, j) in [(0, 0), (size-1, size-1)]:  # Start and goal positions
                row.append(1)  # Minimum cost for start/goal
            elif random.random() < obstacle_prob:
                row.append(9)  # Obstacle
            else:
                row.append(random.randint(1, max_cost))  # Random cost
        grid.append(row)
    return grid

def run_static_simulation(algorithms, configurations, trials=1, max_cost=5, seed=None):

    results = []
    total_configs = len(configurations) * len(algorithms)
    current_config = 0

    logging.info("Starting static pathfinding simulation")
    logging.info(f"Total configurations to test: {total_configs}")
    logging.info(f"Trials per configuration: {trials}")
    logging.info(f"Maximum cell cost: {max_cost}")
    logging.info(f"Random seed: {seed}")
    logging.info("=" * 50)

    for grid_size, obstacle_prob in configurations:
        logging.info(f"\nTesting grid size: {grid_size}×{grid_size} with {obstacle_prob*100}% obstacles")
        # Use different seeds for different configurations but keep them consistent across algorithms
        trial_seeds = [seed + i if seed is not None else None for i in range(trials)]
        
        for algorithm in algorithms:
            current_config += 1
            logging.info(f"\nConfiguration {current_config}/{total_configs}:")
            logging.info(f"Running {algorithm.__name__}")
            successes = 0
            stats = {'times_ms': [], 'lengths': [], 'visited': [], 'costs': [], 'replans': []}

            for trial_num, trial_seed in enumerate(trial_seeds, 1):
                if trial_num % 10 == 0:  # Log progress every 10 trials
                    logging.info(f"Running trial {trial_num}/{trials}")
                
                # Generate static grid with consistent seed
                grid = generate_static_grid(grid_size, obstacle_prob, max_cost, trial_seed)
                start, goal = (0, 0), (grid_size - 1, grid_size - 1)

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

            # Log summary for this configuration
            success_rate = round(successes / trials * 100, 2)
            avg_time = round(mean(stats['times_ms']), 2) if stats['times_ms'] else None
            avg_length = round(mean(stats['lengths']), 2) if stats['lengths'] else None
            avg_cost = round(mean(stats['costs']), 2) if stats['costs'] else None
            
            logging.info(f"Results for {algorithm.__name__} on {grid_size}×{grid_size} grid:")
            logging.info(f"Success rate: {success_rate}%")
            logging.info(f"Average time: {avg_time}ms")
            logging.info(f"Average path length: {avg_length}")
            logging.info(f"Average path cost: {avg_cost}")
            logging.info("-" * 40)

            result = {
                'grid_size': f"{grid_size}×{grid_size}",
                'algorithm': algorithm.__name__,
                'trials': trials,
                'success_rate': success_rate,
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

# Configuration
algorithms = [astar, dijkstra, dstar_lite]
configurations = [
    # (10, 0.2),    # Very small grid
    # (30, 0.2),    # Small grid
    # (50, 0.2),    # Medium grid
    # (70, 0.2),    # Medium-large grid
    (100, 0.2),   # Large grid
    (200, 0.2),   # Very large grid
    # (500, 0.2)    # Huge grid
]

# Run simulation with reproducible results
logging.info("\nStarting Simulation with the following configuration:")
logging.info(f"Algorithms: {[alg.__name__ for alg in algorithms]}")
logging.info(f"Grid configurations: {configurations}")
logging.info("=" * 50)

simulation_results = run_static_simulation(
    algorithms=algorithms,
    configurations=configurations,
    trials=100,  # Number of trials per configuration
    max_cost=5,  # Maximum cost for non-obstacle cells
    seed=random.randint(1,1000)      
)

# Display and log final results
logging.info("\nFINAL SIMULATION RESULTS")
logging.info("=" * 50)

print("\nStatic Pathfinding Simulation Results:")
print("======================================")
for res in simulation_results:
    result_str = f"""
Grid Size: {res['grid_size']}
Algorithm: {res['algorithm']}
Success Rate: {res['success_rate']}%
Average Time: {res['avg_time_ms']:.2f}ms
Average Path Length: {res['avg_path_length']:.2f}
Average Path Cost: {res['avg_path_cost']:.2f}
Average Nodes Visited: {res['avg_nodes_visited']:.2f}
Min/Max Time: {res['min_time_ms']:.2f}ms / {res['max_time_ms']:.2f}ms
Min/Max Path Length: {res['min_path_length']} / {res['max_path_length']}
Min/Max Path Cost: {res['min_path_cost']:.2f} / {res['max_path_cost']:.2f}
----------------------------------------"""
    print(result_str)
    logging.info(result_str)
