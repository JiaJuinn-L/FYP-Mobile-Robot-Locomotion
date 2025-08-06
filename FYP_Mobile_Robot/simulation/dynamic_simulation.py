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

def generate_dynamic_grid(size, static_prob, dynamic_prob=0.2, max_cost=5, seed=None):
    """
    Generate a grid with both static and dynamic obstacles.
    Args:
        size: Grid size (N x N)
        static_prob: Probability of static obstacles (0-1)
        dynamic_prob: Probability of dynamic obstacles (0-1)
        max_cost: Maximum cost for non-obstacle cells (default: 5)
        seed: Random seed for reproducibility
    Returns:
        grid: 2D list with costs (1-max_cost) and obstacles (9)
        dynamic_positions: List of positions that can have dynamic obstacles
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate base grid with static obstacles
    grid = []
    for i in range(size):
        row = []
        for j in range(size):
            if (i, j) in [(0, 0), (size-1, size-1)]:  # Start and goal positions
                row.append(1)  # Minimum cost for start/goal
            elif random.random() < static_prob:
                row.append(9)  # Static obstacle
            else:
                row.append(random.randint(1, max_cost))  # Random cost
        grid.append(row)
    
    # Create mask for potential dynamic obstacle positions
    obstacle_mask = [[True if grid[i][j] == 9 or (i,j) in [(0,0), (size-1,size-1)] else False 
                     for j in range(size)] for i in range(size)]
    
    # Get available positions for dynamic obstacles
    available_positions = [(i, j) for i in range(size) for j in range(size) 
                         if not obstacle_mask[i][j]]
    
    # Calculate and select dynamic obstacle positions
    total_dynamic = int(size * size * dynamic_prob)
    dynamic_positions = random.sample(available_positions, min(total_dynamic, len(available_positions)))
    
    return grid, dynamic_positions

def update_dynamic_obstacles(grid, dynamic_positions, step, size):
    """
    Update dynamic obstacles based on the current step.
    Returns the new grid state and whether obstacles changed.
    """
    new_grid = [row[:] for row in grid]
    random.seed(step)  # Make updates deterministic based on step
    
    changes_made = False
    for pos in dynamic_positions:
        i, j = pos
        if random.random() < 0.2:  # 20% chance to toggle each obstacle
            if new_grid[i][j] == 9:
                new_grid[i][j] = 1  # Remove obstacle
            else:
                new_grid[i][j] = 9  # Add obstacle
            changes_made = True
    
    return new_grid, changes_made

def run_dynamic_simulation(algorithms, configurations, trials=10, max_cost=5, steps_per_trial=50, seed=None):
  
    results = []
    total_configs = len(configurations) * len(algorithms)
    current_config = 0

    logging.info("Starting dynamic pathfinding simulation")
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
                
                # Generate initial grid with dynamic obstacles
                grid, dynamic_positions = generate_dynamic_grid(grid_size, obstacle_prob, 0.2, max_cost, trial_seed)
                start, goal = (0, 0), (grid_size - 1, grid_size - 1)
                current_pos = start
                total_cost = 0
                total_visited = set()
                path_found = False

                # Start timing for entire pathfinding process
                start_time = time.perf_counter()
                
                # Simulate dynamic environment
                for step in range(steps_per_trial):
                    # Update dynamic obstacles
                    new_grid, obstacles_changed = update_dynamic_obstacles(grid, dynamic_positions, step, grid_size)
                    
                    # Only replan if obstacles changed or this is the first step
                    if obstacles_changed or step == 0:
                        res = algorithm(new_grid, current_pos, goal)
                        
                        # Support both 3-value and 4-value returns
                        if len(res) == 4:
                            path, visited, step_cost, replans = res
                        else:
                            path, visited, step_cost = res
                            replans = 0
                        
                        if path:
                            # Move up to 3 steps along the path if possible
                            steps_to_move = min(3, len(path) - 1)
                            if steps_to_move > 0:
                                current_pos = path[steps_to_move]
                                # Add cost for steps taken
                                for step_idx in range(1, steps_to_move + 1):
                                    i, j = path[step_idx]
                                    total_cost += new_grid[i][j]
                            path_found = True
                            total_visited.update(visited)
                    
                    grid = new_grid
                    
                    # Check if we reached the goal
                    if current_pos == goal:
                        break
                
                end_time = time.perf_counter()
                exec_time_ms = (end_time - start_time) * 1000
                
                # Only count as success if we reached the goal
                if not (path_found and current_pos == goal):
                    continue

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

simulation_results = run_dynamic_simulation(
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
