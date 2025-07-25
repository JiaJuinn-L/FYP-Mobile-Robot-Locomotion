import numpy as np
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils import configurations, generate_weighted_grid
from algorithms.astar import astar_wrapped
from algorithms.dijstra import dijkstra_wrapped
from algorithms.dstar_lite import dstar_wrapped
from collections import defaultdict
import random

# Filter configurations to only include grids up to specified size
configurations = [(size, 0.1) for size, _ in configurations if size <= 500]  # Using 10% static obstacle rate

def generate_dynamic_grid(size, static_prob=0.1, dynamic_prob=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate base grid with static obstacles
    grid = generate_weighted_grid(size, static_prob, seed=seed if seed else None)
    
    # Create mask for potential dynamic obstacle positions
    # Avoid placing dynamic obstacles on static obstacles or start/goal positions
    obstacle_mask = np.zeros((size, size), dtype=bool)
    obstacle_mask[grid == np.inf] = True  # Mark static obstacles
    obstacle_mask[0, 0] = True  # Mark start
    obstacle_mask[size-1, size-1] = True  # Mark goal
    
    # Calculate number of dynamic obstacles
    total_dynamic = int(size * size * dynamic_prob)
    
    # Get available positions for dynamic obstacles
    available_positions = [(i, j) for i in range(size) for j in range(size) if not obstacle_mask[i, j]]
    
    # Randomly select positions for dynamic obstacles
    dynamic_positions = random.sample(available_positions, min(total_dynamic, len(available_positions)))
    
    return grid, dynamic_positions

def update_dynamic_obstacles(grid, obstacle_positions, step, size):
    """
    Update obstacles based on the current step.
    All obstacles are dynamic and can appear/disappear.
    Returns the new grid state and whether obstacles changed position.
    """
    # Create a copy of the original grid
    new_grid = grid.copy()
    
    # Deterministic but pseudo-random obstacle movement based on step
    random.seed(step)
    
    # Each step, randomly toggle each obstacle (appear/disappear)
    changes_made = False
    for pos in obstacle_positions:
        i, j = pos  # Unpack the tuple into row and column indices
        # Use the step number to create a pseudo-random but deterministic pattern
        if random.random() < 0.2:  # 20% chance to toggle each obstacle (reduced for more stability)
            if new_grid[i][j] == np.inf:
                new_grid[i][j] = 1  # Remove obstacle
            else:
                new_grid[i][j] = np.inf  # Add obstacle
            changes_made = True
    
    return new_grid, changes_made

def run_dynamic_simulation(num_trials=100, steps_per_trial=200):
    """
    Run simulation with dynamic obstacles.
    num_trials: number of different initial configurations to test
    steps_per_trial: number of steps with dynamic changes per trial
    """
    results = {
        'A*': defaultdict(lambda: defaultdict(list)),
        'Dijkstra': defaultdict(lambda: defaultdict(list)),
        'D* Lite': defaultdict(lambda: defaultdict(list))
    }
    
    algorithms = {
        'A*': astar_wrapped,
        'Dijkstra': dijkstra_wrapped,
        'D* Lite': dstar_wrapped
    }

    for size, static_prob in configurations:
        print(f"\nRunning dynamic simulations for grid size {size}x{size}")
        
        for trial in range(num_trials):
            if trial % 10 == 0:
                print(f"Trial {trial}/{num_trials}")
            
            # Generate initial grid with both static and dynamic obstacles
            initial_grid, dynamic_positions = generate_dynamic_grid(
                size, 
                static_prob=static_prob,  # 10% static obstacles
                dynamic_prob=0.1,        # 10% dynamic obstacles
                seed=trial
            )
            
            # Run each algorithm
            for algo_name, algo_func in algorithms.items():
                start = (0, 0)
                goal = (size-1, size-1)
                current_grid = initial_grid.copy()
                
                # Track metrics for this trial
                total_cost = 0
                total_replans = 0
                total_visited = set()
                path_found = False
                execution_times = []
                
                # Start timing
                trial_start_time = time.time()
                
                current_pos = start
                for step in range(steps_per_trial):
                    step_start_time = time.time()
                    
                    # Update dynamic obstacles while preserving static ones
                    new_grid, obstacles_changed = update_dynamic_obstacles(
                        current_grid, 
                        dynamic_positions, 
                        step, 
                        size
                    )
                    
                    # Only replan if obstacles changed or this is the first step or every 5 steps
                    if obstacles_changed or step == 0 or step % 5 == 0:
                        path, visited, step_cost, replans = algo_func(
                            new_grid,
                            current_pos,  # Start from current position
                            goal
                        )
                        
                        execution_times.append(time.time() - step_start_time)
                        total_replans += replans
                        total_visited.update(visited)
                        
                        if path:
                            # Move up to 3 steps along the path if possible
                            steps_to_move = min(3, len(path) - 1)
                            if steps_to_move > 0:
                                current_pos = path[steps_to_move]  # Move multiple positions
                                i, j = current_pos
                                # Add cost for all steps taken
                                for step_idx in range(1, steps_to_move + 1):
                                    i_step, j_step = path[step_idx]
                                    total_cost += new_grid[i_step][j_step]
                            path_found = True
                    
                    current_grid = new_grid
                    
                    # Check if we reached the goal
                    if current_pos == goal:
                        break
                
                total_execution_time = time.time() - trial_start_time
                
                # Store results only if path was found and goal was reached
                if path_found and current_pos == goal:
                    results[algo_name][size]['path_length'].append(total_cost)
                    results[algo_name][size]['visited_nodes'].append(len(total_visited))
                    results[algo_name][size]['total_cost'].append(total_cost)
                    results[algo_name][size]['success_rate'].append(1)
                    results[algo_name][size]['timeout_rate'].append(0)
                    results[algo_name][size]['execution_time'].append(total_execution_time)
                    results[algo_name][size]['replan_count'].append(total_replans)
                else:
                    # For unsuccessful attempts, only update success rate
                    results[algo_name][size]['success_rate'].append(0)
                    # Initialize other metrics lists if they don't exist yet
                    for metric in ['path_length', 'visited_nodes', 'total_cost', 'execution_time', 'replan_count']:
                        if metric not in results[algo_name][size]:
                            results[algo_name][size][metric] = []
                
    return results

def print_detailed_summary(results):
    """
    Print a detailed summary of the simulation results.
    """
    print("\nDetailed Dynamic Simulation Summary:")
    print("=" * 100)
    
    metrics = ['success_rate', 'timeout_rate', 'path_length', 'visited_nodes', 
               'execution_time', 'replan_count', 'total_cost']
    
    # Print header comparing all algorithms side by side
    for size in sorted(results['A*'].keys()):
        print(f"\nGrid Size: {size}x{size}")
        print("-" * 100)
        print(f"{'Metric':<25} {'A*':<35} {'Dijkstra':<35} {'D* Lite':<35}")
        print("-" * 100)
        
        for metric in metrics:
            metric_name = metric.replace('_', ' ').title()
            values = []
            
            for algo_name in ['A*', 'Dijkstra', 'D* Lite']:
                data = results[algo_name][size][metric]
                
                # Skip empty data lists (unsuccessful trials)
                if not data:
                    values.append("No successful paths")
                    continue

                mean_val = np.mean(data)
                min_val = np.min(data)
                max_val = np.max(data)

                if 'rate' in metric:
                    values.append(f"{mean_val*100:.1f}% ({min_val*100:.1f}%-{max_val*100:.1f}%)")
                elif metric == 'execution_time':
                    values.append(f"{mean_val:.3f}s ({min_val:.3f}s-{max_val:.3f}s)")
                else:
                    values.append(f"{mean_val:.1f} ({min_val:.1f}-{max_val:.1f})")
            
            print(f"{metric_name:<25} {values[0]:<35} {values[1]:<35} {values[2]:<35}")
        print("=" * 100)

if __name__ == "__main__":
    print("Starting dynamic simulation with 100 trials...")
    results = run_dynamic_simulation(num_trials=100, steps_per_trial=50)
    print_detailed_summary(results)
