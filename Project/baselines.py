"""
Baseline methods for warehouse layout optimization.
"""

from typing import List, Dict, Tuple
import numpy as np

from warehouse import Warehouse, OrderGenerator
from mapd_solver import MAPDSimulator


def random_baseline(warehouse: Warehouse,
                    order_generator: OrderGenerator,
                    n_agents: int,
                    n_samples: int = 100,
                    sim_steps: int = 500,
                    order_lambda: float = 0.5,
                    seed: int = None,
                    verbose: bool = True,
                    planner_type: str = "sat") -> Dict:
    """
    Random baseline - evaluate random layouts.

    Args:
        warehouse: Warehouse instance
        order_generator: Order generator
        n_agents: Number of agents
        n_samples: Number of random layouts to try
        sim_steps: Simulation steps
        order_lambda: Order rate
        seed: Random seed
        verbose: Print progress
        planner_type: "sat" or "astar"

    Returns:
        Dictionary with:
        - best_layout: Best layout found
        - best_fitness: Best throughput
        - all_fitness: List of all throughputs
        - mean_fitness: Average throughput
        - std_fitness: Standard deviation
    """
    if seed is not None:
        np.random.seed(seed)

    all_fitness = []
    all_layouts = []
    best_fitness = -float('inf')
    best_layout = None

    for i in range(n_samples):
        # Random layout
        layout = list(range(warehouse.n_inner))
        np.random.shuffle(layout)
        warehouse.set_layout(layout)

        # Run simulation
        sim = MAPDSimulator(warehouse, n_agents, order_generator, seed=i, planner_type=planner_type)
        throughput = sim.run(n_steps=sim_steps, order_lambda=order_lambda)

        all_fitness.append(throughput)
        all_layouts.append(layout)

        if throughput > best_fitness:
            best_fitness = throughput
            best_layout = layout.copy()

        if verbose and (i + 1) % 20 == 0:
            print(f"  Random: {i + 1}/{n_samples} samples, best so far: {best_fitness:.4f}")

    return {
        'best_layout': best_layout,
        'best_fitness': best_fitness,
        'all_fitness': all_fitness,
        'mean_fitness': float(np.mean(all_fitness)),
        'std_fitness': float(np.std(all_fitness))
    }


def greedy_baseline(warehouse: Warehouse,
                    order_generator: OrderGenerator,
                    n_agents: int,
                    sim_steps: int = 500,
                    order_lambda: float = 0.5,
                    seed: int = None,
                    planner_type: str = "sat") -> Dict:
    """
    Greedy baseline - place popular goods closest to edges.

    Strategy: Sort goods by popularity (order probability), then assign
    positions from closest to edge to furthest.

    Args:
        warehouse: Warehouse instance
        order_generator: Order generator
        n_agents: Number of agents
        sim_steps: Simulation steps
        order_lambda: Order rate
        seed: Random seed
        planner_type: "sat" or "astar"

    Returns:
        Dictionary with:
        - layout: Greedy layout
        - fitness: Throughput
    """
    # Get order probabilities
    probs = order_generator.get_probabilities()

    # Sort goods by popularity (most popular first)
    goods_by_popularity = np.argsort(probs)[::-1]

    # Get inner positions sorted by distance to edge (closest first)
    inner_positions = warehouse.get_inner_positions()
    positions_with_dist = [
        (pos, warehouse.distance_to_nearest_edge(pos))
        for pos in inner_positions
    ]
    positions_sorted = sorted(positions_with_dist, key=lambda x: x[1])

    # Create layout: position i gets goods j
    # layout[position_index] = goods_id
    layout = [0] * warehouse.n_inner

    for rank, goods_id in enumerate(goods_by_popularity):
        # Get the rank-th closest position
        pos = positions_sorted[rank][0]
        pos_idx = inner_positions.index(pos)
        layout[pos_idx] = goods_id

    # Evaluate
    warehouse.set_layout(layout)
    sim = MAPDSimulator(warehouse, n_agents, order_generator, seed=seed, planner_type=planner_type)
    throughput = sim.run(n_steps=sim_steps, order_lambda=order_lambda)

    return {
        'layout': layout,
        'fitness': throughput
    }


def inverse_greedy_baseline(warehouse: Warehouse,
                            order_generator: OrderGenerator,
                            n_agents: int,
                            sim_steps: int = 500,
                            order_lambda: float = 0.5,
                            seed: int = None,
                            planner_type: str = "sat") -> Dict:
    """
    Inverse greedy baseline - place popular goods FURTHEST from edges.

    This is intentionally bad, useful for comparison.

    Args:
        warehouse: Warehouse instance
        order_generator: Order generator
        n_agents: Number of agents
        sim_steps: Simulation steps
        order_lambda: Order rate
        seed: Random seed
        planner_type: "sat" or "astar"

    Returns:
        Dictionary with:
        - layout: Inverse greedy layout
        - fitness: Throughput
    """
    # Get order probabilities
    probs = order_generator.get_probabilities()

    # Sort goods by popularity (most popular first)
    goods_by_popularity = np.argsort(probs)[::-1]

    # Get inner positions sorted by distance to edge (furthest first!)
    inner_positions = warehouse.get_inner_positions()
    positions_with_dist = [
        (pos, warehouse.distance_to_nearest_edge(pos))
        for pos in inner_positions
    ]
    positions_sorted = sorted(positions_with_dist, key=lambda x: -x[1])  # Descending

    # Create layout
    layout = [0] * warehouse.n_inner

    for rank, goods_id in enumerate(goods_by_popularity):
        pos = positions_sorted[rank][0]
        pos_idx = inner_positions.index(pos)
        layout[pos_idx] = goods_id

    # Evaluate
    warehouse.set_layout(layout)
    sim = MAPDSimulator(warehouse, n_agents, order_generator, seed=seed, planner_type=planner_type)
    throughput = sim.run(n_steps=sim_steps, order_lambda=order_lambda)

    return {
        'layout': layout,
        'fitness': throughput
    }


def run_all_baselines(warehouse: Warehouse,
                      order_generator: OrderGenerator,
                      n_agents: int,
                      random_samples: int = 100,
                      sim_steps: int = 500,
                      order_lambda: float = 0.5,
                      seed: int = None,
                      verbose: bool = True,
                      planner_type: str = "sat") -> Dict[str, Dict]:
    """
    Run all baseline methods.

    Args:
        planner_type: "sat" or "astar"

    Returns:
        Dictionary with results for each baseline
    """
    results = {}

    if verbose:
        print(f"Running baselines (planner: {planner_type})...")

    # Random
    if verbose:
        print("\n1. Random baseline:")
    results['random'] = random_baseline(
        warehouse, order_generator, n_agents,
        n_samples=random_samples,
        sim_steps=sim_steps,
        order_lambda=order_lambda,
        seed=seed,
        verbose=verbose,
        planner_type=planner_type
    )

    # Greedy
    if verbose:
        print("\n2. Greedy baseline:")
    results['greedy'] = greedy_baseline(
        warehouse, order_generator, n_agents,
        sim_steps=sim_steps,
        order_lambda=order_lambda,
        seed=seed,
        planner_type=planner_type
    )
    if verbose:
        print(f"  Greedy fitness: {results['greedy']['fitness']:.4f}")

    # Inverse greedy
    if verbose:
        print("\n3. Inverse greedy baseline:")
    results['inverse_greedy'] = inverse_greedy_baseline(
        warehouse, order_generator, n_agents,
        sim_steps=sim_steps,
        order_lambda=order_lambda,
        seed=seed,
        planner_type=planner_type
    )
    if verbose:
        print(f"  Inverse greedy fitness: {results['inverse_greedy']['fitness']:.4f}")

    return results


if __name__ == "__main__":
    print("Testing Baselines...")

    # Create warehouse
    wh = Warehouse(7, 7)
    og = OrderGenerator(wh.n_inner, seed=42)

    # Run baselines with small parameters for quick test
    results = run_all_baselines(
        wh, og, n_agents=3,
        random_samples=20,
        sim_steps=200,
        order_lambda=0.3,
        seed=42,
        verbose=True
    )

    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"Random:         best={results['random']['best_fitness']:.4f}, "
          f"mean={results['random']['mean_fitness']:.4f} ± {results['random']['std_fitness']:.4f}")
    print(f"Greedy:         {results['greedy']['fitness']:.4f}")
    print(f"Inverse Greedy: {results['inverse_greedy']['fitness']:.4f}")
