"""
Evolutionary algorithm for warehouse layout optimization using DEAP.
"""

from typing import List, Tuple, Dict, Callable, Optional
import random
import time
import numpy as np
from deap import base, creator, tools, algorithms

from warehouse import Warehouse, OrderGenerator
from mapd_solver import MAPDSimulator
from surrogate import SurrogateModel, extract_features


# Create DEAP types (only once)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def setup_toolbox(n_goods: int) -> base.Toolbox:
    """
    Setup DEAP toolbox for permutation-based optimization.

    Args:
        n_goods: Number of goods (= number of positions)

    Returns:
        Configured DEAP toolbox
    """
    toolbox = base.Toolbox()

    # Individual: random permutation of [0, 1, ..., n_goods-1]
    toolbox.register("indices", random.sample, range(n_goods), n_goods)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operators
    toolbox.register("mate", tools.cxOrdered)  # Order Crossover for permutations
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # Swap mutation
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


class HybridFitnessEvaluator:
    """
    Hybrid fitness evaluator using surrogate model with periodic real evaluation.
    """

    def __init__(self,
                 warehouse: Warehouse,
                 order_generator: OrderGenerator,
                 n_agents: int,
                 surrogate: SurrogateModel,
                 real_eval_interval: int = 10,
                 sim_steps: int = 500,
                 order_lambda: float = 0.5):
        """
        Initialize hybrid evaluator.

        Args:
            warehouse: Warehouse instance
            order_generator: Order generator
            n_agents: Number of agents
            surrogate: Surrogate model instance
            real_eval_interval: Run real simulation every N generations
            sim_steps: Number of steps for MAPD simulation
            order_lambda: Poisson lambda for order generation
        """
        self.warehouse = warehouse
        self.order_generator = order_generator
        self.n_agents = n_agents
        self.surrogate = surrogate
        self.real_eval_interval = real_eval_interval
        self.sim_steps = sim_steps
        self.order_lambda = order_lambda

        self.generation = 0
        self.real_evals = 0
        self.surrogate_evals = 0

    def _run_real_simulation(self, layout: List[int], seed: int = None) -> float:
        """Run actual MAPD simulation."""
        self.warehouse.set_layout(layout)
        sim = MAPDSimulator(
            self.warehouse,
            n_agents=self.n_agents,
            order_generator=self.order_generator,
            seed=seed
        )
        throughput = sim.run(n_steps=self.sim_steps, order_lambda=self.order_lambda)
        self.real_evals += 1
        return throughput

    def _predict_surrogate(self, layout: List[int]) -> float:
        """Predict fitness using surrogate model."""
        self.warehouse.set_layout(layout)
        features = extract_features(
            self.warehouse,
            self.order_generator.get_probabilities(),
            self.n_agents
        )
        self.surrogate_evals += 1
        return self.surrogate.predict(features)

    def evaluate(self, individual: List[int]) -> Tuple[float]:
        """
        Evaluate fitness of individual.

        Uses real simulation every `real_eval_interval` generations,
        surrogate model otherwise.

        Returns:
            Tuple with single fitness value
        """
        use_real = (self.generation % self.real_eval_interval == 0)

        if use_real or not self.surrogate.is_fitted:
            # Real evaluation
            fitness = self._run_real_simulation(individual, seed=self.generation * 1000 + self.real_evals)

            # Update surrogate
            self.warehouse.set_layout(individual)
            features = extract_features(
                self.warehouse,
                self.order_generator.get_probabilities(),
                self.n_agents
            )
            self.surrogate.update(features, fitness)
        else:
            # Surrogate prediction
            fitness = self._predict_surrogate(individual)

        return (fitness,)

    def next_generation(self):
        """Called after each generation."""
        self.generation += 1


def run_evolution(warehouse: Warehouse,
                  order_generator: OrderGenerator,
                  n_agents: int,
                  n_generations: int = 100,
                  pop_size: int = 50,
                  use_surrogate: bool = True,
                  surrogate_type: str = "gp",
                  real_eval_interval: int = 10,
                  sim_steps: int = 500,
                  order_lambda: float = 0.5,
                  cxpb: float = 0.7,
                  mutpb: float = 0.2,
                  seed: int = None,
                  verbose: bool = True,
                  callback: Callable = None) -> Dict:
    """
    Run evolutionary optimization.

    Args:
        warehouse: Warehouse instance
        order_generator: Order generator
        n_agents: Number of agents
        n_generations: Number of generations
        pop_size: Population size
        use_surrogate: Whether to use surrogate model
        surrogate_type: Type of surrogate ('linear', 'gp', 'rf', 'xgboost')
        real_eval_interval: Run real simulation every N generations (if using surrogate)
        sim_steps: Number of steps for MAPD simulation
        order_lambda: Poisson lambda for order generation
        cxpb: Crossover probability
        mutpb: Mutation probability
        seed: Random seed
        verbose: Print progress
        callback: Optional callback(generation, best_fitness, stats)

    Returns:
        Dictionary with results:
        - best_individual: Best layout found
        - best_fitness: Best throughput achieved
        - fitness_history: List of best fitness per generation
        - avg_fitness_history: List of average fitness per generation
        - real_evals: Number of real MAPD simulations
        - surrogate_evals: Number of surrogate predictions
        - wall_time: Total time in seconds
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start_time = time.time()

    # Setup DEAP
    toolbox = setup_toolbox(warehouse.n_inner)

    # Setup evaluator
    if use_surrogate:
        surrogate = SurrogateModel(surrogate_type)
        evaluator = HybridFitnessEvaluator(
            warehouse, order_generator, n_agents, surrogate,
            real_eval_interval=real_eval_interval,
            sim_steps=sim_steps,
            order_lambda=order_lambda
        )
        toolbox.register("evaluate", evaluator.evaluate)
    else:
        # Pure EA - always use real simulation
        def real_evaluate(individual):
            warehouse.set_layout(individual)
            sim = MAPDSimulator(warehouse, n_agents, order_generator, seed=random.randint(0, 100000))
            return (sim.run(n_steps=sim_steps, order_lambda=order_lambda),)

        toolbox.register("evaluate", real_evaluate)
        evaluator = None

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of Fame (best individuals)
    hof = tools.HallOfFame(1)

    # Create initial population
    pop = toolbox.population(n=pop_size)

    # History tracking
    fitness_history = []
    avg_fitness_history = []

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Update statistics
    record = stats.compile(pop)
    hof.update(pop)
    fitness_history.append(record['max'])
    avg_fitness_history.append(record['avg'])

    if verbose:
        print(f"Gen 0: best={record['max']:.4f}, avg={record['avg']:.4f}, std={record['std']:.4f}")

    if callback:
        callback(0, record['max'], record)

    # Evolution loop
    for gen in range(1, n_generations + 1):
        if evaluator:
            evaluator.next_generation()

        # Select offspring
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Update statistics
        record = stats.compile(pop)
        hof.update(pop)
        fitness_history.append(record['max'])
        avg_fitness_history.append(record['avg'])

        if verbose and (gen % 10 == 0 or gen == n_generations):
            real_info = ""
            if evaluator:
                real_info = f", real_evals={evaluator.real_evals}"
            print(f"Gen {gen}: best={record['max']:.4f}, avg={record['avg']:.4f}{real_info}")

        if callback:
            callback(gen, record['max'], record)

    wall_time = time.time() - start_time

    # Prepare results
    results = {
        'best_individual': list(hof[0]),
        'best_fitness': hof[0].fitness.values[0],
        'fitness_history': fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'wall_time': wall_time
    }

    if evaluator:
        results['real_evals'] = evaluator.real_evals
        results['surrogate_evals'] = evaluator.surrogate_evals
    else:
        results['real_evals'] = n_generations * pop_size + pop_size
        results['surrogate_evals'] = 0

    return results


def collect_initial_data(warehouse: Warehouse,
                         order_generator: OrderGenerator,
                         n_agents: int = 1,
                         n_samples: int = 50,
                         sim_steps: int = 200,
                         order_lambda: float = 0.5,
                         seed: int = None,
                         verbose: bool = True) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
    """
    Collect initial training data for surrogate model.

    Args:
        warehouse: Warehouse instance
        order_generator: Order generator
        n_agents: Number of agents
        n_samples: Number of random layouts to evaluate
        sim_steps: Simulation steps
        order_lambda: Order rate
        seed: Random seed
        verbose: Print progress

    Returns:
        (features_list, fitness_list, layouts_list)
    """
    if seed is not None:
        np.random.seed(seed)

    features_list = []
    fitness_list = []
    layouts_list = []

    for i in range(n_samples):
        # Random layout
        layout = list(range(warehouse.n_inner))
        np.random.shuffle(layout)
        warehouse.set_layout(layout)

        # Extract features
        features = extract_features(warehouse, order_generator.get_probabilities(), n_agents)
        features_list.append(features)

        # Run simulation
        sim = MAPDSimulator(warehouse, n_agents, order_generator, seed=i)
        throughput = sim.run(n_steps=sim_steps, order_lambda=order_lambda)
        fitness_list.append(throughput)

        layouts_list.append(layout)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{n_samples} samples")

    return features_list, fitness_list, layouts_list


if __name__ == "__main__":
    print("Testing Evolution Module...")

    # Create warehouse
    wh = Warehouse(7, 7)
    og = OrderGenerator(wh.n_inner, seed=42)

    # Quick test with small parameters
    print("\nRunning short evolution (10 generations, 10 individuals)...")
    results = run_evolution(
        wh, og, n_agents=3,
        n_generations=10,
        pop_size=10,
        use_surrogate=True,
        surrogate_type="rf",
        real_eval_interval=5,
        sim_steps=100,
        order_lambda=0.3,
        seed=42,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Best fitness: {results['best_fitness']:.4f}")
    print(f"  Real evaluations: {results['real_evals']}")
    print(f"  Surrogate evaluations: {results['surrogate_evals']}")
    print(f"  Wall time: {results['wall_time']:.2f}s")
