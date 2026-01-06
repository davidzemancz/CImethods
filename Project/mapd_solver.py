"""
MAPD (Multi-Agent Pickup and Delivery) solver using SAT-based or A* path planning.
"""

from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
import heapq
import numpy as np
from pysat.solvers import Solver
from pysat.formula import CNF

from warehouse import Warehouse, OrderGenerator


class AgentState(Enum):
    """Agent states in MAPD lifecycle."""
    IDLE = "idle"           # Waiting at delivery point for task
    TO_PICKUP = "to_pickup" # Moving to pickup goods
    TO_DELIVERY = "to_delivery"  # Carrying goods to delivery point


class Agent:
    """
    Represents a robot agent in the warehouse.
    """

    def __init__(self, agent_id: int, start_pos: Tuple[int, int]):
        """
        Initialize agent.

        Args:
            agent_id: Unique agent identifier
            start_pos: Starting position (should be a delivery point)
        """
        self.id = agent_id
        self.position = start_pos
        self.state = AgentState.IDLE
        self.target = None          # Target position
        self.path: List[Tuple[int, int]] = []  # Planned path (including current position)
        self.path_index = 0         # Current position in path
        self.carrying = None        # Goods ID being carried

    def is_idle(self) -> bool:
        """Check if agent is available for new task."""
        return self.state == AgentState.IDLE

    def assign_pickup(self, goods_id: int, goods_pos: Tuple[int, int], path: List[Tuple[int, int]]):
        """Assign pickup task to agent."""
        self.carrying = goods_id
        self.target = goods_pos
        self.path = path
        self.path_index = 0
        self.state = AgentState.TO_PICKUP

    def assign_delivery(self, delivery_pos: Tuple[int, int], path: List[Tuple[int, int]]):
        """Assign delivery task after pickup."""
        self.target = delivery_pos
        self.path = path
        self.path_index = 0
        self.state = AgentState.TO_DELIVERY

    def step(self) -> bool:
        """
        Execute one timestep - move along path.

        Returns:
            True if agent reached its target this step
        """
        if not self.path or self.path_index >= len(self.path) - 1:
            return False

        self.path_index += 1
        self.position = self.path[self.path_index]

        # Check if reached target
        if self.position == self.target:
            return True
        return False

    def complete_pickup(self):
        """Called when agent arrives at pickup location."""
        # Goods already assigned, just clear target
        self.target = None
        self.path = []
        self.path_index = 0

    def complete_delivery(self):
        """Called when agent completes delivery."""
        self.carrying = None
        self.target = None
        self.path = []
        self.path_index = 0
        self.state = AgentState.IDLE

    def get_future_positions(self, max_time: int) -> List[Tuple[int, int]]:
        """
        Get agent's planned positions for future timesteps.

        Args:
            max_time: Maximum number of future timesteps

        Returns:
            List of positions (pads with final position if path ends early)
        """
        if not self.path:
            return [self.position] * max_time

        positions = []
        for t in range(max_time):
            idx = self.path_index + t
            if idx < len(self.path):
                positions.append(self.path[idx])
            else:
                positions.append(self.path[-1])
        return positions

    def __repr__(self) -> str:
        return f"Agent({self.id}, {self.state.value}, pos={self.position})"


class SATPathPlanner:
    """
    SAT-based path planner for single agent with frozen obstacle paths.
    """

    def __init__(self, warehouse: Warehouse, timeout: float = 5.0):
        """
        Initialize planner.

        Args:
            warehouse: Warehouse instance
            timeout: Maximum time for SAT solver (seconds)
        """
        self.warehouse = warehouse
        self.timeout = timeout

        # Precompute all positions and create position-to-index mapping
        self.all_positions = self._get_all_positions()
        self.pos_to_idx = {pos: i for i, pos in enumerate(self.all_positions)}

    def _get_all_positions(self) -> List[Tuple[int, int]]:
        """Get list of all valid positions in warehouse."""
        positions = []
        for y in range(self.warehouse.height):
            for x in range(self.warehouse.width):
                positions.append((x, y))
        return positions

    def _var(self, t: int, pos_idx: int, n_positions: int) -> int:
        """
        Get SAT variable number for (timestep, position).

        Variables are 1-indexed for SAT solver.
        """
        return t * n_positions + pos_idx + 1

    def plan_path(self,
                  start: Tuple[int, int],
                  goal: Tuple[int, int],
                  frozen_paths: Dict[int, List[Tuple[int, int]]],
                  makespan_limit: int = None) -> Optional[List[Tuple[int, int]]]:
        """
        Plan collision-free path from start to goal.

        Args:
            start: Starting position
            goal: Goal position
            frozen_paths: Dictionary mapping agent_id -> list of positions over time
            makespan_limit: Maximum path length (None = auto-calculate)

        Returns:
            List of positions forming the path, or None if no solution found
        """
        if start == goal:
            return [start]

        # Calculate makespan limit
        min_dist = self.warehouse.manhattan_distance(start, goal)
        if makespan_limit is None:
            makespan_limit = min_dist * 3 + 5  # Allow some slack

        # Try increasing makespan until solution found or limit reached
        for makespan in range(min_dist, makespan_limit + 1):
            path = self._solve_sat(start, goal, frozen_paths, makespan)
            if path is not None:
                return path

        return None

    def _solve_sat(self,
                   start: Tuple[int, int],
                   goal: Tuple[int, int],
                   frozen_paths: Dict[int, List[Tuple[int, int]]],
                   makespan: int) -> Optional[List[Tuple[int, int]]]:
        """
        Solve SAT for specific makespan.

        Args:
            start: Starting position
            goal: Goal position
            frozen_paths: Frozen agent paths
            makespan: Exact number of timesteps

        Returns:
            Path if solution found, None otherwise
        """
        n_positions = len(self.all_positions)
        cnf = CNF()

        # Helper to get variable
        def var(t, pos_idx):
            return self._var(t, pos_idx, n_positions)

        # 1. Start constraint: agent at start position at t=0
        start_idx = self.pos_to_idx[start]
        cnf.append([var(0, start_idx)])
        for idx in range(n_positions):
            if idx != start_idx:
                cnf.append([-var(0, idx)])

        # 2. Goal constraint: agent at goal position at t=makespan
        goal_idx = self.pos_to_idx[goal]
        cnf.append([var(makespan, goal_idx)])

        # 3. At most one position per timestep (pairwise)
        for t in range(makespan + 1):
            # At least one position
            cnf.append([var(t, idx) for idx in range(n_positions)])

            # At most one position (pairwise negative)
            for i in range(n_positions):
                for j in range(i + 1, n_positions):
                    cnf.append([-var(t, i), -var(t, j)])

        # 4. Movement constraints: can only move to neighbor or stay
        for t in range(makespan):
            for pos_idx, pos in enumerate(self.all_positions):
                neighbors = self.warehouse.get_neighbors(pos)
                neighbor_indices = [self.pos_to_idx[n] for n in neighbors]

                # If at pos at time t, must be at pos or neighbor at time t+1
                valid_next = [pos_idx] + neighbor_indices
                clause = [-var(t, pos_idx)] + [var(t + 1, next_idx) for next_idx in valid_next]
                cnf.append(clause)

        # 5. Collision constraints with frozen agents
        for agent_id, frozen_path in frozen_paths.items():
            for t in range(min(makespan + 1, len(frozen_path))):
                frozen_pos = frozen_path[t]
                if frozen_pos in self.pos_to_idx:
                    frozen_idx = self.pos_to_idx[frozen_pos]
                    # Vertex conflict: cannot be at same position
                    cnf.append([-var(t, frozen_idx)])

            # Edge conflicts (swap)
            for t in range(min(makespan, len(frozen_path) - 1)):
                pos_t = frozen_path[t]
                pos_t1 = frozen_path[t + 1]
                if pos_t != pos_t1:  # Only if frozen agent moved
                    if pos_t in self.pos_to_idx and pos_t1 in self.pos_to_idx:
                        idx_t = self.pos_to_idx[pos_t]
                        idx_t1 = self.pos_to_idx[pos_t1]
                        # Cannot swap: if I'm at pos_t1 at t and pos_t at t+1
                        cnf.append([-var(t, idx_t1), -var(t + 1, idx_t)])

        # Solve
        solver = Solver(name='g3', bootstrap_with=cnf)  # Glucose3

        # Note: python-sat doesn't support direct timeout, so we solve without it
        # For production, consider using time-limited solving
        if solver.solve():
            model = solver.get_model()
            solver.delete()
            return self._decode_path(model, makespan, n_positions)
        else:
            solver.delete()
            return None

    def _decode_path(self, model: List[int], makespan: int, n_positions: int) -> List[Tuple[int, int]]:
        """Decode SAT solution to path."""
        path = []
        for t in range(makespan + 1):
            for pos_idx in range(n_positions):
                v = self._var(t, pos_idx, n_positions)
                if v in model:
                    path.append(self.all_positions[pos_idx])
                    break
        return path


class AStarPathPlanner:
    """
    A* path planner with space-time reservations for collision avoidance.
    Much faster than SAT for simple cases.
    """

    def __init__(self, warehouse: Warehouse):
        """
        Initialize planner.

        Args:
            warehouse: Warehouse instance
        """
        self.warehouse = warehouse

    def plan_path(self,
                  start: Tuple[int, int],
                  goal: Tuple[int, int],
                  frozen_paths: Dict[int, List[Tuple[int, int]]],
                  makespan_limit: int = None) -> Optional[List[Tuple[int, int]]]:
        """
        Plan collision-free path from start to goal using space-time A*.

        Args:
            start: Starting position
            goal: Goal position
            frozen_paths: Dictionary mapping agent_id -> list of positions over time
            makespan_limit: Maximum path length (None = auto-calculate)

        Returns:
            List of positions forming the path, or None if no solution found
        """
        if start == goal:
            return [start]

        # Calculate makespan limit
        min_dist = self.warehouse.manhattan_distance(start, goal)
        if makespan_limit is None:
            makespan_limit = min_dist * 3 + 10

        # Build reservation table from frozen paths
        # reservations[t] = set of occupied positions at time t
        reservations: Dict[int, Set[Tuple[int, int]]] = {}
        for agent_id, path in frozen_paths.items():
            for t, pos in enumerate(path):
                if t not in reservations:
                    reservations[t] = set()
                reservations[t].add(pos)
            # After path ends, agent stays at last position
            if path:
                last_pos = path[-1]
                for t in range(len(path), makespan_limit + 1):
                    if t not in reservations:
                        reservations[t] = set()
                    reservations[t].add(last_pos)

        # A* search in space-time
        # State: (x, y, t)
        # Priority queue: (f_score, counter, state)
        counter = 0
        start_state = (start[0], start[1], 0)

        # f = g + h, where g = time, h = manhattan distance to goal
        h_start = self.warehouse.manhattan_distance(start, goal)
        open_set = [(h_start, counter, start_state)]

        # g_score[state] = cost to reach state
        g_score = {start_state: 0}

        # came_from[state] = previous state
        came_from = {}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            x, y, t = current
            current_pos = (x, y)

            # Check if reached goal
            if current_pos == goal:
                return self._reconstruct_path(came_from, current)

            # Check makespan limit
            if t >= makespan_limit:
                continue

            # Generate neighbors (move or wait)
            neighbors = self.warehouse.get_neighbors(current_pos)
            neighbors.append(current_pos)  # Wait action

            for next_pos in neighbors:
                next_t = t + 1
                next_state = (next_pos[0], next_pos[1], next_t)

                # Check vertex conflict (next position occupied at next time)
                if next_t in reservations and next_pos in reservations[next_t]:
                    continue

                # Check edge conflict (swap)
                if t in reservations and next_pos in reservations.get(t, set()):
                    # Check if the agent at next_pos is moving to current_pos
                    swap_conflict = False
                    for agent_id, path in frozen_paths.items():
                        if t < len(path) and t + 1 < len(path):
                            if path[t] == next_pos and path[t + 1] == current_pos:
                                swap_conflict = True
                                break
                    if swap_conflict:
                        continue

                # Calculate tentative g score
                tentative_g = g_score[current] + 1

                if next_state not in g_score or tentative_g < g_score[next_state]:
                    g_score[next_state] = tentative_g
                    came_from[next_state] = current
                    h = self.warehouse.manhattan_distance(next_pos, goal)
                    f = tentative_g + h
                    counter += 1
                    heapq.heappush(open_set, (f, counter, next_state))

        return None  # No path found

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path = [(current[0], current[1])]
        while current in came_from:
            current = came_from[current]
            path.append((current[0], current[1]))
        path.reverse()
        return path


class MAPDSimulator:
    """
    Multi-Agent Pickup and Delivery simulator.
    """

    def __init__(self, warehouse: Warehouse, n_agents: int, order_generator: OrderGenerator,
                 seed: int = None, planner_type: str = "sat"):
        """
        Initialize simulator.

        Args:
            warehouse: Warehouse instance with layout set
            n_agents: Number of agents
            order_generator: Order generator instance
            seed: Random seed
            planner_type: "sat" for SAT-based planner, "astar" for A* planner
        """
        self.warehouse = warehouse
        self.order_generator = order_generator
        self.rng = np.random.default_rng(seed)

        # Select planner
        if planner_type == "astar":
            self.planner = AStarPathPlanner(warehouse)
        else:
            self.planner = SATPathPlanner(warehouse)
        self.planner_type = planner_type

        # Initialize agents on delivery points
        self.agents = self._init_agents(n_agents)

        # Statistics
        self.completed_orders = 0
        self.failed_assignments = 0
        self.current_time = 0

        # Pending orders queue
        self.pending_orders: List[int] = []

    def _init_agents(self, n: int) -> List[Agent]:
        """Initialize agents on random delivery points."""
        delivery_points = self.warehouse.get_delivery_points()
        selected = self.rng.choice(len(delivery_points), size=min(n, len(delivery_points)), replace=False)
        return [Agent(i, delivery_points[idx]) for i, idx in enumerate(selected)]

    def _get_idle_agents(self) -> List[Agent]:
        """Get list of idle agents."""
        return [a for a in self.agents if a.is_idle()]

    def _get_frozen_paths(self, exclude_agent: Agent, max_time: int) -> Dict[int, List[Tuple[int, int]]]:
        """Get paths of all agents except the specified one."""
        frozen = {}
        for agent in self.agents:
            if agent.id != exclude_agent.id:
                frozen[agent.id] = agent.get_future_positions(max_time)
        return frozen

    def _find_nearest_idle_agent(self, target_pos: Tuple[int, int]) -> Optional[Agent]:
        """Find the nearest idle agent to target position."""
        idle_agents = self._get_idle_agents()
        if not idle_agents:
            return None

        best_agent = None
        best_dist = float('inf')
        for agent in idle_agents:
            dist = self.warehouse.manhattan_distance(agent.position, target_pos)
            if dist < best_dist:
                best_dist = dist
                best_agent = agent
        return best_agent

    def _find_nearest_free_delivery(self, pos: Tuple[int, int], occupied: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find nearest delivery point not occupied by another agent."""
        delivery_points = self.warehouse.get_delivery_points()
        best_pos = None
        best_dist = float('inf')

        for dp in delivery_points:
            if dp not in occupied:
                dist = self.warehouse.manhattan_distance(pos, dp)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = dp

        return best_pos

    def _get_occupied_positions(self) -> Set[Tuple[int, int]]:
        """Get positions currently occupied by agents."""
        return {a.position for a in self.agents}

    def _assign_order(self, goods_id: int) -> bool:
        """
        Assign order to nearest idle agent.

        Returns:
            True if assignment successful
        """
        goods_pos = self.warehouse.get_goods_position(goods_id)
        agent = self._find_nearest_idle_agent(goods_pos)

        if agent is None:
            return False

        # Plan path to pickup
        max_time = self.warehouse.width + self.warehouse.height + 10
        frozen = self._get_frozen_paths(agent, max_time)
        path = self.planner.plan_path(agent.position, goods_pos, frozen)

        if path is None:
            return False

        agent.assign_pickup(goods_id, goods_pos, path)
        return True

    def _handle_pickup_completion(self, agent: Agent):
        """Handle agent completing pickup - assign delivery."""
        agent.complete_pickup()

        # Find nearest free delivery point
        occupied = self._get_occupied_positions()
        occupied.discard(agent.position)  # Agent's current position is fine
        delivery_pos = self._find_nearest_free_delivery(agent.position, occupied)

        if delivery_pos is None:
            # No free delivery point - stay in place
            delivery_pos = agent.position

        # Plan path to delivery
        max_time = self.warehouse.width + self.warehouse.height + 10
        frozen = self._get_frozen_paths(agent, max_time)
        path = self.planner.plan_path(agent.position, delivery_pos, frozen)

        if path is None:
            # Cannot plan path - deliver at current position if it's a delivery point
            if self.warehouse.is_delivery_point(agent.position):
                path = [agent.position]
            else:
                # Stay in place and retry next timestep
                path = [agent.position]

        agent.assign_delivery(delivery_pos, path)

    def step(self):
        """Execute one timestep of simulation."""
        # 1. Try to assign pending orders to idle agents
        orders_to_remove = []
        for i, goods_id in enumerate(self.pending_orders):
            if self._assign_order(goods_id):
                orders_to_remove.append(i)

        # Remove assigned orders (in reverse to preserve indices)
        for i in reversed(orders_to_remove):
            self.pending_orders.pop(i)

        # 2. Move all agents
        for agent in self.agents:
            reached_target = agent.step()

            if reached_target:
                if agent.state == AgentState.TO_PICKUP:
                    self._handle_pickup_completion(agent)
                elif agent.state == AgentState.TO_DELIVERY:
                    agent.complete_delivery()
                    self.completed_orders += 1

        self.current_time += 1

    def add_orders(self, orders: List[int]):
        """Add orders to pending queue."""
        self.pending_orders.extend(orders)

    def run(self, n_steps: int, order_lambda: float = 0.5) -> float:
        """
        Run simulation for n_steps.

        Args:
            n_steps: Number of timesteps to simulate
            order_lambda: Poisson lambda for order generation

        Returns:
            Throughput (completed orders / timesteps)
        """
        for _ in range(n_steps):
            # Generate new orders (Poisson process)
            new_orders = self.order_generator.generate_poisson_orders(order_lambda)
            self.add_orders(new_orders)

            # Execute timestep
            self.step()

        return self.completed_orders / n_steps

    def get_state(self) -> Dict:
        """Get current simulation state (for visualization)."""
        return {
            'time': self.current_time,
            'agents': [(a.id, a.position, a.state.value, a.carrying) for a in self.agents],
            'pending_orders': len(self.pending_orders),
            'completed_orders': self.completed_orders
        }


if __name__ == "__main__":
    # Quick test
    print("Testing MAPD Solver...")

    # Create warehouse with random layout
    wh = Warehouse(7, 7)
    layout = list(range(wh.n_inner))
    np.random.shuffle(layout)
    wh.set_layout(layout)

    # Create order generator
    og = OrderGenerator(wh.n_inner, seed=42)

    # Create simulator
    sim = MAPDSimulator(wh, n_agents=3, order_generator=og, seed=42)

    print(f"Initial state: {sim.get_state()}")

    # Run short simulation
    throughput = sim.run(n_steps=100, order_lambda=0.3)
    print(f"\nAfter 100 steps:")
    print(f"  Throughput: {throughput:.4f} orders/step")
    print(f"  Completed orders: {sim.completed_orders}")
    print(f"  Pending orders: {len(sim.pending_orders)}")
    print(f"  Final state: {sim.get_state()}")
