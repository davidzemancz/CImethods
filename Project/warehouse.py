"""
Warehouse representation and order generation for MAPD optimization.
"""

from typing import List, Tuple
import numpy as np


class Warehouse:
    """
    Represents a warehouse grid with goods and delivery points.

    The grid is M x N cells where:
    - Border cells (edges) are delivery points
    - Inner cells contain goods (one type per cell)
    - Agents can move through all cells
    """

    def __init__(self, width: int = 7, height: int = 7):
        """
        Initialize warehouse.

        Args:
            width: Grid width (M)
            height: Grid height (N)
        """
        self.width = width
        self.height = height
        self.n_inner = (width - 2) * (height - 2)  # Number of inner cells = number of goods
        self.layout = None  # Permutation: layout[position_idx] = goods_id

        # Precompute positions
        self._inner_positions = self._compute_inner_positions()
        self._delivery_points = self._compute_delivery_points()

    def _compute_inner_positions(self) -> List[Tuple[int, int]]:
        """Compute list of inner positions (where goods are stored)."""
        positions = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                positions.append((x, y))
        return positions

    def _compute_delivery_points(self) -> List[Tuple[int, int]]:
        """Compute list of delivery points (border cells)."""
        points = []
        for x in range(self.width):
            points.append((x, 0))  # Top edge
            points.append((x, self.height - 1))  # Bottom edge
        for y in range(1, self.height - 1):
            points.append((0, y))  # Left edge
            points.append((self.width - 1, y))  # Right edge
        return points

    def set_layout(self, permutation: List[int]):
        """
        Set goods layout from permutation.

        Args:
            permutation: List where permutation[i] = goods_id at position i
        """
        if len(permutation) != self.n_inner:
            raise ValueError(f"Permutation length {len(permutation)} != {self.n_inner} inner cells")
        self.layout = list(permutation)

    def get_inner_positions(self) -> List[Tuple[int, int]]:
        """Return list of inner positions (where goods are stored)."""
        return self._inner_positions.copy()

    def get_delivery_points(self) -> List[Tuple[int, int]]:
        """Return list of delivery points (border cells)."""
        return self._delivery_points.copy()

    def get_goods_position(self, goods_id: int) -> Tuple[int, int]:
        """
        Get position of specific goods.

        Args:
            goods_id: ID of the goods (0 to n_inner-1)

        Returns:
            (x, y) position of the goods
        """
        if self.layout is None:
            raise ValueError("Layout not set")
        position_idx = self.layout.index(goods_id)
        return self._inner_positions[position_idx]

    def get_goods_at_position(self, pos: Tuple[int, int]) -> int:
        """
        Get goods ID at a specific position.

        Args:
            pos: (x, y) position

        Returns:
            Goods ID at that position, or -1 if not an inner position
        """
        if self.layout is None:
            raise ValueError("Layout not set")
        if pos in self._inner_positions:
            idx = self._inner_positions.index(pos)
            return self.layout[idx]
        return -1

    def is_delivery_point(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a delivery point."""
        x, y = pos
        return x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells (for agent movement).

        Args:
            pos: Current position

        Returns:
            List of valid neighboring positions (up, down, left, right)
        """
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_position((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors

    def distance_to_nearest_edge(self, pos: Tuple[int, int]) -> int:
        """Calculate minimum Manhattan distance to any edge (delivery point)."""
        x, y = pos
        return min(x, self.width - 1 - x, y, self.height - 1 - y)

    def __repr__(self) -> str:
        return f"Warehouse({self.width}x{self.height}, {self.n_inner} goods)"


class OrderGenerator:
    """
    Generates orders according to a probability distribution.

    Uses Zipf distribution by default (few popular items, many rare ones).
    """

    def __init__(self, n_goods: int, distribution: str = "zipf", zipf_param: float = 1.5, seed: int = None):
        """
        Initialize order generator.

        Args:
            n_goods: Number of different goods types
            distribution: "zipf" or "uniform"
            zipf_param: Parameter for Zipf distribution (higher = more skewed)
            seed: Random seed for reproducibility
        """
        self.n_goods = n_goods
        self.distribution = distribution
        self.zipf_param = zipf_param
        self.rng = np.random.default_rng(seed)
        self.probabilities = self._create_distribution()

    def _create_distribution(self) -> np.ndarray:
        """Create probability distribution for orders."""
        if self.distribution == "uniform":
            return np.ones(self.n_goods) / self.n_goods
        elif self.distribution == "zipf":
            # Zipf: P(k) proportional to 1/k^s
            ranks = np.arange(1, self.n_goods + 1)
            probs = 1.0 / np.power(ranks, self.zipf_param)
            return probs / probs.sum()
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def generate_order(self) -> int:
        """
        Generate a single order.

        Returns:
            Goods ID for the order
        """
        return self.rng.choice(self.n_goods, p=self.probabilities)

    def generate_orders(self, n: int) -> List[int]:
        """
        Generate multiple orders.

        Args:
            n: Number of orders to generate

        Returns:
            List of goods IDs
        """
        return list(self.rng.choice(self.n_goods, size=n, p=self.probabilities))

    def generate_poisson_orders(self, lam: float) -> List[int]:
        """
        Generate orders according to Poisson process.

        Args:
            lam: Expected number of orders (Poisson lambda)

        Returns:
            List of goods IDs (may be empty)
        """
        n_orders = self.rng.poisson(lam)
        if n_orders == 0:
            return []
        return self.generate_orders(n_orders)

    def get_expected_counts(self, n_orders: int) -> np.ndarray:
        """
        Get expected number of orders for each goods type.

        Args:
            n_orders: Total expected number of orders

        Returns:
            Array of expected counts per goods type
        """
        return self.probabilities * n_orders

    def get_probabilities(self) -> np.ndarray:
        """Return the probability distribution."""
        return self.probabilities.copy()

    def __repr__(self) -> str:
        return f"OrderGenerator({self.n_goods} goods, {self.distribution})"


if __name__ == "__main__":
    # Quick test
    wh = Warehouse(7, 7)
    print(f"Created {wh}")
    print(f"Inner positions: {len(wh.get_inner_positions())}")
    print(f"Delivery points: {len(wh.get_delivery_points())}")

    # Set random layout
    layout = list(range(wh.n_inner))
    np.random.shuffle(layout)
    wh.set_layout(layout)
    print(f"Goods 0 is at position: {wh.get_goods_position(0)}")

    # Test order generator
    og = OrderGenerator(wh.n_inner, seed=42)
    print(f"\nCreated {og}")
    print(f"Probabilities (first 5): {og.get_probabilities()[:5]}")
    print(f"Sample orders: {og.generate_orders(10)}")
    print(f"Poisson orders (lambda=2): {og.generate_poisson_orders(2)}")
