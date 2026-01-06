"""
Surrogate models for approximating MAPD simulation fitness.
"""

from typing import List, Tuple, Optional
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, Exception):
    # XGBoost may fail if OpenMP (libomp) is not installed
    HAS_XGBOOST = False
    XGBRegressor = None

from warehouse import Warehouse, OrderGenerator


def extract_features(warehouse: Warehouse,
                     order_distribution: np.ndarray,
                     n_agents: int) -> np.ndarray:
    """
    Extract features from warehouse layout for surrogate model.

    Args:
        warehouse: Warehouse with layout set
        order_distribution: Probability distribution of orders per goods
        n_agents: Number of agents

    Returns:
        Feature vector with layout characteristics
    """
    features = []

    # === Distance-based features ===

    # 1. Weighted average distance to edge
    weighted_avg = _weighted_avg_distance(warehouse, order_distribution)
    features.append(weighted_avg)

    # 2. Weighted standard deviation of distances
    weighted_std = _weighted_std_distance(warehouse, order_distribution)
    features.append(weighted_std)

    # 3. Max distance for top 10% most popular goods
    max_popular = _max_popular_distance(warehouse, order_distribution, top_pct=0.1)
    features.append(max_popular)

    # 4. Average distance of top 5 most popular items
    avg_top5 = _avg_top_n_distance(warehouse, order_distribution, n=5)
    features.append(avg_top5)

    # 5. Average distance of top 20% most popular items
    avg_top20 = _avg_top_n_distance(warehouse, order_distribution, n=max(1, int(warehouse.n_inner * 0.2)))
    features.append(avg_top20)

    # 6. Weighted squared distance (penalizes far items more)
    weighted_sq = _weighted_squared_distance(warehouse, order_distribution)
    features.append(weighted_sq)

    # 7. Min distance of top 5 popular items
    min_top5 = _min_top_n_distance(warehouse, order_distribution, n=5)
    features.append(min_top5)

    # 8. Max distance of top 5 popular items
    max_top5 = _max_top_n_distance(warehouse, order_distribution, n=5)
    features.append(max_top5)

    # === Position-based features ===

    # 9. Number of popular items (top 20%) on edge (distance=1)
    edge_popular = _count_popular_on_edge(warehouse, order_distribution, top_pct=0.2)
    features.append(edge_popular)

    # 10. Number of popular items (top 10%) on edge
    edge_top10 = _count_popular_on_edge(warehouse, order_distribution, top_pct=0.1)
    features.append(edge_top10)

    # 11. Fraction of top 20% items in corners (distance=2)
    corner_frac = _fraction_popular_in_corners(warehouse, order_distribution, top_pct=0.2)
    features.append(corner_frac)

    # === Congestion/clustering features ===

    # 12. Congestion score (how clustered popular goods are)
    congestion = _congestion_score(warehouse, order_distribution)
    features.append(congestion)

    # 13. Average pairwise distance between top 5 items
    avg_pairwise_top5 = _avg_pairwise_distance(warehouse, order_distribution, n=5)
    features.append(avg_pairwise_top5)

    # 14. Dispersion of popular items (std of their positions)
    dispersion = _popular_dispersion(warehouse, order_distribution, top_pct=0.2)
    features.append(dispersion)

    # === Distribution-based features ===

    # 15. Entropy of distance distribution (weighted)
    dist_entropy = _distance_entropy(warehouse, order_distribution)
    features.append(dist_entropy)

    # 16. Gini coefficient of distances (inequality)
    gini = _distance_gini(warehouse, order_distribution)
    features.append(gini)

    # 17. Probability mass at edge (sum of probs for items at distance=1)
    edge_prob_mass = _edge_probability_mass(warehouse, order_distribution)
    features.append(edge_prob_mass)

    # 18. Probability mass in center (distance >= 3)
    center_prob_mass = _center_probability_mass(warehouse, order_distribution, min_dist=3)
    features.append(center_prob_mass)

    # === Quadrant features ===

    # 19-22. Weighted distance in each quadrant
    quadrant_dists = _quadrant_weighted_distances(warehouse, order_distribution)
    features.extend(quadrant_dists)

    # === Agent-aware features ===

    # 23. Ratio of popular items per agent at edge
    edge_popular_per_agent = edge_popular / max(1, n_agents)
    features.append(edge_popular_per_agent)

    # 24. Expected travel distance per order (weighted avg dist * 2 for round trip)
    expected_travel = weighted_avg * 2
    features.append(expected_travel)

    # 25. Theoretical max throughput ratio (how close to optimal)
    min_possible_dist = 1.0
    throughput_ratio = min_possible_dist / max(weighted_avg, 0.1)
    features.append(throughput_ratio)

    # === Path-based features (NEW) ===

    # 26. Average distance to nearest delivery point (weighted)
    avg_to_delivery = _weighted_avg_to_delivery(warehouse, order_distribution)
    features.append(avg_to_delivery)

    # 27. Min distance to delivery for top 5 items
    min_delivery_top5 = _min_delivery_distance_top_n(warehouse, order_distribution, n=5)
    features.append(min_delivery_top5)

    # 28. Max distance to delivery for top 5 items
    max_delivery_top5 = _max_delivery_distance_top_n(warehouse, order_distribution, n=5)
    features.append(max_delivery_top5)

    # === Accessibility features (NEW) ===

    # 29. Number of unique delivery points adjacent to top 20% items
    accessible_deliveries = _count_accessible_deliveries(warehouse, order_distribution, top_pct=0.2)
    features.append(accessible_deliveries)

    # 30. Average number of neighboring delivery points for popular items
    avg_neighbors = _avg_delivery_neighbors(warehouse, order_distribution, top_pct=0.2)
    features.append(avg_neighbors)

    # === Spatial clustering features (NEW) ===

    # 31. Local density of popular items (how many popular items are neighbors)
    local_density = _popular_local_density(warehouse, order_distribution, top_pct=0.2)
    features.append(local_density)

    # 32. Spread along X axis (weighted std of x coordinates)
    spread_x = _weighted_spread_x(warehouse, order_distribution)
    features.append(spread_x)

    # 33. Spread along Y axis (weighted std of y coordinates)
    spread_y = _weighted_spread_y(warehouse, order_distribution)
    features.append(spread_y)

    # 34. Balance ratio (how evenly distributed across quadrants)
    balance = _quadrant_balance(warehouse, order_distribution)
    features.append(balance)

    # === Conflict potential features (NEW) ===

    # 35. Hotspot score (probability mass in high-traffic areas)
    hotspot = _hotspot_score(warehouse, order_distribution)
    features.append(hotspot)

    # 36. Bottleneck score (popular items sharing same nearest delivery)
    bottleneck = _bottleneck_score(warehouse, order_distribution, top_pct=0.3)
    features.append(bottleneck)

    # 37. Path overlap potential (popular items on same row/column)
    path_overlap = _path_overlap_score(warehouse, order_distribution, top_pct=0.2)
    features.append(path_overlap)

    # === Distance distribution features (NEW) ===

    # 38. Median weighted distance
    median_dist = _weighted_median_distance(warehouse, order_distribution)
    features.append(median_dist)

    # 39. 90th percentile distance for popular items
    p90_dist = _percentile_distance(warehouse, order_distribution, percentile=90, top_pct=0.3)
    features.append(p90_dist)

    # 40. Skewness of distance distribution
    skewness = _distance_skewness(warehouse, order_distribution)
    features.append(skewness)

    return np.array(features)


def _weighted_avg_distance(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """
    Calculate weighted average distance to nearest edge.

    Weights are order probabilities.
    """
    total = 0.0
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        weight = distribution[goods_id]
        total += dist * weight
    return total


def _weighted_std_distance(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """
    Calculate weighted standard deviation of distances to edge.
    """
    distances = []
    weights = []
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        distances.append(dist)
        weights.append(distribution[goods_id])

    distances = np.array(distances)
    weights = np.array(weights)

    # Weighted mean
    weighted_mean = np.sum(distances * weights)

    # Weighted variance
    weighted_var = np.sum(weights * (distances - weighted_mean) ** 2)

    return np.sqrt(weighted_var)


def _max_popular_distance(warehouse: Warehouse, distribution: np.ndarray, top_pct: float = 0.1) -> float:
    """
    Calculate maximum distance for top X% most popular goods.
    """
    n_top = max(1, int(warehouse.n_inner * top_pct))

    # Get top goods by probability
    top_goods = np.argsort(distribution)[-n_top:]

    max_dist = 0
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        max_dist = max(max_dist, dist)

    return max_dist


def _congestion_score(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """
    Calculate congestion score - how clustered popular goods are.

    High score = popular goods are close together (potential for conflicts).
    """
    # Get top 30% most popular goods
    n_top = max(2, int(warehouse.n_inner * 0.3))
    top_goods = np.argsort(distribution)[-n_top:]

    # Calculate average pairwise distance between popular goods
    positions = [warehouse.get_goods_position(g) for g in top_goods]

    total_dist = 0
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total_dist += warehouse.manhattan_distance(positions[i], positions[j])
            count += 1

    if count == 0:
        return 0.0

    avg_dist = total_dist / count

    # Invert: lower avg distance = higher congestion
    # Normalize by grid diagonal
    max_dist = warehouse.width + warehouse.height - 2
    congestion = 1 - (avg_dist / max_dist)

    return congestion


def _avg_top_n_distance(warehouse: Warehouse, distribution: np.ndarray, n: int) -> float:
    """
    Calculate average distance to edge for top N most popular goods.
    """
    top_goods = np.argsort(distribution)[-n:]
    total_dist = 0
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        total_dist += warehouse.distance_to_nearest_edge(pos)
    return total_dist / n


def _count_popular_on_edge(warehouse: Warehouse, distribution: np.ndarray, top_pct: float) -> int:
    """
    Count how many popular items are on the edge (distance=1).
    """
    n_top = max(1, int(warehouse.n_inner * top_pct))
    top_goods = np.argsort(distribution)[-n_top:]
    count = 0
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        if warehouse.distance_to_nearest_edge(pos) == 1:
            count += 1
    return count


def _weighted_squared_distance(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """
    Calculate weighted squared distance - penalizes far popular items more.
    """
    total = 0.0
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        weight = distribution[goods_id]
        total += (dist ** 2) * weight
    return total


def _min_top_n_distance(warehouse: Warehouse, distribution: np.ndarray, n: int) -> float:
    """Minimum distance to edge among top N popular items."""
    top_goods = np.argsort(distribution)[-n:]
    min_dist = float('inf')
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        min_dist = min(min_dist, dist)
    return min_dist


def _max_top_n_distance(warehouse: Warehouse, distribution: np.ndarray, n: int) -> float:
    """Maximum distance to edge among top N popular items."""
    top_goods = np.argsort(distribution)[-n:]
    max_dist = 0
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        max_dist = max(max_dist, dist)
    return max_dist


def _fraction_popular_in_corners(warehouse: Warehouse, distribution: np.ndarray, top_pct: float) -> float:
    """Fraction of popular items that are in inner corners (distance=2)."""
    n_top = max(1, int(warehouse.n_inner * top_pct))
    top_goods = np.argsort(distribution)[-n_top:]
    count = 0
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        if warehouse.distance_to_nearest_edge(pos) == 2:
            count += 1
    return count / n_top


def _avg_pairwise_distance(warehouse: Warehouse, distribution: np.ndarray, n: int) -> float:
    """Average pairwise Manhattan distance between top N items."""
    top_goods = np.argsort(distribution)[-n:]
    positions = [warehouse.get_goods_position(g) for g in top_goods]

    total_dist = 0
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total_dist += warehouse.manhattan_distance(positions[i], positions[j])
            count += 1

    return total_dist / count if count > 0 else 0.0


def _popular_dispersion(warehouse: Warehouse, distribution: np.ndarray, top_pct: float) -> float:
    """Standard deviation of positions of popular items (measures spread)."""
    n_top = max(2, int(warehouse.n_inner * top_pct))
    top_goods = np.argsort(distribution)[-n_top:]

    positions = np.array([warehouse.get_goods_position(g) for g in top_goods])

    # Std of x and y coordinates
    std_x = np.std(positions[:, 0])
    std_y = np.std(positions[:, 1])

    return (std_x + std_y) / 2


def _distance_entropy(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Entropy of the weighted distance distribution."""
    # Get distances weighted by probability
    distances = []
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        distances.append(dist)

    distances = np.array(distances)

    # Create histogram of distances weighted by distribution
    max_dist = int(distances.max())
    bins = np.zeros(max_dist + 1)
    for d, p in zip(distances, distribution):
        bins[int(d)] += p

    # Calculate entropy
    bins = bins[bins > 0]  # Remove zeros
    entropy = -np.sum(bins * np.log(bins + 1e-10))
    return entropy


def _distance_gini(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Gini coefficient of weighted distances (inequality measure)."""
    weighted_distances = []
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        weighted_distances.append(dist * distribution[goods_id])

    wd = np.array(sorted(weighted_distances))
    n = len(wd)
    cumsum = np.cumsum(wd)

    if cumsum[-1] == 0:
        return 0.0

    gini = (2 * np.sum((np.arange(1, n + 1) * wd))) / (n * cumsum[-1]) - (n + 1) / n
    return gini


def _edge_probability_mass(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Sum of probabilities for items at distance=1 from edge."""
    total = 0.0
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        if warehouse.distance_to_nearest_edge(pos) == 1:
            total += distribution[goods_id]
    return total


def _center_probability_mass(warehouse: Warehouse, distribution: np.ndarray, min_dist: int) -> float:
    """Sum of probabilities for items at distance >= min_dist from edge."""
    total = 0.0
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        if warehouse.distance_to_nearest_edge(pos) >= min_dist:
            total += distribution[goods_id]
    return total


def _quadrant_weighted_distances(warehouse: Warehouse, distribution: np.ndarray) -> List[float]:
    """Weighted average distance in each quadrant (NW, NE, SW, SE)."""
    mid_x = warehouse.width / 2
    mid_y = warehouse.height / 2

    quadrant_totals = [0.0, 0.0, 0.0, 0.0]  # NW, NE, SW, SE
    quadrant_weights = [0.0, 0.0, 0.0, 0.0]

    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        weight = distribution[goods_id]

        # Determine quadrant
        if pos[0] < mid_x and pos[1] >= mid_y:
            q = 0  # NW
        elif pos[0] >= mid_x and pos[1] >= mid_y:
            q = 1  # NE
        elif pos[0] < mid_x and pos[1] < mid_y:
            q = 2  # SW
        else:
            q = 3  # SE

        quadrant_totals[q] += dist * weight
        quadrant_weights[q] += weight

    # Normalize
    result = []
    for i in range(4):
        if quadrant_weights[i] > 0:
            result.append(quadrant_totals[i] / quadrant_weights[i])
        else:
            result.append(0.0)

    return result


# === NEW HELPER FUNCTIONS ===

def _weighted_avg_to_delivery(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Weighted average distance to nearest delivery point."""
    delivery_points = warehouse.get_delivery_points()
    total = 0.0
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        min_dist = min(warehouse.manhattan_distance(pos, dp) for dp in delivery_points)
        total += min_dist * distribution[goods_id]
    return total


def _min_delivery_distance_top_n(warehouse: Warehouse, distribution: np.ndarray, n: int) -> float:
    """Minimum distance to delivery point among top N popular items."""
    delivery_points = warehouse.get_delivery_points()
    top_goods = np.argsort(distribution)[-n:]
    min_dist = float('inf')
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        dist = min(warehouse.manhattan_distance(pos, dp) for dp in delivery_points)
        min_dist = min(min_dist, dist)
    return min_dist


def _max_delivery_distance_top_n(warehouse: Warehouse, distribution: np.ndarray, n: int) -> float:
    """Maximum distance to delivery point among top N popular items."""
    delivery_points = warehouse.get_delivery_points()
    top_goods = np.argsort(distribution)[-n:]
    max_dist = 0
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        dist = min(warehouse.manhattan_distance(pos, dp) for dp in delivery_points)
        max_dist = max(max_dist, dist)
    return max_dist


def _count_accessible_deliveries(warehouse: Warehouse, distribution: np.ndarray, top_pct: float) -> int:
    """Count unique delivery points adjacent to popular items."""
    n_top = max(1, int(warehouse.n_inner * top_pct))
    top_goods = np.argsort(distribution)[-n_top:]
    delivery_points = set(warehouse.get_delivery_points())

    accessible = set()
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        for neighbor in warehouse.get_neighbors(pos):
            if neighbor in delivery_points:
                accessible.add(neighbor)
    return len(accessible)


def _avg_delivery_neighbors(warehouse: Warehouse, distribution: np.ndarray, top_pct: float) -> float:
    """Average number of delivery point neighbors for popular items."""
    n_top = max(1, int(warehouse.n_inner * top_pct))
    top_goods = np.argsort(distribution)[-n_top:]
    delivery_points = set(warehouse.get_delivery_points())

    total_neighbors = 0
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        count = sum(1 for n in warehouse.get_neighbors(pos) if n in delivery_points)
        total_neighbors += count
    return total_neighbors / n_top


def _popular_local_density(warehouse: Warehouse, distribution: np.ndarray, top_pct: float) -> float:
    """Average number of popular neighbors for each popular item."""
    n_top = max(2, int(warehouse.n_inner * top_pct))
    top_goods = set(np.argsort(distribution)[-n_top:])

    # Map goods to positions
    goods_positions = {goods_id: warehouse.get_goods_position(goods_id) for goods_id in top_goods}
    position_to_goods = {pos: gid for gid, pos in goods_positions.items()}

    total_neighbors = 0
    for goods_id in top_goods:
        pos = goods_positions[goods_id]
        neighbors = warehouse.get_neighbors(pos)
        popular_neighbors = sum(1 for n in neighbors if n in position_to_goods)
        total_neighbors += popular_neighbors

    return total_neighbors / n_top


def _weighted_spread_x(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Weighted standard deviation of x coordinates."""
    x_coords = []
    weights = []
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        x_coords.append(pos[0])
        weights.append(distribution[goods_id])

    x_coords = np.array(x_coords)
    weights = np.array(weights)
    weighted_mean = np.sum(x_coords * weights)
    weighted_var = np.sum(weights * (x_coords - weighted_mean) ** 2)
    return np.sqrt(weighted_var)


def _weighted_spread_y(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Weighted standard deviation of y coordinates."""
    y_coords = []
    weights = []
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        y_coords.append(pos[1])
        weights.append(distribution[goods_id])

    y_coords = np.array(y_coords)
    weights = np.array(weights)
    weighted_mean = np.sum(y_coords * weights)
    weighted_var = np.sum(weights * (y_coords - weighted_mean) ** 2)
    return np.sqrt(weighted_var)


def _quadrant_balance(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """How evenly distributed probability mass is across quadrants (0=uneven, 1=even)."""
    mid_x = warehouse.width / 2
    mid_y = warehouse.height / 2

    quadrant_mass = [0.0, 0.0, 0.0, 0.0]
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        if pos[0] < mid_x and pos[1] >= mid_y:
            q = 0
        elif pos[0] >= mid_x and pos[1] >= mid_y:
            q = 1
        elif pos[0] < mid_x and pos[1] < mid_y:
            q = 2
        else:
            q = 3
        quadrant_mass[q] += distribution[goods_id]

    # Balance = 1 - normalized std (0 = all in one quadrant, 1 = perfectly balanced)
    std = np.std(quadrant_mass)
    max_std = 0.25  # Maximum possible std when all mass in one quadrant
    return 1 - min(std / max_std, 1.0)


def _hotspot_score(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Probability mass in center positions (potential congestion area)."""
    center_x = warehouse.width / 2
    center_y = warehouse.height / 2
    radius = min(warehouse.width, warehouse.height) / 4

    total = 0.0
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist_to_center = abs(pos[0] - center_x) + abs(pos[1] - center_y)
        if dist_to_center <= radius:
            total += distribution[goods_id]
    return total


def _bottleneck_score(warehouse: Warehouse, distribution: np.ndarray, top_pct: float) -> float:
    """How many popular items share the same nearest delivery point."""
    n_top = max(2, int(warehouse.n_inner * top_pct))
    top_goods = np.argsort(distribution)[-n_top:]
    delivery_points = warehouse.get_delivery_points()

    # Count which delivery point is nearest for each popular item
    nearest_counts = {}
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        nearest = min(delivery_points, key=lambda dp: warehouse.manhattan_distance(pos, dp))
        nearest_counts[nearest] = nearest_counts.get(nearest, 0) + 1

    # Return max count (higher = more bottleneck)
    if not nearest_counts:
        return 0.0
    return max(nearest_counts.values()) / n_top


def _path_overlap_score(warehouse: Warehouse, distribution: np.ndarray, top_pct: float) -> float:
    """Score based on how many popular items share rows/columns (path conflicts)."""
    n_top = max(2, int(warehouse.n_inner * top_pct))
    top_goods = np.argsort(distribution)[-n_top:]

    positions = [warehouse.get_goods_position(g) for g in top_goods]

    # Count items per row and column
    row_counts = {}
    col_counts = {}
    for pos in positions:
        row_counts[pos[1]] = row_counts.get(pos[1], 0) + 1
        col_counts[pos[0]] = col_counts.get(pos[0], 0) + 1

    # Score = average items sharing same row/column
    row_overlap = sum(c * (c - 1) for c in row_counts.values()) / max(1, n_top)
    col_overlap = sum(c * (c - 1) for c in col_counts.values()) / max(1, n_top)

    return (row_overlap + col_overlap) / 2


def _weighted_median_distance(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Weighted median distance to edge."""
    distances = []
    weights = []
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        distances.append(dist)
        weights.append(distribution[goods_id])

    # Sort by distance
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    weights = np.array(weights)[sorted_indices]

    # Find weighted median
    cumsum = np.cumsum(weights)
    median_idx = np.searchsorted(cumsum, 0.5)
    return distances[min(median_idx, len(distances) - 1)]


def _percentile_distance(warehouse: Warehouse, distribution: np.ndarray,
                         percentile: int, top_pct: float) -> float:
    """Percentile distance among popular items."""
    n_top = max(1, int(warehouse.n_inner * top_pct))
    top_goods = np.argsort(distribution)[-n_top:]

    distances = []
    for goods_id in top_goods:
        pos = warehouse.get_goods_position(goods_id)
        distances.append(warehouse.distance_to_nearest_edge(pos))

    return np.percentile(distances, percentile)


def _distance_skewness(warehouse: Warehouse, distribution: np.ndarray) -> float:
    """Skewness of weighted distance distribution."""
    distances = []
    weights = []
    for goods_id in range(warehouse.n_inner):
        pos = warehouse.get_goods_position(goods_id)
        dist = warehouse.distance_to_nearest_edge(pos)
        distances.append(dist)
        weights.append(distribution[goods_id])

    distances = np.array(distances)
    weights = np.array(weights)

    # Weighted mean and std
    mean = np.sum(distances * weights)
    var = np.sum(weights * (distances - mean) ** 2)
    std = np.sqrt(var) if var > 0 else 1e-10

    # Weighted skewness
    skewness = np.sum(weights * ((distances - mean) / std) ** 3)
    return skewness


class SurrogateModel:
    """
    Surrogate model for predicting MAPD throughput.

    Supports multiple model types: linear, ridge, gp, rf, xgboost, voting, stacking.
    """

    AVAILABLE_MODELS = ['linear', 'ridge', 'gp', 'rf', 'voting', 'stacking']
    if HAS_XGBOOST:
        AVAILABLE_MODELS.append('xgboost')

    def __init__(self, model_type: str = "ridge", feature_indices: Optional[List[int]] = None):
        """
        Initialize surrogate model.

        Args:
            model_type: 'linear', 'ridge', 'gp', 'rf', 'voting', 'stacking', or 'xgboost'
            feature_indices: Optional list of feature indices to use (for feature selection).
                           If None, all features are used.
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {self.AVAILABLE_MODELS}")

        self.model_type = model_type
        self.feature_indices = feature_indices
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []
        self.is_fitted = False

    def _create_model(self):
        """Create sklearn model based on type."""
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(alpha=10.0)  # Regularized linear model
        elif self.model_type == "gp":
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        elif self.model_type == "rf":
            return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        elif self.model_type == "xgboost":
            return XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        elif self.model_type == "voting":
            return VotingRegressor([
                ('ridge', Ridge(alpha=10.0)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
            ])
        elif self.model_type == "stacking":
            return StackingRegressor(
                estimators=[
                    ('ridge', Ridge(alpha=10.0)),
                    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                    ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
                ],
                final_estimator=Ridge(alpha=1.0),
                cv=5
            )

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection if feature_indices is set."""
        if self.feature_indices is None:
            return X
        if X.ndim == 1:
            return X[self.feature_indices]
        return X[:, self.feature_indices]

    def add_sample(self, features: np.ndarray, fitness: float):
        """
        Add a new training sample.

        Args:
            features: Feature vector from extract_features()
            fitness: Actual throughput from simulation
        """
        self.X_train.append(features)  # Store full features
        self.y_train.append(fitness)

    def fit(self):
        """Fit/refit the model on all training data."""
        if len(self.X_train) < 2:
            return

        X = np.array(self.X_train)
        y = np.array(self.y_train)

        # Apply feature selection
        X = self._select_features(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def update(self, features: np.ndarray, fitness: float):
        """
        Add sample and refit model.

        Args:
            features: Feature vector
            fitness: Actual throughput
        """
        self.add_sample(features, fitness)
        self.fit()

    def predict(self, features: np.ndarray) -> float:
        """
        Predict throughput for given features.

        Args:
            features: Feature vector

        Returns:
            Predicted throughput
        """
        if not self.is_fitted:
            # Return dummy value if not fitted
            return 0.0

        # Apply feature selection
        features = self._select_features(features)
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])

    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """
        Predict throughput for multiple feature vectors.

        Args:
            features_batch: Array of shape (n_samples, n_features)

        Returns:
            Array of predicted throughputs
        """
        if not self.is_fitted:
            return np.zeros(len(features_batch))

        # Apply feature selection
        features_batch = self._select_features(features_batch)
        X_scaled = self.scaler.transform(features_batch)
        return self.model.predict(X_scaled)

    def get_uncertainty(self, features: np.ndarray) -> float:
        """
        Get prediction uncertainty (only for GP model).

        Args:
            features: Feature vector

        Returns:
            Standard deviation of prediction (0 for non-GP models)
        """
        if self.model_type != "gp" or not self.is_fitted:
            return 0.0

        # Apply feature selection
        features = self._select_features(features)
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        _, std = self.model.predict(X_scaled, return_std=True)
        return float(std[0])

    def get_cv_score(self, cv: int = 5) -> Tuple[float, float]:
        """
        Get cross-validation score.

        Args:
            cv: Number of CV folds

        Returns:
            (mean_r2, std_r2) tuple
        """
        if len(self.X_train) < cv:
            return 0.0, 0.0

        X = np.array(self.X_train)
        y = np.array(self.y_train)
        X_scaled = self.scaler.fit_transform(X)

        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='r2')
        return float(np.mean(scores)), float(np.std(scores))

    def n_samples(self) -> int:
        """Return number of training samples."""
        return len(self.X_train)

    def __repr__(self) -> str:
        return f"SurrogateModel({self.model_type}, {self.n_samples()} samples, fitted={self.is_fitted})"


def compare_models(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """
    Compare different surrogate models using cross-validation.

    Args:
        X: Feature matrix
        y: Target values
        cv: Number of CV folds

    Returns:
        Dictionary with model scores
    """
    results = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=10.0),
        'gp': GaussianProcessRegressor(kernel=ConstantKernel() * RBF() + WhiteKernel(), random_state=42),
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
    }

    if HAS_XGBOOST:
        models['xgboost'] = XGBRegressor(n_estimators=100, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        results[name] = {
            'r2_mean': float(np.mean(scores)),
            'r2_std': float(np.std(scores))
        }

    return results


if __name__ == "__main__":
    # Quick test
    print("Testing Surrogate Model...")

    from mapd_solver import MAPDSimulator

    # Create warehouse
    wh = Warehouse(7, 7)
    og = OrderGenerator(wh.n_inner, seed=42)

    # Collect training data
    print("\nCollecting training data (10 random layouts)...")
    X_data = []
    y_data = []

    for i in range(10):
        # Random layout
        layout = list(range(wh.n_inner))
        np.random.shuffle(layout)
        wh.set_layout(layout)

        # Extract features
        features = extract_features(wh, og.get_probabilities(), n_agents=3)
        X_data.append(features)

        # Run simulation
        sim = MAPDSimulator(wh, n_agents=3, order_generator=og, seed=i)
        throughput = sim.run(n_steps=200, order_lambda=0.3)
        y_data.append(throughput)

        print(f"  Layout {i}: throughput = {throughput:.4f}")

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # Test surrogate model
    print("\nTraining surrogate models...")
    for model_type in SurrogateModel.AVAILABLE_MODELS:
        surr = SurrogateModel(model_type)
        for x, y in zip(X_data, y_data):
            surr.add_sample(x, y)
        surr.fit()

        # Predict on first sample
        pred = surr.predict(X_data[0])
        print(f"  {model_type}: predicted = {pred:.4f}, actual = {y_data[0]:.4f}")

    # Compare models
    print("\nCross-validation comparison:")
    results = compare_models(X_data, y_data, cv=3)
    for name, scores in results.items():
        print(f"  {name}: R² = {scores['r2_mean']:.3f} ± {scores['r2_std']:.3f}")
