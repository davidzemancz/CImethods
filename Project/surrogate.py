"""
Surrogate models for approximating MAPD simulation fitness.
"""

from typing import List, Tuple, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
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
        Feature vector: [weighted_avg_dist, weighted_std_dist, max_popular_dist,
                        congestion_score, n_agents, grid_size]
    """
    features = []

    # 1. Weighted average distance to edge
    weighted_avg = _weighted_avg_distance(warehouse, order_distribution)
    features.append(weighted_avg)

    # 2. Weighted standard deviation of distances
    weighted_std = _weighted_std_distance(warehouse, order_distribution)
    features.append(weighted_std)

    # 3. Max distance for top 10% most popular goods
    max_popular = _max_popular_distance(warehouse, order_distribution, top_pct=0.1)
    features.append(max_popular)

    # 4. Congestion score (how clustered popular goods are)
    congestion = _congestion_score(warehouse, order_distribution)
    features.append(congestion)

    # 5. Number of agents
    features.append(n_agents)

    # 6. Grid size
    features.append(warehouse.width * warehouse.height)

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


class SurrogateModel:
    """
    Surrogate model for predicting MAPD throughput.

    Supports multiple model types: linear, gp, rf, xgboost.
    """

    AVAILABLE_MODELS = ['linear', 'gp', 'rf']
    if HAS_XGBOOST:
        AVAILABLE_MODELS.append('xgboost')

    def __init__(self, model_type: str = "gp"):
        """
        Initialize surrogate model.

        Args:
            model_type: 'linear', 'gp', 'rf', or 'xgboost'
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {self.AVAILABLE_MODELS}")

        self.model_type = model_type
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []
        self.is_fitted = False

    def _create_model(self):
        """Create sklearn model based on type."""
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "gp":
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        elif self.model_type == "rf":
            return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        elif self.model_type == "xgboost":
            return XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    def add_sample(self, features: np.ndarray, fitness: float):
        """
        Add a new training sample.

        Args:
            features: Feature vector from extract_features()
            fitness: Actual throughput from simulation
        """
        self.X_train.append(features)
        self.y_train.append(fitness)

    def fit(self):
        """Fit/refit the model on all training data."""
        if len(self.X_train) < 2:
            return

        X = np.array(self.X_train)
        y = np.array(self.y_train)

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
