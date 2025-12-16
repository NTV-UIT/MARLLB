"""
Reward Functions for Load Balancing

This module implements various fairness metrics used as reward functions
in the reinforcement learning environment for load balancing.

Available Metrics:
- Jain's Fairness Index
- Variance-based fairness
- Max-min fairness
- Product fairness (Nash welfare)
- Coefficient of Variation

Reference: MARLLB paper Section 3.1
"""

import numpy as np
from typing import List, Union, Optional


def jain_fairness(values: Union[List[float], np.ndarray], epsilon: float = 1e-10) -> float:
    """
    Compute Jain's Fairness Index.
    
    Jain's index measures fairness of resource allocation:
    J(x) = (Σx_i)² / (n * Σx_i²)
    
    Properties:
    - Range: [1/n, 1] where n is number of values
    - J = 1: Perfect fairness (all values equal)
    - J = 1/n: Worst case (all resources to one entity)
    - Scale-independent
    
    Args:
        values: Array of load values (e.g., flow durations per server)
        epsilon: Small value to avoid division by zero
    
    Returns:
        Jain's fairness index in [1/n, 1]
    
    Example:
        >>> jain_fairness([10, 10, 10, 10])
        1.0
        >>> jain_fairness([40, 0, 0, 0])
        0.25
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 1.0
    
    # Handle all-zero case
    if np.sum(values) < epsilon:
        return 1.0  # No load, considered fair
    
    sum_values = np.sum(values)
    sum_squared = np.sum(values ** 2)
    
    # Avoid division by zero
    if sum_squared < epsilon:
        return 1.0
    
    n = len(values)
    jain_index = (sum_values ** 2) / (n * sum_squared)
    
    # Clip to valid range (numerical stability)
    return np.clip(jain_index, 1.0 / n, 1.0)


def variance_fairness(values: Union[List[float], np.ndarray]) -> float:
    """
    Compute variance-based fairness metric.
    
    Returns negative variance (to maximize fairness).
    Lower variance = more fair = higher reward.
    
    Args:
        values: Array of load values
    
    Returns:
        Negative variance (-∞, 0]
    
    Example:
        >>> variance_fairness([10, 10, 10, 10])
        -0.0
        >>> variance_fairness([40, 0, 0, 0])
        -300.0
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0
    
    return -np.var(values)


def std_fairness(values: Union[List[float], np.ndarray]) -> float:
    """
    Compute standard deviation-based fairness metric.
    
    Returns negative standard deviation (less sensitive to outliers than variance).
    
    Args:
        values: Array of load values
    
    Returns:
        Negative standard deviation (-∞, 0]
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0
    
    return -np.std(values)


def coefficient_of_variation(values: Union[List[float], np.ndarray], epsilon: float = 1e-10) -> float:
    """
    Compute coefficient of variation (CV) fairness metric.
    
    CV = std / mean (scale-independent)
    Returns negative CV to maximize fairness.
    
    Args:
        values: Array of load values
        epsilon: Small value to avoid division by zero
    
    Returns:
        Negative coefficient of variation (-∞, 0]
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0
    
    mean = np.mean(values)
    
    if mean < epsilon:
        return 0.0  # No load
    
    std = np.std(values)
    cv = std / (mean + epsilon)
    
    return -cv


def max_min_fairness(values: Union[List[float], np.ndarray]) -> float:
    """
    Compute max-min fairness metric.
    
    Minimizes the maximum load (avoid overload).
    Returns negative max to maximize fairness.
    
    Args:
        values: Array of load values
    
    Returns:
        Negative maximum value (-∞, 0]
    
    Example:
        >>> max_min_fairness([10, 10, 10, 10])
        -10.0
        >>> max_min_fairness([40, 0, 0, 0])
        -40.0
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0
    
    return -np.max(values)


def min_max_fairness(values: Union[List[float], np.ndarray]) -> float:
    """
    Compute min-max fairness metric.
    
    Maximizes the minimum load (ensure no server is idle).
    
    Args:
        values: Array of load values
    
    Returns:
        Minimum value [0, ∞)
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0
    
    return np.min(values)


def product_fairness(values: Union[List[float], np.ndarray], epsilon: float = 1e-10) -> float:
    """
    Compute product fairness (Nash welfare).
    
    Maximize product of allocations:
    W(x) = Π(x_i + ε)
    
    Use log transform for numerical stability:
    log W = Σ log(x_i + ε)
    
    Args:
        values: Array of load values
        epsilon: Small value to avoid log(0)
    
    Returns:
        Log product of values
    
    Example:
        >>> product_fairness([10, 10, 10, 10])
        9.21  # 4 * log(10)
        >>> product_fairness([40, 10, 10, 10])
        8.66  # log(40) + 3*log(10)
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0
    
    # Add epsilon to avoid log(0)
    log_sum = np.sum(np.log(values + epsilon))
    
    return log_sum


def range_fairness(values: Union[List[float], np.ndarray]) -> float:
    """
    Compute range-based fairness metric.
    
    Returns negative range (max - min).
    Smaller range = more fair.
    
    Args:
        values: Array of load values
    
    Returns:
        Negative range (-∞, 0]
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0
    
    return -(np.max(values) - np.min(values))


def gini_coefficient(values: Union[List[float], np.ndarray]) -> float:
    """
    Compute Gini coefficient (income inequality metric).
    
    Gini = (Σ Σ |x_i - x_j|) / (2n² * mean(x))
    
    Properties:
    - Range: [0, 1]
    - 0: Perfect equality
    - 1: Perfect inequality
    
    Returns negative Gini to maximize fairness.
    
    Args:
        values: Array of load values
    
    Returns:
        Negative Gini coefficient [-1, 0]
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0
    
    n = len(values)
    mean = np.mean(values)
    
    if mean == 0:
        return 0.0
    
    # Compute pairwise differences
    diff_sum = 0.0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(values[i] - values[j])
    
    gini = diff_sum / (2 * n * n * mean)
    
    return -gini


class RewardFunction:
    """
    Configurable reward function for load balancing.
    
    Supports multiple fairness metrics and reward fields.
    """
    
    SUPPORTED_METRICS = {
        'jain': jain_fairness,
        'variance': variance_fairness,
        'std': std_fairness,
        'cv': coefficient_of_variation,
        'max': max_min_fairness,
        'min': min_max_fairness,
        'product': product_fairness,
        'range': range_fairness,
        'gini': gini_coefficient
    }
    
    def __init__(self, metric: str = 'jain', reward_field: str = 'flow_duration_avg_decay'):
        """
        Initialize reward function.
        
        Args:
            metric: Fairness metric name ('jain', 'variance', 'max', etc.)
            reward_field: Feature field to compute reward on
                         (e.g., 'flow_duration_avg_decay', 'fct_mean', 'n_flow_on')
        
        Raises:
            ValueError: If metric is not supported
        """
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric: {metric}. "
                           f"Supported: {list(self.SUPPORTED_METRICS.keys())}")
        
        self.metric = metric
        self.reward_field = reward_field
        self._compute_func = self.SUPPORTED_METRICS[metric]
    
    def compute(self, observations: dict) -> float:
        """
        Compute reward from observations.
        
        Args:
            observations: Dictionary with structure:
                {
                    'active_servers': [0, 1, 2, 3],
                    'server_stats': {
                        0: {'flow_duration_avg_decay': 10.5, ...},
                        1: {'flow_duration_avg_decay': 12.3, ...},
                        ...
                    }
                }
        
        Returns:
            Reward value (higher = better fairness)
        
        Example:
            >>> reward_fn = RewardFunction(metric='jain', reward_field='fct_mean')
            >>> obs = {
            ...     'active_servers': [0, 1, 2, 3],
            ...     'server_stats': {
            ...         0: {'fct_mean': 10},
            ...         1: {'fct_mean': 12},
            ...         2: {'fct_mean': 11},
            ...         3: {'fct_mean': 10}
            ...     }
            ... }
            >>> reward_fn.compute(obs)
            0.99
        """
        active_servers = observations.get('active_servers', [])
        server_stats = observations.get('server_stats', {})
        
        if not active_servers:
            return 0.0  # No active servers
        
        # Extract field values for active servers
        values = []
        for server_id in active_servers:
            if server_id in server_stats:
                stat = server_stats[server_id]
                if self.reward_field in stat:
                    values.append(stat[self.reward_field])
        
        if not values:
            return 0.0  # No valid data
        
        # Compute fairness metric
        reward = self._compute_func(values)
        
        return reward
    
    def __call__(self, observations: dict) -> float:
        """Allow using instance as function."""
        return self.compute(observations)
    
    def __repr__(self):
        return f"RewardFunction(metric='{self.metric}', reward_field='{self.reward_field}')"


# Convenience functions for common configurations
def create_jain_reward(field: str = 'flow_duration_avg_decay') -> RewardFunction:
    """Create Jain fairness reward (default for MARLLB)."""
    return RewardFunction(metric='jain', reward_field=field)


def create_variance_reward(field: str = 'flow_duration_avg_decay') -> RewardFunction:
    """Create variance-based reward."""
    return RewardFunction(metric='variance', reward_field=field)


def create_max_reward(field: str = 'flow_duration_avg_decay') -> RewardFunction:
    """Create max-min fairness reward."""
    return RewardFunction(metric='max', reward_field=field)


if __name__ == '__main__':
    # Demonstration
    print("Fairness Metrics Demonstration")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ([10, 10, 10, 10], "Perfect balance"),
        ([15, 10, 10, 5], "Slight imbalance"),
        ([25, 10, 10, 5], "Moderate imbalance"),
        ([40, 5, 5, 0], "Severe imbalance"),
        ([40, 0, 0, 0], "Worst case"),
    ]
    
    metrics = ['jain', 'variance', 'max', 'product']
    
    for values, description in test_cases:
        print(f"\n{description}: {values}")
        print("-" * 50)
        
        for metric in metrics:
            reward_fn = RewardFunction(metric=metric)
            
            # Create mock observations
            obs = {
                'active_servers': list(range(len(values))),
                'server_stats': {
                    i: {'flow_duration_avg_decay': v}
                    for i, v in enumerate(values)
                }
            }
            
            reward = reward_fn.compute(obs)
            print(f"{metric:>10}: {reward:8.4f}")
    
    print("\n" + "=" * 50)
    print("Jain Index range: [1/n, 1] = [0.25, 1] for n=4")
    print("Variance: (-∞, 0], lower = more fair")
    print("Max: (-∞, 0], higher = avoid overload")
    print("Product: (-∞, ∞), higher = more fair")
