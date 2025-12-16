"""
Reservoir Sampling Implementation for MARLLB

This module implements Algorithm R (Vitter, 1985) for maintaining a fixed-size
random sample from a stream of data. Used for collecting flow statistics in
the VPP load balancer plugin.

Author: MARLLB Implementation Team
Date: December 2025
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import time


class ReservoirSampler:
    """
    Reservoir Sampler maintains a fixed-size random sample from a stream.
    
    This implementation uses Algorithm R for O(1) amortized insertion time
    and provides feature extraction methods for RL state representation.
    
    Attributes:
        capacity (int): Maximum number of samples to maintain (default: 128)
        count (int): Total number of elements seen so far
        values (np.ndarray): Array of sampled values
        timestamps (np.ndarray): Timestamps of samples (for decay weighting)
    """
    
    def __init__(self, capacity: int = 128, seed: Optional[int] = None):
        """
        Initialize reservoir sampler.
        
        Args:
            capacity: Maximum size of reservoir (k in Algorithm R)
            seed: Random seed for reproducibility (optional)
        """
        self.capacity = capacity
        self.count = 0
        self.values = np.zeros(capacity, dtype=np.float32)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Internal state
        self._is_full = False
        
    def add(self, value: float, timestamp: Optional[float] = None) -> bool:
        """
        Add a new value to the reservoir using Algorithm R.
        
        Args:
            value: The value to add (e.g., FCT, flow duration)
            timestamp: Unix timestamp in seconds (default: current time)
            
        Returns:
            True if value was added to reservoir, False if rejected
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Algorithm R implementation
        if self.count < self.capacity:
            # Fill reservoir initially
            self.values[self.count] = value
            self.timestamps[self.count] = timestamp
            self.count += 1
            
            if self.count == self.capacity:
                self._is_full = True
            return True
        else:
            # Reservoir is full - probabilistic replacement
            j = self.rng.randint(0, self.count + 1)
            
            if j < self.capacity:
                self.values[j] = value
                self.timestamps[j] = timestamp
                self.count += 1
                return True
            else:
                self.count += 1
                return False
    
    def get_size(self) -> int:
        """Get current number of samples in reservoir."""
        return min(self.count, self.capacity)
    
    def is_full(self) -> bool:
        """Check if reservoir has reached capacity."""
        return self._is_full
    
    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all samples currently in reservoir.
        
        Returns:
            Tuple of (values, timestamps) as numpy arrays
        """
        size = self.get_size()
        return self.values[:size].copy(), self.timestamps[:size].copy()
    
    def get_features(self, 
                     decay_factor: float = 0.9,
                     current_time: Optional[float] = None) -> Dict[str, float]:
        """
        Compute statistical features from reservoir samples.
        
        This computes 5 features used as RL state:
        - mean: Average value
        - p90: 90th percentile (tail latency)
        - std: Standard deviation
        - mean_decay: Decay-weighted average
        - p90_decay: Decay-weighted 90th percentile
        
        Args:
            decay_factor: Exponential decay factor (default: 0.9)
            current_time: Current timestamp (default: time.time())
            
        Returns:
            Dictionary with 5 feature values
        """
        size = self.get_size()
        
        if size == 0:
            return {
                'mean': 0.0,
                'p90': 0.0,
                'std': 0.0,
                'mean_decay': 0.0,
                'p90_decay': 0.0
            }
        
        values = self.values[:size]
        timestamps = self.timestamps[:size]
        
        if current_time is None:
            current_time = time.time()
        
        # Basic statistics
        mean = np.mean(values)
        p90 = np.percentile(values, 90)
        std = np.std(values)
        
        # Decay-weighted statistics
        time_diffs = current_time - timestamps
        weights = np.power(decay_factor, time_diffs)
        
        # Weighted mean
        mean_decay = np.average(values, weights=weights)
        
        # Weighted percentile (more complex)
        p90_decay = self._weighted_percentile(values, weights, 0.9)
        
        return {
            'mean': float(mean),
            'p90': float(p90),
            'std': float(std),
            'mean_decay': float(mean_decay),
            'p90_decay': float(p90_decay)
        }
    
    def _weighted_percentile(self, 
                            values: np.ndarray,
                            weights: np.ndarray,
                            percentile: float) -> float:
        """
        Compute weighted percentile.
        
        Args:
            values: Array of values
            weights: Array of weights (same shape as values)
            percentile: Percentile to compute (0.0 to 1.0)
            
        Returns:
            Weighted percentile value
        """
        # Sort by values
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Cumulative sum of weights
        cumsum = np.cumsum(sorted_weights)
        cutoff = percentile * cumsum[-1]
        
        # Find index where cumsum exceeds cutoff
        idx = np.searchsorted(cumsum, cutoff)
        
        # Handle edge cases
        if idx >= len(sorted_values):
            idx = len(sorted_values) - 1
        
        return sorted_values[idx]
    
    def get_feature_vector(self,
                          decay_factor: float = 0.9,
                          current_time: Optional[float] = None) -> np.ndarray:
        """
        Get features as a numpy vector (convenient for RL state).
        
        Args:
            decay_factor: Exponential decay factor
            current_time: Current timestamp
            
        Returns:
            Feature vector of shape (5,)
        """
        features = self.get_features(decay_factor, current_time)
        return np.array([
            features['mean'],
            features['p90'],
            features['std'],
            features['mean_decay'],
            features['p90_decay']
        ], dtype=np.float32)
    
    def reset(self):
        """Clear all samples and reset counter."""
        self.count = 0
        self._is_full = False
        self.values.fill(0)
        self.timestamps.fill(0)
    
    def __len__(self) -> int:
        """Return current number of samples."""
        return self.get_size()
    
    def __repr__(self) -> str:
        return (f"ReservoirSampler(capacity={self.capacity}, "
                f"count={self.count}, size={self.get_size()})")


class MultiMetricReservoir:
    """
    Maintains multiple reservoir samplers for different metrics.
    
    This is used in MARLLB where each server tracks both FCT and flow_duration.
    """
    
    def __init__(self, 
                 metrics: List[str] = None,
                 capacity: int = 128,
                 seed: Optional[int] = None):
        """
        Initialize multi-metric reservoir.
        
        Args:
            metrics: List of metric names (default: ['fct', 'flow_duration'])
            capacity: Reservoir capacity for each metric
            seed: Random seed
        """
        if metrics is None:
            metrics = ['fct', 'flow_duration']
        
        self.metrics = metrics
        self.reservoirs = {}
        
        for metric in metrics:
            self.reservoirs[metric] = ReservoirSampler(
                capacity=capacity,
                seed=seed
            )
    
    def add(self, metric: str, value: float, timestamp: Optional[float] = None):
        """
        Add value to specific metric reservoir.
        
        Args:
            metric: Metric name (e.g., 'fct', 'flow_duration')
            value: Value to add
            timestamp: Timestamp (optional)
        """
        if metric not in self.reservoirs:
            raise ValueError(f"Unknown metric: {metric}")
        
        return self.reservoirs[metric].add(value, timestamp)
    
    def get_all_features(self,
                        decay_factor: float = 0.9,
                        current_time: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Get features for all metrics.
        
        Returns:
            Dictionary mapping metric name -> feature dictionary
        """
        result = {}
        for metric, reservoir in self.reservoirs.items():
            result[metric] = reservoir.get_features(decay_factor, current_time)
        return result
    
    def get_feature_vector(self,
                          decay_factor: float = 0.9,
                          current_time: Optional[float] = None) -> np.ndarray:
        """
        Get concatenated feature vector for all metrics.
        
        Returns:
            Feature vector of shape (num_metrics * 5,)
        """
        vectors = []
        for metric in self.metrics:
            vec = self.reservoirs[metric].get_feature_vector(decay_factor, current_time)
            vectors.append(vec)
        return np.concatenate(vectors)
    
    def reset(self):
        """Reset all reservoirs."""
        for reservoir in self.reservoirs.values():
            reservoir.reset()
    
    def __repr__(self) -> str:
        return f"MultiMetricReservoir(metrics={self.metrics})"


if __name__ == "__main__":
    # Simple demonstration
    print("=== Reservoir Sampling Demo ===\n")
    
    # Create reservoir
    reservoir = ReservoirSampler(capacity=128, seed=42)
    
    # Simulate stream of FCT values
    print("Adding 1000 samples to reservoir...")
    for i in range(1000):
        fct = np.random.exponential(scale=0.1)  # Simulate FCT distribution
        reservoir.add(fct, timestamp=time.time())
    
    print(f"Reservoir state: {reservoir}")
    print(f"Filled: {reservoir.is_full()}\n")
    
    # Get features
    features = reservoir.get_features()
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n=== Multi-Metric Demo ===\n")
    
    # Multi-metric reservoir
    multi_reservoir = MultiMetricReservoir(metrics=['fct', 'flow_duration'])
    
    for i in range(1000):
        fct = np.random.exponential(scale=0.1)
        duration = np.random.uniform(0.01, 1.0)
        
        multi_reservoir.add('fct', fct)
        multi_reservoir.add('flow_duration', duration)
    
    all_features = multi_reservoir.get_all_features()
    print("All features:")
    for metric, features in all_features.items():
        print(f"\n{metric}:")
        for key, value in features.items():
            print(f"  {key}: {value:.6f}")
    
    feature_vector = multi_reservoir.get_feature_vector()
    print(f"\nFeature vector shape: {feature_vector.shape}")
    print(f"Feature vector: {feature_vector}")
