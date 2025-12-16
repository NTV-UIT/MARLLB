"""
Feature Engineering for Reservoir Samples

This module provides advanced feature extraction methods for reservoir samples,
including decay weighting and additional statistical measures.

Author: MARLLB Implementation Team
Date: December 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats


class FeatureExtractor:
    """
    Extract statistical features from reservoir samples for RL state.
    
    Features are designed to capture:
    - Central tendency (mean)
    - Tail behavior (90th percentile)
    - Variability (std)
    - Temporal dynamics (decay-weighted versions)
    """
    
    @staticmethod
    def extract_basic_stats(values: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features.
        
        Args:
            values: Array of sampled values
            
        Returns:
            Dictionary with mean, median, p90, std, min, max
        """
        if len(values) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'p90': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    @staticmethod
    def extract_decay_weighted_stats(values: np.ndarray,
                                     timestamps: np.ndarray,
                                     current_time: float,
                                     decay_factor: float = 0.9) -> Dict[str, float]:
        """
        Extract decay-weighted statistics.
        
        Recent samples have higher weights: w_i = decay_factor^(t_current - t_i)
        
        Args:
            values: Array of sampled values
            timestamps: Array of timestamps
            current_time: Current time
            decay_factor: Decay factor (0 < decay < 1)
            
        Returns:
            Dictionary with decay-weighted mean and p90
        """
        if len(values) == 0:
            return {
                'mean_decay': 0.0,
                'p90_decay': 0.0
            }
        
        # Compute weights
        time_diffs = current_time - timestamps
        weights = np.power(decay_factor, time_diffs)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Weighted mean
        mean_decay = np.sum(values * weights)
        
        # Weighted percentile
        p90_decay = FeatureExtractor._weighted_percentile(values, weights, 0.9)
        
        return {
            'mean_decay': float(mean_decay),
            'p90_decay': float(p90_decay)
        }
    
    @staticmethod
    def _weighted_percentile(values: np.ndarray,
                           weights: np.ndarray,
                           percentile: float) -> float:
        """
        Compute weighted percentile.
        
        Args:
            values: Array of values
            weights: Array of weights (normalized)
            percentile: Percentile to compute (0.0 to 1.0)
            
        Returns:
            Weighted percentile value
        """
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumsum = np.cumsum(sorted_weights)
        idx = np.searchsorted(cumsum, percentile)
        
        if idx >= len(sorted_values):
            idx = len(sorted_values) - 1
        
        return sorted_values[idx]
    
    @staticmethod
    def extract_distribution_features(values: np.ndarray) -> Dict[str, float]:
        """
        Extract features describing the distribution shape.
        
        Args:
            values: Array of sampled values
            
        Returns:
            Dictionary with skewness, kurtosis, cv
        """
        if len(values) < 3:
            return {
                'skewness': 0.0,
                'kurtosis': 0.0,
                'cv': 0.0
            }
        
        # Skewness: measure of asymmetry
        skewness = float(stats.skew(values))
        
        # Kurtosis: measure of tail heaviness
        kurtosis = float(stats.kurtosis(values))
        
        # Coefficient of variation: std/mean
        mean = np.mean(values)
        cv = float(np.std(values) / mean) if mean > 0 else 0.0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'cv': cv
        }
    
    @staticmethod
    def extract_all_features(values: np.ndarray,
                           timestamps: np.ndarray,
                           current_time: float,
                           decay_factor: float = 0.9) -> Dict[str, float]:
        """
        Extract all available features.
        
        Args:
            values: Array of sampled values
            timestamps: Array of timestamps
            current_time: Current time
            decay_factor: Decay factor for temporal weighting
            
        Returns:
            Dictionary with all features
        """
        features = {}
        
        # Basic stats
        features.update(FeatureExtractor.extract_basic_stats(values))
        
        # Decay-weighted stats
        features.update(FeatureExtractor.extract_decay_weighted_stats(
            values, timestamps, current_time, decay_factor
        ))
        
        # Distribution features
        features.update(FeatureExtractor.extract_distribution_features(values))
        
        return features
    
    @staticmethod
    def extract_marllb_features(values: np.ndarray,
                               timestamps: np.ndarray,
                               current_time: float,
                               decay_factor: float = 0.9) -> np.ndarray:
        """
        Extract the 5 features used in MARLLB paper.
        
        This is the standard feature set used as RL state:
        [mean, p90, std, mean_decay, p90_decay]
        
        Args:
            values: Array of sampled values
            timestamps: Array of timestamps
            current_time: Current time
            decay_factor: Decay factor
            
        Returns:
            Feature vector of shape (5,)
        """
        if len(values) == 0:
            return np.zeros(5, dtype=np.float32)
        
        # Basic stats
        mean = np.mean(values)
        p90 = np.percentile(values, 90)
        std = np.std(values)
        
        # Decay-weighted stats
        time_diffs = current_time - timestamps
        weights = np.power(decay_factor, time_diffs)
        weights = weights / np.sum(weights)
        
        mean_decay = np.sum(values * weights)
        p90_decay = FeatureExtractor._weighted_percentile(values, weights, 0.9)
        
        return np.array([mean, p90, std, mean_decay, p90_decay], dtype=np.float32)


class PerServerFeatures:
    """
    Manage features for multiple servers in load balancing scenario.
    
    Each server has:
    - 1 counter feature: n_flow_on (active flows)
    - 10 reservoir features: 2 metrics × 5 features each
    Total: 11 features per server
    """
    
    def __init__(self, num_servers: int):
        """
        Initialize per-server feature manager.
        
        Args:
            num_servers: Number of application servers
        """
        self.num_servers = num_servers
        self.n_flow_on = np.zeros(num_servers, dtype=np.int32)
        
    def update_flow_count(self, server_id: int, count: int):
        """Update active flow count for a server."""
        self.n_flow_on[server_id] = count
    
    def get_state_vector(self,
                        reservoir_features: List[np.ndarray],
                        active_servers: Optional[List[int]] = None) -> np.ndarray:
        """
        Construct full state vector for RL agent.
        
        Args:
            reservoir_features: List of feature arrays, one per server
                               Each array has shape (10,) = 2 metrics × 5 features
            active_servers: List of active server IDs (optional)
            
        Returns:
            State array of shape (num_servers, 11)
        """
        state = np.zeros((self.num_servers, 11), dtype=np.float32)
        
        for i in range(self.num_servers):
            # Counter feature
            state[i, 0] = self.n_flow_on[i]
            
            # Reservoir features
            if i < len(reservoir_features):
                state[i, 1:] = reservoir_features[i]
        
        # Mask inactive servers if specified
        if active_servers is not None:
            mask = np.zeros(self.num_servers, dtype=bool)
            mask[active_servers] = True
            state[~mask] = 0
        
        return state
    
    def get_server_features(self, server_id: int,
                          reservoir_features: np.ndarray) -> np.ndarray:
        """
        Get feature vector for a specific server.
        
        Args:
            server_id: Server ID
            reservoir_features: Feature array of shape (10,)
            
        Returns:
            Feature vector of shape (11,)
        """
        features = np.zeros(11, dtype=np.float32)
        features[0] = self.n_flow_on[server_id]
        features[1:] = reservoir_features
        return features


def normalize_features(features: np.ndarray,
                      mean: Optional[np.ndarray] = None,
                      std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Feature array of shape (num_samples, num_features)
        mean: Pre-computed mean (optional)
        std: Pre-computed std (optional)
        
    Returns:
        Tuple of (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
    
    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)
    
    normalized = (features - mean) / std
    
    return normalized, mean, std


if __name__ == "__main__":
    # Demonstration
    print("=== Feature Extraction Demo ===\n")
    
    # Generate synthetic data
    np.random.seed(42)
    values = np.random.exponential(scale=0.1, size=128)
    timestamps = np.linspace(0, 100, 128)
    current_time = 100.0
    
    # Extract all features
    all_features = FeatureExtractor.extract_all_features(
        values, timestamps, current_time, decay_factor=0.9
    )
    
    print("All features:")
    for key, value in all_features.items():
        print(f"  {key:15s}: {value:.6f}")
    
    # Extract MARLLB features
    marllb_features = FeatureExtractor.extract_marllb_features(
        values, timestamps, current_time
    )
    
    print(f"\nMARLLB feature vector: {marllb_features}")
    print(f"Shape: {marllb_features.shape}")
    
    # Per-server features
    print("\n=== Per-Server Features Demo ===\n")
    
    num_servers = 4
    per_server = PerServerFeatures(num_servers)
    
    # Simulate server states
    for i in range(num_servers):
        per_server.update_flow_count(i, np.random.randint(10, 100))
    
    # Generate reservoir features for each server
    reservoir_features_list = []
    for i in range(num_servers):
        fct_features = np.random.rand(5)
        duration_features = np.random.rand(5)
        combined = np.concatenate([fct_features, duration_features])
        reservoir_features_list.append(combined)
    
    # Get full state
    state = per_server.get_state_vector(reservoir_features_list)
    
    print(f"State shape: {state.shape}")
    print("State per server:")
    for i in range(num_servers):
        print(f"  Server {i}: n_flow={state[i, 0]:.0f}, features={state[i, 1:4]}")
