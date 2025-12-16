"""
Basic Usage Example for Reservoir Sampling

This example demonstrates the basic API usage for collecting
flow statistics in a load balancing scenario.

Author: MARLLB Implementation Team
Date: December 2025
"""

import numpy as np
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reservoir import ReservoirSampler, MultiMetricReservoir


def example_single_metric():
    """Example: Single metric reservoir (FCT tracking)"""
    print("=" * 60)
    print("Example 1: Single Metric Reservoir (FCT)")
    print("=" * 60)
    
    # Create reservoir for Flow Completion Time
    fct_sampler = ReservoirSampler(capacity=128, seed=42)
    
    # Simulate incoming flows with exponential FCT distribution
    num_flows = 1000
    fct_values = np.random.exponential(scale=0.1, size=num_flows)
    
    print(f"\nSimulating {num_flows} flows...")
    print(f"True mean FCT: {np.mean(fct_values):.6f}s")
    print(f"True 90th percentile: {np.percentile(fct_values, 90):.6f}s\n")
    
    # Add flows to reservoir
    start_time = time.time()
    for fct in fct_values:
        fct_sampler.add(fct, timestamp=time.time())
    
    # Get statistics
    features = fct_sampler.get_features(decay_factor=0.9)
    
    print(f"Reservoir statistics (from {len(fct_sampler)} samples):")
    print(f"  Mean:       {features['mean']:.6f}s")
    print(f"  P90:        {features['p90']:.6f}s")
    print(f"  Std:        {features['std']:.6f}s")
    print(f"  Mean decay: {features['mean_decay']:.6f}s")
    print(f"  P90 decay:  {features['p90_decay']:.6f}s")
    
    # Compute accuracy
    relative_error = abs(features['mean'] - np.mean(fct_values)) / np.mean(fct_values)
    print(f"\nRelative error in mean: {relative_error * 100:.2f}%")


def example_multi_metric():
    """Example: Multi-metric reservoir (FCT + Duration)"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Metric Reservoir (FCT + Duration)")
    print("=" * 60)
    
    # Create multi-metric reservoir
    multi_reservoir = MultiMetricReservoir(
        metrics=['fct', 'flow_duration'],
        capacity=128,
        seed=42
    )
    
    # Simulate flows
    num_flows = 1000
    print(f"\nSimulating {num_flows} flows with 2 metrics...")
    
    for i in range(num_flows):
        # FCT: exponential distribution
        fct = np.random.exponential(scale=0.1)
        
        # Flow duration: uniform distribution
        duration = np.random.uniform(0.01, 1.0)
        
        # Add to respective reservoirs
        multi_reservoir.add('fct', fct)
        multi_reservoir.add('flow_duration', duration)
    
    # Get all features
    all_features = multi_reservoir.get_all_features()
    
    print("\nFeatures per metric:")
    for metric, features in all_features.items():
        print(f"\n{metric}:")
        for key, value in features.items():
            print(f"  {key:12s}: {value:.6f}")
    
    # Get feature vector for RL
    feature_vector = multi_reservoir.get_feature_vector()
    print(f"\nRL State vector shape: {feature_vector.shape}")
    print(f"RL State vector: {feature_vector}")


def example_temporal_dynamics():
    """Example: Temporal dynamics with decay weighting"""
    print("\n" + "=" * 60)
    print("Example 3: Temporal Dynamics (Workload Change)")
    print("=" * 60)
    
    reservoir = ReservoirSampler(capacity=128, seed=42)
    
    base_time = time.time()
    
    # Phase 1: Low load (FCT ~0.05s)
    print("\nPhase 1: Low load period (0-50s)")
    for i in range(500):
        fct = np.random.exponential(scale=0.05)
        timestamp = base_time + i * 0.1  # 100ms intervals
        reservoir.add(fct, timestamp=timestamp)
    
    phase1_features = reservoir.get_features(
        decay_factor=0.9,
        current_time=base_time + 50
    )
    print(f"  Mean: {phase1_features['mean']:.6f}s")
    print(f"  Mean decay: {phase1_features['mean_decay']:.6f}s")
    
    # Phase 2: High load (FCT ~0.20s)
    print("\nPhase 2: High load period (50-100s)")
    for i in range(500):
        fct = np.random.exponential(scale=0.20)
        timestamp = base_time + 50 + i * 0.1
        reservoir.add(fct, timestamp=timestamp)
    
    phase2_features = reservoir.get_features(
        decay_factor=0.9,
        current_time=base_time + 100
    )
    print(f"  Mean: {phase2_features['mean']:.6f}s")
    print(f"  Mean decay: {phase2_features['mean_decay']:.6f}s")
    
    print("\nObservation:")
    print(f"  Regular mean changed by: "
          f"{(phase2_features['mean'] - phase1_features['mean']) / phase1_features['mean'] * 100:.1f}%")
    print(f"  Decay-weighted mean changed by: "
          f"{(phase2_features['mean_decay'] - phase1_features['mean_decay']) / phase1_features['mean_decay'] * 100:.1f}%")
    print("  → Decay-weighted metric responds faster to workload changes!")


def example_per_server_state():
    """Example: Per-server state for load balancing"""
    print("\n" + "=" * 60)
    print("Example 4: Per-Server State (4 Application Servers)")
    print("=" * 60)
    
    num_servers = 4
    
    # Create reservoir for each server
    server_reservoirs = []
    for i in range(num_servers):
        multi = MultiMetricReservoir(
            metrics=['fct', 'flow_duration'],
            capacity=128,
            seed=i
        )
        server_reservoirs.append(multi)
    
    # Simulate different load patterns on each server
    print("\nSimulating different loads on servers...")
    
    loads = {
        0: {'scale': 0.05, 'flows': 100},  # Light load
        1: {'scale': 0.10, 'flows': 150},  # Medium load
        2: {'scale': 0.15, 'flows': 200},  # Heavy load
        3: {'scale': 0.08, 'flows': 120},  # Medium-light load
    }
    
    for server_id, config in loads.items():
        for _ in range(config['flows']):
            fct = np.random.exponential(scale=config['scale'])
            duration = np.random.uniform(0.01, 1.0)
            
            server_reservoirs[server_id].add('fct', fct)
            server_reservoirs[server_id].add('flow_duration', duration)
    
    # Construct RL state
    print("\nServer states (FCT mean):")
    state_vectors = []
    
    for server_id in range(num_servers):
        features = server_reservoirs[server_id].get_all_features()
        fct_mean = features['fct']['mean']
        active_flows = loads[server_id]['flows']
        
        print(f"  Server {server_id}: FCT={fct_mean:.6f}s, Active flows={active_flows}")
        
        # Get feature vector (10 dims: 2 metrics × 5 features)
        vec = server_reservoirs[server_id].get_feature_vector()
        state_vectors.append(vec)
    
    # Stack into state matrix
    state_matrix = np.stack(state_vectors)
    print(f"\nRL State matrix shape: {state_matrix.shape}")
    print(f"(num_servers={num_servers}, features_per_server=10)")
    
    # Decision: Which server should receive next flow?
    fct_means = [state_vectors[i][0] for i in range(num_servers)]
    best_server = np.argmin(fct_means)
    print(f"\nRecommendation: Send next flow to Server {best_server} (lowest FCT)")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("MARLLB Reservoir Sampling - Usage Examples")
    print("=" * 60)
    
    example_single_metric()
    example_multi_metric()
    example_temporal_dynamics()
    example_per_server_state()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
