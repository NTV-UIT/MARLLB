"""
Unit Tests for Reservoir Sampling Implementation

Tests cover:
- Algorithm correctness (uniform sampling)
- Feature extraction accuracy
- Edge cases and boundary conditions
- Performance characteristics

Author: MARLLB Implementation Team
Date: December 2025
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reservoir import ReservoirSampler, MultiMetricReservoir


class TestReservoirSampler(unittest.TestCase):
    """Test cases for ReservoirSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.capacity = 128
        self.sampler = ReservoirSampler(capacity=self.capacity, seed=42)
    
    def test_initialization(self):
        """Test proper initialization of reservoir."""
        self.assertEqual(self.sampler.capacity, self.capacity)
        self.assertEqual(self.sampler.count, 0)
        self.assertEqual(len(self.sampler), 0)
        self.assertFalse(self.sampler.is_full())
    
    def test_initial_fill(self):
        """Test that first k elements are always added."""
        for i in range(self.capacity):
            result = self.sampler.add(float(i))
            self.assertTrue(result, f"Element {i} should be added")
        
        self.assertEqual(len(self.sampler), self.capacity)
        self.assertTrue(self.sampler.is_full())
    
    def test_probabilistic_replacement(self):
        """Test probabilistic replacement after reservoir is full."""
        # Fill reservoir
        for i in range(self.capacity):
            self.sampler.add(float(i))
        
        initial_count = self.sampler.count
        self.assertEqual(initial_count, self.capacity)
        
        # Add more elements
        total = 1000
        for i in range(total):
            self.sampler.add(float(self.capacity + i))
        
        # Count should increase
        final_count = self.sampler.count
        self.assertEqual(final_count, self.capacity + total)
        
        # Size should stay at capacity
        self.assertEqual(len(self.sampler), self.capacity)
        
        # Reservoir should still be full
        self.assertTrue(self.sampler.is_full())
    
    def test_empty_reservoir_features(self):
        """Test feature extraction from empty reservoir."""
        features = self.sampler.get_features()
        
        for key, value in features.items():
            self.assertEqual(value, 0.0, f"Empty reservoir should have {key}=0")
    
    def test_basic_statistics(self):
        """Test basic statistical feature computation."""
        # Add known values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            self.sampler.add(v)
        
        features = self.sampler.get_features()
        
        # Check mean
        self.assertAlmostEqual(features['mean'], 3.0, places=5)
        
        # Check std
        expected_std = np.std(values)
        self.assertAlmostEqual(features['std'], expected_std, places=5)
    
    def test_percentile_computation(self):
        """Test 90th percentile computation."""
        # Add 100 values: 0, 1, 2, ..., 99
        for i in range(100):
            self.sampler.add(float(i))
        
        features = self.sampler.get_features()
        
        # 90th percentile of 0-99 should be around 89-90
        self.assertGreater(features['p90'], 85)
        self.assertLess(features['p90'], 95)
    
    def test_decay_weighting(self):
        """Test decay-weighted statistics."""
        base_time = 0.0
        
        # Add old samples with value 1.0
        for i in range(64):
            self.sampler.add(1.0, timestamp=base_time)
        
        # Add recent samples with value 10.0
        for i in range(64):
            self.sampler.add(10.0, timestamp=base_time + 100)
        
        # Compute features at current_time = base_time + 100
        features = self.sampler.get_features(
            decay_factor=0.9,
            current_time=base_time + 100
        )
        
        # Regular mean should be ~5.5
        self.assertAlmostEqual(features['mean'], 5.5, delta=0.5)
        
        # Decay-weighted mean should be closer to 10.0 (recent values)
        self.assertGreater(features['mean_decay'], 7.0)
        self.assertLess(features['mean_decay'], 10.0)
    
    def test_reset(self):
        """Test reservoir reset."""
        # Add some values
        for i in range(50):
            self.sampler.add(float(i))
        
        self.assertEqual(len(self.sampler), 50)
        
        # Reset
        self.sampler.reset()
        
        self.assertEqual(len(self.sampler), 0)
        self.assertEqual(self.sampler.count, 0)
        self.assertFalse(self.sampler.is_full())
    
    def test_get_samples(self):
        """Test retrieval of raw samples."""
        values_in = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values_in:
            self.sampler.add(v)
        
        values_out, timestamps = self.sampler.get_samples()
        
        self.assertEqual(len(values_out), 5)
        self.assertEqual(len(timestamps), 5)
        
        # Check that all values are present (order may differ)
        self.assertSetEqual(set(values_out), set(values_in))
    
    def test_feature_vector_shape(self):
        """Test feature vector has correct shape."""
        for i in range(10):
            self.sampler.add(float(i))
        
        feature_vec = self.sampler.get_feature_vector()
        
        self.assertEqual(feature_vec.shape, (5,))
        self.assertEqual(feature_vec.dtype, np.float32)


class TestMultiMetricReservoir(unittest.TestCase):
    """Test cases for MultiMetricReservoir class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = ['fct', 'flow_duration']
        self.multi = MultiMetricReservoir(metrics=self.metrics, capacity=128, seed=42)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(len(self.multi.reservoirs), 2)
        self.assertIn('fct', self.multi.reservoirs)
        self.assertIn('flow_duration', self.multi.reservoirs)
    
    def test_add_to_metric(self):
        """Test adding values to specific metrics."""
        self.multi.add('fct', 0.1)
        self.multi.add('flow_duration', 1.5)
        
        self.assertEqual(len(self.multi.reservoirs['fct']), 1)
        self.assertEqual(len(self.multi.reservoirs['flow_duration']), 1)
    
    def test_invalid_metric(self):
        """Test error handling for invalid metric name."""
        with self.assertRaises(ValueError):
            self.multi.add('invalid_metric', 1.0)
    
    def test_get_all_features(self):
        """Test retrieval of features for all metrics."""
        # Add some data
        for i in range(10):
            self.multi.add('fct', np.random.rand())
            self.multi.add('flow_duration', np.random.rand())
        
        all_features = self.multi.get_all_features()
        
        self.assertEqual(len(all_features), 2)
        self.assertIn('fct', all_features)
        self.assertIn('flow_duration', all_features)
        
        # Each metric should have 5 features
        for metric_features in all_features.values():
            self.assertEqual(len(metric_features), 5)
    
    def test_feature_vector_concatenation(self):
        """Test feature vector concatenation across metrics."""
        for i in range(10):
            self.multi.add('fct', float(i))
            self.multi.add('flow_duration', float(i * 2))
        
        feature_vec = self.multi.get_feature_vector()
        
        # Should have 2 metrics × 5 features = 10 dimensions
        self.assertEqual(feature_vec.shape, (10,))
    
    def test_reset_all(self):
        """Test resetting all reservoirs."""
        for i in range(50):
            self.multi.add('fct', float(i))
            self.multi.add('flow_duration', float(i))
        
        self.multi.reset()
        
        for reservoir in self.multi.reservoirs.values():
            self.assertEqual(len(reservoir), 0)


class TestStatisticalProperties(unittest.TestCase):
    """Test statistical properties of reservoir sampling."""
    
    def test_uniform_sampling_property(self):
        """
        Test that reservoir sampling produces uniform distribution.
        
        This is a statistical test: we sample from a known distribution
        and verify that each element has equal probability of being selected.
        """
        np.random.seed(42)
        
        # Create stream of 1000 distinct elements
        stream_size = 1000
        capacity = 100
        
        # Track how many times each element is selected
        selection_counts = np.zeros(stream_size)
        
        # Run multiple trials
        num_trials = 500
        for trial in range(num_trials):
            sampler = ReservoirSampler(capacity=capacity, seed=trial)
            
            # Add all elements
            for i in range(stream_size):
                sampler.add(float(i))
            
            # Check which elements are in reservoir
            samples, _ = sampler.get_samples()
            for sample in samples:
                selection_counts[int(sample)] += 1
        
        # Expected count for each element: num_trials * capacity / stream_size
        expected_count = num_trials * capacity / stream_size
        
        # Chi-square test for uniformity
        chi_square = np.sum((selection_counts - expected_count) ** 2 / expected_count)
        
        # Degrees of freedom: stream_size - 1
        dof = stream_size - 1
        
        # Critical value at alpha=0.05 for dof=999 is approximately 1073
        # (from chi-square table)
        critical_value = 1100  # Conservative threshold
        
        self.assertLess(chi_square, critical_value,
                       f"Chi-square {chi_square} exceeds critical value {critical_value}")
    
    def test_sample_mean_convergence(self):
        """
        Test that sample mean converges to population mean.
        
        By the law of large numbers, the sample mean should be close
        to the true mean for large enough sample size.
        """
        np.random.seed(42)
        
        # True distribution: exponential with mean 1.0
        true_mean = 1.0
        stream_size = 10000
        capacity = 128
        
        sampler = ReservoirSampler(capacity=capacity, seed=42)
        
        # Generate stream
        for i in range(stream_size):
            value = np.random.exponential(scale=true_mean)
            sampler.add(value)
        
        features = sampler.get_features()
        sample_mean = features['mean']
        
        # Sample mean should be within 20% of true mean
        relative_error = abs(sample_mean - true_mean) / true_mean
        self.assertLess(relative_error, 0.2,
                       f"Sample mean {sample_mean} too far from true mean {true_mean}")


def run_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestReservoirSampler))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiMetricReservoir))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalProperties))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
