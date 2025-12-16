"""
Unit tests for reward functions

Tests all fairness metrics to ensure correctness.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rewards import (
    jain_fairness,
    variance_fairness,
    std_fairness,
    coefficient_of_variation,
    max_min_fairness,
    min_max_fairness,
    product_fairness,
    range_fairness,
    gini_coefficient,
    RewardFunction
)


class TestFairnessMetrics(unittest.TestCase):
    """Test individual fairness metric functions."""
    
    def test_jain_fairness_perfect(self):
        """Test Jain index for perfect fairness."""
        values = [10, 10, 10, 10]
        result = jain_fairness(values)
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_jain_fairness_worst(self):
        """Test Jain index for worst case (all load on one)."""
        values = [40, 0, 0, 0]
        result = jain_fairness(values)
        self.assertAlmostEqual(result, 0.25, places=5)  # 1/4
    
    def test_jain_fairness_moderate(self):
        """Test Jain index for moderate imbalance."""
        values = [15, 10, 10, 5]
        result = jain_fairness(values)
        # (40^2) / (4 * (225+100+100+25)) = 1600 / 1800 = 0.889
        self.assertAlmostEqual(result, 0.889, places=3)
    
    def test_jain_fairness_empty(self):
        """Test Jain index for empty values."""
        values = []
        result = jain_fairness(values)
        self.assertEqual(result, 1.0)
    
    def test_jain_fairness_zeros(self):
        """Test Jain index for all zeros."""
        values = [0, 0, 0, 0]
        result = jain_fairness(values)
        self.assertEqual(result, 1.0)
    
    def test_jain_fairness_range(self):
        """Test that Jain index is in valid range [1/n, 1]."""
        test_cases = [
            [10, 10, 10, 10],
            [20, 10, 5, 5],
            [30, 5, 3, 2],
            [40, 0, 0, 0]
        ]
        
        for values in test_cases:
            n = len(values)
            result = jain_fairness(values)
            self.assertGreaterEqual(result, 1.0 / n)
            self.assertLessEqual(result, 1.0)
    
    def test_variance_fairness_perfect(self):
        """Test variance fairness for perfect balance."""
        values = [10, 10, 10, 10]
        result = variance_fairness(values)
        self.assertEqual(result, 0.0)
    
    def test_variance_fairness_imbalance(self):
        """Test variance fairness for imbalanced load."""
        values = [40, 0, 0, 0]
        result = variance_fairness(values)
        # var = (30^2 + 10^2 + 10^2 + 10^2) / 4 = 1200/4 = 300
        self.assertAlmostEqual(result, -300.0, places=1)
    
    def test_max_min_fairness(self):
        """Test max-min fairness metric."""
        values = [40, 10, 10, 5]
        result = max_min_fairness(values)
        self.assertEqual(result, -40)
    
    def test_min_max_fairness(self):
        """Test min-max fairness metric."""
        values = [40, 10, 10, 5]
        result = min_max_fairness(values)
        self.assertEqual(result, 5)
    
    def test_product_fairness(self):
        """Test product fairness (Nash welfare)."""
        values = [10, 10, 10, 10]
        result = product_fairness(values)
        expected = 4 * np.log(10)
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_range_fairness_perfect(self):
        """Test range fairness for perfect balance."""
        values = [10, 10, 10, 10]
        result = range_fairness(values)
        self.assertEqual(result, 0.0)
    
    def test_range_fairness_imbalance(self):
        """Test range fairness for imbalanced load."""
        values = [40, 10, 10, 5]
        result = range_fairness(values)
        self.assertEqual(result, -35)  # -(40-5)
    
    def test_cv_fairness_perfect(self):
        """Test coefficient of variation for perfect balance."""
        values = [10, 10, 10, 10]
        result = coefficient_of_variation(values)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_cv_fairness_zeros(self):
        """Test CV for all zeros."""
        values = [0, 0, 0, 0]
        result = coefficient_of_variation(values)
        self.assertEqual(result, 0.0)


class TestRewardFunction(unittest.TestCase):
    """Test RewardFunction class."""
    
    def setUp(self):
        """Set up test observations."""
        self.obs_perfect = {
            'active_servers': [0, 1, 2, 3],
            'server_stats': {
                0: {'flow_duration_avg_decay': 10.0},
                1: {'flow_duration_avg_decay': 10.0},
                2: {'flow_duration_avg_decay': 10.0},
                3: {'flow_duration_avg_decay': 10.0}
            }
        }
        
        self.obs_imbalanced = {
            'active_servers': [0, 1, 2, 3],
            'server_stats': {
                0: {'flow_duration_avg_decay': 40.0},
                1: {'flow_duration_avg_decay': 10.0},
                2: {'flow_duration_avg_decay': 10.0},
                3: {'flow_duration_avg_decay': 0.0}
            }
        }
    
    def test_jain_reward_perfect(self):
        """Test Jain reward for perfect balance."""
        reward_fn = RewardFunction(metric='jain', reward_field='flow_duration_avg_decay')
        reward = reward_fn.compute(self.obs_perfect)
        self.assertAlmostEqual(reward, 1.0, places=5)
    
    def test_jain_reward_imbalanced(self):
        """Test Jain reward for imbalanced load."""
        reward_fn = RewardFunction(metric='jain', reward_field='flow_duration_avg_decay')
        reward = reward_fn.compute(self.obs_imbalanced)
        # (60^2) / (4 * (1600+100+100+0)) = 3600/7200 = 0.5
        self.assertAlmostEqual(reward, 0.5, places=3)
    
    def test_variance_reward(self):
        """Test variance reward."""
        reward_fn = RewardFunction(metric='variance', reward_field='flow_duration_avg_decay')
        reward = reward_fn.compute(self.obs_perfect)
        self.assertAlmostEqual(reward, 0.0, places=5)
    
    def test_max_reward(self):
        """Test max-min reward."""
        reward_fn = RewardFunction(metric='max', reward_field='flow_duration_avg_decay')
        reward = reward_fn.compute(self.obs_imbalanced)
        self.assertEqual(reward, -40.0)
    
    def test_empty_servers(self):
        """Test reward with no active servers."""
        reward_fn = RewardFunction(metric='jain', reward_field='flow_duration_avg_decay')
        obs = {'active_servers': [], 'server_stats': {}}
        reward = reward_fn.compute(obs)
        self.assertEqual(reward, 0.0)
    
    def test_missing_field(self):
        """Test reward when field is missing from stats."""
        reward_fn = RewardFunction(metric='jain', reward_field='missing_field')
        reward = reward_fn.compute(self.obs_perfect)
        self.assertEqual(reward, 0.0)
    
    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with self.assertRaises(ValueError):
            RewardFunction(metric='invalid_metric')
    
    def test_callable(self):
        """Test that RewardFunction is callable."""
        reward_fn = RewardFunction(metric='jain')
        reward = reward_fn(self.obs_perfect)
        self.assertIsInstance(reward, float)
    
    def test_repr(self):
        """Test string representation."""
        reward_fn = RewardFunction(metric='jain', reward_field='fct_mean')
        repr_str = repr(reward_fn)
        self.assertIn('jain', repr_str)
        self.assertIn('fct_mean', repr_str)


class TestRewardMetricComparison(unittest.TestCase):
    """Test comparison of different metrics on same data."""
    
    def test_all_metrics_on_perfect_balance(self):
        """All metrics should show high fairness for perfect balance."""
        values = [10, 10, 10, 10]
        
        # Jain should be 1
        self.assertAlmostEqual(jain_fairness(values), 1.0, places=5)
        
        # Variance should be 0
        self.assertEqual(variance_fairness(values), 0.0)
        
        # Range should be 0
        self.assertEqual(range_fairness(values), 0.0)
    
    def test_all_metrics_on_severe_imbalance(self):
        """All metrics should show low fairness for severe imbalance."""
        values = [40, 0, 0, 0]
        
        # Jain should be 1/4
        self.assertAlmostEqual(jain_fairness(values), 0.25, places=5)
        
        # Variance should be high (negative)
        var = variance_fairness(values)
        self.assertLess(var, -100)
        
        # Range should be large (negative)
        rng = range_fairness(values)
        self.assertEqual(rng, -40)
    
    def test_metric_ordering(self):
        """Test that metrics agree on ordering of scenarios."""
        perfect = [10, 10, 10, 10]
        slight = [15, 10, 10, 5]
        severe = [40, 5, 5, 0]
        
        # Jain index ordering
        jain_perfect = jain_fairness(perfect)
        jain_slight = jain_fairness(slight)
        jain_severe = jain_fairness(severe)
        
        self.assertGreater(jain_perfect, jain_slight)
        self.assertGreater(jain_slight, jain_severe)
        
        # Variance ordering (more negative = worse)
        var_perfect = variance_fairness(perfect)
        var_slight = variance_fairness(slight)
        var_severe = variance_fairness(severe)
        
        self.assertGreater(var_perfect, var_slight)
        self.assertGreater(var_slight, var_severe)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_single_value(self):
        """Test with single value."""
        values = [10]
        self.assertEqual(jain_fairness(values), 1.0)
        self.assertEqual(variance_fairness(values), 0.0)
    
    def test_two_equal_values(self):
        """Test with two equal values."""
        values = [10, 10]
        self.assertEqual(jain_fairness(values), 1.0)
        self.assertEqual(variance_fairness(values), 0.0)
    
    def test_two_unequal_values(self):
        """Test with two unequal values."""
        values = [20, 10]
        jain = jain_fairness(values)
        # (30^2) / (2 * (400 + 100)) = 900 / 1000 = 0.9
        self.assertAlmostEqual(jain, 0.9, places=5)
    
    def test_very_large_values(self):
        """Test with very large values."""
        values = [1e6, 1e6, 1e6, 1e6]
        self.assertAlmostEqual(jain_fairness(values), 1.0, places=5)
    
    def test_very_small_values(self):
        """Test with very small values."""
        values = [1e-6, 1e-6, 1e-6, 1e-6]
        self.assertAlmostEqual(jain_fairness(values), 1.0, places=5)
    
    def test_mixed_scales(self):
        """Test with mixed scale values."""
        values = [1000, 1000, 1, 1]
        jain = jain_fairness(values)
        # Not perfect but should be reasonable
        self.assertGreater(jain, 0.5)
        self.assertLess(jain, 1.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
