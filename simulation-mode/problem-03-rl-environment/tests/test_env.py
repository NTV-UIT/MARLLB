"""
Unit tests for LoadBalanceEnv

Tests the Gym environment implementation.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from env import LoadBalanceEnv


class TestLoadBalanceEnvBasics(unittest.TestCase):
    """Test basic environment functionality."""
    
    def setUp(self):
        """Create environment for testing."""
        self.env = LoadBalanceEnv(
            num_servers=4,
            action_type='discrete',
            max_steps=100,
            use_shm=False,  # Simulation mode
            seed=42
        )
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.num_servers, 4)
        self.assertEqual(self.env.action_type, 'discrete')
        self.assertEqual(self.env.max_steps, 100)
        self.assertFalse(self.env.use_shm)
    
    def test_observation_space(self):
        """Test observation space shape and bounds."""
        obs_space = self.env.observation_space
        self.assertEqual(obs_space.shape, (4, 11))
        self.assertTrue(np.all(obs_space.low == 0))
        self.assertTrue(np.all(obs_space.high == np.inf))
    
    def test_action_space_discrete(self):
        """Test discrete action space."""
        action_space = self.env.action_space
        self.assertEqual(len(action_space.nvec), 4)
        self.assertTrue(np.all(action_space.nvec == 3))
    
    def test_action_space_continuous(self):
        """Test continuous action space."""
        env = LoadBalanceEnv(
            num_servers=4,
            action_type='continuous',
            max_steps=100,
            use_shm=False
        )
        action_space = env.action_space
        self.assertEqual(action_space.shape, (4,))
        self.assertTrue(np.all(action_space.low == 0.1))
        self.assertTrue(np.all(action_space.high == 10.0))
    
    def test_reset(self):
        """Test environment reset."""
        obs = self.env.reset()
        
        # Check shape
        self.assertEqual(obs.shape, (4, 11))
        
        # Check all values are finite
        self.assertTrue(np.all(np.isfinite(obs)))
        
        # Check step counter reset
        self.assertEqual(self.env.current_step, 0)
    
    def test_step(self):
        """Test environment step."""
        self.env.reset()
        action = self.env.action_space.sample()
        
        obs, reward, done, info = self.env.step(action)
        
        # Check observation
        self.assertEqual(obs.shape, (4, 11))
        self.assertTrue(np.all(np.isfinite(obs)))
        
        # Check reward
        self.assertIsInstance(reward, (int, float))
        self.assertTrue(np.isfinite(reward))
        
        # Check done
        self.assertIsInstance(done, bool)
        
        # Check info
        self.assertIsInstance(info, dict)
        self.assertIn('step', info)
        self.assertIn('weights', info)
        self.assertIn('active_servers', info)
    
    def test_episode_termination(self):
        """Test that episode terminates at max_steps."""
        env = LoadBalanceEnv(num_servers=4, max_steps=5, use_shm=False)
        env.reset()
        
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            if step < 4:
                self.assertFalse(done)
            else:
                self.assertTrue(done)
    
    def test_seed(self):
        """Test that seed produces reproducible results."""
        env1 = LoadBalanceEnv(num_servers=4, use_shm=False, seed=42)
        env2 = LoadBalanceEnv(num_servers=4, use_shm=False, seed=42)
        
        obs1 = env1.reset()
        obs2 = env2.reset()
        
        # Should be identical with same seed
        np.testing.assert_array_almost_equal(obs1, obs2)


class TestActionConversion(unittest.TestCase):
    """Test action to weight conversion."""
    
    def test_discrete_action_conversion(self):
        """Test discrete action to weights."""
        env = LoadBalanceEnv(
            num_servers=4,
            action_type='discrete',
            discrete_weights=[1.0, 1.5, 2.0],
            use_shm=False
        )
        
        action = np.array([0, 1, 2, 1])
        weights = env._action_to_weights(action)
        
        expected = np.array([1.0, 1.5, 2.0, 1.5])
        np.testing.assert_array_almost_equal(weights, expected)
    
    def test_continuous_action_conversion(self):
        """Test continuous action conversion."""
        env = LoadBalanceEnv(
            num_servers=4,
            action_type='continuous',
            min_weight=0.5,
            max_weight=5.0,
            use_shm=False
        )
        
        action = np.array([1.0, 2.0, 3.0, 4.0])
        weights = env._action_to_weights(action)
        
        # Should be clipped to [0.5, 5.0]
        np.testing.assert_array_almost_equal(weights, action)
    
    def test_continuous_action_clipping(self):
        """Test that continuous actions are clipped to valid range."""
        env = LoadBalanceEnv(
            num_servers=4,
            action_type='continuous',
            min_weight=0.5,
            max_weight=5.0,
            use_shm=False
        )
        
        # Actions outside range
        action = np.array([0.1, 2.0, 6.0, 3.0])
        weights = env._action_to_weights(action)
        
        expected = np.array([0.5, 2.0, 5.0, 3.0])  # Clipped
        np.testing.assert_array_almost_equal(weights, expected)


class TestObservationConversion(unittest.TestCase):
    """Test observation dictionary <-> array conversion."""
    
    def setUp(self):
        """Create environment for testing."""
        self.env = LoadBalanceEnv(num_servers=4, use_shm=False)
    
    def test_dict_to_array(self):
        """Test converting observation dict to array."""
        obs_dict = {
            'active_servers': [0, 1, 2, 3],
            'server_stats': {
                0: {
                    'n_flow_on': 10, 'fct_mean': 5.0, 'fct_p90': 8.0,
                    'fct_std': 2.0, 'fct_mean_decay': 4.5, 'fct_p90_decay': 7.5,
                    'flow_duration_mean': 6.0, 'flow_duration_p90': 9.0,
                    'flow_duration_std': 3.0, 'flow_duration_mean_decay': 5.5,
                    'flow_duration_avg_decay': 5.0
                },
                1: {
                    'n_flow_on': 12, 'fct_mean': 6.0, 'fct_p90': 9.0,
                    'fct_std': 2.5, 'fct_mean_decay': 5.5, 'fct_p90_decay': 8.5,
                    'flow_duration_mean': 7.0, 'flow_duration_p90': 10.0,
                    'flow_duration_std': 3.5, 'flow_duration_mean_decay': 6.5,
                    'flow_duration_avg_decay': 6.0
                },
                2: {
                    'n_flow_on': 8, 'fct_mean': 4.0, 'fct_p90': 7.0,
                    'fct_std': 1.5, 'fct_mean_decay': 3.5, 'fct_p90_decay': 6.5,
                    'flow_duration_mean': 5.0, 'flow_duration_p90': 8.0,
                    'flow_duration_std': 2.5, 'flow_duration_mean_decay': 4.5,
                    'flow_duration_avg_decay': 4.0
                },
                3: {
                    'n_flow_on': 15, 'fct_mean': 7.0, 'fct_p90': 10.0,
                    'fct_std': 3.0, 'fct_mean_decay': 6.5, 'fct_p90_decay': 9.5,
                    'flow_duration_mean': 8.0, 'flow_duration_p90': 11.0,
                    'flow_duration_std': 4.0, 'flow_duration_mean_decay': 7.5,
                    'flow_duration_avg_decay': 7.0
                }
            }
        }
        
        obs_array = self.env._dict_to_array(obs_dict)
        
        # Check shape
        self.assertEqual(obs_array.shape, (4, 11))
        
        # Check first server values
        self.assertEqual(obs_array[0, 0], 10)  # n_flow_on
        self.assertAlmostEqual(obs_array[0, 1], 5.0)  # fct_mean
        self.assertAlmostEqual(obs_array[0, 10], 5.0)  # flow_duration_avg_decay
    
    def test_array_to_dict(self):
        """Test converting observation array to dict."""
        obs_array = np.array([
            [10, 5.0, 8.0, 2.0, 4.5, 7.5, 6.0, 9.0, 3.0, 5.5, 5.0],
            [12, 6.0, 9.0, 2.5, 5.5, 8.5, 7.0, 10.0, 3.5, 6.5, 6.0],
            [8, 4.0, 7.0, 1.5, 3.5, 6.5, 5.0, 8.0, 2.5, 4.5, 4.0],
            [15, 7.0, 10.0, 3.0, 6.5, 9.5, 8.0, 11.0, 4.0, 7.5, 7.0]
        ], dtype=np.float32)
        
        obs_dict = self.env._array_to_dict(obs_array)
        
        # Check structure
        self.assertIn('active_servers', obs_dict)
        self.assertIn('server_stats', obs_dict)
        
        # Check active servers
        self.assertEqual(len(obs_dict['active_servers']), 4)
        
        # Check first server stats
        server_0_stats = obs_dict['server_stats'][0]
        self.assertEqual(server_0_stats['n_flow_on'], 10)
        self.assertAlmostEqual(server_0_stats['fct_mean'], 5.0)
    
    def test_round_trip_conversion(self):
        """Test dict -> array -> dict conversion."""
        obs_dict = {
            'active_servers': [0, 1, 2, 3],
            'server_stats': {
                0: {
                    'n_flow_on': 10, 'fct_mean': 5.0, 'fct_p90': 8.0,
                    'fct_std': 2.0, 'fct_mean_decay': 4.5, 'fct_p90_decay': 7.5,
                    'flow_duration_mean': 6.0, 'flow_duration_p90': 9.0,
                    'flow_duration_std': 3.0, 'flow_duration_mean_decay': 5.5,
                    'flow_duration_avg_decay': 5.0
                }
            }
        }
        
        # Convert dict -> array -> dict
        obs_array = self.env._dict_to_array(obs_dict)
        obs_dict_2 = self.env._array_to_dict(obs_array)
        
        # Check that values match
        self.assertEqual(obs_dict_2['server_stats'][0]['n_flow_on'], 10)
        self.assertAlmostEqual(obs_dict_2['server_stats'][0]['fct_mean'], 5.0, places=5)


class TestRewardComputation(unittest.TestCase):
    """Test reward computation in environment."""
    
    def test_reward_in_step(self):
        """Test that step returns valid reward."""
        env = LoadBalanceEnv(
            num_servers=4,
            reward_metric='jain',
            use_shm=False,
            seed=42
        )
        
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # Reward should be in Jain index range [0.25, 1.0] for 4 servers
        self.assertGreaterEqual(reward, 0.25)
        self.assertLessEqual(reward, 1.0)
    
    def test_episode_return(self):
        """Test episode return accumulation."""
        env = LoadBalanceEnv(num_servers=4, max_steps=5, use_shm=False, seed=42)
        env.reset()
        
        total_reward = 0
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        # Check that episode return matches
        self.assertAlmostEqual(info['episode']['r'], total_reward, places=5)


class TestNormalization(unittest.TestCase):
    """Test observation normalization."""
    
    def test_normalization_updates_statistics(self):
        """Test that normalization updates running statistics."""
        env = LoadBalanceEnv(
            num_servers=4,
            normalize_obs=True,
            use_shm=False,
            seed=42
        )
        
        obs1 = env.reset()
        initial_count = env.obs_count
        
        action = env.action_space.sample()
        obs2, _, _, _ = env.step(action)
        
        # Count should increase
        self.assertEqual(env.obs_count, initial_count + 1)
        
        # Mean and std should be updated
        self.assertTrue(np.any(env.obs_mean != 0))


class TestSimulation(unittest.TestCase):
    """Test simulation mode (without SHM)."""
    
    def test_simulate_observation(self):
        """Test simulated observation generation."""
        env = LoadBalanceEnv(num_servers=4, use_shm=False, seed=42)
        
        obs = env._simulate_observation()
        
        # Check shape
        self.assertEqual(obs.shape, (4, 11))
        
        # Check all values are positive
        self.assertTrue(np.all(obs >= 0))
        
        # Check all values are finite
        self.assertTrue(np.all(np.isfinite(obs)))
    
    def test_simulation_reproducibility(self):
        """Test that simulation is reproducible with seed."""
        env1 = LoadBalanceEnv(num_servers=4, use_shm=False, seed=42)
        env2 = LoadBalanceEnv(num_servers=4, use_shm=False, seed=42)
        
        obs1 = env1._simulate_observation()
        obs2 = env2._simulate_observation()
        
        np.testing.assert_array_almost_equal(obs1, obs2)


class TestRender(unittest.TestCase):
    """Test rendering functionality."""
    
    def test_render_no_crash(self):
        """Test that render doesn't crash."""
        env = LoadBalanceEnv(num_servers=4, use_shm=False)
        env.reset()
        
        # Should not crash
        try:
            env.render(mode='human')
        except Exception as e:
            self.fail(f"render() raised exception: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
