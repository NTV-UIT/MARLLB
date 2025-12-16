"""
OpenAI Gym Environment for Load Balancing

This module implements a Gym-compatible environment for multi-agent
load balancing using reinforcement learning.

The environment interfaces with:
- VPP load balancer via shared memory (Problem 02)
- Reservoir sampling for feature extraction (Problem 01)

Reference: MARLLB paper Section 3
"""

import time
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import gym
    from gym import spaces
except ImportError:
    print("Warning: gym not installed. Install with: pip install gym")
    gym = None
    spaces = None

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'problem-02-shared-memory-ipc' / 'src'))
sys.path.append(str(Path(__file__).parent.parent.parent / 'problem-01-reservoir-sampling' / 'src'))

try:
    from shm_region import SharedMemoryRegion
except ImportError:
    SharedMemoryRegion = None
    print("Warning: SharedMemoryRegion not found. SHM integration disabled.")

from rewards import RewardFunction


class LoadBalanceEnv:
    """
    OpenAI Gym environment for load balancing.
    
    State Space:
        - Per-server features (11 dimensions each):
          [n_flow_on, fct_mean, fct_p90, fct_std, fct_mean_decay, fct_p90_decay,
           duration_mean, duration_p90, duration_std, duration_mean_decay, duration_p90_decay]
        - Shape: (num_servers, 11)
    
    Action Space:
        - Discrete: MultiDiscrete([K] * num_servers), where K is number of weight levels
        - Continuous: Box(low=0, high=max_weight, shape=(num_servers,))
    
    Reward:
        - Fairness metric (Jain index, variance, max-min, etc.)
        - Computed on specified field (default: flow_duration_avg_decay)
    
    Example:
        >>> env = LoadBalanceEnv(num_servers=4, action_type='discrete')
        >>> obs = env.reset()
        >>> obs.shape
        (4, 11)
        >>> action = env.action_space.sample()
        >>> next_obs, reward, done, info = env.step(action)
    """
    
    # Default action weights for discrete actions
    DEFAULT_DISCRETE_WEIGHTS = [1.0, 1.5, 2.0]
    
    def __init__(
        self,
        num_servers: int = 4,
        action_type: str = 'discrete',
        discrete_weights: Optional[List[float]] = None,
        max_weight: float = 10.0,
        min_weight: float = 0.1,
        reward_metric: str = 'jain',
        reward_field: str = 'flow_duration_avg_decay',
        step_interval: float = 0.25,
        max_steps: int = 10000,
        use_shm: bool = False,
        shm_name: Optional[str] = None,
        use_ground_truth: bool = False,
        normalize_obs: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize load balancing environment.
        
        Args:
            num_servers: Number of backend servers
            action_type: 'discrete' or 'continuous'
            discrete_weights: Weight values for discrete actions (default: [1.0, 1.5, 2.0])
            max_weight: Maximum weight for continuous actions
            min_weight: Minimum weight for continuous actions
            reward_metric: Fairness metric ('jain', 'variance', 'max', etc.)
            reward_field: Feature field to compute reward on
            step_interval: Time interval between steps (seconds)
            max_steps: Maximum steps per episode
            use_shm: Whether to use shared memory for VPP integration
            shm_name: Shared memory region name (e.g., 'marllb_lb0')
            use_ground_truth: Include CPU/memory in state
            normalize_obs: Normalize observations
            seed: Random seed
        """
        # Gym compatibility check
        if gym is None:
            raise ImportError("gym package required. Install with: pip install gym")
        
        self.num_servers = num_servers
        self.action_type = action_type
        self.discrete_weights = discrete_weights or self.DEFAULT_DISCRETE_WEIGHTS
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.step_interval = step_interval
        self.max_steps = max_steps
        self.use_shm = use_shm
        self.shm_name = shm_name
        self.use_ground_truth = use_ground_truth
        self.normalize_obs = normalize_obs
        
        # Reward function
        self.reward_fn = RewardFunction(metric=reward_metric, reward_field=reward_field)
        
        # Random seed
        self._np_random = np.random.RandomState(seed)
        
        # Define spaces
        self._setup_spaces()
        
        # Shared memory connection (if enabled)
        self.shm = None
        if self.use_shm and SharedMemoryRegion is not None:
            if self.shm_name is None:
                raise ValueError("shm_name required when use_shm=True")
            try:
                self.shm = SharedMemoryRegion.attach(self.shm_name)
                print(f"Connected to shared memory: {self.shm_name}")
            except Exception as e:
                print(f"Warning: Failed to attach shared memory: {e}")
                print("Falling back to simulation mode")
                self.use_shm = False
        
        # Episode state
        self.current_step = 0
        self.last_observation = None
        self.episode_rewards = []
        self.episode_return = 0.0
        
        # Observation statistics (for normalization)
        self.obs_mean = np.zeros((num_servers, 11))
        self.obs_std = np.ones((num_servers, 11))
        self.obs_count = 0
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space: (num_servers, 11) features per server
        num_features = 11
        if self.use_ground_truth:
            num_features += 3  # CPU, memory, threads
        
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.num_servers, num_features),
            dtype=np.float32
        )
        
        # Action space
        if self.action_type == 'discrete':
            # MultiDiscrete: each server has K discrete actions
            num_actions = len(self.discrete_weights)
            self.action_space = spaces.MultiDiscrete([num_actions] * self.num_servers)
        elif self.action_type == 'continuous':
            # Box: continuous weights for each server
            self.action_space = spaces.Box(
                low=self.min_weight,
                high=self.max_weight,
                shape=(self.num_servers,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation (num_servers, 11)
        """
        self.current_step = 0
        self.episode_rewards = []
        self.episode_return = 0.0
        
        if self.use_shm and self.shm is not None:
            # Read initial observation from shared memory
            try:
                obs_dict = self.shm.read_observation()
                self.last_observation = obs_dict
                obs = self._dict_to_array(obs_dict)
            except Exception as e:
                print(f"Warning: Failed to read from SHM: {e}")
                obs = self._simulate_observation()
        else:
            # Simulation mode
            obs = self._simulate_observation()
        
        if self.normalize_obs:
            obs = self._normalize_observation(obs)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action and return next observation.
        
        Args:
            action: Action to take
                - Discrete: array of action indices [0, 1, 2, 1, ...]
                - Continuous: array of weights [1.2, 0.8, 1.5, ...]
        
        Returns:
            observation: Next state (num_servers, 11)
            reward: Reward value
            done: Whether episode is done
            info: Additional information dictionary
        """
        self.current_step += 1
        
        # Convert action to weights
        weights = self._action_to_weights(action)
        
        if self.use_shm and self.shm is not None:
            # Write action to shared memory
            try:
                seq_id = self.last_observation.get('sequence_id', self.current_step)
                self.shm.write_action(sequence_id=seq_id, weights=weights)
            except Exception as e:
                print(f"Warning: Failed to write action to SHM: {e}")
            
            # Wait for next observation
            time.sleep(self.step_interval)
            
            # Read new observation
            try:
                obs_dict = self.shm.read_observation()
                self.last_observation = obs_dict
                next_obs = self._dict_to_array(obs_dict)
            except Exception as e:
                print(f"Warning: Failed to read from SHM: {e}")
                next_obs = self._simulate_observation()
                obs_dict = self._array_to_dict(next_obs)
        else:
            # Simulation mode
            time.sleep(self.step_interval)
            next_obs = self._simulate_observation()
            obs_dict = self._array_to_dict(next_obs)
        
        # Compute reward
        reward = self.reward_fn.compute(obs_dict)
        self.episode_rewards.append(reward)
        self.episode_return += reward
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'step': self.current_step,
            'weights': weights.tolist(),
            'active_servers': obs_dict.get('active_servers', list(range(self.num_servers))),
            'episode_return': self.episode_return
        }
        
        if done:
            info['episode'] = {
                'r': self.episode_return,
                'l': self.current_step
            }
        
        if self.normalize_obs:
            next_obs = self._normalize_observation(next_obs)
        
        return next_obs, reward, done, info
    
    def render(self, mode: str = 'human'):
        """
        Render environment state.
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
        """
        if mode == 'human':
            print(f"\n{'=' * 60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Episode Return: {self.episode_return:.4f}")
            
            if self.last_observation:
                obs_dict = self.last_observation
                active_servers = obs_dict.get('active_servers', [])
                server_stats = obs_dict.get('server_stats', {})
                
                print(f"Active Servers: {active_servers}")
                print(f"\n{'Server':<10} {'n_flows':<10} {'fct_mean':<12} {'fct_p90':<12} {'dur_decay':<12}")
                print("-" * 60)
                
                for sid in active_servers:
                    if sid in server_stats:
                        stats = server_stats[sid]
                        n_flow = stats.get('n_flow_on', 0)
                        fct_mean = stats.get('fct_mean', 0)
                        fct_p90 = stats.get('fct_p90', 0)
                        dur_decay = stats.get('flow_duration_avg_decay', 0)
                        
                        print(f"{sid:<10} {n_flow:<10.0f} {fct_mean:<12.4f} {fct_p90:<12.4f} {dur_decay:<12.4f}")
            
            print("=" * 60)
    
    def close(self):
        """Clean up resources."""
        if self.shm is not None:
            # Optionally detach/close SHM
            pass
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        self._np_random = np.random.RandomState(seed)
        return [seed]
    
    # ===== Internal Helper Methods =====
    
    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert action to weight array.
        
        Args:
            action: Action from action_space
        
        Returns:
            Weights array (num_servers,)
        """
        if self.action_type == 'discrete':
            # Map action indices to weights
            weights = np.array([self.discrete_weights[int(a)] for a in action], dtype=np.float32)
        else:
            # Continuous actions are already weights
            weights = np.asarray(action, dtype=np.float32)
            # Clip to valid range
            weights = np.clip(weights, self.min_weight, self.max_weight)
        
        return weights
    
    def _dict_to_array(self, obs_dict: dict) -> np.ndarray:
        """
        Convert observation dictionary to array.
        
        Args:
            obs_dict: Observation dictionary with structure:
                {
                    'active_servers': [0, 1, 2, 3],
                    'server_stats': {
                        0: {'n_flow_on': 10, 'fct_mean': 5.2, ...},
                        ...
                    }
                }
        
        Returns:
            Observation array (num_servers, 11)
        """
        obs = np.zeros((self.num_servers, 11), dtype=np.float32)
        
        active_servers = obs_dict.get('active_servers', [])
        server_stats = obs_dict.get('server_stats', {})
        
        feature_names = [
            'n_flow_on', 'fct_mean', 'fct_p90', 'fct_std', 'fct_mean_decay', 'fct_p90_decay',
            'flow_duration_mean', 'flow_duration_p90', 'flow_duration_std',
            'flow_duration_mean_decay', 'flow_duration_avg_decay'
        ]
        
        for sid in active_servers:
            if sid < self.num_servers and sid in server_stats:
                stats = server_stats[sid]
                for i, fname in enumerate(feature_names):
                    obs[sid, i] = stats.get(fname, 0.0)
        
        return obs
    
    def _array_to_dict(self, obs: np.ndarray) -> dict:
        """
        Convert observation array to dictionary (for simulation mode).
        
        Args:
            obs: Observation array (num_servers, 11)
        
        Returns:
            Observation dictionary
        """
        feature_names = [
            'n_flow_on', 'fct_mean', 'fct_p90', 'fct_std', 'fct_mean_decay', 'fct_p90_decay',
            'flow_duration_mean', 'flow_duration_p90', 'flow_duration_std',
            'flow_duration_mean_decay', 'flow_duration_avg_decay'
        ]
        
        server_stats = {}
        active_servers = []
        
        for sid in range(self.num_servers):
            # Check if server is active (has non-zero features)
            if np.any(obs[sid] > 0):
                active_servers.append(sid)
                server_stats[sid] = {
                    fname: float(obs[sid, i])
                    for i, fname in enumerate(feature_names)
                }
        
        return {
            'active_servers': active_servers,
            'server_stats': server_stats,
            'sequence_id': self.current_step
        }
    
    def _simulate_observation(self) -> np.ndarray:
        """
        Simulate observation (for testing without VPP).
        
        Returns:
            Simulated observation array (num_servers, 11)
        """
        obs = np.zeros((self.num_servers, 11), dtype=np.float32)
        
        for sid in range(self.num_servers):
            # Simulate features with random values
            obs[sid, 0] = self._np_random.randint(5, 20)  # n_flow_on
            obs[sid, 1] = self._np_random.uniform(5, 15)  # fct_mean
            obs[sid, 2] = self._np_random.uniform(10, 25)  # fct_p90
            obs[sid, 3] = self._np_random.uniform(1, 5)  # fct_std
            obs[sid, 4] = obs[sid, 1] * 0.9  # fct_mean_decay
            obs[sid, 5] = obs[sid, 2] * 0.9  # fct_p90_decay
            obs[sid, 6] = self._np_random.uniform(8, 18)  # flow_duration_mean
            obs[sid, 7] = self._np_random.uniform(15, 30)  # flow_duration_p90
            obs[sid, 8] = self._np_random.uniform(2, 8)  # flow_duration_std
            obs[sid, 9] = obs[sid, 6] * 0.85  # flow_duration_mean_decay
            obs[sid, 10] = obs[sid, 6] * 0.9  # flow_duration_avg_decay
        
        return obs
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation using running statistics.
        
        Args:
            obs: Observation array (num_servers, 11)
        
        Returns:
            Normalized observation
        """
        # Update running statistics
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_std = np.sqrt(np.maximum((self.obs_std ** 2 * (self.obs_count - 1) + delta * delta2) / self.obs_count, 1e-8))
        
        # Normalize
        normalized = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        
        return normalized


# Alias for Gym compatibility
if gym is not None:
    class LoadBalanceEnvGym(gym.Env, LoadBalanceEnv):
        """Gym-compatible wrapper for LoadBalanceEnv."""
        
        def __init__(self, **kwargs):
            LoadBalanceEnv.__init__(self, **kwargs)
        
        metadata = {'render.modes': ['human']}


if __name__ == '__main__':
    # Demonstration
    print("LoadBalanceEnv Demonstration")
    print("=" * 60)
    
    # Create environment
    env = LoadBalanceEnv(
        num_servers=4,
        action_type='discrete',
        reward_metric='jain',
        max_steps=10,
        use_shm=False  # Simulation mode
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Reward function: {env.reward_fn}")
    
    # Run episode
    obs = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation:\n{obs}")
    
    total_reward = 0
    for step in range(5):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Weights: {info['weights']}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}")
        
        if done:
            break
    
    print(f"\nEpisode finished!")
    print(f"Total reward: {total_reward:.4f}")
    
    env.render()
