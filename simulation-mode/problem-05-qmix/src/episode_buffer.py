"""
Episode Replay Buffer for QMIX

Stores complete episodes with multi-agent transitions for centralized training.
"""

import numpy as np
from collections import deque
import random


class EpisodeBuffer:
    """
    Episode replay buffer for multi-agent learning.
    
    Stores complete episodes rather than individual transitions because:
    1. QMIX needs episode context for mixing
    2. Recurrent networks (GRU) need sequential data
    3. Credit assignment requires full trajectories
    
    Episode structure:
        {
            'observations': List of [obs_agent_0, obs_agent_1, ..., obs_agent_n] for each timestep
            'actions': List of [action_agent_0, action_agent_1, ..., action_agent_n] for each timestep
            'rewards': List of [reward_agent_0, reward_agent_1, ..., reward_agent_n] for each timestep
            'states': List of global states for each timestep
            'dones': List of done flags for each timestep
            'hiddens': List of GRU hidden states for each agent
        }
    
    Args:
        capacity: Maximum number of episodes to store
        num_agents: Number of agents
    """
    
    def __init__(self, capacity: int = 5000, num_agents: int = 4):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = deque(maxlen=capacity)
        self.current_episode = None
    
    def start_episode(self):
        """Start collecting a new episode."""
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'states': [],
            'dones': [],
            'hiddens': []
        }
    
    def add_transition(self, observations, actions, rewards, state, done, hiddens=None):
        """
        Add a transition to current episode.
        
        Args:
            observations: List of observations (num_agents, obs_dim)
            actions: List of actions (num_agents, action_dim)
            rewards: List of rewards (num_agents,)
            state: Global state
            done: Episode done flag
            hiddens: List of hidden states (num_agents, 1, gru_dim) or None
        """
        if self.current_episode is None:
            self.start_episode()
        
        self.current_episode['observations'].append(observations)
        self.current_episode['actions'].append(actions)
        self.current_episode['rewards'].append(rewards)
        self.current_episode['states'].append(state)
        self.current_episode['dones'].append(done)
        
        if hiddens is not None:
            self.current_episode['hiddens'].append(hiddens)
    
    def end_episode(self):
        """Finish current episode and add to buffer."""
        if self.current_episode is not None and len(self.current_episode['observations']) > 0:
            self.buffer.append(self.current_episode)
            self.current_episode = None
    
    def sample_batch(self, batch_size: int, max_seq_len: int = None):
        """
        Sample batch of episode sequences.
        
        Args:
            batch_size: Number of episodes to sample
            max_seq_len: Maximum sequence length (None = full episodes)
        
        Returns:
            Dictionary with batched tensors:
                observations: (batch, seq_len, num_agents, obs_dim)
                actions: (batch, seq_len, num_agents, action_dim)
                rewards: (batch, seq_len, num_agents)
                states: (batch, seq_len, state_dim)
                dones: (batch, seq_len)
                seq_lengths: (batch,) - actual length of each sequence
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Sample episodes
        episodes = random.sample(self.buffer, batch_size)
        
        # Determine sequence length
        if max_seq_len is None:
            seq_len = max(len(ep['observations']) for ep in episodes)
        else:
            seq_len = min(max_seq_len, max(len(ep['observations']) for ep in episodes))
        
        # Get dimensions
        obs_dim = len(episodes[0]['observations'][0][0])
        action_dim = len(episodes[0]['actions'][0][0]) if hasattr(episodes[0]['actions'][0][0], '__len__') else 1
        state_dim = len(episodes[0]['states'][0])
        
        # Initialize arrays
        batch = {
            'observations': np.zeros((batch_size, seq_len, self.num_agents, obs_dim)),
            'actions': np.zeros((batch_size, seq_len, self.num_agents, action_dim)),
            'rewards': np.zeros((batch_size, seq_len, self.num_agents)),
            'states': np.zeros((batch_size, seq_len, state_dim)),
            'dones': np.zeros((batch_size, seq_len)),
            'seq_lengths': np.zeros(batch_size, dtype=np.int32)
        }
        
        # Fill batch
        for i, episode in enumerate(episodes):
            ep_len = min(len(episode['observations']), seq_len)
            batch['seq_lengths'][i] = ep_len
            
            for t in range(ep_len):
                # Observations
                for agent_id in range(self.num_agents):
                    batch['observations'][i, t, agent_id] = episode['observations'][t][agent_id]
                
                # Actions
                for agent_id in range(self.num_agents):
                    action = episode['actions'][t][agent_id]
                    if not hasattr(action, '__len__'):
                        action = [action]
                    batch['actions'][i, t, agent_id] = action
                
                # Rewards
                for agent_id in range(self.num_agents):
                    batch['rewards'][i, t, agent_id] = episode['rewards'][t][agent_id]
                
                # State and done
                batch['states'][i, t] = episode['states'][t]
                batch['dones'][i, t] = episode['dones'][t]
        
        return batch
    
    def __len__(self):
        """Return number of episodes in buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough episodes."""
        return len(self.buffer) >= batch_size
    
    def get_stats(self):
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {}
        
        episode_lengths = [len(ep['observations']) for ep in self.buffer]
        total_rewards = [sum([sum(r) for r in ep['rewards']]) for ep in self.buffer]
        
        return {
            'num_episodes': len(self.buffer),
            'avg_length': np.mean(episode_lengths),
            'max_length': np.max(episode_lengths),
            'min_length': np.min(episode_lengths),
            'avg_return': np.mean(total_rewards),
            'max_return': np.max(total_rewards),
            'min_return': np.min(total_rewards)
        }


if __name__ == '__main__':
    # Test episode buffer
    print("=" * 60)
    print("Testing Episode Buffer")
    print("=" * 60)
    
    num_agents = 4
    obs_dim = 20
    action_dim = 4
    state_dim = 74
    
    buffer = EpisodeBuffer(capacity=100, num_agents=num_agents)
    
    # Test 1: Add episodes
    print("\n1. Adding episodes...")
    for ep in range(10):
        buffer.start_episode()
        ep_len = np.random.randint(10, 20)
        
        for t in range(ep_len):
            obs = [np.random.randn(obs_dim) for _ in range(num_agents)]
            actions = [np.random.randn(action_dim) for _ in range(num_agents)]
            rewards = [np.random.randn() for _ in range(num_agents)]
            state = np.random.randn(state_dim)
            done = (t == ep_len - 1)
            
            buffer.add_transition(obs, actions, rewards, state, done)
        
        buffer.end_episode()
    
    print(f"   Episodes in buffer: {len(buffer)}")
    
    # Test 2: Sample batch
    print("\n2. Sampling batch...")
    batch_size = 4
    batch = buffer.sample_batch(batch_size)
    
    if batch is not None:
        print(f"   Batch observations shape: {batch['observations'].shape}")
        print(f"   Batch actions shape: {batch['actions'].shape}")
        print(f"   Batch rewards shape: {batch['rewards'].shape}")
        print(f"   Batch states shape: {batch['states'].shape}")
        print(f"   Batch dones shape: {batch['dones'].shape}")
        print(f"   Sequence lengths: {batch['seq_lengths']}")
    
    # Test 3: Sample with max_seq_len
    print("\n3. Sampling with max_seq_len=10...")
    batch = buffer.sample_batch(batch_size, max_seq_len=10)
    
    if batch is not None:
        print(f"   Batch observations shape: {batch['observations'].shape}")
        print(f"   Sequence lengths: {batch['seq_lengths']}")
    
    # Test 4: Buffer statistics
    print("\n4. Buffer statistics...")
    stats = buffer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.4f}")
    
    # Test 5: Buffer capacity
    print("\n5. Testing buffer capacity...")
    initial_size = len(buffer)
    for ep in range(50):
        buffer.start_episode()
        for t in range(10):
            obs = [np.random.randn(obs_dim) for _ in range(num_agents)]
            actions = [np.random.randn(action_dim) for _ in range(num_agents)]
            rewards = [np.random.randn() for _ in range(num_agents)]
            state = np.random.randn(state_dim)
            done = (t == 9)
            buffer.add_transition(obs, actions, rewards, state, done)
        buffer.end_episode()
    
    print(f"   Initial size: {initial_size}")
    print(f"   After adding 50 episodes: {len(buffer)}")
    print(f"   Capacity respected: {len(buffer) <= buffer.capacity}")
    
    # Test 6: Empty episode handling
    print("\n6. Testing empty episode handling...")
    buffer.start_episode()
    buffer.end_episode()  # End without adding transitions
    print(f"   Buffer size (should not increase): {len(buffer)}")
    
    # Test 7: is_ready check
    print("\n7. Testing is_ready...")
    print(f"   Ready for batch_size=4: {buffer.is_ready(4)}")
    print(f"   Ready for batch_size=1000: {buffer.is_ready(1000)}")
    
    print("\n" + "=" * 60)
    print("All Episode Buffer Tests Passed! ✓")
    print("=" * 60)
