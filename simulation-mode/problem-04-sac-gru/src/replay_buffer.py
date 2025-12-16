"""
Experience Replay Buffer for SAC-GRU

Stores transitions with GRU hidden states for off-policy learning.
"""

import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """
    Simple replay buffer for off-policy RL algorithms.
    
    Stores: (state, action, reward, next_state, done, hidden_state)
    """
    
    def __init__(self, capacity: int = 1_000_000, seed: int = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state, action, reward, next_state, done, hidden=None):
        """
        Add transition to buffer.
        
        Args:
            state: State array
            action: Action array
            reward: Reward scalar
            next_state: Next state array
            done: Done flag
            hidden: GRU hidden state (optional)
        """
        # Convert to numpy for storage efficiency
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if hidden is not None and isinstance(hidden, torch.Tensor):
            hidden = hidden.cpu().numpy()
        
        self.buffer.append((state, action, reward, next_state, done, hidden))
    
    def sample(self, batch_size: int, device: str = 'cpu'):
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of samples
            device: Device to load tensors to
        
        Returns:
            Tuple of batched tensors: (states, actions, rewards, next_states, dones, hiddens)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones, hiddens = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        
        # Hidden states (may be None)
        if hiddens[0] is not None:
            hiddens_array = np.array(hiddens)
            # Reshape from (batch_size, 1, 1, gru_dim) to (1, batch_size, gru_dim)
            if hiddens_array.ndim == 4:
                hiddens_array = hiddens_array.squeeze(axis=2).squeeze(axis=1)  # Remove extra dims: (batch, gru_dim)
                hiddens_array = np.expand_dims(hiddens_array, axis=0)  # Add seq dim: (1, batch, gru_dim)
            elif hiddens_array.ndim == 2:
                hiddens_array = np.expand_dims(hiddens_array, axis=0)  # (1, batch, gru_dim)
            hiddens = torch.FloatTensor(hiddens_array).to(device)
        else:
            hiddens = None
        
        return states, actions, rewards, next_states, dones, hiddens
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples transitions based on TD error priority.
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        seed: int = None
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sample
            epsilon: Small constant to prevent zero priority
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state, action, reward, next_state, done, hidden=None):
        """Add transition with maximum priority."""
        # Convert to numpy
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if hidden is not None and isinstance(hidden, torch.Tensor):
            hidden = hidden.cpu().numpy()
        
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, hidden))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, hidden)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: str = 'cpu'):
        """
        Sample batch based on priorities.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, hiddens, weights, indices)
        """
        # Sample indices based on priorities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, hiddens = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)
        
        if hiddens[0] is not None:
            hiddens = torch.FloatTensor(np.array(hiddens)).to(device)
        else:
            hiddens = None
        
        return states, actions, rewards, next_states, dones, hiddens, weights, indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        """Return current buffer size."""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size


if __name__ == '__main__':
    # Test replay buffer
    print("Testing Replay Buffers")
    print("=" * 60)
    
    # Test simple buffer
    print("\n1. Testing Simple Replay Buffer...")
    buffer = ReplayBuffer(capacity=1000, seed=42)
    
    # Add some transitions
    for i in range(100):
        state = np.random.randn(44)
        action = np.random.randn(4)
        reward = np.random.randn()
        next_state = np.random.randn(44)
        done = i % 10 == 0
        hidden = np.random.randn(1, 1, 128)
        
        buffer.push(state, action, reward, next_state, done, hidden)
    
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Ready for batch_size=32: {buffer.is_ready(32)}")
    
    # Sample batch
    states, actions, rewards, next_states, dones, hiddens = buffer.sample(32)
    print(f"   Sampled states shape: {states.shape}")
    print(f"   Sampled actions shape: {actions.shape}")
    print(f"   Sampled rewards shape: {rewards.shape}")
    print(f"   Sampled hiddens shape: {hiddens.shape}")
    
    # Test prioritized buffer
    print("\n2. Testing Prioritized Replay Buffer...")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000, seed=42)
    
    # Add transitions
    for i in range(100):
        state = np.random.randn(44)
        action = np.random.randn(4)
        reward = np.random.randn()
        next_state = np.random.randn(44)
        done = i % 10 == 0
        hidden = np.random.randn(1, 1, 128)
        
        pri_buffer.push(state, action, reward, next_state, done, hidden)
    
    print(f"   Buffer size: {len(pri_buffer)}")
    
    # Sample batch
    states, actions, rewards, next_states, dones, hiddens, weights, indices = pri_buffer.sample(32)
    print(f"   Sampled states shape: {states.shape}")
    print(f"   Sampled weights shape: {weights.shape}")
    print(f"   Sampled indices: {len(indices)}")
    
    # Update priorities
    new_priorities = np.random.rand(32)
    pri_buffer.update_priorities(indices, new_priorities)
    print(f"   Priorities updated")
    
    # Test beta increment
    initial_beta = pri_buffer.beta
    pri_buffer.sample(32)
    print(f"   Beta increment: {pri_buffer.beta - initial_beta:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
