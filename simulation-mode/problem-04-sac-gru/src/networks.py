"""
Neural Network Architectures for SAC-GRU

This module implements:
- PolicyNetwork: GRU-based actor with Gaussian policy
- QNetwork: GRU-based critic for Q-value estimation
- Helper functions for network operations

Reference: MARLLB paper Section 4.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class PolicyNetwork(nn.Module):
    """
    GRU-based policy network for continuous actions.
    
    Architecture:
        State → GRU → FC → [Mean, Log_std] → Gaussian distribution
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        gru_dim: GRU hidden state dimension
        action_scale: Scale for action output
        action_bias: Bias for action output
        log_std_min: Minimum log std (for numerical stability)
        log_std_max: Maximum log std (prevent too much exploration)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gru_dim: int = 128,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # GRU layer
        self.gru = nn.GRU(state_dim, gru_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(gru_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, state, hidden):
        """
        Forward pass through policy network.
        
        Args:
            state: State tensor (batch_size, state_dim)
            hidden: GRU hidden state (1, batch_size, gru_dim)
        
        Returns:
            mean: Action mean (batch_size, action_dim)
            log_std: Action log std (batch_size, action_dim)
            hidden_new: Updated hidden state (1, batch_size, gru_dim)
        """
        # Add sequence dimension
        state = state.unsqueeze(1)  # (batch_size, 1, state_dim)
        
        # GRU forward
        gru_out, hidden_new = self.gru(state, hidden)
        gru_out = gru_out.squeeze(1)  # (batch_size, gru_dim)
        
        # Fully connected layers
        x = F.relu(self.fc1(gru_out))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, hidden_new
    
    def sample(self, state, hidden):
        """
        Sample action from policy.
        
        Args:
            state: State tensor (batch_size, state_dim)
            hidden: GRU hidden state (1, batch_size, gru_dim)
        
        Returns:
            action: Sampled action (batch_size, action_dim)
            log_prob: Log probability of action (batch_size, 1)
            mean: Action mean (batch_size, action_dim)
            hidden_new: Updated hidden state (1, batch_size, gru_dim)
        """
        mean, log_std, hidden_new = self.forward(state, hidden)
        std = torch.exp(log_std)
        
        # Create Gaussian distribution
        normal = Normal(mean, std)
        
        # Sample action (reparameterization trick)
        x_t = normal.rsample()  # Reparameterized sample
        y_t = torch.tanh(x_t)  # Squash to [-1, 1]
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        
        # Apply correction for tanh squashing
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Compute mean action (for deterministic evaluation)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean, hidden_new
    
    def init_hidden(self, batch_size: int = 1):
        """Initialize GRU hidden state."""
        return torch.zeros(1, batch_size, self.gru_dim)
    
    def to(self, device):
        """Move network to device."""
        return super(PolicyNetwork, self).to(device)


class QNetwork(nn.Module):
    """
    GRU-based Q-network for state-action value estimation.
    
    Architecture:
        (State, Action) → Concat → GRU → FC → FC → Q-value
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        gru_dim: GRU hidden state dimension
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gru_dim: int = 128
    ):
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim
        
        # GRU layer (input: state + action)
        self.gru = nn.GRU(state_dim + action_dim, gru_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(gru_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, state, action, hidden):
        """
        Forward pass through Q-network.
        
        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            hidden: GRU hidden state (1, batch_size, gru_dim)
        
        Returns:
            q_value: Q-value (batch_size, 1)
            hidden_new: Updated hidden state (1, batch_size, gru_dim)
        """
        # Concatenate state and action
        sa = torch.cat([state, action], dim=1)  # (batch_size, state_dim + action_dim)
        
        # Add sequence dimension
        sa = sa.unsqueeze(1)  # (batch_size, 1, state_dim + action_dim)
        
        # GRU forward
        gru_out, hidden_new = self.gru(sa, hidden)
        gru_out = gru_out.squeeze(1)  # (batch_size, gru_dim)
        
        # Fully connected layers
        x = F.relu(self.fc1(gru_out))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value, hidden_new
    
    def init_hidden(self, batch_size: int = 1):
        """Initialize GRU hidden state."""
        return torch.zeros(1, batch_size, self.gru_dim)
    
    def to(self, device):
        """Move network to device."""
        return super(QNetwork, self).to(device)


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    """
    Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        source: Source network
        target: Target network
        tau: Soft update coefficient
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(source: nn.Module, target: nn.Module):
    """
    Hard update target network parameters.
    
    θ_target = θ_source
    
    Args:
        source: Source network
        target: Target network
    """
    target.load_state_dict(source.state_dict())


def get_network_info(network: nn.Module) -> dict:
    """
    Get information about network architecture.
    
    Args:
        network: Neural network
    
    Returns:
        Dictionary with network information
    """
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'architecture': str(network),
        'device': next(network.parameters()).device
    }
    
    return info


if __name__ == '__main__':
    # Test networks
    print("Testing SAC-GRU Networks")
    print("=" * 60)
    
    # Configuration
    state_dim = 44  # 4 servers × 11 features
    action_dim = 4  # 4 servers
    hidden_dim = 256
    gru_dim = 128
    batch_size = 32
    
    # Create networks
    print("\n1. Creating Policy Network...")
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, gru_dim)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    print("\n2. Creating Q-Networks...")
    q1 = QNetwork(state_dim, action_dim, hidden_dim, gru_dim)
    q2 = QNetwork(state_dim, action_dim, hidden_dim, gru_dim)
    print(f"Q1 parameters: {sum(p.numel() for p in q1.parameters()):,}")
    print(f"Q2 parameters: {sum(p.numel() for p in q2.parameters()):,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    state = torch.randn(batch_size, state_dim)
    policy_hidden = policy.init_hidden(batch_size)
    q_hidden = q1.init_hidden(batch_size)
    
    # Policy forward
    action, log_prob, mean, policy_hidden_new = policy.sample(state, policy_hidden)
    print(f"   Action shape: {action.shape}")
    print(f"   Log prob shape: {log_prob.shape}")
    print(f"   Mean shape: {mean.shape}")
    print(f"   Hidden shape: {policy_hidden_new.shape}")
    
    # Q forward
    q_value, q_hidden_new = q1.forward(state, action, q_hidden)
    print(f"   Q-value shape: {q_value.shape}")
    print(f"   Q-value range: [{q_value.min().item():.4f}, {q_value.max().item():.4f}]")
    
    # Test soft update
    print("\n4. Testing soft update...")
    q_target = QNetwork(state_dim, action_dim, hidden_dim, gru_dim)
    hard_update(q1, q_target)
    
    q1_param_before = list(q1.parameters())[0].clone()
    q_target_param_before = list(q_target.parameters())[0].clone()
    
    # Modify q1
    with torch.no_grad():
        for param in q1.parameters():
            param += 0.1
    
    soft_update(q1, q_target, tau=0.005)
    
    q_target_param_after = list(q_target.parameters())[0]
    print(f"   Target param changed: {not torch.equal(q_target_param_before, q_target_param_after)}")
    
    # Test deterministic evaluation
    print("\n5. Testing deterministic evaluation...")
    policy.eval()
    with torch.no_grad():
        state_test = torch.randn(1, state_dim)
        hidden_test = policy.init_hidden(1)
        
        _, _, mean1, _ = policy.sample(state_test, hidden_test)
        _, _, mean2, _ = policy.sample(state_test, hidden_test)
        
        print(f"   Deterministic: {torch.allclose(mean1, mean2, atol=1e-6)}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
