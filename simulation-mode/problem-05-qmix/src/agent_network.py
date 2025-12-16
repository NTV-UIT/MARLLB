"""
Agent Q-Network for QMIX

This network outputs Q-values for all actions (unlike SAC's Q-network which
evaluates a specific state-action pair).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentQNetwork(nn.Module):
    """
    Agent Q-network for QMIX.
    
    Outputs Q-values for all actions given observation and hidden state.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension (number of discrete actions)
        hidden_dim: Hidden layer dimension
        gru_dim: GRU hidden state dimension
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gru_dim: int = 64
    ):
        super(AgentQNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim
        
        # GRU layer
        self.gru = nn.GRU(obs_dim, gru_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(gru_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # Output Q for each action
        
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
    
    def forward(self, obs, hidden):
        """
        Forward pass through Q-network.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim)
            hidden: GRU hidden state (1, batch_size, gru_dim)
        
        Returns:
            q_values: Q-values for all actions (batch_size, action_dim)
            hidden_new: Updated hidden state (1, batch_size, gru_dim)
        """
        # Add sequence dimension
        obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
        
        # GRU forward
        gru_out, hidden_new = self.gru(obs, hidden)
        gru_out = gru_out.squeeze(1)  # (batch_size, gru_dim)
        
        # Fully connected layers
        x = F.relu(self.fc1(gru_out))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # (batch_size, action_dim)
        
        return q_values, hidden_new
    
    def init_hidden(self, batch_size: int = 1):
        """Initialize GRU hidden state."""
        return torch.zeros(1, batch_size, self.gru_dim)
    
    def to(self, device):
        """Move network to device."""
        return super(AgentQNetwork, self).to(device)


if __name__ == '__main__':
    # Test agent Q-network
    print("=" * 60)
    print("Testing Agent Q-Network")
    print("=" * 60)
    
    batch_size = 8
    obs_dim = 20
    action_dim = 4
    hidden_dim = 64
    gru_dim = 32
    
    # Create network
    print("\n1. Creating network...")
    net = AgentQNetwork(obs_dim, action_dim, hidden_dim, gru_dim)
    print(f"   Obs dim: {obs_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   GRU dim: {gru_dim}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    obs = torch.randn(batch_size, obs_dim)
    hidden = net.init_hidden(batch_size)
    
    q_values, hidden_new = net.forward(obs, hidden)
    
    print(f"   Input shape: {obs.shape}")
    print(f"   Hidden shape: {hidden.shape}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   New hidden shape: {hidden_new.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    
    # Test action selection
    print("\n3. Testing action selection...")
    actions = q_values.argmax(dim=1)
    print(f"   Actions shape: {actions.shape}")
    print(f"   Actions: {actions.tolist()}")
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    loss = q_values.mean()
    loss.backward()
    
    has_grad = False
    for param in net.parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_grad = True
            break
    
    print(f"   Gradients computed: {has_grad}")
    
    # Test different batch sizes
    print("\n5. Testing different batch sizes...")
    for bs in [1, 4, 16, 32]:
        obs_test = torch.randn(bs, obs_dim)
        hidden_test = net.init_hidden(bs)
        q_test, _ = net.forward(obs_test, hidden_test)
        print(f"   Batch size {bs:2d}: {q_test.shape} ✓")
    
    print("\n" + "=" * 60)
    print("All Agent Q-Network Tests Passed! ✓")
    print("=" * 60)
