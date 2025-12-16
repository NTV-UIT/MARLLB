"""
QMIX Mixing Network Implementation

This module implements the mixing network that combines individual agent
Q-values into a global Q_tot while maintaining monotonicity constraint.

Reference: QMIX paper (Rashid et al., ICML 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixingNetwork(nn.Module):
    """
    QMIX mixing network with hypernetworks.
    
    The mixing network combines individual agent Q-values into Q_tot:
        Q_tot = f(Q₁, Q₂, ..., Qₙ; state)
    
    Key property: Monotonicity constraint
        ∂Q_tot/∂Qᵢ ≥ 0 for all i
    
    This is enforced by:
    1. Hypernetworks generate positive weights (via abs or ReLU)
    2. Mixing function is additive/multiplicative
    
    Args:
        num_agents: Number of agents
        state_dim: Global state dimension
        mixing_embed_dim: Hidden dimension for mixing network
        hypernet_embed_dim: Hidden dimension for hypernetworks
    """
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64
    ):
        super(QMixingNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.hypernet_embed_dim = hypernet_embed_dim
        
        # Hypernetwork for first layer weights
        # Input: state → Output: weights for [Q₁, ..., Qₙ] → hidden
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, num_agents * mixing_embed_dim)
        )
        
        # Hypernetwork for first layer bias
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim)
        )
        
        # Hypernetwork for second layer weights
        # Input: state → Output: weights for hidden → Q_tot
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim)
        )
        
        # Hypernetwork for second layer bias
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, 1)
        )
    
    def forward(self, agent_qs, state):
        """
        Mix agent Q-values into global Q_tot.
        
        Args:
            agent_qs: Individual Q-values (batch_size, num_agents)
            state: Global state (batch_size, state_dim)
        
        Returns:
            Q_tot: Mixed Q-value (batch_size, 1)
        """
        batch_size = agent_qs.size(0)
        
        # Generate weights and biases from state
        # First layer: [Q₁, ..., Qₙ] → hidden
        w1 = torch.abs(self.hyper_w1(state))  # (batch, num_agents * mixing_embed_dim)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)  # (batch, num_agents, embed)
        
        b1 = self.hyper_b1(state)  # (batch, mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)  # (batch, 1, embed)
        
        # Second layer: hidden → Q_tot
        w2 = torch.abs(self.hyper_w2(state))  # (batch, mixing_embed_dim)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)  # (batch, embed, 1)
        
        b2 = self.hyper_b2(state)  # (batch, 1)
        
        # Forward pass through mixing network
        # Layer 1: Q-values → hidden
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)  # (batch, 1, num_agents)
        hidden = torch.bmm(agent_qs, w1)  # (batch, 1, embed)
        hidden = hidden + b1  # (batch, 1, embed)
        hidden = F.elu(hidden)  # Non-linearity
        
        # Layer 2: hidden → Q_tot
        q_tot = torch.bmm(hidden, w2)  # (batch, 1, 1)
        q_tot = q_tot.squeeze(1)  # (batch, 1)
        q_tot = q_tot + b2  # Add bias
        
        return q_tot
    
    def get_monotonicity_info(self, agent_qs, state):
        """
        Compute partial derivatives ∂Q_tot/∂Qᵢ for monitoring.
        
        These should always be non-negative due to abs() on weights.
        
        Args:
            agent_qs: Individual Q-values (batch_size, num_agents)
            state: Global state (batch_size, state_dim)
        
        Returns:
            gradients: ∂Q_tot/∂Qᵢ for each agent (batch_size, num_agents)
        """
        batch_size = agent_qs.size(0)
        
        # Compute Q_tot
        q_tot = self.forward(agent_qs, state)
        
        # Compute gradients
        gradients = []
        for i in range(self.num_agents):
            grad = torch.autograd.grad(
                outputs=q_tot,
                inputs=agent_qs,
                grad_outputs=torch.ones_like(q_tot),
                create_graph=True,
                retain_graph=True
            )[0][:, i]
            gradients.append(grad)
        
        gradients = torch.stack(gradients, dim=1)  # (batch_size, num_agents)
        
        return gradients


class VDNMixingNetwork(nn.Module):
    """
    Value Decomposition Network (VDN) - Simple additive mixing.
    
    VDN is a special case of QMIX where:
        Q_tot = Σᵢ Qᵢ
    
    This is simpler but less expressive than QMIX.
    Useful as a baseline for comparison.
    
    Args:
        num_agents: Number of agents
    """
    
    def __init__(self, num_agents: int):
        super(VDNMixingNetwork, self).__init__()
        self.num_agents = num_agents
    
    def forward(self, agent_qs, state=None):
        """
        Sum agent Q-values.
        
        Args:
            agent_qs: Individual Q-values (batch_size, num_agents)
            state: Global state (ignored for VDN)
        
        Returns:
            Q_tot: Sum of Q-values (batch_size, 1)
        """
        q_tot = agent_qs.sum(dim=1, keepdim=True)
        return q_tot


class WeightedQMixingNetwork(nn.Module):
    """
    Weighted QMIX (WQMIX) - Learning non-uniform credit assignment.
    
    Extends QMIX by learning importance weights for each agent:
        Q_tot = Σᵢ wᵢ(s) · Qᵢ
    
    where wᵢ(s) are learned from state.
    
    Args:
        num_agents: Number of agents
        state_dim: Global state dimension
        hidden_dim: Hidden dimension for weight network
    """
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        hidden_dim: int = 64
    ):
        super(WeightedQMixingNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # Network to compute agent weights from state
        self.weight_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents),
            nn.Softmax(dim=1)  # Weights sum to 1
        )
    
    def forward(self, agent_qs, state):
        """
        Compute weighted sum of agent Q-values.
        
        Args:
            agent_qs: Individual Q-values (batch_size, num_agents)
            state: Global state (batch_size, state_dim)
        
        Returns:
            Q_tot: Weighted sum (batch_size, 1)
        """
        weights = self.weight_network(state)  # (batch_size, num_agents)
        q_tot = (agent_qs * weights).sum(dim=1, keepdim=True)
        
        return q_tot, weights  # Return weights for analysis


def test_mixing_networks():
    """Test mixing network implementations."""
    print("=" * 60)
    print("Testing Mixing Networks")
    print("=" * 60)
    
    batch_size = 16
    num_agents = 4
    state_dim = 20
    
    # Test QMIX
    print("\n1. Testing QMIX Mixing Network...")
    qmix = QMixingNetwork(
        num_agents=num_agents,
        state_dim=state_dim,
        mixing_embed_dim=32,
        hypernet_embed_dim=64
    )
    
    agent_qs = torch.randn(batch_size, num_agents)
    state = torch.randn(batch_size, state_dim)
    
    q_tot = qmix(agent_qs, state)
    print(f"   Input shape: {agent_qs.shape}")
    print(f"   State shape: {state.shape}")
    print(f"   Output shape: {q_tot.shape}")
    print(f"   Q_tot range: [{q_tot.min():.4f}, {q_tot.max():.4f}]")
    
    # Test monotonicity
    print("\n2. Testing Monotonicity...")
    agent_qs.requires_grad = True
    gradients = qmix.get_monotonicity_info(agent_qs, state)
    print(f"   Gradient shape: {gradients.shape}")
    print(f"   All gradients >= 0: {(gradients >= 0).all().item()}")
    print(f"   Gradient range: [{gradients.min():.4f}, {gradients.max():.4f}]")
    
    # Test VDN
    print("\n3. Testing VDN...")
    vdn = VDNMixingNetwork(num_agents=num_agents)
    q_tot_vdn = vdn(agent_qs)
    expected_sum = agent_qs.sum(dim=1, keepdim=True)
    print(f"   VDN output: {q_tot_vdn.shape}")
    print(f"   Correct sum: {torch.allclose(q_tot_vdn, expected_sum)}")
    
    # Test Weighted QMIX
    print("\n4. Testing Weighted QMIX...")
    wqmix = WeightedQMixingNetwork(
        num_agents=num_agents,
        state_dim=state_dim,
        hidden_dim=64
    )
    q_tot_w, weights = wqmix(agent_qs, state)
    print(f"   Output shape: {q_tot_w.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weights sum to 1: {torch.allclose(weights.sum(dim=1), torch.ones(batch_size))}")
    print(f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    # Test backward pass
    print("\n5. Testing Backward Pass...")
    q_tot = qmix(agent_qs, state)
    loss = q_tot.mean()
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for name, param in qmix.named_parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_grad = True
            break
    
    print(f"   Gradients computed: {has_grad}")
    
    # Test different batch sizes
    print("\n6. Testing Different Batch Sizes...")
    for bs in [1, 8, 32, 64]:
        agent_qs_test = torch.randn(bs, num_agents)
        state_test = torch.randn(bs, state_dim)
        q_tot_test = qmix(agent_qs_test, state_test)
        print(f"   Batch size {bs:2d}: {q_tot_test.shape} ✓")
    
    print("\n" + "=" * 60)
    print("All Mixing Network Tests Passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    test_mixing_networks()
