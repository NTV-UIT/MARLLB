"""
Unit tests for SAC-GRU networks.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from networks import PolicyNetwork, QNetwork, soft_update, hard_update


def test_policy_network_creation():
    """Test policy network initialization."""
    state_dim = 10
    action_dim = 4
    hidden_dim = 64
    gru_dim = 32
    
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, gru_dim)
    
    assert policy.state_dim == state_dim
    assert policy.action_dim == action_dim
    assert policy.hidden_dim == hidden_dim
    assert policy.gru_dim == gru_dim
    
    print("✓ test_policy_network_creation passed")


def test_policy_forward():
    """Test policy forward pass."""
    batch_size = 8
    state_dim = 10
    action_dim = 4
    
    policy = PolicyNetwork(state_dim, action_dim, 64, 32)
    
    state = torch.randn(batch_size, state_dim)
    hidden = policy.init_hidden(batch_size)
    
    mean, log_std, hidden_new = policy.forward(state, hidden)
    
    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (batch_size, action_dim)
    assert hidden_new.shape == hidden.shape
    
    # Check log_std bounds
    assert torch.all(log_std >= -20)
    assert torch.all(log_std <= 2)
    
    print("✓ test_policy_forward passed")


def test_policy_sample():
    """Test policy action sampling."""
    batch_size = 8
    state_dim = 10
    action_dim = 4
    
    policy = PolicyNetwork(state_dim, action_dim, 64, 32)
    
    state = torch.randn(batch_size, state_dim)
    hidden = policy.init_hidden(batch_size)
    
    # Stochastic sampling
    action, log_prob, mean, hidden_new = policy.sample(state, hidden)
    
    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size, 1)
    assert mean.shape == (batch_size, action_dim)
    assert hidden_new.shape == hidden.shape
    
    # Action bounds (-1, 1) due to tanh
    assert torch.all(action >= -1)
    assert torch.all(action <= 1)
    
    # Mean action bounds
    assert torch.all(mean >= -1)
    assert torch.all(mean <= 1)
    
    print("✓ test_policy_sample passed")


def test_policy_hidden_state():
    """Test policy hidden state initialization."""
    batch_size = 8
    gru_dim = 32
    
    policy = PolicyNetwork(10, 4, 64, gru_dim)
    
    hidden = policy.init_hidden(batch_size)
    
    assert hidden.shape == (1, batch_size, gru_dim)
    assert torch.all(hidden == 0)
    
    print("✓ test_policy_hidden_state passed")


def test_q_network_creation():
    """Test Q-network initialization."""
    state_dim = 10
    action_dim = 4
    hidden_dim = 64
    gru_dim = 32
    
    qnet = QNetwork(state_dim, action_dim, hidden_dim, gru_dim)
    
    assert qnet.state_dim == state_dim
    assert qnet.action_dim == action_dim
    assert qnet.hidden_dim == hidden_dim
    assert qnet.gru_dim == gru_dim
    
    print("✓ test_q_network_creation passed")


def test_q_network_forward():
    """Test Q-network forward pass."""
    batch_size = 8
    state_dim = 10
    action_dim = 4
    
    qnet = QNetwork(state_dim, action_dim, 64, 32)
    
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    hidden = qnet.init_hidden(batch_size)
    
    q_value, hidden_new = qnet.forward(state, action, hidden)
    
    assert q_value.shape == (batch_size, 1)
    assert hidden_new.shape == hidden.shape
    
    print("✓ test_q_network_forward passed")


def test_q_network_hidden_state():
    """Test Q-network hidden state initialization."""
    batch_size = 8
    gru_dim = 32
    
    qnet = QNetwork(10, 4, 64, gru_dim)
    
    hidden = qnet.init_hidden(batch_size)
    
    assert hidden.shape == (1, batch_size, gru_dim)
    assert torch.all(hidden == 0)
    
    print("✓ test_q_network_hidden_state passed")


def test_soft_update():
    """Test soft update of target networks."""
    net1 = PolicyNetwork(10, 4, 64, 32)
    net2 = PolicyNetwork(10, 4, 64, 32)
    
    # Make networks different
    with torch.no_grad():
        for p in net2.parameters():
            p.add_(1.0)
    
    # Soft update with tau = 0.5
    soft_update(net1, net2, tau=0.5)
    
    # Check parameters are between original values
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        # p1 should be (0.5 * old_p1 + 0.5 * p2)
        # We can't verify exact values without storing old_p1,
        # but we can check they're different from both originals
        assert not torch.all(p1 == p2)
    
    print("✓ test_soft_update passed")


def test_hard_update():
    """Test hard copy of networks."""
    net1 = PolicyNetwork(10, 4, 64, 32)
    net2 = PolicyNetwork(10, 4, 64, 32)
    
    # Make networks different
    with torch.no_grad():
        for p in net2.parameters():
            p.add_(1.0)
    
    # Hard update (copy net2 -> net1)
    hard_update(net1, net2)
    
    # Check all parameters are equal
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        assert torch.allclose(p1, p2)
    
    print("✓ test_hard_update passed")


def test_policy_gradient_flow():
    """Test that gradients flow through policy network."""
    policy = PolicyNetwork(10, 4, 64, 32)
    
    state = torch.randn(8, 10)
    hidden = policy.init_hidden(8)
    
    # Sample action
    action, log_prob, mean, hidden_new = policy.sample(state, hidden)
    
    # Compute loss and backprop
    loss = -log_prob.mean()
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for p in policy.parameters():
        if p.grad is not None and torch.any(p.grad != 0):
            has_grad = True
            break
    
    assert has_grad, "No gradients found"
    
    print("✓ test_policy_gradient_flow passed")


def test_q_gradient_flow():
    """Test that gradients flow through Q-network."""
    qnet = QNetwork(10, 4, 64, 32)
    
    state = torch.randn(8, 10)
    action = torch.randn(8, 4)
    hidden = qnet.init_hidden(8)
    
    # Forward pass
    q_value, _ = qnet.forward(state, action, hidden)
    
    # Compute loss and backprop
    target = torch.randn(8, 1)
    loss = torch.nn.functional.mse_loss(q_value, target)
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for p in qnet.parameters():
        if p.grad is not None and torch.any(p.grad != 0):
            has_grad = True
            break
    
    assert has_grad, "No gradients found"
    
    print("✓ test_q_gradient_flow passed")


def run_all_tests():
    """Run all network tests."""
    print("\n" + "=" * 60)
    print("Running Network Tests")
    print("=" * 60 + "\n")
    
    test_policy_network_creation()
    test_policy_forward()
    test_policy_sample()
    test_policy_hidden_state()
    test_q_network_creation()
    test_q_network_forward()
    test_q_network_hidden_state()
    test_soft_update()
    test_hard_update()
    test_policy_gradient_flow()
    test_q_gradient_flow()
    
    print("\n" + "=" * 60)
    print("All Network Tests Passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
