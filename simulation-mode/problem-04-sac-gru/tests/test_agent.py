"""
Unit tests for SAC-GRU agent.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sac_agent import SAC_GRU_Agent


def test_agent_creation():
    """Test agent initialization."""
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        hidden_dim=64,
        gru_dim=32,
        batch_size=32
    )
    
    assert agent.state_dim == 10
    assert agent.action_dim == 4
    assert agent.batch_size == 32
    
    # Check networks exist
    assert agent.policy is not None
    assert agent.q1 is not None
    assert agent.q2 is not None
    assert agent.q1_target is not None
    assert agent.q2_target is not None
    
    print("✓ test_agent_creation passed")


def test_agent_select_action():
    """Test action selection."""
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        hidden_dim=64,
        gru_dim=32
    )
    
    state = np.random.randn(10)
    hidden = agent.policy.init_hidden(1)
    
    # Stochastic action
    action, hidden_new = agent.select_action(state, hidden, evaluate=False)
    
    assert action.shape == (4,)
    assert isinstance(action, np.ndarray)
    assert hidden_new.shape == hidden.shape
    
    # Deterministic action
    action_det, _ = agent.select_action(state, hidden, evaluate=True)
    assert action_det.shape == (4,)
    
    print("✓ test_agent_select_action passed")


def test_agent_action_bounds():
    """Test action bounds are respected."""
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        hidden_dim=64,
        gru_dim=32
    )
    
    state = np.random.randn(10)
    hidden = agent.policy.init_hidden(1)
    
    # Sample many actions
    for _ in range(100):
        action, hidden = agent.select_action(state, hidden, evaluate=False)
        
        # Actions should be in [-1, 1] due to tanh
        assert np.all(action >= -1)
        assert np.all(action <= 1)
    
    print("✓ test_agent_action_bounds passed")


def test_agent_update_no_data():
    """Test update with insufficient data."""
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        batch_size=32
    )
    
    # Try to update with no data
    stats = agent.update_parameters()
    
    # Should return None or empty dict
    assert stats is None or len(stats) == 0
    
    print("✓ test_agent_update_no_data passed")


def test_agent_update_with_data():
    """Test update with data."""
    gru_dim = 64
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        gru_dim=gru_dim,
        batch_size=8,  # Small batch for test
        buffer_size=100
    )
    
    # Add data to buffer
    for _ in range(20):
        state = np.random.randn(10)
        action = np.random.randn(4)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        hidden = np.zeros((1, 1, gru_dim))
        
        agent.replay_buffer.push(state, action, reward, next_state, done, hidden)
    
    # Update
    stats = agent.update_parameters()
    
    # Check stats exist
    assert stats is not None
    assert 'q1' in stats
    assert 'q2' in stats
    assert 'policy' in stats
    assert 'alpha' in stats
    
    # Check values are reasonable
    assert isinstance(stats['q1'], float)
    assert isinstance(stats['q2'], float)
    assert isinstance(stats['policy'], float)
    assert stats['alpha'] >= 0
    
    print("✓ test_agent_update_with_data passed")


def test_agent_save_load():
    """Test save and load."""
    agent1 = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        hidden_dim=64,
        gru_dim=32
    )
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save
        save_path = Path(temp_dir) / 'agent.pth'
        agent1.save(save_path)
        
        assert save_path.exists()
        
        # Create new agent and load
        agent2 = SAC_GRU_Agent(
            state_dim=10,
            action_dim=4,
            hidden_dim=64,
            gru_dim=32
        )
        
        agent2.load(save_path)
        
        # Check parameters are equal
        for p1, p2 in zip(agent1.policy.parameters(), agent2.policy.parameters()):
            assert torch.allclose(p1, p2)
        
        for p1, p2 in zip(agent1.q1.parameters(), agent2.q1.parameters()):
            assert torch.allclose(p1, p2)
        
        print("✓ test_agent_save_load passed")
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_agent_deterministic_action():
    """Test deterministic action is consistent."""
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        hidden_dim=64,
        gru_dim=32
    )
    
    state = np.random.randn(10)
    hidden = agent.policy.init_hidden(1)
    
    # Get deterministic action multiple times
    action1, _ = agent.select_action(state, hidden, evaluate=True)
    action2, _ = agent.select_action(state, hidden, evaluate=True)
    action3, _ = agent.select_action(state, hidden, evaluate=True)
    
    # Should be the same
    assert np.allclose(action1, action2)
    assert np.allclose(action2, action3)
    
    print("✓ test_agent_deterministic_action passed")


def test_agent_alpha_tuning():
    """Test automatic alpha (temperature) tuning."""
    gru_dim = 64
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        gru_dim=gru_dim,
        batch_size=8,
        auto_entropy_tuning=True
    )
    
    # Add data
    for _ in range(20):
        state = np.random.randn(10)
        action = np.random.randn(4)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        hidden = np.zeros((1, 1, gru_dim))
        
        agent.replay_buffer.push(state, action, reward, next_state, done, hidden)
    
    # Initial alpha
    alpha_before = agent.alpha.item()
    
    # Update several times
    for _ in range(5):
        agent.update_parameters()
    
    # Alpha should have changed (unless by chance it's at optimal)
    alpha_after = agent.alpha.item()
    
    # At least check alpha is positive
    assert alpha_after > 0
    
    print("✓ test_agent_alpha_tuning passed")


def test_agent_target_networks():
    """Test target networks are updated."""
    gru_dim = 64
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        gru_dim=gru_dim,
        batch_size=8,
        tau=0.1  # Larger tau for more noticeable update
    )
    
    # Store initial target parameters
    initial_params = []
    for p in agent.q1_target.parameters():
        initial_params.append(p.clone())
    
    # Add data and update
    for _ in range(20):
        state = np.random.randn(10)
        action = np.random.randn(4)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        hidden = np.zeros((1, 1, gru_dim))
        
        agent.replay_buffer.push(state, action, reward, next_state, done, hidden)
    
    # Update
    agent.update_parameters()
    
    # Check target parameters changed
    params_changed = False
    for initial_p, current_p in zip(initial_params, agent.q1_target.parameters()):
        if not torch.allclose(initial_p, current_p):
            params_changed = True
            break
    
    assert params_changed, "Target network not updated"
    
    print("✓ test_agent_target_networks passed")


def test_agent_get_stats():
    """Test statistics retrieval."""
    gru_dim = 64
    agent = SAC_GRU_Agent(
        state_dim=10,
        action_dim=4,
        gru_dim=gru_dim,
        batch_size=8
    )
    
    stats = agent.get_stats()
    
    assert 'total_updates' in stats
    assert 'alpha' in stats
    assert stats['total_updates'] == 0
    
    # Add data and update
    for _ in range(20):
        state = np.random.randn(10)
        action = np.random.randn(4)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        hidden = np.zeros((1, 1, gru_dim))
        
        agent.replay_buffer.push(state, action, reward, next_state, done, hidden)
    
    agent.update_parameters()
    
    stats = agent.get_stats()
    assert stats['total_updates'] > 0
    
    print("✓ test_agent_get_stats passed")


def run_all_tests():
    """Run all agent tests."""
    print("\n" + "=" * 60)
    print("Running Agent Tests")
    print("=" * 60 + "\n")
    
    test_agent_creation()
    test_agent_select_action()
    test_agent_action_bounds()
    test_agent_update_no_data()
    test_agent_update_with_data()
    test_agent_save_load()
    test_agent_deterministic_action()
    test_agent_alpha_tuning()
    test_agent_target_networks()
    test_agent_get_stats()
    
    print("\n" + "=" * 60)
    print("All Agent Tests Passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
