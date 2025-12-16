"""
Tests for Problem 06: VPP Integration

Test các Python components:
- RL Controller
- Training Pipeline
- Shared Memory Communication
- Agent Integration
"""

import sys
import os
import time
import numpy as np
import tempfile
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-03-rl-environment' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-04-sac-gru' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-05-qmix' / 'src'))


def test_alias_table():
    """Test alias table construction cho O(1) sampling."""
    print("\n=== Test 1: Alias Table Construction ===")
    
    from rl_controller import RLController
    
    # Create mock controller
    with tempfile.NamedTemporaryFile() as tmp:
        controller = RLController(
            agent_type='qmix',
            num_servers=4,
            num_agents=2,
            shm_path=tmp.name,
            online_training=False
        )
        
        # Test weights
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Build alias table
        alias_table = controller._build_alias_table(weights)
        
        print(f"Weights: {weights}")
        print(f"Alias table:")
        for i, (prob, alias) in enumerate(alias_table):
            print(f"  [{i}] prob={prob:.3f}, alias={alias}")
        
        # Test sampling distribution
        num_samples = 10000
        samples = []
        
        for _ in range(num_samples):
            i = np.random.randint(0, len(weights))
            u = np.random.uniform()
            
            if u < alias_table[i][0]:
                samples.append(i)
            else:
                samples.append(alias_table[i][1])
        
        # Check distribution
        sample_counts = np.bincount(samples, minlength=len(weights))
        sample_freqs = sample_counts / num_samples
        
        print(f"\nSampled distribution (n={num_samples}):")
        print(f"  Expected: {weights}")
        print(f"  Actual:   {sample_freqs}")
        print(f"  Error:    {np.abs(weights - sample_freqs)}")
        
        # Assert close to expected
        assert np.allclose(weights, sample_freqs, atol=0.01), \
            "Sampled distribution should match weights"
        
        print("✓ Alias table sampling correct")


def test_stats_to_observation():
    """Test conversion từ VPP stats sang RL observation."""
    print("\n=== Test 2: Stats to Observation Conversion ===")
    
    from rl_controller import RLController
    
    with tempfile.NamedTemporaryFile() as tmp:
        # QMIX controller
        controller = RLController(
            agent_type='qmix',
            num_servers=16,
            num_agents=4,
            shm_path=tmp.name,
            online_training=False
        )
        
        # Mock stats
        msg_out = {
            'id': 1,
            'timestamp': time.time(),
            'server_stats': [
                {
                    'as_index': i,
                    'n_flow_on': 10 + i,
                    'cpu_util': 0.5 + i * 0.01,
                    'queue_depth': 5 + i,
                    'response_time': 10.0 + i
                }
                for i in range(16)
            ]
        }
        
        # Convert
        observations = controller._stats_to_observation(msg_out)
        
        print(f"Num agents: {len(observations)}")
        for i, obs in enumerate(observations):
            print(f"Agent {i}: shape={obs.shape}, range=[{obs.min():.3f}, {obs.max():.3f}]")
        
        assert len(observations) == 4, "Should have 4 agent observations"
        assert all(obs.shape[0] == 18 for obs in observations), \
            "Each observation should have 18 features (4 servers × 4 + 2 global)"
        
        print("✓ Stats to observation conversion correct")


def test_action_to_weights():
    """Test conversion từ RL actions sang server weights."""
    print("\n=== Test 3: Action to Weights Conversion ===")
    
    from rl_controller import RLController
    
    with tempfile.NamedTemporaryFile() as tmp:
        controller = RLController(
            agent_type='qmix',
            num_servers=16,
            num_agents=4,
            shm_path=tmp.name,
            online_training=False
        )
        
        # Mock observation (20 dims per agent)
        observations = [np.random.randn(20).astype(np.float32) for _ in range(4)]
        
        # Get weights
        weights = controller._get_action(observations)
        
        print(f"Weights shape: {weights.shape}")
        print(f"Weights: {weights}")
        print(f"Sum: {weights.sum():.6f}")
        print(f"Min: {weights.min():.6f}, Max: {weights.max():.6f}")
        
        assert weights.shape == (16,), "Should have 16 weights"
        assert abs(weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
        assert np.all(weights >= 0), "Weights should be non-negative"
        
        print("✓ Action to weights conversion correct")


def test_reward_computation():
    """Test reward computation từ server stats."""
    print("\n=== Test 4: Reward Computation ===")
    
    from rl_controller import RLController
    
    with tempfile.NamedTemporaryFile() as tmp:
        controller = RLController(
            agent_type='sac-gru',
            num_servers=16,
            shm_path=tmp.name,
            online_training=True
        )
        
        # Balanced load (should give high reward)
        balanced_stats = {
            'server_stats': [
                {
                    'as_index': i,
                    'n_flow_on': 50,
                    'cpu_util': 0.5,
                    'queue_depth': 10,
                    'response_time': 10.0
                }
                for i in range(16)
            ]
        }
        
        reward_balanced = controller._compute_reward(balanced_stats)
        
        # Imbalanced load (should give low reward)
        imbalanced_stats = {
            'server_stats': [
                {
                    'as_index': i,
                    'n_flow_on': 10 if i < 8 else 90,
                    'cpu_util': 0.2 if i < 8 else 0.9,
                    'queue_depth': 5 if i < 8 else 50,
                    'response_time': 5.0 if i < 8 else 50.0
                }
                for i in range(16)
            ]
        }
        
        reward_imbalanced = controller._compute_reward(imbalanced_stats)
        
        print(f"Balanced load reward: {reward_balanced:.4f}")
        print(f"Imbalanced load reward: {reward_imbalanced:.4f}")
        print(f"Difference: {reward_balanced - reward_imbalanced:.4f}")
        
        assert reward_balanced > reward_imbalanced, \
            "Balanced load should give higher reward"
        
        print("✓ Reward computation correct")


def test_controller_integration():
    """Test full controller integration."""
    print("\n=== Test 5: Controller Integration ===")
    
    from rl_controller import RLController
    
    with tempfile.NamedTemporaryFile() as tmp:
        # Create controller
        controller = RLController(
            agent_type='qmix',
            num_servers=16,
            num_agents=4,
            shm_path=tmp.name,
            online_training=False
        )
        
        # Mock stats in shared memory
        msg_out = {
            'id': 0,
            'timestamp': time.time(),
            'server_stats': [
                {
                    'as_index': i,
                    'n_flow_on': 10,
                    'cpu_util': 0.5,
                    'queue_depth': 5,
                    'response_time': 10.0
                }
                for i in range(64)
            ]
        }
        controller.shm.write_msg_out(msg_out)
        
        # Read and process
        msg_out_read = controller.shm.read_msg_out()
        observation = controller._stats_to_observation(msg_out_read)
        weights = controller._get_action(observation)
        controller._write_action(weights, msg_out_read['id'])
        
        # Read back
        msg_in = controller.shm.read_msg_in()
        
        print(f"Wrote msg_out id: {msg_out['id']}")
        print(f"Read msg_out id: {msg_out_read['id']}")
        print(f"Wrote msg_in id: {msg_in['id']}")
        print(f"Server weights sum: {msg_in['server_weights'].sum():.6f}")
        print(f"Alias table length: {len(msg_in['alias_table'])}")
        
        assert msg_in['id'] == msg_out['id'] + 1, "Message IDs should increment"
        assert abs(msg_in['server_weights'].sum() - 1.0) < 1e-6, "Weights should sum to 1"
        assert len(msg_in['alias_table']) == controller.num_servers, \
            "Alias table should have entry for each server"
        
        print("✓ Controller integration working")


def test_training_pipeline():
    """Test training pipeline."""
    print("\n=== Test 6: Training Pipeline ===")
    
    from training_pipeline import TrainingPipeline
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create pipeline
        pipeline = TrainingPipeline(
            agent_type='sac-gru',
            num_servers=4,
            trace_dir='data/trace',
            checkpoint_dir=tmpdir
        )
        
        print(f"Loaded {len(pipeline.traces)} traces")
        
        # Run few training episodes
        print("\nTraining for 5 episodes...")
        
        for episode in range(5):
            # Select trace by index
            trace_idx = np.random.randint(0, len(pipeline.traces))
            trace = pipeline.traces[trace_idx]
            reward, length, loss = pipeline._run_episode(trace, episode)
            
            print(f"  Episode {episode}: reward={reward:.2f}, length={length}, loss={loss}")
        
        # Test evaluation
        print("\nEvaluating...")
        eval_reward = pipeline._evaluate(num_episodes=3)
        print(f"  Average eval reward: {eval_reward:.2f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')
        pipeline.agent.save(checkpoint_path)
        
        assert os.path.exists(checkpoint_path), "Checkpoint should be saved"
        
        print("✓ Training pipeline working")


def test_sac_gru_integration():
    """Test SAC-GRU agent integration."""
    print("\n=== Test 7: SAC-GRU Integration ===")
    
    from rl_controller import RLController
    
    with tempfile.NamedTemporaryFile() as tmp:
        controller = RLController(
            agent_type='sac-gru',
            num_servers=16,
            shm_path=tmp.name,
            online_training=False
        )
        
        # Generate random observation
        obs = np.random.randn(74).astype(np.float32)
        
        # Get action
        weights = controller._get_action(obs)
        
        print(f"SAC-GRU weights: {weights}")
        print(f"Sum: {weights.sum():.6f}")
        print(f"Entropy: {-np.sum(weights * np.log(weights + 1e-8)):.3f}")
        
        assert weights.shape == (16,), "Should have 16 weights"
        assert abs(weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
        
        print("✓ SAC-GRU integration working")


def test_qmix_integration():
    """Test QMIX agent integration."""
    print("\n=== Test 8: QMIX Integration ===")
    
    from rl_controller import RLController
    
    with tempfile.NamedTemporaryFile() as tmp:
        controller = RLController(
            agent_type='qmix',
            num_servers=16,
            num_agents=4,
            shm_path=tmp.name,
            online_training=False
        )
        
        # Generate random observations (20 dims per agent)
        observations = [np.random.randn(20).astype(np.float32) for _ in range(4)]
        
        # Get action
        weights = controller._get_action(observations)
        
        print(f"QMIX weights: {weights}")
        print(f"Sum: {weights.sum():.6f}")
        print(f"Non-zero servers: {np.count_nonzero(weights)}")
        
        assert weights.shape == (16,), "Should have 16 weights"
        assert abs(weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
        
        print("✓ QMIX integration working")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Problem 06: VPP Integration Tests")
    print("=" * 70)
    
    tests = [
        test_alias_table,
        test_stats_to_observation,
        test_action_to_weights,
        test_reward_computation,
        test_controller_integration,
        test_training_pipeline,
        test_sac_gru_integration,
        test_qmix_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
