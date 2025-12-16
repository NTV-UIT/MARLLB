"""
Random Policy Example

Demonstrates using a random policy with LoadBalanceEnv.
This is the simplest baseline to verify environment functionality.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from env import LoadBalanceEnv


def run_random_policy(num_episodes=5, max_steps=20):
    """
    Run random policy for specified number of episodes.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    # Create environment
    env = LoadBalanceEnv(
        num_servers=4,
        action_type='discrete',
        reward_metric='jain',
        reward_field='flow_duration_avg_decay',
        max_steps=max_steps,
        use_shm=False,  # Simulation mode
        seed=42
    )
    
    print("=" * 70)
    print("Random Policy Baseline")
    print("=" * 70)
    print(f"Environment: {env.num_servers} servers")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Reward function: {env.reward_fn}")
    print("=" * 70)
    
    episode_returns = []
    
    for episode in range(num_episodes):
        print(f"\n{'=' * 70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print("=" * 70)
        
        obs = env.reset()
        episode_reward = 0
        rewards_history = []
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            rewards_history.append(reward)
            
            # Print step info
            print(f"Step {step + 1:3d}: action={action}, "
                  f"weights={[f'{w:.1f}' for w in info['weights']]}, "
                  f"reward={reward:.4f}")
            
            obs = next_obs
            
            if done:
                break
        
        episode_returns.append(episode_reward)
        
        print(f"\nEpisode Summary:")
        print(f"  Total steps: {step + 1}")
        print(f"  Episode return: {episode_reward:.4f}")
        print(f"  Average reward: {np.mean(rewards_history):.4f}")
        print(f"  Reward std: {np.std(rewards_history):.4f}")
        print(f"  Min reward: {np.min(rewards_history):.4f}")
        print(f"  Max reward: {np.max(rewards_history):.4f}")
    
    # Overall statistics
    print(f"\n{'=' * 70}")
    print("Overall Statistics")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Average episode return: {np.mean(episode_returns):.4f} ± {np.std(episode_returns):.4f}")
    print(f"Min episode return: {np.min(episode_returns):.4f}")
    print(f"Max episode return: {np.max(episode_returns):.4f}")
    print("=" * 70)
    
    env.close()
    
    return episode_returns


def compare_reward_metrics():
    """
    Compare different reward metrics with random policy.
    """
    metrics = ['jain', 'variance', 'max', 'product']
    num_steps = 50
    
    print("\n" + "=" * 70)
    print("Comparing Reward Metrics with Random Policy")
    print("=" * 70)
    
    results = {}
    
    for metric in metrics:
        print(f"\n--- Testing metric: {metric} ---")
        
        env = LoadBalanceEnv(
            num_servers=4,
            reward_metric=metric,
            max_steps=num_steps,
            use_shm=False,
            seed=42
        )
        
        obs = env.reset()
        rewards = []
        
        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            
            if done:
                break
        
        results[metric] = rewards
        
        print(f"  Mean reward: {np.mean(rewards):.4f}")
        print(f"  Std reward: {np.std(rewards):.4f}")
        print(f"  Min reward: {np.min(rewards):.4f}")
        print(f"  Max reward: {np.max(rewards):.4f}")
        
        env.close()
    
    print("\n" + "=" * 70)
    print("Summary: All metrics tested successfully")
    print("=" * 70)
    
    return results


def test_discrete_vs_continuous():
    """
    Compare discrete vs continuous action spaces.
    """
    print("\n" + "=" * 70)
    print("Comparing Action Space Types")
    print("=" * 70)
    
    num_steps = 30
    
    # Discrete actions
    print("\n--- Discrete Actions ---")
    env_discrete = LoadBalanceEnv(
        num_servers=4,
        action_type='discrete',
        discrete_weights=[1.0, 1.5, 2.0],
        max_steps=num_steps,
        use_shm=False,
        seed=42
    )
    
    obs = env_discrete.reset()
    discrete_rewards = []
    
    for _ in range(num_steps):
        action = env_discrete.action_space.sample()
        obs, reward, done, info = env_discrete.step(action)
        discrete_rewards.append(reward)
        if done:
            break
    
    print(f"  Mean reward: {np.mean(discrete_rewards):.4f}")
    print(f"  Std reward: {np.std(discrete_rewards):.4f}")
    
    env_discrete.close()
    
    # Continuous actions
    print("\n--- Continuous Actions ---")
    env_continuous = LoadBalanceEnv(
        num_servers=4,
        action_type='continuous',
        min_weight=0.5,
        max_weight=2.5,
        max_steps=num_steps,
        use_shm=False,
        seed=42
    )
    
    obs = env_continuous.reset()
    continuous_rewards = []
    
    for _ in range(num_steps):
        action = env_continuous.action_space.sample()
        obs, reward, done, info = env_continuous.step(action)
        continuous_rewards.append(reward)
        if done:
            break
    
    print(f"  Mean reward: {np.mean(continuous_rewards):.4f}")
    print(f"  Std reward: {np.std(continuous_rewards):.4f}")
    
    env_continuous.close()
    
    print("\n" + "=" * 70)


def visualize_episode():
    """
    Run one episode with detailed visualization.
    """
    print("\n" + "=" * 70)
    print("Detailed Episode Visualization")
    print("=" * 70)
    
    env = LoadBalanceEnv(
        num_servers=4,
        action_type='discrete',
        reward_metric='jain',
        max_steps=10,
        use_shm=False,
        seed=42
    )
    
    obs = env.reset()
    
    print(f"\nInitial Observation Shape: {obs.shape}")
    print(f"Initial Observation:\n{obs}")
    
    for step in range(10):
        print(f"\n{'=' * 70}")
        print(f"Step {step + 1}")
        print("=" * 70)
        
        # Sample action
        action = env.action_space.sample()
        print(f"Action (indices): {action}")
        
        # Take step
        next_obs, reward, done, info = env.step(action)
        
        print(f"Weights: {info['weights']}")
        print(f"Active servers: {info['active_servers']}")
        print(f"Reward: {reward:.6f}")
        print(f"Done: {done}")
        
        # Show observation for first server
        print(f"\nServer 0 features:")
        feature_names = ['n_flow', 'fct_mean', 'fct_p90', 'fct_std', 
                        'fct_mean_decay', 'fct_p90_decay',
                        'dur_mean', 'dur_p90', 'dur_std', 
                        'dur_mean_decay', 'dur_avg_decay']
        for i, fname in enumerate(feature_names):
            print(f"  {fname:20s}: {next_obs[0, i]:8.4f}")
        
        obs = next_obs
        
        if done:
            print(f"\nEpisode finished!")
            print(f"Episode return: {info['episode']['r']:.4f}")
            print(f"Episode length: {info['episode']['l']}")
            break
    
    env.close()


if __name__ == '__main__':
    # Run demonstrations
    print("\n" + "=" * 70)
    print("LoadBalanceEnv - Random Policy Examples")
    print("=" * 70)
    
    # 1. Basic random policy
    print("\n[1] Running random policy baseline...")
    run_random_policy(num_episodes=3, max_steps=15)
    
    # 2. Compare reward metrics
    print("\n[2] Comparing reward metrics...")
    compare_reward_metrics()
    
    # 3. Compare action types
    print("\n[3] Comparing action space types...")
    test_discrete_vs_continuous()
    
    # 4. Detailed visualization
    print("\n[4] Detailed episode visualization...")
    visualize_episode()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
