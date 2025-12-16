"""
Training utilities and trainer class for SAC-GRU
"""

import sys
from pathlib import Path
import time
import numpy as np
from collections import deque

# Add path for environment
sys.path.append(str(Path(__file__).parent.parent.parent / 'problem-03-rl-environment' / 'src'))

try:
    from env import LoadBalanceEnv
except ImportError:
    LoadBalanceEnv = None
    print("Warning: LoadBalanceEnv not found")

from sac_agent import SAC_GRU_Agent


class Trainer:
    """
    Training loop for SAC-GRU agent.
    """
    
    def __init__(
        self,
        env,
        agent: SAC_GRU_Agent,
        max_episodes: int = 1000,
        max_steps: int = 100,
        start_steps: int = 10000,
        updates_per_step: int = 1,
        eval_interval: int = 10,
        save_interval: int = 100,
        save_dir: str = './checkpoints',
        log_interval: int = 1
    ):
        """
        Initialize trainer.
        
        Args:
            env: Environment
            agent: SAC-GRU agent
            max_episodes: Maximum training episodes
            max_steps: Maximum steps per episode
            start_steps: Random exploration steps before training
            updates_per_step: Number of gradient updates per step
            eval_interval: Episodes between evaluations
            save_interval: Episodes between saves
            save_dir: Directory to save checkpoints
            log_interval: Episodes between logging
        """
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.start_steps = start_steps
        self.updates_per_step = updates_per_step
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.total_steps = 0
    
    def train(self):
        """Run training loop."""
        print("=" * 70)
        print("Starting SAC-GRU Training")
        print("=" * 70)
        print(f"Max episodes: {self.max_episodes}")
        print(f"Max steps per episode: {self.max_steps}")
        print(f"Start steps (random): {self.start_steps}")
        print(f"Updates per step: {self.updates_per_step}")
        print(f"Device: {self.agent.device}")
        print("=" * 70)
        
        start_time = time.time()
        recent_rewards = deque(maxlen=100)
        
        for episode in range(1, self.max_episodes + 1):
            episode_reward = 0
            episode_steps = 0
            
            # Reset environment and hidden state
            state = self.env.reset()
            if len(state.shape) > 1:
                state = state.flatten()
            
            hidden = self.agent.policy.init_hidden(1).to(self.agent.device)
            
            for step in range(self.max_steps):
                # Select action
                if self.total_steps < self.start_steps:
                    # Random exploration
                    action = self.env.action_space.sample()
                    hidden_new = hidden  # Don't update hidden for random actions
                else:
                    action, hidden_new = self.agent.select_action(state, hidden, evaluate=False)
                
                # Environment step
                next_state, reward, done, info = self.env.step(action)
                if len(next_state.shape) > 1:
                    next_state = next_state.flatten()
                
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                
                # Store transition
                self.agent.replay_buffer.push(state, action, reward, next_state, done, hidden.cpu().numpy())
                
                # Update parameters
                if self.total_steps >= self.start_steps:
                    for _ in range(self.updates_per_step):
                        self.agent.update_parameters()
                
                state = next_state
                hidden = hidden_new
                
                if done:
                    break
            
            # Episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            recent_rewards.append(episode_reward)
            
            # Logging
            if episode % self.log_interval == 0:
                avg_reward = np.mean(recent_rewards)
                elapsed_time = time.time() - start_time
                
                print(f"Episode {episode:4d} | "
                      f"Steps: {episode_steps:3d} | "
                      f"Reward: {episode_reward:7.4f} | "
                      f"Avg(100): {avg_reward:7.4f} | "
                      f"Alpha: {self.agent.alpha.item():.4f} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # Evaluation
            if episode % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                print(f"--- Evaluation | Reward: {eval_reward:.4f} ---")
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                checkpoint_path = self.save_dir / f"agent_episode_{episode}.pth"
                self.agent.save(checkpoint_path)
        
        # Final save
        final_path = self.save_dir / "agent_final.pth"
        self.agent.save(final_path)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Total steps: {self.total_steps}")
        print(f"Final avg reward (100 episodes): {np.mean(recent_rewards):.4f}")
        print("=" * 70)
    
    def evaluate(self, num_episodes: int = 5):
        """
        Evaluate agent performance.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Average reward over evaluation episodes
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            if len(state.shape) > 1:
                state = state.flatten()
            
            hidden = self.agent.policy.init_hidden(1).to(self.agent.device)
            episode_reward = 0
            
            for step in range(self.max_steps):
                action, hidden = self.agent.select_action(state, hidden, evaluate=True)
                next_state, reward, done, info = self.env.step(action)
                
                if len(next_state.shape) > 1:
                    next_state = next_state.flatten()
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def get_stats(self):
        """Get training statistics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eval_rewards': self.eval_rewards,
            'total_steps': self.total_steps,
            'agent_stats': self.agent.get_stats()
        }


if __name__ == '__main__':
    # Test trainer
    print("Testing Trainer")
    print("=" * 60)
    
    if LoadBalanceEnv is None:
        print("LoadBalanceEnv not available, skipping test")
    else:
        # Create environment
        env = LoadBalanceEnv(
            num_servers=4,
            action_type='continuous',
            reward_metric='jain',
            max_steps=20,
            use_shm=False
        )
        
        # Create agent
        agent = SAC_GRU_Agent(
            state_dim=44,
            action_dim=4,
            hidden_dim=128,
            gru_dim=64,
            batch_size=32
        )
        
        # Create trainer
        trainer = Trainer(
            env=env,
            agent=agent,
            max_episodes=5,
            max_steps=20,
            start_steps=50,
            updates_per_step=1,
            eval_interval=2,
            save_interval=5,
            log_interval=1
        )
        
        # Run training
        print("\nRunning short training test...")
        trainer.train()
        
        # Get statistics
        stats = trainer.get_stats()
        print(f"\nFinal stats:")
        print(f"  Episodes: {len(stats['episode_rewards'])}")
        print(f"  Avg reward: {np.mean(stats['episode_rewards']):.4f}")
        print(f"  Total steps: {stats['total_steps']}")
        
        print("\n" + "=" * 60)
        print("Test passed! ✓")
