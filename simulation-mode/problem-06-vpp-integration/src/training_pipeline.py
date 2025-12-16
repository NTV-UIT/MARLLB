"""
Training Pipeline for VPP Load Balancer

Offline training pipeline using historical traffic traces.
Train SAC-GRU or QMIX agents trước khi deploy lên VPP.

Traces:
- Poisson arrival process (synthetic)
- Wikipedia hourly traces (real-world)
- Custom traces
"""

import sys
import os
import time
import argparse
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-03-rl-environment' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-04-sac-gru' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-05-qmix' / 'src'))

from env import LoadBalanceEnv
from sac_agent import SAC_GRU_Agent as SACGRUAgent
from qmix_agent import QMIXAgent
from multi_agent_env import MultiAgentLoadBalanceEnv


class TrainingPipeline:
    """
    Offline training pipeline using historical traces.
    
    Pipeline:
    1. Load traffic traces từ data/trace/
    2. Initialize environment và agent
    3. Run training episodes với trace replay
    4. Save checkpoints periodically
    5. Evaluate và log metrics
    """
    
    def __init__(self, 
                 agent_type='qmix',
                 num_servers=16,
                 num_agents=4,
                 trace_dir='data/trace',
                 checkpoint_dir='checkpoints',
                 config=None):
        """
        Args:
            agent_type: 'sac-gru' hoặc 'qmix'
            num_servers: Tổng số servers
            num_agents: Số agents (cho QMIX)
            trace_dir: Directory chứa traffic traces
            checkpoint_dir: Directory lưu checkpoints
            config: Additional configuration
        """
        self.agent_type = agent_type
        self.num_servers = num_servers
        self.num_agents = num_agents
        self.trace_dir = trace_dir
        self.checkpoint_dir = checkpoint_dir
        self.config = config or {}
        
        print(f"[TrainingPipeline] Initializing...")
        print(f"  Agent: {agent_type}")
        print(f"  Servers: {num_servers}")
        print(f"  Agents: {num_agents}")
        print(f"  Trace Dir: {trace_dir}")
        print(f"  Checkpoint Dir: {checkpoint_dir}")
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load traces
        self.traces = self._load_traces()
        print(f"  ✓ Loaded {len(self.traces)} traces")
        
        # Initialize environment
        self.env = self._init_env()
        print(f"  ✓ Environment initialized")
        
        # Initialize agent
        self.agent = self._init_agent()
        print(f"  ✓ Agent initialized")
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        print(f"[TrainingPipeline] Ready to train!\n")
    
    def _load_traces(self):
        """
        Load traffic traces from data directory.
        
        Returns:
            traces: List of arrival time arrays
        """
        traces = []
        
        # Load Poisson traces
        poisson_files = glob.glob(f'{self.trace_dir}/poisson_*/*.csv')
        for file in poisson_files:
            try:
                df = pd.read_csv(file)
                if 'arrival_time' in df.columns:
                    traces.append(df['arrival_time'].values)
                elif len(df.columns) > 0:
                    traces.append(df.iloc[:, 0].values)
            except Exception as e:
                print(f"  ⚠ Failed to load {file}: {e}")
        
        # Load Wikipedia traces
        wiki_files = glob.glob(f'{self.trace_dir}/wiki/*.csv')
        for file in wiki_files:
            try:
                df = pd.read_csv(file)
                if 'arrival_time' in df.columns:
                    traces.append(df['arrival_time'].values)
                elif len(df.columns) > 0:
                    traces.append(df.iloc[:, 0].values)
            except Exception as e:
                print(f"  ⚠ Failed to load {file}: {e}")
        
        if len(traces) == 0:
            print(f"  ⚠ No traces found, generating synthetic Poisson trace")
            # Generate synthetic trace
            rate = 500  # requests per second
            duration = 3600  # 1 hour
            arrivals = self._generate_poisson_trace(rate, duration)
            traces.append(arrivals)
        
        return traces
    
    def _generate_poisson_trace(self, rate, duration):
        """
        Generate Poisson arrival process.
        
        Args:
            rate: Arrival rate (requests per second)
            duration: Trace duration (seconds)
        
        Returns:
            arrivals: Array of arrival times
        """
        num_arrivals = int(rate * duration)
        inter_arrivals = np.random.exponential(1.0 / rate, num_arrivals)
        arrivals = np.cumsum(inter_arrivals)
        return arrivals[arrivals < duration]
    
    def _init_env(self):
        """Initialize RL environment."""
        if self.agent_type == 'qmix':
            servers_per_agent = self.num_servers // self.num_agents
            return MultiAgentLoadBalanceEnv(
                num_agents=self.num_agents,
                servers_per_agent=servers_per_agent
            )
        else:
            return LoadBalanceEnv(num_servers=self.num_servers)
    
    def _init_agent(self):
        """Initialize RL agent."""
        if self.agent_type == 'qmix':
            servers_per_agent = self.num_servers // self.num_agents
            state_dim = 74
            obs_dim = 20
            action_dim = servers_per_agent
            
            return QMIXAgent(
                num_agents=self.num_agents,
                state_dim=state_dim,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=64,
                mixing_embed_dim=32,
                lr=self.config.get('learning_rate', 0.0005),
                gamma=self.config.get('gamma', 0.99)
            )
        else:
            state_dim = 74
            action_dim = self.num_servers
            
            lr = self.config.get('learning_rate', 0.0003)
            return SACGRUAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=256,
                lr_policy=lr,
                lr_q=lr,
                lr_alpha=lr,
                gamma=self.config.get('gamma', 0.99)
            )
    
    def train(self, num_episodes=10000, save_interval=100, eval_interval=100):
        """
        Train agent using trace replay.
        
        Args:
            num_episodes: Number of training episodes
            save_interval: Save checkpoint every N episodes
            eval_interval: Evaluate every N episodes
        """
        print(f"[Training] Starting training for {num_episodes} episodes...")
        print(f"  Save interval: {save_interval}")
        print(f"  Eval interval: {eval_interval}\n")
        
        start_time = time.time()
        best_reward = -np.inf
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # Sample random trace (select index first)
            trace_idx = np.random.randint(0, len(self.traces))
            trace = self.traces[trace_idx]
            
            # Run episode
            episode_reward, episode_length, loss = self._run_episode(trace, episode)
            
            # Track stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if loss is not None:
                self.losses.append(loss)
            
            # Log
            episode_time = time.time() - episode_start
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_loss = np.mean(self.losses[-10:]) if self.losses else 0.0
                
                print(f"[{episode+1:05d}/{num_episodes}] "
                      f"Reward: {episode_reward:7.2f} (avg: {avg_reward:7.2f}) "
                      f"Length: {episode_length:3d} (avg: {avg_length:5.1f}) "
                      f"Loss: {avg_loss:7.4f} "
                      f"Time: {episode_time:5.2f}s")
            
            # Evaluate
            if (episode + 1) % eval_interval == 0:
                eval_reward = self._evaluate(num_episodes=10)
                print(f"\n  [Eval] Average reward: {eval_reward:.2f}")
                
                # Save best model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    best_path = os.path.join(self.checkpoint_dir, f'{self.agent_type}_best.pth')
                    self.agent.save(best_path)
                    print(f"  ✓ New best model saved to {best_path}\n")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, 
                                              f'{self.agent_type}_ep{episode+1}.pth')
                self.agent.save(checkpoint_path)
                
                # Save training stats
                stats_path = os.path.join(self.checkpoint_dir, 'training_stats.json')
                with open(stats_path, 'w') as f:
                    json.dump({
                        'episode_rewards': self.episode_rewards,
                        'episode_lengths': self.episode_lengths,
                        'losses': self.losses,
                        'best_reward': float(best_reward),
                        'total_episodes': episode + 1,
                        'total_time': time.time() - start_time
                    }, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\n[Training] Complete!")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print(f"  Best reward: {best_reward:.2f}")
        print(f"  Final checkpoint: {checkpoint_path}")
    
    def _run_episode(self, trace, episode_num):
        """
        Run one training episode.
        
        Args:
            trace: Arrival time trace
            episode_num: Episode number (for epsilon decay)
        
        Returns:
            episode_reward: Total reward
            episode_length: Number of steps
            loss: Training loss (if updated)
        """
        # Reset environment
        observations = self.env.reset()
        
        # Initialize hidden states
        import torch
        if self.agent_type == 'qmix':
            hiddens = [torch.zeros((1, 1, 64), dtype=torch.float32) for _ in range(self.num_agents)]
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'states': [],
                'dones': []
            }
        else:
            # SAC-GRU uses gru_dim=128
            hiddens = torch.zeros((1, 1, 128), dtype=torch.float32)
        
        done = False
        episode_reward = 0
        episode_length = 0
        loss = None
        
        # Epsilon decay
        epsilon = max(0.01, 0.1 - episode_num / 5000.0)
        
        while not done and episode_length < 200:
            # Select actions
            if self.agent_type == 'qmix':
                actions, hiddens, _ = self.agent.select_actions(
                    observations, hiddens, epsilon=epsilon
                )
                state = self.env.get_state()
                
                # Take step
                next_observations, rewards, done, info = self.env.step(actions)
                
                # Store transition
                episode_data['observations'].append(observations)
                episode_data['actions'].append(actions)
                episode_data['rewards'].append(rewards)
                episode_data['states'].append(state)
                episode_data['dones'].append(done)
                
                observations = next_observations
                episode_reward += sum(rewards)
                
            else:
                action, hiddens = self.agent.select_action(
                    observations, hiddens, evaluate=False
                )
                
                # Take step
                next_observations, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(observations, action, reward, next_observations, done)
                
                observations = next_observations
                episode_reward += reward
            
            episode_length += 1
        
        # Update agent
        if self.agent_type == 'qmix':
            self.agent.store_episode(episode_data)
            if len(self.agent.buffer) >= self.agent.batch_size:
                loss = self.agent.update()
        else:
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                losses = self.agent.update(num_updates=1)
                loss = losses.get('critic_loss', 0.0)
        
        return episode_reward, episode_length, loss
    
    def _evaluate(self, num_episodes=10):
        """
        Evaluate agent performance.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            avg_reward: Average episode reward
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            # Random trace (select index first, then get trace)
            trace_idx = np.random.randint(0, len(self.traces))
            trace = self.traces[trace_idx]
            
            # Reset
            observations = self.env.reset()
            
            import torch
            if self.agent_type == 'qmix':
                hiddens = [torch.zeros((1, 1, 64), dtype=torch.float32) for _ in range(self.num_agents)]
            else:
                # SAC-GRU uses gru_dim=128
                hiddens = torch.zeros((1, 1, 128), dtype=torch.float32)
            
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 200:
                # Greedy actions
                if self.agent_type == 'qmix':
                    actions, hiddens, _ = self.agent.select_actions(
                        observations, hiddens, epsilon=0.0
                    )
                    observations, rewards, done, _ = self.env.step(actions)
                    episode_reward += sum(rewards)
                else:
                    action, hiddens = self.agent.select_action(
                        observations, hiddens, evaluate=True
                    )
                    observations, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                
                steps += 1
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)


def main():
    parser = argparse.ArgumentParser(description='Training Pipeline for VPP Load Balancer')
    
    parser.add_argument('--agent', type=str, default='qmix',
                        choices=['sac-gru', 'qmix'],
                        help='Agent type')
    parser.add_argument('--servers', type=int, default=16,
                        help='Number of servers')
    parser.add_argument('--agents', type=int, default=4,
                        help='Number of agents (for QMIX)')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes')
    parser.add_argument('--trace-dir', type=str, default='data/trace',
                        help='Directory with traffic traces')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Evaluate every N episodes')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create pipeline
    pipeline = TrainingPipeline(
        agent_type=args.agent,
        num_servers=args.servers,
        num_agents=args.agents,
        trace_dir=args.trace_dir,
        checkpoint_dir=args.checkpoint_dir,
        config=config
    )
    
    # Train
    pipeline.train(
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval
    )


if __name__ == '__main__':
    main()
