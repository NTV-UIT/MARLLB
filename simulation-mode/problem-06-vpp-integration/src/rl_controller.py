"""
RL Controller for VPP Load Balancer

Controller chạy RL agent (SAC-GRU hoặc QMIX) và giao tiếp với VPP data plane
qua shared memory. Controller đọc statistics từ VPP, chạy inference để chọn
server weights, và ghi kết quả trả lại VPP để sử dụng.

Integration:
- Problem 02: Shared Memory IPC
- Problem 03: LoadBalanceEnv 
- Problem 04: SAC-GRU Agent
- Problem 05: QMIX Agent
"""

import sys
import os
import time
import argparse
import json
import numpy as np
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-03-rl-environment' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-04-sac-gru' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-05-qmix' / 'src'))

from shm_interface import SharedMemoryInterface, SHMLayout
from env import LoadBalanceEnv
from sac_agent import SAC_GRU_Agent as SACGRUAgent
from qmix_agent import QMIXAgent
from multi_agent_env import MultiAgentLoadBalanceEnv


class RLController:
    """
    RL Controller cho VPP Load Balancer.
    
    Controller chạy RL agent và đồng bộ với VPP data plane:
    1. Đọc server statistics từ VPP via shared memory
    2. Convert stats thành observation cho RL agent
    3. Chạy inference để chọn action (server weights)
    4. Ghi action vào shared memory để VPP sử dụng
    5. Optionally train agent online
    """
    
    def __init__(self, 
                 agent_type='qmix',
                 num_servers=16,
                 num_agents=4,
                 model_path=None,
                 shm_path='/dev/shm/lb_rl_shm',
                 update_interval=0.2,
                 online_training=False,
                 config=None):
        """
        Args:
            agent_type: 'sac-gru' hoặc 'qmix'
            num_servers: Tổng số servers
            num_agents: Số agents (cho QMIX)
            model_path: Path to pretrained model
            shm_path: Shared memory file path
            update_interval: Agent update interval (seconds)
            online_training: Enable online learning
            config: Additional config dict
        """
        self.agent_type = agent_type
        self.num_servers = num_servers
        self.num_agents = num_agents
        self.update_interval = update_interval
        self.online_training = online_training
        self.config = config or {}
        
        print(f"[RLController] Initializing {agent_type} controller...")
        print(f"  Servers: {num_servers}")
        print(f"  Agents: {num_agents}")
        print(f"  SHM Path: {shm_path}")
        print(f"  Online Training: {online_training}")
        
        # Initialize shared memory interface
        self.shm = self._init_shm(shm_path)
        print(f"  ✓ Shared memory initialized")
        
        # Initialize RL environment (for training)
        if online_training:
            self.env = self._init_env()
            print(f"  ✓ Environment initialized")
        else:
            self.env = None
        
        # Initialize RL agent
        self.agent = self._init_agent()
        print(f"  ✓ Agent initialized")
        
        # Load pretrained model if available
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"  ✓ Loaded model from {model_path}")
        else:
            print(f"  ⚠ No pretrained model found at {model_path}")
        
        # State tracking
        import torch
        self.hiddens = None
        if agent_type == 'qmix':
            self.hiddens = [torch.zeros((1, 1, 64), dtype=torch.float32) for _ in range(num_agents)]
        else:
            # SAC-GRU uses gru_dim=128 by default
            self.hiddens = torch.zeros((1, 1, 128), dtype=torch.float32)
        
        self.last_update_time = time.time()
        self.total_updates = 0
        self.episode_rewards = []
        
        print(f"[RLController] Initialization complete!\n")
    
    def _init_shm(self, shm_path):
        """Initialize shared memory interface."""
        # Create layout for 64 servers (max)
        layout = SHMLayout(num_servers=64)
        
        # Create shared memory file if not exists
        if not os.path.exists(shm_path):
            print(f"  Creating shared memory file: {shm_path}")
            with open(shm_path, 'wb') as f:
                f.write(b'\x00' * layout.total_size)
        
        # Open interface
        return SharedMemoryInterface(shm_path, layout)
    
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
            state_dim = 74  # From MultiAgentLoadBalanceEnv
            obs_dim = 20    # Per-agent observation
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
    
    def run(self):
        """
        Main control loop.
        
        Loop:
        1. Read stats from VPP
        2. Convert to observation
        3. Get action from agent
        4. Write action to VPP
        5. Optionally train agent
        """
        print("[RLController] Starting control loop...")
        
        iteration = 0
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Read stats from VPP
                msg_out = self.shm.read_msg_out()
                
                # 2. Convert to observation
                observation = self._stats_to_observation(msg_out)
                
                # 3. Get action from RL agent
                server_weights = self._get_action(observation)
                
                # 4. Write action to VPP
                self._write_action(server_weights, msg_out['id'])
                
                # 5. Optionally train agent
                if self.online_training and (time.time() - self.last_update_time) > self.update_interval:
                    self._update_agent(msg_out)
                    self.last_update_time = time.time()
                
                # Stats
                iteration += 1
                elapsed = time.time() - start_time
                
                if iteration % 100 == 0:
                    avg_weight = np.mean(server_weights)
                    std_weight = np.std(server_weights)
                    print(f"[{iteration:06d}] "
                          f"Weights: μ={avg_weight:.3f} σ={std_weight:.3f} "
                          f"Updates: {self.total_updates} "
                          f"Time: {elapsed*1000:.2f}ms")
                
                # Sleep to maintain update frequency (50ms = 20 Hz)
                sleep_time = max(0, 0.05 - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[RLController] Stopping...")
            self.cleanup()
    
    def _stats_to_observation(self, msg_out):
        """
        Convert VPP stats to RL observation.
        
        Args:
            msg_out: Dict from shared memory
        
        Returns:
            observation: numpy array or list of arrays (for QMIX)
        """
        server_stats = msg_out['server_stats'][:self.num_servers]
        
        if self.agent_type == 'qmix':
            # Multi-agent: split servers among agents
            observations = []
            servers_per_agent = self.num_servers // self.num_agents
            
            for i in range(self.num_agents):
                start_idx = i * servers_per_agent
                end_idx = start_idx + servers_per_agent
                agent_servers = server_stats[start_idx:end_idx]
                
                # Build observation for this agent (20 dims total)
                obs = []
                for server in agent_servers:
                    obs.extend([
                        server['n_flow_on'] / 100.0,     # Normalize
                        server['cpu_util'],
                        server['queue_depth'] / 100.0,
                        server['response_time'] / 1000.0,  # ms to s
                    ])
                
                # Global metrics (4 features to reach 20 total)
                total_flows = sum(s['n_flow_on'] for s in server_stats)
                avg_cpu = np.mean([s['cpu_util'] for s in server_stats])
                std_cpu = np.std([s['cpu_util'] for s in server_stats])
                max_queue = max(s['queue_depth'] for s in server_stats)
                
                obs.extend([total_flows / 1000.0, avg_cpu, std_cpu, max_queue / 100.0])
                
                observations.append(np.array(obs, dtype=np.float32))
            
            return observations
        
        else:
            # Single agent: all servers in one observation
            obs = []
            for server in server_stats:
                obs.extend([
                    server['n_flow_on'] / 100.0,
                    server['cpu_util'],
                    server['queue_depth'] / 100.0,
                    server['response_time'] / 1000.0,
                ])
            
            # Global metrics
            total_flows = sum(s['n_flow_on'] for s in server_stats)
            avg_cpu = np.mean([s['cpu_util'] for s in server_stats])
            std_cpu = np.std([s['cpu_util'] for s in server_stats])
            
            obs.extend([total_flows / 1000.0, avg_cpu, std_cpu])
            
            return np.array(obs, dtype=np.float32)
    
    def _get_action(self, observation):
        """
        Get action from RL agent.
        
        Args:
            observation: Array or list of arrays
        
        Returns:
            server_weights: Normalized weights for each server (sum to 1)
        """
        if self.agent_type == 'qmix':
            # Multi-agent
            actions, self.hiddens, _ = self.agent.select_actions(
                observation, 
                self.hiddens, 
                epsilon=0.0  # Greedy (no exploration in production)
            )
            
            # Convert discrete actions to weights
            # Each agent outputs action in [0, servers_per_agent-1]
            weights = np.zeros(self.num_servers)
            servers_per_agent = self.num_servers // self.num_agents
            
            for i, action in enumerate(actions):
                server_idx = i * servers_per_agent + action
                weights[server_idx] += 1.0
            
            # Normalize
            weights = weights / weights.sum()
            
        else:
            # Single agent
            action, self.hiddens = self.agent.select_action(
                observation, 
                self.hiddens, 
                evaluate=True  # No exploration
            )
            
            # Convert action to weights via softmax
            weights = np.exp(action) / np.exp(action).sum()
        
        return weights
    
    def _write_action(self, server_weights, msg_id):
        """
        Write action to shared memory for VPP.
        
        Args:
            server_weights: Array of weights (sum to 1)
            msg_id: Message sequence ID
        """
        # Build alias table for O(1) sampling in VPP
        alias_table = self._build_alias_table(server_weights)
        
        msg_in = {
            'id': msg_id + 1,
            'timestamp': time.time(),
            'server_weights': server_weights,
            'alias_table': alias_table
        }
        
        self.shm.write_msg_in(msg_in)
    
    def _build_alias_table(self, weights):
        """
        Build alias table for O(1) server sampling.
        
        Alias method allows constant-time sampling:
        1. Generate random index i
        2. Generate random u ~ Uniform(0, 1)
        3. If u < prob[i]: return i
        4. Else: return alias[i]
        
        Args:
            weights: Array of probabilities (sum to 1)
        
        Returns:
            alias_table: List of (probability, alias) tuples
        """
        n = len(weights)
        weights = np.array(weights)
        
        # Scale to n
        prob = weights * n
        alias = np.arange(n, dtype=np.int32)
        
        # Separate small and large
        small = []
        large = []
        
        for i, p in enumerate(prob):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        # Build table
        while small and large:
            l = small.pop()
            g = large.pop()
            
            alias[l] = g
            prob[g] = prob[g] + prob[l] - 1.0
            
            if prob[g] < 1.0:
                small.append(g)
            else:
                large.append(g)
        
        return [(prob[i], alias[i]) for i in range(n)]
    
    def _update_agent(self, msg_out):
        """
        Update agent (online learning).
        
        Args:
            msg_out: Current stats from VPP
        """
        # Compute reward from current state
        reward = self._compute_reward(msg_out)
        
        # Store transition (simplified - in practice would need full trajectory)
        # This is placeholder for online learning logic
        
        # Update agent
        if self.agent_type == 'qmix':
            if len(self.agent.buffer) >= self.agent.batch_size:
                loss = self.agent.update()
                self.total_updates += 1
                
                if self.total_updates % 10 == 0:
                    print(f"  [Update {self.total_updates}] Loss: {loss:.4f}, Reward: {reward:.2f}")
        else:
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                losses = self.agent.update(num_updates=1)
                self.total_updates += 1
                
                if self.total_updates % 10 == 0:
                    print(f"  [Update {self.total_updates}] "
                          f"Critic Loss: {losses['critic_loss']:.4f}, "
                          f"Reward: {reward:.2f}")
    
    def _compute_reward(self, msg_out):
        """
        Compute reward from current state.
        
        Reward = combination of:
        - Fairness (low std of server loads)
        - Low latency (fast response times)
        - High throughput (many completed requests)
        
        Args:
            msg_out: Stats dict
        
        Returns:
            reward: Scalar reward
        """
        server_stats = msg_out['server_stats'][:self.num_servers]
        
        # CPU utilization fairness
        cpu_utils = [s['cpu_util'] for s in server_stats]
        cpu_mean = np.mean(cpu_utils)
        cpu_std = np.std(cpu_utils)
        
        # Jain's fairness index
        cpu_sum_sq = sum(u**2 for u in cpu_utils)
        fairness = (cpu_mean * len(cpu_utils))**2 / (len(cpu_utils) * cpu_sum_sq + 1e-8)
        
        # Average response time
        response_times = [s['response_time'] for s in server_stats]
        avg_response = np.mean(response_times)
        
        # Total throughput
        total_flows = sum(s['n_flow_on'] for s in server_stats)
        
        # Combined reward
        reward = (
            10 * fairness +           # Fairness weight
            -0.01 * avg_response +    # Latency penalty
            0.001 * total_flows       # Throughput bonus
        )
        
        return reward
    
    def cleanup(self):
        """Cleanup resources."""
        print("[RLController] Cleaning up...")
        self.shm.close()
        print("[RLController] Goodbye!")


def main():
    parser = argparse.ArgumentParser(description='RL Controller for VPP Load Balancer')
    
    parser.add_argument('--agent', type=str, default='qmix',
                        choices=['sac-gru', 'qmix'],
                        help='Agent type')
    parser.add_argument('--servers', type=int, default=16,
                        help='Number of servers')
    parser.add_argument('--agents', type=int, default=4,
                        help='Number of agents (for QMIX)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--shm', type=str, default='/dev/shm/lb_rl_shm',
                        help='Shared memory path')
    parser.add_argument('--interval', type=float, default=0.2,
                        help='Update interval (seconds)')
    parser.add_argument('--online-training', action='store_true',
                        help='Enable online training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create controller
    controller = RLController(
        agent_type=args.agent,
        num_servers=args.servers,
        num_agents=args.agents,
        model_path=args.model,
        shm_path=args.shm,
        update_interval=args.interval,
        online_training=args.online_training,
        config=config
    )
    
    # Run
    controller.run()


if __name__ == '__main__':
    main()
