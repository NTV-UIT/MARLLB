"""
Multi-Agent Environment Wrapper

Wraps single-agent LoadBalanceEnv to support multiple agents.
Each agent controls a subset of servers.
"""

import sys
from pathlib import Path
import numpy as np

# Add path for Problem 03 environment
sys.path.append(str(Path(__file__).parent.parent.parent / 'problem-03-rl-environment' / 'src'))

try:
    from env import LoadBalanceEnv
except ImportError:
    LoadBalanceEnv = None
    print("Warning: LoadBalanceEnv not found")


class MultiAgentLoadBalanceEnv:
    """
    Multi-agent load balancing environment.
    
    Wraps LoadBalanceEnv to support multiple agents, where each agent
    controls a subset of servers.
    
    Example with 4 agents and 16 servers:
        Agent 0: controls servers 0-3
        Agent 1: controls servers 4-7
        Agent 2: controls servers 8-11
        Agent 3: controls servers 12-15
    
    Args:
        num_agents: Number of agents
        servers_per_agent: Number of servers per agent
        action_type: 'continuous' or 'discrete'
        reward_metric: Reward metric ('jain', 'variance', etc.)
        max_steps: Maximum steps per episode
        use_shm: Whether to use shared memory
    """
    
    def __init__(
        self,
        num_agents: int = 4,
        servers_per_agent: int = 4,
        action_type: str = 'continuous',
        reward_metric: str = 'jain',
        max_steps: int = 100,
        use_shm: bool = False,
        global_reward: bool = True
    ):
        if LoadBalanceEnv is None:
            raise ImportError("LoadBalanceEnv not found. Run Problem 03 first.")
        
        self.num_agents = num_agents
        self.servers_per_agent = servers_per_agent
        self.total_servers = num_agents * servers_per_agent
        self.global_reward = global_reward
        
        # Create underlying environment
        self.env = LoadBalanceEnv(
            num_servers=self.total_servers,
            action_type=action_type,
            reward_metric=reward_metric,
            max_steps=max_steps,
            use_shm=use_shm
        )
        
        # Agent server assignments
        self.agent_servers = {}
        for i in range(num_agents):
            start_idx = i * servers_per_agent
            end_idx = start_idx + servers_per_agent
            self.agent_servers[i] = list(range(start_idx, end_idx))
        
        # Observation and action spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # State dimensions
        self.obs_dim = self._get_obs_dim()
        self.state_dim = self._get_state_dim()
    
    def _get_obs_dim(self):
        """Get observation dimension for single agent."""
        # Each agent observes: own servers + global stats
        # Own servers: servers_per_agent * (queue + cpu + mem + latency)
        # Global: total_queue, total_throughput, fairness, time
        own_server_dim = self.servers_per_agent * 4
        global_dim = 4
        return own_server_dim + global_dim
    
    def _get_state_dim(self):
        """Get global state dimension."""
        # Global state includes all server states + global metrics
        return self.total_servers * 4 + 10  # 10 global metrics
    
    def reset(self):
        """
        Reset environment.
        
        Returns:
            observations: List of observations for each agent
        """
        global_obs = self.env.reset()
        
        # Split observation for each agent
        observations = []
        for i in range(self.num_agents):
            obs = self._get_agent_observation(global_obs, i)
            observations.append(obs)
        
        return observations
    
    def step(self, actions):
        """
        Execute actions for all agents.
        
        Args:
            actions: List of actions, one per agent (num_agents, action_dim_per_agent)
        
        Returns:
            observations: List of observations (num_agents, obs_dim)
            rewards: List of rewards (num_agents,)
            done: Episode termination flag
            info: Additional information
        """
        # Combine agent actions into global action
        global_action = self._combine_actions(actions)
        
        # Execute in environment
        global_obs, global_reward, done, info = self.env.step(global_action)
        
        # Split observation for each agent
        observations = []
        for i in range(self.num_agents):
            obs = self._get_agent_observation(global_obs, i)
            observations.append(obs)
        
        # Compute rewards
        if self.global_reward:
            # All agents get same global reward
            rewards = [global_reward] * self.num_agents
        else:
            # Each agent gets local reward based on own servers
            rewards = self._compute_local_rewards(info)
        
        return observations, rewards, done, info
    
    def _get_agent_observation(self, global_obs, agent_id):
        """
        Extract observation for specific agent.
        
        Args:
            global_obs: Global observation from environment
            agent_id: Agent ID
        
        Returns:
            agent_obs: Observation for this agent
        """
        # Flatten global_obs if needed
        if len(global_obs.shape) > 1:
            global_obs = global_obs.flatten()
        
        # global_obs shape: (total_servers * features + global_features,)
        # Split into server-specific and global parts
        
        server_features_per_server = 4  # queue, cpu, mem, latency
        server_obs_dim = self.total_servers * server_features_per_server
        
        # Extract own server observations
        agent_server_indices = self.agent_servers[agent_id]
        own_obs = []
        
        for server_idx in agent_server_indices:
            start = server_idx * server_features_per_server
            end = start + server_features_per_server
            own_obs.extend(global_obs[start:end].tolist() if hasattr(global_obs[start:end], 'tolist') else global_obs[start:end])
        
        # Extract global features (last few dimensions)
        global_features = global_obs[server_obs_dim:]
        
        # Combine (both should be 1D now)
        agent_obs = np.concatenate([np.array(own_obs).flatten(), global_features.flatten()])
        
        return agent_obs
    
    def _combine_actions(self, actions):
        """
        Combine agent actions into global action.
        
        Args:
            actions: List of actions (num_agents, servers_per_agent)
        
        Returns:
            global_action: Combined action (total_servers,)
        """
        global_action = np.zeros(self.total_servers)
        
        for agent_id, action in enumerate(actions):
            agent_servers = self.agent_servers[agent_id]
            for i, server_idx in enumerate(agent_servers):
                if i < len(action):
                    global_action[server_idx] = action[i]
        
        return global_action
    
    def _compute_local_rewards(self, info):
        """
        Compute local rewards for each agent based on own servers.
        
        Args:
            info: Info dict from environment step
        
        Returns:
            rewards: List of rewards (num_agents,)
        """
        rewards = []
        
        for agent_id in range(self.num_agents):
            agent_servers = self.agent_servers[agent_id]
            
            # Compute fairness among own servers
            server_loads = [info['server_loads'][idx] for idx in agent_servers]
            
            if sum(server_loads) == 0:
                reward = 0.0
            else:
                # Local Jain's fairness index
                sum_loads = sum(server_loads)
                sum_squared = sum(x**2 for x in server_loads)
                fairness = (sum_loads**2) / (self.servers_per_agent * sum_squared + 1e-8)
                reward = fairness
            
            rewards.append(reward)
        
        return rewards
    
    def get_state(self):
        """
        Get global state for centralized training.
        
        Returns:
            state: Global state vector
        """
        # Get full environment state
        if hasattr(self.env, 'last_observation') and self.env.last_observation is not None:
            obs = self.env.last_observation
            if len(obs.shape) > 1:
                obs = obs.flatten()
        else:
            obs = np.zeros(self.total_servers * 4)
        
        # Add additional global metrics
        info = {'total_requests': 0, 'total_throughput': 0, 'avg_latency': 0, 
                'fairness': 0, 'utilization': 0, 'server_loads': [0] * self.total_servers}
        if hasattr(self.env, 'total_requests'):
            info['total_requests'] = self.env.total_requests
            info['total_throughput'] = sum(self.env.server_throughputs)
            info['avg_latency'] = np.mean(self.env.server_latencies)
            info['fairness'] = self.env.last_fairness if hasattr(self.env, 'last_fairness') else 0
            info['utilization'] = np.mean([s > 0 for s in self.env.server_loads])
            info['server_loads'] = self.env.server_loads
        
        global_metrics = [
            info['total_requests'],
            info['total_throughput'],
            info['avg_latency'],
            info['fairness'],
            info['utilization'],
            np.std(info['server_loads']),
            np.max(info['server_loads']),
            np.min(info['server_loads']),
            self.env.current_step / self.env.max_steps,  # Time progress
            self.num_agents
        ]
        
        state = np.concatenate([obs, global_metrics])
        
        return state
    
    def render(self, mode='human'):
        """Render environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close environment."""
        self.env.close()


if __name__ == '__main__':
    # Test multi-agent environment
    print("=" * 60)
    print("Testing Multi-Agent Load Balance Environment")
    print("=" * 60)
    
    if LoadBalanceEnv is None:
        print("LoadBalanceEnv not available, skipping test")
    else:
        # Create environment
        env = MultiAgentLoadBalanceEnv(
            num_agents=4,
            servers_per_agent=4,
            action_type='continuous',
            reward_metric='jain',
            max_steps=20,
            global_reward=True
        )
        
        print(f"\n1. Environment created")
        print(f"   Num agents: {env.num_agents}")
        print(f"   Servers per agent: {env.servers_per_agent}")
        print(f"   Total servers: {env.total_servers}")
        print(f"   Observation dim: {env.obs_dim}")
        print(f"   State dim: {env.state_dim}")
        
        # Test reset
        print(f"\n2. Testing reset...")
        observations = env.reset()
        print(f"   Observations shape: {len(observations)} agents")
        for i, obs in enumerate(observations):
            print(f"   Agent {i}: {obs.shape}")
        
        # Test step
        print(f"\n3. Testing step...")
        actions = []
        for i in range(env.num_agents):
            action = np.random.rand(env.servers_per_agent)
            actions.append(action)
        
        observations, rewards, done, info = env.step(actions)
        
        print(f"   Observations: {len(observations)} agents")
        print(f"   Rewards: {rewards}")
        print(f"   Done: {done}")
        print(f"   Info keys: {list(info.keys())}")
        
        # Test global state
        print(f"\n4. Testing global state...")
        state = env.get_state()
        print(f"   State shape: {state.shape}")
        print(f"   State range: [{state.min():.4f}, {state.max():.4f}]")
        
        # Test episode
        print(f"\n5. Testing full episode...")
        observations = env.reset()
        episode_rewards = [0.0] * env.num_agents
        
        for step in range(10):
            actions = [np.random.rand(env.servers_per_agent) for _ in range(env.num_agents)]
            observations, rewards, done, info = env.step(actions)
            
            for i, r in enumerate(rewards):
                episode_rewards[i] += r
            
            if done:
                break
        
        print(f"   Episode length: {step + 1}")
        print(f"   Episode rewards: {episode_rewards}")
        print(f"   Average reward: {np.mean(episode_rewards):.4f}")
        
        # Test agent server assignments
        print(f"\n6. Agent server assignments:")
        for agent_id, servers in env.agent_servers.items():
            print(f"   Agent {agent_id}: servers {servers}")
        
        env.close()
        
        print("\n" + "=" * 60)
        print("All Tests Passed! ✓")
        print("=" * 60)
