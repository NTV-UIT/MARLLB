"""
QMIX Implementation for Multi-Agent Load Balancing

This package implements QMIX (Q-Mixing Networks) for coordinating multiple
load balancers in a distributed system.

Components:
- mixing_network: QMIX, VDN, and Weighted QMIX networks
- agent_network: Individual agent Q-networks with GRU
- episode_buffer: Episode replay buffer for multi-agent trajectories
- qmix_agent: Main QMIX agent coordinating all components
- multi_agent_env: Multi-agent environment wrapper

Usage:
    from qmix_agent import QMIXAgent
    from multi_agent_env import MultiAgentLoadBalanceEnv
    
    env = MultiAgentLoadBalanceEnv(num_agents=4)
    agent = QMIXAgent(num_agents=4, state_dim=74, obs_dim=20, action_dim=4)
    
    # Training loop
    observations = env.reset()
    episode_data = {...}
    
    while not done:
        actions, hiddens, _ = agent.select_actions(observations)
        next_obs, rewards, done, info = env.step(actions)
        # Store transitions...
    
    agent.store_episode(episode_data)
    stats = agent.update()
"""

__version__ = "1.0.0"

from .mixing_network import QMixingNetwork, VDNMixingNetwork, WeightedQMixingNetwork
from .agent_network import AgentQNetwork
from .episode_buffer import EpisodeBuffer
from .qmix_agent import QMIXAgent
from .multi_agent_env import MultiAgentLoadBalanceEnv

__all__ = [
    'QMixingNetwork',
    'VDNMixingNetwork', 
    'WeightedQMixingNetwork',
    'AgentQNetwork',
    'EpisodeBuffer',
    'QMIXAgent',
    'MultiAgentLoadBalanceEnv'
]
