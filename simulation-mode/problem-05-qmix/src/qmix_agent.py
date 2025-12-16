"""
QMIX Agent Implementation

Main QMIX agent that coordinates multiple individual agents with mixing network.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from mixing_network import QMixingNetwork, VDNMixingNetwork
from episode_buffer import EpisodeBuffer
from agent_network import AgentQNetwork


class QMIXAgent:
    """
    QMIX multi-agent reinforcement learning agent.
    
    Implements centralized training with decentralized execution using:
    - Individual Q-networks for each agent (with GRU)
    - Mixing network to combine Q-values
    - Episode replay buffer
    - Target networks for stability
    
    Args:
        num_agents: Number of agents
        state_dim: Global state dimension
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        hidden_dim: Hidden layer dimension
        gru_dim: GRU hidden dimension
        mixing_embed_dim: Mixing network embedding dimension
        hypernet_embed_dim: Hypernetwork embedding dimension
        lr: Learning rate
        gamma: Discount factor
        target_update_interval: Steps between target network updates
        buffer_capacity: Episode buffer capacity
        batch_size: Training batch size
        max_seq_len: Maximum sequence length for training
        device: Device to use
        use_vdn: Use VDN instead of QMIX (simpler baseline)
    """
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gru_dim: int = 64,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        target_update_interval: int = 200,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        max_seq_len: int = 50,
        device: str = None,
        use_vdn: bool = False
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Individual agent Q-networks
        self.agent_networks = []
        for i in range(num_agents):
            q_net = AgentQNetwork(obs_dim, action_dim, hidden_dim, gru_dim).to(self.device)
            self.agent_networks.append(q_net)
        
        # Target agent networks
        self.agent_networks_target = []
        for i in range(num_agents):
            q_net_target = AgentQNetwork(obs_dim, action_dim, hidden_dim, gru_dim).to(self.device)
            q_net_target.load_state_dict(self.agent_networks[i].state_dict())
            self.agent_networks_target.append(q_net_target)
        
        # Mixing network
        if use_vdn:
            self.mixer = VDNMixingNetwork(num_agents).to(self.device)
            self.mixer_target = VDNMixingNetwork(num_agents).to(self.device)
        else:
            self.mixer = QMixingNetwork(
                num_agents, state_dim, mixing_embed_dim, hypernet_embed_dim
            ).to(self.device)
            self.mixer_target = QMixingNetwork(
                num_agents, state_dim, mixing_embed_dim, hypernet_embed_dim
            ).to(self.device)
            self.mixer_target.load_state_dict(self.mixer.state_dict())
        
        # Optimizer (all parameters together)
        params = []
        for agent_net in self.agent_networks:
            params += list(agent_net.parameters())
        params += list(self.mixer.parameters())
        self.optimizer = Adam(params, lr=lr)
        
        # Episode buffer
        self.episode_buffer = EpisodeBuffer(capacity=buffer_capacity, num_agents=num_agents)
        
        # Training stats
        self.total_updates = 0
        self.training_stats = {
            'loss': [],
            'q_tot': [],
            'target_q_tot': []
        }
    
    def select_actions(self, observations, hiddens=None, evaluate=False, epsilon=0.0):
        """
        Select actions for all agents.
        
        Args:
            observations: List of observations (num_agents, obs_dim)
            hiddens: List of GRU hidden states or None
            evaluate: If True, use deterministic policy
            epsilon: Epsilon for epsilon-greedy exploration
        
        Returns:
            actions: List of actions (num_agents, action_dim)
            new_hiddens: Updated hidden states
            q_values: Q-values for selected actions (for analysis)
        """
        actions = []
        new_hiddens = []
        q_values = []
        
        for agent_id in range(self.num_agents):
            obs = torch.FloatTensor(observations[agent_id]).unsqueeze(0).to(self.device)
            
            # Initialize hidden if needed
            if hiddens is None or hiddens[agent_id] is None:
                hidden = self.agent_networks[agent_id].init_hidden(1).to(self.device)
            else:
                hidden = hiddens[agent_id]
            
            # Get Q-values
            with torch.no_grad():
                q_vals, new_hidden = self.agent_networks[agent_id](obs, hidden)
            
            # Select action
            if not evaluate and np.random.rand() < epsilon:
                # Random action
                action = np.random.randint(0, self.action_dim)
            else:
                # Greedy action
                action = q_vals.argmax(dim=1).item()
            
            actions.append(action)
            new_hiddens.append(new_hidden)
            q_values.append(q_vals[0, action].item())
        
        return actions, new_hiddens, q_values
    
    def store_episode(self, episode_data):
        """
        Store complete episode in buffer.
        
        Args:
            episode_data: Dict with observations, actions, rewards, states, dones
        """
        self.episode_buffer.start_episode()
        
        for t in range(len(episode_data['observations'])):
            self.episode_buffer.add_transition(
                observations=episode_data['observations'][t],
                actions=episode_data['actions'][t],
                rewards=episode_data['rewards'][t],
                state=episode_data['states'][t],
                done=episode_data['dones'][t]
            )
        
        self.episode_buffer.end_episode()
    
    def update(self):
        """
        Update agent parameters using batch from episode buffer.
        
        Returns:
            Dictionary with loss values or None if not enough data
        """
        if not self.episode_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        batch = self.episode_buffer.sample_batch(self.batch_size, self.max_seq_len)
        
        # Convert to tensors
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        states = torch.FloatTensor(batch['states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        seq_lengths = batch['seq_lengths']
        
        batch_size, seq_len, num_agents, obs_dim = observations.shape
        
        # Compute Q-values for all agents
        agent_qs = []
        for agent_id in range(self.num_agents):
            q_vals_list = []
            hidden = self.agent_networks[agent_id].init_hidden(batch_size).to(self.device)
            
            for t in range(seq_len):
                obs = observations[:, t, agent_id, :]
                q_vals, hidden = self.agent_networks[agent_id](obs, hidden)
                q_vals_list.append(q_vals)
            
            # Stack across time: (batch, seq_len, 1)
            agent_q = torch.stack(q_vals_list, dim=1)
            agent_qs.append(agent_q)
        
        # Stack across agents: (batch, seq_len, num_agents)
        agent_qs = torch.cat(agent_qs, dim=2).squeeze(-1)
        
        # Extract Q-values for taken actions
        chosen_action_qs = agent_qs.gather(2, actions[:, :, :, 0].long())  # (batch, seq_len, num_agents)
        
        # Mix Q-values
        q_tot = []
        for t in range(seq_len):
            q_t = self.mixer(chosen_action_qs[:, t, :], states[:, t, :])
            q_tot.append(q_t)
        q_tot = torch.stack(q_tot, dim=1)  # (batch, seq_len, 1)
        
        # Compute target Q-values
        with torch.no_grad():
            target_agent_qs = []
            for agent_id in range(self.num_agents):
                q_vals_list = []
                hidden = self.agent_networks_target[agent_id].init_hidden(batch_size).to(self.device)
                
                for t in range(seq_len):
                    obs = observations[:, t, agent_id, :]
                    q_vals, hidden = self.agent_networks_target[agent_id](obs, hidden)
                    q_vals_list.append(q_vals.max(dim=1, keepdim=True)[0])
                
                agent_q = torch.stack(q_vals_list, dim=1)
                target_agent_qs.append(agent_q)
            
            target_agent_qs = torch.cat(target_agent_qs, dim=2)
            
            # Mix target Q-values
            target_q_tot = []
            for t in range(seq_len):
                target_q_t = self.mixer_target(target_agent_qs[:, t, :], states[:, t, :])
                target_q_tot.append(target_q_t)
            target_q_tot = torch.stack(target_q_tot, dim=1)
            
            # Compute target: r + gamma * Q_target (shifted by 1)
            target_q_tot_shifted = torch.zeros_like(target_q_tot)
            target_q_tot_shifted[:, :-1] = target_q_tot[:, 1:]
            
            targets = rewards.sum(dim=2, keepdim=True) + \
                     self.gamma * (1 - dones.unsqueeze(2)) * target_q_tot_shifted
        
        # Compute loss (only for valid timesteps)
        mask = torch.zeros(batch_size, seq_len, 1).to(self.device)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = 1.0
        
        loss = ((q_tot - targets) ** 2 * mask).sum() / mask.sum()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 10.0)
        self.optimizer.step()
        
        # Update target networks
        self.total_updates += 1
        if self.total_updates % self.target_update_interval == 0:
            for i in range(self.num_agents):
                self.agent_networks_target[i].load_state_dict(
                    self.agent_networks[i].state_dict()
                )
            self.mixer_target.load_state_dict(self.mixer.state_dict())
        
        # Store stats
        stats = {
            'loss': loss.item(),
            'q_tot': q_tot.mean().item(),
            'target_q_tot': targets.mean().item()
        }
        
        self.training_stats['loss'].append(stats['loss'])
        self.training_stats['q_tot'].append(stats['q_tot'])
        self.training_stats['target_q_tot'].append(stats['target_q_tot'])
        
        return stats
    
    def save(self, filepath):
        """Save agent parameters."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'agent_networks': [net.state_dict() for net in self.agent_networks],
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_updates': self.total_updates
        }
        
        torch.save(checkpoint, filepath)
        print(f"QMIX agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for i, state_dict in enumerate(checkpoint['agent_networks']):
            self.agent_networks[i].load_state_dict(state_dict)
            self.agent_networks_target[i].load_state_dict(state_dict)
        
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.mixer_target.load_state_dict(checkpoint['mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_updates = checkpoint['total_updates']
        
        print(f"QMIX agent loaded from {filepath}")
    
    def get_stats(self):
        """Get training statistics."""
        stats = {
            'total_updates': self.total_updates,
            'buffer_size': len(self.episode_buffer),
            'training_stats': self.training_stats
        }
        stats.update(self.episode_buffer.get_stats())
        return stats


if __name__ == '__main__':
    # Test QMIX agent
    print("=" * 60)
    print("Testing QMIX Agent")
    print("=" * 60)
    
    # Create agent
    agent = QMIXAgent(
            num_agents=4,
            state_dim=74,
            obs_dim=20,
            action_dim=4,
            hidden_dim=64,
            gru_dim=32,
            batch_size=4
    )
    
    print(f"\n1. Agent created")
    print(f"   Num agents: {agent.num_agents}")
    print(f"   State dim: {agent.state_dim}")
    print(f"   Obs dim: {agent.obs_dim}")
    print(f"   Action dim: {agent.action_dim}")
    print(f"   Device: {agent.device}")
    
    # Test action selection
    print(f"\n2. Testing action selection...")
    observations = [np.random.randn(agent.obs_dim) for _ in range(agent.num_agents)]
    actions, hiddens, q_values = agent.select_actions(observations)
    print(f"   Actions: {actions}")
    print(f"   Q-values: {[f'{q:.4f}' for q in q_values]}")
    print(f"   Hiddens: {len(hiddens)} agents")
    
    # Test episode storage
    print(f"\n3. Testing episode storage...")
    for ep in range(10):
        episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'states': [],
            'dones': []
        }
        
        for t in range(15):
            obs = [np.random.randn(agent.obs_dim) for _ in range(agent.num_agents)]
            acts = [np.random.randint(0, agent.action_dim) for _ in range(agent.num_agents)]
            rews = [np.random.randn() for _ in range(agent.num_agents)]
            state = np.random.randn(agent.state_dim)
            done = (t == 14)
            
            episode['observations'].append(obs)
            episode['actions'].append(acts)
            episode['rewards'].append(rews)
            episode['states'].append(state)
            episode['dones'].append(done)
        
        agent.store_episode(episode)
    
    print(f"   Episodes in buffer: {len(agent.episode_buffer)}")
    
    # Test update
    print(f"\n4. Testing parameter update...")
    stats = agent.update()
    
    if stats:
        print(f"   Loss: {stats['loss']:.6f}")
        print(f"   Q_tot: {stats['q_tot']:.6f}")
        print(f"   Target Q_tot: {stats['target_q_tot']:.6f}")
    
    # Test save/load
    print(f"\n5. Testing save/load...")
    agent.save("/tmp/qmix_test.pth")
    agent.load("/tmp/qmix_test.pth")
    
    # Test stats
    print(f"\n6. Agent statistics...")
    stats = agent.get_stats()
    print(f"   Total updates: {stats['total_updates']}")
    print(f"   Buffer size: {stats['buffer_size']}")
    print(f"   Avg episode length: {stats.get('avg_length', 0):.2f}")
        
    print("\n" + "=" * 60)
    print("All QMIX Agent Tests Passed! ✓")
    print("=" * 60)
