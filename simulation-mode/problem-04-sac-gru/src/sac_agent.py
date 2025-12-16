"""
SAC-GRU Agent Implementation

Main agent class implementing Soft Actor-Critic with GRU networks.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from pathlib import Path

from networks import PolicyNetwork, QNetwork, soft_update, hard_update
from replay_buffer import ReplayBuffer


class SAC_GRU_Agent:
    """
    Soft Actor-Critic agent with GRU networks.
    
    Implements:
    - Policy network (actor) with GRU
    - Twin Q-networks (critics) with GRU
    - Automatic entropy temperature tuning
    - Soft target network updates
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gru_dim: int = 128,
        lr_policy: float = 3e-4,
        lr_q: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: float = None,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        device: str = None
    ):
        """
        Initialize SAC-GRU agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            gru_dim: GRU hidden state dimension
            lr_policy: Policy learning rate
            lr_q: Q-network learning rate
            lr_alpha: Temperature learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Initial temperature
            auto_entropy_tuning: Whether to auto-tune temperature
            target_entropy: Target entropy (default: -dim(A))
            buffer_size: Replay buffer size
            batch_size: Training batch size
            device: Device to use (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, gru_dim).to(self.device)
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim, gru_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim, gru_dim).to(self.device)
        
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim, gru_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim, gru_dim).to(self.device)
        
        # Initialize target networks
        hard_update(self.q1, self.q1_target)
        hard_update(self.q2, self.q2_target)
        
        # Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_policy)
        self.q1_optimizer = Adam(self.q1.parameters(), lr=lr_q)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=lr_q)
        
        # Entropy temperature
        if auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim  # Heuristic
            else:
                self.target_entropy = target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
            self.target_entropy = None
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Training statistics
        self.total_steps = 0
        self.training_stats = {
            'q1_loss': [],
            'q2_loss': [],
            'policy_loss': [],
            'alpha_loss': [],
            'alpha': []
        }
    
    def select_action(self, state, hidden, evaluate=False):
        """
        Select action from policy.
        
        Args:
            state: State array or tensor
            hidden: GRU hidden state
            evaluate: If True, use deterministic policy
        
        Returns:
            action: Selected action
            hidden_new: Updated hidden state
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if hidden is None:
            hidden = self.policy.init_hidden(batch_size=1).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                _, _, action, hidden_new = self.policy.sample(state, hidden)
            else:
                action, _, _, hidden_new = self.policy.sample(state, hidden)
        
        return action.cpu().numpy()[0], hidden_new
    
    def update_parameters(self, updates=1):
        """
        Update agent parameters using batch from replay buffer.
        
        Args:
            updates: Number of update steps
        
        Returns:
            Dictionary with loss values
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        losses = {'q1': 0, 'q2': 0, 'policy': 0, 'alpha': 0}
        
        for _ in range(updates):
            # Sample batch
            states, actions, rewards, next_states, dones, hiddens = \
                self.replay_buffer.sample(self.batch_size, self.device)
            
            # Initialize hidden states if not stored
            if hiddens is None:
                policy_hidden = self.policy.init_hidden(self.batch_size).to(self.device)
                q_hidden = self.q1.init_hidden(self.batch_size).to(self.device)
            else:
                policy_hidden = hiddens
                q_hidden = hiddens
            
            # --- Update Q-networks ---
            with torch.no_grad():
                # Sample next actions
                next_actions, next_log_probs, _, _ = self.policy.sample(next_states, policy_hidden)
                
                # Compute target Q-values
                q1_next, _ = self.q1_target.forward(next_states, next_actions, q_hidden)
                q2_next, _ = self.q2_target.forward(next_states, next_actions, q_hidden)
                q_next = torch.min(q1_next, q2_next)
                
                # Target value: r + γ * (min(Q1', Q2') - α * log π)
                q_target = rewards + (1 - dones) * self.gamma * (q_next - self.alpha * next_log_probs)
            
            # Current Q-values
            q1_current, _ = self.q1.forward(states, actions, q_hidden)
            q2_current, _ = self.q2.forward(states, actions, q_hidden)
            
            # Q-losses
            q1_loss = F.mse_loss(q1_current, q_target)
            q2_loss = F.mse_loss(q2_current, q_target)
            
            # Update Q-networks
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()
            
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()
            
            # --- Update Policy ---
            new_actions, log_probs, _, _ = self.policy.sample(states, policy_hidden)
            
            q1_new, _ = self.q1.forward(states, new_actions, q_hidden)
            q2_new, _ = self.q2.forward(states, new_actions, q_hidden)
            q_new = torch.min(q1_new, q2_new)
            
            policy_loss = (self.alpha * log_probs - q_new).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # --- Update Temperature ---
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp()
                losses['alpha'] += alpha_loss.item()
            
            # --- Soft Update Target Networks ---
            soft_update(self.q1, self.q1_target, self.tau)
            soft_update(self.q2, self.q2_target, self.tau)
            
            # Accumulate losses
            losses['q1'] += q1_loss.item()
            losses['q2'] += q2_loss.item()
            losses['policy'] += policy_loss.item()
            
            self.total_steps += 1
        
        # Average losses
        for key in losses:
            losses[key] /= updates
        
        # Store statistics
        self.training_stats['q1_loss'].append(losses['q1'])
        self.training_stats['q2_loss'].append(losses['q2'])
        self.training_stats['policy_loss'].append(losses['policy'])
        self.training_stats['alpha_loss'].append(losses['alpha'])
        self.training_stats['alpha'].append(self.alpha.item())
        
        return losses
    
    def save(self, filepath: str):
        """Save agent parameters."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'auto_entropy_tuning': self.auto_entropy_tuning,
                'target_entropy': self.target_entropy
            }
        }
        
        if self.auto_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        if self.auto_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.total_steps = checkpoint['total_steps']
        
        print(f"Agent loaded from {filepath}")
    
    def get_stats(self):
        """Get training statistics."""
        return {
            'total_updates': self.total_steps,
            'alpha': self.alpha.item(),
            'training_stats': self.training_stats
        }


if __name__ == '__main__':
    # Test agent
    print("Testing SAC-GRU Agent")
    print("=" * 60)
    
    # Configuration
    state_dim = 44
    action_dim = 4
    
    # Create agent
    print("\n1. Creating agent...")
    agent = SAC_GRU_Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        gru_dim=128,
        batch_size=32
    )
    print(f"   Device: {agent.device}")
    print(f"   Auto entropy tuning: {agent.auto_entropy_tuning}")
    print(f"   Target entropy: {agent.target_entropy}")
    
    # Test action selection
    print("\n2. Testing action selection...")
    state = np.random.randn(state_dim)
    hidden = agent.policy.init_hidden(1).to(agent.device)
    
    action, hidden_new = agent.select_action(state, hidden, evaluate=False)
    print(f"   Action shape: {action.shape}")
    print(f"   Action range: [{action.min():.4f}, {action.max():.4f}]")
    
    # Test replay buffer
    print("\n3. Testing replay buffer...")
    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i % 10 == 0
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(agent.replay_buffer)}")
    
    # Test parameter update
    print("\n4. Testing parameter update...")
    losses = agent.update_parameters(updates=5)
    print(f"   Q1 loss: {losses['q1']:.6f}")
    print(f"   Q2 loss: {losses['q2']:.6f}")
    print(f"   Policy loss: {losses['policy']:.6f}")
    print(f"   Alpha: {agent.alpha.item():.6f}")
    
    # Test save/load
    print("\n5. Testing save/load...")
    agent.save("/tmp/test_agent.pth")
    agent.load("/tmp/test_agent.pth")
    print(f"   Save/load successful")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
