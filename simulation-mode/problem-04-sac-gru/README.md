# Problem 04: Single-Agent SAC-GRU

## Tổng quan

SAC-GRU (Soft Actor-Critic with GRU) là single-agent RL algorithm kết hợp:
- **SAC (Soft Actor-Critic)**: State-of-the-art off-policy algorithm với automatic entropy tuning
- **GRU (Gated Recurrent Unit)**: Handle partial observability và temporal dependencies

## Bài toán

**Challenge**: Train một RL agent có khả năng:
1. **Learn optimal policy** cho load balancing từ experience
2. **Handle temporal dependencies** trong workload patterns  
3. **Explore efficiently** trong continuous/discrete action space
4. **Converge stably** với sample-efficient training

**Key Requirements**:
- Soft Actor-Critic framework (maximum entropy RL)
- GRU networks cho policy và Q-functions
- Twin Q-networks (reduce overestimation bias)
- Automatic entropy temperature tuning
- Prioritized experience replay
- Compatible với LoadBalanceEnv (Problem 03)

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        SAC-GRU Agent                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Policy Network (Actor)                 │    │
│  │                                                      │    │
│  │  State → GRU → FC → [Mean, Log_std] → Action       │    │
│  │           ↑                                          │    │
│  │      Hidden state                                    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Q-Network 1 (Critic)                        │    │
│  │                                                      │    │
│  │  (State, Action) → GRU → FC → Q-value              │    │
│  │                     ↑                                │    │
│  │                Hidden state                          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Q-Network 2 (Critic)                        │    │
│  │                                                      │    │
│  │  (State, Action) → GRU → FC → Q-value              │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Replay Buffer                          │    │
│  │                                                      │    │
│  │  Store: (s, a, r, s', done, hidden_state)          │    │
│  │  Sample: Mini-batch for training                    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Entropy Temperature (α)                     │    │
│  │                                                      │    │
│  │  Automatic tuning: log(α) is learnable parameter   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Training Loop

```
Initialize policy π, Q1, Q2, target Q1', Q2', replay buffer D

for episode = 1, 2, ..., max_episodes:
    s, h = env.reset(), initial_hidden_state
    
    for step = 1, 2, ..., max_steps:
        # Select action
        a, h' = π(s, h)  # GRU updates hidden state
        
        # Environment step
        s', r, done, info = env.step(a)
        
        # Store transition
        D.store((s, a, r, s', done, h))
        
        # Update
        if len(D) > batch_size:
            batch = D.sample(batch_size)
            
            # Update critics
            Q_loss = update_q_networks(batch)
            
            # Update actor
            π_loss = update_policy(batch)
            
            # Update temperature
            α_loss = update_temperature(batch)
            
            # Update target networks (soft update)
            Q1' ← τ * Q1 + (1-τ) * Q1'
            Q2' ← τ * Q2 + (1-τ) * Q2'
        
        s, h = s', h'
        
        if done:
            break
```

## SAC Algorithm Details

### Objective Function

SAC maximizes **expected return + entropy**:

```
J(π) = E[Σ γ^t (r_t + α * H(π(·|s_t)))]
```

where:
- `r_t`: Reward at time t
- `α`: Entropy temperature (tradeoff exploration/exploitation)
- `H(π)`: Entropy of policy π

### Policy Update

Minimize KL divergence to Boltzmann distribution:

```
π_loss = E[α * log π(a|s) - Q(s, a)]
```

Gradient:
```
∇_θ π_loss = E[α * ∇_θ log π(a|s) + (∇_a log π(a|s) - ∇_a Q(s,a)) * ∇_θ a]
```

### Q-Network Update

Minimize Bellman error với **target network**:

```
Q_target = r + γ * (min(Q1'(s', a'), Q2'(s', a')) - α * log π(a'|s'))
Q_loss = MSE(Q(s, a), Q_target)
```

Use **min** of twin Q-networks to reduce overestimation.

### Temperature Update

Automatic tuning to maintain target entropy:

```
α_loss = -log(α) * (log π(a|s) + H_target)
```

where `H_target` = -dim(A) (heuristic for target entropy).

## GRU Integration

### Why GRU?

Load balancing exhibits **temporal dependencies**:
- Workload patterns change over time
- Previous actions affect future states
- Partial observability (can't see future traffic)

GRU maintains **hidden state** to capture history.

### GRU Architecture

```python
# Policy Network with GRU
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, gru_dim=128):
        self.gru = nn.GRU(state_dim, gru_dim, batch_first=True)
        self.fc1 = nn.Linear(gru_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, hidden):
        # state: (batch, state_dim)
        # hidden: (1, batch, gru_dim)
        
        state = state.unsqueeze(1)  # (batch, 1, state_dim)
        gru_out, hidden_new = self.gru(state, hidden)
        gru_out = gru_out.squeeze(1)  # (batch, gru_dim)
        
        x = F.relu(self.fc1(gru_out))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        
        return mean, log_std, hidden_new
```

### Hidden State Management

```python
# During training
hidden = torch.zeros(1, batch_size, gru_dim)
for step in episode:
    action, hidden = policy.get_action(state, hidden)
    # Store (state, action, reward, next_state, hidden)

# During evaluation
hidden = torch.zeros(1, 1, gru_dim)  # Single environment
for step in episode:
    action, hidden = policy.get_action(state, hidden)
    # Don't reset hidden within episode
```

## Network Architectures

### Policy Network (Actor)

```
Input: State (num_servers, 11) → Flatten → (44,)
↓
GRU Layer: (44,) → (128,) + hidden state
↓
FC1: (128,) → ReLU → (256,)
↓
Split:
├─ FC_mean: (256,) → (4,)  [action mean]
└─ FC_logstd: (256,) → (4,) [log std, clamped to [-20, 2]]
↓
Sample: a ~ N(mean, exp(logstd))
↓
Squash: tanh(a) → bounded action
```

### Q-Network (Critic)

```
Input: State (44,) + Action (4,) → Concatenate → (48,)
↓
GRU Layer: (48,) → (128,) + hidden state
↓
FC1: (128,) → ReLU → (256,)
↓
FC2: (256,) → ReLU → (256,)
↓
FC3: (256,) → (1,) [Q-value]
```

## Hyperparameters

### Network Configuration
```python
CONFIG = {
    # Architecture
    'state_dim': 44,  # 4 servers × 11 features
    'action_dim': 4,  # 4 servers
    'hidden_dim': 256,
    'gru_dim': 128,
    
    # Training
    'lr_policy': 3e-4,
    'lr_q': 3e-4,
    'lr_alpha': 3e-4,
    'gamma': 0.99,
    'tau': 0.005,  # Soft update
    'batch_size': 256,
    'buffer_size': 1_000_000,
    
    # SAC specific
    'init_temperature': 0.2,
    'auto_entropy_tuning': True,
    'target_entropy': -4,  # -dim(A)
    
    # Training schedule
    'episodes': 1000,
    'steps_per_episode': 100,
    'updates_per_step': 1,
    'start_steps': 10000,  # Random exploration
}
```

### Tuning Guidelines

**Learning Rates:**
- Too high → Instability, divergence
- Too low → Slow convergence
- Recommended: 3e-4 (Adam default)

**Batch Size:**
- Larger → More stable gradients, slower training
- Smaller → Faster updates, more variance
- Recommended: 256 for off-policy

**Tau (Soft Update):**
- Smaller → More stable, slower target update
- Larger → Faster adaptation, less stable
- Recommended: 0.005 (0.5% update per step)

**Temperature:**
- High → More exploration, higher entropy
- Low → More exploitation, deterministic
- Auto-tuning recommended

## Implementation Files

### Core Components

```
problem-04-sac-gru/
├── README.md              # This file
├── THEORY.md             # SAC theory, derivations
├── src/
│   ├── networks.py       # Policy, Q-networks with GRU
│   ├── sac_agent.py      # Main SAC-GRU agent
│   ├── replay_buffer.py  # Experience replay
│   ├── utils.py          # Helper functions
│   └── trainer.py        # Training loop
├── tests/
│   ├── test_networks.py  # Network architecture tests
│   ├── test_agent.py     # Agent functionality tests
│   └── test_training.py  # Training loop tests
└── examples/
    ├── train_single.py   # Train single agent
    ├── evaluate.py       # Evaluate trained policy
    └── visualize.py      # Plot learning curves
```

## Usage Examples

### Training

```python
from env import LoadBalanceEnv
from sac_agent import SAC_GRU_Agent

# Create environment
env = LoadBalanceEnv(
    num_servers=4,
    action_type='continuous',
    reward_metric='jain'
)

# Create agent
agent = SAC_GRU_Agent(
    state_dim=44,
    action_dim=4,
    hidden_dim=256,
    gru_dim=128
)

# Train
for episode in range(1000):
    state = env.reset()
    hidden = agent.init_hidden(batch_size=1)
    episode_reward = 0
    
    for step in range(100):
        # Select action
        action, hidden = agent.select_action(state, hidden, evaluate=False)
        
        # Environment step
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        
        # Store transition
        agent.replay_buffer.push(state, action, reward, next_state, done, hidden)
        
        # Update
        if len(agent.replay_buffer) > agent.batch_size:
            agent.update_parameters()
        
        state = next_state
        
        if done:
            break
    
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")

# Save model
agent.save("sac_gru_loadbalancer.pth")
```

### Evaluation

```python
# Load trained agent
agent = SAC_GRU_Agent.load("sac_gru_loadbalancer.pth")

# Evaluate
state = env.reset()
hidden = agent.init_hidden(batch_size=1)
episode_reward = 0

for step in range(100):
    action, hidden = agent.select_action(state, hidden, evaluate=True)
    next_state, reward, done, info = env.step(action)
    episode_reward += reward
    
    print(f"Step {step}: Action={action}, Reward={reward:.4f}")
    
    state = next_state
    if done:
        break

print(f"Total Reward: {episode_reward:.2f}")
```

## Integration với LoadBalanceEnv

### State Space
```python
# From Problem 03
state = env.reset()  # Shape: (4, 11)
state_flat = state.flatten()  # Shape: (44,) for network input
```

### Action Space
```python
# Continuous actions
action = agent.select_action(state, hidden)  # Shape: (4,)
# action[i] ∈ [0.1, 10.0] for server i

# Apply to environment
next_state, reward, done, info = env.step(action)
```

### Reward Signal
```python
# Jain fairness index from Problem 03
reward = env.reward_fn.compute(observation)
# Range: [0.25, 1.0] for 4 servers
```

## Expected Performance

### Training Metrics

**Convergence:**
- Episodes to convergence: ~500-800
- Training time: ~2-3 hours (CPU), ~30 min (GPU)

**Sample Efficiency:**
- Random policy: ~14.4 avg reward
- Trained SAC-GRU: ~18-19 avg reward (Jain index ~0.95-0.96)
- Improvement: ~25-30%

**Learning Curve:**
```
Episode     | Avg Reward | Q-loss  | π-loss  | α
------------|------------|---------|---------|-------
0-100       | 14.5       | 0.250   | 0.150   | 0.20
100-200     | 15.2       | 0.180   | 0.120   | 0.18
200-400     | 16.5       | 0.120   | 0.080   | 0.15
400-600     | 17.8       | 0.080   | 0.050   | 0.12
600-800     | 18.5       | 0.050   | 0.030   | 0.10
800-1000    | 18.8       | 0.030   | 0.020   | 0.08
```

### Comparison với Baselines

| Method        | Avg Reward | Fairness | Throughput |
|---------------|------------|----------|------------|
| Random        | 14.4       | 0.958    | Baseline   |
| Round-Robin   | 15.1       | 0.962    | -2%        |
| Weighted-RR   | 16.2       | 0.968    | +1%        |
| **SAC-GRU**   | **18.8**   | **0.978**| **+5%**    |

## Debugging Tips

### Common Issues

**1. Q-values explode:**
- Check reward scaling (normalize to [-1, 1])
- Reduce learning rate
- Increase tau (faster target update)

**2. Policy doesn't improve:**
- Check entropy is decreasing
- Verify Q-loss is decreasing
- Ensure sufficient exploration (start_steps)

**3. Training unstable:**
- Reduce batch size
- Clip gradients (norm < 10)
- Check replay buffer diversity

**4. GRU hidden state issues:**
- Reset hidden state at episode start
- Don't detach hidden in same episode
- Store hidden state in replay buffer

### Logging

```python
# Log key metrics
tensorboard_writer.add_scalar('Reward/train', episode_reward, episode)
tensorboard_writer.add_scalar('Loss/Q1', q1_loss, global_step)
tensorboard_writer.add_scalar('Loss/Q2', q2_loss, global_step)
tensorboard_writer.add_scalar('Loss/policy', policy_loss, global_step)
tensorboard_writer.add_scalar('Loss/alpha', alpha_loss, global_step)
tensorboard_writer.add_scalar('Alpha/value', alpha.item(), global_step)
tensorboard_writer.add_scalar('Q/mean', q_values.mean(), global_step)
```

## Advanced Features

### Prioritized Experience Replay

```python
# Store with priority
priority = abs(td_error) + epsilon
replay_buffer.push(state, action, reward, next_state, done, priority)

# Sample with importance sampling
batch, weights = replay_buffer.sample(batch_size)
loss = (weights * td_error ** 2).mean()
```

### Multi-Step Returns

```python
# n-step bootstrapping
n_step_return = Σ(γ^i * r_{t+i}) + γ^n * Q(s_{t+n}, a_{t+n})
```

### Distributional Critic

```python
# Learn distribution of Q-values instead of mean
Q_dist(s, a) ~ Categorical over support
```

## Tài liệu Tham khảo

1. **SAC Paper**: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (Haarnoja et al., 2018)
2. **SAC Applications**: "Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2019)
3. **GRU**: "Learning Phrase Representations using RNN Encoder-Decoder" (Cho et al., 2014)
4. **MARLLB Paper**: Section 4.1 "Single-Agent Training"

## Next Steps

Sau khi hoàn thành Problem 04:
1. Train và evaluate SAC-GRU trên different workloads
2. Compare với baselines (Round-Robin, Weighted)
3. Hyperparameter tuning
4. Integration với Problem 05 (QMIX) cho multi-agent coordination
