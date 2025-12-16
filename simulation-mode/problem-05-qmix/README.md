# Problem 05: Multi-Agent QMIX Coordination

## Overview

This problem implements **QMIX (Q-Mixing Networks)**, a multi-agent reinforcement learning algorithm for coordinating multiple load balancers in a distributed system. QMIX enables decentralized execution while maintaining centralized training through value function factorization.

### Key Concept: CTDE Paradigm

**Centralized Training, Decentralized Execution (CTDE)**:
- **Training**: Uses global state information and coordination
- **Execution**: Each agent acts independently using only local observations

This paradigm is crucial for scalable multi-agent systems in production environments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QMIX Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Agent 1    │  │   Agent 2    │  │   Agent N    │      │
│  │  Q-Network   │  │  Q-Network   │  │  Q-Network   │      │
│  │   (GRU)      │  │   (GRU)      │  │   (GRU)      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘              │
│                            │                                  │
│                            ▼                                  │
│                  ┌──────────────────┐                        │
│                  │  Mixing Network  │                        │
│                  │                  │                        │
│                  │  w₁ = f₁(state) │                        │
│                  │  w₂ = f₂(state) │                        │
│                  │  b  = f₃(state) │                        │
│                  │                  │                        │
│                  │  Q_tot = Σ(w·Q) │                        │
│                  └──────────────────┘                        │
│                            │                                  │
│                            ▼                                  │
│                      Global Q-value                          │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Agent Networks**: Individual Q-networks with GRU (from Problem 04)
2. **Mixing Network**: Combines individual Q-values into global Q_tot
3. **Hypernetworks**: Generate mixing network weights from global state
4. **Replay Buffer**: Stores multi-agent episodes
5. **Coordinator**: Manages agent communication (TCP-based)

---

## QMIX Algorithm

### Value Factorization

QMIX factorizes the joint action-value function Q_tot(τ, **a**) into individual agent Q-values:

```
Q_tot(τ, a₁, a₂, ..., aₙ) = f(Q₁(τ₁, a₁), Q₂(τ₂, a₂), ..., Qₙ(τₙ, aₙ); s)
```

Where:
- `τᵢ`: Action-observation history for agent i
- `aᵢ`: Action of agent i
- `s`: Global state
- `f(·)`: Mixing function (monotonic in each Qᵢ)

### Monotonicity Constraint

**Key Property**: The mixing network enforces monotonicity:

```
∂Q_tot / ∂Qᵢ ≥ 0  for all i
```

This ensures that:
- Maximizing Q_tot also maximizes each Qᵢ
- Individual Greedy Maximization (IGM) principle holds
- Decentralized execution is optimal

### Mixing Network Architecture

The mixing network uses **hypernetworks** to generate weights:

```python
# State-dependent weights (always positive)
w₁ = |hypernetwork₁(state)|     # ReLU activation
w₂ = |hypernetwork₂(state)|     # ReLU activation
b  = hypernetwork₃(state)       # Can be negative

# Mixing computation (2-layer example)
hidden = ELU(w₁ · [Q₁, Q₂, ..., Qₙ] + b₁)
Q_tot = w₂ · hidden + b₂
```

**Why absolute weights?**
- Ensures `∂Q_tot/∂Qᵢ ≥ 0`
- Maintains monotonicity constraint
- Allows decentralized execution

---

## Training Process

### 1. Episode Collection (Decentralized)

```
For each agent i in parallel:
    1. Observe local state: oᵢ
    2. Compute Qᵢ(τᵢ, aᵢ) for all actions
    3. Select action: aᵢ = argmax Qᵢ(τᵢ, aᵢ)
    4. Execute action
    5. Observe reward: rᵢ
```

### 2. Centralized Training

```
Sample batch of episodes from replay buffer

For each episode:
    1. Compute individual Q-values: Q₁, Q₂, ..., Qₙ
    2. Mix Q-values: Q_tot = MixingNetwork(Q₁, ..., Qₙ, state)
    3. Compute target: y = r + γ · max Q_tot'
    4. Loss: L = (Q_tot - y)²
    5. Backpropagate through mixing network AND agent networks
```

### 3. Communication Protocol

**TCP-based Coordination** (for distributed deployment):

```
┌──────────────┐      TCP       ┌──────────────┐
│   Agent 1    │ ◄────────────► │ Coordinator  │
│ (Load Bal 1) │                │   (Central)  │
└──────────────┘                └──────────────┘
                                       ▲
┌──────────────┐      TCP              │
│   Agent 2    │ ◄─────────────────────┤
│ (Load Bal 2) │                       │
└──────────────┘                       │
                                       │
┌──────────────┐      TCP              │
│   Agent N    │ ◄─────────────────────┘
│ (Load Bal N) │
└──────────────┘
```

**Message Types**:
- `OBSERVATION`: Agent sends local observation to coordinator
- `ACTION`: Coordinator sends action to agent
- `REWARD`: Agent sends reward to coordinator
- `EPISODE_END`: Signal episode completion
- `SYNC`: Synchronize weights across agents

---

## Implementation Details

### Network Architecture

```python
class QMixingNetwork(nn.Module):
    """
    QMIX mixing network with hypernetworks.
    
    Architecture:
        state → hypernetwork₁ → w₁ (positive)
        state → hypernetwork₂ → w₂ (positive)
        state → hypernetwork₃ → b
        
        Q_tot = w₂ · ELU(w₁ · [Q₁, ..., Qₙ] + b₁) + b₂
    """
```

### Individual Agent Network

Each agent uses the SAC-GRU network from Problem 04:
- Input: Local observation (server stats, queue length)
- Hidden: GRU state (temporal dependencies)
- Output: Q-values for each action

### Replay Buffer Structure

```python
Episode = {
    'observations': [o₁, o₂, ..., oₙ],      # Per-agent observations
    'actions': [a₁, a₂, ..., aₙ],           # Per-agent actions
    'rewards': [r₁, r₂, ..., rₙ],           # Per-agent rewards
    'global_state': s,                       # Shared state
    'done': bool                             # Episode termination
}
```

---

## Hyperparameters

### Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_agents` | 4-8 | Number of load balancers |
| `mixing_embed_dim` | 32 | Mixing network hidden dimension |
| `hypernet_embed_dim` | 64 | Hypernetwork hidden dimension |
| `learning_rate` | 5e-4 | Optimizer learning rate |
| `gamma` | 0.99 | Discount factor |
| `target_update_interval` | 200 | Steps between target network updates |
| `batch_size` | 32 | Training batch size |
| `buffer_size` | 5000 | Episode replay buffer size |

### Network Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| `agent_hidden_dim` | 128 | Agent Q-network hidden dimension |
| `agent_gru_dim` | 64 | Agent GRU dimension |
| `mixing_layers` | 2 | Number of mixing layers |
| `hypernet_layers` | 2 | Number of hypernetwork layers |

### Communication

| Parameter | Value | Description |
|-----------|-------|-------------|
| `coordinator_host` | "localhost" | Coordinator TCP address |
| `coordinator_port` | 9000 | Coordinator TCP port |
| `timeout` | 5.0 | Connection timeout (seconds) |
| `max_retries` | 3 | Connection retry attempts |

---

## Usage

### 1. Single Machine Training

```python
from qmix_agent import QMIXAgent
from multi_agent_env import MultiAgentLoadBalanceEnv

# Create multi-agent environment
env = MultiAgentLoadBalanceEnv(num_agents=4)

# Create QMIX agent
agent = QMIXAgent(
    num_agents=4,
    state_dim=44,
    action_dim=4,
    mixing_embed_dim=32
)

# Train
for episode in range(1000):
    observations = env.reset()
    episode_reward = 0
    
    while not done:
        # Get actions from all agents
        actions = agent.select_actions(observations)
        
        # Environment step
        next_observations, rewards, done, info = env.step(actions)
        
        # Store transition
        agent.store_transition(observations, actions, rewards, 
                              next_observations, done, env.get_state())
        
        # Update
        agent.update()
        
        observations = next_observations
        episode_reward += sum(rewards)
    
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

### 2. Distributed Training (with TCP)

**Coordinator**:
```python
from coordinator import QMIXCoordinator

coordinator = QMIXCoordinator(
    num_agents=4,
    host="0.0.0.0",
    port=9000
)

coordinator.start()  # Wait for agents to connect
coordinator.train(num_episodes=1000)
```

**Agent** (on each load balancer):
```python
from distributed_agent import DistributedQMIXAgent

agent = DistributedQMIXAgent(
    agent_id=0,
    coordinator_host="192.168.1.100",
    coordinator_port=9000
)

agent.connect()
agent.run()  # Execute episodes
```

### 3. Evaluation

```python
# Load trained model
agent.load("checkpoints/qmix_final.pth")

# Evaluate
total_reward = 0
for episode in range(100):
    observations = env.reset()
    done = False
    
    while not done:
        # Deterministic actions
        actions = agent.select_actions(observations, evaluate=True)
        observations, rewards, done, _ = env.step(actions)
        total_reward += sum(rewards)

print(f"Average reward: {total_reward / 100:.2f}")
```

---

## Key Differences from Single-Agent SAC

| Aspect | Single-Agent SAC | Multi-Agent QMIX |
|--------|------------------|------------------|
| **Training** | Off-policy, individual | Centralized with global state |
| **Execution** | Single agent | Decentralized, parallel agents |
| **Value Function** | Single Q(s, a) | Factorized Q_tot = f(Q₁, ..., Qₙ) |
| **Communication** | None | TCP-based coordination |
| **Replay Buffer** | Single transitions | Episode trajectories |
| **Target Network** | Soft updates | Hard updates (periodic) |
| **Credit Assignment** | Direct | Through value factorization |

---

## Expected Performance

### Metrics

1. **Episode Return**: Sum of all agents' rewards
   - Baseline (random): ~50-60
   - Trained QMIX: ~75-85 (25-40% improvement)

2. **Fairness (Jain's Index)**:
   - Baseline: ~0.92
   - Trained: ~0.97+ (better load distribution)

3. **Convergence**:
   - Episodes to convergence: ~500-800
   - Training time: ~2-3 hours (4 agents, CPU)

4. **Scalability**:
   - 2 agents: ~80 return
   - 4 agents: ~75 return
   - 8 agents: ~70 return (coordination overhead)

### Learning Curves

```
Episode Return (expected trajectory):
  
100 ┤                                        ┌────────
 90 ┤                               ┌────────┘
 80 ┤                      ┌────────┘
 70 ┤             ┌────────┘
 60 ┤    ┌────────┘
 50 ┤────┘
    └────┴────┴────┴────┴────┴────┴────┴────┴────┴────
    0   100  200  300  400  500  600  700  800  900 1000
                        Episodes
```

---

## Advantages of QMIX

1. **Scalability**: Decentralized execution scales to many agents
2. **Coordination**: Mixing network learns implicit coordination
3. **Credit Assignment**: Value factorization assigns credit properly
4. **Sample Efficiency**: Shared replay buffer improves learning
5. **Deployment**: Agents can run independently in production

---

## Limitations

1. **Communication Overhead**: TCP coordination adds latency
2. **Training Complexity**: Centralized training requires global state
3. **Monotonicity Assumption**: May be restrictive for some tasks
4. **Scalability Bound**: Performance degrades with >10 agents
5. **Exploration**: Decentralized exploration can be suboptimal

---

## Extensions

### 1. Weighted QMIX (WQMIX)
- Learns non-uniform credit assignment weights
- Better for heterogeneous agents

### 2. QTRAN
- Relaxes monotonicity constraint
- More expressive factorization

### 3. QPLEX
- Duplex dueling architecture
- Improved credit assignment

### 4. Asynchronous Training
- Agents train at different rates
- Better for distributed systems

---

## References

1. **QMIX Paper**: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning", ICML 2018

2. **VDN**: Sunehag et al., "Value-Decomposition Networks For Cooperative Multi-Agent Learning", AAMAS 2018

3. **CTDE Survey**: Oliehoek & Amato, "A Concise Introduction to Decentralized POMDPs", 2016

4. **MARLLB Paper**: (Original paper describing this load balancing application)

---

## File Structure

```
problem-05-qmix/
├── README.md                 # This file
├── THEORY.md                 # Mathematical derivations
├── src/
│   ├── __init__.py
│   ├── mixing_network.py     # QMIX mixing network
│   ├── qmix_agent.py         # Main QMIX agent
│   ├── multi_agent_env.py    # Multi-agent environment wrapper
│   ├── episode_buffer.py     # Episode replay buffer
│   ├── coordinator.py        # TCP coordinator
│   └── distributed_agent.py  # Distributed agent implementation
├── tests/
│   ├── test_mixing_network.py
│   ├── test_qmix_agent.py
│   ├── test_multi_agent_env.py
│   └── test_coordinator.py
└── examples/
    ├── train_single.py       # Single-machine training
    ├── train_distributed.py  # Distributed training
    ├── evaluate.py           # Evaluation script
    └── visualize.py          # Visualization utilities
```

---

## Integration with MARLLB

This QMIX implementation integrates with:
- **Problem 03**: Uses LoadBalanceEnv as base environment
- **Problem 04**: Each agent uses SAC-GRU Q-network
- **Problem 06**: Deploys to VPP plugin for production

The complete pipeline:
1. Problem 03: Environment + rewards
2. Problem 04: Single-agent learning
3. **Problem 05: Multi-agent coordination** ← You are here
4. Problem 06: VPP integration

---

## Troubleshooting

### Common Issues

1. **Agents not coordinating**:
   - Check mixing network weights are positive
   - Verify global state contains relevant information
   - Increase mixing network capacity

2. **TCP connection failures**:
   - Check firewall settings
   - Verify coordinator is running before agents
   - Increase timeout parameter

3. **Training instability**:
   - Reduce learning rate
   - Increase target update interval
   - Use gradient clipping

4. **Poor scalability**:
   - Reduce number of agents
   - Increase coordination frequency
   - Use hierarchical architecture

---

## Next Steps

After completing this problem:
1. ✅ Test with 2, 4, 8 agents
2. ✅ Compare with independent learners baseline
3. ✅ Measure communication overhead
4. → Integrate with VPP plugin (Problem 06)
5. → Deploy to testbed with real traffic

**Status**: Implementation in progress... 🚧
