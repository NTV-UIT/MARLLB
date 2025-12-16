# Problem 03: RL Environment Integration

## Tổng quan

RL Environment là cầu nối giữa load balancing system (VPP + shared memory) và RL algorithms (SAC-GRU, QMIX). Nó implement OpenAI Gym interface để tương thích với các RL frameworks như Stable-Baselines3, RLlib.

## Bài toán

**Challenge**: Thiết kế Gym environment sao cho:
1. **State space** phản ánh đúng server load conditions
2. **Action space** cho phép RL agent điều khiển weights
3. **Reward function** encourage fairness và performance

**Key Requirements**:
- OpenAI Gym compatible (methods: `reset()`, `step()`, `render()`)
- Support cả discrete và continuous action spaces
- Multiple fairness metrics (Jain index, variance, max-min)
- Integration với shared memory (Problem 02)
- Integration với reservoir sampling (Problem 01)

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    LoadBalanceEnv                            │
│                                                              │
│  reset() ────────────────────────────────────────────┐      │
│     │                                                 │      │
│     ▼                                                 │      │
│  Read observation from SharedMemory                  │      │
│     │                                                 │      │
│     ▼                                                 │      │
│  Extract state = [active_servers, features, gt]     │      │
│     │                                                 │      │
│     └─────────────────────────────────────────────────      │
│                                                              │
│  step(action) ────────────────────────────────────┐         │
│     │                                              │         │
│     ▼                                              │         │
│  Convert action → weights                         │         │
│     │                                              │         │
│     ▼                                              │         │
│  Write weights to SharedMemory                    │         │
│     │                                              │         │
│     ▼                                              │         │
│  Wait for next observation (e.g., 250ms)          │         │
│     │                                              │         │
│     ▼                                              │         │
│  Read new observation                             │         │
│     │                                              │         │
│     ▼                                              │         │
│  Compute reward (fairness metric)                 │         │
│     │                                              │         │
│     ▼                                              │         │
│  Return (state, reward, done, info)               │         │
│     │                                              │         │
│     └──────────────────────────────────────────────         │
└─────────────────────────────────────────────────────────────┘
```

## State Space Design

### State Components

Từ MARLLB paper, state gồm 3 phần:

```python
state = {
    'active_as': [0, 1, 2, 3],  # List of active server IDs
    'feature_as': np.array([    # Features per server [N_servers × N_features]
        # Server 0
        [n_flow_on, fct_mean, fct_p90, fct_std, fct_mean_decay, fct_p90_decay,
         duration_mean, duration_p90, duration_std, duration_mean_decay, duration_p90_decay],
        # Server 1
        [...],
        ...
    ]),
    'gt': np.array([            # Ground truth (optional) [N_servers × 3]
        [cpu_usage, memory_usage, apache_threads],
        ...
    ])
}
```

### Feature Dimensions

**Per-server features** (11 dimensions):
1. `n_flow_on`: Active flows (counter)
2-6. FCT reservoir features (5): mean, p90, std, mean_decay, p90_decay
7-11. Duration reservoir features (5): mean, p90, std, mean_decay, p90_decay

**Total state size**: `N_servers × 11` (e.g., 4 servers → 44 dims)

### State Representation Options

**Option 1: Flat vector** (for feedforward networks)
```python
state = np.concatenate([
    active_bitmap,  # [N_servers] binary
    features.flatten()  # [N_servers × 11]
])
# Shape: (N_servers + N_servers × 11,) = (N_servers × 12,)
```

**Option 2: Structured** (for GRU/LSTM)
```python
state = {
    'features': features,  # [N_servers, 11]
    'mask': active_mask    # [N_servers] for inactive servers
}
```

**MARLLB uses Option 2** với GRU networks.

## Action Space Design

### Discrete Action Space

Each server has K discrete actions (e.g., K=3):
```python
action_space = spaces.MultiDiscrete([K] * N_servers)
# Example: [0, 1, 2, 1] = server 0: action 0, server 1: action 1, etc.

# Action meanings
ACTION_WEIGHTS = [1.0, 1.5, 2.0]  # Predefined weights

# Convert action to weights
weights[i] = ACTION_WEIGHTS[action[i]]
```

**Advantages**:
- Simple, bounded
- Easy to explore
- Stable training

**Disadvantages**:
- Limited expressiveness
- Discrete jumps

### Continuous Action Space

```python
action_space = spaces.Box(low=0.0, high=np.inf, shape=(N_servers,))
# Example: [1.23, 0.87, 1.54, 1.01]

# Actions are direct weights (often need clipping/normalization)
weights = np.clip(action, 0.1, 10.0)
```

**Advantages**:
- Fine-grained control
- More expressive

**Disadvantages**:
- Harder to explore
- May need careful tuning

**MARLLB primarily uses discrete actions** for stability.

## Reward Function Design

### Fairness Metrics

The core challenge: Balance load fairly across servers.

#### 1. Jain's Fairness Index

```python
def jain_fairness(loads):
    """
    Jain's index: 1/N <= J <= 1
    J = 1: Perfect fairness
    J = 1/N: Worst case (all load on one server)
    """
    n = len(loads)
    sum_loads = np.sum(loads)
    sum_squared = np.sum(loads ** 2)
    
    if sum_loads == 0:
        return 1.0
    
    return (sum_loads ** 2) / (n * sum_squared)
```

**Properties**:
- Scale-independent
- Bounded [1/N, 1]
- Smooth

#### 2. Variance-Based

```python
def variance_reward(loads):
    """
    Penalize variance (prefer uniform distribution)
    """
    return -np.var(loads)
```

**Variants**:
- `-np.var(loads)`: Direct penalty
- `np.exp(-k * np.var(loads))`: Exponential (smoother)

#### 3. Max-Min Fairness

```python
def max_min_reward(loads):
    """
    Minimize maximum load (avoid overload)
    """
    return -np.max(loads)
```

**Use case**: Prioritize worst-case performance

#### 4. Product Fairness

```python
def product_fairness(loads):
    """
    Maximize product (encourages balance)
    """
    return np.prod(loads + epsilon)  # Add epsilon to avoid 0
```

### Reward Field Selection

Which metric to use for computing reward?

**Options**:
1. `n_flow_on`: Active flows
2. `fct_mean`: Average FCT
3. `fct_p90`: Tail latency
4. `flow_duration_avg_decay`: Recent flow duration

**MARLLB default**: `flow_duration_avg_decay`
- Captures recent workload
- Responsive to changes
- Correlates with user experience

### Complete Reward Function

```python
def compute_reward(obs, reward_field='flow_duration_avg_decay', 
                  reward_type='jain'):
    # Extract field values for all servers
    values = [obs['server_stats'][sid][reward_field] 
             for sid in obs['active_servers']]
    
    # Compute fairness metric
    if reward_type == 'jain':
        reward = jain_fairness(values)
    elif reward_type == 'var':
        reward = -np.var(values)
    elif reward_type == 'max':
        reward = -np.max(values)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    return reward
```

## Gym Interface Implementation

### Core Methods

```python
class LoadBalanceEnv(gym.Env):
    """OpenAI Gym environment for load balancing."""
    
    def __init__(self, shm_name, num_servers, action_type='discrete'):
        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(num_servers, 11), 
            dtype=np.float32
        )
        
        if action_type == 'discrete':
            self.action_space = spaces.MultiDiscrete([3] * num_servers)
        else:
            self.action_space = spaces.Box(
                low=0, high=10, 
                shape=(num_servers,), 
                dtype=np.float32
            )
        
        # Shared memory connection
        self.shm = SharedMemoryRegion.attach(shm_name)
    
    def reset(self):
        """Reset environment, return initial observation."""
        obs = self.shm.read_observation()
        return self._process_observation(obs)
    
    def step(self, action):
        """Take action, return (obs, reward, done, info)."""
        # Convert action to weights
        weights = self._action_to_weights(action)
        
        # Write to shared memory
        self.shm.write_action(
            sequence_id=self.last_seq_id,
            weights=weights
        )
        
        # Wait for next observation
        time.sleep(self.step_interval)
        
        # Read new observation
        obs = self.shm.read_observation()
        
        # Compute reward
        reward = self._compute_reward(obs)
        
        # Check if done
        done = self._check_done()
        
        # Additional info
        info = {
            'sequence_id': obs['sequence_id'],
            'active_servers': obs['active_servers']
        }
        
        return self._process_observation(obs), reward, done, info
    
    def render(self, mode='human'):
        """Visualize environment state."""
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"Active servers: {self.last_obs['active_servers']}")
            # Print server stats
```

## Integration with Previous Components

### Using Reservoir Sampling (Problem 01)

```python
# Reservoir features are already computed and in shared memory
# Just extract from observation
features = obs['server_stats'][server_id]['reservoir_features']

# Or compute on-the-fly if needed
from reservoir import ReservoirSampler
reservoir = ReservoirSampler(capacity=128)
# ... add samples ...
features = reservoir.get_feature_vector()
```

### Using Shared Memory (Problem 02)

```python
from shm_region import SharedMemoryRegion

class LoadBalanceEnv(gym.Env):
    def __init__(self, shm_name='marllb_lb0'):
        self.shm = SharedMemoryRegion.attach(shm_name)
    
    def step(self, action):
        # Write action
        self.shm.write_action(seq_id, weights)
        
        # Read observation
        obs = self.shm.read_observation()
```

## Configuration

### Environment Parameters

```python
ENV_CONFIG = {
    'shm_name': 'marllb_lb0',
    'num_servers': 4,
    'action_type': 'discrete',  # or 'continuous'
    'step_interval': 0.25,  # seconds between steps
    'reward_field': 'flow_duration_avg_decay',
    'reward_type': 'jain',  # 'jain', 'var', 'max', 'product'
    'max_steps': 10000,
    'use_ground_truth': False,  # Include CPU/memory in state
}
```

## Testing Strategy

### 1. Basic Functionality
- Environment creation
- reset() returns valid state
- step() returns (state, reward, done, info)
- Action space sampling works

### 2. Integration Tests
- Shared memory communication
- Reward computation correctness
- Episode termination

### 3. Policy Tests
- Random policy can run
- Reward accumulation reasonable
- No crashes during long runs

### 4. Performance Tests
- Step latency < 300ms (250ms wait + 50ms overhead)
- Memory usage stable
- No memory leaks

## Expected Results

### State Space Validation
```python
env = LoadBalanceEnv(num_servers=4)
obs = env.reset()

assert obs.shape == (4, 11)  # 4 servers × 11 features
assert np.all(obs >= 0)  # All non-negative
assert np.all(np.isfinite(obs))  # No NaN/Inf
```

### Reward Range
```python
# Jain index: [0.25, 1.0] for 4 servers
# Variance: [-inf, 0]
# Max: depends on load distribution
```

### Episode Statistics
```python
episode_return = 0
for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    episode_return += reward

print(f"Episode return: {episode_return}")
# Expected: Depends on fairness, typically -10 to +10 for Jain
```

## File Structure

```
problem-03-rl-environment/
├── README.md              # This file
├── THEORY.md             # RL theory, MDP formulation
├── src/
│   ├── env.py            # LoadBalanceEnv class
│   ├── rewards.py        # Reward functions
│   └── wrappers.py       # Gym wrappers (normalization, etc.)
├── tests/
│   ├── test_env.py       # Basic env tests
│   ├── test_rewards.py   # Reward computation tests
│   └── test_integration.py  # Integration with shm
└── examples/
    ├── random_policy.py  # Random agent
    └── manual_control.py # Interactive testing
```

## Integration với RL Algorithms

### Compatible Algorithms

1. **Single-Agent**:
   - DQN (discrete actions)
   - PPO (discrete or continuous)
   - SAC (continuous)
   - A3C/A2C

2. **Multi-Agent**:
   - QMIX (value-based)
   - MADDPG (policy-based)
   - MAPPO

### Stable-Baselines3 Example

```python
from stable_baselines3 import PPO
from env import LoadBalanceEnv

env = LoadBalanceEnv(num_servers=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("ppo_loadbalance")
```

### Custom Algorithm (SAC-GRU)

```python
from env import LoadBalanceEnv
from sac_gru import SAC_GRU  # Problem 04

env = LoadBalanceEnv(num_servers=4)

agent = SAC_GRU(
    state_dim=44,  # 4 servers × 11 features
    action_dim=3,  # Discrete actions
    num_heads=4    # One per server
)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state)
        state = next_state
    
    agent.update()
```

## Tài liệu Tham khảo

1. **OpenAI Gym**: https://gym.openai.com/docs/
2. **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
3. **MARLLB Paper**: Section 3.1 "MDP Formulation"
4. **Fairness Metrics**: Jain et al. "A Quantitative Measure of Fairness"

## Next Steps

Sau khi hoàn thành Problem 03:
1. Test với random policy
2. Integrate với Problem 04 (SAC-GRU) để train real agent
3. Evaluate trên different workloads từ `data/trace/`
4. Compare với baseline methods (Maglev, ECMP, LSQ)
