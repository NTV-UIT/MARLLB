# THEORY.md - RL Environment Integration Theory

## Table of Contents
1. [Markov Decision Process (MDP) Foundation](#mdp-foundation)
2. [State Space Design Theory](#state-space-theory)
3. [Action Space Design Theory](#action-space-theory)
4. [Reward Functions & Fairness Metrics](#reward-theory)
5. [Gym Interface Principles](#gym-theory)
6. [Load Balancing as MDP](#lb-mdp)
7. [Advanced Topics](#advanced-topics)

---

## 1. Markov Decision Process (MDP) Foundation {#mdp-foundation}

### 1.1 MDP Definition

Một MDP được định nghĩa bởi tuple `(S, A, P, R, γ)`:

**S (State Space)**: Tập hợp tất cả states có thể có
- Trong load balancing: server features, active connections, resource usage

**A (Action Space)**: Tập hợp tất cả actions có thể có
- Trong load balancing: weight assignments cho mỗi server

**P (Transition Function)**: `P(s'|s, a)` - Xác suất chuyển từ state s sang s' khi thực hiện action a
- Stochastic vì: traffic arrival random, server processing time variable

**R (Reward Function)**: `R(s, a, s')` - Reward nhận được khi chuyển từ s sang s' với action a
- Trong load balancing: fairness metrics

**γ (Discount Factor)**: `γ ∈ [0, 1]` - Trọng số cho future rewards
- γ = 0: Chỉ quan tâm immediate reward
- γ = 1: Tất cả rewards equally important
- Typical: γ = 0.99

### 1.2 Policy Definition

**Policy π**: Mapping từ states sang actions

**Deterministic Policy**: `π(s) → a`
- Mỗi state có exact một action

**Stochastic Policy**: `π(a|s)` - Xác suất chọn action a trong state s
- More exploration capability
- Useful for continuous actions (Gaussian policy)

### 1.3 Value Functions

**State Value Function** `V^π(s)`:
```
V^π(s) = E[Σ(γ^t * R_t) | s_0 = s, π]
```
Expected cumulative discounted reward starting from state s following policy π

**Action Value Function (Q-function)** `Q^π(s, a)`:
```
Q^π(s, a) = E[Σ(γ^t * R_t) | s_0 = s, a_0 = a, π]
```
Expected return starting from s, taking action a, then following π

**Bellman Equation**:
```
V^π(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V^π(s')]

Q^π(s, a) = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * Σ_{a'} π(a'|s') * Q^π(s', a')]
```

### 1.4 Optimal Policy

**Optimal Policy** `π*`:
```
π* = argmax_π V^π(s), ∀s ∈ S
```

**Bellman Optimality Equation**:
```
V*(s) = max_a Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V*(s')]

Q*(s, a) = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * max_{a'} Q*(s', a')]
```

---

## 2. State Space Design Theory {#state-space-theory}

### 2.1 State Representation Requirements

**Markov Property**: State phải capture tất cả thông tin cần thiết để predict future
```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

**Observability**: State phải observable (đo được)
- Full observability: Agent thấy toàn bộ state
- Partial observability (POMDP): Agent chỉ thấy observation o_t

**Dimensionality**: Tradeoff giữa expressiveness và complexity
- Too few dimensions: Không capture đủ info (violate Markov property)
- Too many dimensions: Slow learning, overfitting

### 2.2 Feature Engineering Principles

**Temporal Information**:
- Current metrics: `n_flow_on`, `fct_mean`
- Historical metrics: `fct_mean_decay` (exponential moving average)
- Reason: Capture trends, predict future load

**Normalization**:
```python
normalized_feature = (feature - mean) / std
# Or
normalized_feature = (feature - min) / (max - min)
```
- Helps neural network convergence
- Prevents features với large magnitude dominate

**Aggregation Levels**:
- Per-server: `[n_flows, fct_mean, ...]` for each server
- Global: Total flows, average FCT across all servers
- MARLLB uses per-server cho fine-grained control

### 2.3 State Space Structures

**Flat Vector** (for feedforward nets):
```
state = [server_0_features, server_1_features, ..., server_N_features]
Shape: (N_servers × N_features,)
```

**Structured Tensor** (for recurrent nets):
```
state = [[f_0_0, f_0_1, ..., f_0_M],
         [f_1_0, f_1_1, ..., f_1_M],
         ...
         [f_N_0, f_N_1, ..., f_N_M]]
Shape: (N_servers, N_features)
```

**Graph** (for graph neural networks):
```
nodes = server_features
edges = server_to_server connections
```

MARLLB uses **structured tensor** vì:
- Compatible với GRU (sequential processing)
- Mask inactive servers
- Permutation invariance (order doesn't matter)

### 2.4 Partial Observability Handling

**Problem**: Agent không observe được complete state (e.g., future traffic)

**Solutions**:
1. **History Window**: Stack multiple observations
   ```
   state = [obs_t, obs_{t-1}, obs_{t-2}, ..., obs_{t-k}]
   ```

2. **Recurrent Networks**: GRU/LSTM maintain hidden state
   ```
   h_t = GRU(obs_t, h_{t-1})
   action = policy(h_t)
   ```

3. **Belief State**: Maintain probability distribution over states
   ```
   b(s) = P(state = s | obs_1, ..., obs_t, a_1, ..., a_{t-1})
   ```

MARLLB uses **GRU** để handle partial observability.

---

## 3. Action Space Design Theory {#action-space-theory}

### 3.1 Discrete vs. Continuous Actions

**Discrete Actions**:
```
A = {a_1, a_2, ..., a_K}  # Finite set
```

**Advantages**:
- Easy exploration (uniform sampling)
- Q-learning applicable (Q(s,a) for each action)
- Stable training

**Disadvantages**:
- Limited expressiveness
- Combinatorial explosion (N_servers × K_actions)

**Continuous Actions**:
```
A = [a_min, a_max]^D  # D-dimensional continuous space
```

**Advantages**:
- Fine-grained control
- Compact representation (D << K^N)

**Disadvantages**:
- Harder exploration
- Need policy gradient methods (DDPG, SAC)

### 3.2 Action Space for Load Balancing

**Option 1: Direct Weights** (continuous)
```
action = [w_0, w_1, ..., w_N]  # Weights for each server
```

**Option 2: Weight Adjustments** (continuous)
```
action = [Δw_0, Δw_1, ..., Δw_N]
weights_{t+1} = weights_t + action
```

**Option 3: Categorical Weights** (discrete)
```
action = [c_0, c_1, ..., c_N]  # Category for each server
c_i ∈ {0, 1, 2}  # e.g., low (1.0), medium (1.5), high (2.0)
```

**MARLLB uses Option 3** vì:
- Stable training
- Sufficient expressiveness (3 levels enough)
- Compatible với QMIX (value-based multi-agent)

### 3.3 Multi-Agent Action Spaces

**Joint Action Space**: Cartesian product của individual actions
```
A = A_1 × A_2 × ... × A_N
|A| = |A_1| × |A_2| × ... × |A_N|
```

**Challenge**: Exponential growth với number of agents

**Solutions**:
1. **Factored Actions**: Independent per-agent actions
2. **Action Masking**: Restrict invalid actions
3. **Hierarchical Actions**: Higher-level coordinator + local agents

### 3.4 Action Constraints

**Hard Constraints**:
```
Σ weights = constant  # Total capacity constraint
weights ≥ 0           # Non-negative weights
```

**Soft Constraints** (via reward shaping):
```
reward = fairness_metric(loads) - λ * constraint_violation
```

**Implementation**:
- Projection: Project actions onto constraint manifold
- Clipping: `weights = np.clip(weights, min_weight, max_weight)`
- Normalization: `weights = weights / np.sum(weights) * total_capacity`

---

## 4. Reward Functions & Fairness Metrics {#reward-theory}

### 4.1 Reward Function Design Principles

**Alignment**: Reward should align với true objective
- Bad: Reward = throughput (might overload servers)
- Good: Reward = fairness (balanced load)

**Smoothness**: Smooth rewards → stable gradients
```
r(x) = -x^2        # Smooth
r(x) = -|x|        # Non-differentiable at 0
r(x) = -1{x > 0}   # Discrete, hard to optimize
```

**Scale**: Reward magnitude affects learning
- Too small: Slow learning (gradients vanish)
- Too large: Unstable (large gradient updates)
- Typical: Normalize rewards to [-1, 1] or [0, 1]

**Sparsity**: Dense rewards > sparse rewards
- Sparse: Reward only at episode end (hard credit assignment)
- Dense: Reward at every step (easier learning)

### 4.2 Jain's Fairness Index

**Definition**:
```
J(x_1, x_2, ..., x_n) = (Σx_i)^2 / (n * Σx_i^2)
```

**Properties**:
- Range: `[1/n, 1]`
- J = 1: Perfect fairness (all x_i equal)
- J = 1/n: Worst case (all load on one server)

**Proof of Range**:

Lower bound (Cauchy-Schwarz):
```
(Σx_i)^2 ≤ n * Σx_i^2
(Σx_i)^2 / (n * Σx_i^2) ≤ 1
```
Equality when all x_i equal.

Upper bound:
```
If x_1 = X, x_2 = ... = x_n = 0:
J = X^2 / (n * X^2) = 1/n
```

**Derivatives** (for gradient-based optimization):
```
∂J/∂x_i = (2/n) * [(Σx_j) / (Σx_j^2) - (Σx_j)^2 * x_i / (Σx_j^2)^2]
```

**Gradient interpretation**: Increases J by increasing small x_i, decreasing large x_i

### 4.3 Variance-Based Fairness

**Definition**:
```
Var(x) = E[(x - μ)^2] = (1/n) * Σ(x_i - μ)^2
where μ = (1/n) * Σx_i
```

**Reward**:
```
r = -Var(x)  # Penalize variance
```

**Properties**:
- Range: `[-∞, 0]`
- Var = 0: Perfect fairness
- Larger variance → lower reward

**Standard Deviation** (alternative):
```
r = -√Var(x) = -σ
```
Less sensitive to outliers than variance.

**Coefficient of Variation** (scale-independent):
```
CV = σ / μ
r = -CV
```

### 4.4 Max-Min Fairness

**Definition**: Maximize minimum allocation
```
r = min(x_1, x_2, ..., x_n)
```

**Properties**:
- Prioritize worst-case server
- Range: `[0, ∞]` (depends on load)

**Dual formulation**: Minimize maximum load
```
r = -max(x_1, x_2, ..., x_n)
```
Useful for avoiding overload.

### 4.5 Product Fairness (Nash Welfare)

**Definition**:
```
W(x) = Π(x_i + ε)
Log W = Σ log(x_i + ε)
```
Add ε to avoid log(0).

**Properties**:
- Encourages balanced allocation
- Log transform → sum (easier optimization)

**Gradient**:
```
∂(log W)/∂x_i = 1 / (x_i + ε)
```
Larger gradient for smaller x_i → increase fairness.

### 4.6 Reward Shaping

**Potential-Based Shaping** (Ng et al., 1999):
```
F(s, s') = γ * Φ(s') - Φ(s)
r'(s, a, s') = r(s, a, s') + F(s, s')
```

**Theorem**: Potential-based shaping không thay đổi optimal policy.

**Example cho load balancing**:
```
Φ(s) = Jain_index(loads in state s)
F = γ * Jain(s') - Jain(s)
r' = base_reward + F
```
Encourages states với higher Jain index.

### 4.7 Reward Field Selection

**Problem**: Which metric to use for reward computation?

**Options**:
1. `n_flow_on`: Active connections
2. `fct_mean`: Average flow completion time
3. `fct_p90`: Tail latency
4. `flow_duration_avg_decay`: Recent flow duration

**Analysis**:

**n_flow_on**:
- Pro: Simple, directly measurable
- Con: Doesn't capture server capacity (CPU, memory)

**fct_mean**:
- Pro: User-centric metric
- Con: Aggregates over all flows (may hide tail latency)

**fct_p90**:
- Pro: Captures tail latency (important for SLA)
- Con: Sensitive to outliers

**flow_duration_avg_decay**:
- Pro: Recent workload (responsive to changes)
- Pro: Smooth (exponential moving average)
- Con: Tuning decay parameter

**MARLLB choice**: `flow_duration_avg_decay`
- Balance responsiveness và stability
- Correlates well với server load

---

## 5. Gym Interface Principles {#gym-theory}

### 5.1 OpenAI Gym API

**Core Methods**:

```python
class Env:
    def reset(self) -> observation:
        """Reset environment to initial state."""
    
    def step(self, action) -> (observation, reward, done, info):
        """Take action, return next state."""
    
    def render(self, mode='human'):
        """Visualize environment."""
    
    def close(self):
        """Clean up resources."""
```

**Spaces**:
```python
from gym import spaces

# Discrete
action_space = spaces.Discrete(n)  # {0, 1, ..., n-1}

# Box (continuous)
observation_space = spaces.Box(low=0, high=1, shape=(4,))

# MultiDiscrete
action_space = spaces.MultiDiscrete([3, 3, 3, 3])  # 4 agents, 3 actions each

# Dict
observation_space = spaces.Dict({
    'image': spaces.Box(low=0, high=255, shape=(64, 64, 3)),
    'position': spaces.Box(low=-1, high=1, shape=(2,))
})
```

### 5.2 Episode Structure

```
s_0 ← env.reset()

for t = 0, 1, 2, ...:
    a_t ← policy(s_t)
    s_{t+1}, r_t, done, info ← env.step(a_t)
    
    if done:
        break
```

**Episode Termination**:
- Success: Goal reached
- Failure: Constraint violated
- Timeout: Max steps reached
- Truncation: External signal (e.g., time limit)

### 5.3 Gym Wrappers

**Purpose**: Modify environment behavior without changing core code

**Common Wrappers**:

**Normalization**:
```python
class NormalizeObservation(gym.ObservationWrapper):
    def observation(self, obs):
        return (obs - self.mean) / (self.std + 1e-8)
```

**Reward Clipping**:
```python
class ClipReward(gym.RewardWrapper):
    def reward(self, reward):
        return np.clip(reward, -1, 1)
```

**Frame Stacking**:
```python
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        self.k = k
        self.frames = deque([], maxlen=k)
    
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate(self.frames, axis=0)
```

**Time Limit**:
```python
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info
```

### 5.4 VecEnv (Vectorized Environments)

**Purpose**: Run multiple environments in parallel

```python
class VecEnv:
    def reset(self) -> observations:
        """Reset all envs, return stacked observations."""
    
    def step(self, actions) -> (observations, rewards, dones, infos):
        """Step all envs with actions."""
```

**Benefits**:
- Faster data collection
- Better GPU utilization
- Decorrelate experiences (important for on-policy algorithms)

**Implementations**:
- `DummyVecEnv`: Sequential execution (for debugging)
- `SubprocVecEnv`: Multiprocessing (parallel)

---

## 6. Load Balancing as MDP {#lb-mdp}

### 6.1 MDP Formulation for MARLLB

**State Space S**:
```
s_t = {
    active_servers: [s_0, s_1, ..., s_k],  # Subset of N servers
    features: {
        s_i: [n_flow, fct_mean, fct_p90, fct_std, fct_mean_decay, fct_p90_decay,
              dur_mean, dur_p90, dur_std, dur_mean_decay, dur_p90_decay]
    },
    ground_truth: {  # Optional
        s_i: [cpu_usage, mem_usage, threads]
    }
}
```

**Action Space A**:
```
a_t = [w_0, w_1, ..., w_{N-1}]  # Weights for N servers
w_i ∈ {1.0, 1.5, 2.0}  # Discrete levels
```

**Transition Function P**:
```
s_{t+1} ~ P(·|s_t, a_t)
```
Stochastic vì:
- Random traffic arrival (Poisson, wiki trace, etc.)
- Variable request processing time
- Server failures/recoveries

**Reward Function R**:
```
r_t = Jain_fairness([load_0, load_1, ..., load_k])
where load_i = flow_duration_avg_decay for server i
```

**Discount Factor γ**:
```
γ = 0.99  # Value future rewards
```

### 6.2 Challenges in Load Balancing MDP

**1. High-Dimensional State Space**:
- N servers × M features → N×M dimensions
- Example: 4 servers × 11 features = 44D
- Solution: Function approximation (neural networks)

**2. Continuous State Space**:
- Feature values unbounded (flow count, FCT, etc.)
- Solution: Normalization, discretization (reservoir sampling)

**3. Partial Observability**:
- Cannot observe future traffic
- Cannot observe internal server state (queue lengths)
- Solution: Recurrent policies (GRU/LSTM)

**4. Multi-Agent Coordination**:
- Multiple load balancers (Problem 05)
- Need coordination protocol
- Solution: QMIX (centralized training, decentralized execution)

**5. Non-Stationary Environment**:
- Traffic patterns change over time
- Server capacities vary (thermal throttling, contention)
- Solution: Online learning, continual adaptation

**6. Safety Constraints**:
- Cannot overload servers (cause failures)
- Must maintain minimum performance
- Solution: Constrained RL (safe exploration)

### 6.3 Horizon and Episode Design

**Finite Horizon**:
```
Episode length T = 1000 steps
Step interval Δt = 250ms
Total episode time = 250 seconds
```

**Infinite Horizon** (more realistic):
```
Episode never ends, just reset periodically for training
```

**Tradeoff**:
- Finite: Easier credit assignment, clear episode returns
- Infinite: More realistic, but need value function bootstrapping

**MARLLB approach**: Finite episodes for training, infinite for deployment

### 6.4 State Transition Dynamics

**Deterministic Components**:
```
features_{t+1} = update_reservoir(features_t, new_flows)
weights_t = action_t
```

**Stochastic Components**:
```
new_flows ~ Poisson(λ) or WikiTrace(t)
server_processing_time ~ Exponential(μ_i)
server_failure ~ Bernoulli(p_fail)
```

**Transition Model**:
```
s_{t+1} = {
    active_servers: update_active(s_t.active_servers, failures, recoveries),
    features: {
        s_i: update_features(s_t.features[s_i], 
                            new_flows_to_s_i, 
                            completed_flows_from_s_i)
    }
}
```

---

## 7. Advanced Topics {#advanced-topics}

### 7.1 Partially Observable MDPs (POMDPs)

**POMDP Definition**: `(S, A, O, P, R, Z, γ)`
- S, A, P, R, γ: Same as MDP
- O: Observation space
- Z: Observation function `Z(o|s, a)` - probability of observing o given state s and action a

**Belief State**:
```
b(s) = P(state = s | o_1, ..., o_t, a_1, ..., a_{t-1})
```

**POMDP Solution Methods**:
1. **Belief MDP**: Solve over belief states (exponential complexity)
2. **History-Based**: Use observation history as state
3. **Recurrent Policies**: RNN maintains internal state

**MARLLB Approach**: GRU-based policy (recurrent)

### 7.2 Constrained MDPs

**Definition**: MDP with constraints
```
maximize E[Σ γ^t * r_t]
subject to E[Σ γ^t * c_t] ≤ d
```
where c_t is cost at time t, d is constraint threshold.

**Example for Load Balancing**:
```
maximize E[fairness]
subject to E[max_server_load] ≤ 80% capacity
```

**Solution Methods**:
- Lagrangian relaxation
- Primal-dual methods
- Safe RL (CPO, TRPO with constraints)

### 7.3 Transfer Learning in RL

**Problem**: Train on one task, deploy on another
- Train on simulated traffic, deploy on real traffic
- Train on 4 servers, deploy on 6 servers

**Approaches**:
1. **Domain Randomization**: Train on diverse conditions
2. **Fine-Tuning**: Pre-train then adapt
3. **Meta-Learning**: Learn to adapt quickly (MAML)

**MARLLB**: Train on various traces (Poisson, wiki) for robustness

### 7.4 Multi-Task RL

**Problem**: Single agent handles multiple tasks
- Different traffic patterns
- Different fairness objectives

**Approaches**:
1. **Shared Representation**: Common feature extractor + task-specific heads
2. **Contextual Policies**: Policy conditioned on task ID
3. **Hierarchical RL**: High-level selects task, low-level executes

### 7.5 Offline RL

**Problem**: Learn from fixed dataset (no environment interaction)

**Use Case for Load Balancing**:
- Learn from production logs
- Avoid risky exploration in live system

**Challenges**:
- Distribution shift (behavior policy ≠ learned policy)
- Extrapolation error

**Algorithms**:
- Conservative Q-Learning (CQL)
- Batch-Constrained Q-Learning (BCQ)

### 7.6 Sim-to-Real Transfer

**Challenge**: Policy trained in simulator fails in real environment

**Causes**:
- Modeling errors (transition dynamics, reward function)
- Sensor noise
- Latency

**Solutions**:
1. **Domain Adaptation**: Align simulator and real distributions
2. **Robust RL**: Train policy robust to perturbations
3. **Model Calibration**: Improve simulator accuracy
4. **Online Adaptation**: Fine-tune policy in real environment

**MARLLB Sim-to-Real**:
- Simulator: Client replay + VPP simulation
- Real: KVM testbed with Apache servers
- Gap: Network delay, server processing variance

---

## Mathematical Appendix

### A. Jain Index Derivation

**Claim**: `J(x) = (Σx_i)^2 / (n * Σx_i^2)` measures fairness.

**Proof of Properties**:

1. **Range [1/n, 1]**:
   - Upper bound: By Cauchy-Schwarz, `(Σx_i)^2 ≤ n * Σx_i^2`, so `J ≤ 1`.
   - Equality when all x_i equal.
   - Lower bound: If x_1 = X, rest = 0, then `J = X^2 / (n * X^2) = 1/n`.

2. **Monotonicity**: J increases when variance decreases (for fixed mean).

3. **Scale Invariance**: `J(c * x) = J(x)` for c > 0.

### B. Bellman Equation Proof

**Claim**: `V^π(s) = E_π[r + γ * V^π(s')]`

**Proof**:
```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γ * G_{t+1} | S_t = s]
       = E_π[R_{t+1} | S_t = s] + γ * E_π[G_{t+1} | S_t = s]
       = Σ_a π(a|s) Σ_{s'} P(s'|s,a) R(s,a,s') + γ * Σ_{s'} P(s'|s') V^π(s')
```

### C. Policy Gradient Theorem

**Theorem** (Sutton et al., 2000):
```
∇_θ J(θ) = E_π [∇_θ log π_θ(a|s) * Q^π(s, a)]
```

**Intuition**: Increase probability of actions with high Q-value.

**Proof Sketch**:
1. J(θ) = E_π[V^π(s_0)]
2. Expand using performance difference lemma
3. Apply log-derivative trick

---

## References

1. **Sutton & Barto** (2018). "Reinforcement Learning: An Introduction" (2nd ed.)
2. **Jain et al.** (1984). "A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems"
3. **Brockman et al.** (2016). "OpenAI Gym"
4. **Ng et al.** (1999). "Policy Invariance Under Reward Transformations"
5. **MARLLB Paper** (Section 3). "Multi-Agent Reinforcement Learning Load Balancer"

---

## Summary

Trong Problem 03, chúng ta implement OpenAI Gym environment cho load balancing:

**Key Concepts**:
- MDP formulation: States (server features), Actions (weights), Rewards (fairness)
- State design: Per-server features từ reservoir sampling
- Action design: Discrete weights (3 levels)
- Reward design: Jain fairness index trên flow_duration_avg_decay

**Integration**:
- Use SharedMemoryRegion (Problem 02) để communicate với VPP
- Use ReservoirSampler features (Problem 01) trong state

**Next**: Problem 04 sẽ implement SAC-GRU algorithm để learn optimal policy trong environment này.
