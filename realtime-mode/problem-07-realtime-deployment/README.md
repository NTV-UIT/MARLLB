# Problem 07: Real-time VPP Deployment

**Purpose**: Production deployment với VPP data plane và real-time packet processing

**Status**: 🚧 Phase 2 - Real-time Implementation  
**Completion**: 0%  
**Dependencies**: Problems 01-06 (Required)

---

## 📋 Overview

Problem 07 implements **real-time production deployment** của MARLLB load balancer:
- ✅ VPP C plugin cho packet processing
- ✅ Hardware integration với 10+ Gbps NICs
- ✅ Real-time RL inference với shared memory IPC
- ✅ Production monitoring & health checks
- ✅ Performance benchmarking

**Khác với Problem 06** (Python-only simulation):
- Problem 06: Python controller + simulated environment (training)
- Problem 07: **C plugin + real VPP + actual network traffic** (production)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      CLIENT TRAFFIC                          │
│                   (HTTP/TCP requests)                        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  VPP Load Balancer (C)                       │
│  ┌────────────────────────────────────────────────────┐     │
│  │  lb_rl_node.c (Packet Processing)                  │     │
│  │  - Parse packets (L3/L4)                           │     │
│  │  - Read weights from SHM (msg_in)                  │     │
│  │  - Select server via alias table (O(1))            │     │
│  │  - Forward packet to selected backend              │     │
│  │  - Collect statistics (latency, throughput)        │     │
│  │  - Write stats to SHM (msg_out)                    │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
                              ↕
                    Shared Memory (Problem 02)
                     /dev/shm/lb_rl_shm
                              ↕
┌──────────────────────────────────────────────────────────────┐
│              Python RL Controller (Problem 06)               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  RLController.control_loop():                      │     │
│  │    1. Read stats from SHM (msg_out)                │     │
│  │    2. Convert to observation (20 dims)             │     │
│  │    3. Agent inference (SAC-GRU/QMIX)               │     │
│  │    4. Convert action to weights                    │     │
│  │    5. Write weights to SHM (msg_in)                │     │
│  │    6. Sleep(update_interval=200ms)                 │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND SERVERS                           │
│         [Server 1] [Server 2] ... [Server N]                 │
│         Apache/Nginx with actual workloads                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
problem-07-realtime-deployment/
├── README.md                    # This file
├── DEPLOYMENT.md                # Deployment guide
├── THEORY.md                    # Real-time systems theory
│
├── vpp-plugin/                  # VPP C Plugin (Phase 2A)
│   ├── lb_rl_node.c            # Packet processing node
│   ├── lb_rl_cli.c             # VPP CLI commands
│   ├── lb_rl_api.c             # Binary API handlers
│   ├── lb_rl.h                 # Header file
│   ├── shm_reader.c            # Read msg_in from Python
│   ├── shm_writer.c            # Write msg_out to Python
│   ├── alias_table.c           # O(1) server selection
│   ├── stats_collector.c       # Collect server metrics
│   ├── CMakeLists.txt          # Build configuration
│   ├── lb_rl.api               # API definition
│   └── README.md               # Plugin documentation
│
├── src/                        # Python Integration (Phase 2B)
│   ├── realtime_controller.py # Real-time RL controller
│   ├── health_monitor.py      # Server health checks
│   ├── metrics_collector.py   # Prometheus metrics
│   └── failover_handler.py    # Graceful failover
│
├── config/                     # Configuration Files
│   ├── vpp_startup.conf       # VPP startup config
│   ├── lb_servers.yaml        # Server list & IPs
│   ├── agent_config.yaml      # RL agent parameters
│   └── monitoring.yaml        # Metrics & alerts
│
├── scripts/                    # Deployment Scripts
│   ├── build_plugin.sh        # Compile VPP plugin
│   ├── install_vpp.sh         # Install VPP on Ubuntu
│   ├── deploy.sh              # Full deployment
│   ├── start_controller.sh    # Start Python controller
│   ├── stop_all.sh            # Stop all services
│   └── benchmark.sh           # Run performance tests
│
└── tests/                      # Integration Tests
    ├── test_vpp_plugin.py     # Test C plugin
    ├── test_realtime_loop.py  # Test control loop
    ├── test_throughput.py     # Throughput benchmarks
    └── test_latency.py        # Latency benchmarks
```

---

## 🎯 Implementation Phases

### Phase 2A: VPP C Plugin (Current - 0%)

**Files to create**:
1. ✅ `vpp-plugin/lb_rl_node.c` (500+ lines)
   - Packet processing graph node
   - Read weights from shared memory
   - Alias table for O(1) sampling
   - Forward packets to selected server

2. ✅ `vpp-plugin/shm_reader.c` (200 lines)
   - Read `msg_in` from Python (weights)
   - Parse action array into alias table
   - Handle synchronization (seq numbers)

3. ✅ `vpp-plugin/shm_writer.c` (200 lines)
   - Write `msg_out` to Python (stats)
   - Aggregate per-server statistics
   - Update statistics periodically (e.g., every 100ms)

4. ✅ `vpp-plugin/alias_table.c` (300 lines)
   - Build alias table from weights: O(n) construction
   - Sample server: O(1) lookup
   - Update table when weights change

5. ✅ `vpp-plugin/lb_rl_cli.c` (200 lines)
   - VPP CLI commands:
     - `lb rl enable/disable`
     - `lb rl set-servers <ip-list>`
     - `lb rl show stats`
     - `lb rl show weights`

6. ✅ `vpp-plugin/CMakeLists.txt`
   - Build system integration
   - Link with VPP libraries

### Phase 2B: Production Controller (30%)

**Files to create**:
1. ✅ `src/realtime_controller.py` (400 lines)
   - Extends `RLController` from Problem 06
   - Add health checks
   - Add failover logic
   - Add metrics export (Prometheus)

2. ✅ `src/health_monitor.py` (250 lines)
   - Ping backend servers
   - Detect dead servers
   - Update server list dynamically

3. ✅ `src/metrics_collector.py` (200 lines)
   - Export to Prometheus/Grafana
   - Track: latency, throughput, fairness, CPU usage

### Phase 2C: Deployment & Testing (10%)

**Files to create**:
1. ✅ `scripts/deploy.sh`
   - Full deployment automation
   - VPP installation
   - Plugin compilation
   - Controller startup

2. ✅ `tests/test_throughput.py`
   - Benchmark 1-10 Gbps traffic
   - Compare RL vs baseline (round-robin, weighted)

3. ✅ `DEPLOYMENT.md`
   - Hardware requirements
   - Installation guide
   - Troubleshooting

---

## 🔧 Key Differences from Problem 06

| Aspect | Problem 06 (Simulation) | Problem 07 (Real-time) |
|--------|------------------------|------------------------|
| **Traffic** | Trace files (CSV) | Real network packets |
| **Processing** | Python env.step() | VPP C plugin (microseconds) |
| **Timing** | Discrete steps | Continuous real-time |
| **Backend** | Simulated queues | Real Apache/Nginx servers |
| **SHM Usage** | Optional | **Required** |
| **Performance** | ~1M steps/min | ~10 Gbps (hardware limited) |
| **Agent Role** | Training + inference | **Inference only** |
| **Update Rate** | Every step | Every 100-200ms |

---

## 🚀 Quick Start (When Implemented)

### 1. Build VPP Plugin
```bash
cd vpp-plugin
./build.sh
# Output: liblb_rl_plugin.so
```

### 2. Install Plugin to VPP
```bash
sudo cp liblb_rl_plugin.so /usr/lib/x86_64-linux-gnu/vpp_plugins/
sudo systemctl restart vpp
```

### 3. Configure VPP
```bash
sudo vppctl lb rl enable
sudo vppctl lb rl set-servers 192.168.1.10 192.168.1.11 192.168.1.12 192.168.1.13
sudo vppctl lb rl set-shm /dev/shm/lb_rl_shm
sudo vppctl lb rl start
```

### 4. Start Python Controller
```bash
cd ../src
python realtime_controller.py \
    --agent-type qmix \
    --model ../../problem-06-vpp-integration/checkpoints/qmix_best.pt \
    --shm-path /dev/shm/lb_rl_shm \
    --update-interval 0.2
```

### 5. Monitor
```bash
# Terminal 1: VPP stats
sudo vppctl lb rl show stats

# Terminal 2: Python metrics
curl http://localhost:9090/metrics

# Terminal 3: Grafana dashboard
firefox http://localhost:3000
```

---

## 📊 Performance Goals

| Metric | Target | Notes |
|--------|--------|-------|
| **Throughput** | 10 Gbps | Limited by 10G NIC |
| **Latency (avg)** | < 10 ms | VPP processing + network |
| **Latency (P95)** | < 20 ms | Worst-case |
| **Fairness (Jain)** | > 0.95 | Better than baseline (0.85) |
| **Agent Update** | 200 ms | Python inference overhead |
| **Packet Loss** | < 0.01% | High reliability |
| **Stability** | 24+ hours | No crashes |

---

## 🔬 Components Detail

### 1. VPP Packet Processing Node

```c
// vpp-plugin/lb_rl_node.c
static uword
lb_rl_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node, vlib_frame_t *frame)
{
    u32 n_left_from, *from;
    from = vlib_frame_vector_args(frame);
    n_left_from = frame->n_vectors;
    
    while (n_left_from > 0) {
        // 1. Parse packet
        vlib_buffer_t *b0 = vlib_get_buffer(vm, from[0]);
        ip4_header_t *ip0 = vlib_buffer_get_current(b0);
        
        // 2. Read weights from SHM (cached)
        lb_rl_weights_t *weights = lb_rl_get_weights();
        
        // 3. Sample server using alias table (O(1))
        u32 server_idx = alias_table_sample(weights->alias_table);
        
        // 4. Update statistics
        lb_rl_update_stats(server_idx, packet_size, timestamp);
        
        // 5. Forward to selected server
        vnet_buffer(b0)->ip.adj_index[VLIB_TX] = server_adj[server_idx];
        
        from++;
        n_left_from--;
    }
    
    // Periodically write stats to SHM (every 100ms)
    if (should_update_shm()) {
        lb_rl_write_stats_to_shm();
    }
    
    return frame->n_vectors;
}
```

### 2. Alias Table (O(1) Sampling)

```c
// vpp-plugin/alias_table.c

// Vose's Alias Method: O(n) build, O(1) sample
typedef struct {
    u32 num_servers;
    f32 *prob;      // Probability table
    u32 *alias;     // Alias table
} alias_table_t;

void alias_table_build(alias_table_t *table, f32 *weights, u32 n)
{
    // Step 1: Normalize weights to probabilities
    f32 sum = 0.0;
    for (u32 i = 0; i < n; i++) sum += weights[i];
    for (u32 i = 0; i < n; i++) table->prob[i] = n * weights[i] / sum;
    
    // Step 2: Partition into small/large
    u32 *small = ..., *large = ...;
    // ... (Vose's algorithm implementation)
}

u32 alias_table_sample(alias_table_t *table)
{
    // O(1) sampling
    u32 i = random() % table->num_servers;
    f32 r = random_float();
    return (r < table->prob[i]) ? i : table->alias[i];
}
```

### 3. Real-time Controller

```python
# src/realtime_controller.py

class RealtimeController(RLController):
    """Production-ready controller with monitoring & failover."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Production features
        self.health_monitor = HealthMonitor(self.servers)
        self.metrics_exporter = PrometheusExporter(port=9090)
        self.failover_handler = FailoverHandler()
        
    def _control_loop(self):
        """Main control loop with error handling."""
        while self.running:
            try:
                # 1. Health check
                dead_servers = self.health_monitor.check()
                if dead_servers:
                    self.failover_handler.handle(dead_servers)
                
                # 2. Normal RL control
                stats = self.shm.read_msg_out()
                obs = self._stats_to_observation(stats)
                action = self.agent.select_action(obs)
                weights = self._action_to_weights(action)
                self.shm.write_msg_in(weights)
                
                # 3. Export metrics
                self.metrics_exporter.update({
                    'latency_avg': np.mean(stats['latency']),
                    'fairness': compute_jain_index(stats['load']),
                    'throughput': stats['total_requests'] / self.update_interval
                })
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Control loop error: {e}")
                self.failover_handler.emergency_fallback()
```

---

## 🧪 Testing Strategy

### Unit Tests (C Plugin)
```bash
# Test alias table
./test_alias_table
# ✓ Build: O(n) complexity verified
# ✓ Sample: O(1) lookup verified
# ✓ Distribution: Chi-square test passed (p > 0.05)

# Test SHM reader/writer
./test_shm_io
# ✓ Read msg_in: Weights parsed correctly
# ✓ Write msg_out: Stats written correctly
# ✓ Seq numbers: No race conditions
```

### Integration Tests (End-to-End)
```bash
# Test full pipeline
python tests/test_realtime_loop.py
# ✓ VPP receives packets
# ✓ SHM communication works
# ✓ Agent inference successful
# ✓ Weights applied to VPP
```

### Performance Tests
```bash
# Throughput benchmark
./scripts/benchmark.sh --mode throughput --rate 10gbps
# Target: 10 Gbps sustained

# Latency benchmark
./scripts/benchmark.sh --mode latency --duration 3600
# Target: <10ms avg, <20ms P95
```

---

## 📚 References

1. **VPP Documentation**: https://fd.io/docs/vpp/
2. **VPP Plugin Development**: https://wiki.fd.io/view/VPP/How_To_Create_A_Plugin
3. **Alias Method**: Walker 1977 (O(1) weighted sampling)
4. **Shared Memory IPC**: Problem 02 documentation
5. **RL Agents**: Problems 04-05 (SAC-GRU, QMIX)

---

## ✅ Success Criteria

- [ ] VPP plugin compiles without errors
- [ ] Plugin loads successfully in VPP
- [ ] Shared memory communication works bidirectionally
- [ ] Agent inference completes in <50ms
- [ ] Throughput reaches 10 Gbps
- [ ] Latency stays <10ms average
- [ ] Fairness exceeds baseline (Jain > 0.95)
- [ ] System runs stable for 24+ hours
- [ ] Graceful failover when server dies
- [ ] Monitoring dashboard shows real-time metrics

---

## 🚧 Current Status

**Phase 2A (VPP C Plugin)**: 0% - Not started  
**Phase 2B (Production Controller)**: 30% - Skeleton code in Problem 06  
**Phase 2C (Deployment)**: 10% - Docs only

**Next Steps**:
1. Implement `lb_rl_node.c` (packet processing)
2. Implement `alias_table.c` (O(1) sampling)
3. Implement `shm_reader.c` & `shm_writer.c`
4. Test with VPP development environment
5. Benchmark on hardware testbed

---

**Author**: MARLLB Implementation Team  
**Date**: December 14, 2025  
**Status**: Phase 2 - Real-time Implementation  
**Depends On**: Problems 01-06 ✅
