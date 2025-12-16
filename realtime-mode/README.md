# MARLLB - Real-time Mode

**Multi-Agent Reinforcement Learning Load Balancer - Production Deployment**

---

## 📋 Overview

Thư mục này chứa **Problem 07** cho **real-time mode** - triển khai production với VPP data plane và real network traffic.

**Mục đích**:
- ✅ Production deployment trong data centers
- ✅ Real-time packet processing (10+ Gbps)
- ✅ Hardware load balancer với RL intelligence
- ✅ Live traffic với actual backend servers

**Khác với Simulation Mode**:
- Simulation (`../simulation-mode/`): Training với trace files, Python-only
- Real-time (Folder này): VPP C plugin + real packets + hardware NICs

---

## 🗂️ Structure

```
realtime-mode/
└── problem-07-realtime-deployment/
    ├── README.md              # This problem's documentation
    ├── DEPLOYMENT.md          # Production deployment guide
    ├── vpp-plugin/            # VPP C plugin (data plane)
    │   ├── lb_rl_node.c      # Packet processing
    │   └── alias_table.h     # O(1) server selection
    ├── src/                   # Python controller (control plane)
    │   └── realtime_controller.py
    ├── scripts/               # Deployment scripts
    │   └── start_controller.sh
    ├── config/                # Configuration files
    └── tests/                 # Integration tests
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   CLIENT TRAFFIC (Internet)                  │
│                     HTTP/TCP Requests                        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              VPP Load Balancer (C - Data Plane)              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  lb_rl_node.c                                      │     │
│  │  - Parse packets at wire speed (10+ Gbps)         │     │
│  │  - Read weights from shared memory                 │     │
│  │  - Select server via alias table (O(1))            │     │
│  │  - Forward packets to backend                      │     │
│  │  - Collect statistics (latency, throughput)        │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
                              ↕
                    Shared Memory IPC
                   (from simulation-mode/problem-02)
                              ↕
┌──────────────────────────────────────────────────────────────┐
│        Python RL Controller (Control Plane)                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │  realtime_controller.py                            │     │
│  │  - Read stats from VPP (every 200ms)               │     │
│  │  - Run agent inference (SAC-GRU or QMIX)           │     │
│  │  - Compute new weights                             │     │
│  │  - Write weights back to VPP                       │     │
│  │  - Health monitoring & failover                    │     │
│  │  - Prometheus metrics export                       │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  Uses trained models from:                                  │
│  ../simulation-mode/problem-06-vpp-integration/checkpoints/ │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND SERVERS                           │
│         [Server 1] [Server 2] ... [Server N]                 │
│         Real Apache/Nginx with actual workloads              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Problem 07: Real-time Deployment

### Status: 🚧 30% Complete (Skeleton + Documentation)

**Completed**:
- ✅ VPP C plugin skeleton (lb_rl_node.c, alias_table.h)
- ✅ Production controller (realtime_controller.py)
- ✅ Deployment scripts (start_controller.sh)
- ✅ Comprehensive documentation (README + DEPLOYMENT)

**Pending** (Phase 2):
- ⏳ Complete VPP plugin (shm_reader.c, shm_writer.c, CLI)
- ⏳ CMakeLists.txt for build system
- ⏳ Integration tests
- ⏳ Hardware testbed validation

### Files & Statistics

| Category | Lines | Files | Status |
|----------|-------|-------|--------|
| C/Headers | 631 | 2 | ✅ Skeleton |
| Python | 515 | 1 | ✅ Complete |
| Shell Scripts | 207 | 1 | ✅ Complete |
| Markdown | 988 | 2 | ✅ Complete |
| **Total** | **2,341** | **6** | **30%** |

---

## 🎯 Key Components

### 1. VPP C Plugin (Data Plane)

**File**: `problem-07-realtime-deployment/vpp-plugin/lb_rl_node.c`

**Features**:
- Packet processing at 10+ Gbps
- O(1) server selection via alias table
- Statistics collection (latency, throughput)
- Shared memory communication

**Performance**:
- Throughput: 10 Gbps (hardware limited)
- Latency: <1 microsecond per packet
- CPU: 4-8 cores for packet processing

```c
// Packet processing loop
static uword
lb_rl_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node, vlib_frame_t *frame)
{
    // 1. Read weights from SHM (cached, throttled)
    lb_rl_update_weights(lm, vm);
    
    // 2. Process packets
    while (n_left_from > 0) {
        // Parse packet
        vlib_buffer_t *b0 = vlib_get_buffer(vm, from[0]);
        
        // Select server (O(1))
        u32 server_idx = alias_table_sample(lm->alias_table);
        
        // Forward packet
        vnet_buffer(b0)->ip.adj_index[VLIB_TX] = lm->server_adj_index[server_idx];
        
        // Update stats
        lb_rl_update_server_stats(lm, server_idx, packet_size, latency);
    }
    
    // 3. Write stats to SHM (periodic)
    lb_rl_write_stats(lm, vm);
}
```

---

### 2. Production Controller (Control Plane)

**File**: `problem-07-realtime-deployment/src/realtime_controller.py`

**Features**:
- Uses pretrained models from simulation mode
- Health monitoring (detect dead servers)
- Graceful failover
- Prometheus metrics export
- Error handling & auto-restart

**Performance**:
- Agent inference: <50ms
- Update interval: 200ms (5 Hz)
- Metrics export: Real-time

```python
class RealtimeController(RLController):
    """Production controller with monitoring & failover."""
    
    def _control_loop(self):
        while self.running:
            # 1. Health check
            dead_servers = self.health_monitor.check()
            if dead_servers:
                self.failover_handler.handle(dead_servers)
            
            # 2. Read VPP stats
            stats = self.shm.read_msg_out()
            
            # 3. Agent inference
            obs = self._stats_to_observation(stats)
            action = self.agent.select_action(obs)
            
            # 4. Write weights
            weights = self._action_to_weights(action)
            self.shm.write_msg_in(weights)
            
            # 5. Export metrics
            self.metrics_exporter.update({
                'latency_avg': np.mean(stats['latency']),
                'fairness': compute_jain_index(stats['load']),
                'throughput': stats['total_requests'] / self.update_interval
            })
            
            time.sleep(self.update_interval)
```

---

## 🚀 Deployment Workflow

### Prerequisites

1. **Hardware**:
   - Server with 8+ cores, 16+ GB RAM
   - 10 Gbps NIC (Intel X520/X710, DPDK-compatible)
   - Ubuntu 20.04/22.04 LTS

2. **Trained Model**:
   - Train in simulation mode first
   - Model file: `../simulation-mode/problem-06-vpp-integration/checkpoints/qmix_best.pt`

3. **Backend Servers**:
   - 4+ servers with Apache/Nginx
   - Reachable from VPP load balancer

### Deployment Steps

#### Step 1: Train Agent (Simulation)
```bash
cd ../simulation-mode/problem-06-vpp-integration

# Train QMIX agent offline
python src/training_pipeline.py \
    --agent-type qmix \
    --num-servers 16 \
    --episodes 10000 \
    --save-path checkpoints/qmix_prod.pt

# Expected: 2-4 hours, final model saved
```

#### Step 2: Install VPP
```bash
# Add VPP repository
curl -s https://packagecloud.io/install/repositories/fdio/release/script.deb.sh | sudo bash

# Install VPP
sudo apt-get update
sudo apt-get install -y vpp vpp-plugin-core vpp-plugin-dpdk

# Verify
vpp -version  # Should be v23.06+
```

#### Step 3: Build & Install Plugin
```bash
cd realtime-mode/problem-07-realtime-deployment/vpp-plugin

# Build plugin (when fully implemented)
mkdir build && cd build
cmake ..
make -j8

# Install
sudo cp liblb_rl_plugin.so /usr/lib/x86_64-linux-gnu/vpp_plugins/
```

#### Step 4: Configure VPP
```bash
# Edit /etc/vpp/startup.conf
sudo vim /etc/vpp/startup.conf

# Start VPP
sudo systemctl start vpp

# Configure via CLI
sudo vppctl
vpp# lb rl enable
vpp# lb rl set-servers 192.168.1.10 192.168.1.11 192.168.1.12 192.168.1.13
vpp# lb rl set-vip 10.0.1.100 port 80
```

#### Step 5: Start Controller
```bash
cd realtime-mode/problem-07-realtime-deployment

# Start production controller
./scripts/start_controller.sh \
    --agent qmix \
    --model ../../simulation-mode/problem-06-vpp-integration/checkpoints/qmix_prod.pt \
    --servers "192.168.1.10 192.168.1.11 192.168.1.12 192.168.1.13" \
    --prometheus-port 9090

# ✓ Controller started (PID: 12345)
```

#### Step 6: Monitor
```bash
# Terminal 1: VPP stats
sudo vppctl lb rl show stats

# Terminal 2: Python logs
tail -f realtime_controller.log

# Terminal 3: Metrics
curl http://localhost:9090/metrics

# Terminal 4: Send traffic
curl http://10.0.1.100/
```

---

## 📊 Performance Goals

| Metric | Target | Notes |
|--------|--------|-------|
| **Throughput** | 10 Gbps | Limited by 10G NIC |
| **Latency (avg)** | <10 ms | VPP + network + backend |
| **Latency (P95)** | <20 ms | Worst-case |
| **Fairness (Jain)** | >0.95 | Better than baseline (0.85) |
| **Agent Update** | 200 ms | Python inference overhead |
| **Packet Loss** | <0.01% | High reliability |
| **Uptime** | 99.9%+ | 24/7 operation |

---

## 🔄 Integration with Simulation

### Data Flow

```
┌─────────────────────────────────────────┐
│   SIMULATION MODE (Offline Training)    │
│   ../simulation-mode/                   │
│                                         │
│   1. Train agent with traces            │
│   2. Validate in simulated env          │
│   3. Save model checkpoints             │
│                                         │
│   Output: qmix_best.pt                  │
└─────────────────────────────────────────┘
                  ↓ (Deploy model)
┌─────────────────────────────────────────┐
│   REAL-TIME MODE (Production)           │
│   ./realtime-mode/                      │
│                                         │
│   1. Load pretrained model              │
│   2. Run inference on real traffic      │
│   3. Monitor & adapt                    │
│                                         │
│   Output: Production metrics            │
└─────────────────────────────────────────┘
```

### Shared Components

Both modes use:
- ✅ **Problem 02**: Shared memory protocol
- ✅ **Problem 04/05**: Agent architectures (SAC-GRU, QMIX)
- ✅ **Reward functions**: Same fairness metrics

**Difference**: 
- Simulation: env.step() with traces
- Real-time: VPP packets with real servers

---

## 📚 Documentation

- **README.md**: Overview & architecture (this file)
- **DEPLOYMENT.md**: Step-by-step production deployment guide
- **problem-07-*/README.md**: Detailed problem documentation

---

## ⚠️ Important Notes

### When to Use Real-time Mode

Use real-time mode when:
1. ✅ Agent training completed in simulation
2. ✅ Have hardware infrastructure (10G NIC, servers)
3. ✅ Need production validation
4. ✅ Ready for actual deployment

**Do NOT** use real-time for:
- ❌ Training RL agents (too slow, use simulation)
- ❌ Algorithm development (use simulation)
- ❌ Hyperparameter tuning (use simulation)

### Development Workflow

**Recommended workflow**:
```
1. Develop in simulation (../simulation-mode/)
   - Train agents
   - Test algorithms
   - Validate performance

2. Deploy to real-time (./realtime-mode/)
   - Load pretrained models
   - Run in production
   - Monitor metrics
   
3. Iterate
   - If issues: Go back to simulation, retrain
   - If success: Monitor & maintain
```

---

## 🔗 Related Folders

- **Simulation Mode**: `../simulation-mode/` - Training & development
- **Data**: `../data/` - Trace files for simulation
- **Config**: `../config/` - Configuration files

---

## 📖 References

1. **VPP**: https://fd.io/docs/vpp/
2. **DPDK**: https://doc.dpdk.org/
3. **Alias Method**: Walker (1977) - O(1) weighted sampling
4. **Production RL**: Best practices for deploying RL in production

---

**Status**: 🚧 **30% Complete** (Skeleton + Documentation)  
**Pending**: VPP plugin full implementation, hardware validation  
**Timeline**: 2-3 weeks additional work  

**Prerequisites**: Complete simulation mode first (✅ Done)  
**Next Step**: Implement VPP C plugin components (shm_reader, shm_writer, CLI)
