# Problem 07: Deployment Guide

**Real-time VPP Deployment for Production Load Balancing**

---

## 📋 Overview

This guide covers end-to-end deployment of MARLLB in production:
1. Hardware requirements & setup
2. VPP installation & configuration
3. RL controller deployment
4. Monitoring & troubleshooting
5. Performance tuning

---

## 🖥️ Hardware Requirements

### Minimum Configuration

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **CPU** | 8+ cores (2.0+ GHz) | VPP is CPU-intensive |
| **RAM** | 16 GB | 8 GB for VPP, 8 GB for OS/Python |
| **NIC** | 10 Gbps (Intel X520/X710) | DPDK-compatible |
| **Storage** | 50 GB SSD | For logs & checkpoints |
| **OS** | Ubuntu 20.04/22.04 LTS | Debian-based preferred |

### Recommended Configuration

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **CPU** | 16+ cores (3.0+ GHz) | Better for high throughput |
| **RAM** | 32 GB | Allows larger buffer pools |
| **NIC** | 2× 10 Gbps (bonded) | Redundancy & higher capacity |
| **Storage** | 100 GB NVMe SSD | Fast I/O for logging |

### Network Topology

```
                    Internet
                        ↓
                  ┌─────────┐
                  │ Router  │
                  └─────────┘
                        ↓
              ┌──────────────────┐
              │  VPP Load Balancer│  (This machine)
              │  + RL Controller  │
              │  10.0.1.1         │
              └──────────────────┘
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │Backend 1│    │Backend 2│... │Backend N│
   │10.0.2.10│    │10.0.2.11│    │10.0.2.N │
   └─────────┘    └─────────┘    └─────────┘
```

---

## 🚀 Quick Start (Production Deployment)

### Step 1: Install VPP

```bash
# Add VPP repository
curl -s https://packagecloud.io/install/repositories/fdio/release/script.deb.sh | sudo bash

# Install VPP (version 23.06+)
sudo apt-get update
sudo apt-get install -y vpp vpp-plugin-core vpp-plugin-dpdk

# Verify installation
vpp -version
# Expected: vpp v23.06 or newer
```

### Step 2: Configure VPP

```bash
# Edit startup config
sudo vim /etc/vpp/startup.conf
```

Add/modify these sections:

```conf
unix {
  nodaemon
  log /var/log/vpp/vpp.log
  full-coredump
  cli-listen /run/vpp/cli.sock
  gid vpp
}

api-trace {
  on
}

dpdk {
  dev 0000:03:00.0    # Your 10G NIC PCIe address
  num-mbufs 65536
  no-multi-seg
  uio-driver vfio-pci
}

plugins {
  path /usr/lib/x86_64-linux-gnu/vpp_plugins
  plugin lb_rl_plugin.so { enable }
}

cpu {
  main-core 1
  corelist-workers 2-7
}

buffers {
  buffers-per-numa 128000
}
```

Find your NIC PCIe address:
```bash
lspci | grep Ethernet
# Example output: 03:00.0 Ethernet controller: Intel Corporation ...
```

### Step 3: Bind NIC to DPDK

```bash
# Install DPDK tools
sudo apt-get install -y dpdk dpdk-dev

# Bind NIC to vfio-pci driver
sudo dpdk-devbind.py --bind=vfio-pci 0000:03:00.0

# Verify binding
dpdk-devbind.py --status
```

### Step 4: Build & Install RL Plugin

```bash
cd implementations/problem-07-realtime-deployment/vpp-plugin

# Build plugin
mkdir build && cd build
cmake ..
make -j8

# Install to VPP plugins directory
sudo cp liblb_rl_plugin.so /usr/lib/x86_64-linux-gnu/vpp_plugins/

# Verify
ls -la /usr/lib/x86_64-linux-gnu/vpp_plugins/liblb_rl_plugin.so
```

### Step 5: Start VPP

```bash
# Start VPP service
sudo systemctl start vpp
sudo systemctl enable vpp  # Auto-start on boot

# Check status
sudo systemctl status vpp

# Connect to VPP CLI
sudo vppctl
```

### Step 6: Configure Load Balancer

Inside VPP CLI:

```bash
# Create shared memory region
lb rl create-shm /dev/shm/lb_rl_shm

# Configure backend servers (example: 4 servers)
lb rl add-server 10.0.2.10
lb rl add-server 10.0.2.11
lb rl add-server 10.0.2.12
lb rl add-server 10.0.2.13

# Set VIP (Virtual IP for load balancer)
lb rl set-vip 10.0.1.100 port 80

# Enable RL load balancing
lb rl enable

# Verify configuration
lb rl show config
lb rl show servers
```

### Step 7: Train RL Agent (Offline)

```bash
cd implementations/problem-06-vpp-integration

# Train QMIX agent
python src/training_pipeline.py \
    --agent-type qmix \
    --num-servers 16 \
    --num-agents 4 \
    --trace-type wiki \
    --episodes 10000 \
    --save-path checkpoints/qmix_prod.pt

# Wait for training to complete (~2-4 hours)
# Final model saved to: checkpoints/qmix_prod.pt
```

### Step 8: Start Python Controller

```bash
cd implementations/problem-07-realtime-deployment

# Start controller in background
./scripts/start_controller.sh \
    --agent qmix \
    --model ../problem-06-vpp-integration/checkpoints/qmix_prod.pt \
    --num-servers 4 \
    --servers "10.0.2.10 10.0.2.11 10.0.2.12 10.0.2.13" \
    --prometheus-port 9090

# Controller started successfully (PID: 12345)
# Monitor logs: tail -f realtime_controller.log
```

### Step 9: Verify System

```bash
# Terminal 1: Monitor VPP stats
sudo vppctl lb rl show stats

# Terminal 2: Monitor Python controller
tail -f realtime_controller.log

# Terminal 3: Check Prometheus metrics
curl http://localhost:9090/metrics | grep lb_

# Terminal 4: Send test traffic
curl http://10.0.1.100/
```

---

## 📊 Monitoring & Dashboards

### Prometheus Metrics

Controller exports these metrics on port 9090:

```
# Latency
lb_latency_avg_ms         # Average latency
lb_latency_p95_ms         # P95 latency
lb_latency_p99_ms         # P99 latency

# Fairness
lb_fairness_jain          # Jain's fairness index (0-1)
lb_fairness_cv            # Coefficient of variation

# Throughput
lb_throughput_rps         # Requests per second
lb_total_requests         # Total requests (counter)

# Agent performance
lb_agent_inference_seconds  # Inference time histogram
```

### Grafana Dashboard

Create dashboard at `http://localhost:3000`:

```json
{
  "dashboard": {
    "title": "MARLLB Production",
    "panels": [
      {
        "title": "Latency",
        "targets": [
          "lb_latency_avg_ms",
          "lb_latency_p95_ms"
        ]
      },
      {
        "title": "Fairness",
        "targets": ["lb_fairness_jain"]
      },
      {
        "title": "Throughput",
        "targets": ["lb_throughput_rps"]
      }
    ]
  }
}
```

---

## 🔧 Troubleshooting

### Issue 1: VPP Fails to Start

**Symptoms**: `systemctl status vpp` shows failed

**Solutions**:
```bash
# Check logs
sudo journalctl -u vpp -n 50

# Common issues:
# 1. NIC not bound to DPDK
sudo dpdk-devbind.py --bind=vfio-pci 0000:03:00.0

# 2. Insufficient hugepages
echo 1024 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# 3. Permission issues
sudo usermod -aG vpp $USER
```

### Issue 2: Shared Memory Not Found

**Symptoms**: Controller logs "Failed to attach to SHM"

**Solutions**:
```bash
# Check if SHM exists
ls -la /dev/shm/lb_rl_shm

# Create manually via VPP CLI
sudo vppctl lb rl create-shm /dev/shm/lb_rl_shm

# Check permissions
sudo chmod 666 /dev/shm/lb_rl_shm
```

### Issue 3: Low Throughput

**Symptoms**: Throughput < 1 Gbps

**Solutions**:
```bash
# 1. Check CPU cores
sudo vppctl show threads
# Increase worker threads in startup.conf: corelist-workers 2-15

# 2. Check buffer allocation
sudo vppctl show buffers
# Increase in startup.conf: buffers-per-numa 256000

# 3. Check NIC stats
sudo vppctl show hardware-interfaces
# Look for drops, errors
```

### Issue 4: Agent Inference Too Slow

**Symptoms**: Agent inference > 100ms

**Solutions**:
```bash
# 1. Use GPU if available
export CUDA_VISIBLE_DEVICES=0
python src/realtime_controller.py --device cuda ...

# 2. Increase update interval (less frequent inference)
./scripts/start_controller.sh --update-interval 0.5 ...

# 3. Use smaller model
# Train with smaller hidden dimensions (gru_dim=64 instead of 128)
```

---

## 🎯 Performance Tuning

### VPP Optimization

```conf
# startup.conf optimizations
cpu {
  main-core 0           # Dedicate core 0 to main thread
  corelist-workers 1-7  # Use 7 cores for packet processing
  skip-cores 8          # Reserve cores 8+ for Python
}

buffers {
  buffers-per-numa 256000  # Increase buffer pool (for 10 Gbps)
}

dpdk {
  dev 0000:03:00.0 {
    num-rx-queues 4     # Multi-queue for better scaling
    num-tx-queues 4
  }
}
```

### Python Controller Optimization

```bash
# Use PyTorch JIT compilation
python src/realtime_controller.py --jit-compile ...

# Pin Python to specific cores (avoid interference with VPP)
taskset -c 8-11 python src/realtime_controller.py ...

# Increase update interval (trade latency for stability)
--update-interval 0.3  # Instead of 0.2
```

### System-level Tuning

```bash
# 1. Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# 2. Disable IRQ balancing
sudo systemctl stop irqbalance

# 3. Set NIC IRQ affinity (core 8-11 for Python)
sudo set_irq_affinity.sh 8-11 eth0

# 4. Increase network buffer sizes
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

---

## 📈 Performance Benchmarks

### Expected Performance (16-core server, 10G NIC)

| Metric | Target | Baseline | RL-based |
|--------|--------|----------|----------|
| **Throughput** | 10 Gbps | 9.8 Gbps | 9.9 Gbps |
| **Avg Latency** | <10 ms | 12.3 ms | **8.2 ms** ✓ |
| **P95 Latency** | <20 ms | 28.5 ms | **18.1 ms** ✓ |
| **Fairness (Jain)** | >0.95 | 0.85 | **0.96** ✓ |
| **CPU Usage** | <80% | 65% | 72% |

### Benchmarking Tools

```bash
# 1. Apache Bench (HTTP)
ab -n 1000000 -c 100 http://10.0.1.100/

# 2. TRex (high-rate traffic generator)
./t-rex-64 -f cfg/lb_test.yaml --duration 60

# 3. iperf3 (TCP throughput)
iperf3 -c 10.0.1.100 -t 60 -P 10

# 4. wrk (HTTP benchmark)
wrk -t4 -c100 -d60s http://10.0.1.100/
```

---

## 🔒 Security Considerations

1. **Shared Memory Permissions**
   ```bash
   # Restrict SHM access
   sudo chown vpp:vpp /dev/shm/lb_rl_shm
   sudo chmod 660 /dev/shm/lb_rl_shm
   ```

2. **Network Isolation**
   - Run VPP in dedicated VLAN
   - Firewall rules for management interface only

3. **Model Security**
   - Store model checkpoints securely
   - Use encrypted storage for sensitive models
   - Validate model checksums before loading

---

## 📚 References

- VPP Documentation: https://fd.io/docs/vpp/
- DPDK Setup: https://doc.dpdk.org/guides/linux_gsg/
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/

---

**Next Steps**:
1. Review this guide thoroughly
2. Prepare hardware according to requirements
3. Follow deployment steps sequentially
4. Test with low traffic first, then scale up
5. Monitor metrics and tune as needed

**Support**:
- Issues: File bug reports in GitHub
- Questions: Check FAQ in main README.md
