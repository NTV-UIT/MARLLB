# MARLLB Implementation Guide - Hướng dẫn Triển khai

Đây là kho lưu trữ các bài toán implementation từng phần của hệ thống Multi-Agent Reinforcement Learning Load Balancer (MARLLB).

## Tổng quan Kiến trúc

MARLLB là hệ thống load balancer thông minh sử dụng Deep Reinforcement Learning để phân phối traffic động dựa trên tải thực tế của các application servers. Hệ thống tích hợp trực tiếp vào VPP (Vector Packet Processing) data plane để đạt hiệu năng cao.

### Kiến trúc Tổng thể

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Clients   │────────▶│ Edge Router  │────────▶│Load Balancer│
│  (Traffic   │         │              │         │  (VPP + RL) │
│  Replayer)  │         │              │         │             │
└─────────────┘         └──────────────┘         └──────┬──────┘
                                                          │
                                     ┌────────────────────┴────────────────────┐
                                     │                                         │
                                     ▼                                         ▼
                             ┌───────────────┐                        ┌───────────────┐
                             │Application    │                        │Application    │
                             │Server 1       │          ...           │Server N       │
                             │(Apache+MySQL) │                        │(Apache+MySQL) │
                             └───────────────┘                        └───────────────┘
```

## Các Bài Toán Implementation

Mỗi bài toán được tổ chức trong thư mục riêng với code và documentation đầy đủ:

### 1. Reservoir Sampling (`problem-01-reservoir-sampling/`)
**Mục tiêu**: Implement thuật toán reservoir sampling để thu thập và duy trì thống kê về flow completion time (FCT) và flow duration mà không cần lưu trữ toàn bộ flows.

**Kỹ thuật chính**:
- Random sampling với xác suất đồng đều
- Feature engineering (mean, 90th percentile, std, decay-weighted)
- Memory-efficient với fixed-size buffer (128 samples)

**Output**: Library Python/C có thể tái sử dụng cho VPP plugin

---

### 2. Shared Memory IPC (`problem-02-shared-memory-ipc/`)
**Mục tiêu**: Thiết kế và implement giao tiếp giữa VPP plugin (C) và RL agent (Python) qua shared memory.

**Kỹ thuật chính**:
- POSIX shared memory (`/dev/shm/`)
- Ring buffer cho observations
- Lock-free synchronization
- Memory layout design

**Output**: Python wrapper class và C header files

---

### 3. RL Environment Integration (`problem-03-rl-environment/`)
**Mục tiêu**: Xây dựng OpenAI Gym environment cho bài toán load balancing.

**Kỹ thuật chính**:
- State space design (server features, active server bitmap)
- Action space (discrete/continuous weights)
- Reward function (fairness metrics: Jain index, variance, max-min)
- Integration với shared memory

**Output**: LoadBalanceEnv class tương thích với Stable-Baselines3

---

### 4. Single-Agent SAC-GRU (`problem-04-sac-gru/`)
**Mục tiêu**: Implement thuật toán Soft Actor-Critic với GRU network cho temporal modeling.

**Kỹ thuật chính**:
- Policy network với GRU layers
- Twin Q-networks với GRU
- Automatic entropy tuning
- Episode-based replay buffer

**Output**: Trainable SAC agent cho single load balancer

---

### 5. Multi-Agent QMIX Coordination (`problem-05-qmix/`)
**Mục tiêu**: Implement QMIX algorithm cho multi-agent learning với decentralized execution.

**Kỹ thuật chính**:
- Individual agent networks
- Mixing network với monotonicity constraint
- TCP-based coordination protocol
- Centralized training, decentralized execution

**Output**: QMIX agent cho distributed load balancers

---

### 6. VPP Plugin Integration (`problem-06-vpp-plugin/`)
**Mục tiêu**: Tích hợp toàn bộ components vào VPP plugin để xử lý packet thực tế.

**Kỹ thuật chính**:
- VPP node implementation
- Hash-based session stickiness
- Statistics collection trong packet processing path
- Conditional compilation cho các load balancing methods

**Output**: Complete VPP plugin với RL-based load balancing

---

## Workflow Triển khai

### Giai đoạn 1: Foundation (Problems 1-2)
1. Implement reservoir sampling với unit tests
2. Thiết lập shared memory communication
3. Verify synchronization giữa Python và C

### Giai đoạn 2: RL Core (Problems 3-4)
4. Xây dựng Gym environment
5. Train SAC-GRU agent trong simulated environment
6. Evaluate với synthetic workload

### Giai đoạn 3: Multi-Agent (Problem 5)
7. Implement QMIX algorithm
8. Test coordination protocol với 2+ agents
9. Compare với independent learners

### Giai đoạn 4: Integration (Problem 6)
10. Tích hợp vào VPP plugin
11. Setup KVM testbed
12. Full system evaluation với real traces

---

## Yêu cầu Hệ thống

### Software
- Python 3.6+
- PyTorch 1.8+
- NumPy, SciPy
- OpenAI Gym
- GCC 7+ (cho C code)
- VPP 20.05+ (cho full integration)

### Hardware
- Minimum: 8 CPU cores, 16GB RAM (cho development)
- Recommended: 40+ CPU cores, 64GB RAM (cho testbed)
- Full cluster: 4 machines × 48 cores (cho paper reproduction)

---

## Cách Sử dụng Repo

### Development Mode
```bash
# Clone repo
cd /path/to/MARLLB/implementations

# Chuyển đến bài toán cụ thể
cd problem-01-reservoir-sampling

# Đọc README và làm theo hướng dẫn
cat README.md

# Run tests
python test_reservoir.py
```

### Testing Individual Components
Mỗi bài toán có test suite riêng để verify correctness trước khi integration.

### Integration Testing
Sau khi hoàn thành các bài toán riêng lẻ, chạy integration tests trong `problem-06-vpp-plugin/tests/`

---

## Documentation Structure

Mỗi thư mục bài toán chứa:
- `README.md`: Giải thích chi tiết bài toán, approach, và implementation
- `THEORY.md`: Background lý thuyết và toán học
- `src/`: Source code
- `tests/`: Unit tests và integration tests
- `examples/`: Code examples và usage demonstrations
- `docs/`: Additional documentation (nếu cần)

---

## Contributing

Khi implement mỗi bài toán:
1. Đọc kỹ THEORY.md để hiểu background
2. Follow coding style trong existing codebase
3. Viết tests trước khi implement (TDD)
4. Document code với comments chi tiết
5. Update README với usage examples

---

## References

- **Original Paper**: "Aquarius: A Multi-Agent Reinforcement Learning Load Balancer"
- **VPP Documentation**: https://fd.io/
- **QMIX Paper**: "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning"
- **SAC Paper**: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"

---

## License

Tuân theo LICENSE của MARLLB repository gốc.

---

## Contact

Nếu có câu hỏi về implementation, vui lòng tạo issue trong repository.
