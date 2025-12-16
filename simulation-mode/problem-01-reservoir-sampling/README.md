# Problem 01: Reservoir Sampling

## Tổng quan

Reservoir Sampling là kỹ thuật quan trọng trong MARLLB để thu thập và duy trì thống kê về network flows mà không cần lưu trữ toàn bộ flows trong memory. VPP plugin xử lý hàng triệu packets/giây, không thể lưu trữ thông tin tất cả flows, nên sử dụng reservoir sampling để maintain một sample ngẫu nhiên có kích thước cố định.

## Bài toán

**Input**: Stream vô hạn các flows với metrics (Flow Completion Time, Flow Duration)

**Output**: 
- Random sample gồm N=128 flows được chọn đồng đều
- Các features thống kê: mean, 90th percentile, std, decay-weighted variants

**Constraints**:
- Memory cố định: 128 slots per server per metric
- Phải update trong O(1) time (không block packet processing)
- Đảm bảo uniform random sampling

## Lý thuyết

### Algorithm R - Reservoir Sampling

Thuật toán cơ bản cho reservoir sampling (từ paper của Jeffrey Vitter):

```
Initialization:
    reservoir[0..k-1] = stream[0..k-1]  // k = 128

For i = k to n:
    j = random(0, i)  // Random integer from 0 to i
    if j < k:
        reservoir[j] = stream[i]
```

**Proof of Correctness**: Mỗi phần tử trong stream có xác suất k/n được chọn vào reservoir sau n elements.

### Feature Engineering

Từ 128 samples, tính các features cho RL agent:

1. **Basic Statistics**:
   - `mean`: Trung bình
   - `p90`: 90th percentile (quan trọng hơn max vì robust với outliers)
   - `std`: Standard deviation

2. **Decay-Weighted Statistics**:
   - `mean_decay`: Weighted average với decay factor α=0.9
   - `p90_decay`: 90th percentile của decay-weighted values
   
   Decay weight: $w_i = \alpha^{t_{current} - t_i}$
   
   Mục đích: Ưu tiên samples gần đây hơn để phản ánh workload changes

### Use Case trong MARLLB

**Metrics được sample**:
- **FCT (Flow Completion Time)**: Thời gian từ SYN đến FIN/RST
- **Flow Duration**: Thời gian từ first packet đến last packet

**Features per server**: 
- 1 counter feature: `n_flow_on` (active flows)
- 2 reservoir metrics × 5 features each = 10 features
- **Total**: 11 dimensions per server

## Implementation

### Structure

```
problem-01-reservoir-sampling/
├── README.md              # This file
├── THEORY.md             # Mathematical background
├── src/
│   ├── reservoir.py      # Python implementation
│   ├── reservoir.c       # C implementation (for VPP)
│   ├── reservoir.h       # C header
│   └── features.py       # Feature engineering
├── tests/
│   ├── test_reservoir.py # Unit tests
│   ├── test_features.py  # Feature tests
│   └── test_uniformity.py # Statistical tests
└── examples/
    ├── basic_usage.py    # Simple example
    └── benchmark.py      # Performance benchmark
```

### API Design

#### Python API

```python
from reservoir import ReservoirSampler

# Initialize
sampler = ReservoirSampler(capacity=128)

# Add samples
for flow in flows:
    sampler.add(flow.fct, timestamp=flow.end_time)

# Get statistics
stats = sampler.get_features(decay_factor=0.9)
# Returns: {
#   'mean': ..., 
#   'p90': ..., 
#   'std': ...,
#   'mean_decay': ..., 
#   'p90_decay': ...
# }
```

#### C API

```c
#include "reservoir.h"

// Initialize
reservoir_t reservoir;
reservoir_init(&reservoir, 128);

// Add sample
float fct = 0.123;  // seconds
uint64_t timestamp = get_timestamp();
reservoir_add(&reservoir, fct, timestamp);

// Get statistics (called by RL agent via shared memory)
reservoir_stats_t stats;
reservoir_compute_stats(&reservoir, &stats, 0.9);
```

## Testing Strategy

### 1. Correctness Tests
- Verify uniform sampling distribution
- Check boundary conditions (empty, full, overflow)
- Validate feature computations

### 2. Statistical Tests
- Chi-square test for uniformity
- Compare with ground truth on known distributions
- Test decay weighting correctness

### 3. Performance Tests
- Benchmark add() operation (should be < 1μs)
- Memory footprint verification
- Cache performance (important for VPP)

## Expected Results

### Uniformity Test
Sau 10,000 samples vào reservoir size 128, mỗi position nên có ~78 updates (10000/128) ± 10%.

### Performance Target
- **Add operation**: < 1 microsecond
- **Feature computation**: < 100 microseconds (chỉ call mỗi 250ms)
- **Memory**: 128 × (4 bytes float + 8 bytes timestamp) = 1.5KB per metric

## Integration với MARLLB

Reservoir sampling được sử dụng trong:

1. **VPP Plugin** (`src/vpp/lb/node.c`):
   ```c
   // Khi flow kết thúc (RST packet)
   if (tcp_flag == RSTACK) {
       float fct = current_time - flow_start_time;
       reservoir_add(&reservoir_fct[server_id], fct, current_time);
   }
   ```

2. **Shared Memory** (`src/vpp/lb/shm.h`):
   ```c
   typedef struct {
       tv_pair_f_t fct[128];           // FCT reservoir
       tv_pair_f_t flow_duration[128]; // Duration reservoir
   } reservoir_as_t;
   ```

3. **RL Agent** (`src/lb/shm_proxy.py`):
   ```python
   # Read from shared memory
   reservoir_data = read_reservoir_from_shm(server_id)
   features = compute_features(reservoir_data, decay=0.9)
   state[server_id] = features
   ```

## Tài liệu Tham khảo

1. Vitter, J. S. (1985). "Random sampling with a reservoir". ACM Transactions on Mathematical Software.
2. MARLLB Paper: Section 3.2 "State Space Design"
3. VPP Source: `src/vpp/lb/lbhash.h` (original implementation)

## Next Steps

Sau khi hoàn thành Problem 01:
1. Integrate với Problem 02 (Shared Memory) để transfer reservoir data
2. Use features trong Problem 03 (RL Environment) làm state input
3. Test với real traffic traces từ `data/trace/wiki/`
