# Problem 02: Shared Memory IPC

## Tổng quan

Shared Memory Inter-Process Communication (IPC) là cơ chế giao tiếp giữa VPP plugin (C) và RL agent (Python) trong MARLLB. Đây là component quan trọng vì:

1. **Performance Critical**: VPP xử lý millions packets/sec, không thể dùng sockets hoặc pipes (quá chậm)
2. **Zero-Copy**: Shared memory cho phép cả hai processes truy cập cùng memory region
3. **Low Latency**: Sub-microsecond communication overhead

## Bài toán

**Challenge**: Thiết kế memory layout và synchronization protocol để:
- VPP plugin ghi observations (server stats, reservoir samples)
- RL agent đọc observations và ghi actions (server weights)
- Đảm bảo consistency mà không cần locks (lock-free)

**Constraints**:
- Total shared memory: 1MB (limited resource)
- Update frequency: VPP writes mỗi 200ms, RL reads mỗi 250ms
- Max 64 servers
- Lock-free để không block VPP packet processing

## Architecture

### Memory Layout

```
/dev/shm/marllb_lb0:
├─ msg_out[4]           # Ring buffer: VPP → RL
│  ├─ sequence_id       # Monotonic counter
│  ├─ timestamp         # Microseconds
│  ├─ active_as_bitmap  # 64-bit bitmap
│  └─ as_stats[64]      # Per-server statistics
│     ├─ n_flow_on      # Active flows
│     └─ reservoir_features[10]  # FCT + duration features
│
├─ msg_in[1]            # Action: RL → VPP
│  ├─ sequence_id
│  ├─ weights[64]       # Server weights
│  └─ alias_table[64]   # For O(1) weighted sampling
│
└─ reservoir_data[64]   # Raw reservoir samples (optional)
   └─ samples[128 × 2]  # FCT and duration arrays
```

### Communication Flow

```
Time    VPP Plugin                  Shared Memory              RL Agent
────────────────────────────────────────────────────────────────────────
0.0s    Process packets
        Update reservoir
        
0.2s    Compute features
        Write msg_out[0] ────────▶  msg_out[0].seq=1
                                    msg_out[0].stats=...
        
0.25s                                                      Read msg_out[0]
                                                           Compute action
                                    msg_in[0].seq=1    ◀── Write msg_in[0]
                                    msg_in[0].weights=...

0.4s    Read msg_in[0] if new
        Update weights
        Write msg_out[1] ────────▶  msg_out[1].seq=2
        
0.5s                                                      Read msg_out[1]
                                                           ...
```

## Memory Layout Design

### Message Out (Observation): VPP → RL

```c
#define MAX_AS 64
#define RESERVOIR_CAPACITY 128
#define NUM_FEATURES 5
#define RING_BUFFER_SIZE 4

typedef struct {
    // Header
    uint64_t sequence_id;        // Monotonic counter
    uint64_t timestamp_us;       // Microseconds since epoch
    uint64_t active_as_bitmap;   // Bitmap of active servers
    uint32_t num_active_as;      // Number of active servers
    uint32_t reserved;           // Padding
    
    // Per-server statistics
    struct {
        uint32_t n_flow_on;      // Active flows
        float reservoir_features[10];  // 2 metrics × 5 features
    } as_stats[MAX_AS];
    
} msg_out_t;

// Ring buffer for observations
typedef struct {
    msg_out_t messages[RING_BUFFER_SIZE];
    uint64_t write_index;  // Atomic counter
} msg_out_ring_t;
```

**Size calculation**:
- Header: 8 + 8 + 8 + 4 + 4 = 32 bytes
- Per-server: 4 + 40 = 44 bytes
- Total per message: 32 + 64 × 44 = 2,848 bytes
- Ring buffer: 2,848 × 4 = **11,392 bytes** (~11KB)

### Message In (Action): RL → VPP

```c
typedef struct {
    // Header
    uint64_t sequence_id;        // Must match msg_out
    uint64_t timestamp_us;       // When action was computed
    uint32_t num_servers;        // Number of servers
    uint32_t reserved;
    
    // Server weights for load balancing
    float weights[MAX_AS];       // Weight per server
    
    // Alias method tables for O(1) sampling
    struct {
        float prob;              // Probability
        uint32_t alias;          // Alias index
    } alias_table[MAX_AS];
    
} msg_in_t;
```

**Size**: 16 + 256 + 512 = **784 bytes**

### Optional: Raw Reservoir Data

```c
typedef struct {
    float values[RESERVOIR_CAPACITY];
    uint64_t timestamps[RESERVOIR_CAPACITY];
} reservoir_raw_t;

typedef struct {
    reservoir_raw_t fct[MAX_AS];
    reservoir_raw_t flow_duration[MAX_AS];
} reservoir_data_t;
```

**Size**: (512 + 1024) × 2 × 64 = **196KB**

### Total Shared Memory

- msg_out ring: 11KB
- msg_in: 1KB
- reservoir_data: 196KB (optional)
- **Total: ~12KB** (or 208KB with raw reservoir data)

Well within 1MB limit! ✅

## Synchronization Protocol

### Lock-Free Communication

**Key insight**: Single writer, single reader per direction → no locks needed!

**VPP writes msg_out**:
```c
// Atomic increment write_index
uint64_t idx = __atomic_fetch_add(&ring->write_index, 1, __ATOMIC_SEQ_CST);
uint64_t slot = idx % RING_BUFFER_SIZE;

// Write data
msg_out_t *msg = &ring->messages[slot];
msg->sequence_id = idx + 1;
// ... fill other fields ...

// Memory barrier to ensure write completes
__atomic_thread_fence(__ATOMIC_RELEASE);
```

**RL reads msg_out**:
```python
# Read latest message
latest_idx = ring_buffer.write_index
slot = latest_idx % 4

msg = ring_buffer.messages[slot]

# Check sequence_id to detect missed messages
if msg.sequence_id > last_seen_seq + 1:
    print(f"Warning: Missed {msg.sequence_id - last_seen_seq - 1} observations")
```

**No locks, no blocking!** 🎉

### Sequence ID Protocol

Sequence IDs enable:
1. **Ordering**: Know which message is newest
2. **Loss detection**: Detect if messages were overwritten
3. **Synchronization**: Match msg_in to corresponding msg_out

```python
# RL agent
obs = read_msg_out()
action = compute_action(obs)
write_msg_in(action, sequence_id=obs.sequence_id)

# VPP plugin
action = read_msg_in()
if action.sequence_id > last_applied_seq:
    apply_action(action)
    last_applied_seq = action.sequence_id
```

## Implementation

### File Structure

```
problem-02-shared-memory-ipc/
├── README.md              # This file
├── THEORY.md             # IPC theory, POSIX shm, synchronization
├── Makefile              # Build system
├── src/
│   ├── shm_layout.py     # Memory layout definitions
│   ├── shm_writer.py     # Python writer (for testing)
│   ├── shm_reader.py     # Python reader
│   ├── shm_layout.h      # C memory layout
│   ├── shm_writer.c      # C writer (VPP side)
│   └── shm_reader.c      # C reader (for testing)
├── tests/
│   ├── test_shm_basic.py      # Basic read/write
│   ├── test_shm_concurrent.py # Concurrent access
│   └── test_shm_performance.py # Latency benchmark
└── examples/
    ├── producer_consumer.py   # Simple example
    └── vpp_rl_simulation.py   # Simulated VPP↔RL
```

## API Design

### Python API

```python
from shm_layout import SharedMemoryRegion

# Initialize (RL agent side)
shm = SharedMemoryRegion(name="marllb_lb0", create=False)

# Read observation
obs = shm.read_observation()
# Returns: {
#   'sequence_id': 42,
#   'timestamp': 1234567890.123,
#   'active_servers': [0, 1, 2, 3],
#   'server_stats': {
#       0: {'n_flow_on': 10, 'fct_mean': 0.1, ...},
#       1: {'n_flow_on': 15, 'fct_mean': 0.15, ...},
#       ...
#   }
# }

# Write action
shm.write_action(
    sequence_id=obs['sequence_id'],
    weights=[1.0, 1.5, 2.0, 1.2, ...],
    num_servers=4
)
```

### C API

```c
#include "shm_layout.h"

// Initialize (VPP plugin side)
shm_region_t *shm = shm_create("marllb_lb0", sizeof(shm_data_t));

// Write observation
msg_out_t obs;
obs.sequence_id = get_next_seq_id();
obs.timestamp_us = get_time_us();
obs.active_as_bitmap = compute_active_bitmap();

for (int i = 0; i < num_servers; i++) {
    obs.as_stats[i].n_flow_on = count_active_flows(i);
    compute_reservoir_features(i, obs.as_stats[i].reservoir_features);
}

shm_write_observation(shm, &obs);

// Read action
msg_in_t *action = shm_read_action(shm);
if (action && action->sequence_id > last_applied) {
    apply_weights(action->weights, action->num_servers);
    last_applied = action->sequence_id;
}
```

## Testing Strategy

### 1. Basic Functionality
- Create/attach shared memory
- Write/read simple data
- Verify data integrity

### 2. Concurrent Access
- VPP writes while RL reads
- No data corruption
- Sequence ID consistency

### 3. Performance
- Write latency: < 1μs
- Read latency: < 1μs
- End-to-end: < 10μs

### 4. Edge Cases
- Memory full (ring buffer wrap)
- Fast writer, slow reader
- Process crash and recovery

## Integration with MARLLB

### VPP Plugin Integration

```c
// In node.c packet processing loop
static uword lb_node_fn(vlib_main_t *vm, ...) {
    // ... process packets ...
    
    // Every 200ms, write observation
    if (should_update_shm()) {
        msg_out_t obs;
        prepare_observation(&obs);
        shm_write_observation(shm, &obs);
    }
    
    // Read action (if available)
    msg_in_t *action = shm_read_action(shm);
    if (action && is_new_action(action)) {
        update_lb_weights(action->weights);
    }
}
```

### RL Agent Integration

```python
# In env.py step() method
def step(self, action):
    # Write action to shared memory
    self.shm.write_action(
        sequence_id=self.last_obs['sequence_id'],
        weights=action_to_weights(action)
    )
    
    # Wait for next observation
    time.sleep(0.25)
    
    # Read new observation
    obs = self.shm.read_observation()
    reward = compute_reward(obs)
    
    return obs, reward, done, info
```

## Performance Considerations

### Memory Alignment
Align structures to cache lines (64 bytes) để avoid false sharing:

```c
typedef struct __attribute__((aligned(64))) {
    uint64_t sequence_id;
    // ... rest of struct ...
} msg_out_t;
```

### NUMA Awareness
Trên multi-socket systems, pin VPP và RL agent vào cùng NUMA node:

```bash
numactl --cpunodebind=0 --membind=0 vpp ...
numactl --cpunodebind=0 --membind=0 python rl_agent.py
```

### Hugepages
Optionally use hugepages cho shared memory để reduce TLB misses:

```bash
# Allocate 2MB hugepages
echo 128 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Use in code
shm_fd = shm_open(...);
ftruncate(shm_fd, size);
void *ptr = mmap(..., MAP_HUGETLB | MAP_HUGE_2MB, ...);
```

## Expected Results

### Latency Benchmarks

| Operation | Target | Typical |
|-----------|--------|---------|
| Write msg_out | < 1μs | ~0.3μs |
| Read msg_out | < 1μs | ~0.2μs |
| Write msg_in | < 1μs | ~0.3μs |
| Read msg_in | < 1μs | ~0.2μs |
| End-to-end | < 10μs | ~1μs |

### Throughput
- Updates per second: 5000 (200μs interval)
- Bandwidth: 5000 × 3KB = 15 MB/s
- Well within memory bandwidth (> 100 GB/s)

## Tài liệu Tham khảo

1. **POSIX Shared Memory**: `man shm_overview`
2. **Memory Barriers**: Linux kernel memory-barriers.txt
3. **Lock-Free Programming**: Herb Sutter's articles
4. **MARLLB Paper**: Section 3.3 "System Implementation"
5. **VPP Source**: `src/vpp/lb/shm.h` (original implementation)

## Next Steps

Sau khi hoàn thành Problem 02:
1. Use shared memory trong Problem 03 (RL Environment)
2. Integrate với Problem 01 (Reservoir Sampling) để transfer features
3. Test với real VPP plugin trong Problem 06
