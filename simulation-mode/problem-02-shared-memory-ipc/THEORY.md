# Shared Memory IPC - Lý Thuyết Chi Tiết

## 1. Inter-Process Communication (IPC) Mechanisms

### 1.1 Các phương pháp IPC phổ biến

| Method | Bandwidth | Latency | Use Case |
|--------|-----------|---------|----------|
| **Pipes** | Low | ~10μs | Simple parent-child |
| **Sockets** | Medium | ~5μs | Network-style |
| **Message Queues** | Medium | ~3μs | Async messaging |
| **Shared Memory** | High | ~0.1μs | High-performance |

**Tại sao chọn Shared Memory?**

1. **Bandwidth**: Giới hạn chỉ bởi memory bandwidth (~100 GB/s), không bởi kernel syscalls
2. **Latency**: Sub-microsecond, chỉ cần memory access
3. **Zero-copy**: Không cần copy data giữa processes

### 1.2 POSIX Shared Memory

POSIX cung cấp API chuẩn cho shared memory:

```c
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

// Create shared memory object
int shm_fd = shm_open("/myshm", O_CREAT | O_RDWR, 0666);

// Set size
ftruncate(shm_fd, SIZE);

// Map to address space
void *ptr = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
```

**Lifetime**: Shared memory objects persist sau khi process exits (until `shm_unlink()`)

**Location**: Linux stores trong `/dev/shm/` (tmpfs - RAM-based filesystem)

## 2. Memory Consistency và Synchronization

### 2.1 Memory Ordering Problem

Modern CPUs reorder memory operations để tối ưu performance:

```c
// Thread 1
x = 1;        // Write 1
flag = true;  // Write 2

// Thread 2
if (flag) {   // Read 2
    print(x); // Read 1 - Might see x=0!
}
```

**Problem**: CPU có thể reorder "Write 2" trước "Write 1", Thread 2 thấy `flag=true` nhưng `x=0`.

### 2.2 Memory Barriers

Memory barriers đảm bảo ordering:

```c
// C11 atomics
#include <stdatomic.h>

atomic_int x = 0;
atomic_bool flag = false;

// Thread 1
atomic_store_explicit(&x, 1, memory_order_release);
atomic_store_explicit(&flag, true, memory_order_release);

// Thread 2
if (atomic_load_explicit(&flag, memory_order_acquire)) {
    int val = atomic_load_explicit(&x, memory_order_acquire);
    // Guaranteed: val == 1
}
```

**Memory order types**:

- `memory_order_relaxed`: No ordering guarantees (fastest)
- `memory_order_acquire`: Loads after this không thể reorder trước
- `memory_order_release`: Stores before this không thể reorder sau
- `memory_order_seq_cst`: Sequential consistency (strongest, slowest)

### 2.3 Lock-Free Programming

**Goal**: Avoid locks để không block critical threads (như VPP packet processing)

**Single-Writer, Single-Reader** (SWSR) queue:
- Simplest lock-free structure
- Writer updates write_index, reader updates read_index
- No conflicts if indices don't overlap

```c
typedef struct {
    uint64_t write_index;  // Written by producer only
    uint64_t read_index;   // Written by consumer only
    data_t buffer[SIZE];
} ring_buffer_t;

// Producer
void enqueue(ring_buffer_t *rb, data_t item) {
    uint64_t idx = rb->write_index;
    rb->buffer[idx % SIZE] = item;
    __atomic_store_n(&rb->write_index, idx + 1, __ATOMIC_RELEASE);
}

// Consumer
bool dequeue(ring_buffer_t *rb, data_t *item) {
    uint64_t read_idx = rb->read_index;
    uint64_t write_idx = __atomic_load_n(&rb->write_index, __ATOMIC_ACQUIRE);
    
    if (read_idx == write_idx) return false; // Empty
    
    *item = rb->buffer[read_idx % SIZE];
    __atomic_store_n(&rb->read_index, read_idx + 1, __ATOMIC_RELEASE);
    return true;
}
```

## 3. Ring Buffer Design

### 3.1 Tại sao dùng Ring Buffer?

**Alternatives**:
- **Single buffer**: Overwrite problem nếu reader chậm
- **Unbounded queue**: Memory growth

**Ring buffer advantages**:
- Fixed size (predictable memory)
- Overwrite oldest data nếu writer quá nhanh
- Lock-free với SWSR pattern

### 3.2 Size Selection

```
Ring size = ceil(max_write_rate / min_read_rate)
```

MARLLB:
- Write rate: 5 Hz (200ms interval)
- Read rate: 4 Hz (250ms interval)
- Ratio: 5/4 = 1.25
- Add safety margin: 4× = **5 slots**

**Actual**: MARLLB dùng 4 slots (acceptable với ~20% margin)

### 3.3 Wrap-Around Handling

**Naïve approach** (WRONG):
```c
idx = (idx + 1) % SIZE;  // Modulo is expensive (div instruction)
```

**Optimized approach** (if SIZE is power of 2):
```c
idx = (idx + 1) & (SIZE - 1);  // Bitwise AND is fast
```

**MARLLB approach** (SIZE=4):
```c
slot = sequence_id % 4;  // Use sequence_id directly
```

## 4. Sequence Numbers

### 4.1 Purpose

Sequence numbers solve multiple problems:

1. **Ordering**: Which message is newest?
2. **Loss detection**: Did we miss messages?
3. **Synchronization**: Match actions to observations

### 4.2 Generation

```c
static uint64_t next_seq_id = 0;

uint64_t get_next_seq_id() {
    return __atomic_fetch_add(&next_seq_id, 1, __ATOMIC_SEQ_CST);
}
```

**Properties**:
- Monotonic increasing
- Never repeats (64-bit → won't overflow in lifetime of universe)
- Thread-safe với atomic operations

### 4.3 Comparison Protocol

**Writer**:
```c
msg->sequence_id = get_next_seq_id();
```

**Reader**:
```c
if (msg->sequence_id > last_seen_seq) {
    // New message
    if (msg->sequence_id > last_seen_seq + 1) {
        printf("Missed %llu messages\n", 
               msg->sequence_id - last_seen_seq - 1);
    }
    process_message(msg);
    last_seen_seq = msg->sequence_id;
}
```

## 5. Data Layout Optimization

### 5.1 Struct Alignment

**Bad layout** (compiler adds padding):
```c
struct bad {
    char a;     // 1 byte
    // 3 bytes padding
    int b;      // 4 bytes
    char c;     // 1 byte
    // 7 bytes padding
    double d;   // 8 bytes
}; // Total: 24 bytes (58% wasted!)
```

**Good layout** (manual ordering):
```c
struct good {
    double d;   // 8 bytes
    int b;      // 4 bytes
    char a;     // 1 byte
    char c;     // 1 byte
    // 2 bytes padding
}; // Total: 16 bytes (12.5% wasted)
```

**Best practice**: Order fields từ lớn đến nhỏ

### 5.2 Cache Line Alignment

Cache line size = 64 bytes trên x86-64

**False sharing problem**:
```c
struct shared {
    int counter1;  // Updated by CPU 0
    int counter2;  // Updated by CPU 1
}; // Both in same cache line → ping-pong!
```

**Solution**: Align to cache lines
```c
struct shared {
    int counter1 __attribute__((aligned(64)));
    int counter2 __attribute__((aligned(64)));
};
```

### 5.3 Struct of Arrays (SoA) vs Array of Structs (AoS)

**AoS** (traditional):
```c
struct point { float x, y, z; };
struct point points[1000];

// Access: points[i].x
// Problem: Loading x also loads unused y, z
```

**SoA** (better for SIMD):
```c
struct points {
    float x[1000];
    float y[1000];
    float z[1000];
};

// Access: points.x[i]
// Benefit: Can load 4 x values in one cache line
```

**MARLLB uses hybrid**: AoS per server (small), SoA for features (vectorizable)

## 6. Python ↔ C Interop

### 6.1 Memory Mapping in Python

```python
import mmap
import os

# Open shared memory
fd = os.open('/dev/shm/myshm', os.O_RDWR)

# Memory map
mm = mmap.mmap(fd, length=SIZE, access=mmap.ACCESS_WRITE)

# Read/write
mm.seek(0)
data = mm.read(100)
mm.seek(0)
mm.write(b'hello')

mm.close()
os.close(fd)
```

### 6.2 Struct Packing

Python's `struct` module para serialize/deserialize:

```python
import struct

# Pack data
data = struct.pack('=Qdd', sequence_id, timestamp, value)
# '=' = native byte order
# 'Q' = unsigned long long (8 bytes)
# 'd' = double (8 bytes)

# Unpack
seq, ts, val = struct.unpack('=Qdd', data)
```

**Format characters**:
- `c`: char (1 byte)
- `b`/`B`: signed/unsigned char
- `h`/`H`: short (2 bytes)
- `i`/`I`: int (4 bytes)
- `q`/`Q`: long long (8 bytes)
- `f`: float (4 bytes)
- `d`: double (8 bytes)

### 6.3 NumPy Memory Views

Efficient access với NumPy:

```python
import numpy as np

# Map shared memory as NumPy array
arr = np.ndarray(shape=(1000,), dtype=np.float32, buffer=mm)

# Now can use NumPy operations
mean = arr.mean()
arr[:] = arr * 2  # In-place modification
```

## 7. Error Handling và Recovery

### 7.1 Detecting Stale Data

**Problem**: Process crashes, shared memory contains garbage

**Solution**: Heartbeat timestamps

```c
typedef struct {
    uint64_t last_write_time;
    // ... data ...
} msg_t;

// Writer
msg.last_write_time = get_time_us();

// Reader
uint64_t now = get_time_us();
if (now - msg.last_write_time > TIMEOUT_US) {
    // Data is stale
}
```

### 7.2 Process Crash Recovery

**Problem**: Writer crashes mid-write, partial data in memory

**Solution 1**: Double buffering
```c
typedef struct {
    msg_t buffer[2];
    atomic_int active_buffer;  // 0 or 1
} double_buffer_t;

// Writer
int next_buf = !active_buffer;
write_to_buffer(&buffer[next_buf]);
__atomic_store(&active_buffer, next_buf, __ATOMIC_RELEASE);

// Reader
int buf_idx = __atomic_load(&active_buffer, __ATOMIC_ACQUIRE);
read_from_buffer(&buffer[buf_idx]);
```

**Solution 2**: Sequence ID validation
```c
// Reader
if (msg->sequence_id != expected_seq) {
    // Corrupted or missed message
    skip_or_request_resend();
}
```

### 7.3 Memory Leak Prevention

```python
import atexit

class SharedMemory:
    def __init__(self, name):
        self.name = name
        self.mm = open_shm(name)
        atexit.register(self.cleanup)
    
    def cleanup(self):
        if self.mm:
            self.mm.close()
        # Don't unlink - other process might still use it
```

## 8. Performance Analysis

### 8.1 Latency Breakdown

**Write operation**:
```
1. Compute slot index:           ~1 ns
2. Write header fields:           ~10 ns
3. Write data (2KB):              ~100 ns
4. Memory barrier:                ~10 ns
5. Update sequence ID:            ~10 ns
────────────────────────────────────────
Total:                            ~130 ns
```

**Read operation**:
```
1. Load sequence ID:              ~10 ns (L1 cache hit)
2. Check if new:                  ~1 ns
3. Read header:                   ~10 ns
4. Read data (2KB):               ~100 ns
────────────────────────────────────────
Total:                            ~120 ns
```

### 8.2 Cache Effects

**L1 cache**: 32-64 KB, ~4 cycles (~1 ns)
**L2 cache**: 256-512 KB, ~12 cycles (~3 ns)
**L3 cache**: 8-32 MB, ~40 cycles (~10 ns)
**RAM**: 100+ cycles (~30 ns)

**MARLLB message**: 2.8 KB fits completely in L1 cache!

### 8.3 NUMA Effects

**Non-Uniform Memory Access**:
- Local memory: ~30 ns latency
- Remote memory: ~100 ns latency (3× slower!)

**Mitigation**:
```bash
# Pin both processes to same NUMA node
numactl --cpunodebind=0 --membind=0 ./vpp_process
numactl --cpunodebind=0 --membind=0 ./rl_process
```

## 9. Advanced Topics

### 9.1 Transactional Memory

Hardware Transactional Memory (HTM) - Intel TSX:

```c
#include <immintrin.h>

unsigned int status;
if ((status = _xbegin()) == _XBEGIN_STARTED) {
    // Transaction
    shared_var++;
    _xend();
} else {
    // Fallback
    pthread_mutex_lock(&lock);
    shared_var++;
    pthread_mutex_unlock(&lock);
}
```

**Limitation**: Không widely available, có thể abort

### 9.2 RDMA (Remote DMA)

For distributed shared memory:

```c
#include <infiniband/verbs.h>

// Zero-copy network transfer
ibv_post_send(qp, &send_wr, NULL);
```

**Use case**: Multi-machine MARLLB setup

### 9.3 Persistent Memory

Intel Optane PMem - shared memory survives reboot:

```c
#include <libpmem.h>

void *pmem_addr = pmem_map_file("/mnt/pmem/myfile", size, ...);
pmem_memcpy_persist(pmem_addr, data, len);
```

## 10. Testing Strategies

### 10.1 Unit Tests

```python
def test_write_read():
    shm = SharedMemory.create("test", size=1024)
    
    # Write
    shm.write(offset=0, data=b"hello")
    
    # Read
    data = shm.read(offset=0, size=5)
    assert data == b"hello"
    
    shm.close()
```

### 10.2 Stress Tests

```python
def test_concurrent_access():
    import multiprocessing
    
    def writer(shm_name):
        shm = SharedMemory.attach(shm_name)
        for i in range(10000):
            shm.write_message(i)
    
    def reader(shm_name):
        shm = SharedMemory.attach(shm_name)
        for i in range(10000):
            msg = shm.read_message()
            assert msg.sequence_id > prev_seq
    
    p1 = multiprocessing.Process(target=writer, args=("test",))
    p2 = multiprocessing.Process(target=reader, args=("test",))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
```

### 10.3 Benchmarking

```python
import time

def benchmark_latency():
    shm = SharedMemory.create("bench", size=4096)
    
    # Warm-up
    for _ in range(1000):
        shm.write_message(...)
        shm.read_message()
    
    # Measure
    start = time.perf_counter()
    for _ in range(100000):
        shm.write_message(...)
        shm.read_message()
    end = time.perf_counter()
    
    latency = (end - start) / 100000 * 1e6  # microseconds
    print(f"Round-trip latency: {latency:.2f} μs")
```

## 11. Tài liệu Tham khảo

### Papers

1. **Lamport, L.** (1979). "How to Make a Multiprocessor Computer That Correctly Executes Multiprocess Programs". *IEEE Transactions on Computers*.
   - Sequential consistency definition

2. **Herlihy, M. & Shavit, N.** (2008). "The Art of Multiprocessor Programming".
   - Comprehensive lock-free programming guide

3. **McKenney, P.** (2017). "Is Parallel Programming Hard, And, If So, What Can You Do About It?"
   - Linux kernel perspective

### Documentation

4. **POSIX.1-2017**: `shm_overview(7)`, `mmap(2)`, `shm_open(3)`
5. **Linux kernel**: `Documentation/memory-barriers.txt`
6. **Intel**: "Intel® 64 and IA-32 Architectures Software Developer's Manual"
7. **C11 Standard**: ISO/IEC 9899:2011 - Section 7.17 (Atomics)

### Online Resources

8. **Preshing on Programming**: https://preshing.com/
   - Excellent lock-free programming tutorials
9. **1024cores**: http://www.1024cores.net/
   - Lock-free algorithms
10. **Linux kernel mailing list**: Discussion of memory ordering

## 12. Common Pitfalls

### 12.1 Forgot Memory Barriers

```c
// WRONG
shared_data = new_value;
ready_flag = true;  // Might reorder before write!

// CORRECT
shared_data = new_value;
__atomic_thread_fence(__ATOMIC_RELEASE);
ready_flag = true;
```

### 12.2 Integer Overflow

```c
// WRONG - Wraps at 2^32
uint32_t seq_id = 0;
seq_id++;  // Overflows after 4 billion

// CORRECT - Practically infinite
uint64_t seq_id = 0;
seq_id++;  // Won't overflow in lifetime
```

### 12.3 Struct Padding Assumption

```c
// WRONG - Assumes no padding
struct msg { char a; int b; };
memcpy(&msg, buffer, 5);  // Might miss padding bytes!

// CORRECT
#pragma pack(push, 1)
struct msg { char a; int b; };
#pragma pack(pop)
```

### 12.4 Forgetting to Unlink

```python
# WRONG - Shared memory persists forever
shm = SharedMemory.create("myshm", 1024)
# ... use ...
shm.close()  # Only closes handle, memory still exists!

# CORRECT
shm = SharedMemory.create("myshm", 1024)
try:
    # ... use ...
finally:
    shm.close()
    shm.unlink()  # Actually delete
```

---

**Key Takeaways**:

1. Shared memory = fastest IPC (~100ns latency)
2. Lock-free với SWSR pattern
3. Memory barriers critical cho correctness
4. Sequence IDs solve ordering và loss detection
5. Cache alignment matters cho performance
6. Test concurrent access thoroughly
