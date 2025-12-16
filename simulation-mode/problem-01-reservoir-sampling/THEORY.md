# Reservoir Sampling - Lý Thuyết Chi Tiết

## 1. Giới thiệu Vấn đề

### 1.1 Streaming Data Problem

Trong nhiều ứng dụng thực tế, ta phải xử lý luồng dữ liệu vô hạn (hoặc rất lớn) mà không thể lưu trữ toàn bộ trong memory:

- **Network monitoring**: Millions of flows per second
- **Log analysis**: Continuous event streams
- **Database sampling**: Large tables với millions of rows

**Yêu cầu**: Maintain một sample ngẫu nhiên có kích thước cố định k từ stream có n elements (n >> k và có thể không biết trước).

### 1.2 Tại sao không dùng Simple Random Sampling?

Simple random sampling yêu cầu:
1. Biết trước tổng số elements n
2. Có khả năng random access toàn bộ dataset

Cả hai điều này không khả thi với streaming data!

## 2. Algorithm R - Reservoir Sampling Cơ bản

### 2.1 Thuật toán

```
Algorithm R (Vitter, 1985):
Input: Stream S of unknown length, reservoir size k
Output: Reservoir R containing k uniformly random samples

1. Initialize: R[0..k-1] = S[0..k-1]
2. For i = k to n-1:
   a. j = random(0, i)  // Uniform random integer in [0, i]
   b. if j < k:
      R[j] = S[i]
```

### 2.2 Proof of Correctness

**Theorem**: Sau khi xử lý n elements, mỗi element có xác suất chính xác k/n được chọn vào reservoir.

**Proof by Induction**:

**Base case** (i = k):
- k elements đầu tiên đều trong reservoir → P = k/k = 1 ✓

**Inductive step**: Giả sử đúng cho i-1, chứng minh cho i.

Element thứ i (S[i-1]):
- Xác suất được chọn: P(j < k) = k/i
- ✓ Đúng

Element cũ S[m] (m < i-1):
- Đã trong reservoir với xác suất k/(i-1) (theo giả thiết quy nạp)
- Xác suất bị thay thế bởi S[i-1]:
  - P(j = position của S[m]) = 1/i
- Xác suất còn lại:
  ```
  P(S[m] in R after step i) = k/(i-1) × (1 - 1/i)
                             = k/(i-1) × (i-1)/i
                             = k/i
  ```
- ✓ Đúng

**Conclusion**: Thuật toán đảm bảo uniform sampling với xác suất k/n.

### 2.3 Time Complexity

- **Add operation**: O(1) amortized
  - Generate random number: O(1)
  - Conditional replacement: O(1)
  
- **Space**: O(k) - constant size reservoir

## 3. Feature Engineering

### 3.1 Basic Statistics

Từ reservoir R = {x₁, x₂, ..., xₖ}, tính các features:

#### Mean (Trung bình)
```
μ = (1/k) Σᵢ xᵢ
```

#### 90th Percentile
```
p₉₀ = value tại position ⌊0.9k⌋ trong sorted array
```

Tại sao dùng p90 thay vì max?
- **Robust against outliers**: 1 flow bị timeout không ảnh hưởng toàn bộ metric
- **Reflects typical worst-case**: Capture "tail latency" mà không bị skew bởi extreme values

#### Standard Deviation
```
σ = √[(1/k) Σᵢ (xᵢ - μ)²]
```

### 3.2 Decay-Weighted Statistics

**Motivation**: Workload thay đổi theo thời gian. Samples gần đây nên có trọng số cao hơn.

#### Exponential Decay Weight
```
wᵢ = α^(t_current - tᵢ)
```
Trong đó:
- α = 0.9 (decay factor)
- tᵢ = timestamp của sample i
- t_current = thời điểm hiện tại

#### Weighted Mean
```
μ_decay = (Σᵢ wᵢxᵢ) / (Σᵢ wᵢ)
```

#### Weighted 90th Percentile
Phức tạp hơn: cần sort và accumulate weights.

```python
def weighted_percentile(values, weights, percentile):
    sorted_idx = np.argsort(values)
    sorted_weights = weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    cutoff = percentile * cumsum[-1]
    idx = np.searchsorted(cumsum, cutoff)
    return values[sorted_idx[idx]]
```

### 3.3 Tại sao cần 5 features cho mỗi metric?

| Feature | Meaning | Use in RL |
|---------|---------|-----------|
| `mean` | Average load | Overall server utilization |
| `p90` | Tail latency | Worst-case performance |
| `std` | Variability | Stability/predictability |
| `mean_decay` | Recent average | Detect load changes |
| `p90_decay` | Recent tail | Detect performance degradation |

## 4. Advanced Topics

### 4.1 Algorithm L - Optimized Reservoir Sampling

Vitter cũng đề xuất Algorithm L với time complexity tốt hơn khi k << n:

**Idea**: Thay vì generate random number cho mỗi element, skip một số elements.

```
Algorithm L:
1. w = exp(ln(random())/k)
2. skip = floor(ln(random())/ln(1-w))
3. Process next element at position i + skip
```

**Advantage**: Expected number of random number generations: O(k(1 + log(n/k)))

**Disadvantage**: Phức tạp hơn, ít cache-friendly hơn cho k lớn (128).

### 4.2 Weighted Reservoir Sampling

Nếu mỗi element có weight wᵢ (không phải tất cả đều equal importance):

```
Algorithm A-Res (Efraimidis & Spirakis):
1. For each element i, compute key = random()^(1/wᵢ)
2. Maintain heap of k elements với largest keys
```

**Use case trong MARLLB**: Có thể dùng để ưu tiên large flows hơn small flows.

### 4.3 Distributed Reservoir Sampling

Khi có multiple load balancers, làm sao merge reservoirs?

**Challenge**: Không thể simply concatenate và re-sample (mất tính uniform).

**Solution**: 
1. Each LB maintains local reservoir với size k
2. Central aggregator:
   - Collect n total samples từ m LBs
   - Apply Algorithm R với k' samples
   
**Note**: MARLLB không cần global reservoir, mỗi LB chỉ cần local view.

## 5. Implementation Considerations

### 5.1 Random Number Generation

**Yêu cầu**:
- Fast (< 100ns per call)
- Good randomness (uniform distribution)
- Thread-safe (nếu multi-threaded)

**Options**:

1. **`rand()` (C stdlib)**: 
   - ❌ Không thread-safe
   - ❌ Poor quality (LCG)
   - ✓ Fast

2. **`random()` (POSIX)**:
   - ✓ Better quality
   - ❌ Slow
   
3. **`xorshift128+`**:
   - ✓ Excellent speed
   - ✓ Good quality
   - ✓ Simple implementation

```c
uint64_t xorshift128plus(uint64_t s[2]) {
    uint64_t x = s[0];
    uint64_t const y = s[1];
    s[0] = y;
    x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s[1] + y;
}
```

### 5.2 Memory Layout

**Cache efficiency**: Quan trọng trong VPP (tight loop với high packet rate).

```c
// BAD: Array of structs (AoS)
struct sample {
    float value;
    uint64_t timestamp;
};
struct sample reservoir[128];  // 12 bytes × 128 = 1.5KB

// GOOD: Struct of arrays (SoA)
struct reservoir {
    float values[128];       // 512 bytes (8 cache lines)
    uint64_t timestamps[128]; // 1KB (16 cache lines)
};
```

**Why SoA better?**
- Feature computation chỉ cần access values array
- Better prefetching
- SIMD-friendly

### 5.3 Percentile Computation

**Naive approach**: Full sort O(k log k)

```python
def percentile_naive(values, p):
    sorted_values = sorted(values)
    return sorted_values[int(p * len(values))]
```

**Optimized approach**: Quickselect O(k) average case

```python
def percentile_quickselect(values, p):
    k = int(p * len(values))
    return np.partition(values, k)[k]
```

**VPP approach**: Pre-sort reservoir periodically (every 1 second) để amortize cost.

## 6. Mathematical Properties

### 6.1 Variance of Sample Mean

Reservoir sample mean $\bar{X}$ là unbiased estimator của population mean μ:

```
E[X̄] = μ
Var[X̄] = σ²/k × (N-k)/(N-1)
```

Trong đó:
- σ² = population variance
- k = reservoir size
- N = total stream size

**Implication**: Larger k → lower variance → more accurate estimates.

### 6.2 Coverage Probability

Xác suất ít nhất một element từ subgroup S ⊆ Stream được chọn:

```
P(at least one from S) = 1 - [(N-|S|)!/(N-k)!] / [(N)!/(N-k)!]
                        ≈ 1 - (1 - k/N)^|S|  (khi N lớn)
```

**Example**: N=10000, k=128, |S|=100
```
P ≈ 1 - (1 - 128/10000)^100 ≈ 0.72
```

## 7. Experimental Validation

### 7.1 Uniformity Test

**Null hypothesis**: Reservoir samples theo uniform distribution.

**Test procedure**:
1. Stream n=10000 elements với known distribution
2. Track số lần mỗi element được chọn
3. Apply chi-square test:

```
χ² = Σᵢ (Oᵢ - Eᵢ)² / Eᵢ
```

Trong đó:
- Oᵢ = observed frequency
- Eᵢ = expected frequency = k/n × số lần stream qua i

**Acceptance criterion**: χ² < critical value at α=0.05 significance level.

### 7.2 Feature Accuracy Test

So sánh reservoir-based features với ground truth:

```python
# Ground truth
true_mean = np.mean(all_flows)
true_p90 = np.percentile(all_flows, 90)

# Reservoir estimate
reservoir_mean = np.mean(reservoir.samples)
reservoir_p90 = np.percentile(reservoir.samples, 90)

# Relative error
error_mean = abs(reservoir_mean - true_mean) / true_mean
error_p90 = abs(reservoir_p90 - true_p90) / true_p90

# Should be < 5% with k=128
assert error_mean < 0.05
assert error_p90 < 0.05
```

## 8. Tài liệu Tham khảo

### Papers

1. **Vitter, J. S.** (1985). "Random sampling with a reservoir". *ACM Transactions on Mathematical Software*, 11(1), 37-57.
   - Original Algorithm R và Algorithm L

2. **Efraimidis, P. S., & Spirakis, P. G.** (2006). "Weighted random sampling with a reservoir". *Information Processing Letters*, 97(5), 181-185.
   - Weighted reservoir sampling

3. **Li, K.** (1994). "Reservoir-sampling algorithms of time complexity O(n(1+log(N/n)))". *ACM Transactions on Mathematical Software*, 20(4), 481-493.
   - Further optimizations

### Books

4. **Knuth, D.** "The Art of Computer Programming, Volume 2: Seminumerical Algorithms", Section 3.4.2.
   - Mathematical foundations

5. **Cormode, G., et al.** "Synopses for Massive Data: Samples, Histograms, Wavelets, Sketches".
   - Modern streaming algorithms

### Online Resources

6. **VPP Documentation**: https://fd.io/
7. **numpy.random documentation**: https://numpy.org/doc/stable/reference/random/
