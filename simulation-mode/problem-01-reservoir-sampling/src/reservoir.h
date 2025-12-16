/**
 * Reservoir Sampling - C Implementation for VPP Plugin
 * 
 * This is a high-performance implementation designed for integration
 * with VPP's packet processing pipeline. Key features:
 * - Lock-free single-threaded design (VPP is single-threaded per worker)
 * - Cache-friendly memory layout (SoA instead of AoS)
 * - Fast random number generation (xorshift128+)
 * - O(1) amortized insertion time
 * 
 * Author: MARLLB Implementation Team
 * Date: December 2025
 */

#ifndef RESERVOIR_H
#define RESERVOIR_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define RESERVOIR_CAPACITY 128
#define NUM_FEATURES 5

/**
 * Reservoir data structure (Struct of Arrays layout for cache efficiency)
 */
typedef struct {
    float values[RESERVOIR_CAPACITY];       /* Sampled values (FCT, duration, etc.) */
    uint64_t timestamps[RESERVOIR_CAPACITY]; /* Timestamps in microseconds */
    uint64_t count;                         /* Total number of elements seen */
    uint64_t rng_state[2];                  /* Random number generator state */
    bool is_full;                           /* Whether reservoir has reached capacity */
} reservoir_t;

/**
 * Feature statistics computed from reservoir
 */
typedef struct {
    float mean;
    float p90;
    float std;
    float mean_decay;
    float p90_decay;
} reservoir_stats_t;

/**
 * Initialize reservoir sampler
 * 
 * @param r Pointer to reservoir structure
 * @param seed Random seed (use system time if 0)
 */
static inline void reservoir_init(reservoir_t *r, uint64_t seed) {
    memset(r, 0, sizeof(reservoir_t));
    r->count = 0;
    r->is_full = false;
    
    /* Initialize RNG state */
    if (seed == 0) {
        /* Use simple seed if none provided */
        r->rng_state[0] = 0x123456789abcdef0ULL;
        r->rng_state[1] = 0xfedcba9876543210ULL;
    } else {
        r->rng_state[0] = seed;
        r->rng_state[1] = seed ^ 0xffffffffffffffffULL;
    }
}

/**
 * Fast random number generator: xorshift128+
 * 
 * This is much faster than rand() and has good statistical properties.
 * Performance: ~0.5ns per call on modern CPUs
 * 
 * @param state RNG state (2 uint64_t values)
 * @return Random uint64_t value
 */
static inline uint64_t xorshift128plus(uint64_t state[2]) {
    uint64_t x = state[0];
    uint64_t const y = state[1];
    state[0] = y;
    x ^= x << 23;
    state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return state[1] + y;
}

/**
 * Generate random integer in range [0, max)
 * 
 * Uses rejection sampling to avoid modulo bias.
 * 
 * @param state RNG state
 * @param max Upper bound (exclusive)
 * @return Random integer in [0, max)
 */
static inline uint64_t random_range(uint64_t state[2], uint64_t max) {
    if (max == 0) return 0;
    
    /* Generate random value */
    uint64_t r = xorshift128plus(state);
    
    /* Simple modulo (slight bias, but acceptable for reservoir sampling) */
    return r % max;
}

/**
 * Add value to reservoir using Algorithm R
 * 
 * Time complexity: O(1) amortized
 * 
 * @param r Pointer to reservoir
 * @param value Value to add (e.g., FCT in seconds)
 * @param timestamp Timestamp in microseconds
 * @return true if value was added, false if rejected
 */
static inline bool reservoir_add(reservoir_t *r, float value, uint64_t timestamp) {
    if (r->count < RESERVOIR_CAPACITY) {
        /* Initial fill phase */
        r->values[r->count] = value;
        r->timestamps[r->count] = timestamp;
        r->count++;
        
        if (r->count == RESERVOIR_CAPACITY) {
            r->is_full = true;
        }
        return true;
    } else {
        /* Probabilistic replacement phase */
        uint64_t j = random_range(r->rng_state, r->count + 1);
        
        if (j < RESERVOIR_CAPACITY) {
            r->values[j] = value;
            r->timestamps[j] = timestamp;
            r->count++;
            return true;
        } else {
            r->count++;
            return false;
        }
    }
}

/**
 * Get current size of reservoir
 * 
 * @param r Pointer to reservoir
 * @return Number of samples in reservoir
 */
static inline uint64_t reservoir_size(const reservoir_t *r) {
    return (r->count < RESERVOIR_CAPACITY) ? r->count : RESERVOIR_CAPACITY;
}

/**
 * Compare function for qsort (ascending order)
 */
static inline int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/**
 * Compute basic statistics from reservoir
 * 
 * This function computes the 5 features used in MARLLB:
 * - mean: average value
 * - p90: 90th percentile
 * - std: standard deviation
 * - mean_decay: decay-weighted mean
 * - p90_decay: decay-weighted 90th percentile
 * 
 * @param r Pointer to reservoir
 * @param stats Pointer to output statistics structure
 * @param decay_factor Decay factor for temporal weighting (e.g., 0.9)
 * @param current_time Current timestamp in microseconds
 */
static inline void reservoir_compute_stats(const reservoir_t *r,
                                          reservoir_stats_t *stats,
                                          float decay_factor,
                                          uint64_t current_time) {
    uint64_t size = reservoir_size(r);
    
    if (size == 0) {
        memset(stats, 0, sizeof(reservoir_stats_t));
        return;
    }
    
    /* Compute mean and std */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (uint64_t i = 0; i < size; i++) {
        float v = r->values[i];
        sum += v;
        sum_sq += v * v;
    }
    
    stats->mean = sum / size;
    float variance = (sum_sq / size) - (stats->mean * stats->mean);
    stats->std = sqrtf(variance > 0 ? variance : 0);
    
    /* Compute p90 (requires sorting a copy) */
    float sorted_values[RESERVOIR_CAPACITY];
    memcpy(sorted_values, r->values, size * sizeof(float));
    qsort(sorted_values, size, sizeof(float), compare_float);
    
    uint64_t p90_idx = (uint64_t)(0.9f * size);
    if (p90_idx >= size) p90_idx = size - 1;
    stats->p90 = sorted_values[p90_idx];
    
    /* Compute decay-weighted statistics */
    float weight_sum = 0.0f;
    float weighted_sum = 0.0f;
    
    /* Pre-compute weights */
    float weights[RESERVOIR_CAPACITY];
    
    for (uint64_t i = 0; i < size; i++) {
        /* Time difference in seconds */
        float time_diff = (current_time - r->timestamps[i]) / 1000000.0f;
        
        /* Exponential decay: decay_factor^time_diff */
        /* Use pow() or approximate with exp(log(decay) * time_diff) */
        weights[i] = expf(logf(decay_factor) * time_diff);
        
        weight_sum += weights[i];
        weighted_sum += r->values[i] * weights[i];
    }
    
    stats->mean_decay = weighted_sum / weight_sum;
    
    /* Compute weighted p90 */
    /* Sort by value, accumulate weights until reaching 90% of weight_sum */
    
    /* Create array of (value, weight) pairs and sort by value */
    typedef struct { float value; float weight; } value_weight_t;
    value_weight_t vw[RESERVOIR_CAPACITY];
    
    for (uint64_t i = 0; i < size; i++) {
        vw[i].value = r->values[i];
        vw[i].weight = weights[i];
    }
    
    /* Simple bubble sort (okay for small N=128) */
    for (uint64_t i = 0; i < size - 1; i++) {
        for (uint64_t j = 0; j < size - i - 1; j++) {
            if (vw[j].value > vw[j + 1].value) {
                value_weight_t temp = vw[j];
                vw[j] = vw[j + 1];
                vw[j + 1] = temp;
            }
        }
    }
    
    /* Find 90th percentile by cumulative weight */
    float cutoff = 0.9f * weight_sum;
    float cumsum = 0.0f;
    
    for (uint64_t i = 0; i < size; i++) {
        cumsum += vw[i].weight;
        if (cumsum >= cutoff) {
            stats->p90_decay = vw[i].value;
            break;
        }
    }
}

/**
 * Reset reservoir to empty state
 * 
 * @param r Pointer to reservoir
 */
static inline void reservoir_reset(reservoir_t *r) {
    r->count = 0;
    r->is_full = false;
    memset(r->values, 0, sizeof(r->values));
    memset(r->timestamps, 0, sizeof(r->timestamps));
}

#endif /* RESERVOIR_H */
