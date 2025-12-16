/*
 * Alias Table Implementation for O(1) Weighted Server Selection
 * 
 * Implements Vose's Alias Method:
 * - Build: O(n) time complexity
 * - Sample: O(1) time complexity
 * - Space: O(n)
 * 
 * Reference:
 * - Walker, A. J. (1977). "An Efficient Method for Generating 
 *   Discrete Random Variables with General Distributions"
 * - Vose, M. D. (1991). "A Linear Algorithm for Generating Random 
 *   Numbers with a Given Distribution"
 * 
 * Author: MARLLB Implementation Team
 * Date: December 14, 2025
 */

#ifndef _ALIAS_TABLE_H_
#define _ALIAS_TABLE_H_

#include <vppinfra/types.h>
#include <vppinfra/random.h>

/*
 * Alias table structure
 */
typedef struct {
    u32 num_servers;    // Number of servers (n)
    f32 *prob;          // Probability table [n]
    u32 *alias;         // Alias table [n]
    
    // Random state
    u64 random_seed;
    u32 random_state;
} alias_table_t;

/*
 * Create alias table
 */
static inline alias_table_t *
alias_table_create(u32 num_servers)
{
    alias_table_t *table = clib_mem_alloc(sizeof(alias_table_t));
    
    table->num_servers = num_servers;
    table->prob = clib_mem_alloc(num_servers * sizeof(f32));
    table->alias = clib_mem_alloc(num_servers * sizeof(u32));
    
    // Initialize random seed
    table->random_seed = time(NULL);
    table->random_state = (u32)table->random_seed;
    
    return table;
}

/*
 * Free alias table
 */
static inline void
alias_table_free(alias_table_t *table)
{
    if (!table)
        return;
    
    clib_mem_free(table->prob);
    clib_mem_free(table->alias);
    clib_mem_free(table);
}

/*
 * Build alias table from weights (Vose's Algorithm)
 * 
 * Time complexity: O(n)
 * 
 * Args:
 *   table: Alias table to build
 *   weights: Array of weights [num_servers]
 *   n: Number of servers
 */
static inline void
alias_table_build(alias_table_t *table, f32 *weights, u32 n)
{
    if (!table || !weights || n == 0)
        return;
    
    ASSERT(n == table->num_servers);
    
    // Step 1: Normalize weights to probabilities (sum = n)
    f32 sum = 0.0;
    for (u32 i = 0; i < n; i++) {
        sum += weights[i];
    }
    
    if (sum <= 0.0) {
        // All weights are zero, use uniform distribution
        for (u32 i = 0; i < n; i++) {
            table->prob[i] = 1.0;
            table->alias[i] = i;
        }
        return;
    }
    
    // Normalize: prob[i] = n * weights[i] / sum
    f32 *prob_scaled = clib_mem_alloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++) {
        prob_scaled[i] = n * weights[i] / sum;
    }
    
    // Step 2: Partition into small and large
    u32 *small = clib_mem_alloc(n * sizeof(u32));
    u32 *large = clib_mem_alloc(n * sizeof(u32));
    u32 small_count = 0;
    u32 large_count = 0;
    
    for (u32 i = 0; i < n; i++) {
        if (prob_scaled[i] < 1.0) {
            small[small_count++] = i;
        } else {
            large[large_count++] = i;
        }
    }
    
    // Step 3: Construct alias table
    while (small_count > 0 && large_count > 0) {
        u32 s = small[--small_count];
        u32 l = large[--large_count];
        
        table->prob[s] = prob_scaled[s];
        table->alias[s] = l;
        
        prob_scaled[l] = (prob_scaled[l] + prob_scaled[s]) - 1.0;
        
        if (prob_scaled[l] < 1.0) {
            small[small_count++] = l;
        } else {
            large[large_count++] = l;
        }
    }
    
    // Step 4: Fill remaining (should be close to 1.0)
    while (small_count > 0) {
        u32 s = small[--small_count];
        table->prob[s] = 1.0;
        table->alias[s] = s;
    }
    
    while (large_count > 0) {
        u32 l = large[--large_count];
        table->prob[l] = 1.0;
        table->alias[l] = l;
    }
    
    // Cleanup
    clib_mem_free(prob_scaled);
    clib_mem_free(small);
    clib_mem_free(large);
}

/*
 * Fast random number generator (xorshift32)
 */
static inline u32
alias_table_random(alias_table_t *table)
{
    u32 x = table->random_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    table->random_state = x;
    return x;
}

/*
 * Generate random float in [0, 1)
 */
static inline f32
alias_table_random_float(alias_table_t *table)
{
    u32 r = alias_table_random(table);
    return (f32)r / (f32)0xFFFFFFFF;
}

/*
 * Sample server index from alias table
 * 
 * Time complexity: O(1)
 * 
 * Args:
 *   table: Alias table
 *   
 * Returns:
 *   Server index in [0, num_servers-1]
 */
static inline u32
alias_table_sample(alias_table_t *table)
{
    if (!table || table->num_servers == 0)
        return 0;
    
    // Step 1: Generate random index i in [0, n-1]
    u32 i = alias_table_random(table) % table->num_servers;
    
    // Step 2: Generate random float r in [0, 1)
    f32 r = alias_table_random_float(table);
    
    // Step 3: Return i or alias[i] based on prob[i]
    return (r < table->prob[i]) ? i : table->alias[i];
}

/*
 * Verify alias table distribution (for testing)
 * 
 * Samples n_samples times and returns histogram.
 * 
 * Args:
 *   table: Alias table
 *   n_samples: Number of samples to draw
 *   histogram: Output array [num_servers]
 */
static inline void
alias_table_test_distribution(alias_table_t *table, u64 n_samples, u64 *histogram)
{
    if (!table || !histogram)
        return;
    
    // Initialize histogram
    for (u32 i = 0; i < table->num_servers; i++) {
        histogram[i] = 0;
    }
    
    // Sample n_samples times
    for (u64 s = 0; s < n_samples; s++) {
        u32 idx = alias_table_sample(table);
        histogram[idx]++;
    }
}

#endif /* _ALIAS_TABLE_H_ */
