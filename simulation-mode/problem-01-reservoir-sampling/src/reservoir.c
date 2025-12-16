/**
 * Reservoir Sampling - C Implementation (Source File)
 * 
 * This file provides implementations for functions that are too large
 * to be inlined in the header.
 * 
 * Author: MARLLB Implementation Team
 * Date: December 2025
 */

#include "reservoir.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * Print reservoir statistics (for debugging)
 */
void reservoir_print_stats(const reservoir_t *r, const reservoir_stats_t *stats) {
    printf("Reservoir Statistics:\n");
    printf("  Count: %llu\n", (unsigned long long)r->count);
    printf("  Size: %llu\n", (unsigned long long)reservoir_size(r));
    printf("  Full: %s\n", r->is_full ? "true" : "false");
    printf("\nFeatures:\n");
    printf("  mean:       %.6f\n", stats->mean);
    printf("  p90:        %.6f\n", stats->p90);
    printf("  std:        %.6f\n", stats->std);
    printf("  mean_decay: %.6f\n", stats->mean_decay);
    printf("  p90_decay:  %.6f\n", stats->p90_decay);
}

/**
 * Example: Simulate flow statistics collection
 */
void example_flow_tracking() {
    printf("=== Reservoir Sampling Demo (C) ===\n\n");
    
    /* Initialize reservoir */
    reservoir_t reservoir;
    uint64_t seed = (uint64_t)time(NULL);
    reservoir_init(&reservoir, seed);
    
    printf("Initialized reservoir with capacity %d\n", RESERVOIR_CAPACITY);
    
    /* Simulate incoming flows */
    int num_flows = 1000;
    uint64_t current_time = 1000000000; /* Start time in microseconds */
    
    printf("Simulating %d flows...\n", num_flows);
    
    for (int i = 0; i < num_flows; i++) {
        /* Simulate FCT: exponential distribution with mean 0.1s */
        float fct = -0.1f * logf((float)rand() / RAND_MAX);
        
        /* Advance time by 1ms */
        current_time += 1000;
        
        /* Add to reservoir */
        reservoir_add(&reservoir, fct, current_time);
    }
    
    printf("Added %d flows to reservoir\n", num_flows);
    printf("Reservoir size: %llu\n\n", (unsigned long long)reservoir_size(&reservoir));
    
    /* Compute statistics */
    reservoir_stats_t stats;
    reservoir_compute_stats(&reservoir, &stats, 0.9f, current_time);
    
    /* Print results */
    reservoir_print_stats(&reservoir, &stats);
}

/**
 * Benchmark: Measure add() performance
 */
void benchmark_add_performance() {
    printf("\n=== Performance Benchmark ===\n\n");
    
    reservoir_t reservoir;
    reservoir_init(&reservoir, 12345);
    
    int num_operations = 1000000;
    
    clock_t start = clock();
    
    for (int i = 0; i < num_operations; i++) {
        float value = (float)i * 0.001f;
        reservoir_add(&reservoir, value, (uint64_t)i);
    }
    
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double ops_per_sec = num_operations / elapsed;
    double ns_per_op = (elapsed * 1e9) / num_operations;
    
    printf("Operations: %d\n", num_operations);
    printf("Time: %.3f seconds\n", elapsed);
    printf("Throughput: %.2f M ops/sec\n", ops_per_sec / 1e6);
    printf("Latency: %.2f ns/op\n", ns_per_op);
}

/**
 * Main function for testing
 */
int main() {
    /* Run example */
    example_flow_tracking();
    
    /* Run benchmark */
    benchmark_add_performance();
    
    return 0;
}
