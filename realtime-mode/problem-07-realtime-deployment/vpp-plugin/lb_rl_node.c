/*
 * VPP Load Balancer RL Node
 * 
 * Packet processing node for RL-based load balancing.
 * Integrates with Python RL controller via shared memory.
 * 
 * Features:
 * - Read server weights from shared memory (msg_in)
 * - Select backend server using alias table (O(1))
 * - Forward packets to selected server
 * - Collect per-server statistics
 * - Write stats to shared memory (msg_out)
 * 
 * Author: MARLLB Implementation Team
 * Date: December 14, 2025
 * 
 * Usage:
 *   vppctl lb rl enable
 *   vppctl lb rl set-servers 192.168.1.10-25
 *   vppctl lb rl start
 */

#include <vnet/vnet.h>
#include <vnet/plugin/plugin.h>
#include <vnet/ip/ip.h>
#include <vppinfra/error.h>
#include <vppinfra/hash.h>

#include "lb_rl.h"
#include "alias_table.h"
#include "shm_reader.h"
#include "shm_writer.h"

// Plugin metadata
VLIB_PLUGIN_REGISTER() = {
    .version = "1.0",
    .description = "RL-based Load Balancer Plugin",
};

// Node registration
vlib_node_registration_t lb_rl_node;

// Global state
typedef struct {
    // Configuration
    u8 enabled;
    u32 num_servers;
    ip4_address_t *server_ips;
    u32 *server_adj_index;  // Adjacency index for each server
    
    // Shared memory
    shm_region_t *shm;
    char *shm_path;
    
    // Alias table (O(1) server selection)
    alias_table_t *alias_table;
    
    // Statistics (per-server)
    lb_rl_stats_t *stats;
    
    // Update control
    f64 last_weight_update;
    f64 last_stats_write;
    f64 stats_write_interval;  // Default: 0.1s (100ms)
    
    // Counters
    u64 total_packets;
    u64 total_bytes;
    
} lb_rl_main_t;

lb_rl_main_t lb_rl_main;

// Per-server statistics
typedef struct {
    u64 n_packets;
    u64 n_bytes;
    u64 n_flows;
    
    f64 latency_sum;
    f64 latency_sum_sq;
    u64 latency_samples;
    
    f64 last_update_time;
} lb_rl_stats_t;


/*
 * Update alias table from shared memory weights
 */
static inline void
lb_rl_update_weights(lb_rl_main_t *lm, vlib_main_t *vm)
{
    f64 now = vlib_time_now(vm);
    
    // Throttle: Update at most every 100ms
    if (now - lm->last_weight_update < 0.1)
        return;
    
    lm->last_weight_update = now;
    
    // Read weights from shared memory (msg_in)
    f32 *weights = shm_reader_get_weights(lm->shm, lm->num_servers);
    if (!weights)
        return;
    
    // Rebuild alias table: O(n) construction
    alias_table_build(lm->alias_table, weights, lm->num_servers);
    
    clib_mem_free(weights);
}

/*
 * Write statistics to shared memory
 */
static inline void
lb_rl_write_stats(lb_rl_main_t *lm, vlib_main_t *vm)
{
    f64 now = vlib_time_now(vm);
    
    // Write every 100ms by default
    if (now - lm->last_stats_write < lm->stats_write_interval)
        return;
    
    lm->last_stats_write = now;
    
    // Prepare msg_out data
    shm_msg_out_t msg;
    msg.num_servers = lm->num_servers;
    
    for (u32 i = 0; i < lm->num_servers; i++) {
        lb_rl_stats_t *s = &lm->stats[i];
        
        msg.servers[i].n_flow_on = s->n_flows;
        msg.servers[i].n_packets = s->n_packets;
        msg.servers[i].n_bytes = s->n_bytes;
        
        // Compute average latency
        if (s->latency_samples > 0) {
            msg.servers[i].latency_avg = s->latency_sum / s->latency_samples;
            
            // Compute std dev
            f64 mean = msg.servers[i].latency_avg;
            f64 variance = (s->latency_sum_sq / s->latency_samples) - (mean * mean);
            msg.servers[i].latency_std = sqrt(variance);
        } else {
            msg.servers[i].latency_avg = 0.0;
            msg.servers[i].latency_std = 0.0;
        }
        
        // Reset counters (for next interval)
        s->n_packets = 0;
        s->n_bytes = 0;
        s->latency_sum = 0.0;
        s->latency_sum_sq = 0.0;
        s->latency_samples = 0;
    }
    
    // Write to shared memory (msg_out)
    shm_writer_write_stats(lm->shm, &msg);
}

/*
 * Update per-server statistics
 */
static inline void
lb_rl_update_server_stats(lb_rl_main_t *lm, u32 server_idx, 
                          u32 packet_size, f64 latency)
{
    if (server_idx >= lm->num_servers)
        return;
    
    lb_rl_stats_t *s = &lm->stats[server_idx];
    
    s->n_packets++;
    s->n_bytes += packet_size;
    
    if (latency > 0) {
        s->latency_sum += latency;
        s->latency_sum_sq += latency * latency;
        s->latency_samples++;
    }
}

/*
 * Main packet processing function
 */
static uword
lb_rl_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node, vlib_frame_t *frame)
{
    lb_rl_main_t *lm = &lb_rl_main;
    u32 n_left_from, *from, *to_next;
    u32 next_index;
    u32 pkts_processed = 0;
    
    if (!lm->enabled)
        return 0;
    
    from = vlib_frame_vector_args(frame);
    n_left_from = frame->n_vectors;
    next_index = node->cached_next_index;
    
    // Update weights from shared memory (throttled)
    lb_rl_update_weights(lm, vm);
    
    while (n_left_from > 0) {
        u32 n_left_to_next;
        
        vlib_get_next_frame(vm, node, next_index, to_next, n_left_to_next);
        
        // Process 4 packets at a time (SIMD-friendly)
        while (n_left_from >= 4 && n_left_to_next >= 2) {
            u32 bi0, bi1;
            vlib_buffer_t *b0, *b1;
            ip4_header_t *ip0, *ip1;
            u32 next0, next1;
            u32 server_idx0, server_idx1;
            
            // Prefetch next iteration
            {
                vlib_buffer_t *p2, *p3;
                p2 = vlib_get_buffer(vm, from[2]);
                p3 = vlib_get_buffer(vm, from[3]);
                vlib_prefetch_buffer_header(p2, LOAD);
                vlib_prefetch_buffer_header(p3, LOAD);
            }
            
            // Get buffers
            bi0 = from[0];
            bi1 = from[1];
            to_next[0] = bi0;
            to_next[1] = bi1;
            from += 2;
            to_next += 2;
            n_left_from -= 2;
            n_left_to_next -= 2;
            
            b0 = vlib_get_buffer(vm, bi0);
            b1 = vlib_get_buffer(vm, bi1);
            
            // Parse IP header
            ip0 = vlib_buffer_get_current(b0);
            ip1 = vlib_buffer_get_current(b1);
            
            // Select server using alias table (O(1))
            server_idx0 = alias_table_sample(lm->alias_table);
            server_idx1 = alias_table_sample(lm->alias_table);
            
            // Update statistics
            lb_rl_update_server_stats(lm, server_idx0, 
                                     vlib_buffer_length_in_chain(vm, b0), 0.0);
            lb_rl_update_server_stats(lm, server_idx1,
                                     vlib_buffer_length_in_chain(vm, b1), 0.0);
            
            // Set next hop (adjacency index)
            vnet_buffer(b0)->ip.adj_index[VLIB_TX] = lm->server_adj_index[server_idx0];
            vnet_buffer(b1)->ip.adj_index[VLIB_TX] = lm->server_adj_index[server_idx1];
            
            next0 = next1 = IP4_REWRITE_NEXT_LOOKUP;
            
            // Verify speculative enqueue
            vlib_validate_buffer_enqueue_x2(vm, node, next_index,
                                           to_next, n_left_to_next,
                                           bi0, bi1, next0, next1);
            
            pkts_processed += 2;
        }
        
        // Process remaining packets one by one
        while (n_left_from > 0 && n_left_to_next > 0) {
            u32 bi0;
            vlib_buffer_t *b0;
            ip4_header_t *ip0;
            u32 next0;
            u32 server_idx;
            
            bi0 = from[0];
            to_next[0] = bi0;
            from += 1;
            to_next += 1;
            n_left_from -= 1;
            n_left_to_next -= 1;
            
            b0 = vlib_get_buffer(vm, bi0);
            ip0 = vlib_buffer_get_current(b0);
            
            // Select server (O(1))
            server_idx = alias_table_sample(lm->alias_table);
            
            // Update stats
            lb_rl_update_server_stats(lm, server_idx,
                                     vlib_buffer_length_in_chain(vm, b0), 0.0);
            
            // Set next hop
            vnet_buffer(b0)->ip.adj_index[VLIB_TX] = lm->server_adj_index[server_idx];
            next0 = IP4_REWRITE_NEXT_LOOKUP;
            
            vlib_validate_buffer_enqueue_x1(vm, node, next_index,
                                           to_next, n_left_to_next,
                                           bi0, next0);
            
            pkts_processed++;
        }
        
        vlib_put_next_frame(vm, node, next_index, n_left_to_next);
    }
    
    // Update global counters
    lm->total_packets += pkts_processed;
    
    // Periodically write stats to shared memory
    lb_rl_write_stats(lm, vm);
    
    return frame->n_vectors;
}

/*
 * Error strings
 */
#define foreach_lb_rl_error \
    _(PROCESSED, "Packets processed by RL load balancer")

typedef enum {
#define _(sym, str) LB_RL_ERROR_##sym,
    foreach_lb_rl_error
#undef _
    LB_RL_N_ERROR,
} lb_rl_error_t;

static char *lb_rl_error_strings[] = {
#define _(sym, string) string,
    foreach_lb_rl_error
#undef _
};

/*
 * Node registration
 */
VLIB_REGISTER_NODE(lb_rl_node) = {
    .function = lb_rl_node_fn,
    .name = "lb-rl",
    .vector_size = sizeof(u32),
    .format_trace = NULL,
    .type = VLIB_NODE_TYPE_INTERNAL,
    
    .n_errors = ARRAY_LEN(lb_rl_error_strings),
    .error_strings = lb_rl_error_strings,
    
    .n_next_nodes = 1,
    .next_nodes = {
        [0] = "ip4-rewrite",
    },
};

/*
 * Feature arc registration (insert into ip4-unicast)
 */
VNET_FEATURE_INIT(lb_rl, static) = {
    .arc_name = "ip4-unicast",
    .node_name = "lb-rl",
    .runs_before = VNET_FEATURES("ip4-lookup"),
};

/*
 * Plugin initialization
 */
static clib_error_t *
lb_rl_init(vlib_main_t *vm)
{
    lb_rl_main_t *lm = &lb_rl_main;
    
    // Initialize state
    lm->enabled = 0;
    lm->num_servers = 0;
    lm->server_ips = NULL;
    lm->server_adj_index = NULL;
    lm->shm = NULL;
    lm->shm_path = strdup("/dev/shm/lb_rl_shm");
    lm->alias_table = NULL;
    lm->stats = NULL;
    lm->last_weight_update = 0.0;
    lm->last_stats_write = 0.0;
    lm->stats_write_interval = 0.1;  // 100ms
    lm->total_packets = 0;
    lm->total_bytes = 0;
    
    vlib_cli_output(vm, "RL Load Balancer plugin initialized\n");
    
    return 0;
}

VLIB_INIT_FUNCTION(lb_rl_init);
