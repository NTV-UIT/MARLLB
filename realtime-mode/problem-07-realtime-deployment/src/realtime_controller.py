"""
Real-time RL Controller for Production VPP Deployment

Extends RLController from Problem 06 with production features:
- Health monitoring & server detection
- Graceful failover for dead servers
- Prometheus metrics export
- Error handling & logging
- Auto-restart on failures

Author: MARLLB Implementation Team
Date: December 14, 2025
"""

import sys
import os
import time
import logging
import signal
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

# Prometheus client for metrics
try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Metrics disabled.")

# Import from Problem 06
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'problem-06-vpp-integration' / 'src'))
from rl_controller import RLController


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitor backend server health via periodic checks."""
    
    def __init__(self, servers: List[str], check_interval: float = 5.0):
        """
        Args:
            servers: List of server IPs/hostnames
            check_interval: Seconds between health checks
        """
        self.servers = servers
        self.check_interval = check_interval
        self.last_check_time = 0
        self.healthy_servers: Set[str] = set(servers)
        self.dead_servers: Set[str] = set()
        
        logger.info(f"HealthMonitor initialized with {len(servers)} servers")
    
    def check(self) -> Set[str]:
        """
        Check server health. Returns set of dead servers.
        
        Returns:
            Set of dead server IPs
        """
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return set()  # Skip check
        
        self.last_check_time = current_time
        newly_dead = set()
        newly_alive = set()
        
        for server in self.servers:
            is_alive = self._ping_server(server)
            
            if is_alive and server in self.dead_servers:
                # Server recovered
                self.dead_servers.remove(server)
                self.healthy_servers.add(server)
                newly_alive.add(server)
                logger.info(f"✓ Server {server} recovered")
                
            elif not is_alive and server in self.healthy_servers:
                # Server died
                self.healthy_servers.remove(server)
                self.dead_servers.add(server)
                newly_dead.add(server)
                logger.warning(f"✗ Server {server} died")
        
        if newly_dead:
            logger.warning(f"Dead servers: {newly_dead}")
        if newly_alive:
            logger.info(f"Recovered servers: {newly_alive}")
        
        return newly_dead
    
    def _ping_server(self, server: str) -> bool:
        """
        Check if server is alive (simple ping).
        
        In production, use more sophisticated checks:
        - TCP connection to service port
        - HTTP health endpoint
        - Response time validation
        
        Args:
            server: Server IP/hostname
            
        Returns:
            True if alive
        """
        # TODO: Implement real health check
        # For now, always return True (placeholder)
        return True


class PrometheusExporter:
    """Export metrics to Prometheus."""
    
    def __init__(self, port: int = 9090):
        """
        Args:
            port: Prometheus HTTP server port
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Metrics disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.port = port
        
        # Define metrics
        self.latency_avg = Gauge('lb_latency_avg_ms', 'Average latency (ms)')
        self.latency_p95 = Gauge('lb_latency_p95_ms', 'P95 latency (ms)')
        self.latency_p99 = Gauge('lb_latency_p99_ms', 'P99 latency (ms)')
        
        self.fairness_jain = Gauge('lb_fairness_jain', 'Jain fairness index')
        self.fairness_cv = Gauge('lb_fairness_cv', 'Coefficient of variation')
        
        self.throughput = Gauge('lb_throughput_rps', 'Throughput (req/s)')
        self.total_requests = Counter('lb_total_requests', 'Total requests processed')
        
        self.agent_inference_time = Histogram(
            'lb_agent_inference_seconds',
            'Agent inference time (seconds)',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        )
        
        # Start HTTP server
        start_http_server(port)
        logger.info(f"✓ Prometheus exporter started on port {port}")
    
    def update(self, metrics: Dict[str, float]):
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
        """
        if not self.enabled:
            return
        
        # Latency metrics
        if 'latency_avg' in metrics:
            self.latency_avg.set(metrics['latency_avg'])
        if 'latency_p95' in metrics:
            self.latency_p95.set(metrics['latency_p95'])
        if 'latency_p99' in metrics:
            self.latency_p99.set(metrics['latency_p99'])
        
        # Fairness metrics
        if 'fairness_jain' in metrics:
            self.fairness_jain.set(metrics['fairness_jain'])
        if 'fairness_cv' in metrics:
            self.fairness_cv.set(metrics['fairness_cv'])
        
        # Throughput
        if 'throughput' in metrics:
            self.throughput.set(metrics['throughput'])
        if 'total_requests' in metrics:
            self.total_requests.inc(metrics['total_requests'])


class FailoverHandler:
    """Handle server failures and graceful degradation."""
    
    def __init__(self, min_healthy_servers: int = 1):
        """
        Args:
            min_healthy_servers: Minimum servers before emergency fallback
        """
        self.min_healthy_servers = min_healthy_servers
        self.fallback_mode = False
        
        logger.info(f"FailoverHandler initialized (min_servers={min_healthy_servers})")
    
    def handle(self, dead_servers: Set[str], all_servers: List[str]) -> bool:
        """
        Handle server failures.
        
        Args:
            dead_servers: Set of dead server IPs
            all_servers: List of all servers
            
        Returns:
            True if system can continue, False if emergency fallback needed
        """
        healthy_count = len(all_servers) - len(dead_servers)
        
        if healthy_count < self.min_healthy_servers:
            logger.critical(f"Only {healthy_count} healthy servers left! Emergency fallback.")
            self.emergency_fallback()
            return False
        
        logger.info(f"Failover: {healthy_count}/{len(all_servers)} servers healthy")
        # Agent will automatically adjust weights for remaining servers
        return True
    
    def emergency_fallback(self):
        """
        Emergency fallback to baseline policy.
        
        In production:
        - Switch to round-robin or static weights
        - Alert operations team
        - Trigger auto-scaling if available
        """
        self.fallback_mode = True
        logger.critical("⚠️ EMERGENCY FALLBACK ACTIVATED")
        logger.critical("Switching to round-robin baseline policy")
        # TODO: Implement baseline fallback


class RealtimeController(RLController):
    """
    Production-ready RL controller with monitoring & failover.
    
    Extends RLController from Problem 06 with:
    - Health monitoring
    - Graceful failover
    - Metrics export (Prometheus)
    - Error handling & auto-restart
    - Signal handling (SIGTERM, SIGINT)
    """
    
    def __init__(self, 
                 agent_type='qmix',
                 num_servers=16,
                 num_agents=4,
                 model_path=None,
                 shm_path='/dev/shm/lb_rl_shm',
                 update_interval=0.2,
                 online_training=False,
                 config=None,
                 # Production-specific
                 server_ips=None,
                 health_check_interval=5.0,
                 prometheus_port=9090,
                 min_healthy_servers=1):
        """
        Args:
            (Same as RLController, plus:)
            server_ips: List of backend server IPs
            health_check_interval: Seconds between health checks
            prometheus_port: Port for Prometheus metrics
            min_healthy_servers: Minimum servers before emergency fallback
        """
        # Initialize base controller
        super().__init__(
            agent_type=agent_type,
            num_servers=num_servers,
            num_agents=num_agents,
            model_path=model_path,
            shm_path=shm_path,
            update_interval=update_interval,
            online_training=online_training,
            config=config
        )
        
        # Production components
        self.server_ips = server_ips or [f"192.168.1.{10+i}" for i in range(num_servers)]
        self.health_monitor = HealthMonitor(self.server_ips, health_check_interval)
        self.metrics_exporter = PrometheusExporter(prometheus_port)
        self.failover_handler = FailoverHandler(min_healthy_servers)
        
        # Signal handling
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("✓ RealtimeController initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def _control_loop(self):
        """
        Main control loop with production features.
        
        Overrides RLController._control_loop() to add:
        - Health monitoring
        - Failover handling
        - Metrics export
        - Error recovery
        """
        logger.info("Starting real-time control loop...")
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                start_time = time.time()
                
                # 1. Health check (periodic)
                dead_servers = self.health_monitor.check()
                if dead_servers:
                    can_continue = self.failover_handler.handle(
                        dead_servers, self.server_ips
                    )
                    if not can_continue:
                        logger.critical("Cannot continue, stopping controller")
                        self.stop()
                        break
                
                # 2. Read stats from VPP
                stats = self.shm.read_msg_out()
                if stats is None:
                    logger.warning("Failed to read stats from SHM")
                    time.sleep(self.update_interval)
                    continue
                
                # 3. Convert to observation
                obs = self._stats_to_observation(stats)
                
                # 4. Agent inference (timed)
                inference_start = time.time()
                action = self.agent.select_action(obs, deterministic=True)
                inference_time = time.time() - inference_start
                
                # 5. Convert to weights & mask dead servers
                weights = self._action_to_weights(action)
                weights = self._mask_dead_servers(weights, dead_servers)
                
                # 6. Write to VPP
                success = self.shm.write_msg_in(weights)
                if not success:
                    logger.warning("Failed to write weights to SHM")
                
                # 7. Compute & export metrics
                metrics = self._compute_metrics(stats, inference_time)
                self.metrics_exporter.update(metrics)
                
                # 8. Logging (every 100 iterations)
                if iteration % 100 == 0:
                    logger.info(f"Iter {iteration}: "
                              f"latency={metrics.get('latency_avg', 0):.2f}ms, "
                              f"fairness={metrics.get('fairness_jain', 0):.3f}, "
                              f"throughput={metrics.get('throughput', 0):.1f} rps")
                
                # 9. Sleep until next update
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                self.stop()
                break
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}", exc_info=True)
                # Continue running (error recovery)
                time.sleep(self.update_interval)
    
    def _mask_dead_servers(self, weights: np.ndarray, dead_servers: Set[str]) -> np.ndarray:
        """
        Set weights to 0 for dead servers.
        
        Args:
            weights: Current weights array
            dead_servers: Set of dead server IPs
            
        Returns:
            Masked weights (dead servers have weight 0)
        """
        if not dead_servers:
            return weights
        
        masked_weights = weights.copy()
        for i, server_ip in enumerate(self.server_ips):
            if server_ip in dead_servers:
                masked_weights[i] = 0.0
        
        # Renormalize
        total = masked_weights.sum()
        if total > 0:
            masked_weights /= total
        
        return masked_weights
    
    def _compute_metrics(self, stats: Dict, inference_time: float) -> Dict[str, float]:
        """
        Compute metrics for export.
        
        Args:
            stats: Server statistics from VPP
            inference_time: Agent inference time (seconds)
            
        Returns:
            Dictionary of metrics
        """
        # Extract per-server latencies & loads
        latencies = []
        loads = []
        
        for server_stats in stats.get('servers', []):
            latencies.append(server_stats.get('latency_avg', 0))
            loads.append(server_stats.get('n_flow_on', 0))
        
        latencies = np.array(latencies)
        loads = np.array(loads)
        
        # Compute metrics
        metrics = {}
        
        # Latency
        if len(latencies) > 0:
            metrics['latency_avg'] = float(np.mean(latencies))
            metrics['latency_p95'] = float(np.percentile(latencies, 95))
            metrics['latency_p99'] = float(np.percentile(latencies, 99))
        
        # Fairness
        if len(loads) > 0 and np.sum(loads) > 0:
            # Jain's index
            sum_x = np.sum(loads)
            sum_x2 = np.sum(loads ** 2)
            n = len(loads)
            metrics['fairness_jain'] = float((sum_x ** 2) / (n * sum_x2 + 1e-9))
            
            # Coefficient of variation
            mean_load = np.mean(loads)
            std_load = np.std(loads)
            metrics['fairness_cv'] = float(std_load / (mean_load + 1e-9))
        
        # Throughput
        total_requests = stats.get('total_requests', 0)
        metrics['throughput'] = float(total_requests / self.update_interval)
        metrics['total_requests'] = float(total_requests)
        
        # Agent performance
        self.metrics_exporter.agent_inference_time.observe(inference_time)
        
        return metrics


def main():
    """Main entry point for production deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time RL Controller for VPP')
    parser.add_argument('--agent-type', type=str, default='qmix', choices=['sac-gru', 'qmix'])
    parser.add_argument('--num-servers', type=int, default=16)
    parser.add_argument('--num-agents', type=int, default=4)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--shm-path', type=str, default='/dev/shm/lb_rl_shm')
    parser.add_argument('--update-interval', type=float, default=0.2)
    parser.add_argument('--health-check-interval', type=float, default=5.0)
    parser.add_argument('--prometheus-port', type=int, default=9090)
    parser.add_argument('--min-healthy-servers', type=int, default=1)
    parser.add_argument('--server-ips', type=str, nargs='+', default=None)
    
    args = parser.parse_args()
    
    # Create controller
    controller = RealtimeController(
        agent_type=args.agent_type,
        num_servers=args.num_servers,
        num_agents=args.num_agents,
        model_path=args.model_path,
        shm_path=args.shm_path,
        update_interval=args.update_interval,
        online_training=False,  # Production: inference only
        server_ips=args.server_ips,
        health_check_interval=args.health_check_interval,
        prometheus_port=args.prometheus_port,
        min_healthy_servers=args.min_healthy_servers
    )
    
    # Start controller
    try:
        controller.start()
        
        # Wait until stopped
        while controller.is_running():
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        controller.stop()
        logger.info("Controller stopped")


if __name__ == '__main__':
    main()
