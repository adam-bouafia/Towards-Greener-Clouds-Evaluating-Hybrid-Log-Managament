"""
Metrics collector for experiment tracking.

Tracks performance metrics like latency, throughput, success rates.
"""

import time
import psutil
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LogMetrics:
    """Metrics for a single log entry."""
    log_id: int
    timestamp: float
    backend: str
    routing_latency_ms: float
    write_latency_ms: float
    total_latency_ms: float
    energy_joules: float
    success: bool
    cpu_percent: float
    memory_mb: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics for an experiment."""
    total_logs: int = 0
    successful_logs: int = 0
    failed_logs: int = 0
    total_latency_ms: float = 0.0
    total_energy_joules: float = 0.0
    avg_latency_ms: float = 0.0
    avg_energy_joules: float = 0.0
    throughput_logs_per_sec: float = 0.0
    success_rate: float = 0.0
    duration_seconds: float = 0.0
    
    # Per-backend stats
    backend_counts: Dict[str, int] = field(default_factory=dict)
    backend_latencies: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and aggregate performance metrics during experiments.
    
    Tracks:
    - Latency (routing + write)
    - Energy consumption
    - Success rates
    - Resource usage (CPU, memory)
    - Per-backend statistics
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.log_metrics: List[LogMetrics] = []
        self.backend_data = defaultdict(lambda: {"count": 0, "latency": 0.0})
        self.start_time = None
        self.end_time = None
        
        # Process handle for resource monitoring
        self.process = psutil.Process()
    
    def start_experiment(self):
        """Mark experiment start time."""
        self.start_time = time.time()
        self.log_metrics.clear()
        self.backend_data.clear()
    
    def record_log(
        self,
        log_id: int,
        backend: str,
        routing_latency_ms: float,
        write_latency_ms: float,
        energy_joules: float,
        success: bool
    ):
        """
        Record metrics for a single log entry.
        
        Args:
            log_id: Log identifier
            backend: Target backend name
            routing_latency_ms: Time to route (ms)
            write_latency_ms: Time to write (ms)
            energy_joules: Energy consumed (J)
            success: Whether write succeeded
        """
        metrics = LogMetrics(
            log_id=log_id,
            timestamp=time.time(),
            backend=backend,
            routing_latency_ms=routing_latency_ms,
            write_latency_ms=write_latency_ms,
            total_latency_ms=routing_latency_ms + write_latency_ms,
            energy_joules=energy_joules,
            success=success,
            cpu_percent=self.process.cpu_percent(),
            memory_mb=self.process.memory_info().rss / 1024 / 1024
        )
        
        self.log_metrics.append(metrics)
        
        # Update backend stats
        self.backend_data[backend]["count"] += 1
        self.backend_data[backend]["latency"] += metrics.total_latency_ms
    
    def end_experiment(self):
        """Mark experiment end time."""
        self.end_time = time.time()
    
    def get_aggregate_metrics(self) -> AggregateMetrics:
        """
        Calculate aggregate metrics from collected data.
        
        Returns:
            AggregateMetrics object with summary statistics
        """
        if not self.log_metrics:
            return AggregateMetrics()
        
        total_logs = len(self.log_metrics)
        successful_logs = sum(1 for m in self.log_metrics if m.success)
        failed_logs = total_logs - successful_logs
        
        total_latency = sum(m.total_latency_ms for m in self.log_metrics)
        total_energy = sum(m.energy_joules for m in self.log_metrics)
        
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        duration = max(duration, 0.001)  # Avoid division by zero
        
        # Backend stats
        backend_counts = {k: v["count"] for k, v in self.backend_data.items()}
        backend_latencies = {
            k: v["latency"] / v["count"] if v["count"] > 0 else 0.0
            for k, v in self.backend_data.items()
        }
        
        return AggregateMetrics(
            total_logs=total_logs,
            successful_logs=successful_logs,
            failed_logs=failed_logs,
            total_latency_ms=total_latency,
            total_energy_joules=total_energy,
            avg_latency_ms=total_latency / total_logs,
            avg_energy_joules=total_energy / total_logs,
            throughput_logs_per_sec=total_logs / duration,
            success_rate=successful_logs / total_logs,
            duration_seconds=duration,
            backend_counts=backend_counts,
            backend_latencies=backend_latencies
        )
    
    def get_log_metrics(self) -> List[LogMetrics]:
        """
        Get detailed per-log metrics.
        
        Returns:
            List of LogMetrics objects
        """
        return self.log_metrics
