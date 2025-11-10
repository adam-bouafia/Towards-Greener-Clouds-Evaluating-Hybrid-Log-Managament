"""
Base router interface - abstract class for all routing algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict


class BaseRouter(ABC):
    """
    Abstract base class for all log routers.
    
    All routers must implement:
    - get_route(): Decide where to route a log
    - observe(): Optional feedback for adaptive routers
    """
    
    @abstractmethod
    def get_route(self, log_entry: Dict) -> str:
        """
        Decide where to route a log entry.
        
        Args:
            log_entry: Dictionary containing log fields (Level, Component, Content, etc.)
        
        Returns:
            Backend name: "clickhouse" or "minio"
        """
        raise NotImplementedError
    
    def observe(
        self,
        log_entry: Dict,
        destination: str,
        success: bool,
        latency_ms: float,
        energy_joules: float
    ):
        """
        Optional: Observe routing outcome for adaptive learning.
        
        Args:
            log_entry: The log that was routed
            destination: Where it was routed ("clickhouse" or "minio")
            success: Whether write succeeded
            latency_ms: Backend write latency
            energy_joules: Energy consumed
        """
        pass  # Default: no-op for non-adaptive routers
