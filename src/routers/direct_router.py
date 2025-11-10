"""
Direct routers - Baseline routers that always route to same backend.

Used for comparison and baseline experiments.
"""

from typing import Dict
from .base import BaseRouter


class DirectRouter(BaseRouter):
    """
    Direct router that always routes to a specific backend.
    
    Used as baseline for comparison:
    - DirectRouter("clickhouse") → All logs to hot storage
    - DirectRouter("minio") → All logs to cold storage
    """
    
    def __init__(self, backend: str):
        """
        Initialize direct router.
        
        Args:
            backend: Target backend ("clickhouse" or "minio")
        """
        if backend not in ["clickhouse", "minio"]:
            raise ValueError(f"Invalid backend: {backend}. Must be 'clickhouse' or 'minio'")
        
        self.backend = backend
        print(f"✅ DirectRouter initialized → all logs to {backend}")
    
    def get_route(self, log_entry: Dict) -> str:
        """
        Always return the configured backend.
        
        Returns:
            Configured backend name
        """
        return self.backend
