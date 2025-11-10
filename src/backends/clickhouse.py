"""
ClickHouse backend for hot storage.

High-performance columnar database for frequently queried logs.
"""

import time
import requests
from typing import Dict, Tuple
from datetime import datetime


class ClickHouseBackend:
    """
    ClickHouse client for log storage.
    
    Features:
    - Columnar storage with compression
    - Fast aggregation queries
    - Time-based partitioning
    - Automatic TTL (30 days)
    """
    
    def __init__(self):
        """Initialize ClickHouse client."""
        from src.config import (
            CLICKHOUSE_HOST,
            CLICKHOUSE_PORT,
            CLICKHOUSE_DB,
            CLICKHOUSE_USER,
            CLICKHOUSE_PASSWORD
        )
        
        self.host = CLICKHOUSE_HOST
        self.port = CLICKHOUSE_PORT
        self.db = CLICKHOUSE_DB
        self.user = CLICKHOUSE_USER
        self.password = CLICKHOUSE_PASSWORD
        
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Initialize schema
        self._init_schema()
        
        print(f"✅ ClickHouseBackend initialized ({self.host}:{self.port}/{self.db})")
    
    def _init_schema(self):
        """Create logs table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS logs (
            log_id UInt64,
            timestamp DateTime,
            level LowCardinality(String),
            component LowCardinality(String),
            log_source LowCardinality(String),
            event_template String,
            content String,
            backend String,
            routing_latency_ms Float32,
            write_latency_ms Float32,
            energy_joules Float32,
            success UInt8,
            cpu_percent Float32,
            memory_mb Float32,
            blockchain_hash String DEFAULT '',
            inserted_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, log_id)
        TTL timestamp + INTERVAL 30 DAY
        SETTINGS index_granularity = 8192;
        """
        
        try:
            self.query(create_table_sql)
        except Exception as e:
            print(f"⚠️  Schema init warning: {e}")
    
    def write(self, log_entry: Dict) -> Tuple[bool, float]:
        """
        Write a single log entry.
        
        Args:
            log_entry: Dictionary with log fields
        
        Returns:
            Tuple of (success, latency_ms)
        """
        start_time = time.time()
        
        try:
            # Parse timestamp
            ts = log_entry.get("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Build INSERT statement
            blockchain_hash = log_entry.get('blockchain_hash', '')
            
            insert_sql = """
            INSERT INTO logs (
                log_id, timestamp, level, component, log_source,
                event_template, content, backend,
                routing_latency_ms, write_latency_ms, energy_joules,
                success, cpu_percent, memory_mb, blockchain_hash
            ) VALUES
            """
            
            values = f"""(
                {log_entry.get('LogID', 0)},
                '{ts}',
                '{log_entry.get('Level', 'INFO')}',
                '{self._escape(log_entry.get('Component', 'unknown'))}',
                '{self._escape(log_entry.get('LogSource', 'unknown'))}',
                '{self._escape(log_entry.get('EventTemplate', ''))}',
                '{self._escape(log_entry.get('Content', ''))}',
                'clickhouse',
                0.0, 0.0, 0.0, 1, 0.0, 0.0,
                '{blockchain_hash}'
            )"""
            
            self.query(insert_sql + values)
            
            latency_ms = (time.time() - start_time) * 1000
            return (True, latency_ms)
        
        except Exception as e:
            print(f"❌ ClickHouse write error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return (False, latency_ms)
    
    def _escape(self, value: str) -> str:
        """Escape single quotes in SQL values."""
        if not isinstance(value, str):
            # Handle NaN/None values
            if value is None or (isinstance(value, float) and value != value):  # NaN check
                return "unknown"
            return str(value)
        return value.replace("'", "''")
    
    def query(self, sql: str) -> list:
        """
        Execute SQL query.
        
        Args:
            sql: SQL statement
        
        Returns:
            List of result rows (as dicts)
        """
        url = f"{self.base_url}/"
        params = {
            "query": sql,
            "database": self.db,
            "user": self.user,
            "password": self.password,
            "default_format": "JSONEachRow"
        }
        
        response = requests.post(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse NDJSON response
        if response.text.strip():
            return [eval(line) for line in response.text.strip().split('\n')]
        return []
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with stats
        """
        try:
            # Count logs
            count_result = self.query("SELECT count(*) as count FROM logs")
            total_logs = count_result[0]["count"] if count_result else 0
            
            # Get size
            size_result = self.query("""
                SELECT 
                    sum(bytes) as total_bytes,
                    sum(data_compressed_bytes) as compressed_bytes
                FROM system.parts
                WHERE table = 'logs' AND active
            """)
            
            if size_result and size_result[0]:
                total_bytes = size_result[0].get("total_bytes", 0) or 0
                compressed_bytes = size_result[0].get("compressed_bytes", 0) or 0
            else:
                total_bytes = 0
                compressed_bytes = 0
            
            return {
                "total_logs": total_logs,
                "total_size_mb": total_bytes / 1024 / 1024,
                "compressed_size_mb": compressed_bytes / 1024 / 1024
            }
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {"total_logs": 0, "total_size_mb": 0, "compressed_size_mb": 0}
    
    def health_check(self) -> bool:
        """
        Check if ClickHouse is accessible.
        
        Returns:
            True if healthy
        """
        try:
            response = requests.get(f"{self.base_url}/ping", timeout=5)
            return response.status_code == 200
        except:
            return False
