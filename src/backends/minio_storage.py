"""
MinIO/S3 Backend - Object storage for cold/archive logs.

Handles:
- Logs older than 30 days (cold storage)
- Compliance archives (immutable, versioned)
- Cost-effective long-term retention
- Parquet format (compressed, queryable)
"""

import io
import json
import time
from datetime import datetime
from typing import Dict, List
from minio import Minio
from minio.error import S3Error
import pandas as pd

from ..config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET,
    MINIO_SECURE,
)


class MinIOBackend:
    """
    MinIO/S3-compatible object storage for cold log archives.
    
    Features:
    - Immutable storage (versioned)
    - Parquet format (10:1 compression)
    - Lifecycle policies
    - S3-compatible API
    """
    
    def __init__(self):
        self.endpoint = MINIO_ENDPOINT
        self.access_key = MINIO_ACCESS_KEY
        self.secret_key = MINIO_SECRET_KEY
        self.bucket = MINIO_BUCKET
        
        # Initialize MinIO client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=MINIO_SECURE
        )
        
        # Initialize bucket
        self._init_bucket()
        
        # Buffer for batch writes
        self.buffer = []
        self.buffer_size = 1000  # Write every 1000 logs
    
    def _init_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                print(f"✅ MinIO bucket '{self.bucket}' created")
            else:
                print(f"✅ MinIO bucket '{self.bucket}' exists")
        except S3Error as e:
            print(f"MinIO bucket error: {e}")
    
    def write(self, log_entry: Dict) -> tuple[bool, float]:
        """
        Write a log entry to MinIO (buffered for efficiency).
        
        Returns:
            (success, latency_ms)
        """
        start_time = time.time()
        
        try:
            # Add to buffer
            self.buffer.append({
                "timestamp": datetime.now().isoformat(),
                "log_id": log_entry.get("LineId", ""),
                "level": log_entry.get("Level", ""),
                "component": log_entry.get("Component", ""),
                "log_source": log_entry.get("LogSource", ""),
                "event_template": log_entry.get("EventTemplate", ""),
                "content": log_entry.get("Content", ""),
                "backend": "minio",
                "routing_latency_ms": log_entry.get("routing_latency_ms", 0.0),
                "energy_joules": log_entry.get("energy_joules", 0.0),
                "blockchain_hash": log_entry.get("blockchain_hash", ""),
            })
            
            # Flush if buffer full
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
            
            latency_ms = (time.time() - start_time) * 1000
            return True, latency_ms
            
        except Exception as e:
            print(f"MinIO write error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return False, latency_ms
    
    def _flush_buffer(self):
        """Flush buffered logs to MinIO as Parquet file."""
        if not self.buffer:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.buffer)
            
            # Write as Parquet to bytes buffer
            buffer = io.BytesIO()
            df.to_parquet(buffer, engine='pyarrow', compression='snappy')
            buffer.seek(0)
            
            # Generate filename with timestamp
            filename = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            
            # Upload to MinIO
            self.client.put_object(
                self.bucket,
                filename,
                buffer,
                length=len(buffer.getvalue()),
                content_type='application/octet-stream'
            )
            
            print(f"✅ Flushed {len(self.buffer)} logs to MinIO: {filename}")
            
            # Clear buffer
            self.buffer = []
            
        except Exception as e:
            print(f"MinIO flush error: {e}")
    
    def flush(self):
        """Manually flush buffer (call at end of experiment)."""
        self._flush_buffer()
    
    def list_files(self) -> List[str]:
        """List all Parquet files in bucket."""
        try:
            objects = self.client.list_objects(self.bucket)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"MinIO list error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        try:
            files = self.list_files()
            total_size = 0
            
            for filename in files:
                stat = self.client.stat_object(self.bucket, filename)
                total_size += stat.size
            
            return {
                "total_files": len(files),
                "total_size_mb": total_size / (1024 * 1024),
                "total_size_human": f"{total_size / (1024**2):.2f} MB"
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {"total_files": 0, "total_size_mb": 0, "total_size_human": "0 MB"}
    
    def health_check(self) -> bool:
        """Check if MinIO is accessible."""
        try:
            return self.client.bucket_exists(self.bucket)
        except:
            return False
