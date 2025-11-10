"""
Feature extractor for intelligent log routing.

Converts raw log entries into ML-ready feature vectors.
"""

import hashlib
from typing import Dict, List
import re


class FeatureExtractor:
    """
    Extract features from log entries for ML models.
    
    Features extracted:
    - level_encoded: Numeric severity (0=INFO, 1=WARN, 2=ERROR, 3=CRITICAL)
    - component_hash: Bucketed hash of log component (0-99)
    - log_source_hash: Bucketed hash of log source (0-99)
    - content_length: Character count of log content
    - has_error_keywords: Binary flag for error-related keywords
    - is_security: Binary flag for security-related keywords
    """
    
    ERROR_KEYWORDS = [
        "error", "exception", "failed", "failure", "fatal", 
        "crash", "abort", "panic", "segfault", "timeout"
    ]
    
    SECURITY_KEYWORDS = [
        "authentication", "authorization", "password", "token",
        "certificate", "encryption", "decrypt", "ssh", "ssl",
        "tls", "credential", "privilege", "permission", "access denied"
    ]
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract(self, log_entry: Dict) -> List[float]:
        """
        Extract feature vector from log entry.
        
        Args:
            log_entry: Dictionary with keys like:
                - Level: severity level
                - Component: log component
                - LogSource: log source
                - Content: log message
                - EventTemplate: optional template
        
        Returns:
            Feature vector [level, component_hash, source_hash, length, has_error, is_security]
        """
        # Feature 1: Level encoding
        level = log_entry.get("Level", "INFO").upper()
        level_map = {"INFO": 0, "WARN": 1, "WARNING": 1, "ERROR": 2, "CRITICAL": 3, "FATAL": 3}
        level_encoded = level_map.get(level, 0)
        
        # Feature 2: Component hash (bucketed to 0-99)
        component = log_entry.get("Component", "unknown")
        component_hash = int(hashlib.md5(component.encode()).hexdigest(), 16) % 100
        
        # Feature 3: Log source hash (bucketed to 0-99)
        log_source = log_entry.get("LogSource", "unknown")
        log_source_hash = int(hashlib.md5(log_source.encode()).hexdigest(), 16) % 100
        
        # Feature 4: Content length
        content = log_entry.get("Content", "")
        content_length = len(content)
        
        # Feature 5: Has error keywords
        content_lower = content.lower()
        has_error = int(any(kw in content_lower for kw in self.ERROR_KEYWORDS))
        
        # Feature 6: Is security-related
        is_security = int(any(kw in content_lower for kw in self.SECURITY_KEYWORDS))
        
        return [
            float(level_encoded),
            float(component_hash),
            float(log_source_hash),
            float(content_length),
            float(has_error),
            float(is_security)
        ]
    
    def extract_batch(self, log_entries: List[Dict]) -> List[List[float]]:
        """
        Extract features for multiple logs (batch processing).
        
        Args:
            log_entries: List of log dictionaries
        
        Returns:
            List of feature vectors
        """
        return [self.extract(log) for log in log_entries]
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names in order.
        
        Returns:
            List of feature names
        """
        return [
            "level_encoded",
            "component_hash",
            "log_source_hash",
            "content_length",
            "has_error_keywords",
            "is_security"
        ]
