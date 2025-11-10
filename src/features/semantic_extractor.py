"""
Semantic Feature Extractor using DistilBERT for log content understanding.

This module extracts 768-dimensional embeddings from log content to enable
semantic understanding of log messages.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available - semantic features disabled")


class SemanticFeatureExtractor:
    """Extract semantic embeddings from log content using DistilBERT."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", cache_size: int = 1000):
        """
        Initialize semantic extractor.
        
        Args:
            model_name: HuggingFace model identifier
            cache_size: Number of embeddings to cache (for repeated logs)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library required for semantic features")
        
        logger.info(f"Loading {model_name} for semantic feature extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Inference mode
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Simple LRU cache for repeated log patterns
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Semantic extractor ready (device: {self.device})")
    
    def extract_embedding(self, text: str) -> np.ndarray:
        """
        Extract 768-dim embedding from text.
        
        Args:
            text: Log content to encode
            
        Returns:
            768-dimensional numpy array
        """
        # Check cache first
        if text in self.cache:
            self.cache_hits += 1
            return self.cache[text]
        
        self.cache_misses += 1
        
        # Tokenize and encode
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # Cache management (simple: evict random if full)
        if len(self.cache) >= self.cache_size:
            # Remove random item
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[text] = embedding
        return embedding
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2)
        }


class EnhancedFeatureExtractor:
    """
    Extract comprehensive features for XGBoost routing including:
    - Semantic embeddings (768-dim from DistilBERT)
    - Temporal features (time_of_day, day_of_week)
    - Contextual features (error_rate, frequency)
    - Structural features (level, component, source)
    """
    
    def __init__(self, enable_semantic: bool = True):
        """
        Initialize enhanced feature extractor.
        
        Args:
            enable_semantic: Whether to enable DistilBERT semantic features
        """
        self.enable_semantic = enable_semantic
        self.semantic_extractor = None
        
        if enable_semantic:
            try:
                self.semantic_extractor = SemanticFeatureExtractor()
            except Exception as e:
                logger.warning(f"Failed to initialize semantic extractor: {e}")
                self.enable_semantic = False
        
        # Historical tracking for contextual features
        self.component_error_counts = {}
        self.component_total_counts = {}
        self.log_frequency_window = []
        self.window_size = 100  # Track last 100 logs for frequency
        
        logger.info(f"Enhanced feature extractor initialized (semantic: {self.enable_semantic})")
    
    def extract_features(self, log_entry: Dict) -> np.ndarray:
        """
        Extract all features from log entry.
        
        Args:
            log_entry: Dict with keys: Level, Component, LogSource, Content, Timestamp
            
        Returns:
            Feature vector (774-dim if semantic enabled, 6-dim otherwise)
        """
        features = []
        
        # 1. SEMANTIC FEATURES (768-dim)
        if self.enable_semantic and self.semantic_extractor:
            content = str(log_entry.get("Content", ""))
            embedding = self.semantic_extractor.extract_embedding(content)
            features.extend(embedding.tolist())
        
        # 2. STRUCTURAL FEATURES
        level_map = {"INFO": 0, "WARN": 1, "WARNING": 1, "ERROR": 2, "CRITICAL": 3, "FATAL": 3}
        level = str(log_entry.get("Level", "INFO")).upper()
        level_encoded = level_map.get(level, 0)
        features.append(level_encoded)
        
        # Component category (clustered, not just hashed)
        component = str(log_entry.get("Component", "unknown"))
        component_category = self._categorize_component(component)
        features.append(component_category)
        
        # Log source hash (keep this simple)
        log_source = str(log_entry.get("LogSource", "unknown"))
        source_hash = hash(log_source) % 100
        features.append(source_hash)
        
        # Content length
        content_length = len(str(log_entry.get("Content", "")))
        features.append(content_length)
        
        # 3. CONTENT ANALYSIS FEATURES (ML-based, not just keywords)
        error_severity = self._compute_error_severity(log_entry)
        features.append(error_severity)
        
        security_risk = self._compute_security_risk(log_entry)
        features.append(security_risk)
        
        # 4. TEMPORAL FEATURES
        if "Timestamp" in log_entry:
            time_features = self._extract_temporal_features(log_entry["Timestamp"])
            features.extend(time_features)
        else:
            features.extend([0, 0])  # time_of_day, day_of_week
        
        # 5. CONTEXTUAL/HISTORICAL FEATURES
        component_error_rate = self._get_component_error_rate(component, level_encoded >= 2)
        features.append(component_error_rate)
        
        recent_frequency = self._get_recent_frequency(component)
        features.append(recent_frequency)
        
        return np.array(features, dtype=np.float32)
    
    def _categorize_component(self, component: str) -> int:
        """
        Categorize component into meaningful groups.
        
        Categories:
        0: System (kernel, systemd, etc.)
        1: Network (sshd, network, etc.)
        2: Application (custom apps)
        3: Database (mysql, postgres, etc.)
        4: Web (nginx, apache, etc.)
        5: Unknown
        """
        # Handle NaN/None/float values
        if not isinstance(component, str):
            return 5
        
        component_lower = component.lower()
        
        if any(x in component_lower for x in ["kernel", "systemd", "init"]):
            return 0
        elif any(x in component_lower for x in ["ssh", "network", "net", "tcp"]):
            return 1
        elif any(x in component_lower for x in ["mysql", "postgres", "mongo", "redis"]):
            return 3
        elif any(x in component_lower for x in ["nginx", "apache", "http"]):
            return 4
        elif component == "unknown":
            return 5
        else:
            return 2  # Application
    
    def _compute_error_severity(self, log_entry: Dict) -> float:
        """
        Compute error severity score (0.0-1.0) based on content analysis.
        More sophisticated than simple keyword matching.
        """
        content = log_entry.get("Content", "").lower()
        level = log_entry.get("Level", "INFO").upper()
        
        # Base score from level
        level_scores = {"INFO": 0.0, "WARN": 0.3, "WARNING": 0.3, "ERROR": 0.7, "CRITICAL": 1.0, "FATAL": 1.0}
        score = level_scores.get(level, 0.0)
        
        # Increase score based on error keywords (weighted)
        high_severity_keywords = ["fatal", "critical", "panic", "segfault", "crash"]
        medium_severity_keywords = ["error", "fail", "exception", "timeout", "denied"]
        
        for keyword in high_severity_keywords:
            if keyword in content:
                score = min(1.0, score + 0.3)
        
        for keyword in medium_severity_keywords:
            if keyword in content:
                score = min(1.0, score + 0.1)
        
        return score
    
    def _compute_security_risk(self, log_entry: Dict) -> float:
        """
        Compute security risk score (0.0-1.0) based on content analysis.
        """
        content = log_entry.get("Content", "").lower()
        
        score = 0.0
        
        # High-risk security keywords
        high_risk = ["authentication failed", "unauthorized", "intrusion", "breach", "attack", "malicious"]
        medium_risk = ["auth", "permission", "denied", "invalid credentials", "security"]
        
        for keyword in high_risk:
            if keyword in content:
                score = min(1.0, score + 0.4)
        
        for keyword in medium_risk:
            if keyword in content:
                score = min(1.0, score + 0.2)
        
        return score
    
    def _extract_temporal_features(self, timestamp) -> List[float]:
        """
        Extract temporal features from timestamp.
        
        Returns:
            [time_of_day (0-23), day_of_week (0-6)]
        """
        from datetime import datetime
        
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp)
            else:
                dt = timestamp
            
            time_of_day = dt.hour
            day_of_week = dt.weekday()
            
            return [float(time_of_day), float(day_of_week)]
        except Exception:
            return [0.0, 0.0]
    
    def _get_component_error_rate(self, component: str, is_error: bool) -> float:
        """
        Track and return error rate for this component.
        
        Args:
            component: Component name
            is_error: Whether current log is an error
            
        Returns:
            Error rate (0.0-1.0)
        """
        # Update counts
        self.component_total_counts[component] = self.component_total_counts.get(component, 0) + 1
        if is_error:
            self.component_error_counts[component] = self.component_error_counts.get(component, 0) + 1
        
        # Calculate rate
        total = self.component_total_counts.get(component, 1)
        errors = self.component_error_counts.get(component, 0)
        
        return errors / total if total > 0 else 0.0
    
    def _get_recent_frequency(self, component: str) -> float:
        """
        Get recent frequency of logs from this component.
        
        Returns:
            Frequency (0.0-1.0) - proportion in recent window
        """
        # Add to window
        self.log_frequency_window.append(component)
        
        # Maintain window size
        if len(self.log_frequency_window) > self.window_size:
            self.log_frequency_window.pop(0)
        
        # Calculate frequency
        count = self.log_frequency_window.count(component)
        return count / len(self.log_frequency_window) if self.log_frequency_window else 0.0
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability."""
        names = []
        
        if self.enable_semantic:
            names.extend([f"embedding_{i}" for i in range(768)])
        
        names.extend([
            "level_encoded",
            "component_category",
            "log_source_hash",
            "content_length",
            "error_severity_score",
            "security_risk_score",
            "time_of_day",
            "day_of_week",
            "component_error_rate",
            "recent_frequency"
        ])
        
        return names
    
    def get_stats(self) -> Dict:
        """Get extractor statistics."""
        stats = {
            "semantic_enabled": self.enable_semantic,
            "feature_dim": 778 if self.enable_semantic else 10,
            "components_tracked": len(self.component_total_counts),
            "frequency_window_size": len(self.log_frequency_window)
        }
        
        if self.enable_semantic and self.semantic_extractor:
            stats["semantic_cache"] = self.semantic_extractor.get_cache_stats()
        
        return stats
