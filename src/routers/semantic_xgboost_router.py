"""
Semantic XGBoost Router - Intelligent routing using gradient boosting with semantic features.

This router uses DistilBERT embeddings plus contextual features for intelligent routing decisions.
"""

import os
import pickle
from typing import Dict
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import numpy as np

from .base import BaseRouter
from ..config import TRAINED_MODELS_DIR
from ..features import EnhancedFeatureExtractor


class SemanticXGBoostRouter(BaseRouter):
    """
    Semantic XGBoost-based intelligent log router.
    
    Features (778 total):
    - 768-dim DistilBERT semantic embeddings
    - Structural features (level, component, source, length)
    - Content analysis (error severity, security risk)
    - Temporal features (time of day, day of week)
    - Contextual features (error rate, frequency)
    
    Training creates binary classifier:
    - Class 0: clickhouse (hot storage)
    - Class 1: minio (cold storage)
    """
    
    def __init__(
        self, 
        model_path: str = "xgboost_semantic_router",
        enable_semantic: bool = True,
        blockchain_logger=None
    ):
        """
        Initialize Semantic XGBoost router.
        
        Args:
            model_path: Path to trained model (without extension)
            enable_semantic: Enable DistilBERT semantic features (778 features vs 10)
            blockchain_logger: Optional BlockchainLogger for sensitive log verification
        """
        self.model_path = None
        if model_path:
            if not os.path.isabs(model_path):
                self.model_path = str(TRAINED_MODELS_DIR / f"{model_path}.json")
            else:
                self.model_path = model_path
        
        self.enable_semantic = enable_semantic
        self.blockchain_logger = blockchain_logger
        self.blockchain_count = 0
        self.total_logs = 0
        
        # Initialize feature extractor
        print(f"🔧 Initializing {'semantic' if enable_semantic else 'basic'} feature extractor...")
        self.feature_extractor = EnhancedFeatureExtractor(enable_semantic=enable_semantic)
        
        # Load model
        self.model = None
        if self.model_path:
            self._load_model()
        else:
            print("⚠️  No model path provided. Using fallback routing.")
    
    def _load_model(self):
        """Load trained XGBoost model."""
        if not XGBOOST_AVAILABLE:
            print("⚠️  XGBoost not available. Using fallback routing.")
            self.model = None
            return
        
        try:
            # Load XGBoost model
            if os.path.exists(self.model_path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                print(f"✅ Loaded semantic XGBoost model from {self.model_path}")
                
                # Verify feature dimensions
                expected_features = 778 if self.enable_semantic else 10
                if hasattr(self.model, 'n_features_in_'):
                    actual_features = self.model.n_features_in_
                    if actual_features != expected_features:
                        print(f"⚠️  Feature dimension mismatch: model expects {actual_features}, "
                              f"but extractor provides {expected_features}")
                        print(f"   Falling back to default routing")
                        self.model = None
            else:
                print(f"⚠️  Model not found: {self.model_path}")
                print("   Using fallback routing")
                self.model = None
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def get_route(self, log_entry: Dict) -> str:
        """
        Get optimal routing decision for this log using semantic features.
        
        Args:
            log_entry: Dict with keys: Level, Component, LogSource, Content, Timestamp
            
        Returns:
            'clickhouse' (hot) or 'minio' (cold)
        """
        self.total_logs += 1
        
        # Handle blockchain asynchronously (non-blocking)
        if self.blockchain_logger and self.blockchain_logger.enabled:
            if self.blockchain_logger.is_sensitive(log_entry):
                # Store hash asynchronously after routing decision
                # (actual async implementation would use threading/asyncio)
                # For now, we mark it but don't block
                log_entry['_needs_blockchain'] = True
        
        # If model not loaded, use fallback
        if self.model is None:
            return self._fallback_routing(log_entry)
        
        try:
            # Extract semantic features (778-dim or 10-dim)
            features = self.feature_extractor.extract_features(log_entry)
            features = features.reshape(1, -1)
            
            # Predict with XGBoost
            prediction = self.model.predict(features)[0]
            
            # Binary classification: 0 = clickhouse, 1 = minio
            backend = "minio" if prediction == 1 else "clickhouse"
            
            # Handle blockchain storage after routing (async in production)
            if log_entry.get('_needs_blockchain'):
                tx_hash = self.blockchain_logger.store_hash(log_entry, backend)
                if tx_hash:
                    log_entry['blockchain_hash'] = tx_hash
                    self.blockchain_count += 1
            
            return backend
            
        except Exception as e:
            print(f"❌ XGBoost prediction error: {e}")
            return self._fallback_routing(log_entry)
    
    def _fallback_routing(self, log_entry: Dict) -> str:
        """
        Fallback routing when model is unavailable.
        
        Strategy:
        - Security logs → minio (long-term archive)
        - Error/critical logs → clickhouse (hot for debugging)
        - High frequency → clickhouse (hot for analytics)
        - Default → clickhouse (hot storage)
        """
        level = str(log_entry.get('Level', 'info')).lower()
        content = str(log_entry.get('Content', '')).lower()
        component = str(log_entry.get('Component', '')).lower()
        
        # Security-critical → cold archive
        security_keywords = ['ssh', 'login', 'password', 'auth', 'kernel', 'security', 
                           'authentication', 'unauthorized', 'breach']
        if any(kw in content or kw in component for kw in security_keywords):
            return "minio"
        
        # Errors → hot for quick debugging
        if level in ['error', 'err', 'crit', 'critical', 'alert', 'emerg', 'fatal']:
            return "clickhouse"
        
        # Default: hot storage (most logs benefit from fast access)
        return "clickhouse"
    
    def get_prediction_confidence(self, log_entry: Dict) -> float:
        """
        Get confidence score for prediction.
        
        Returns:
            Probability of predicted class (0.5 to 1.0)
        """
        if self.model is None:
            return 0.5  # No confidence when using fallback
        
        try:
            features = self.feature_extractor.extract_features(log_entry)
            features = features.reshape(1, -1)
            probabilities = self.model.predict_proba(features)[0]
            return float(max(probabilities))
        except:
            return 0.5
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from trained model.
        
        Returns:
            Dict mapping feature names to importance scores
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        try:
            feature_names = self.feature_extractor.get_feature_names()
            importances = self.model.feature_importances_
            
            # Sort by importance
            feature_importance = dict(zip(feature_names, importances))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Return top 20 features
            return dict(sorted_features[:20])
        except:
            return {}
    
    def get_stats(self) -> Dict:
        """
        Get router statistics including feature extraction and blockchain usage.
        
        Returns:
            Dictionary with routing, feature, and blockchain stats
        """
        stats = {
            'total_logs': self.total_logs,
            'model_loaded': self.model is not None,
            'semantic_enabled': self.enable_semantic,
            'blockchain_enabled': self.blockchain_logger is not None and self.blockchain_logger.enabled,
            'blockchain_logs': self.blockchain_count
        }
        
        # Add feature extractor stats
        extractor_stats = self.feature_extractor.get_stats()
        stats['feature_extractor'] = extractor_stats
        
        # Add blockchain percentage
        if stats['blockchain_enabled'] and self.total_logs > 0:
            stats['blockchain_percentage'] = round((self.blockchain_count / self.total_logs) * 100, 2)
        
        return stats
    
    def benchmark_latency(self, log_entry: Dict, iterations: int = 100) -> Dict:
        """
        Benchmark routing latency.
        
        Args:
            log_entry: Sample log entry to route
            iterations: Number of iterations for averaging
            
        Returns:
            Dict with latency statistics (ms)
        """
        import time
        
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.get_route(log_entry)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean_ms': round(np.mean(latencies), 3),
            'median_ms': round(np.median(latencies), 3),
            'min_ms': round(np.min(latencies), 3),
            'max_ms': round(np.max(latencies), 3),
            'std_ms': round(np.std(latencies), 3),
            'p95_ms': round(np.percentile(latencies, 95), 3),
            'p99_ms': round(np.percentile(latencies, 99), 3)
        }
