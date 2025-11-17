"""
XGBoost Router - Intelligent routing using gradient boosting.

The core intelligent router that learns optimal routing decisions
based on log characteristics and system state.
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


class XGBoostRouter(BaseRouter):
    """
    XGBoost-based intelligent log router.
    
    Features:
    - Learns from historical routing data
    - Predicts optimal backend (hot vs cold storage)
    - Fast inference (~2ms per prediction)
    - Explainable decisions (feature importance)
    
    Training creates binary classifier:
    - Class 0: clickhouse (hot storage)
    - Class 1: minio (cold storage)
    """
    
    def __init__(self, model_path: str = "xgboost_router", blockchain_logger=None):
        """
        Initialize XGBoost router.
        
        Args:
            model_path: Path to trained model (without extension), or None for fallback
            blockchain_logger: Optional BlockchainLogger for sensitive log verification
        """
        self.model_path = None
        if model_path:
            if not os.path.isabs(model_path):
                self.model_path = str(TRAINED_MODELS_DIR / f"{model_path}.json")
            else:
                self.model_path = model_path
        
        self.blockchain_logger = blockchain_logger
        self.blockchain_count = 0
        self.total_logs = 0
        
        # Load model
        self.model = None
        self.feature_encoders = None
        if self.model_path:
            self._load_model()
    
    def _load_model(self):
        """Load trained XGBoost model and encoders."""
        if not XGBOOST_AVAILABLE:
            print("⚠️  XGBoost not available. Using fallback routing.")
            self.model = None
            return
        
        try:
            # Load XGBoost model
            if os.path.exists(self.model_path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                print(f"✅ Loaded XGBoost model from {self.model_path}")
            else:
                print(f"⚠️  XGBoost model not found: {self.model_path}")
                print("   Using fallback routing")
                self.model = None
            
            # Load encoders (for categorical features)
            encoder_path = self.model_path.replace('.json', '_encoders.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.feature_encoders = pickle.load(f)
                print(f"✅ Loaded feature encoders")
            else:
                self.feature_encoders = None
                
        except Exception as e:
            print(f"❌ Error loading XGBoost model: {e}")
            self.model = None
    
    def _extract_features(self, log_entry: Dict) -> np.ndarray:
        """
        Extract features from log entry for XGBoost prediction.
        
        Features:
        1. level_encoded: Error severity (0=info, 1=warn, 2=error, 3=critical)
        2. component_hash: Hash of component name (bucketed)
        3. log_source_hash: Hash of log source (bucketed)
        4. content_length: Length of log content
        5. has_error_keywords: Binary flag for error-related terms
        6. is_security: Binary flag for security-related terms
        
        Returns:
            Feature vector as numpy array [6 features]
        """
        # Level encoding
        level_map = {
            'debug': 0, 'info': 0, 'notice': 0,
            'warn': 1, 'warning': 1,
            'error': 2, 'err': 2,
            'crit': 3, 'critical': 3, 'alert': 3, 'emerg': 3
        }
        level = log_entry.get('Level', 'info').lower()
        level_encoded = level_map.get(level, 0)
        
        # Component hash (bucketed to 100 values)
        component = log_entry.get('Component', '')
        component_hash = hash(component) % 100
        
        # Log source hash (bucketed to 100 values)
        log_source = log_entry.get('LogSource', '')
        log_source_hash = hash(log_source) % 100
        
        # Content length
        content = log_entry.get('Content', '')
        content_length = len(content)
        
        # Error keywords
        error_keywords = ['error', 'fail', 'denied', 'reject', 'timeout', 'exception']
        has_error_keywords = any(kw in content.lower() for kw in error_keywords)
        
        # Security keywords
        security_keywords = ['ssh', 'login', 'password', 'auth', 'permission', 'access', 'security']
        is_security = any(kw in content.lower() for kw in security_keywords)
        
        return np.array([
            level_encoded,
            component_hash,
            log_source_hash,
            content_length,
            int(has_error_keywords),
            int(is_security)
        ]).reshape(1, -1)
    
    def get_route(self, log_entry: Dict) -> str:
        """
        Get optimal routing decision for this log.
        
        Returns 'clickhouse' or 'minio'.
        Also handles blockchain verification for sensitive logs.
        """
        self.total_logs += 1
        
        if self.blockchain_logger and self.blockchain_logger.enabled:
            if self.blockchain_logger.is_sensitive(log_entry):
                tx_hash = self.blockchain_logger.store_hash(log_entry, backend="hybrid")
                if tx_hash:
                    log_entry['blockchain_hash'] = tx_hash
                    self.blockchain_count += 1
        
        # If model not loaded, use fallback
        if self.model is None:
            return self._fallback_routing(log_entry)
        
        try:
            # Extract features
            features = self._extract_features(log_entry)
            
            # Predict with XGBoost
            prediction = self.model.predict(features)[0]
            
            # Binary classification: 0 = clickhouse, 1 = minio
            return "minio" if prediction == 1 else "clickhouse"
            
        except Exception as e:
            print(f"XGBoost prediction error: {e}")
            return self._fallback_routing(log_entry)
    
    def _fallback_routing(self, log_entry: Dict) -> str:
        """
        Fallback routing when model is unavailable.
        
        Strategy:
        - Security logs → minio (long-term archive)
        - Error/critical logs → clickhouse (hot for debugging)
        - Default → clickhouse (hot storage)
        """
        level = str(log_entry.get('Level', 'info')).lower()
        content = str(log_entry.get('Content', '')).lower()
        component = str(log_entry.get('Component', '')).lower()
        
        # Security-critical → cold archive
        security_keywords = ['ssh', 'login', 'password', 'auth', 'kernel', 'security']
        if any(kw in content or kw in component for kw in security_keywords):
            return "minio"
        
        # Errors → hot for quick debugging
        if level in ['error', 'err', 'crit', 'critical', 'alert', 'emerg']:
            return "clickhouse"
        
        # Default: hot storage
        return "clickhouse"
    
    def get_prediction_confidence(self, log_entry: Dict) -> float:
        """
        Get confidence score for prediction (if model loaded).
        
        Returns:
            Probability of predicted class (0.5 to 1.0)
        """
        if self.model is None:
            return 0.5  # No confidence when using fallback
        
        try:
            features = self._extract_features(log_entry)
            probabilities = self.model.predict_proba(features)[0]
            return float(max(probabilities))
        except:
            return 0.5
    
    def get_stats(self) -> Dict:
        """
        Get router statistics including blockchain usage.
        
        Returns:
            Dictionary with routing and blockchain stats
        """
        stats = {
            'total_logs': self.total_logs,
            'blockchain_enabled': self.blockchain_logger is not None and self.blockchain_logger.enabled,
            'blockchain_logs': self.blockchain_count
        }
        
        if stats['blockchain_enabled'] and self.total_logs > 0:
            stats['blockchain_percentage'] = (self.blockchain_count / self.total_logs) * 100
        
        return stats
