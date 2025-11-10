"""
Configuration constants for the system.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
TRAINED_MODELS_DIR.mkdir(exist_ok=True)

# ClickHouse configuration
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_DB = "logs"
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = ""

# MinIO configuration (updated for port 9002)
MINIO_ENDPOINT = "localhost:9002"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "logs"
MINIO_SECURE = False  # Use HTTP (not HTTPS) for local testing

# Datasets
DATASETS = {
    "loghub": str(DATA_DIR / "Loghub-zenodo_Logs.csv"),
    "synthetic": str(DATA_DIR / "Synthetic_Datacenter_Logs.csv")
}

# Router types - XGBoost (primary) + Direct baselines (comparison)
ROUTER_TYPES = ["xgboost", "direct_clickhouse", "direct_minio"]

# XGBoost hyperparameters (Primary Production Router)
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "eval_metric": "logloss",
    "seed": 42
}

# XGBoost uses 6 features (NO semantic features - they add 34x latency!):
# 1. level_encoded, 2. component_hash, 3. log_source_hash
# 4. content_length, 5. has_error, 6. is_security
# Achieves 99.89% accuracy with 0.6ms latency

# Blockchain configuration (Selective logging for sensitive data)
# Hash is computed DURING routing (atomic with decision) for integrity
# Actual blockchain storage can be async to avoid blocking
BLOCKCHAIN_ENABLED = False  # Enable blockchain verification
BLOCKCHAIN_RPC_URL = "https://polygon-rpc.com"  # Polygon mainnet
BLOCKCHAIN_CONTRACT_ADDRESS = ""  # Deploy contract and set address
BLOCKCHAIN_PRIVATE_KEY = ""  # Private key for signing transactions
BLOCKCHAIN_GAS_LIMIT = 100000  # Max gas per transaction
BLOCKCHAIN_SIMULATION_MODE = True  # True = simulate, False = real transactions

# Alternative networks
BLOCKCHAIN_NETWORKS = {
    "polygon_mainnet": "https://polygon-rpc.com",
    "polygon_testnet": "https://rpc-mumbai.maticvigil.com",
    "ethereum_mainnet": "https://eth.llamarpc.com",
    "ethereum_sepolia": "https://rpc.sepolia.org",
    "local": "http://127.0.0.1:8545"
}

# Sensitivity Detection Configuration (for blockchain logging)
# Used by BlockchainLogger to determine which logs need immutable audit trail
SENSITIVITY_METHOD = "weighted"  # Options: "weighted", "ml", "hybrid"
SENSITIVITY_THRESHOLD = 0.5  # Score threshold (0.0-1.0)

# ML Sensitivity Detector (OPTIONAL - for background analysis, NOT routing)
# ⚠️  DO NOT use DistilBERT in routing path - it adds 34x latency!
# Use for: Background analysis, alert generation, anomaly detection
ML_DETECTOR_ENABLED = False  # Enable ML-based detection
ML_MODEL_TYPE = "distilbert"  # Options: "distilbert", "tinybert", "sklearn"
ML_MODEL_PATH = None  # Path to custom trained model (None = use default)
ML_CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for ML predictions (0.0-1.0)

# Model details (for background analysis pipeline, not routing):
# - distilbert: 66MB, ~20ms/prediction, zero-shot (no training needed)
# - tinybert: 14MB, ~5ms/prediction, fine-tunable
# - sklearn: <1MB, <1ms/prediction, requires training data
