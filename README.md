# Semantic ML Routing for Green Cloud Computing

**Hybrid Log Management System with Intelligent Semantic Routing**

This system uses **machine learning with semantic understanding** to intelligently route logs to hot (ClickHouse) or cold (MinIO) storage, optimizing for both **performance** and **energy efficiency**.

## 🎯 Core Innovation

**XGBoost + DistilBERT Semantic Features**: Our intelligent routing system uses:
- **768-dimensional semantic embeddings** from DistilBERT to understand log content
- **Temporal features** (time of day, day of week) for pattern detection
- **Contextual features** (component error rate, log frequency) for adaptive routing
- **Fast inference** (~5-8ms per log) with semantic understanding

**Result**: ~95% accuracy in routing decisions with semantic understanding.

## 📊 System Architecture

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  Log Entry  │────▶│  EnhancedFeatureExtractor            │
└─────────────┘     │  - DistilBERT Embeddings (768-dim)   │
                    │  - Temporal Features (2-dim)          │
                    │  - Contextual Features (2-dim)        │
                    │  - Structural Features (6-dim)        │
                    └──────────────────┬───────────────────┘
                                       │ 778-dim vector
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  SemanticXGBoostRouter               │
                    │  - Trained XGBoost Classifier        │
                    │  - Binary: Hot (0) vs Cold (1)       │
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       │ Backend decision
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  Storage Backend                     │
                    │  - ClickHouse (Hot) → Fast queries   │
                    │  - MinIO (Cold) → Long-term archive  │
                    └──────────────────┬───────────────────┘
                                       │ (async, non-blocking)
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  BlockchainLogger                    │
                    │  - Blockchain verification           │
                    │  - Only for sensitive logs           │
                    │  - Runs asynchronously               │
                    └──────────────────────────────────────┘
```

## 🔬 How It Works

## 🔗 Blockchain Integration

Asynchronous blockchain verification for sensitive logs is a core component of the system. The default setup for development uses a local Ganache instance; this provides deterministic accounts and instant blocks so experiments are reproducible and cheap.

- Local (recommended for experiments): Ganache on port 8545
- Production/Testnet: Any EVM-compatible RPC (set via `POLYGON_RPC_URL`)

Important notes:
- Do NOT store private keys in repository files. `run_all_experiments.sh` now reads `BLOCKCHAIN_PRIVATE_KEY` from the environment.
- A local draft with full deployment steps is available in `BLOCKCHAIN_EVAL_DRAFT.md` (this file is intentionally ignored by Git).


### Semantic ML Routing
```python
# Extract semantic features from log content
embedding = distilbert(log.content)  # 768-dim semantic understanding
features = [embedding, temporal, contextual, structural]  # 778-dim total

# Intelligent routing decision
backend = xgboost.predict(features)  # Learned optimal routing
```
**Key Benefits**: 
- Semantic understanding of log content
- Adaptive to evolving log patterns
- ~95% routing accuracy
- ~5-8ms inference time

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies (including semantic features)
pip install -r requirements.txt
```

### 2. Train Semantic Router

```bash
# Train with semantic features (DistilBERT embeddings)
python -m src.training.train_semantic_xgboost \
    --data data/Loghub-zenodo_Logs.csv \
    --output xgboost_semantic_router \
    --semantic \
    --balance \
    --test_split 0.2
```

**Expected output**: 
- Training time: ~10-30 minutes (depending on dataset size)
- Model accuracy: ~95%+ 
- Feature dimension: 778
- Model file: `trained_models/xgboost_semantic_router.json`

### 3. Test Feature Extraction

```bash
# Test semantic feature extractor
python test_semantic_features.py
```

This will:
- Initialize DistilBERT extractor
- Extract features from sample logs
- Benchmark latency (~5-8ms expected)
- Test cache performance

### 4. Run Routing Experiment

```bash
# Route logs using semantic router
python -m src.experiment \
    --log_filepath data/Loghub-zenodo_Logs.csv \
    --routers xgboost_semantic \
    --limit 1000 \
    --output_dir results/semantic_routing
```

## 📁 Project Structure

```
hybrid-log-management/
├── src/
│   ├── features/
│   │   ├── semantic_extractor.py      # DistilBERT feature extraction
│   │   └── extractor.py                # Basic feature extraction
│   ├── routers/
│   │   ├── semantic_xgboost_router.py  # Semantic XGBoost router (PRIMARY)
│   │   ├── xgboost_router.py           # Basic XGBoost router (legacy)
│   │   └── direct_router.py            # Direct baselines
│   ├── backends/
│   │   ├── clickhouse.py               # Hot storage backend
│   │   └── minio_storage.py            # Cold storage backend
│   ├── training/
│   │   └── train_semantic_xgboost.py   # Training script
│   └── blockchain_logger.py            # Blockchain verification
├── data/
│   ├── Loghub-zenodo_Logs.csv          # Real-world logs
│   └── Synthetic_Datacenter_Logs.csv   # Synthetic logs
├── trained_models/                     # Trained XGBoost models
├── results/                            # Experiment results
├── requirements.txt                    # Dependencies
└── test_semantic_features.py           # Feature extraction tests
```

## 🧪 Automated Experiments Framework

### Quick Start - Run All Experiments

**Option 1: Using Shell Script (Recommended)**
```bash
# Run all thesis experiments (automated)
./run_experiments.sh --all

# Quick test mode (1000 logs, ~10-15 minutes)
./run_experiments.sh --all --quick

# Run specific research question
./run_experiments.sh --rq1  # Basic vs Semantic features
./run_experiments.sh --rq2  # XGBoost accuracy analysis
./run_experiments.sh --rq3  # ML vs baseline comparison
./run_experiments.sh --rq4  # Blockchain overhead
```

**Option 2: Using Main CLI (Integrated)**
```bash
# Run all experiments through main CLI
python -m src --run-experiments

# Quick test mode
python -m src --run-experiments --experiments-quick

# Run specific research question
python -m src --run-experiments --experiments-mode rq1
python -m src --run-experiments --experiments-mode rq2
python -m src --run-experiments --experiments-mode rq3
python -m src --run-experiments --experiments-mode rq4

# Custom output directory
python -m src --run-experiments --experiments-output results/my_experiment
```

## 🔥 Key Features

### 1. Semantic Understanding with DistilBERT
- **768-dimensional embeddings** capture log semantics
- Understands similar logs even with different wording
- Cache mechanism for repeated patterns (~50% hit rate)

### 2. Contextual & Temporal Features
- **Error rate tracking** per component (adaptive to patterns)
- **Log frequency** analysis (detects bursts)
- **Time-based features** (hour of day, day of week)

### 3. Asynchronous Blockchain Verification
- **Non-blocking** blockchain logging for sensitive logs
- Polygon blockchain for immutable audit trail
- Only ~5-10% of logs require blockchain (selective)

### 4. Green Computing Focus
- Optimized for **energy efficiency** (reduce hot storage usage)
- **Cost optimization** (cold storage 10x cheaper)
- **Performance maintained** (~5-8ms routing overhead)

## 📈 Performance Targets

- **Routing Latency**: <10ms per log (target: 5-8ms)
- **Accuracy**: >90% routing correctness (target: ~95%)
- **Energy Savings**: 30-50% vs all-hot storage
- **Cost Savings**: 40-60% vs all-hot storage

## 🎓 Thesis Contribution

**Title**: *"Semantic Machine Learning for Green Cloud Log Management: A Hybrid Hot-Cold Storage Approach"*

**Key Contributions**:
1. **Semantic feature engineering** for log routing (DistilBERT + contextual)
2. **XGBoost-based intelligent routing** with <10ms latency
3. **Energy-aware storage optimization** (hot vs cold)
4. **Asynchronous blockchain** verification for sensitive logs

**What We're NOT claiming**:
- ❌ We're not using LLMs for routing (too slow, too expensive)
- ❌ We're not doing real-time blockchain (asynchronous only)
- ❌ We're not replacing human judgment (ML-assisted decisions)

## 🔧 Configuration

Key settings in `src/config.py`:

```python
# Router types available
ROUTER_TYPES = ["xgboost", "xgboost_semantic", "direct_clickhouse", "direct_minio"]

# Backends
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123

MINIO_ENDPOINT = "localhost:9002"
MINIO_BUCKET = "logs"

# Blockchain (optional)
BLOCKCHAIN_ENABLED = False  # Enable for sensitive log verification
BLOCKCHAIN_SIMULATION_MODE = True  # Simulation mode for testing
```

## 📚 Dependencies

**Core**: 
- `xgboost>=2.0.0` - Gradient boosting classifier
- `transformers>=4.30.0` - DistilBERT for semantic embeddings
- `torch>=2.0.0` - Deep learning backend

**Storage**:
- `clickhouse-connect>=0.6.0` - ClickHouse client
- `minio>=7.2.0` - MinIO S3-compatible storage

**ML/Data**:
- `scikit-learn>=1.3.0` - Train/test split, metrics
- `pandas>=2.0.0`, `numpy>=1.24.0` - Data processing

## 📝 License

Academic research project - see institution guidelines.
