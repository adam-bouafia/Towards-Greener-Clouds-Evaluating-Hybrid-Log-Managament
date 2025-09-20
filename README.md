# Hybrid Log Management System

A research implementation of an intelligent, energy-aware log routing system. This system dynamically routes log entries to one of three specialized storage tiers (MySQL, ELK, IPFS) based on Intelligent Routing balancing performance, analytical capability, and cryptographic integrity.

## 🆕 New: Content-Based Routing (CBR)

The project now includes an adaptive **Content-Based Router (CBR)** inspired by Bizarro et al. (VLDB'05). Unlike hand-crafted static rules, CBR:

- Learns online which log attributes (e.g., `Level`, `Component`, `LogSource`) correlate with backend cost (latency / energy).
- Hash-partitions attribute values into buckets and keeps per-backend cost statistics per bucket.
- Periodically scores attributes using a variance-reduction heuristic (proxy for gain) and picks the highest scoring attribute as its current classifier.
- Routes each log to the backend with the lowest expected cost for that bucket.
- Falls back gracefully to global averages, then to the original static policy when insufficient data is available.

Supported cost metrics (`--cbr_cost_metric`):

- `latency` (default)
- `energy` (uses per-log CPU package Joules when available)
- `combined` (simple weighted sum: latency_ms + 1000 * energy_j)

Optional JSON diagnostics: set `--cbr_json_dump_path path/to/cbr_diag.json --cbr_json_dump_interval 500` to periodically dump internal state (attribute scores, selected classifier, sample counts).

## 📁 Repository Structure

```text
hybrid-log-management/
├── data/                           # Input log datasets
│   ├── Loghub-zenodo_Logs.csv      # Real-world logs from Loghub 2.0 (~14k entries)
│   └── Synthetic_Datacenter_Logs.csv # Generated synthetic logs (~200k entries)
├── docker-compose.yml              # Defines MySQL, ELK Stack, and IPFS services
├── figures/                        # Generated plots and visualizations (results)
├── plots.ipynb                     # Jupyter notebook for results analysis & plotting
├── requirements.txt                # Python dependencies
├── results/                        # Detailed CSV results from experiments
├── run_all.sh                      # Main script to run the entire experiment pipeline
├── src/                            # Core application source code
│   ├── backends.py                 # Handles connections to MySQL, Elasticsearch, IPFS
│   ├── config.py                   # Configuration and constants
│   ├── feature_extractor.py        # Creates state vectors (DistilBERT + system metrics)
│   ├── log_provider.py             # Reads and samples from log datasets
│   ├── __main__.py                 # Main application entry point
│   ├── metrics.py                  # Energy and performance measurement (RAPL)
│   ├── routers.py                  # Static, CBR, Q-learning, A2C, direct baselines
│   ├── train.py                    # Script to train the A2C model
│   └── train_qlearning.py          # Script to train the Q-learning agent
├── tables/                         # Processed data tables for the thesis
├── tests/                          # Automated tests (CBR unit tests etc.)
├── test_ipfs.py                    # Utility script to test IPFS connection
└── trained_models/                 # Persisted models from the training phase
```

## 🎯 Key Features

- **Three-Storage Hybrid Architecture:** Routes logs to the optimal backend:
  - **Performance (MySQL):** High-throughput, low-latency storage for routine operational logs.
  - **Analytics (ELK Stack):** Powerful search and aggregation for debugging and metrics.
  - **Integrity (IPFS):** Immutable, tamper-evident storage for security-critical audit trails.
- **Adaptive Routing Strategies:**
  - **Static Policy (Rules):** Deterministic baseline using log severity/content heuristics.
  - **CBR (Content-Based Routing):** Online adaptive attribute-based cost minimization (no offline training).
  - **Q-learning:** Tabular RL with PCA + discretization and a teacher-guided exploration.
    - Artifacts now include scaler (mean/std) and metadata JSON for compatibility validation.
  - **A2C:** Deep RL using Stable-Baselines3.
  - **Direct Baselines:** Always route to a single backend (`direct_mysql`, etc.).
- **Semantic Log Understanding:** DistilBERT embeddings of log content.
- **Energy-Aware Evaluation:** RAPL CPU package Joules → Wh/log + CO₂ estimation.
- **Diagnostics & Explainability:** Optional JSON dumps for CBR; per-log CSV traces; attribute scoring.

## ⚙️ Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Intel CPU (for accurate RAPL measurements; otherwise energy will register as zero)

## 🚀 Quick Start

```bash
git clone https://github.com/adam-bouafia/Towards-Greener-Clouds-Evaluating-Hybrid-Log-Managament
cd hybrid-log-management
pip install -r requirements.txt
docker-compose up -d
```

Run an experiment with CBR (real dataset example):

```bash
python -m src --router cbr \
  --log_source real_world \
  --log_filepath data/Loghub-zenodo_Logs.csv \
  --cbr_cost_metric combined \
  --cbr_num_buckets 32 \
  --cbr_sample_prob 0.08 \
  --cbr_warm_samples 120 \
  --cbr_json_dump_path results/cbr_diag.json \
  --cbr_json_dump_interval 500
```

Switch to A2C for comparison:

```bash
python -m src --router a2c --model_path trained_models/a2c_log_router --log_source real_world --log_filepath data/Loghub-zenodo_Logs.csv
```

## 🔧 CBR Parameters

| Flag | Purpose | Default |
|------|---------|---------|
| `--cbr_num_buckets` | Hash buckets per attribute | 24 |
| `--cbr_sample_prob` | Probability of sampling a log for stats | 0.06 |
| `--cbr_warm_samples` | Samples required before first attribute selection | 150 |
| `--cbr_recompute_interval` | Decisions between rescoring | 200 |
| `--cbr_cost_metric` | `latency`, `energy`, or `combined` | latency |
| `--cbr_latency_weight` | Weight for latency in combined cost | 1.0 |
| `--cbr_energy_weight` | Weight for energy (J) in combined cost | 1000.0 |
| `--cbr_json_dump_path` | Optional path to JSON diagnostics | None |
| `--cbr_json_dump_interval` | Dump frequency (decisions) | 0 (disabled) |
| `--cbr_json_dump_mode` | JSON writing mode: overwrite / append / timestamp | overwrite |
| `--cbr_state_path` | Persist / load CBR learned stats JSON | None |


### JSON Dump Modes

- `overwrite`: Replace file each interval with latest snapshot.
- `append`: Write one JSON object per line (NDJSON stream).
- `timestamp`: Emit separate file per dump (suffix epoch seconds).

### State Persistence

If `--cbr_state_path` is provided, CBR will attempt to load prior statistics at startup and save updated state on exit, enabling faster adaptation across repeated runs.

## 🧪 Tests

Basic unit tests for CBR are located in `tests/test_cbr_router.py` covering:

- Fallback behavior prior to warm-up.
- Attribute selection after sufficient samples.
- JSON diagnostic dump creation.

Run tests with:

```bash
pytest -q
```

## 🔬 Two-Phase Experiment (Learning Agents)

(Description unchanged; CBR operates online and does not require a separate training phase.)

### Q-learning Artifacts & Compatibility

When you train the Q-learning agent via `train_qlearning.py`, the following files are produced (prefix defaults to `trained_models/q_learning`):

| Suffix | Purpose |
|--------|---------|
| `_q_table.pkl` | Learned Q-values per discretized state |
| `_pca.pkl` | PCA transformer (after scaling) |
| `_binner.pkl` | KBinsDiscretizer for quantile binning of PCA output |
| `_scaler.pkl` | Dict with `mean` and `std` vectors (float32) for z-score normalization |
| `_metadata.json` | Training metadata (obs_dim, pca_components, n_bins, hyperparams, version) |

At inference time `QLearningRouter` will:

1. Load scaler → normalize `[system_state + embedding]`.
2. Apply PCA and KBins discretization.
3. Look up action in Q-table, falling back to the static policy for unknown states.
4. If artifact dimensions mismatch (e.g., embedding size change), it invalidates the model and gracefully reverts to the static router.

This ensures forward compatibility when you upgrade feature extraction without silently producing invalid actions.

#### Adaptive Exploration (Epsilon Scheduling)

New optional adaptive epsilon decay monitors recent episode reward plateaus:

| Flag | Meaning | Default |
|------|---------|---------|
| `--adaptive_eps_window` | Number of recent episodes to average for plateau detection (0 disables) | 20 |
| `--adaptive_eps_patience` | Consecutive non-improving windows before forcing extra decay | 3 |
| `--adaptive_eps_drop` | Multiplicative factor applied to epsilon on plateau event | 0.7 |
| `--adaptive_eps_min_episodes` | Minimum episodes before adaptive logic activates | 10 |
| `--no_static_eps_decay` | Disable baseline geometric decay (adaptive only) | false |
| `--reward_history_inline_threshold` | If total episodes <= threshold, embed full reward list in metadata | 120 |

Metadata additions (`version` now 3):

- `embedding_dim`, `reward_mean/median/std`
- `adaptive_events` (list of {episode, epsilon})
- `final_epsilon`, `best_recent_avg`
- Adaptive configuration flags persisted for reproducibility.
- `episode_rewards` (only when episodes <= reward_history_inline_threshold)
- `reward_history_inline_threshold`

#### Reward & Adaptive Event Outputs

For large training runs where full reward history is not embedded in metadata you can export:

| Flag | Output | Format |
|------|--------|--------|
| `--reward_history_csv_path PATH` | Per-episode rewards (when episodes > inline threshold) | CSV (episode,reward) |
| `--adaptive_events_path PATH` | Adaptive epsilon change events | NDJSON (one JSON object per line) |

These files facilitate plotting learning curves without parsing large metadata blobs.

## 📊 Results & Plots

See `results/` and `figures/` as before; adding CBR will generate `cbr_<dataset>.csv` and `summary_cbr_<dataset>.csv`.

## 🔍 Future Improvements

- Enhanced cost modeling (include variance, success penalties).
- Persistence of CBR learned stats across runs.
- Dynamic bucket refinement for high-cardinality attributes.
- Unified evaluation harness for energy-normalized comparisons.

## 👤 Author

**Adam Bouafia**  
[LinkedIn](hhttps://www.linkedin.com/in/adam-bouafia-b597ab86/)

---

*Research prototype: absolute metrics vary by host; comparative behavior is the key insight.*
