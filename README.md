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
  - **A2C:** Deep RL using Stable-Baselines3 (now with configurable hyperparameters, evaluation + metadata artifacts, optional observation scaler, reward CSV export).
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

### A2C Training Enhancements

Train the A2C agent with the upgraded script:

```bash
python -m src.train \
  --timesteps 120000 \
  --model_path trained_models/a2c_log_router \
  --log_source real_world \
  --log_filepath data/Loghub-zenodo_Logs.csv \
  --sample_mode balanced \
  --learning_rate 5e-4 \
  --policy_hidden 256,128 \
  --n_steps 10 \
  --n_envs 1 \
  --eval_interval 10000 \
  --eval_episodes 5 \
  --save_best \
  --scaler_warmup 4000 \
  --reward_csv_path results/a2c_rewards.csv \
  --tensorboard_log runs/a2c \
  --seed 42
```

Artifacts produced (prefix `trained_models/a2c_log_router`):

| Suffix | Purpose |
|--------|---------|
| `.zip` | Final trained policy (A2C) |
| `_best.zip` | Best eval model (if `--save_best` and eval enabled) |
| `_metadata.json` | Training metadata (version=1: hyperparams, reward stats, seed, git commit, scaler flags) |
| `_scaler.pkl` | Optional z-score scaler dict (`mean`, `std`) when `--scaler_warmup > 0` |
| `a2c_rewards.csv` | (If specified) Per-episode total rewards for plotting |

Key CLI Flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--learning_rate` | Optimizer learning rate | 7e-4 |
| `--gamma` | Discount factor | 0.99 |
| `--ent_coef` | Entropy coefficient | 0.0 |
| `--vf_coef` | Value function loss weight | 0.5 |
| `--n_steps` | Rollout horizon per update | 5 |
| `--n_envs` | Parallel env count (DummyVecEnv) | 1 |
| `--policy_hidden` | Comma list hidden layer sizes (pi & vf) | 128,128 |
| `--eval_interval` | Evaluate every N steps (0 disables) | 0 |
| `--eval_episodes` | Episodes per evaluation | 5 |
| `--save_best` | Save best eval model copy | off |
| `--scaler_warmup` | Collect this many obs to build mean/std | 0 |
| `--reward_csv_path` | Write per-episode rewards CSV | None |
| `--tensorboard_log` | TensorBoard log directory | None |
| `--seed` | Global RNG seed | None |
| `--lr_linear_decay` | Linearly decay LR to 0 over training | off |
| `--ent_anneal` | Enable entropy coefficient annealing | off |
| `--ent_target` | Target entropy coef (with anneal) | None |
| `--ent_anneal_steps` | Steps to reach target entropy | None |
| `--checkpoint_interval` | Save interim checkpoints every N steps | None |

Metadata Schema (version=1):
```jsonc
{
  "version": 1,
  "timestamp": 1730...,          // epoch seconds
  "obs_dim": 774,                // observation vector length seen by network
  "embedding_dim": 768,          // embedding component length (current model)
  "policy_hidden": [256,128],
  "total_timesteps": 120000,
  "learning_rate": 0.0005,
  "gamma": 0.99,
  "ent_coef": 0.0,
  "vf_coef": 0.5,
  "n_steps": 10,
  "n_envs": 1,
  "scaler_present": true,
  "scaler_warmup": 4000,
  "reward_mean": 0.42,
  "reward_std": 0.77,
  "reward_min": -1.9,
  "reward_max": 1.8,
  "eval_best_mean_reward": 0.65,
  "seed": 42,
  "git_commit": "a1b2c3d4e5f6"
  "lr_linear_decay": true,
  "final_learning_rate": 0.0,
  "ent_anneal": true,
  "ent_target": 0.01,
  "ent_anneal_steps": 60000,
  "final_ent_coef": 0.01,
  "checkpoint_interval": 10000,
  "num_checkpoints": 11
}
```

Inference (router) automatically loads `_scaler.pkl` and applies it before prediction if present.

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

## 🔬 Unified Six-Router Experiment Orchestrator

A single command now prepares (training if needed) and evaluates the six comparative routers:

`a2c`, `cbr`, `q_learning`, `direct_mysql`, `direct_elk`, `direct_ipfs`

The legacy `static` policy is retained only as an internal fallback inside adaptive routers and is no longer part of the headline comparisons.

### Why a Unified Orchestrator?

Previously you had to (1) train Q-learning, (2) train A2C, and (3) run separate evaluation invocations. The new orchestrator (`src/experiment.py`) automates this flow, ensuring consistent dataset sampling, metric collection, and aggregated reporting (latency distribution, success rate, destination mix, energy).

### Command Example

```bash
python -m src.experiment \
  --log_filepath data/Loghub-zenodo_Logs.csv \
  --sample_mode head \
  --q_episodes 120 \
  --a2c_timesteps 120000 \
  --routers all \
  --cbr_cost_metric combined \
  --output_dir results \
  --write_markdown
```

What happens:
 
1. Checks for Q-learning artifacts (`trained_models/q_learning_*`). Trains if missing or `--force_retrain` supplied.
2. Checks for A2C artifacts (`trained_models/a2c_log_router.zip`, metadata). Trains if missing or forced.
3. Runs evaluation for each selected router (default `all` → six routers). CBR always starts cold (online learner by design). RL agents are frozen (no learning during evaluation phase).
4. Produces per-router CSVs (already implemented in `__main__.py`).
5. Aggregates all per-log CSVs into `combined_summary_<dataset>.csv` (+ optional markdown table) with:
   - mean / median / p95 total latency
   - success rate
   - sensitive log fraction
   - destination mix (mysql / elk / ipfs)
   - average energy per log (J)
6. Writes `experiment_metadata_<dataset>.json` capturing training reuse, durations, and statuses.

### Key Orchestrator Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `--log_filepath` | Real-world log CSV path | (required) |
| `--sample_mode` | Log sampling mode (`head`,`random`,`balanced`) | head |
| `--routers` | Comma list or `all` | all |
| `--q_episodes` | Q-learning training episodes | 120 |
| `--q_prefix` | Q-learning artifact prefix (under `trained_models/`) | q_learning |
| `--a2c_timesteps` | A2C training timesteps | 120000 |
| `--a2c_base` | A2C base model name (under `trained_models/`) | a2c_log_router |
| `--a2c_scaler_warmup` | A2C scaler warmup obs (0 disables) | 0 |
| `--a2c_lr_linear_decay` | Enable linear LR decay | off |
| `--a2c_ent_anneal` | Enable entropy anneal (requires target + steps) | off |
| `--a2c_ent_target` | Target entropy coef (with anneal) | None |
| `--a2c_ent_steps` | Anneal steps | None |
| `--cbr_cost_metric` | CBR cost focus (`latency`,`energy`,`combined`) | combined |
| `--cbr_state_path` | Optional persistent CBR state JSON | None |
| `--force_retrain` | Retrain RL even if artifacts present | off |
| `--seed` | RNG seed (passed to A2C) | None |
| `--output_dir` | Directory for combined outputs | results |
| `--write_markdown` | Emit Markdown summary table | off |
| `--compliance_enable` | Force sensitive logs (pattern match) to IPFS | off |
| `--compliance_patterns` | Extra comma patterns merged with defaults | None |

### Fairness & Interpretation

- Q-learning & A2C are trained offline and evaluated frozen.
- CBR adapts online during the evaluation stream (that *is* the method).
- Direct baselines give lower/upper reference bounds for each individual backend.
- Energy values rely on RAPL; on unsupported hardware they may read near zero—still comparable across routers.
- `sample_mode` should be kept constant across runs to avoid distribution-induced bias.

### Outputs

| File | Description |
|------|-------------|
| `results/<router>_<dataset>.csv` | Per-log trace (existing behavior) |
| `results/summary_<router>_<dataset>.csv` | Per-router summary (existing) |
| `results/combined_summary_<dataset>.csv` | Cross-router aggregate metrics |
| `results/combined_summary_<dataset>.md` | (Optional) Markdown table |
| `results/experiment_metadata_<dataset>.json` | Run metadata (timings, training reused status) |
| `results/combined_summary_<dataset>.csv` | Includes compliance metrics when enabled |

### Extending

Add new router names to `DEFAULT_ROUTERS` in `src/experiment.py` and ensure `__main__.py` can build them. The aggregator will pick up any additional per-log CSV automatically if included in `--routers`.

---

## 🛡️ Compliance (Hard Enforcement to IPFS)

When `--compliance_enable` is provided (either to a direct `python -m src ...` run or via the orchestrator / menu), **all logs matching sensitive patterns are forcibly routed to IPFS**, regardless of the adaptive router's chosen destination.

### Default Sensitive Patterns

```text
sessionid, token, secret key, permission denied, login, /home/
```

You may extend these with `--compliance_patterns "email,credential"` (case-insensitive substring matching). The effective pattern list is recorded in each per-router summary CSV when compliance is enabled.

### Rationale

IPFS is treated as the immutable, tamper-evident audit store. Hard enforcement ensures 100% coverage (zero leakage) for security / regulatory contexts while allowing adaptive optimization on the remaining discretionary logs.

### Per-Log Fields Added

| Column | Meaning |
|--------|---------|
| `raw_destination` | Router's original choice prior to compliance override |
| `compliance_forced` | True if override changed destination to IPFS |

### Aggregated Compliance Metrics (in combined summary)

| Metric | Definition |
|--------|------------|
| `sensitive_total` | Number of logs classified sensitive |
| `sensitive_to_ipfs` | Sensitive logs actually stored in IPFS |
| `sensitive_coverage` | sensitive_to_ipfs / sensitive_total (==1.0 expected) |
| `sensitive_leakage` | sensitive_total - sensitive_to_ipfs (==0 expected) |
| `leakage_rate` | sensitive_leakage / sensitive_total |
| `non_sensitive_ipfs_fraction` | Fraction of non-sensitive logs sent to IPFS |
| `compliance_score` | 1.0 if leakage == 0 else 0.0 |

### Example (Orchestrator with Compliance)

```bash
python -m src.experiment \
  --log_filepath data/Loghub-zenodo_Logs.csv \
  --routers q_learning,a2c,cbr \
  --q_episodes 20 \
  --a2c_timesteps 20000 \
  --compliance_enable \
  --compliance_patterns sessionid,token
```

---

## 🧭 Interactive Experiment Menu

A convenience interface to launch common experiment profiles without memorizing long command lines.

Run:

```bash
python -m src.menu --log_filepath data/Loghub-zenodo_Logs.csv
```

### Menu Options

| Option | Description |
|--------|-------------|
| 1 | Smoke run (fast; limited episodes/timesteps) |
| 2 | Full run (longer training; all routers) |
| 3 | Compliance smoke run (smoke + hard compliance) |
| 4 | Train only (retrain RL artifacts, no multi-router evaluation) |
| 5 | Custom (prompt for parameters including compliance) |

Outputs are placed in the specified `--output_dir` (default `results/menu_runs`).

### Non-Interactive Presets

```bash
# Smoke
python -m src.menu --preset smoke --log_filepath data/Loghub-zenodo_Logs.csv

# Compliance smoke (explicit)
python -m src.menu --preset compliance_smoke --log_filepath data/Loghub-zenodo_Logs.csv

# Force compliance on any preset + add patterns
python -m src.menu --preset smoke --compliance --extra_patterns sessionid,token,permission denied --log_filepath data/Loghub-zenodo_Logs.csv
```

Internally this wraps `python -m src.experiment` with the appropriate arguments.

---

---

The sections below (training artifacts) remain for detailed reference.

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
