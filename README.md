# Hybrid Log Management System

A research implementation of an intelligent, energy-aware log routing system. This system dynamically routes log entries to one of three specialized storage tiers (MySQL, ELK, IPFS) based on Intelligent Routing balancing performance, analytical capability, and cryptographic integrity.

## 📁 Repository Structure

```
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
│   ├── routers.py                  # Implements Static, Q-learning, and A2C routers
│   ├── train.py                    # Script to train the A2C model
│   └── train_qlearning.py          # Script to train the Q-learning agent
├── tables/                         # Processed data tables for the thesis
├── test_ipfs.py                    # Utility script to test IPFS connection
└── trained_models/                 # Persisted models from the training phase
    ├── a2c_Loghub-zenodo_Logs.zip  # Trained A2C model checkpoint
    ├── q_learning_binner.pkl       # Q-learning's discretization model
    ├── q_learning_pca.pkl          # Q-learning's PCA model
    └── q_learning_q_table.pkl      # Q-learning's Q-table
```

## 🎯 Key Features

- **Three-Storage Hybrid Architecture:** Routes logs to the optimal backend:
  - **Performance (MySQL):** High-throughput, low-latency storage for routine operational logs.
  - **Analytics (ELK Stack):** Powerful search and aggregation for debugging and metrics.
  - **Integrity (IPFS):** Immutable, tamper-evident storage for security-critical audit trails.
- **Intelligent Routing:** Evaluates three routing strategies:
  - **Static Policy:** Rule-based routing (Content-Based Routing).
  - **Q-learning:** Tabular Reinforcement Learning agent.
  - **A2C (Advantage Actor-Critic):** Deep Reinforcement Learning agent.
- **Semantic Log Understanding:** Uses a pre-trained **DistilBERT** model to generate embeddings from log text, enabling content-aware routing decisions.
- **Energy-Aware Evaluation:** Measures CPU energy consumption using Intel's RAPL interface and estimates carbon emissions.
- **Full Reproducibility:** Containerized backends, pinned versions, and a frozen evaluation phase ensure all results are reproducible.

## ⚙️ Prerequisites

- **Docker & Docker Compose:** Required to run the storage backends.
- **Python 3.9+**
- **Bash Shell** (to run `run_all.sh`)
- An Intel CPU (required for accurate energy measurement via RAPL)

## 🚀 Quick Start

1. **Clone the repository and navigate into it:**
   ```bash
   git clone https://github.com/adam-bouafia/Towards-Greener-Clouds-Evaluating-Hybrid-Log-Managament
   cd hybrid-log-management
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend services (MySQL, Elasticsearch, Kibana, IPFS):**
   ```bash
   docker-compose up -d
   ```
   *Wait a few minutes for Elasticsearch to fully start. Check logs with `docker-compose logs elasticsearch`.*

4. **Run the complete experiment pipeline:**  
   This script runs the two-phase experiment (training followed by frozen evaluation) for all routers and both datasets.
   ```bash
   ./run_all.sh
   ```

5. **Analyze the results:**
   - **Raw Data:** See CSV files in the `results/` directory.
   - **Plots & Figures:** Run the `plots.ipynb` Jupyter notebook to regenerate all graphs saved in the `figures/` folder.
   - **Aggregate Tables:** The `tables/` directory contains aggregated results for the thesis.

## 🔬 The Two-Phase Experiment

The experiment is designed to ensure a fair, reproducible comparison between routing strategies.

1. **Phase 1: Training**
   - **Goal:** Train the learning-based models (Q-learning, A2C) and create all necessary artifacts.
   - **Process:** The agents interact with the environment, learning a routing policy based on a reward signal that balances latency, energy, and integrity.
   - **Output:** Frozen model files (`.pkl` for Q-learning, `.zip` for A2C) are saved to the `trained_models/` directory. The Static router requires no training.

2. **Phase 2: Frozen Evaluation**
   - **Goal:** Compare all routers on an *identical* sequence of log entries.
   - **Process:** The pre-trained models are loaded. Each log from the dataset is processed by every router in a deterministic manner, and key metrics (latency, energy, success) are meticulously measured.
   - **Output:** Detailed per-log and aggregate results in the `results/` directory. This phase guarantees that performance differences are due to the algorithms' effectiveness, not random variation or data leakage.

 

## 📊 Results

The `figures/` directory contains the key findings from the thesis, comparing the routers across:

- Latency vs. Throughput trade-offs
- Energy efficiency (Wh per log)
- Scaling behavior under load (14k vs. 200k logs)
- Tail latency (90th, 95th, 99th percentiles)
- Destination selection patterns (adaptivity)

 
## 👤 Author

**Adam Bouafia**  
[LinkedIn](hhttps://www.linkedin.com/in/adam-bouafia-b597ab86/)  
---

**Note:** This is a research prototype. Absolute performance numbers will vary based on hardware. The relative comparisons and trade-offs between the strategies are the key findings.