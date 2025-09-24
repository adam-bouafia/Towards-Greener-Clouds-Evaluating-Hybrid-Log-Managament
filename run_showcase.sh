#!/usr/bin/env bash
# One-shot showcase pipeline: trains (if needed) Q-learning + A2C for both datasets,
# then runs evaluation WITH and WITHOUT compliance. Designed to be simple.
#
# Usage:
#   bash run_showcase.sh            # full long run (can take many hours)
#   FAST=1 bash run_showcase.sh     # quick smoke to verify setup
#   FORCE_RETRAIN=1 bash run_showcase.sh  # retrain even if artifacts exist
#
# You can override defaults, e.g.:
#   QL_EPISODES_REAL=10000 A2C_TIMESTEPS_REAL=3000000 bash run_showcase.sh

set -euo pipefail

# -------------------------
# Config (overridable env)
# -------------------------
FAST=${FAST:-0}
FORCE_RETRAIN=${FORCE_RETRAIN:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
OUT_DIR="results/showcase_${STAMP}"
mkdir -p "${OUT_DIR}" || true

# Long-run defaults
QL_EPISODES_REAL=${QL_EPISODES_REAL:-20000}
QL_STEPS_PER_EPISODE_REAL=${QL_STEPS_PER_EPISODE_REAL:-300}
QL_EPISODES_SYN=${QL_EPISODES_SYN:-12000}
QL_STEPS_PER_EPISODE_SYN=${QL_STEPS_PER_EPISODE_SYN:-400}
A2C_TIMESTEPS_REAL=${A2C_TIMESTEPS_REAL:-5000000}
A2C_TIMESTEPS_SYN=${A2C_TIMESTEPS_SYN:-4800000}

# Smoke overrides if FAST=1
if [ "${FAST}" = "1" ]; then
  echo "[MODE] FAST=1 → using tiny numbers for a smoke test"
  QL_EPISODES_REAL=300
  QL_STEPS_PER_EPISODE_REAL=150
  QL_EPISODES_SYN=300
  QL_STEPS_PER_EPISODE_SYN=150
  A2C_TIMESTEPS_REAL=120000
  A2C_TIMESTEPS_SYN=120000
fi

# Artifact prefixes
Q_REAL_PREFIX=trained_models/q_learning_real_long
Q_SYN_PREFIX=trained_models/q_learning_synth_long
A2C_REAL_PREFIX=trained_models/a2c_real_long
A2C_SYN_PREFIX=trained_models/a2c_synth_long

# Helper: check if artifact exists (Q-learning needs q_table, A2C needs .zip)
function have_q_artifacts() { [ -f "$1_q_table.pkl" ]; }
function have_a2c_artifacts() { [ -f "$1.zip" ]; }

# -------------------------
# Q-learning Real
# -------------------------
if [ "${FORCE_RETRAIN}" = "1" ] || ! have_q_artifacts "${Q_REAL_PREFIX}"; then
  echo "[TRAIN] Q-learning REAL dataset (${QL_EPISODES_REAL} eps × ${QL_STEPS_PER_EPISODE_REAL} steps)"
  python -m src.train_qlearning \
    --log_filepath data/Loghub-zenodo_Logs.csv \
    --dataset_name LoghubReal \
    --episodes ${QL_EPISODES_REAL} \
    --max_steps_per_episode ${QL_STEPS_PER_EPISODE_REAL} \
    --alpha 0.25 \
    --gamma 0.985 \
    --eps_start 1.0 \
    --eps_end 0.02 \
    --eps_decay 0.9992 \
    --adaptive_eps_window 40 \
    --adaptive_eps_patience 3 \
    --adaptive_eps_drop 0.6 \
    --adaptive_eps_min_episodes 400 \
    --warmup_steps 5000 \
    --pca_components 16 \
    --n_bins 8 \
    --guided_prob 0.45 \
    --prior_bonus 0.35 \
    --model_path_prefix ${Q_REAL_PREFIX} \
    --sample_mode balanced \
    --reward_history_inline_threshold 100 \
    --reward_history_csv_path ${OUT_DIR}/qlearn_real_reward_history.csv \
    --adaptive_events_path ${OUT_DIR}/qlearn_real_adaptive_events.ndjson
else
  echo "[SKIP] Q-learning REAL artifacts present (use FORCE_RETRAIN=1 to redo)"
fi

# -------------------------
# Q-learning Synthetic
# -------------------------
if [ "${FORCE_RETRAIN}" = "1" ] || ! have_q_artifacts "${Q_SYN_PREFIX}"; then
  echo "[TRAIN] Q-learning SYNTH dataset (${QL_EPISODES_SYN} eps × ${QL_STEPS_PER_EPISODE_SYN} steps)"
  python -m src.train_qlearning \
    --log_filepath data/Synthetic_Datacenter_Logs.csv \
    --dataset_name SyntheticDC \
    --episodes ${QL_EPISODES_SYN} \
    --max_steps_per_episode ${QL_STEPS_PER_EPISODE_SYN} \
    --alpha 0.25 \
    --gamma 0.985 \
    --eps_start 1.0 \
    --eps_end 0.02 \
    --eps_decay 0.9992 \
    --adaptive_eps_window 40 \
    --adaptive_eps_patience 3 \
    --adaptive_eps_drop 0.6 \
    --adaptive_eps_min_episodes 400 \
    --warmup_steps 6000 \
    --pca_components 16 \
    --n_bins 8 \
    --guided_prob 0.40 \
    --prior_bonus 0.35 \
    --model_path_prefix ${Q_SYN_PREFIX} \
    --sample_mode balanced \
    --reward_history_inline_threshold 100 \
    --reward_history_csv_path ${OUT_DIR}/qlearn_synth_reward_history.csv \
    --adaptive_events_path ${OUT_DIR}/qlearn_synth_adaptive_events.ndjson
else
  echo "[SKIP] Q-learning SYNTH artifacts present (use FORCE_RETRAIN=1 to redo)"
fi

# -------------------------
# A2C Real
# -------------------------
if [ "${FORCE_RETRAIN}" = "1" ] || ! have_a2c_artifacts "${A2C_REAL_PREFIX}"; then
  echo "[TRAIN] A2C REAL (${A2C_TIMESTEPS_REAL} timesteps)"
  python -m src.train \
    --timesteps ${A2C_TIMESTEPS_REAL} \
    --model_path ${A2C_REAL_PREFIX} \
    --log_source real_world \
    --log_filepath data/Loghub-zenodo_Logs.csv \
    --sample_mode balanced \
    --learning_rate 5e-4 \
    --lr_linear_decay \
    --policy_hidden 256,128 \
    --n_steps 10 \
    --n_envs 1 \
    --gamma 0.99 \
    --ent_coef 0.002 \
    --ent_anneal \
    --ent_target 0.0005 \
    --ent_anneal_steps $(( A2C_TIMESTEPS_REAL * 3 / 5 )) \
    --eval_interval $(( A2C_TIMESTEPS_REAL / 20 )) \
    --eval_episodes 6 \
    --save_best \
    --scaler_warmup $(( A2C_TIMESTEPS_REAL / 125 )) \
    --reward_csv_path ${OUT_DIR}/a2c_real_rewards.csv \
    --tensorboard_log runs/a2c_real_long \
    --checkpoint_interval $(( A2C_TIMESTEPS_REAL / 10 )) \
    --seed 42
else
  echo "[SKIP] A2C REAL artifacts present (use FORCE_RETRAIN=1 to retrain)"
fi

# -------------------------
# A2C Synthetic
# -------------------------
if [ "${FORCE_RETRAIN}" = "1" ] || ! have_a2c_artifacts "${A2C_SYN_PREFIX}"; then
  echo "[TRAIN] A2C SYNTH (${A2C_TIMESTEPS_SYN} timesteps)"
  python -m src.train \
    --timesteps ${A2C_TIMESTEPS_SYN} \
    --model_path ${A2C_SYN_PREFIX} \
    --log_source real_world \
    --log_filepath data/Synthetic_Datacenter_Logs.csv \
    --sample_mode balanced \
    --learning_rate 5e-4 \
    --lr_linear_decay \
    --policy_hidden 256,128 \
    --n_steps 10 \
    --n_envs 1 \
    --gamma 0.99 \
    --ent_coef 0.002 \
    --ent_anneal \
    --ent_target 0.0005 \
    --ent_anneal_steps $(( A2C_TIMESTEPS_SYN * 3 / 5 )) \
    --eval_interval $(( A2C_TIMESTEPS_SYN / 20 )) \
    --eval_episodes 6 \
    --save_best \
    --scaler_warmup $(( A2C_TIMESTEPS_SYN / 125 )) \
    --reward_csv_path ${OUT_DIR}/a2c_synth_rewards.csv \
    --tensorboard_log runs/a2c_synth_long \
    --checkpoint_interval $(( A2C_TIMESTEPS_SYN / 10 )) \
    --seed 43
else
  echo "[SKIP] A2C SYNTH artifacts present (use FORCE_RETRAIN=1 to retrain)"
fi

# -------------------------
# Evaluations (Compliance + Baseline)
# -------------------------
# Real - compliance
python -m src.experiment \
  --log_filepath data/Loghub-zenodo_Logs.csv \
  --routers all \
  --q_prefix $(basename ${Q_REAL_PREFIX}) \
  --q_episodes 1 \
  --a2c_base $(basename ${A2C_REAL_PREFIX}) \
  --a2c_timesteps 0 \
  --cbr_cost_metric combined \
  --sample_mode balanced \
  --output_dir ${OUT_DIR}/eval_real_compliance \
  --write_markdown \
  --compliance_enable

# Real - baseline
python -m src.experiment \
  --log_filepath data/Loghub-zenodo_Logs.csv \
  --routers all \
  --q_prefix $(basename ${Q_REAL_PREFIX}) \
  --q_episodes 1 \
  --a2c_base $(basename ${A2C_REAL_PREFIX}) \
  --a2c_timesteps 0 \
  --cbr_cost_metric combined \
  --sample_mode balanced \
  --output_dir ${OUT_DIR}/eval_real_baseline \
  --write_markdown

# Synthetic - compliance
python -m src.experiment \
  --log_filepath data/Synthetic_Datacenter_Logs.csv \
  --routers all \
  --q_prefix $(basename ${Q_SYN_PREFIX}) \
  --q_episodes 1 \
  --a2c_base $(basename ${A2C_SYN_PREFIX}) \
  --a2c_timesteps 0 \
  --cbr_cost_metric combined \
  --sample_mode balanced \
  --output_dir ${OUT_DIR}/eval_synth_compliance \
  --write_markdown \
  --compliance_enable

# Synthetic - baseline
python -m src.experiment \
  --log_filepath data/Synthetic_Datacenter_Logs.csv \
  --routers all \
  --q_prefix $(basename ${Q_SYN_PREFIX}) \
  --q_episodes 1 \
  --a2c_base $(basename ${A2C_SYN_PREFIX}) \
  --a2c_timesteps 0 \
  --cbr_cost_metric combined \
  --sample_mode balanced \
  --output_dir ${OUT_DIR}/eval_synth_baseline \
  --write_markdown

# -------------------------
# Merge summary (optional consolidation)
# -------------------------
python - <<'PY'
import pandas as pd, glob, os, json
root = "${OUT_DIR}"  # substituted by shell before heredoc passes to python? No; it's literal—so we expand manually.
# Workaround: use environment variable expansion by printing from bash into python
PY
python - <<PY
import pandas as pd, glob, os
root = "${OUT_DIR}"
rows = []
for path in glob.glob(f"{root}/eval_*_*/combined_summary_*.csv"):
    df = pd.read_csv(path)
    dataset = path.split("combined_summary_")[1].replace(".csv","")
    compliance = ("compliance" in path)
    df['dataset'] = dataset
    df['compliance'] = compliance
    rows.append(df)
if rows:
    merged = pd.concat(rows, ignore_index=True)
    out_path = f"{root}/merged_all_results.csv"
    merged.to_csv(out_path, index=False)
    print(f"[MERGE] Wrote {out_path} ({len(merged)} rows)")
else:
    print("[MERGE] No combined summary files found.")
PY

echo "\n===================================="
echo "DONE. Key outputs under: ${OUT_DIR}"
echo "Look at: ${OUT_DIR}/eval_*/*/combined_summary_*.csv" 
echo "Merged table (if created): ${OUT_DIR}/merged_all_results.csv" 
echo "Use FAST=1 for a quick smoke before full run."