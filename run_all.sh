#!/usr/bin/env bash
set -euo pipefail

# Trains A2C + Q-learning per dataset, then runs:
#   static, q_learning, a2c, direct_mysql, direct_elk, direct_ipfs
# ENV:
#   USE_OPENVINO=1|0
#   FAST=1 (use LIMIT default 400)
#   LIMIT=...
#   SAMPLE_MODE=head|random|balanced
#   Q_EPISODES (default 80), WARMUP (default 3000)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_DIR="./data"
RESULTS_DIR="./results"
MODELS_DIR="./trained_models"
mkdir -p "$RESULTS_DIR" "$MODELS_DIR"

USE_OPENVINO="${USE_OPENVINO:-1}"
if [[ "$USE_OPENVINO" == "1" ]]; then
  export EMBED_BACKEND=openvino
  export EMBED_DEVICE=GPU
  echo "[OV] OpenVINO enabled (GPU)"
else
  unset EMBED_BACKEND EMBED_DEVICE
  echo "[OV] OpenVINO disabled; using PyTorch CPU"
fi

if [[ "${FAST:-0}" == "1" ]]; then
  LIMIT="${LIMIT:-400}"
  SAMPLE_MODE="${SAMPLE_MODE:-balanced}"
  echo "[FAST] Limiting each run to ${LIMIT} logs (sample_mode=${SAMPLE_MODE})"
else
  LIMIT="${LIMIT:-0}"
  SAMPLE_MODE="${SAMPLE_MODE:-head}"
fi

Q_EPISODES="${Q_EPISODES:-80}"
WARMUP="${WARMUP:-3000}"

DATASETS=("Loghub-zenodo_Logs.csv" "Synthetic_Datacenter_Logs.csv")
LABELS=("Loghub-zenodo_Logs" "Synthetic_Datacenter_Logs")

for f in "${DATASETS[@]}"; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "ERROR: Missing dataset $DATA_DIR/$f" >&2; exit 1; }
done

start_ts=$(date +%s)
echo "=== Starting full training + experiments ==="

for i in "${!DATASETS[@]}"; do
  ds="${DATASETS[$i]}"
  label="${LABELS[$i]}"
  base="${ds%.csv}"

  echo
  echo "=================================================================="
  echo "Dataset: $label ($DATA_DIR/$ds)"
  echo "=================================================================="

  # ---- A2C train (per dataset) ----
  A2C_MODEL="trained_models/a2c_${base}.zip"
  if [[ ! -f "$A2C_MODEL" ]]; then
    echo "[A2C] Training model for $label ..."
    python3 -m src.train \
      --log_source real_world \
      --log_filepath "$DATA_DIR/$ds" \
      --timesteps 50000 \
      --model_path "trained_models/a2c_${base}"
    echo "[A2C] Saved $A2C_MODEL"
  else
    echo "[A2C] Reusing existing model $A2C_MODEL"
  fi

  # ---- Q-learning train (shared filenames) ----
  echo "[QLEARN] Training Q-table for $label ..."
  python3 -m src.train_qlearning \
    --log_filepath "$DATA_DIR/$ds" \
    --episodes "$Q_EPISODES" \
    --warmup_steps "$WARMUP" \
    --pca_components 8 \
    --n_bins 6 \
    --eps_decay 0.997 \
    --guided_prob 0.5 \
    --prior_bonus 0.3 \
    --dataset_name "$label" \
    --sample_mode "$SAMPLE_MODE" 

  # ---- Run 6 methods ----
  for router in static a2c q_learning direct_mysql direct_elk direct_ipfs; do
    echo "[RUN] $router on $label ..."
    python3 -m src \
      --router "$router" \
      --log_source real_world \
      --log_filepath "$DATA_DIR/$ds" \
      --model_path "trained_models/a2c_${base}.zip" \
      --sample_mode "$SAMPLE_MODE" \
      --limit "$LIMIT" \
      --sample_mode "$SAMPLE_MODE" 

    out="./results/${router}_${label}.csv"
    sum="./results/summary_${router}_${label}.csv"
    [[ -f "$out" ]] && echo "  -> wrote $out" || echo "  !! missing: $out"
    [[ -f "$sum" ]] && echo "  -> wrote $sum" || true
  done

done

end_ts=$(date +%s)
echo "------------------------------------------------------------------"
echo "All training + experiments finished in $((end_ts - start_ts))s."
echo "Results in $RESULTS_DIR:"
ls -1 "$RESULTS_DIR"
