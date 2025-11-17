#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Export blockchain config (Ganache local)
export POLYGON_RPC_URL="http://127.0.0.1:8545"
export BLOCKCHAIN_CONTRACT_ADDRESS="0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab"
# IMPORTANT: Do NOT hardcode private keys in scripts. Set your private key in the environment
# before running the script. For local Ganache you can export the development key locally,
# but avoid committing any secret to the repository.
# If BLOCKCHAIN_PRIVATE_KEY is not set, the script will attempt to operate without sending
# signed transactions (read-only checks may still work depending on the code path).
export BLOCKCHAIN_PRIVATE_KEY="${BLOCKCHAIN_PRIVATE_KEY:-}"

# Legacy aliases (for backward compatibility)
export WEB3_PROVIDER_URI="$POLYGON_RPC_URL"
export CONTRACT_ADDRESS="$BLOCKCHAIN_CONTRACT_ADDRESS"
export ACCOUNT_ADDRESS="0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1"
export ACCOUNT_PRIVATE_KEY="$BLOCKCHAIN_PRIVATE_KEY"

# Start time tracking
TOTAL_START=$(date +%s)

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}COMPLETE EVALUATION - LOGHUB + SYNTHETIC${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${GREEN}Blockchain: Ganache (Local)${NC}"
echo "  RPC URL: $POLYGON_RPC_URL"
echo "  Contract: $BLOCKCHAIN_CONTRACT_ADDRESS"
echo ""
echo "This will run:"
echo "  • Loghub dataset (14,000 logs)"
echo "    - ClickHouse baseline"
echo "    - MinIO baseline"
echo "    - Create training data"
echo "    - Train XGBoost model"
echo "    - XGBoost integrated with blockchain"
echo ""
echo "  • Synthetic dataset (200,000 logs)"
echo "    - ClickHouse baseline"
echo "    - MinIO baseline"
echo "    - Create training data"
echo "    - Train XGBoost model"
echo "    - XGBoost integrated with blockchain"
echo ""
echo "Estimated total time: ~8-10 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# ============================================
# LOGHUB DATASET (14,000 logs)
# ============================================

echo -e "\n${BLUE}############################################${NC}"
echo -e "${BLUE}## LOGHUB DATASET EVALUATION (14K logs)${NC}"
echo -e "${BLUE}############################################${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Loghub Step 1/5: ClickHouse Baseline${NC}"
echo -e "${GREEN}========================================${NC}"
STEP_START=$(date +%s)

python3 -m src \
    --router direct_clickhouse \
    --log_source loghub \
    --limit 14000 \
    --no-blockchain

LOGHUB_CH_TIME=$(($(date +%s) - STEP_START))
echo -e "${GREEN}✅ Loghub ClickHouse baseline: ${LOGHUB_CH_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Loghub Step 2/5: MinIO Baseline${NC}"
echo -e "${GREEN}========================================${NC}"
STEP_START=$(date +%s)

python3 -m src \
    --router direct_minio \
    --log_source loghub \
    --limit 14000 \
    --no-blockchain

LOGHUB_MINIO_TIME=$(($(date +%s) - STEP_START))
echo -e "${GREEN}✅ Loghub MinIO baseline: ${LOGHUB_MINIO_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Loghub Step 3/5: Create Training Data${NC}"
echo -e "${GREEN}========================================${NC}"

python3 create_training_data.py \
    results/direct_clickhouse_loghub.csv \
    results/direct_minio_loghub.csv \
    results/training_data_loghub.csv

echo -e "${GREEN}✅ Loghub training data created${NC}"

# Check training data distribution
echo ""
echo "Training data distribution:"
tail -n +2 results/training_data_loghub.csv | cut -d',' -f15 | sort | uniq -c
echo ""
echo "Routing reasons:"
tail -n +2 results/training_data_loghub.csv | cut -d',' -f16 | sort | uniq -c

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Loghub Step 4/5: Train XGBoost Model${NC}"
echo -e "${GREEN}========================================${NC}"
STEP_START=$(date +%s)

python3 -m src.training.train_xgboost \
    results/training_data_loghub.csv

LOGHUB_TRAIN_TIME=$(($(date +%s) - STEP_START))
echo -e "${GREEN}✅ Loghub XGBoost trained: ${LOGHUB_TRAIN_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Loghub Step 5/5: XGBoost with Blockchain${NC}"
echo -e "${GREEN}========================================${NC}"

# Restart blockchain for clean state
echo "Restarting blockchain..."
pkill -f 'ganache.*8545' || true
sleep 2
nohup ganache -d -p 8545 > /tmp/ganache.log 2>&1 &
sleep 3
python3 deploy_contract.py > /tmp/deploy.log 2>&1
echo -e "${GREEN}✅ Blockchain ready${NC}"

STEP_START=$(date +%s)

python3 -m src \
    --router xgboost \
    --log_source loghub \
    --limit 14000

LOGHUB_XGBOOST_TIME=$(($(date +%s) - STEP_START))
echo -e "${GREEN}✅ Loghub XGBoost evaluation: ${LOGHUB_XGBOOST_TIME}s${NC}"

# Show Loghub routing distribution
echo ""
echo "Loghub XGBoost routing distribution:"
tail -n +2 results/xgboost_loghub.csv | cut -d',' -f4 | sort | uniq -c

# ============================================
# SYNTHETIC DATASET (200,000 logs)
# ============================================

echo -e "\n${BLUE}############################################${NC}"
echo -e "${BLUE}## SYNTHETIC DATASET EVALUATION (200K logs)${NC}"
echo -e "${BLUE}############################################${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Synthetic Step 1/5: ClickHouse Baseline${NC}"
echo -e "${GREEN}========================================${NC}"
STEP_START=$(date +%s)

python3 -m src \
    --router direct_clickhouse \
    --log_source synthetic \
    --limit 200000 \
    --no-blockchain

SYNTHETIC_CH_TIME=$(($(date +%s) - STEP_START))
echo -e "${GREEN}✅ Synthetic ClickHouse baseline: ${SYNTHETIC_CH_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Synthetic Step 2/5: MinIO Baseline${NC}"
echo -e "${GREEN}========================================${NC}"
STEP_START=$(date +%s)

python3 -m src \
    --router direct_minio \
    --log_source synthetic \
    --limit 200000 \
    --no-blockchain

SYNTHETIC_MINIO_TIME=$(($(date +%s) - STEP_START))
echo -e "${GREEN}✅ Synthetic MinIO baseline: ${SYNTHETIC_MINIO_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Synthetic Step 3/5: Create Training Data${NC}"
echo -e "${GREEN}========================================${NC}"

python3 create_training_data.py \
    results/direct_clickhouse_synthetic.csv \
    results/direct_minio_synthetic.csv \
    results/training_data_synthetic.csv

echo -e "${GREEN}✅ Synthetic training data created${NC}"

# Check training data distribution
echo ""
echo "Training data distribution:"
tail -n +2 results/training_data_synthetic.csv | cut -d',' -f15 | sort | uniq -c
echo ""
echo "Routing reasons:"
tail -n +2 results/training_data_synthetic.csv | cut -d',' -f16 | sort | uniq -c

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Synthetic Step 4/5: Train XGBoost Model${NC}"
echo -e "${GREEN}========================================${NC}"
STEP_START=$(date +%s)

python3 -m src.training.train_xgboost \
    results/training_data_synthetic.csv

SYNTHETIC_TRAIN_TIME=$(($(date +%s) - STEP_START))
echo -e "${GREEN}✅ Synthetic XGBoost trained: ${SYNTHETIC_TRAIN_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Synthetic Step 5/5: XGBoost with Blockchain${NC}"
echo -e "${GREEN}========================================${NC}"

# Restart blockchain for clean state
echo "Restarting blockchain..."
pkill -f 'ganache.*8545' || true
sleep 2
nohup ganache -d -p 8545 > /tmp/ganache.log 2>&1 &
sleep 3
python3 deploy_contract.py > /tmp/deploy.log 2>&1
echo -e "${GREEN}✅ Blockchain ready${NC}"

STEP_START=$(date +%s)

python3 -m src \
    --router xgboost \
    --log_source synthetic \
    --limit 200000

SYNTHETIC_XGBOOST_TIME=$(($(date +%s) - STEP_START))
echo -e "${GREEN}✅ Synthetic XGBoost evaluation: ${SYNTHETIC_XGBOOST_TIME}s${NC}"

# Show Synthetic routing distribution
echo ""
echo "Synthetic XGBoost routing distribution:"
tail -n +2 results/xgboost_synthetic.csv | cut -d',' -f4 | sort | uniq -c

# ============================================
# FINAL SUMMARY
# ============================================

TOTAL_TIME=$(($(date +%s) - TOTAL_START))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}ALL EXPERIMENTS COMPLETE!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "LOGHUB (14,000 logs):"
echo "  Step 1 - ClickHouse baseline:  ${LOGHUB_CH_TIME}s"
echo "  Step 2 - MinIO baseline:       ${LOGHUB_MINIO_TIME}s"
echo "  Step 4 - XGBoost training:     ${LOGHUB_TRAIN_TIME}s"
echo "  Step 5 - XGBoost integrated:   ${LOGHUB_XGBOOST_TIME}s"
echo ""
echo "SYNTHETIC (200,000 logs):"
echo "  Step 1 - ClickHouse baseline:  ${SYNTHETIC_CH_TIME}s"
echo "  Step 2 - MinIO baseline:       ${SYNTHETIC_MINIO_TIME}s"
echo "  Step 4 - XGBoost training:     ${SYNTHETIC_TRAIN_TIME}s"
echo "  Step 5 - XGBoost integrated:   ${SYNTHETIC_XGBOOST_TIME}s"
echo ""
echo "Total execution time: ${HOURS}h ${MINUTES}m (${TOTAL_TIME}s)"
echo ""
echo "Results saved in:"
echo "  LOGHUB:"
echo "    - results/direct_clickhouse_loghub.csv"
echo "    - results/direct_minio_loghub.csv"
echo "    - results/training_data_loghub.csv"
echo "    - results/xgboost_loghub.csv"
echo "    - results/summary_xgboost_loghub.csv"
echo ""
echo "  SYNTHETIC:"
echo "    - results/direct_clickhouse_synthetic.csv"
echo "    - results/direct_minio_synthetic.csv"
echo "    - results/training_data_synthetic.csv"
echo "    - results/xgboost_synthetic.csv"
echo "    - results/summary_xgboost_synthetic.csv"
echo ""
echo "  MODELS:"
echo "    - trained_models/xgboost_router.json"
echo "    - trained_models/xgboost_router_encoders.pkl"
echo ""
echo -e "${GREEN}✅ Ready for analysis and thesis writing!${NC}"
