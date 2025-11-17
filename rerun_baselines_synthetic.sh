#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Re-running Baseline Experiments (Synthetic)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "This will:"
echo "1. Re-run ClickHouse baseline (200,000 logs) with metadata"
echo "2. Re-run MinIO baseline (200,000 logs) with metadata"
echo "3. Create intelligent training data"
echo "4. Train new XGBoost model"
echo "5. Run XGBoost evaluation with intelligent routing"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Export blockchain config
export WEB3_PROVIDER_URI="http://127.0.0.1:8545"
export CONTRACT_ADDRESS="0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab"
export ACCOUNT_ADDRESS="0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1"
# Do NOT hardcode private keys. Export ACCOUNT_PRIVATE_KEY in your shell before running.
export ACCOUNT_PRIVATE_KEY="${ACCOUNT_PRIVATE_KEY:-}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Step 1: ClickHouse Baseline (w/ metadata)${NC}"
echo -e "${GREEN}========================================${NC}"
START_TIME=$(date +%s)

python3 -m src \
    --router direct_clickhouse \
    --log_source synthetic \
    --limit 200000 \
    --no-blockchain

STEP1_TIME=$(($(date +%s) - START_TIME))
echo -e "${GREEN}✅ Step 1 complete in ${STEP1_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Step 2: MinIO Baseline (w/ metadata)${NC}"
echo -e "${GREEN}========================================${NC}"
START_TIME=$(date +%s)

python3 -m src \
    --router direct_minio \
    --log_source synthetic \
    --limit 200000 \
    --no-blockchain

STEP2_TIME=$(($(date +%s) - START_TIME))
echo -e "${GREEN}✅ Step 2 complete in ${STEP2_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Step 3: Create Intelligent Training Data${NC}"
echo -e "${GREEN}========================================${NC}"

python3 create_training_data.py \
    results/direct_clickhouse_synthetic.csv \
    results/direct_minio_synthetic.csv \
    results/training_data_synthetic_intelligent.csv

echo -e "${GREEN}✅ Step 3 complete${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Step 4: Train Intelligent XGBoost Model${NC}"
echo -e "${GREEN}========================================${NC}"
START_TIME=$(date +%s)

python3 -m src.training.train_xgboost \
    results/training_data_synthetic_intelligent.csv

STEP4_TIME=$(($(date +%s) - START_TIME))
echo -e "${GREEN}✅ Step 4 complete in ${STEP4_TIME}s${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Step 5: Restart Ganache & Deploy Contract${NC}"
echo -e "${GREEN}========================================${NC}"

# Kill existing Ganache
pkill -f 'ganache.*8545' || true
sleep 2

# Start fresh Ganache
nohup ganache -d -p 8545 > /tmp/ganache.log 2>&1 &
sleep 3

# Deploy contract
python3 deploy_contract.py > /tmp/deploy.log 2>&1
echo -e "${GREEN}✅ Blockchain ready${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Step 6: XGBoost with Intelligent Routing${NC}"
echo -e "${GREEN}========================================${NC}"
START_TIME=$(date +%s)

python3 -m src \
    --router xgboost \
    --log_source synthetic \
    --limit 200000

STEP6_TIME=$(($(date +%s) - START_TIME))
echo -e "${GREEN}✅ Step 6 complete in ${STEP6_TIME}s${NC}"

# Summary
TOTAL_TIME=$((STEP1_TIME + STEP2_TIME + STEP4_TIME + STEP6_TIME))
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Evaluation Complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Step 1 (ClickHouse baseline): ${STEP1_TIME}s"
echo "Step 2 (MinIO baseline): ${STEP2_TIME}s"
echo "Step 4 (Training): ${STEP4_TIME}s"
echo "Step 6 (XGBoost integrated): ${STEP6_TIME}s"
echo "Total time: ${TOTAL_TIME}s"
echo ""
echo "Results:"
echo "  - results/direct_clickhouse_synthetic.csv"
echo "  - results/direct_minio_synthetic.csv"
echo "  - results/training_data_synthetic_intelligent.csv"
echo "  - trained_models/xgboost_router.json"
echo "  - results/xgboost_synthetic.csv"
echo "  - results/summary_xgboost_synthetic.csv"
echo ""
echo "Next: Compare results to see intelligent routing distribution"
