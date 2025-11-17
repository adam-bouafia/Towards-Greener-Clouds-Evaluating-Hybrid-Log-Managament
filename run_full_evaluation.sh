#!/bin/bash
# Full Evaluation Workflow for Loghub Dataset (14,000 logs)
# This script runs all 5 steps of the thesis experiment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Full Thesis Evaluation - Loghub Dataset${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate virtual environment
source venv/bin/activate

# Blockchain configuration - Ganache Local Blockchain
# Note: Blockchain is DISABLED for baseline experiments (Steps 1-4) to measure pure storage performance
#       Blockchain is ENABLED only for Step 5 (integrated system evaluation)
export POLYGON_RPC_URL="http://127.0.0.1:8545"
export BLOCKCHAIN_CONTRACT_ADDRESS="0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab"
# Do NOT hardcode private keys. Export BLOCKCHAIN_PRIVATE_KEY in your shell before running this script.
export BLOCKCHAIN_PRIVATE_KEY="${BLOCKCHAIN_PRIVATE_KEY:-}"

# Step 1: Run baseline ClickHouse (all to hot storage) - NO BLOCKCHAIN
echo -e "${GREEN}Step 1/5: Running ClickHouse Baseline (14,000 logs)...${NC}"
echo "Measuring pure ClickHouse performance (blockchain disabled)"
echo "Expected duration: ~20 minutes"
START_TIME=$(date +%s)
python3 -m src --router direct_clickhouse --log_source loghub --limit 14000 --no-blockchain
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ ClickHouse baseline completed in ${DURATION}s${NC}"
echo ""

# Step 2: Run baseline MinIO (all to cold storage) - NO BLOCKCHAIN
echo -e "${GREEN}Step 2/5: Running MinIO Baseline (14,000 logs)...${NC}"
echo "Measuring pure MinIO performance (blockchain disabled)"
echo "Expected duration: ~25 minutes"
START_TIME=$(date +%s)
python3 -m src --router direct_minio --log_source loghub --limit 14000 --no-blockchain
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ MinIO baseline completed in ${DURATION}s${NC}"
echo ""

# Step 3: Create training data (label logs with best backend) - NO BLOCKCHAIN
echo -e "${GREEN}Step 3/5: Creating Training Data...${NC}"
echo "Expected duration: <1 minute"
START_TIME=$(date +%s)
python3 create_training_data.py
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ Training data created in ${DURATION}s${NC}"
echo ""

# Step 4: Train XGBoost model - NO BLOCKCHAIN
echo -e "${GREEN}Step 4/5: Training XGBoost Model...${NC}"
echo "Expected duration: ~2-3 minutes"
START_TIME=$(date +%s)
python3 -m src.training.train_xgboost results/training_data_loghub.csv
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ XGBoost model trained in ${DURATION}s${NC}"
echo ""

# Step 5: Run XGBoost experiment (intelligent routing) - WITH BLOCKCHAIN
echo -e "${GREEN}Step 5/5: Running XGBoost Experiment with Blockchain (14,000 logs)...${NC}"
echo "Testing complete system: XGBoost routing + selective blockchain verification"
echo "Expected duration: ~20-25 minutes"
echo ""
echo "🔄 Restarting Ganache blockchain (fresh state for Step 5)..."
pkill -f 'ganache.*8545' 2>/dev/null || true
sleep 2
nohup ganache -d -p 8545 > /tmp/ganache.log 2>&1 &
sleep 3
echo "✅ Ganache restarted"
echo ""
echo "📜 Redeploying smart contract..."
python3 deploy_contract.py > /tmp/deploy.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Smart contract deployed successfully"
else
    echo "❌ Contract deployment failed! Check /tmp/deploy.log"
    exit 1
fi
echo ""
START_TIME=$(date +%s)
python3 -m src --router xgboost --log_source loghub --limit 14000
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ XGBoost experiment completed in ${DURATION}s${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved in results/ directory:"
echo "  - results/direct_clickhouse_loghub.csv"
echo "  - results/direct_minio_loghub.csv"
echo "  - results/xgboost_loghub.csv"
echo "  - results/summary_direct_clickhouse_loghub.csv"
echo "  - results/summary_direct_minio_loghub.csv"
echo "  - results/summary_xgboost_loghub.csv"
echo ""
echo "Next steps:"
echo "  1. Analyze summary files for thesis tables"
echo "  2. Compare avg latency, energy, blockchain metrics"
echo "  3. Check XGBoost routing accuracy"
echo ""
echo "View results with: cat results/summary_*.csv"
