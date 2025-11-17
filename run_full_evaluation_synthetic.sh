#!/bin/bash

# Full Evaluation Script for Synthetic Dataset
# Runs all 5 steps of the thesis evaluation with Synthetic_Datacenter_Logs.csv (200,000 logs)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Full Evaluation - Synthetic Dataset${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Dataset: Synthetic_Datacenter_Logs.csv (200,000 logs)"
echo "Steps: 5 (Baselines + Training + XGBoost with Blockchain)"
echo "Expected total time: ~3.5-4 hours"
echo ""

# Blockchain configuration - Ganache Local Blockchain
# Note: Blockchain is DISABLED for baseline experiments (Steps 1-4) to measure pure storage performance
#       Blockchain is ENABLED only for Step 5 (integrated system evaluation)
export POLYGON_RPC_URL="http://127.0.0.1:8545"
export BLOCKCHAIN_CONTRACT_ADDRESS="0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab"
# Do NOT hardcode private keys. Export BLOCKCHAIN_PRIVATE_KEY in your shell before running this script.
export BLOCKCHAIN_PRIVATE_KEY="${BLOCKCHAIN_PRIVATE_KEY:-}"

# Step 1: Run baseline ClickHouse (all to hot storage) - NO BLOCKCHAIN
echo -e "${GREEN}Step 1/5: Running ClickHouse Baseline (200,000 logs)...${NC}"
echo "Measuring pure ClickHouse performance (blockchain disabled)"
echo "Expected duration: ~2.5-3 hours"
START_TIME=$(date +%s)
python3 -m src --router direct_clickhouse --log_source synthetic --limit 200000 --no-blockchain
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ ClickHouse baseline completed in ${DURATION}s${NC}"
echo ""

# Step 2: Run baseline MinIO (all to cold storage) - NO BLOCKCHAIN
echo -e "${GREEN}Step 2/5: Running MinIO Baseline (200,000 logs)...${NC}"
echo "Measuring pure MinIO performance (blockchain disabled)"
echo "Expected duration: ~30-40 minutes"
START_TIME=$(date +%s)
python3 -m src --router direct_minio --log_source synthetic --limit 200000 --no-blockchain
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ MinIO baseline completed in ${DURATION}s${NC}"
echo ""

# Step 3: Create training data from baseline results
echo -e "${GREEN}Step 3/5: Creating Training Data...${NC}"
echo "Expected duration: <1 minute"
START_TIME=$(date +%s)
python3 << 'PYTHON_EOF'
import pandas as pd
from pathlib import Path
import sys

print("=" * 80)
print("📊 Creating Training Data for XGBoost Router")
print("=" * 80)

# File paths
clickhouse_csv = "results/direct_clickhouse_synthetic.csv"
minio_csv = "results/direct_minio_synthetic.csv"
output_csv = "results/training_data_synthetic.csv"

# Check files exist
if not Path(clickhouse_csv).exists():
    print(f"❌ File not found: {clickhouse_csv}")
    sys.exit(1)

if not Path(minio_csv).exists():
    print(f"❌ File not found: {minio_csv}")
    sys.exit(1)

# Read both baseline results
print(f"\n📁 Reading {clickhouse_csv}...")
clickhouse_df = pd.read_csv(clickhouse_csv)

print(f"📁 Reading {minio_csv}...")
minio_df = pd.read_csv(minio_csv)

# Ensure same number of logs
if len(clickhouse_df) != len(minio_df):
    print(f"❌ Baseline results have different lengths:")
    print(f"   ClickHouse: {len(clickhouse_df)} logs")
    print(f"   MinIO: {len(minio_df)} logs")
    sys.exit(1)

print(f"✅ Both files have {len(clickhouse_df)} logs")

# Create combined dataset
training_data = []

print("\n🔄 Labeling logs with best backend...")

for i in range(len(clickhouse_df)):
    ch_row = clickhouse_df.iloc[i]
    minio_row = minio_df.iloc[i]
    
    # Get latencies (backend write latency)
    ch_latency = ch_row.get('backend_write_latency_ms', 0.0)
    minio_latency = minio_row.get('backend_write_latency_ms', 0.0)
    
    # Choose best backend (lowest latency)
    best_backend = 'clickhouse' if ch_latency <= minio_latency else 'minio'
    
    # Create training sample
    sample = {
        'LogID': i,
        'Level': ch_row.get('Level', 'INFO'),
        'Component': ch_row.get('Component', 'unknown'),
        'LogSource': ch_row.get('LogSource', 'synthetic'),
        'Content': ch_row.get('Content', ''),
        'EventTemplate': ch_row.get('EventTemplate', ''),
        'clickhouse_latency': ch_latency,
        'minio_latency': minio_latency,
        'latency_diff': abs(ch_latency - minio_latency),
        'best_backend': best_backend
    }
    
    training_data.append(sample)

# Create DataFrame
training_df = pd.DataFrame(training_data)

# Statistics
total = len(training_df)
clickhouse_count = (training_df['best_backend'] == 'clickhouse').sum()
minio_count = (training_df['best_backend'] == 'minio').sum()

print("\n" + "=" * 80)
print("📊 Training Data Statistics")
print("=" * 80)
print(f"Total samples: {total}")
print(f"Best backend = clickhouse: {clickhouse_count} ({100*clickhouse_count/total:.1f}%)")
print(f"Best backend = minio: {minio_count} ({100*minio_count/total:.1f}%)")
print(f"\nAvg latency difference: {training_df['latency_diff'].mean():.2f} ms")
print(f"Max latency difference: {training_df['latency_diff'].max():.2f} ms")

# Save
training_df.to_csv(output_csv, index=False)
print(f"\n✅ Training data saved to {output_csv}")
print(f"   Columns: {list(training_df.columns)}")
print(f"   Ready for: python -m src.training.train_xgboost {output_csv}")
print("=" * 80)
PYTHON_EOF
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ Training data created in ${DURATION}s${NC}"
echo ""

# Step 4: Train XGBoost model
echo -e "${GREEN}Step 4/5: Training XGBoost Model...${NC}"
echo "Expected duration: ~10-15 minutes"
START_TIME=$(date +%s)
python3 -m src.training.train_xgboost results/training_data_synthetic.csv
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}✓ XGBoost model trained in ${DURATION}s${NC}"
echo ""

# Step 5: Run XGBoost experiment (intelligent routing) - WITH BLOCKCHAIN
echo -e "${GREEN}Step 5/5: Running XGBoost Experiment with Blockchain (200,000 logs)...${NC}"
echo "Testing complete system: XGBoost routing + selective blockchain verification"
echo "Expected duration: ~2.5-3 hours"
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
python3 -m src --router xgboost --log_source synthetic --limit 200000
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
echo "  - results/direct_clickhouse_synthetic.csv"
echo "  - results/direct_minio_synthetic.csv"
echo "  - results/xgboost_synthetic.csv"
echo "  - results/summary_direct_clickhouse_synthetic.csv"
echo "  - results/summary_direct_minio_synthetic.csv"
echo "  - results/summary_xgboost_synthetic.csv"
echo ""
echo "Next steps:"
echo "  1. Analyze summary files for thesis tables"
echo "  2. Compare avg latency, energy, blockchain metrics"
echo "  3. Check XGBoost routing accuracy"
echo ""
echo "View results with: cat results/summary_*_synthetic.csv"
