#!/bin/bash

# Monitor the Synthetic dataset evaluation progress

echo "=========================================="
echo "Synthetic Dataset Evaluation Monitor"
echo "=========================================="
echo ""

# Check if evaluation is running
if ps aux | grep -q "[p]ython3.*synthetic"; then
    echo "✅ Evaluation is RUNNING"
    echo ""
    
    # Show current step
    echo "Current Process:"
    ps aux | grep "[p]ython3.*synthetic" | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Memory:", $6/1024"MB", "| Command:", $11, $12, $13, $14, $15}'
    echo ""
    
    # Show last 15 lines of log
    echo "Recent Log Output:"
    echo "----------------------------------------"
    tail -15 evaluation_synthetic.log
    echo "----------------------------------------"
    echo ""
    
    # Estimate progress based on which step
    if grep -q "Step 1/5" evaluation_synthetic.log | tail -1; then
        echo "📊 Current Step: Step 1 - ClickHouse Baseline"
        echo "   Expected: ~2.5-3 hours"
    elif grep -q "Step 2/5" evaluation_synthetic.log | tail -1; then
        echo "📊 Current Step: Step 2 - MinIO Baseline"
        echo "   Expected: ~30-40 minutes"
    elif grep -q "Step 3/5" evaluation_synthetic.log | tail -1; then
        echo "📊 Current Step: Step 3 - Training Data Creation"
        echo "   Expected: <1 minute"
    elif grep -q "Step 4/5" evaluation_synthetic.log | tail -1; then
        echo "📊 Current Step: Step 4 - XGBoost Training"
        echo "   Expected: ~10-15 minutes"
    elif grep -q "Step 5/5" evaluation_synthetic.log | tail -1; then
        echo "📊 Current Step: Step 5 - XGBoost with Blockchain"
        echo "   Expected: ~2.5-3 hours"
    fi
    
else
    echo "⏸️  Evaluation is NOT running"
    echo ""
    
    # Check if it completed
    if grep -q "Evaluation Complete!" evaluation_synthetic.log 2>/dev/null; then
        echo "✅ Evaluation COMPLETED!"
        echo ""
        echo "Results available at:"
        echo "  - results/summary_direct_clickhouse_synthetic.csv"
        echo "  - results/summary_direct_minio_synthetic.csv"
        echo "  - results/summary_xgboost_synthetic.csv"
    else
        echo "❌ Evaluation may have stopped unexpectedly"
        echo ""
        echo "Last 20 lines of log:"
        echo "----------------------------------------"
        tail -20 evaluation_synthetic.log 2>/dev/null || echo "No log file found"
        echo "----------------------------------------"
    fi
fi

echo ""
echo "Commands:"
echo "  - Watch live: tail -f evaluation_synthetic.log"
echo "  - Full log: cat evaluation_synthetic.log"
echo "  - This monitor: ./monitor_synthetic.sh"
