#!/bin/bash
# View the most recent experiment results

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Finding most recent experiment results...${NC}"

# Find most recent results directory
LATEST_DIR=$(find results -maxdepth 1 -type d -name "experiments_*" 2>/dev/null | sort -r | head -n 1)

if [ -z "$LATEST_DIR" ]; then
    echo -e "${RED}❌ No experiment results found${NC}"
    echo ""
    echo "Run experiments first:"
    echo "  ./run_experiments.sh --all --quick"
    exit 1
fi

echo -e "${GREEN}✓ Found: $LATEST_DIR${NC}"
echo ""

# Check for report
REPORT="$LATEST_DIR/EXPERIMENT_REPORT.md"
if [ -f "$REPORT" ]; then
    echo -e "${BLUE}📊 Experiment Report:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Show summary section
    if command -v bat &> /dev/null; then
        bat --style=plain --pager=never "$REPORT" 2>/dev/null || cat "$REPORT"
    else
        cat "$REPORT"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo -e "${YELLOW}⚠ No report found in $LATEST_DIR${NC}"
fi

# Check for master summary
SUMMARY="$LATEST_DIR/master_summary.json"
if [ -f "$SUMMARY" ]; then
    echo ""
    echo -e "${BLUE}📈 Quick Stats:${NC}"
    
    # Extract key metrics if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        
        # RQ1 Results
        if [ -f "$LATEST_DIR/rq1_semantic_vs_basic/comparison_summary.json" ]; then
            echo -e "${GREEN}RQ1 - Basic vs Semantic:${NC}"
            jq -r '.comparison | 
                "  Basic Accuracy:    \(.basic_accuracy * 100 | floor)%\n" +
                "  Semantic Accuracy: \(.semantic_accuracy * 100 | floor)%\n" +
                "  Improvement:       +\(.improvement_percent | floor)%"' \
                "$LATEST_DIR/rq1_semantic_vs_basic/comparison_summary.json" 2>/dev/null || \
                echo "  (Results parsing failed)"
            echo ""
        fi
        
        # RQ2 Results
        if [ -f "$LATEST_DIR/rq2_xgboost_accuracy/accuracy_metrics.json" ]; then
            echo -e "${GREEN}RQ2 - XGBoost Accuracy:${NC}"
            jq -r '"  Accuracy:     \(.accuracy * 100 | floor)%\n" +
                "  Hot Ratio:    \(.hot_ratio * 100 | floor)%\n" +
                "  Cold Ratio:   \(.cold_ratio * 100 | floor)%\n" +
                "  Cost Savings: \(.cost_savings * 100 | floor)%"' \
                "$LATEST_DIR/rq2_xgboost_accuracy/accuracy_metrics.json" 2>/dev/null || \
                echo "  (Results parsing failed)"
            echo ""
        fi
        
        # RQ3 Results
        if [ -f "$LATEST_DIR/rq3_ml_vs_baseline/comparison_summary.json" ]; then
            echo -e "${GREEN}RQ3 - ML vs Baseline:${NC}"
            jq -r '.routers[] | 
                "  \(.name):\n" +
                "    Accuracy: \(.accuracy * 100 | floor)%\n" +
                "    Cost Savings: \(.cost_savings * 100 | floor)%"' \
                "$LATEST_DIR/rq3_ml_vs_baseline/comparison_summary.json" 2>/dev/null || \
                echo "  (Results parsing failed)"
            echo ""
        fi
        
        # RQ4 Results
        if [ -f "$LATEST_DIR/rq4_async_blockchain/overhead_summary.json" ]; then
            echo -e "${GREEN}RQ4 - Blockchain Overhead:${NC}"
            jq -r '"  Baseline Latency:   \(.baseline_p50_ms)ms (P50)\n" +
                "  Blockchain Latency: \(.blockchain_p50_ms)ms (P50)\n" +
                "  Overhead:           +\(.overhead_ms)ms"' \
                "$LATEST_DIR/rq4_async_blockchain/overhead_summary.json" 2>/dev/null || \
                echo "  (Results parsing failed)"
            echo ""
        fi
    else
        echo "  (Install 'jq' for formatted stats display)"
        echo ""
        echo "Raw summary available at:"
        echo "  $SUMMARY"
    fi
else
    echo -e "${YELLOW}⚠ No master summary found${NC}"
fi

# Show directory structure
echo ""
echo -e "${BLUE}📁 Results Directory Structure:${NC}"
echo ""
tree -L 2 "$LATEST_DIR" 2>/dev/null || ls -lah "$LATEST_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}Full results location:${NC}"
echo "  $LATEST_DIR"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  • Review detailed report: less '$REPORT'"
echo "  • Check individual RQ results: cd '$LATEST_DIR'"
echo "  • Generate plots: python scripts/plot_results.py '$LATEST_DIR'"
echo "  • Copy tables to thesis: Use metrics from JSON files"
echo ""
