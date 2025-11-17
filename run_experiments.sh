#!/bin/bash
#
# Research Question Experiments Runner
# Easy wrapper for running thesis experiments
#

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Research Question Experiments Runner             ║${NC}"
echo -e "${BLUE}║     Hybrid Log Management Thesis - Answer RQs 1-4        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
else
    source .venv/bin/activate
fi

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "import transformers, torch, xgboost, sklearn" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Missing dependencies. Installing...${NC}"
    pip install -q transformers torch xgboost scikit-learn
}
echo -e "${GREEN}✅ Dependencies OK${NC}"
echo ""

# Parse arguments
MODE="help"
QUICK=""

case "$1" in
    --all)
        MODE="all"
        ;;
    --rq1)
        MODE="rq1"
        ;;
    --rq2)
        MODE="rq2"
        ;;
    --rq3)
        MODE="rq3"
        ;;
    --rq4)
        MODE="rq4"
        ;;
    --help|-h|"")
        MODE="help"
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        MODE="help"
        ;;
esac

# Check for --quick flag
if [ "$2" == "--quick" ] || [ "$1" == "--quick" ]; then
    QUICK="--quick"
    echo -e "${YELLOW}🚀 Quick test mode enabled (1000 logs per experiment)${NC}"
    echo ""
fi

# Display help
if [ "$MODE" == "help" ]; then
    echo "Usage: $0 [OPTION] [--quick]"
    echo ""
    echo "Options:"
    echo "  --all       Run all RQ experiments (RQ1-RQ4)"
    echo "  --rq1       Run RQ1: Semantic vs Basic Features"
    echo "  --rq2       Run RQ2: XGBoost Routing Accuracy"
    echo "  --rq3       Run RQ3: ML vs Baseline Routing"
    echo "  --rq4       Run RQ4: Async Blockchain Performance"
    echo "  --help      Show this help message"
    echo ""
    echo "Flags:"
    echo "  --quick     Quick test mode (1000 logs, ~10-15 minutes)"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Run all experiments (full dataset, 2-4 hours)"
    echo "  $0 --all --quick            # Run all experiments (quick test, 10-15 min)"
    echo "  $0 --rq1                    # Run only RQ1 experiment"
    echo "  $0 --rq3 --quick            # Run only RQ3 (quick test)"
    echo ""
    exit 0
fi

# Run experiments
echo -e "${GREEN}Starting experiments: ${MODE}${NC}"
echo -e "${BLUE}Output will be saved to: results/experiments_TIMESTAMP/${NC}"
echo ""

case "$MODE" in
    all)
        echo -e "${BLUE}Running ALL Research Question Experiments...${NC}"
        python3 -m src.experiments --all $QUICK
        ;;
    rq1)
        echo -e "${BLUE}Running RQ1: Semantic vs. Basic Statistical Features...${NC}"
        python3 -m src.experiments --rq1 $QUICK
        ;;
    rq2)
        echo -e "${BLUE}Running RQ2: XGBoost Routing Accuracy...${NC}"
        python3 -m src.experiments --rq2 $QUICK
        ;;
    rq3)
        echo -e "${BLUE}Running RQ3: ML vs. Baseline Routing...${NC}"
        python3 -m src.experiments --rq3 $QUICK
        ;;
    rq4)
        echo -e "${BLUE}Running RQ4: Async Blockchain Performance...${NC}"
        python3 -m src.experiments --rq4 $QUICK
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              ✅ EXPERIMENTS COMPLETE! ✅                  ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}📁 Results saved to: results/experiments_*/${NC}"
    echo -e "${BLUE}📄 Report generated: results/experiments_*/EXPERIMENT_REPORT.md${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review EXPERIMENT_REPORT.md"
    echo "  2. Analyze CSV files in result directories"
    echo "  3. Use data to answer research questions in thesis"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Experiments failed. Check logs for errors.${NC}"
    exit 1
fi
