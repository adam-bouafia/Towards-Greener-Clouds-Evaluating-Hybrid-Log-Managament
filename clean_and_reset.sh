#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}============================================${NC}"
echo -e "${RED}CLEAN AND RESET ALL DATA${NC}"
echo -e "${RED}============================================${NC}"
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will DELETE:${NC}"
echo "  1. All ClickHouse databases and tables"
echo "  2. All MinIO buckets and objects"
echo "  3. Ganache blockchain data (restart fresh)"
echo "  4. All trained models (you'll delete manually)"
echo "  5. All result files (you'll delete manually)"
echo ""
read -p "Are you sure you want to continue? Type 'yes' to proceed: " -r
echo
if [[ ! $REPLY == "yes" ]]
then
    echo "Aborted."
    exit 1
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Step 1: Clean ClickHouse Database${NC}"
echo -e "${BLUE}========================================${NC}"

echo "Dropping 'logs' database..."
curl -s "http://localhost:8123/?query=DROP+DATABASE+IF+EXISTS+logs" && echo ""
echo -e "${GREEN}✅ ClickHouse cleaned${NC}"

# Verify it's gone
echo "Verifying databases..."
curl -s "http://localhost:8123/?query=SHOW+DATABASES" && echo ""

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Step 2: Clean MinIO Buckets${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if mc (MinIO Client) is installed
if ! command -v mc &> /dev/null; then
    echo -e "${YELLOW}MinIO Client (mc) not found. Installing...${NC}"
    wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O /tmp/mc
    chmod +x /tmp/mc
    sudo mv /tmp/mc /usr/local/bin/
    echo -e "${GREEN}✅ MinIO Client installed${NC}"
fi

# Configure MinIO alias (if not already)
mc alias set myminio http://localhost:9002 minioadmin minioadmin 2>/dev/null || true

# List and remove all buckets
echo "Listing buckets..."
mc ls myminio/ 2>/dev/null || echo "No buckets found or MinIO not accessible"

echo "Removing 'logs' bucket if it exists..."
mc rb --force myminio/logs 2>/dev/null || echo "Bucket 'logs' doesn't exist or already removed"

echo -e "${GREEN}✅ MinIO cleaned${NC}"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Step 3: Restart Ganache Blockchain${NC}"
echo -e "${BLUE}========================================${NC}"

# Kill existing Ganache
echo "Stopping any running Ganache instances..."
pkill -f 'ganache.*8545' || echo "No Ganache process found"
sleep 2

# Start fresh Ganache with deterministic accounts
echo "Starting fresh Ganache..."
nohup ganache -d -p 8545 > /tmp/ganache.log 2>&1 &
sleep 3

# Check if Ganache started
if pgrep -f 'ganache.*8545' > /dev/null; then
    echo -e "${GREEN}✅ Ganache blockchain restarted${NC}"
else
    echo -e "${RED}❌ Failed to start Ganache${NC}"
    exit 1
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Step 4: Deploy Fresh Smart Contract${NC}"
echo -e "${BLUE}========================================${NC}"

# Deploy contract
python3 deploy_contract.py > /tmp/deploy.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Smart contract deployed${NC}"
    echo ""
    cat deployment_info.json | python3 -m json.tool
else
    echo -e "${RED}❌ Failed to deploy contract${NC}"
    cat /tmp/deploy.log
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}SYSTEM RESET COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "✅ ClickHouse: All databases dropped"
echo "✅ MinIO: All buckets removed"
echo "✅ Ganache: Fresh blockchain with new contract"
echo ""
echo -e "${YELLOW}📝 MANUAL STEPS REQUIRED:${NC}"
echo ""
echo "1. Delete trained models:"
echo "   rm -rf trained_models/*"
echo ""
echo "2. Delete result files:"
echo "   rm -rf results/*.csv"
echo ""
echo "After manual cleanup, you can run:"
echo "   ./run_all_experiments.sh"
echo ""
