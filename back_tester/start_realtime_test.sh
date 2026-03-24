#!/bin/bash

# Real-Time Trading Strategy Tester
# This script monitors multiple symbol/interval combinations using real-time data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          REAL-TIME TRADING STRATEGY TESTER                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Default values
BALANCE=10000
RISK=1.0
DURATION=""
SAFE_ONLY=""
NO_TRAILING=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --balance)
            BALANCE="$2"
            shift 2
            ;;
        --risk)
            RISK="$2"
            shift 2
            ;;
        --duration)
            DURATION="--duration $2"
            shift 2
            ;;
        --safe-only)
            SAFE_ONLY="--safe-only"
            shift
            ;;
        --no-trailing)
            NO_TRAILING="--no-trailing"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --balance VALUE    Initial balance (default: 10000)"
            echo "  --risk VALUE       Risk percentage per trade (default: 1.0)"
            echo "  --duration HOURS   Test duration in hours (default: unlimited)"
            echo "  --safe-only        Only use safe combinations (BTCUSDT 1h, ETHUSDT 5m/15m)"
            echo "  --no-trailing      Disable trailing stop"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run with defaults (all combinations)"
            echo "  $0 --safe-only               # Only safe combinations"
            echo "  $0 --duration 2              # Run for 2 hours"
            echo "  $0 --balance 5000 --risk 2   # Custom balance and risk"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Initial Balance: \$${BALANCE}"
echo "  Risk per Trade:  ${RISK}%"
echo "  Duration:        ${DURATION:-Unlimited (Ctrl+C to stop)}"
echo "  Mode:            ${SAFE_ONLY:-All Combinations}"
echo "  Trailing Stop:   ${NO_TRAILING:-Enabled}"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the real-time tester
echo -e "${GREEN}Starting real-time trading test...${NC}"
echo "Press Ctrl+C to stop"
echo ""

python3 "$SCRIPT_DIR/realtime_test.py" \
    --balance "$BALANCE" \
    --risk "$RISK" \
    $DURATION \
    $SAFE_ONLY \
    $NO_TRAILING

echo ""
echo -e "${GREEN}Test completed.${NC}"








