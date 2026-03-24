#!/bin/bash
#
# Start Live Trading Synchronization with Bybit
#
# This script starts the live trading system that synchronizes
# realtime test signals with your actual Bybit account.
#
# Usage:
#   ./start_live_trading.sh [testnet|live] [options]
#
# Examples:
#   ./start_live_trading.sh testnet                   # Start on testnet
#   ./start_live_trading.sh live --safe-only          # Live trading, safe pairs only
#   ./start_live_trading.sh testnet --no-telegram     # Disable Telegram alerts
#   ./start_live_trading.sh testnet --telegram-id 123 # Custom Telegram ID
#

# Change to the script directory
cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           BYBIT LIVE TRADING SYNCHRONIZATION                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check for .env file
if [ ! -f "../.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found in project root${NC}"
    echo "Make sure BYBIT_API_KEY and BYBIT_API_SECRET are set as environment variables"
    echo ""
fi

# Default values
MODE="testnet"
EXTRA_ARGS=""

# Parse first argument as mode
if [ "$1" == "live" ] || [ "$1" == "mainnet" ]; then
    MODE="live"
    shift
elif [ "$1" == "testnet" ] || [ "$1" == "test" ]; then
    MODE="testnet"
    shift
fi

# Collect remaining arguments
EXTRA_ARGS="$@"

# Build command
if [ "$MODE" == "testnet" ]; then
    echo -e "${GREEN}Starting in TESTNET mode${NC}"
    echo ""
    CMD="python bybit_sync.py --testnet $EXTRA_ARGS"
else
    echo -e "${RED}⚠️  WARNING: Starting in LIVE TRADING mode!${NC}"
    echo -e "${RED}   This will execute REAL trades with REAL money!${NC}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
    CMD="python bybit_sync.py --confirm $EXTRA_ARGS"
fi

echo "Command: $CMD"
echo ""
echo "═════════════════════════════════════════════════════════════════"
echo ""

# Run the trading script
$CMD

