#!/bin/bash
#
# Self-Learning Backtesting Launcher for BTC and ETH
#
# This script launches comprehensive backtesting with self-learning
# on BTCUSDT and ETHUSDT across all time intervals.
#
# Usage:
#   ./start_learning_backtest.sh [options]
#
# Options:
#   --iterations N    Number of learning iterations (default: 3)
#   --candles N       Override candle count for all intervals
#   --balance N       Initial balance (default: 10000)
#   --risk N          Risk percentage per trade (default: 1.0)
#   --no-clickhouse   Skip ClickHouse startup (if already running)
#   --help            Show this help message
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLICKHOUSE_DIR="${SCRIPT_DIR}/clickhouse"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Default parameters - AGGRESSIVE LEARNING (20 iterations, not 3!)
ITERATIONS=20
LEARNING_RATE=0.15
BALANCE=10000
SKIP_CLICKHOUSE=false
STORAGE_PATH="${SCRIPT_DIR}/data/btc_eth_learning"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${SCRIPT_DIR}/logs/learning_backtest_${TIMESTAMP}.log"

# Banner
print_banner() {
    echo -e "${MAGENTA}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║   🤖 SELF-LEARNING BACKTESTING SYSTEM                            ║"
    echo "║                                                                  ║"
    echo "║   Symbols: BTCUSDT, ETHUSDT                                      ║"
    echo "║   Intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d                        ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Self-Learning Backtesting for BTCUSDT and ETHUSDT"
    echo ""
    echo "Options:"
    echo "  --iterations N     Number of learning iterations (default: 20)"
    echo "  --learning-rate N  How fast weights change (default: 0.15)"
    echo "  --balance N        Initial balance in USD (default: 10000)"
    echo "  --storage PATH     Path to store learning data"
    echo "  --no-clickhouse    Skip ClickHouse startup"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run 20 iterations (default)"
    echo "  $0 --iterations 50              # Run 50 learning iterations"
    echo "  $0 --learning-rate 0.25         # More aggressive learning"
    echo ""
}

# Check if ClickHouse is running
is_clickhouse_running() {
    nc -z localhost 9000 &> /dev/null
    return $?
}

# Start ClickHouse
start_clickhouse() {
    echo -e "${YELLOW}Starting ClickHouse server...${NC}"
    
    if is_clickhouse_running; then
        echo -e "${GREEN}✓ ClickHouse is already running${NC}"
        return 0
    fi
    
    if [ ! -d "${CLICKHOUSE_DIR}" ]; then
        echo -e "${RED}✗ ClickHouse directory not found: ${CLICKHOUSE_DIR}${NC}"
        echo -e "${YELLOW}Creating minimal ClickHouse config...${NC}"
        mkdir -p "${CLICKHOUSE_DIR}"
    fi
    
    cd "${CLICKHOUSE_DIR}"
    
    if ! command -v clickhouse &> /dev/null; then
        echo -e "${RED}✗ ClickHouse not found in PATH${NC}"
        echo "Please install ClickHouse: brew install clickhouse"
        exit 1
    fi
    
    # Start ClickHouse in background
    if [ -f "config.xml" ]; then
        clickhouse server --config-file=config.xml &> "${SCRIPT_DIR}/logs/clickhouse_${TIMESTAMP}.log" &
    else
        clickhouse server &> "${SCRIPT_DIR}/logs/clickhouse_${TIMESTAMP}.log" &
    fi
    
    CLICKHOUSE_PID=$!
    
    echo -e "${YELLOW}Waiting for ClickHouse to start (PID: ${CLICKHOUSE_PID})...${NC}"
    
    for i in {1..15}; do
        if is_clickhouse_running; then
            echo -e "${GREEN}✓ ClickHouse started successfully${NC}"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    
    echo ""
    echo -e "${RED}✗ Failed to start ClickHouse within timeout${NC}"
    return 1
}

# Stop ClickHouse on exit
cleanup() {
    if [ "$SKIP_CLICKHOUSE" = false ] && [ -n "$CLICKHOUSE_PID" ]; then
        echo -e "${YELLOW}Stopping ClickHouse...${NC}"
        kill $CLICKHOUSE_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --balance)
            BALANCE="$2"
            shift 2
            ;;
        --storage)
            STORAGE_PATH="$2"
            shift 2
            ;;
        --no-clickhouse)
            SKIP_CLICKHOUSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_banner
    
    # Create logs directory
    mkdir -p "${SCRIPT_DIR}/logs"
    mkdir -p "${STORAGE_PATH}"
    
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Iterations: ${ITERATIONS}"
    echo "  Learning Rate: ${LEARNING_RATE}"
    echo "  Balance: \$${BALANCE}"
    echo "  Storage: ${STORAGE_PATH}"
    echo "  Log File: ${LOG_FILE}"
    echo ""
    
    # Estimate time
    ESTIMATED_TIME=$((ITERATIONS * 2))
    echo -e "${YELLOW}Estimated time: ~${ESTIMATED_TIME} minutes${NC}"
    echo ""
    
    # Start ClickHouse if needed
    if [ "$SKIP_CLICKHOUSE" = false ]; then
        start_clickhouse
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to start ClickHouse. Exiting.${NC}"
            exit 1
        fi
        echo ""
        sleep 2
    else
        echo -e "${YELLOW}Skipping ClickHouse startup (--no-clickhouse)${NC}"
        if ! is_clickhouse_running; then
            echo -e "${RED}Warning: ClickHouse doesn't appear to be running!${NC}"
        fi
        echo ""
    fi
    
    # Set Python path
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
    
    # Check Python
    PYTHON_CMD=""
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}✗ Python not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Using Python: $(${PYTHON_CMD} --version)${NC}"
    echo ""
    
    # Build command
    CMD="${PYTHON_CMD} ${SCRIPT_DIR}/run_learning_backtest.py"
    CMD="${CMD} --iterations ${ITERATIONS}"
    CMD="${CMD} --learning-rate ${LEARNING_RATE}"
    CMD="${CMD} --balance ${BALANCE}"
    CMD="${CMD} --storage ${STORAGE_PATH}"
    
    echo -e "${CYAN}Executing:${NC}"
    echo "  ${CMD}"
    echo ""
    
    # Run backtest
    echo -e "${YELLOW}Starting backtests... (logging to ${LOG_FILE})${NC}"
    echo ""
    
    # Run and tee to both console and log file
    ${CMD} 2>&1 | tee "${LOG_FILE}"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ BACKTESTING COMPLETED SUCCESSFULLY                            ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${CYAN}Results saved to:${NC}"
        echo "  Learning Data: ${STORAGE_PATH}"
        echo "  Weights: ${STORAGE_PATH}/final_weights.json"
        echo "  Report: ${STORAGE_PATH}/backtest_report.json"
        echo "  Log: ${LOG_FILE}"
    else
        echo -e "${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ✗ BACKTESTING FAILED (exit code: ${EXIT_CODE})                          ║${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Check log file for details: ${LOG_FILE}"
    fi
    
    return $EXIT_CODE
}

# Run main
main

