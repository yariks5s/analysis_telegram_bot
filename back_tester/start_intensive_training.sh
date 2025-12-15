#!/bin/bash
#
# Intensive Parameter Optimization Launcher
#
# Optimizes ALL trading parameters:
# - 18 Signal Weights
# - Stop Loss (ATR multiplier, min distance)
# - Take Profit Levels (TP1, TP2, TP3 ratios)
# - Trailing Stop (distance, activation)
# - Risk Percentage
#
# Usage:
#   ./start_intensive_training.sh                    # Default: 2 hours
#   ./start_intensive_training.sh --hours 8          # Run for 8 hours
#   ./start_intensive_training.sh --generations 100  # Run 100 generations
#

set -e

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

# Defaults
GENERATIONS=100
HOURS=4
POPULATION=8
TESTS=10
BALANCE=10000
STORAGE_PATH="${SCRIPT_DIR}/data/intensive_training"
SKIP_CLICKHOUSE=false

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${SCRIPT_DIR}/logs/intensive_training_${TIMESTAMP}.log"

print_banner() {
    echo -e "${MAGENTA}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║   🧬 INTENSIVE PARAMETER OPTIMIZATION                           ║"
    echo "║                                                                  ║"
    echo "║   Optimizing:                                                   ║"
    echo "║   • 18 Signal Weights                                           ║"
    echo "║   • Stop Loss Parameters (ATR, distance)                        ║"
    echo "║   • Take Profit Ratios (TP1, TP2, TP3)                          ║"
    echo "║   • Trailing Stop (distance, activation)                        ║"
    echo "║   • Risk Percentage                                             ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Intensive parameter optimization for trading strategy"
    echo ""
    echo "Options:"
    echo "  --hours N         Maximum training time in hours (default: 4)"
    echo "  --generations N   Maximum generations (default: 100)"
    echo "  --population N    Population size per generation (default: 8)"
    echo "  --tests N         Backtests per evaluation (default: 10)"
    echo "  --balance N       Initial balance (default: 10000)"
    echo "  --storage PATH    Storage path for results"
    echo "  --no-clickhouse   Skip ClickHouse startup"
    echo "  --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --hours 8                    # Run for 8 hours"
    echo "  $0 --generations 200 --hours 12 # Long training run"
    echo ""
}

is_clickhouse_running() {
    nc -z localhost 9000 &> /dev/null
    return $?
}

start_clickhouse() {
    echo -e "${YELLOW}Starting ClickHouse...${NC}"
    
    if is_clickhouse_running; then
        echo -e "${GREEN}✓ ClickHouse already running${NC}"
        return 0
    fi
    
    cd "${CLICKHOUSE_DIR}" 2>/dev/null || mkdir -p "${CLICKHOUSE_DIR}"
    
    if ! command -v clickhouse &> /dev/null; then
        echo -e "${RED}✗ ClickHouse not found. Install: brew install clickhouse${NC}"
        exit 1
    fi
    
    if [ -f "config.xml" ]; then
        clickhouse server --config-file=config.xml &> "${SCRIPT_DIR}/logs/clickhouse_${TIMESTAMP}.log" &
    else
        clickhouse server &> "${SCRIPT_DIR}/logs/clickhouse_${TIMESTAMP}.log" &
    fi
    
    CLICKHOUSE_PID=$!
    
    for i in {1..15}; do
        if is_clickhouse_running; then
            echo -e "${GREEN}✓ ClickHouse started${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}✗ ClickHouse failed to start${NC}"
    return 1
}

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
        --hours)
            HOURS="$2"
            shift 2
            ;;
        --generations)
            GENERATIONS="$2"
            shift 2
            ;;
        --population)
            POPULATION="$2"
            shift 2
            ;;
        --tests)
            TESTS="$2"
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

main() {
    print_banner
    
    mkdir -p "${SCRIPT_DIR}/logs"
    mkdir -p "${STORAGE_PATH}"
    
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Max Generations: ${GENERATIONS}"
    echo "  Max Time: ${HOURS} hours"
    echo "  Population Size: ${POPULATION}"
    echo "  Tests per Eval: ${TESTS}"
    echo "  Initial Balance: \$${BALANCE}"
    echo "  Storage: ${STORAGE_PATH}"
    echo "  Log: ${LOG_FILE}"
    echo ""
    
    # Estimate time
    ESTIMATED_BACKTESTS=$((GENERATIONS * POPULATION * TESTS))
    echo -e "${YELLOW}Estimated backtests: ~${ESTIMATED_BACKTESTS}${NC}"
    echo -e "${YELLOW}This may take several hours. Progress will be saved automatically.${NC}"
    echo ""
    
    if [ "$SKIP_CLICKHOUSE" = false ]; then
        start_clickhouse || exit 1
        sleep 2
    fi
    
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
    
    # Find Python
    PYTHON_CMD=""
    for cmd in python3.11 python3 python; do
        if command -v $cmd &> /dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    done
    
    if [ -z "$PYTHON_CMD" ]; then
        echo -e "${RED}✗ Python not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Using: $(${PYTHON_CMD} --version)${NC}"
    echo ""
    
    CMD="${PYTHON_CMD} ${SCRIPT_DIR}/intensive_training.py"
    CMD="${CMD} --generations ${GENERATIONS}"
    CMD="${CMD} --hours ${HOURS}"
    CMD="${CMD} --population ${POPULATION}"
    CMD="${CMD} --tests ${TESTS}"
    CMD="${CMD} --balance ${BALANCE}"
    CMD="${CMD} --storage ${STORAGE_PATH}"
    
    echo -e "${CYAN}Command:${NC} ${CMD}"
    echo ""
    
    echo -e "${YELLOW}Starting optimization... (Ctrl+C to stop safely)${NC}"
    echo ""
    
    ${CMD} 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ OPTIMIZATION COMPLETED                                        ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${CYAN}Results:${NC}"
        echo "  Parameters: ${STORAGE_PATH}/best_params.json"
        echo "  Log: ${LOG_FILE}"
        
        # Show best params if file exists
        if [ -f "${STORAGE_PATH}/best_params.json" ]; then
            echo ""
            echo -e "${CYAN}Best Parameters Summary:${NC}"
            ${PYTHON_CMD} -c "
import json
with open('${STORAGE_PATH}/best_params.json') as f:
    p = json.load(f)
print(f\"  Fitness: {p.get('fitness', 'N/A'):.1f}\")
print(f\"  TP Ratios: {p.get('tp1_ratio', 1.5):.1f} / {p.get('tp2_ratio', 2.5):.1f} / {p.get('tp3_ratio', 4.0):.1f}\")
print(f\"  ATR Mult: {p.get('atr_multiplier', 2.0):.2f}\")
print(f\"  Trailing: {p.get('trailing_stop_distance', 0.5):.2f}% at TP{p.get('trailing_activation_tp', 1)}\")
print(f\"  Risk: {p.get('risk_percentage', 1.0):.2f}%\")
" 2>/dev/null || true
        fi
    else
        echo -e "${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ✗ OPTIMIZATION FAILED (code: ${EXIT_CODE})                              ║${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}"
        echo "Check log: ${LOG_FILE}"
    fi
    
    return $EXIT_CODE
}

main




