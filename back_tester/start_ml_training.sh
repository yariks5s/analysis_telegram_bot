#!/bin/bash

# Exit on error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLICKHOUSE_DIR="${SCRIPT_DIR}/clickhouse"
BACKTESTER_DIR="${SCRIPT_DIR}/back_tester"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "======================================================"
echo "      CryptoBot ML Training System Launcher           "
echo "======================================================"
echo -e "${NC}"

function is_clickhouse_running() {
  nc -z localhost 9000 &> /dev/null
  return $?
}

function start_clickhouse() {
  echo -e "${YELLOW}Starting ClickHouse server from ${CLICKHOUSE_DIR}...${NC}"
  
  cd "${CLICKHOUSE_DIR}"
  
  if is_clickhouse_running; then
    echo -e "${GREEN}ClickHouse is already running.${NC}"
    return 0
  fi
  
  if ! command -v clickhouse server &> /dev/null; then
    echo -e "${RED}Error: clickhouse server command not found.${NC}"
    echo "Please install ClickHouse or make sure it's in your PATH."
    exit 1
  fi
  
  echo -e "${YELLOW}Starting ClickHouse server...${NC}"
  clickhouse server --config-file=config.xml &
  
  echo -e "${YELLOW}Waiting for ClickHouse to start...${NC}"
  for i in {1..10}; do
    if is_clickhouse_running; then
      echo -e "${GREEN}ClickHouse started successfully!${NC}"
      return 0
    fi
    sleep 1
  done
  
  echo -e "${RED}Failed to start ClickHouse within timeout.${NC}"
  return 1
}

function run_ml_trainer() {
  echo -e "${YELLOW}Starting ML training system...${NC}"
  
  cd "${SCRIPT_DIR}"
  
  local symbol=${1:-"BTCUSDT"}
  local interval=${2:-"1h"}
  local candles=${3:-2000}
  local balance=${4:-10000.0}
  local risk=${5:-1.0}
  local iterations=${6:-50}
  
  echo -e "${YELLOW}Running ML training with parameters:${NC}"
  echo "Symbol: $symbol"
  echo "Interval: $interval"
  echo "Candles: $candles"
  echo "Initial Balance: $balance"
  echo "Risk Percentage: $risk"
  echo "Training Iterations: $iterations"
  
  # Add project root directory to PYTHONPATH
  local project_root="$(cd "${SCRIPT_DIR}/.." && pwd)"
  export PYTHONPATH="${project_root}:${PYTHONPATH}"
  
  # Ensure directories exist
  mkdir -p "${project_root}/models/training"
  mkdir -p "${project_root}/reports/training"
  mkdir -p "${project_root}/logs"
  
  local cmd="python3.11 -m back_tester.ml_trainer --symbol $symbol --interval $interval --candles $candles --balance $balance --risk $risk --iterations $iterations"
  
  echo -e "${YELLOW}Executing: ${cmd}${NC}"
  eval $cmd
  
  local status=$?
  if [ $status -eq 0 ]; then
    echo -e "${GREEN}ML training completed successfully!${NC}"
    echo "Models should be available in the models directory."
    echo "Logs are available in the logs directory."
  else
    echo -e "${RED}ML training failed with status $status${NC}"
  fi
  
  return $status
}

function show_usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --symbol SYMBOL      Trading pair symbol (default: BTCUSDT)"
  echo "  --interval INTERVAL  Candle interval (default: 1h)"
  echo "  --candles NUMBER     Number of candles to fetch (default: 2000)"
  echo "  --balance NUMBER     Initial balance for backtest (default: 10000.0)"
  echo "  --risk NUMBER        Risk percentage for backtest (default: 1.0)"
  echo "  --iterations NUMBER  Training iterations (default: 50)"
  echo "  --help              Show this help message"
  echo ""
  echo "Example: $0 --symbol ETHUSDT --interval 4h --iterations 100"
}

SYMBOL="BTCUSDT"
INTERVAL="1h"
CANDLES=2000
BALANCE=10000.0
RISK=1.0
ITERATIONS=50

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --symbol)
      SYMBOL="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --candles)
      CANDLES="$2"
      shift 2
      ;;
    --balance)
      BALANCE="$2"
      shift 2
      ;;
    --risk)
      RISK="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --help)
      show_usage
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $key${NC}"
      show_usage
      exit 1
      ;;
  esac
done

echo -e "${YELLOW}Preparing ML training environment...${NC}"

start_clickhouse

if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to start ClickHouse. Exiting.${NC}"
  exit 1
fi

sleep 5

run_ml_trainer "$SYMBOL" "$INTERVAL" "$CANDLES" "$BALANCE" "$RISK" "$ITERATIONS"

echo ""
echo -e "${YELLOW}ML training session completed.${NC}"

killall clickhouse server

sleep 5

echo -e "${YELLOW}Note: ClickHouse server may still run in the background.${NC}"
echo -e "${YELLOW}To stop ClickHouse, run: killall clickhouse-server${NC}"
echo ""

exit 0
