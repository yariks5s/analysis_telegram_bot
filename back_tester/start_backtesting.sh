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
echo "      CryptoBot Backtesting System Launcher           "
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
  
  if ! command -v clickhouse-server &> /dev/null; then
    echo -e "${RED}Error: clickhouse-server command not found.${NC}"
    echo "Please install ClickHouse or make sure it's in your PATH."
    exit 1
  fi
  
  echo -e "${YELLOW}Starting ClickHouse server...${NC}"
  clickhouse-server --config-file=config.xml &
  
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

function run_backtester() {
  echo -e "${YELLOW}Starting backtesting system...${NC}"
  
  cd "${SCRIPT_DIR}"
  
  local symbol=${1:-"BTCUSDT"}
  local interval=${2:-"1h"}
  local candles=${3:-1000}
  local window=${4:-300}
  local balance=${5:-10000.0}
  local risk=${6:-1.0}
  local optimize=${7:-false}
  
  echo -e "${YELLOW}Running backtesting with parameters:${NC}"
  echo "Symbol: $symbol"
  echo "Interval: $interval"
  echo "Candles: $candles"
  echo "Window: $window"
  echo "Initial Balance: $balance"
  echo "Risk Percentage: $risk"
  echo "Optimize: $optimize"
  
  local cmd="python3 -m back_tester.enhanced_backtester --symbol $symbol --interval $interval --candles $candles --window $window --balance $balance --risk $risk"
  if [[ "$optimize" == "true" ]]; then
    cmd="$cmd --optimize"
  fi
  
  echo -e "${YELLOW}Executing: ${cmd}${NC}"
  eval $cmd
  
  local status=$?
  if [ $status -eq 0 ]; then
    echo -e "${GREEN}Backtesting completed successfully!${NC}"
    echo "Reports should be available in the reports directory."
  else
    echo -e "${RED}Backtesting failed with status $status${NC}"
  fi
  
  return $status
}

function show_usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --symbol SYMBOL      Trading pair symbol (default: BTCUSDT)"
  echo "  --interval INTERVAL  Candle interval (default: 1h)"
  echo "  --candles NUMBER     Number of candles to fetch (default: 1000)"
  echo "  --window NUMBER      Lookback window (default: 300)"
  echo "  --balance NUMBER     Initial balance (default: 10000.0)"
  echo "  --risk NUMBER        Risk percentage (default: 1.0)"
  echo "  --optimize          Run parameter optimization"
  echo "  --help              Show this help message"
  echo ""
  echo "Example: $0 --symbol ETHUSDT --interval 4h --optimize"
}

SYMBOL="BTCUSDT"
INTERVAL="1h"
CANDLES=1000
WINDOW=300
BALANCE=10000.0
RISK=1.0
OPTIMIZE=false

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
    --window)
      WINDOW="$2"
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
    --optimize)
      OPTIMIZE=true
      shift
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

echo -e "${YELLOW}Preparing backtesting environment...${NC}"

start_clickhouse

if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to start ClickHouse. Exiting.${NC}"
  exit 1
fi
run_backtester "$SYMBOL" "$INTERVAL" "$CANDLES" "$WINDOW" "$BALANCE" "$RISK" "$OPTIMIZE"

echo ""
echo -e "${YELLOW}Backtesting session completed.${NC}"
echo -e "${YELLOW}Note: ClickHouse server is still running in the background.${NC}"
echo -e "${YELLOW}To stop ClickHouse, run: killall clickhouse-server${NC}"
echo ""

exit 0
