#!/bin/bash
# Start the Adaptive Learning Dashboard

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate virtual environment if it exists
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Install requirements if needed
pip install -q flask flask-cors

# Start the dashboard
echo "🚀 Starting CryptoBot Adaptive Learning Dashboard..."
echo "📊 Open http://localhost:5050 in your browser"
echo ""

cd "$SCRIPT_DIR"
python app.py




