#!/bin/bash
set -e

echo "=================================================="
echo " World Modeling for Autonomous Agents - Demo"
echo "=================================================="

# Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python3."
    exit 1
fi

echo "Running validation script..."
python3 scripts/validate_submission.py

echo ""
echo "Starting Demo..."
echo "=================================================="
OLLAMA_TIMEOUT=120 python3 scripts/visual_demo.py examples/demo.z8
