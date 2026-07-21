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
python3 scripts/demo_live_textworld.py
