#!/bin/bash
# HydrAI Simulation Runner Script
# ===============================
# This script automatically activates the virtual environment and runs simulations
# Usage: ./run_simulation.sh [reactant] [options]
# Examples:
#   ./run_simulation.sh --list
#   ./run_simulation.sh ethane
#   ./run_simulation.sh propane
#   ./run_simulation.sh naphtha
#   ./run_simulation.sh n-hexane

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "ct-env" ]; then
    echo "Error: Virtual environment 'ct-env' not found!"
    echo "Please ensure the virtual environment is properly set up."
    exit 1
fi

# Activate virtual environment and run the simulation
echo "Activating virtual environment and running simulation..."
echo "========================================================"

source ct-env/bin/activate && python Main_GeneralizedPFR.py "$@"
