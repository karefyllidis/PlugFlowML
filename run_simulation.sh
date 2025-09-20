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

# Define the external virtual environment path
CT_ENV_PATH="/Users/nikolaskarefyllidis/venv/ct-env"

# Check if virtual environment exists
if [ ! -d "$CT_ENV_PATH" ]; then
    echo "Error: Virtual environment not found at $CT_ENV_PATH!"
    echo "Please ensure the virtual environment is properly set up."
    exit 1
fi

# Activate virtual environment and run the simulation
echo "Activating virtual environment from $CT_ENV_PATH and running simulation..."
echo "========================================================================"

# Use the python executable directly from the environment
"$CT_ENV_PATH/bin/python3.13" Main_GeneralizedPFR.py "$@"
