#!/bin/bash
# HydrAI Simulation Runner Script
# ===============================
# This script launches the Jupyter notebook for interactive simulations
# Usage: ./run_simulation.sh
# 
# Note: For interactive use, launch the notebook directly:
#   jupyter notebook notebooks/Main_1_run_pfr.ipynb
#   jupyter lab notebooks/Main_1_run_pfr.ipynb

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project root
cd "$SCRIPT_DIR/.."

# Check if Jupyter is installed
if ! python3 -c "import jupyter" 2>/dev/null; then
    echo "Error: Jupyter is not installed!"
    echo "Please install it with: pip install jupyter"
    exit 1
fi

# Launch Jupyter notebook
echo "Launching Jupyter notebook for HydrAI simulations..."
echo "========================================================================"
echo "Opening: notebooks/Main_1_run_pfr.ipynb"
echo "========================================================================"

python3 -m jupyter notebook notebooks/Main_1_run_pfr.ipynb
