#!/bin/bash
# HydrAI Simulation Runner Script
# ===============================
# Launches Jupyter with Main_1 (same behaviour as scripts/notebook/run_simulation.sh).
# Usage from repo root:
#   ./scripts/local/run_main1_local_simulation.sh
#
# Or open the notebook directly:
#   jupyter notebook notebooks/Main_1_run_pfr.ipynb
#   jupyter lab notebooks/Main_1_run_pfr.ipynb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# scripts/local -> repo root
cd "$SCRIPT_DIR/../.." || exit 1

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
