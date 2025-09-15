#!/bin/bash
# HydrAI Core Structure Display Script
# ====================================
# Shows the core project structure excluding generated files

echo "HydrAI Core Project Structure (excluding generated files)"
echo "========================================================"
echo ""

tree -I 'ct-env|__pycache__|*.pyc|.git|.DS_Store|*.csv|*.png|*.dat' -a

echo ""
echo "Generated Files Summary:"
echo "- fig/ directory: Contains PNG plot files (auto-generated)"
echo "- results/ directory: Contains CSV data and DAT summary files (auto-generated)"
echo ""
echo "Total Core Files: $(find . -name "*.py" -o -name "*.json" -o -name "*.md" -o -name "*.txt" -o -name "*.yaml" -o -name "*.sh" | grep -v ct-env | wc -l | tr -d ' ') files"
echo "Total Core Directories: $(find . -type d | grep -v ct-env | grep -v __pycache__ | wc -l | tr -d ' ') directories"
