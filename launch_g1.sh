#!/usr/bin/env bash

# G1 Humanoid Dual-Arm Manipulation - Launch Script
# Exits if error occurs
set -e

MODE=${MODE:-jupyter}  # default: jupyter

if [ "$MODE" = "standalone" ]; then
    echo "=== G1 Standalone Data Generation Mode ==="
    cd /workspace/isaaclab
    ./_isaac_sim/python.sh -u generate_data_standalone.py
else
    echo "=== G1 Jupyter Lab Mode ==="
    ./_isaac_sim/python.sh -m jupyter lab /workspace/isaaclab/generate_dataset.ipynb \
        --allow-root --ip=0.0.0.0 --no-browser \
        --NotebookApp.token='' --NotebookApp.password='' \
        --NotebookApp.default_url='/tree/generate_dataset.ipynb'
fi
