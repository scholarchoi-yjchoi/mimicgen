#!/usr/bin/env bash

# Exits if error occurs
set -e

MODE=${MODE:-jupyter}  # 기본값: jupyter

if [ "$MODE" = "standalone" ]; then
    echo "=== Standalone Data Generation Mode ==="
    # 필수 패키지 설치 (스크립트 실행 전)
    /isaac-sim/kit/python/bin/python3 -m pip install nest_asyncio toml -q
    cd /workspace/isaaclab
    ./_isaac_sim/python.sh -u generate_data_standalone.py
else
    echo "=== Jupyter Lab Mode ==="
    /isaac-sim/kit/python/bin/python3 -m pip install jupyter
    ./_isaac_sim/python.sh -m jupyter lab /workspace/isaaclab/generate_dataset.ipynb --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.default_url='/tree/generate_dataset.ipynb'
fi
