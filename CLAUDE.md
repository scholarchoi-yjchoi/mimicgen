# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA Omniverse Blueprint for generating synthetic manipulation motion trajectories for robot learning. Takes a small number of human demonstrations and creates large-scale diverse training datasets by:
1. Simulating robotic manipulation in Isaac Lab
2. GPU-accelerated video encoding with semantic segmentation and surface normals
3. Applying NVIDIA Cosmos AI for visual diversity transformation

**Hardware Requirements:**
- Local: Ubuntu 22.04, RTX A6000 (48GB VRAM), NVIDIA Driver 535.129.03+
- Cosmos: Separate H100+ node (80GB VRAM) - cannot run on same machine as Isaac Lab

## Commands

```bash
# Launch development environment
xhost +local:
docker compose -f docker-compose.yml up -d

# Access Jupyter at http://localhost:8888/lab/tree/generate_dataset.ipynb

# Shutdown
docker compose -f docker-compose.yml down
```

All development happens through the Jupyter notebook - there is no CLI interface.

## Architecture

### Three-Stage Pipeline

**Stage 1: Isaac Lab Simulation** → Generates frames with semantic segmentation and surface normals
- Task: `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0`
- Output: `{camera}_{modality}_trial_{trial}_tile_{env}_step_{frame}.png`

**Stage 2: GPU Video Encoding** (`notebook_utils.py`) → Warp/CUDA kernels apply Lambertian shading
- Output: MP4 videos in `_isaaclab_out/`

**Stage 3: Cosmos Transformation** → Generates diverse visual variations of same motion
- Output: Processed videos in `_cosmos_out/`
- Communication via Flask REST API (`app.py`) to remote Cosmos node

### Key Files

| File | Purpose |
|------|---------|
| `notebook/generate_dataset.ipynb` | Main interactive workflow - start here |
| `notebook/app.py` | Flask API for async Cosmos job processing |
| `notebook/cosmos_request.py` | HTTP client for Cosmos server communication |
| `notebook/notebook_utils.py` | GPU-accelerated video encoding (Warp kernels) |
| `notebook/notebook_widgets.py` | Jupyter UI components and PromptManager |
| `notebook/stacking_prompt.toml` | Scene description templates with variables |

### Cosmos API Endpoints (app.py)

- `POST /canny/submit` - Submit video for processing
- `GET /canny/status/<job_id>` - Poll job status
- `GET /canny/result/<job_id>` - Download result

## Key Patterns

**Frame naming convention** is strict and must be preserved:
```
{camera_name}_{modality}_trial_{trial_num}_tile_{env_num}_step_{frame_idx}.png
```

**Scene prompts** are template-driven via `stacking_prompt.toml` - modify templates here, not in code.

**Async job processing** - Cosmos operations use background threading with in-memory status tracking.

**All GPU operations** use Warp (CUDA) for parallel processing.
