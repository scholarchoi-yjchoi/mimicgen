#!/usr/bin/env python3
"""
Isaac Lab MimicGen Data Generation - Standalone Script
nest_asyncio를 사용하여 노트북과 동일한 환경 구성 (v5)
"""

import sys
import os
import glob

# ============================================================
# 1. 경로 설정 (pip_overrides 우선)
# ============================================================
POD = "/workspace/isaaclab/pip_overrides"
if POD not in sys.path:
    sys.path.insert(0, POD)
sys.path = [p for p in sys.path if "pip_prebundle" not in p]

os.environ["PYTHONPATH"] = POD + ":" + os.environ.get("PYTHONPATH", "")
libs = glob.glob(f"{POD}/nvidia/*/lib")
libs.append(f"{POD}/torch/lib")
os.environ["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 필수 모듈 로드
import torch
import torchvision
import numpy as np
import scipy

# nest_asyncio 적용 (노트북과 동일하게)
import nest_asyncio
nest_asyncio.apply()

import sys
print("=" * 60, flush=True)
print("[Standalone] Data Generation 시작 (v5 - nest_asyncio)", flush=True)
print("=" * 60, flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"Torchvision: {torchvision.__version__}", flush=True)
sys.stdout.flush()

# ============================================================
# 2. Isaac Sim AppLauncher 초기화
# ============================================================
from argparse import ArgumentParser, Namespace
from isaaclab.app import AppLauncher

parser = ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.enable_cameras = True
args_cli.kit_args = "--enable omni.videoencoding"
args_cli.headless = True

# 환경 변수에서 생성 횟수 읽기 (기본값: 1)
num_trials = int(os.environ.get("NUM_TRIALS", "1"))
print(f"[설정] 목표 생성 횟수: {num_trials}", flush=True)

config = {
    "task": "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    "num_envs": 1,
    "generation_num_trials": num_trials,
    "input_file": "datasets/annotated_dataset.hdf5",
    "output_file": "datasets/generated_dataset.hdf5",
    "pause_subtask": False,
    "enable": "omni.kit.renderer.capture",
}

args_dict = vars(args_cli)
args_dict.update(config)
args_cli = Namespace(**args_dict)

print("[1/5] AppLauncher 시작 중...")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("      -> AppLauncher 완료")

# ============================================================
# 3. Isaac Lab 모듈 임포트 (AppLauncher 이후!)
# ============================================================
print("[2/5] 모듈 임포트 중...")
import gymnasium as gym
import random
import asyncio
import contextlib

import isaaclab_mimic.envs
from isaaclab_mimic.datagen.generation import setup_env_config, env_loop, setup_async_generation
from isaaclab_mimic.datagen.utils import setup_output_paths, reset_env
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab_tasks

ISAACLAB_OUTPUT_DIR = "/workspace/isaaclab/output"
os.makedirs(ISAACLAB_OUTPUT_DIR, exist_ok=True)
print("      -> 모듈 임포트 완료")

# ============================================================
# 4. 환경 설정
# ============================================================
print("[3/5] 환경 설정 중...")
output_dir, output_file_name = setup_output_paths(args_cli.output_file)
env_name = args_cli.task

env_cfg, success_term = setup_env_config(
    env_name=env_name,
    output_dir=output_dir,
    output_file_name=output_file_name,
    num_envs=args_cli.num_envs,
    device=args_cli.device,
    generation_num_trials=args_cli.generation_num_trials,
)

for obs in vars(env_cfg.observations.rgb_camera).values():
    if not isinstance(obs, ObsTerm):
        continue
    obs.params["image_path"] = os.path.join(ISAACLAB_OUTPUT_DIR, obs.params["image_path"])

env = gym.make(env_name, cfg=env_cfg).unwrapped

random.seed(env.cfg.datagen_config.seed)
np.random.seed(env.cfg.datagen_config.seed)
torch.manual_seed(env.cfg.datagen_config.seed)

reset_env(env, 100)
print("      -> 환경 설정 완료")

# ============================================================
# 5. 공식 API로 비동기 생성 설정
# ============================================================
print("[4/5] 비동기 생성 설정 중...")

async_gen = setup_async_generation(
    env=env,
    num_envs=args_cli.num_envs,
    input_file=args_cli.input_file,
    success_term=success_term,
    pause_subtask=args_cli.pause_subtask
)

print("      -> 비동기 생성 설정 완료")

# ============================================================
# 6. 공식 env_loop 실행
# ============================================================
print("[5/5] 데이터 생성 시작!")
print("-" * 60)
print(f"목표: {env.cfg.datagen_config.generation_num_trials}개 성공")

# 공식 env_loop 사용
env_loop(
    env=env,
    env_action_queue=async_gen['action_queue'],
    shared_datagen_info_pool=async_gen['info_pool'],
    asyncio_event_loop=async_gen['event_loop']
)

# ============================================================
# 7. 결과 요약
# ============================================================
from isaaclab_mimic.datagen.generation import num_success, num_failures, num_attempts

print("\n" + "=" * 60)
print("[완료]")
print(f"      총 시도: {num_attempts}회")
print(f"      성공: {num_success}, 실패: {num_failures}")
print("=" * 60)

# ============================================================
# 8. 출력 파일 구조 정리
# ============================================================
def organize_output_files(output_dir):
    """출력 이미지를 체계적인 폴더 구조로 정리

    Isaac Lab은 성공한 trial만 이미지로 저장하므로,
    trial_N = demo_N으로 직접 매핑됨

    변환 전: output/{camera}_{modality}_trial_{N}_tile_{X}_step_{Y}.png
    변환 후: output/{camera}_{modality}/demo_{N}/step_{Y:04d}.png
    """
    import re
    import shutil

    print("\n[후처리] 출력 파일 구조 정리 중...")

    # 파일명 패턴: {camera}_{modality}_trial_{trial}_tile_{env}_step_{frame}.png
    pattern = re.compile(
        r"(?P<camera>table_(?:high_)?cam)_"
        r"(?P<modality>normals|semantic_segmentation)_"
        r"trial_(?P<trial>\d+)_"
        r"tile_(?P<env>\d+)_"
        r"step_(?P<frame>\d+)\.png"
    )

    # 파일 목록 수집
    files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    if not files:
        print("      -> 정리할 이미지 파일 없음")
        return

    # 파일 분류: {trial: {folder_key: [(frame, filename), ...]}}
    trial_files = {}
    for filename in files:
        match = pattern.match(filename)
        if match:
            trial = int(match.group('trial'))
            camera = match.group('camera')
            modality = match.group('modality')
            frame = int(match.group('frame'))

            # modality 이름 정규화 (semantic_segmentation -> segmentation)
            if modality == 'semantic_segmentation':
                modality = 'segmentation'

            folder_key = f"{camera}_{modality}"

            if trial not in trial_files:
                trial_files[trial] = {}
            if folder_key not in trial_files[trial]:
                trial_files[trial][folder_key] = []

            trial_files[trial][folder_key].append((frame, filename))

    # 폴더 구조 생성 및 파일 이동
    # trial 번호 = demo 번호 (성공한 trial만 저장되므로)
    moved_count = 0
    for trial in sorted(trial_files.keys()):
        for folder_key, frame_files in trial_files[trial].items():
            # 폴더 생성
            demo_folder = os.path.join(output_dir, folder_key, f"demo_{trial}")
            os.makedirs(demo_folder, exist_ok=True)

            # 파일 이동 및 이름 변경
            for frame, filename in sorted(frame_files):
                src = os.path.join(output_dir, filename)
                dst = os.path.join(demo_folder, f"step_{frame:04d}.png")
                shutil.move(src, dst)
                moved_count += 1

    num_demos = len(trial_files)
    print(f"      -> {num_demos}개 demo, {moved_count}개 파일 정리 완료")

# 파일 정리 실행
organize_output_files(ISAACLAB_OUTPUT_DIR)

# 정리 - 블로킹 방지를 위해 강제 종료
print("\n[완료] 프로세스 종료 중...")
sys.stdout.flush()

os._exit(0)  # simulation_app.close() 대신 강제 종료
