#!/usr/bin/env python3
"""
G1 Humanoid Dual-Arm Manipulation - Standalone Data Generation Script
Isaac Lab 2.3.2 기반, multi-EEF MimicGen 지원
"""

import sys
import os

# IMPORTANT: pinocchio must be imported before Isaac Sim to register C++ bindings
import pinocchio  # noqa: F401 — pre-import for Pink IK controller compatibility

# nest_asyncio 적용
import nest_asyncio
nest_asyncio.apply()

import torch
import numpy as np

print("=" * 60, flush=True)
print("[G1 Standalone] Data Generation 시작", flush=True)
print("=" * 60, flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
sys.stdout.flush()

# ============================================================
# Isaac Sim AppLauncher 초기화
# ============================================================
from argparse import ArgumentParser, Namespace
from isaaclab.app import AppLauncher

parser = ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.enable_cameras = True
args_cli.headless = True

# 환경 변수에서 설정 읽기
num_trials = int(os.environ.get("NUM_TRIALS", "1"))
task = os.environ.get("TASK", "")

# Task 자동 선택: 환경 변수 TASK가 없으면 기본 G1 locomanipulation 사용
# Available G1 environments:
#   Isaac-Locomanipulation-G1-Abs-Mimic-v0       — G1 dual-arm MimicGen (locomanipulation)
#   Isaac-PickPlace-Locomanipulation-G1-Abs-v0   — G1 base pick-place (no MimicGen)
#   Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 — G1 fixed base upper body IK
#   Isaac-PickPlace-G1-InspireFTP-Abs-v0         — G1 with Inspire hand
TASK_MAP = {
    "locomanipulation": "Isaac-Locomanipulation-G1-Abs-Mimic-v0",
    "pick_place": "Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
    "fixed_base": "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
    "inspire_hand": "Isaac-PickPlace-G1-InspireFTP-Abs-v0",
}

# Task 이름 결정
if task in TASK_MAP:
    env_name = TASK_MAP[task]
elif task:
    # 직접 Gymnasium 환경 이름이 지정된 경우
    env_name = task
else:
    # 기본값: G1 locomanipulation MimicGen
    env_name = "Isaac-Locomanipulation-G1-Abs-Mimic-v0"

# 입력 파일: task별로 다른 데이터셋 사용
input_file = os.environ.get("INPUT_FILE", "datasets/annotated_dataset.hdf5")
output_file = os.environ.get("OUTPUT_FILE", "datasets/generated_dataset.hdf5")

print(f"[설정] Task: {env_name}", flush=True)
print(f"[설정] 목표 생성 횟수: {num_trials}", flush=True)
print(f"[설정] 입력 파일: {input_file}", flush=True)
print(f"[설정] 출력 파일: {output_file}", flush=True)

config = {
    "task": env_name,
    "num_envs": 1,
    "generation_num_trials": num_trials,
    "input_file": input_file,
    "output_file": output_file,
    "pause_subtask": False,
}

args_dict = vars(args_cli)
args_dict.update(config)
args_cli = Namespace(**args_dict)

print("[1/5] AppLauncher 시작 중...")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("      -> AppLauncher 완료")

# ============================================================
# Isaac Lab 모듈 임포트 (AppLauncher 이후!)
# ============================================================
print("[2/5] 모듈 임포트 중...")
import gymnasium as gym
import random

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401 — G1 locomanipulation MimicGen
from isaaclab_mimic.datagen.generation import setup_env_config, env_loop, setup_async_generation
from isaaclab_mimic.datagen.utils import setup_output_paths, reset_env
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401 — G1 locomanipulation envs

# Pre-download G1 URDF for Pink IK controller
from isaaclab.utils.assets import retrieve_file_path, ISAACLAB_NUCLEUS_DIR
_urdf_nucleus = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
retrieve_file_path(_urdf_nucleus)

ISAACLAB_OUTPUT_DIR = "/workspace/isaaclab/output"
os.makedirs(ISAACLAB_OUTPUT_DIR, exist_ok=True)
print("      -> 모듈 임포트 완료")

# ============================================================
# 환경 설정
# ============================================================
print("[3/5] 환경 설정 중...")
output_dir, output_file_name = setup_output_paths(args_cli.output_file)

env_cfg, success_term = setup_env_config(
    env_name=env_name,
    output_dir=output_dir,
    output_file_name=output_file_name,
    num_envs=args_cli.num_envs,
    device=args_cli.device,
    generation_num_trials=args_cli.generation_num_trials,
)

# Set observation output directory (if rgb_camera observations exist)
if hasattr(env_cfg.observations, 'rgb_camera'):
    for obs in vars(env_cfg.observations.rgb_camera).values():
        if not isinstance(obs, ObsTerm):
            continue
        if "image_path" in obs.params:
            obs.params["image_path"] = os.path.join(ISAACLAB_OUTPUT_DIR, obs.params["image_path"])

env = gym.make(env_name, cfg=env_cfg).unwrapped

random.seed(env.cfg.datagen_config.seed)
np.random.seed(env.cfg.datagen_config.seed)
torch.manual_seed(env.cfg.datagen_config.seed)

reset_env(env, 100)
print("      -> 환경 설정 완료")

# ============================================================
# 비동기 생성 설정
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
# 데이터 생성 실행
# ============================================================
print("[5/5] 데이터 생성 시작!")
print("-" * 60)
print(f"목표: {env.cfg.datagen_config.generation_num_trials}개 성공")

env_loop(
    env=env,
    env_action_queue=async_gen['action_queue'],
    shared_datagen_info_pool=async_gen['info_pool'],
    asyncio_event_loop=async_gen['event_loop']
)

# ============================================================
# 결과 요약
# ============================================================
from isaaclab_mimic.datagen.generation import num_success, num_failures, num_attempts

print("\n" + "=" * 60)
print("[완료]")
print(f"      총 시도: {num_attempts}회")
print(f"      성공: {num_success}, 실패: {num_failures}")
print("=" * 60)

# ============================================================
# 출력 파일 구조 정리
# ============================================================
def organize_output_files(output_dir):
    """출력 이미지를 체계적인 폴더 구조로 정리"""
    import re
    import shutil

    print("\n[후처리] 출력 파일 구조 정리 중...")

    pattern = re.compile(
        r"(?P<camera>[\w]+_cam)_"
        r"(?P<modality>normals|semantic_segmentation)_"
        r"trial_(?P<trial>\d+)_"
        r"tile_(?P<env>\d+)_"
        r"step_(?P<frame>\d+)\.png"
    )

    files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    if not files:
        print("      -> 정리할 이미지 파일 없음")
        return

    trial_files = {}
    for filename in files:
        match = pattern.match(filename)
        if match:
            trial = int(match.group('trial'))
            camera = match.group('camera')
            modality = match.group('modality')
            frame = int(match.group('frame'))

            if modality == 'semantic_segmentation':
                modality = 'segmentation'

            folder_key = f"{camera}_{modality}"

            if trial not in trial_files:
                trial_files[trial] = {}
            if folder_key not in trial_files[trial]:
                trial_files[trial][folder_key] = []

            trial_files[trial][folder_key].append((frame, filename))

    moved_count = 0
    for trial in sorted(trial_files.keys()):
        for folder_key, frame_files in trial_files[trial].items():
            demo_folder = os.path.join(output_dir, folder_key, f"demo_{trial}")
            os.makedirs(demo_folder, exist_ok=True)

            for frame, filename in sorted(frame_files):
                src = os.path.join(output_dir, filename)
                dst = os.path.join(demo_folder, f"step_{frame:04d}.png")
                shutil.move(src, dst)
                moved_count += 1

    num_demos = len(trial_files)
    print(f"      -> {num_demos}개 demo, {moved_count}개 파일 정리 완료")

organize_output_files(ISAACLAB_OUTPUT_DIR)

print("\n[완료] 프로세스 종료 중...")
sys.stdout.flush()

os._exit(0)
