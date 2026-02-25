#!/usr/bin/env python3
"""
Isaac Lab / MimicGen 내부 API 소스 추출 스크립트
Docker 컨테이너 내부에서 실행하여 핵심 API 소스를 파일로 추출합니다.

사용법 (Docker 내부):
  cd /workspace/isaaclab
  ./_isaac_sim/python.sh -u inspect_isaaclab.py
"""

import sys
import os
import glob

# pip_overrides 경로 설정 (generate_data_standalone.py와 동일)
POD = "/workspace/isaaclab/pip_overrides"
if POD not in sys.path:
    sys.path.insert(0, POD)
sys.path = [p for p in sys.path if "pip_prebundle" not in p]

os.environ["PYTHONPATH"] = POD + ":" + os.environ.get("PYTHONPATH", "")
libs = glob.glob(f"{POD}/nvidia/*/lib")
libs.append(f"{POD}/torch/lib")
os.environ["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import numpy as np
import nest_asyncio
nest_asyncio.apply()

OUTPUT_DIR = "/workspace/isaaclab/analysis_results/isaaclab_api_sources"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_FILE = "/workspace/isaaclab/analysis_results/isaaclab_inspection.txt"


def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  -> 저장: {filepath}")


def write_summary(msg):
    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


# Clear summary file
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("Isaac Lab / MimicGen 내부 분석 결과\n")
    f.write("=" * 80 + "\n\n")

# ============================================================
# Phase 1: AppLauncher 초기화 (소스 추출에 필요)
# ============================================================
print("=" * 60)
print("[1/7] AppLauncher 초기화...")
print("=" * 60)

from argparse import ArgumentParser, Namespace
from isaaclab.app import AppLauncher

parser = ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("  -> AppLauncher 완료")

# ============================================================
# Phase 2: 모듈 임포트 및 위치 파악
# ============================================================
print("\n" + "=" * 60)
print("[2/7] 모듈 위치 파악...")
print("=" * 60)

import inspect
import gymnasium as gym
import isaaclab_mimic.envs
import isaaclab_tasks
from isaaclab_mimic.datagen.generation import setup_env_config, env_loop, setup_async_generation
from isaaclab_mimic.datagen.utils import setup_output_paths, reset_env

modules_to_locate = {
    "isaaclab_mimic": __import__("isaaclab_mimic"),
    "isaaclab_mimic.envs": __import__("isaaclab_mimic.envs"),
    "isaaclab_mimic.datagen.generation": __import__("isaaclab_mimic.datagen.generation", fromlist=["generation"]),
    "isaaclab_mimic.datagen.utils": __import__("isaaclab_mimic.datagen.utils", fromlist=["utils"]),
    "isaaclab_tasks": isaaclab_tasks,
    "isaaclab.app": __import__("isaaclab.app", fromlist=["app"]),
}

write_summary("[모듈 위치]")
for name, mod in modules_to_locate.items():
    try:
        src = inspect.getfile(mod)
        write_summary(f"  {name}: {src}")
    except (TypeError, AttributeError):
        write_summary(f"  {name}: (위치 확인 불가)")

# ============================================================
# Phase 3: 등록된 환경 목록
# ============================================================
print("\n" + "=" * 60)
print("[3/7] 등록된 환경 목록...")
print("=" * 60)

write_summary("\n[등록된 MimicGen 환경]")
all_envs = sorted(gym.envs.registry.keys())
mimic_envs = [e for e in all_envs if "Mimic" in e or "Blueprint" in e]
for env_name in mimic_envs:
    spec = gym.envs.registry[env_name]
    write_summary(f"  {env_name}")
    write_summary(f"    entry_point: {spec.entry_point}")
    if hasattr(spec, "kwargs") and spec.kwargs:
        write_summary(f"    kwargs: {spec.kwargs}")

# Franka 관련 환경도
franka_envs = [e for e in all_envs if "Franka" in e]
write_summary(f"\n[Franka 관련 환경] (총 {len(franka_envs)}개)")
for env_name in franka_envs[:20]:
    write_summary(f"  {env_name}")
if len(franka_envs) > 20:
    write_summary(f"  ... 외 {len(franka_envs) - 20}개")

# ============================================================
# Phase 4: 핵심 API 소스 추출
# ============================================================
print("\n" + "=" * 60)
print("[4/7] API 소스 코드 추출...")
print("=" * 60)

functions_to_extract = {
    "setup_env_config": setup_env_config,
    "setup_async_generation": setup_async_generation,
    "env_loop": env_loop,
    "setup_output_paths": setup_output_paths,
    "reset_env": reset_env,
}

for func_name, func_obj in functions_to_extract.items():
    try:
        source = inspect.getsource(func_obj)
        filepath = os.path.join(OUTPUT_DIR, f"{func_name}.py")
        write_file(filepath, f"# Source of {func_name}\n# From: {inspect.getfile(func_obj)}\n\n{source}")
    except (OSError, TypeError) as e:
        write_summary(f"  {func_name}: 소스 추출 실패 - {e}")

# generation.py 전체 모듈도 추출
gen_module = __import__("isaaclab_mimic.datagen.generation", fromlist=["generation"])
try:
    gen_source = inspect.getsource(gen_module)
    write_file(os.path.join(OUTPUT_DIR, "generation_full.py"),
               f"# Full module: isaaclab_mimic.datagen.generation\n# From: {inspect.getfile(gen_module)}\n\n{gen_source}")
except (OSError, TypeError) as e:
    write_summary(f"  generation 전체 모듈: 소스 추출 실패 - {e}")

# utils.py 전체 모듈도 추출
utils_module = __import__("isaaclab_mimic.datagen.utils", fromlist=["utils"])
try:
    utils_source = inspect.getsource(utils_module)
    write_file(os.path.join(OUTPUT_DIR, "utils_full.py"),
               f"# Full module: isaaclab_mimic.datagen.utils\n# From: {inspect.getfile(utils_module)}\n\n{utils_source}")
except (OSError, TypeError) as e:
    write_summary(f"  utils 전체 모듈: 소스 추출 실패 - {e}")

# ============================================================
# Phase 5: Task 환경 설정 분석
# ============================================================
print("\n" + "=" * 60)
print("[5/7] Task 환경 설정 분석...")
print("=" * 60)

task_name = "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0"
write_summary(f"\n[Task 분석] {task_name}")

# Task spec 상세
spec = gym.envs.registry[task_name]
write_summary(f"  entry_point: {spec.entry_point}")
write_summary(f"  kwargs: {spec.kwargs}")

# env_cfg 생성
try:
    env_cfg, success_term = setup_env_config(
        env_name=task_name,
        output_dir="/tmp/inspect_output",
        output_file_name="inspect_test.hdf5",
        num_envs=1,
        device="cuda:0",
        generation_num_trials=1,
    )

    write_summary(f"\n[env_cfg 타입] {type(env_cfg).__name__}")
    write_summary(f"[env_cfg 모듈] {type(env_cfg).__module__}")
    write_summary(f"[success_term] {success_term}")

    # env_cfg 소스 클래스 추출
    try:
        cfg_source = inspect.getsource(type(env_cfg))
        write_file(os.path.join(OUTPUT_DIR, "env_cfg_class.py"),
                   f"# env_cfg class: {type(env_cfg).__name__}\n"
                   f"# From: {inspect.getfile(type(env_cfg))}\n\n{cfg_source}")
    except (OSError, TypeError) as e:
        write_summary(f"  env_cfg 클래스 소스 추출 실패: {e}")

    # Scene 분석
    write_summary(f"\n[Scene 분석]")
    if hasattr(env_cfg, "scene"):
        scene = env_cfg.scene
        write_summary(f"  Scene 타입: {type(scene).__name__}")
        for attr_name in sorted(dir(scene)):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(scene, attr_name)
                if callable(val):
                    continue
                val_type = type(val).__name__
                write_summary(f"  scene.{attr_name}: {val_type}")
                # 로봇 관련이면 상세 출력
                if "robot" in attr_name.lower() or "franka" in attr_name.lower():
                    write_summary(f"    = {val}")
            except Exception:
                pass

    # Actions 분석
    write_summary(f"\n[Actions 분석]")
    if hasattr(env_cfg, "actions"):
        actions = env_cfg.actions
        write_summary(f"  Actions 타입: {type(actions).__name__}")
        for attr_name in sorted(dir(actions)):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(actions, attr_name)
                if callable(val):
                    continue
                write_summary(f"  actions.{attr_name}: {type(val).__name__} = {val}")
            except Exception:
                pass

    # Observations 분석
    write_summary(f"\n[Observations 분석]")
    if hasattr(env_cfg, "observations"):
        obs = env_cfg.observations
        write_summary(f"  Observations 타입: {type(obs).__name__}")
        for group_name in sorted(dir(obs)):
            if group_name.startswith("_"):
                continue
            try:
                group = getattr(obs, group_name)
                if callable(group):
                    continue
                write_summary(f"  observations.{group_name}: {type(group).__name__}")
            except Exception:
                pass

    # Events (Randomization) 분석
    write_summary(f"\n[Events/Randomization 분석]")
    if hasattr(env_cfg, "events"):
        events = env_cfg.events
        write_summary(f"  Events 타입: {type(events).__name__}")
        for attr_name in sorted(dir(events)):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(events, attr_name)
                if callable(val):
                    continue
                write_summary(f"  events.{attr_name}: {type(val).__name__}")
                if "franka" in attr_name.lower() or "random" in attr_name.lower():
                    write_summary(f"    = {val}")
            except Exception:
                pass

    # Terminations 분석
    write_summary(f"\n[Terminations 분석]")
    if hasattr(env_cfg, "terminations"):
        terms = env_cfg.terminations
        write_summary(f"  Terminations 타입: {type(terms).__name__}")
        for attr_name in sorted(dir(terms)):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(terms, attr_name)
                if callable(val):
                    continue
                write_summary(f"  terminations.{attr_name}: {type(val).__name__} = {val}")
            except Exception:
                pass

    # DatagenConfig 분석
    write_summary(f"\n[DatagenConfig 분석]")
    if hasattr(env_cfg, "datagen_config"):
        dg = env_cfg.datagen_config
        write_summary(f"  DatagenConfig 타입: {type(dg).__name__}")
        for attr_name in sorted(dir(dg)):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(dg, attr_name)
                if callable(val):
                    continue
                write_summary(f"  datagen_config.{attr_name}: {type(val).__name__} = {val}")
            except Exception:
                pass

        # DatagenConfig 소스 추출
        try:
            dg_source = inspect.getsource(type(dg))
            write_file(os.path.join(OUTPUT_DIR, "datagen_config_class.py"),
                       f"# DatagenConfig class: {type(dg).__name__}\n"
                       f"# From: {inspect.getfile(type(dg))}\n\n{dg_source}")
        except (OSError, TypeError) as e:
            write_summary(f"  DatagenConfig 소스 추출 실패: {e}")

except Exception as e:
    write_summary(f"  env_cfg 생성 실패: {e}")
    import traceback
    write_summary(traceback.format_exc())

# ============================================================
# Phase 6: Franka 로봇 asset 검색
# ============================================================
print("\n" + "=" * 60)
print("[6/7] Franka 로봇 asset 검색...")
print("=" * 60)

write_summary(f"\n[Franka 로봇 Asset 검색]")

import subprocess

# Franka 관련 파일 검색
searches = [
    ("Franka USD/URDF", "find /workspace/isaaclab -name 'franka*' -o -name 'panda*' 2>/dev/null | head -30"),
    ("robots 디렉토리", "find /workspace/isaaclab -type d -name 'robots' 2>/dev/null | head -10"),
    ("Franka Python 설정", "grep -r 'class.*Franka' /workspace/isaaclab/source --include='*.py' -l 2>/dev/null | head -20"),
]

for label, cmd in searches:
    write_summary(f"\n  --- {label} ---")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            write_summary(f"    {line}")
    else:
        write_summary("    (결과 없음)")

# ============================================================
# Phase 7: env_cfg 소스 파일들 추출
# ============================================================
print("\n" + "=" * 60)
print("[7/7] Task 환경 소스 파일 추출...")
print("=" * 60)

# Task의 실제 환경 클래스 소스 추출
try:
    entry_point = spec.entry_point
    # 'module:class' 형식 파싱
    if ":" in entry_point:
        mod_path, class_name = entry_point.rsplit(":", 1)
    else:
        mod_path = entry_point.rsplit(".", 1)[0]
        class_name = entry_point.rsplit(".", 1)[1]

    env_module = __import__(mod_path, fromlist=[class_name])
    env_class = getattr(env_module, class_name)

    env_source = inspect.getsource(env_class)
    write_file(os.path.join(OUTPUT_DIR, "env_class.py"),
               f"# Environment class: {class_name}\n"
               f"# From: {inspect.getfile(env_class)}\n"
               f"# Entry point: {entry_point}\n\n{env_source}")

    # 해당 모듈 전체도 추출
    mod_source = inspect.getsource(env_module)
    write_file(os.path.join(OUTPUT_DIR, "env_module_full.py"),
               f"# Full module for task env\n"
               f"# From: {inspect.getfile(env_module)}\n\n{mod_source}")

except Exception as e:
    write_summary(f"  환경 클래스 소스 추출 실패: {e}")

# kwargs에 env_cfg_entry_point이 있는지 확인
if spec.kwargs:
    for k, v in spec.kwargs.items():
        write_summary(f"\n  kwargs.{k} = {v}")
        if "entry_point" in k.lower() or "cfg" in k.lower():
            try:
                # 모듈 경로:클래스명 파싱 시도
                if isinstance(v, str) and ":" in v:
                    cfg_mod_path, cfg_class_name = v.rsplit(":", 1)
                    cfg_module = __import__(cfg_mod_path, fromlist=[cfg_class_name])
                    cfg_class = getattr(cfg_module, cfg_class_name)
                    cfg_source = inspect.getsource(cfg_class)
                    safe_name = cfg_class_name.replace("/", "_")
                    write_file(os.path.join(OUTPUT_DIR, f"cfg_{safe_name}.py"),
                               f"# Config class: {cfg_class_name}\n"
                               f"# From: {inspect.getfile(cfg_class)}\n\n{cfg_source}")
                    # 해당 모듈 전체도
                    cfg_mod_source = inspect.getsource(cfg_module)
                    write_file(os.path.join(OUTPUT_DIR, f"cfg_{safe_name}_module_full.py"),
                               f"# Full config module\n"
                               f"# From: {inspect.getfile(cfg_module)}\n\n{cfg_mod_source}")
            except Exception as e2:
                write_summary(f"    소스 추출 실패: {e2}")

print("\n" + "=" * 60)
print("[완료] 분석 결과 저장됨")
print(f"  요약: {SUMMARY_FILE}")
print(f"  소스: {OUTPUT_DIR}/")
print("=" * 60)

# 강제 종료
os._exit(0)
