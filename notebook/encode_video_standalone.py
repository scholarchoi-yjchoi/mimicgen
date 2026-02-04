#!/usr/bin/env python3
"""
Isaac Lab Video Encoding - Standalone Script
이미지 시퀀스를 MP4 비디오로 인코딩 (GPU 가속)
"""

import sys
import os
import glob
import json
import re

# ============================================================
# 1. 경로 설정 (generate_data_standalone.py와 동일)
# ============================================================
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

print("=" * 60, flush=True)
print("[Standalone] Video Encoding 시작", flush=True)
print("=" * 60, flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
sys.stdout.flush()

# ============================================================
# 2. Isaac Sim AppLauncher 초기화 (최소 설정)
# ============================================================
print("[1/4] AppLauncher 시작 중...", flush=True)

from argparse import ArgumentParser, Namespace
from isaaclab.app import AppLauncher

parser = ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.kit_args = "--enable omni.videoencoding"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("      -> AppLauncher 완료", flush=True)

# ============================================================
# 3. Isaac Sim 모듈 임포트 (AppLauncher 이후!)
# ============================================================
print("[2/4] 모듈 임포트 중...", flush=True)

import warp as wp
from PIL import Image
from video_encoding import get_video_encoding_interface

print("      -> 모듈 임포트 완료", flush=True)

# ============================================================
# 4. 설정 읽기
# ============================================================
print("[3/4] 설정 로드 중...", flush=True)

VIDEO_CONFIG = json.loads(os.environ.get("VIDEO_CONFIG", "{}"))
ISAACLAB_OUTPUT_DIR = "/workspace/isaaclab/output"
VIDEO_OUTPUT_DIR = "/workspace/isaaclab/output/videos"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

camera = VIDEO_CONFIG.get("camera", "table_cam")
video_length = VIDEO_CONFIG.get("video_length", 120)

print(f"      카메라: {camera}", flush=True)
print(f"      비디오 길이: {video_length} 프레임", flush=True)
print("      -> 설정 로드 완료", flush=True)

# ============================================================
# 5. GPU 셰이딩 커널
# ============================================================
DEFAULT_FRAMERATE = 24.0
DEFAULT_LIGHT_DIRECTION = (0.0, 0.0, 1.0)

@wp.kernel
def _shade_segmentation(
    segmentation: wp.array3d(dtype=wp.uint8),
    normals: wp.array3d(dtype=wp.float32),
    shading_out: wp.array3d(dtype=wp.uint8),
    light_source: wp.array(dtype=wp.vec3f),
):
    """Apply Lambertian shading to semantic segmentation using surface normals."""
    i, j = wp.tid()
    normal = normals[i, j]
    light_source_vec = wp.normalize(light_source[0])
    shade = 0.5 + wp.dot(wp.vec3f(normal[0], normal[1], normal[2]), light_source_vec) * 0.5

    shading_out[i, j, 0] = wp.uint8(wp.float32(segmentation[i, j, 0]) * shade)
    shading_out[i, j, 1] = wp.uint8(wp.float32(segmentation[i, j, 1]) * shade)
    shading_out[i, j, 2] = wp.uint8(wp.float32(segmentation[i, j, 2]) * shade)
    shading_out[i, j, 3] = wp.uint8(255)

# ============================================================
# 6. 비디오 인코딩 함수
# ============================================================
def encode_video(camera_name, demo_num, start_frame, num_frames):
    """단일 demo의 비디오 인코딩"""

    def get_frame_path(modality, frame_idx):
        return os.path.join(
            ISAACLAB_OUTPUT_DIR,
            f"{camera_name}_{modality}",
            f"demo_{demo_num}",
            f"step_{frame_idx:04d}.png"
        )

    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{camera_name}_demo_{demo_num}.mp4")

    # 프레임 존재 확인
    for frame_idx in range(start_frame, start_frame + num_frames):
        if not os.path.exists(get_frame_path("normals", frame_idx)):
            raise ValueError(f"Missing normals frame {frame_idx}")
        if not os.path.exists(get_frame_path("segmentation", frame_idx)):
            raise ValueError(f"Missing segmentation frame {frame_idx}")

    # 비디오 인코딩 초기화
    video_encoding = get_video_encoding_interface()
    first_frame = np.array(Image.open(get_frame_path("segmentation", start_frame)))
    height, width = first_frame.shape[:2]

    # GPU 버퍼 할당
    normals_wp = wp.empty((height, width, 3), dtype=wp.float32, device="cuda")
    segmentation_wp = wp.empty((height, width, 4), dtype=wp.uint8, device="cuda")
    shaded_wp = wp.empty_like(segmentation_wp)
    light_source = wp.array(DEFAULT_LIGHT_DIRECTION, dtype=wp.vec3f, device="cuda")

    video_encoding.start_encoding(
        video_filename=output_path,
        framerate=DEFAULT_FRAMERATE,
        nframes=num_frames,
        overwrite_video=True,
    )

    # 프레임별 인코딩
    for frame_idx in range(start_frame, start_frame + num_frames):
        # 이미지 로드
        normals_np = np.array(Image.open(get_frame_path("normals", frame_idx))).astype(np.float32) / 255.0
        wp.copy(normals_wp, wp.from_numpy(normals_np))

        segmentation_np = np.array(Image.open(get_frame_path("segmentation", frame_idx)))
        wp.copy(segmentation_wp, wp.from_numpy(segmentation_np))

        # GPU 셰이딩
        wp.launch(_shade_segmentation, dim=(height, width),
                  inputs=[segmentation_wp, normals_wp, shaded_wp, light_source])

        # 프레임 인코딩
        video_encoding.encode_next_frame_from_buffer(
            shaded_wp.numpy().tobytes(), width=width, height=height)

    return output_path

# ============================================================
# 7. 메인 실행
# ============================================================
print("[4/4] 비디오 인코딩 시작!", flush=True)
print("-" * 60, flush=True)

# demo 폴더 스캔
segmentation_dir = os.path.join(ISAACLAB_OUTPUT_DIR, f"{camera}_segmentation")

if not os.path.isdir(segmentation_dir):
    print(f"[ERROR] 디렉토리 없음: {segmentation_dir}", flush=True)
    sys.stdout.flush()
    os._exit(1)

demos = []
for entry in os.listdir(segmentation_dir):
    match = re.match(r"demo_(\d+)", entry)
    if match:
        demos.append(int(match.group(1)))

demos.sort()
print(f"[발견] {len(demos)}개 demo: {demos}", flush=True)

# 각 demo 인코딩
encoded_videos = []
for demo_num in demos:
    demo_path = os.path.join(segmentation_dir, f"demo_{demo_num}")
    frames = []
    for f in os.listdir(demo_path):
        step_match = re.match(r"step_(\d+)\.png", f)
        if step_match:
            frames.append(int(step_match.group(1)))
    frames.sort()

    if len(frames) < video_length:
        print(f"[스킵] demo_{demo_num}: 프레임 부족 ({len(frames)} < {video_length})", flush=True)
        continue

    # 마지막 video_length 프레임 사용
    start_frame = max(frames[0], frames[-1] - video_length + 1)

    print(f"[인코딩] demo_{demo_num}: 프레임 {start_frame}~{start_frame + video_length - 1}", flush=True)

    try:
        output_path = encode_video(camera, demo_num, start_frame, video_length)
        encoded_videos.append(output_path)
        print(f"      -> {output_path}", flush=True)
    except Exception as e:
        print(f"      -> [ERROR] {e}", flush=True)

# ============================================================
# 8. 결과 요약
# ============================================================
print("\n" + "=" * 60, flush=True)
print("[완료]", flush=True)
print(f"      생성된 비디오: {len(encoded_videos)}개", flush=True)
for v in encoded_videos:
    print(f"      - {v}", flush=True)
print("=" * 60, flush=True)

print("\n[완료] 프로세스 종료 중...", flush=True)
sys.stdout.flush()

os._exit(0)
