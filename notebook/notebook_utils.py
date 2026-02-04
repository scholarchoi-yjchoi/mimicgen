import os
import warp as wp
import numpy as np
from PIL import Image


# Set directories to use for inputs/outputs
ISAACLAB_OUTPUT_DIR = "output"           # 이미지 입력 소스 (v18 계층 구조)
VIDEO_OUTPUT_DIR = "output/videos"       # 비디오 출력 (로컬 PC에서 접근 가능)
COSMOS_OUTPUT_DIR = "output/cosmos"      # Cosmos 처리 결과 (로컬 접근 가능)

# Video and rendering settings
DEFAULT_FRAMERATE = 24.0
DEFAULT_LIGHT_DIRECTION = (0.0, 0.0, 1.0)  # Points straight down at surface


@wp.kernel
def _shade_segmentation(
    segmentation: wp.array3d(dtype=wp.uint8),
    normals: wp.array3d(dtype=wp.float32),
    shading_out: wp.array3d(dtype=wp.uint8),
    light_source: wp.array(dtype=wp.vec3f),
):
    """Apply shading to semantic segmentation using surface normals.

    Args:
        segmentation: Input semantic segmentation image (H,W,C)
        normals: Surface normal vectors (H,W,3)
        shading_out: Output shaded segmentation image (H,W,C)
        light_source: Position of light source
    """
    i, j = wp.tid()
    normal = normals[i, j]
    light_source_vec = wp.normalize(light_source[0])
    shade = 0.5 + wp.dot(wp.vec3f(normal[0], normal[1], normal[2]), light_source_vec) * 0.5

    shading_out[i, j, 0] = wp.uint8(wp.float32(segmentation[i, j, 0]) * shade)
    shading_out[i, j, 1] = wp.uint8(wp.float32(segmentation[i, j, 1]) * shade)
    shading_out[i, j, 2] = wp.uint8(wp.float32(segmentation[i, j, 2]) * shade)
    shading_out[i, j, 3] = wp.uint8(255)

def get_env_trial_frames(root_dir: str, camera_name: str, min_frames: int = 30) -> dict:
    """Get frame ranges for each demo from the hierarchical folder structure.

    Args:
        root_dir: Directory containing the camera folders (e.g., "output")
        camera_name: Name of the camera (e.g., "table_cam")
        min_frames: Minimum number of frames required for a valid demo

    Returns:
        dict: {env_num: {demo_num: (start_frame, end_frame)}}
              env_num is always 0 (single environment)
    """
    import re

    # 새 구조: {camera}_segmentation/demo_{N}/step_{frame:04d}.png
    segmentation_dir = os.path.join(root_dir, f"{camera_name}_segmentation")

    if not os.path.isdir(segmentation_dir):
        return {}

    valid_trials = {}

    # demo_N 폴더들 스캔
    for demo_folder in os.listdir(segmentation_dir):
        match = re.match(r"demo_(\d+)", demo_folder)
        if not match:
            continue

        demo_num = int(match.group(1))
        demo_path = os.path.join(segmentation_dir, demo_folder)

        if not os.path.isdir(demo_path):
            continue

        # step_{frame:04d}.png 파일들 스캔
        frames = []
        for filename in os.listdir(demo_path):
            step_match = re.match(r"step_(\d+)\.png", filename)
            if step_match:
                frames.append(int(step_match.group(1)))

        if len(frames) < min_frames:
            continue

        frames.sort()
        start_frame = frames[0]
        end_frame = frames[-1]

        # 연속성 검증
        expected = set(range(start_frame, end_frame + 1))
        if len(expected - set(frames)) > 0:
            continue

        # env_num = 0 (단일 환경)
        valid_trials.setdefault(0, {})[demo_num] = (start_frame, end_frame)

    return valid_trials

def encode_video(root_dir: str, start_frame: int, num_frames: int, camera_name: str, output_path: str, env_num: int, trial_num: int) -> None:
    """Encode a sequence of shaded segmentation frames into a video.

    새 구조: {root_dir}/{camera}_{modality}/demo_{trial}/step_{frame:04d}.png

    Args:
        root_dir: Directory containing the camera folders (e.g., "output")
        start_frame: Starting frame index
        num_frames: Number of frames to encode
        camera_name: Name of the camera (e.g., "table_cam")
        output_path: Output path for the encoded video
        env_num: Environment number (unused, kept for compatibility)
        trial_num: Demo number for the sequence

    Raises:
        ValueError: If start_frame is negative or if any required frame is missing
    """
    from video_encoding import get_video_encoding_interface

    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    # 새 경로 함수
    def get_frame_path(modality: str, frame_idx: int) -> str:
        return os.path.join(
            root_dir,
            f"{camera_name}_{modality}",
            f"demo_{trial_num}",
            f"step_{frame_idx:04d}.png"
        )

    # 모든 프레임 존재 확인
    for frame_idx in range(start_frame, start_frame + num_frames):
        normals_path = get_frame_path("normals", frame_idx)
        segmentation_path = get_frame_path("segmentation", frame_idx)
        if not os.path.exists(normals_path) or not os.path.exists(segmentation_path):
            raise ValueError(f"Missing frame at frame index {frame_idx} for demo {trial_num}")

    # 비디오 인코딩 초기화
    video_encoding = get_video_encoding_interface()

    first_frame = np.array(Image.open(get_frame_path("segmentation", start_frame)))
    height, width = first_frame.shape[:2]

    # 버퍼 할당
    normals_wp = wp.empty((height, width, 3), dtype=wp.float32, device="cuda")
    segmentation_wp = wp.empty((height, width, 4), dtype=wp.uint8, device="cuda")
    shaded_segmentation_wp = wp.empty_like(segmentation_wp)
    light_source = wp.array(DEFAULT_LIGHT_DIRECTION, dtype=wp.vec3f, device="cuda")

    video_encoding.start_encoding(
        video_filename=output_path,
        framerate=DEFAULT_FRAMERATE,
        nframes=num_frames,
        overwrite_video=True,
    )

    for frame_idx in range(start_frame, start_frame + num_frames):
        normals_np = np.array(Image.open(get_frame_path("normals", frame_idx))).astype(np.float32) / 255.0
        wp.copy(normals_wp, wp.from_numpy(normals_np))

        segmentation_np = np.array(Image.open(get_frame_path("segmentation", frame_idx)))
        wp.copy(segmentation_wp, wp.from_numpy(segmentation_np))

        wp.launch(_shade_segmentation, dim=(height, width),
                  inputs=[segmentation_wp, normals_wp, shaded_segmentation_wp, light_source])

        video_encoding.encode_next_frame_from_buffer(
            shaded_segmentation_wp.numpy().tobytes(), width=width, height=height)
