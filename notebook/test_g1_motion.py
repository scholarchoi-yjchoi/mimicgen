#!/usr/bin/env python3
"""
G1 Humanoid Dual-Arm Motion Visualization Test
- Uses env.step() with EEF target actions (Pink IK) for proper PhysX→Fabric→Renderer sync
- fabric=True (default) enables Fabric render bridge for articulation updates
- 4 camera views (front/back/left/right) in separate folders
- 8 motion phases x 8 frames = 64 frames per camera (256 total)
- 1280x720 resolution, black bg + white text overlay
"""

import sys
import os
import traceback

try:
    # CRITICAL: pinocchio must be imported before Isaac Sim
    import pinocchio  # noqa: F401

    import nest_asyncio
    nest_asyncio.apply()

    import torch
    import numpy as np

    print("=" * 60, flush=True)
    print("[G1 Motion Test] Starting (env.step EEF version)...", flush=True)
    print("=" * 60, flush=True)

    # ============================================================
    # Isaac Sim AppLauncher
    # ============================================================
    from argparse import ArgumentParser
    from isaaclab.app import AppLauncher

    parser = ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args([])
    args_cli.enable_cameras = True
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("[1/4] AppLauncher ready", flush=True)

    # ============================================================
    # Imports (after AppLauncher)
    # ============================================================
    import gymnasium as gym
    import isaaclab_mimic.envs  # noqa: F401
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    from isaaclab_mimic.datagen.generation import setup_env_config
    from isaaclab_mimic.datagen.utils import setup_output_paths, reset_env
    from isaaclab.utils.assets import retrieve_file_path, ISAACLAB_NUCLEUS_DIR

    # Pre-download G1 URDF
    _urdf_nucleus = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
    retrieve_file_path(_urdf_nucleus)
    print("[2/4] Modules imported", flush=True)

    # ============================================================
    # Environment Setup (fabric=True for PhysX→Fabric→Renderer sync)
    # ============================================================
    env_name = "Isaac-Locomanipulation-G1-Abs-Mimic-v0"
    output_dir, output_file_name = setup_output_paths("datasets/motion_test.hdf5")

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=1,
        device="cuda:0",
        generation_num_trials=1,
    )

    # fabric=True (default): PhysX → Fabric buffers → Renderer sees updates
    # fabric=False would freeze rendering — PhysX runs but renderer never gets updates

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    reset_env(env, 100)
    fabric_status = env.sim.cfg.use_fabric
    print(f"[3/4] Environment ready (use_fabric={fabric_status})", flush=True)
    if not fabric_status:
        print("  WARNING: Fabric is NOT enabled! Rendering will likely fail.", flush=True)

    # ============================================================
    # Verify action format at runtime
    # ============================================================
    print("\n--- Action Manager Terms ---", flush=True)
    action_dim_total = 0
    for name, term in env.action_manager._terms.items():
        dim = term.action_dim
        print(f"  {name}: dim={dim}", flush=True)
        action_dim_total += dim
    print(f"  TOTAL action_dim = {action_dim_total}", flush=True)

    # Read initial EEF positions from observations
    obs = env.observation_manager.compute_group("policy")
    init_left_eef = obs["left_eef_pos"][0].cpu().numpy()
    init_right_eef = obs["right_eef_pos"][0].cpu().numpy()
    init_left_quat = obs["left_eef_quat"][0].cpu().numpy()
    init_right_quat = obs["right_eef_quat"][0].cpu().numpy()
    print(f"\nInitial EEF positions:", flush=True)
    print(f"  Left EEF pos:  ({init_left_eef[0]:.4f}, {init_left_eef[1]:.4f}, {init_left_eef[2]:.4f})", flush=True)
    print(f"  Left EEF quat: ({init_left_quat[0]:.4f}, {init_left_quat[1]:.4f}, {init_left_quat[2]:.4f}, {init_left_quat[3]:.4f})", flush=True)
    print(f"  Right EEF pos:  ({init_right_eef[0]:.4f}, {init_right_eef[1]:.4f}, {init_right_eef[2]:.4f})", flush=True)
    print(f"  Right EEF quat: ({init_right_quat[0]:.4f}, {init_right_quat[1]:.4f}, {init_right_quat[2]:.4f}, {init_right_quat[3]:.4f})", flush=True)

    # ============================================================
    # Setup cameras
    # ============================================================
    import omni.replicator.core as rep

    OUTPUT_DIR = "/workspace/isaaclab/output/motion_test"
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    CAM_NAMES = ["front", "back", "left", "right"]
    CAMERA_CONFIGS = {
        "front": {"pos": (0.0, 2.0, 1.2), "target": (0.0, 0.0, 0.9)},
        "back":  {"pos": (0.0, -1.5, 1.2), "target": (0.0, 0.0, 0.9)},
        "left":  {"pos": (-2.0, 0.3, 1.2), "target": (0.0, 0.0, 0.9)},
        "right": {"pos": (2.0, 0.3, 1.2), "target": (0.0, 0.0, 0.9)},
    }

    for cam_name in CAM_NAMES:
        os.makedirs(os.path.join(OUTPUT_DIR, cam_name), exist_ok=True)

    cameras = {}
    render_products = {}
    annotators = {}

    RES_W, RES_H = 1280, 720

    for cam_name in CAM_NAMES:
        cam_cfg = CAMERA_CONFIGS[cam_name]
        cam = rep.create.camera(
            position=cam_cfg["pos"],
            look_at=cam_cfg["target"],
        )
        rp = rep.create.render_product(cam, (RES_W, RES_H))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        cameras[cam_name] = cam
        render_products[cam_name] = rp
        annotators[cam_name] = ann

    print(f"\n[4/4] {len(cameras)} cameras created ({RES_W}x{RES_H})", flush=True)

    # ============================================================
    # Helper functions
    # ============================================================
    from PIL import Image, ImageDraw

    def step_with_eef_targets(env, left_pos, left_quat, right_pos, right_quat, num_substeps=5):
        """Step the environment using EEF target poses via Pink IK.

        This uses env.step() which triggers the full pipeline:
        Pink IK solver → actuator model → scene.write_data_to_sim() →
        sim.step() → Fabric → Renderer.
        """
        action = torch.zeros(1, action_dim_total, device="cuda:0")
        action[0, 0:3] = torch.tensor(left_pos, device="cuda:0", dtype=torch.float32)
        action[0, 3:7] = torch.tensor(left_quat, device="cuda:0", dtype=torch.float32)
        action[0, 7:10] = torch.tensor(right_pos, device="cuda:0", dtype=torch.float32)
        action[0, 10:14] = torch.tensor(right_quat, device="cuda:0", dtype=torch.float32)
        # [14:21] left hand joints = 0 (neutral)
        # [21:28] right hand joints = 0 (neutral)
        # [28:32] lower body = 0 (stand still)

        for _ in range(num_substeps):
            obs, _, terminated, truncated, info = env.step(action)
            if terminated.any() or truncated.any():
                print("  WARNING: Episode terminated/truncated during stepping!", flush=True)

        # Pump Kit event loop for Fabric→Renderer sync + annotator processing
        for _ in range(3):
            simulation_app.update()

    def capture_frame(frame_idx, phase_name, left_eef, right_eef):
        """Capture from all 4 cameras with black bg + white text overlay."""
        for cam_name in CAM_NAMES:
            ann = annotators[cam_name]
            data = ann.get_data()
            if data is None or data.size == 0:
                continue
            if len(data.shape) == 3 and data.shape[2] == 4:
                img = Image.fromarray(data[:, :, :3])
            else:
                img = Image.fromarray(data)

            draw = ImageDraw.Draw(img)
            text_lines = [
                f"Frame {frame_idx:03d} | {phase_name}",
                f"L_eef: ({left_eef[0]:.3f}, {left_eef[1]:.3f}, {left_eef[2]:.3f})",
                f"R_eef: ({right_eef[0]:.3f}, {right_eef[1]:.3f}, {right_eef[2]:.3f})",
                f"View: {cam_name}",
            ]
            line_h = 16
            padding = 6
            max_text_w = 0
            for line in text_lines:
                bbox = draw.textbbox((0, 0), line)
                tw = bbox[2] - bbox[0]
                if tw > max_text_w:
                    max_text_w = tw
            block_h = len(text_lines) * line_h + padding * 2
            block_w = max_text_w + padding * 2

            draw.rectangle([(8, 8), (8 + block_w, 8 + block_h)], fill=(0, 0, 0))

            y = 8 + padding
            for line in text_lines:
                draw.text((8 + padding, y), line, fill=(255, 255, 255))
                y += line_h

            filename = f"frame_{frame_idx:03d}.png"
            img.save(os.path.join(OUTPUT_DIR, cam_name, filename))

    def lerp_pos(start, end, alpha):
        """Linear interpolation between two 3D positions."""
        return tuple(s + alpha * (e - s) for s, e in zip(start, end))

    # ============================================================
    # Define motion keyframes as EEF target poses
    # ============================================================
    # Robot at (0, 0, 0.75), facing +Y
    # Initial EEF positions read from obs above — use those as rest pose
    rest_left = tuple(init_left_eef.tolist())
    rest_right = tuple(init_right_eef.tolist())
    identity_quat = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z

    print(f"\nRest pose: L={rest_left}, R={rest_right}", flush=True)

    MOTION_PHASES = [
        {"name": "T-pose (arms out)",    "frames": 8,
         "left_pos": (-0.45, 0.15, 0.95), "right_pos": (0.45, 0.15, 0.95)},
        {"name": "Arms forward",         "frames": 8,
         "left_pos": (-0.15, 0.55, 0.95), "right_pos": (0.15, 0.55, 0.95)},
        {"name": "Arms up high",         "frames": 8,
         "left_pos": (-0.15, 0.20, 1.25), "right_pos": (0.15, 0.20, 1.25)},
        {"name": "L up R down",          "frames": 8,
         "left_pos": (-0.15, 0.20, 1.25), "right_pos": (0.30, 0.40, 0.70)},
        {"name": "R up L down",          "frames": 8,
         "left_pos": (-0.30, 0.40, 0.70), "right_pos": (0.15, 0.20, 1.25)},
        {"name": "Both reach table",     "frames": 8,
         "left_pos": (-0.20, 0.50, 0.75), "right_pos": (0.20, 0.50, 0.75)},
        {"name": "Wide spread",          "frames": 8,
         "left_pos": (-0.40, 0.30, 0.90), "right_pos": (0.40, 0.30, 0.90)},
        {"name": "Return to rest",       "frames": 8,
         "left_pos": rest_left, "right_pos": rest_right},
    ]

    # ============================================================
    # Execute motion and capture
    # ============================================================
    total_frames = sum(p["frames"] for p in MOTION_PHASES)
    print(f"\n{'=' * 60}", flush=True)
    print(f"Starting motion capture (env.step with EEF targets)...", flush=True)
    print(f"  Phases: {len(MOTION_PHASES)}", flush=True)
    print(f"  Frames per camera: {total_frames}", flush=True)
    print(f"  Total images: {total_frames * len(CAM_NAMES)}", flush=True)
    print(f"  Substeps per frame: 5", flush=True)
    print(f"  Resolution: {RES_W}x{RES_H}", flush=True)
    print(f"  Output: {OUTPUT_DIR}/{{front,back,left,right}}/", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Initial render warmup — use env.step with current EEF to stabilize
    print("\nWarmup: 20 env.step() calls...", flush=True)
    warmup_action = torch.zeros(1, action_dim_total, device="cuda:0")
    warmup_action[0, 0:3] = torch.tensor(init_left_eef, device="cuda:0", dtype=torch.float32)
    warmup_action[0, 3:7] = torch.tensor(init_left_quat, device="cuda:0", dtype=torch.float32)
    warmup_action[0, 7:10] = torch.tensor(init_right_eef, device="cuda:0", dtype=torch.float32)
    warmup_action[0, 10:14] = torch.tensor(init_right_quat, device="cuda:0", dtype=torch.float32)
    for i in range(20):
        env.step(warmup_action)
    for _ in range(5):
        simulation_app.update()
    print("Warmup done.", flush=True)

    # Validate rendering: check that cameras produce non-blank frames
    test_data = annotators["front"].get_data()
    if test_data is not None and test_data.size > 0:
        mean_val = test_data[:, :, :3].mean()
        print(f"  Render validation: mean pixel value = {mean_val:.1f} (should be > 0)", flush=True)
        if mean_val < 1.0:
            print("  WARNING: Render appears blank! Fabric→Renderer sync may not be working.", flush=True)
    else:
        print("  WARNING: No render data from annotator!", flush=True)

    # Read actual EEF after warmup to use as starting point
    obs = env.observation_manager.compute_group("policy")
    current_left = tuple(obs["left_eef_pos"][0].cpu().numpy().tolist())
    current_right = tuple(obs["right_eef_pos"][0].cpu().numpy().tolist())
    print(f"Post-warmup EEF: L={current_left}, R={current_right}", flush=True)

    global_frame = 0

    for phase_idx, phase in enumerate(MOTION_PHASES):
        phase_name = phase["name"]
        num_frames = phase["frames"]
        target_left = phase["left_pos"]
        target_right = phase["right_pos"]

        print(f"\n--- Phase {phase_idx+1}/{len(MOTION_PHASES)}: {phase_name} ({num_frames} frames) ---", flush=True)
        print(f"    Target L: {target_left}", flush=True)
        print(f"    Target R: {target_right}", flush=True)

        start_left = current_left
        start_right = current_right

        for f in range(num_frames):
            alpha = (f + 1) / num_frames

            interp_left = lerp_pos(start_left, target_left, alpha)
            interp_right = lerp_pos(start_right, target_right, alpha)

            # Step environment with EEF targets (Pink IK handles joint solving)
            step_with_eef_targets(env, interp_left, identity_quat, interp_right, identity_quat)

            # Read actual EEF from observations
            obs = env.observation_manager.compute_group("policy")
            left_eef = obs["left_eef_pos"][0].cpu().numpy()
            right_eef = obs["right_eef_pos"][0].cpu().numpy()

            # Capture rendered frames
            capture_frame(global_frame, phase_name, left_eef, right_eef)

            print(f"  Frame {global_frame:03d}: "
                  f"L_eef=({left_eef[0]:.3f},{left_eef[1]:.3f},{left_eef[2]:.3f}) "
                  f"R_eef=({right_eef[0]:.3f},{right_eef[1]:.3f},{right_eef[2]:.3f})",
                  flush=True)

            global_frame += 1

        # Update current positions for next phase interpolation start
        obs = env.observation_manager.compute_group("policy")
        current_left = tuple(obs["left_eef_pos"][0].cpu().numpy().tolist())
        current_right = tuple(obs["right_eef_pos"][0].cpu().numpy().tolist())

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 60}", flush=True)
    print(f"[DONE] Motion capture complete!", flush=True)
    for cam_name in CAM_NAMES:
        cam_dir = os.path.join(OUTPUT_DIR, cam_name)
        count = len([f for f in os.listdir(cam_dir) if f.endswith(".png")])
        print(f"  {cam_name}/: {count} frames", flush=True)
    print(f"{'=' * 60}", flush=True)

except Exception as e:
    print(f"\n[ERROR] {e}", flush=True)
    traceback.print_exc()

finally:
    print("\n[EXIT] Shutting down...", flush=True)
    sys.stdout.flush()
    os._exit(0)
