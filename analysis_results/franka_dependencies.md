# Franka/Panda Pipeline 종속성 분석 보고서

> 생성일: 2026-02-23
> 목적: G1 휴머노이드 로봇 마이그레이션을 위한 현재 Franka Panda 종속 지점 완전 식별

---

## 1. 전체 파이프라인 요약

### 1.1 3단계 파이프라인 구조

```
Stage 1: Isaac Lab Simulation
  annotated_dataset.hdf5 (10 demos, Franka)
    → MimicGen DataGenerator
    → generated_dataset.hdf5 (40 demos) + generated_dataset_failed.hdf5 (36 demos)
    → PNG frames (normals + segmentation, 2 cameras)

Stage 2: GPU Video Encoding (Warp/CUDA)
  PNG frames → MP4 videos (Lambertian shading)

Stage 3: Cosmos AI Transformation
  MP4 + stacking_prompt.toml → Visually diverse MP4 videos
```

### 1.2 핵심 수치

| 항목 | 값 |
|------|-----|
| Task 환경 | `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0` |
| 로봇 | Franka Panda 7-DoF + 2-finger gripper |
| Action dim | 7 (dx, dy, dz, d_ax, d_ay, d_az, gripper) |
| Joint dim | 9 (7 revolute + 2 gripper finger joints) |
| EEF 표현 | 4x4 pose matrix (position + rotation) |
| 오브젝트 | 3 cubes (cube_1, cube_2, cube_3) |
| 서브태스크 | 4개 (grasp_1 → stack_1 → grasp_2 → final) |
| 성공률 | ~52.6% (40/76 trials) |
| 카메라 | table_cam, table_high_cam (normals + segmentation) |

---

## 2. Franka 종속 지점 상세 목록

### 2.1 프로젝트 로컬 파일 (직접 수정 가능)

#### `notebook/generate_data_standalone.py`

| 라인 | 코드 | 종속 유형 | G1 변경 필요 |
|------|------|----------|:---:|
| 60 | `"task": "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0"` | Task 이름 하드코딩 | **O** |
| 63 | `"input_file": "datasets/annotated_dataset.hdf5"` | Franka 데모 HDF5 | **O** |

#### `notebook/notebook_widgets.py`

| 라인 | 코드 | 종속 유형 | G1 변경 필요 |
|------|------|----------|:---:|
| 252-254 | `available_tasks = ["Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0"]` | Task 이름 리스트 | **O** |

#### `notebook/stacking_prompt.toml`

| 라인 | 현재 내용 | 종속 유형 | G1 변경 필요 |
|------|----------|----------|:---:|
| 3 | `"a robotic arm performing a precise block-stacking operation"` | Franka=단일 팔 | **O** |
| 5 | `"a sleek industrial design clad in smooth white plastic with metallic joints"` | **Franka 외형** (흰색 플라스틱, 금속 관절) | **O** |
| 6 | `"extends from its base to manipulate the cubes"` | 고정형 로봇 팔 전제 | **O** |
| 3-4 | `"block-stacking operation"`, `"three cubes"` | Task 종속 (큐브 스태킹) | **O** (task 변경 시) |
| 11-16 | `cube_description` 변수 리스트 | Task 종속 (큐브 재질) | **O** (task 변경 시) |
| 17-21 | `table_material` 변수 리스트 | 범용 | 유지 가능 |
| 22-27 | `location` 변수 리스트 | 범용 | 유지 가능 |
| 30-32 | `negative_prompt`: `"human workers or human arms"` | 범용 | 유지 가능 |

#### `datasets/annotated_dataset.hdf5` (입력 데이터)

| 필드 | 내용 | 종속 유형 |
|------|------|----------|
| `env_args/env_name` | `"Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"` | Franka task 이름 |
| `data/demo_*/actions` | shape (T, 7) — Franka 7D IK-Rel | Franka action space |
| `data/demo_*/obs/joint_pos` | shape (T, 9) — 7 joints + 2 gripper | Franka joint space |
| `data/demo_*/obs/joint_vel` | shape (T, 9) | Franka joint space |
| `data/demo_*/obs/eef_pos` | shape (T, 3) — panda_hand position | Franka EEF |
| `data/demo_*/obs/eef_quat` | shape (T, 4) — w,x,y,z | Franka EEF |
| `data/demo_*/obs/gripper_pos` | shape (T, 2) — 2 finger positions | Franka gripper |
| `data/demo_*/obs/datagen_info/eef_pose/franka` | shape (T, 4, 4) | EEF name = "franka" |
| `data/demo_*/obs/datagen_info/target_eef_pose/franka` | shape (T, 4, 4) | EEF name = "franka" |
| `data/demo_*/obs/datagen_info/subtask_term_signals/*` | grasp_1, grasp_2, stack_1 | Task-specific subtask |
| `data/demo_*/initial_state/articulation/robot/joint_position` | shape (1, 9) | Franka joint space |

> **핵심**: 입력 HDF5의 `datagen_info/eef_pose/franka` 키에서 `franka`는 EEF 이름으로 사용됨. G1에서는 이것이 다른 이름(예: `g1`)이 될 것임.

---

### 2.2 Isaac Lab 컨테이너 내부 파일 (Isaac Lab 패키지)

#### 환경 등록: `isaaclab_mimic/envs/__init__.py`

| 라인 | 코드 | 설명 |
|------|------|------|
| 27-33 | `gym.register(id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0", entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv", env_cfg_entry_point=FrankaCubeStackIKRelBlueprintMimicEnvCfg)` | Gymnasium 환경 등록 |
| 18-24 | `gym.register(id="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0", ...)` | 입력 데이터 환경 등록 |

#### 환경 Wrapper: `isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env.py`

| 라인 | 메서드 | Franka 종속 내용 |
|------|--------|-----------------|
| 18-36 | `get_robot_eef_pose()` | `obs_buf["policy"]["eef_pos"]`, `obs_buf["policy"]["eef_quat"]` → 4x4 pose matrix |
| 38-83 | `target_eef_pose_to_action()` | Delta position + delta axis-angle rotation + gripper → 7D action |
| 85-123 | `action_to_target_eef_pose()` | 7D action → target 4x4 EEF pose (역변환) |
| 125-136 | `actions_to_gripper_actions()` | `actions[:, -1:]` (마지막 dim이 gripper) |
| 138-161 | `get_subtask_term_signals()` | `obs_buf["subtask_terms"]["grasp_1/grasp_2/stack_1"]` |

> **핵심**: 이 클래스는 `ManagerBasedRLMimicEnv`를 상속하며, G1용으로 새로 작성해야 함. Action space가 다르면 `target_eef_pose_to_action()`, `action_to_target_eef_pose()` 로직이 달라짐.

#### 환경 설정 (Blueprint): `isaaclab_tasks/.../stack_ik_rel_blueprint_env_cfg.py`

| 라인 | 설정 | Franka 종속 내용 |
|------|------|-----------------|
| 27 | `from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG` | Franka 로봇 asset |
| 219 | `self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(...)` | Franka USD/URDF 모델 |
| 220 | `self.scene.robot.spawn.semantic_tags = [("class", "robot")]` | 로봇 semantic tag |
| 223-230 | `self.actions.arm_action = DifferentialInverseKinematicsActionCfg(joint_names=["panda_joint.*"], body_name="panda_hand", ...)` | **Franka 관절/바디 이름** |
| 228 | `controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")` | IK 컨트롤러 설정 |
| 229 | `body_offset=...OffsetCfg(pos=[0.0, 0.0, 0.107])` | **Franka hand 오프셋** |
| 107-109 | `joint_pos = ObsTerm(func=mdp.joint_pos_rel)`, `joint_vel = ObsTerm(func=mdp.joint_vel_rel)` | Joint observation |
| 113-115 | `eef_pos = ObsTerm(func=mdp.ee_frame_pos)`, `eef_quat = ObsTerm(func=mdp.ee_frame_quat)`, `gripper_pos = ObsTerm(func=mdp.gripper_pos)` | EEF/Gripper observation |
| 244-271 | `CameraCfg(...)` | 카메라 위치/방향 (로봇 크기에 종속) |

#### 환경 설정 (MimicGen): `isaaclab_mimic/envs/franka_stack_ik_rel_blueprint_mimic_env_cfg.py`

| 라인 | 설정 | 내용 |
|------|------|------|
| 15 | `class FrankaCubeStackIKRelBlueprintMimicEnvCfg(FrankaCubeStackBlueprintEnvCfg, MimicEnvCfg)` | 상속 체인 |
| 25 | `datagen_config.name = "isaac_lab_franka_stack_ik_rel_blueprint_D0"` | 데이터 이름 |
| 37-126 | `subtask_configs` (4개 SubTaskConfig) | 서브태스크 정의 |
| 127 | `self.subtask_configs["franka"] = subtask_configs` | **EEF 이름 = "franka"** |

> **핵심**: `self.subtask_configs["franka"]`에서 키 `"franka"`가 EEF 이름으로 사용됨. 이것이 HDF5의 `datagen_info/eef_pose/franka` 경로와 매핑됨.

#### 기본 환경 설정 (IK-Rel): `isaaclab_tasks/.../stack_ik_rel_env_cfg.py`

| 라인 | 설정 | 내용 |
|------|------|------|
| 15 | `from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG` | Franka 로봇 import |
| 26 | `self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(...)` | Franka 모델 적용 |
| 29-36 | `DifferentialInverseKinematicsActionCfg(joint_names=["panda_joint.*"], body_name="panda_hand", ...)` | Franka IK 설정 |

#### Randomization 이벤트: `isaaclab_tasks/.../franka_stack_events.py`

| 함수 | Franka 종속 내용 |
|------|-----------------|
| `randomize_joint_by_gaussian_offset()` | `joint_pos[:, :-2]` (마지막 2개 = gripper finger 제외) |
| `set_default_joint_pose()` | `env.scene["robot"].data.default_joint_pos` 참조 |
| `randomize_object_pose()` | 큐브 위치 랜덤화 (Franka와 무관, task 종속) |
| `randomize_scene_lighting_domelight()` | 조명 랜덤화 (범용) |

#### 환경 등록 (Task): `isaaclab_tasks/.../franka/__init__.py`

| 라인 | 환경 ID | 설명 |
|------|---------|------|
| 25-32 | `Isaac-Stack-Cube-Franka-v0` | Joint position control |
| 48-56 | `Isaac-Stack-Cube-Franka-IK-Rel-v0` | IK relative control |
| 67-74 | `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0` | Blueprint (카메라 포함) |

---

### 2.3 MimicGen 핵심 API (로봇 비종속적)

아래 API들은 로봇에 **직접 종속되지 않으며**, 환경 인터페이스를 통해 추상화되어 있음:

| 파일 | 클래스/함수 | 역할 | 로봇 종속? |
|------|-----------|------|:---:|
| `data_generator.py` | `DataGenerator` | 궤적 생성 메인 클래스 | X (env 인터페이스 사용) |
| `datagen_info.py` | `DatagenInfo` | 데이터 구조 (eef_pose, object_poses, subtask signals) | X |
| `datagen_info_pool.py` | `DataGenInfoPool` | HDF5에서 DatagenInfo 로드/관리 | **부분** (eef_name 키 사용) |
| `waypoint.py` | `Waypoint`, `WaypointSequence`, `WaypointTrajectory` | 6-DoF 궤적 표현 및 실행 | X |
| `selection_strategy.py` | `NearestNeighborObjectStrategy`, `RandomStrategy` | 소스 데모 선택 전략 | X |
| `generation.py` | `setup_env_config()`, `env_loop()`, `setup_async_generation()` | 생성 루프 관리 | X |
| `utils.py` | `reset_env()`, `setup_output_paths()` | 유틸리티 | X |

> **핵심 발견**: MimicGen의 핵심 알고리즘(DataGenerator, Waypoint, SelectionStrategy)은 로봇 비종속적.
> 로봇 종속성은 환경 래퍼(`FrankaCubeStackIKRelMimicEnv`)와 환경 설정(`FrankaCubeStackBlueprintEnvCfg`)에 집중됨.

---

## 3. 데이터 흐름 분석

### 3.1 입력 → 출력 데이터 변환

```
annotated_dataset.hdf5 (입력)
  └─ data/demo_0/
       ├─ actions: (T, 7)                     ← Franka IK-Rel 7D
       ├─ obs/
       │   ├─ joint_pos: (T, 9)               ← Franka 9D
       │   ├─ joint_vel: (T, 9)               ← Franka 9D
       │   ├─ eef_pos: (T, 3)
       │   ├─ eef_quat: (T, 4)               ← w,x,y,z
       │   ├─ gripper_pos: (T, 2)             ← Franka 2-finger
       │   ├─ object: (T, 39)                 ← 3 cubes × 13D
       │   ├─ cube_positions: (T, 9)          ← 3 cubes × 3D
       │   ├─ cube_orientations: (T, 12)      ← 3 cubes × 4D
       │   └─ datagen_info/                   ← MimicGen annotation (입력만)
       │       ├─ eef_pose/franka: (T, 4, 4)  ← EEF name="franka"
       │       ├─ target_eef_pose/franka: (T, 4, 4)
       │       ├─ object_pose/cube_*/root_pose: (T, 13)
       │       └─ subtask_term_signals/
       │           ├─ grasp_1: (T, 1)         ← 0→1 edge = subtask 완료
       │           ├─ stack_1: (T, 1)
       │           └─ grasp_2: (T, 1)
       ├─ states/articulation/robot/: ...
       └─ initial_state/: ...

  ↓ MimicGen DataGenerator (spatial transformation per subtask)
  ↓  - select_source_demo(): nearest_neighbor_object (nn_k=3)
  ↓  - transform_poses_from_frame_A_to_frame_B(): 오브젝트 프레임 기준 좌표 변환
  ↓  - WaypointTrajectory.execute(): target_eef_pose → action → env.step()

generated_dataset.hdf5 (출력, 성공)
  └─ data/demo_0/
       ├─ actions: (T', 7)                    ← 동일 action space
       ├─ obs/ (datagen_info 없음)
       │   ├─ joint_pos, joint_vel, eef_pos, eef_quat, gripper_pos
       │   ├─ object, cube_positions, cube_orientations
       │   └─ (rgb_camera images 별도 PNG 저장)
       ├─ states/: ...
       └─ initial_state/: ...                 ← 랜덤화된 초기 상태
```

### 3.2 MimicGen 궤적 생성 알고리즘

```
for each trial:
    env.reset() with randomized initial state

    for each subtask (4개):
        1. select_source_demo():
           - 현재 object_pose와 가장 가까운 소스 데모 선택 (top-3 중 랜덤)

        2. Extract source subtask segment:
           - src_eef_poses, src_target_poses, src_gripper_actions
           - subtask boundaries from subtask_term_signals (0→1 edge detection)

        3. Spatial transformation:
           - transform_poses_from_frame_A_to_frame_B()
           - frame_A = current object pose, frame_B = source object pose
           - 결과: transformed_eef_poses (새 오브젝트 위치에 맞게 변환된 궤적)

        4. Build WaypointTrajectory:
           - Interpolation segment (5 steps, linear interp)
           - Transformed subtask segment
           - Action noise (0.03)

        5. Execute trajectory:
           - For each waypoint:
             - target_eef_pose_to_action(): 4x4 pose → 7D delta action
             - env.step(action)
             - Check success_term

    if success at any point → export to generated_dataset.hdf5
    else → export to generated_dataset_failed.hdf5
```

---

## 4. G1 마이그레이션 영향도 분석

### 4.1 변경 필요 항목 (우선순위순)

#### P0: 필수 변경 (MimicGen이 동작하려면 반드시 필요)

| # | 항목 | 파일 위치 | 설명 |
|---|------|----------|------|
| 1 | **G1 로봇 asset 등록** | Isaac Lab 컨테이너 내부 | G1 manipulation용 USD/URDF + PD controller config |
| 2 | **G1 task 환경 설정** | 새 파일 필요 (Isaac Lab 내부) | `G1CubeStackBlueprintEnvCfg` 클래스 — 로봇 모델, joint names, body name, IK 설정 |
| 3 | **G1 Mimic 환경 래퍼** | 새 파일 필요 (Isaac Lab 내부) | `G1CubeStackIKRelMimicEnv` — EEF pose 변환, action 변환 |
| 4 | **G1 Mimic 환경 설정** | 새 파일 필요 (Isaac Lab 내부) | subtask configs, EEF name (`"g1"` 등) |
| 5 | **Gymnasium 등록** | 새 `__init__.py` | `"Isaac-Stack-Cube-G1-IK-Rel-Blueprint-Mimic-v0"` 등록 |
| 6 | **G1 annotated dataset** | 새 HDF5 파일 | G1으로 수집한 human demonstration (실제 데이터 or 텔레옵) |

#### P1: 프로젝트 로컬 변경

| # | 항목 | 파일 | 변경 내용 |
|---|------|------|----------|
| 7 | Task 이름 | `generate_data_standalone.py:60` | `"Isaac-Stack-Cube-G1-IK-Rel-Blueprint-Mimic-v0"` |
| 8 | Task 리스트 | `notebook_widgets.py:252-254` | G1 task 추가 |
| 9 | 입력 데이터 | `generate_data_standalone.py:63` | G1 annotated dataset 경로 |
| 10 | Cosmos 프롬프트 | `stacking_prompt.toml:3,5-6` | G1 외형 묘사로 변경 |

#### P2: 선택적 변경

| # | 항목 | 파일 | 변경 내용 |
|---|------|------|----------|
| 11 | 카메라 위치 | env cfg | G1 크기/자세에 맞는 카메라 재배치 |
| 12 | Randomization | events.py | G1 joint space에 맞는 랜덤화 |
| 13 | Semantic mapping | env cfg | 로봇 색상 매핑 변경 |

### 4.2 변경이 **불필요한** 항목

| 항목 | 이유 |
|------|------|
| MimicGen 핵심 알고리즘 (DataGenerator, WaypointTrajectory) | 로봇 비종속적, 환경 인터페이스로 추상화 |
| 선택 전략 (NearestNeighborObjectStrategy) | 오브젝트 pose 기반, 로봇 무관 |
| 데이터 구조 (DatagenInfo, DataGenInfoPool) | EEF name 키만 변경하면 됨 |
| GPU 비디오 인코딩 (notebook_utils.py) | 이미지 처리, 로봇 무관 |
| Cosmos API 통신 (cosmos_request.py, app.py) | 비디오 처리, 로봇 무관 |
| HDF5 스키마 구조 | 동일한 구조 유지 (dim만 다를 수 있음) |

### 4.3 G1 특이사항

| 항목 | Franka Panda | G1 (예상) |
|------|-------------|-----------|
| DoF | 7 (단일 팔) | TBD (듀얼 암 가능) |
| EEF | panda_hand (1개) | 양손 (2개 가능) |
| Action space | 7D (6D pose + 1D gripper) | TBD |
| Gripper | 2-finger parallel | TBD |
| 로봇 형태 | 고정형 팔 | 휴머노이드 (상체/하체) |
| MimicGen 제한 | N/A | **현재 single EEF만 지원** (data_generator.py:51) |

> **주의**: `DataGenerator.__init__()`에서 `len(self.env_cfg.subtask_configs) != 1` 이면 에러 발생.
> G1이 듀얼 암을 사용하면 MimicGen 코드 자체 수정이 필요할 수 있음.

---

## 5. Isaac Lab 상속 체인 (환경 설정)

```
stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg          ← 기본 task (scene, obs, rewards, events)
    ↓ 상속
stack_ik_rel_env_cfg.FrankaCubeStackEnvCfg             ← IK-Rel action 추가
    ↓ 상속
stack_ik_rel_blueprint_env_cfg.FrankaCubeStackBlueprintEnvCfg  ← 카메라, RGB obs 추가
    ↓ 상속 (+ MimicEnvCfg)
franka_stack_ik_rel_blueprint_mimic_env_cfg.FrankaCubeStackIKRelBlueprintMimicEnvCfg
                                                       ← MimicGen subtask configs 추가

환경 래퍼:
ManagerBasedRLMimicEnv
    ↓ 상속
FrankaCubeStackIKRelMimicEnv                           ← EEF/action 변환 메서드
```

G1에서는 이 전체 상속 체인을 G1용으로 새로 구성해야 함.

---

## 6. Cosmos 프롬프트 종속성

### 6.1 프롬프트 구조

```toml
[prompt]
description = "The scene depicts {description}... {cube_description} cubes... {table_material} table... {location}..."

[variables]
cube_description = ["glass", "brightly colored plastic", ...]
table_material = ["wood", "marble", ...]
location = ["cluttered workshop", ...]

[negative_prompt]
description = "The scene shows human workers or human arms."
```

### 6.2 Franka 종속 표현

| 표현 | 위치 | G1 변경안 |
|------|------|----------|
| `"a robotic arm"` | description, line 3 | `"a humanoid robot"` 또는 `"a bipedal robot with two arms"` |
| `"smooth white plastic with metallic joints"` | description, line 5 | G1 외형: `"dark grey/black humanoid torso with articulated dual arms"` |
| `"extends from its base"` | description, line 6 | `"stands upright and uses its hands"` |
| `"block-stacking operation"` | description, line 3 | task에 따라 변경 |

### 6.3 프롬프트 사용 경로

```
stacking_prompt.toml
  → notebook_widgets.py:PromptManager.update_prompt()
    → description.format(**variables)
  → cosmos_request.py → POST /canny/submit
    → Cosmos AI 서버 (H100)
```

---

## 7. DataGenInfoPool HDF5 파싱 로직

`datagen_info_pool.py:_add_episode()` (라인 81-180)에서 HDF5를 파싱하는 두 가지 경로:

### 경로 1: `datagen_info` 키가 있는 경우 (annotated dataset)

```python
eef_pose = ep_grp["obs"]["datagen_info"]["eef_pose"][eef_name]          # (T, 4, 4)
target_eef_pose = ep_grp["obs"]["datagen_info"]["target_eef_pose"][eef_name]  # (T, 4, 4)
object_poses_dict = ep_grp["obs"]["datagen_info"]["object_pose"]        # dict
subtask_term_signals = ep_grp["obs"]["datagen_info"]["subtask_term_signals"]  # dict
```

### 경로 2: `datagen_info` 키가 없는 경우 (fallback)

```python
eef_pos = ep_grp["obs"]["eef_pos"]
eef_quat = ep_grp["obs"]["eef_quat"]   # (w, x, y, z)
eef_pose = PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

object_poses_dict = {}  # ep_grp["obs"]["object_pose"] 에서 root_pose 추출
target_eef_pose = ep_grp["obs"]["target_eef_pose"]
subtask_term_signals = ep_grp["obs"]["subtask_term_signals"]
```

> **G1 데이터 준비 시**: 경로 1을 따르려면 `datagen_info` 구조를 포함한 HDF5를 만들어야 하고,
> 경로 2를 따르려면 `eef_pos`, `eef_quat`, `object_pose`, `target_eef_pose`, `subtask_term_signals`를 직접 제공해야 함.

---

## 8. 결론 및 다음 단계

### 핵심 발견

1. **MimicGen 알고리즘은 로봇 비종속적** — 환경 래퍼와 설정만 새로 작성하면 됨
2. **EEF name `"franka"`가 HDF5 키와 subtask_configs 딕셔너리 키로 사용됨** — G1용으로 변경 필요
3. **현재 MimicGen은 single EEF만 지원** — G1 듀얼 암 사용 시 확장 필요
4. **Cosmos 프롬프트에 Franka 외형이 하드코딩** — G1 외형으로 전체 재작성 필요
5. **입력 HDF5 (annotated_dataset)가 핵심 병목** — G1용 human demonstration 수집이 선행되어야 함

### G1 마이그레이션 로드맵

```
Phase 1: G1 Isaac Lab 환경 구축
  - G1 manipulation task 환경 설정 (scene, robot, objects, cameras)
  - G1 IK controller 설정 (joint names, body names, offsets)
  - G1 MimicGen 래퍼 (EEF pose ↔ action 변환)
  - Gymnasium 환경 등록

Phase 2: G1 데모 데이터 수집
  - G1 텔레오퍼레이션 또는 실제 데이터 수집
  - annotated_dataset.hdf5 형식으로 변환
  - subtask annotation (grasp, stack 등)

Phase 3: 프로젝트 로컬 변경
  - generate_data_standalone.py task 이름 변경
  - notebook_widgets.py task 리스트 업데이트
  - stacking_prompt.toml G1 외형 프롬프트 작성

Phase 4: 검증
  - G1 MimicGen 데이터 생성 테스트
  - 성공률 확인 및 파라미터 튜닝
  - Cosmos 비주얼 변환 테스트
```
