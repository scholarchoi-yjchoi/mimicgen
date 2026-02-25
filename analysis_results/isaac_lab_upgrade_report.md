# Isaac Lab 2.3+ Docker 이미지 확인 보고서

## 조사 일자: 2026-02-24 (업데이트: 2026-02-25)

---

## 1. 현재 환경 vs 필요 환경

| 항목 | 현재 (`gr00t-smmg-bp:1.0`) | 신규 (`isaac-lab:2.3.2`) |
|------|:-------------------------:|:------------------------:|
| Isaac Sim | **4.5.0**-rc.36 | **5.1.0**-rc.19 |
| Isaac Lab | **0.36.2** | **0.54.2** (이미지 태그 2.3.2) |
| isaaclab_mimic | 1.0.3 | **1.0.16** (multi-EEF 지원) |
| Python | 3.10 | **3.11** |
| PyTorch | 2.4.x | **2.7.0+cu128** |
| Ubuntu | 22.04 | 22.04 |
| GUI 지원 | O (X11 forwarding) | **X (headless only)** |
| Jupyter 지원 | O (기본 지원) | **별도 설정 필요** |

## 2. Isaac Lab 2.3.2 Docker 이미지 상태

| 항목 | 상태 |
|------|------|
| NGC 이미지 존재 | **O** (`nvcr.io/nvidia/isaac-lab:2.3.2`) |
| NGC 인증 | **O** (저장됨) |
| 아키텍처 | amd64, arm64 |
| Pull 완료 | **O** (로컬에 다운로드 완료) |
| 디스크 여유 | 116 GB (충분) |

## 3. GPU 호환성 — 실제 테스트 결과

| 항목 | 상태 |
|------|------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q |
| 드라이버 | 580.126.09 |
| **CUDA 인식** | **O** (torch.cuda.is_available() = True) |
| **GPU 이름 인식** | **O** (NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition) |
| **PyTorch 버전** | **2.7.0+cu128** (Blackwell 공식 지원) |
| **CUDA 버전** | **12.8** |

> **결론**: Blackwell GPU가 Isaac Lab 2.3.2 컨테이너에서 **정상 인식**됩니다. PyTorch 2.7.0+cu128은 Blackwell을 공식 지원합니다.

### 알려진 Blackwell 이슈 및 해결책

1. **PhysX GPU 파이프라인 실패** → CPU fallback
   - 해결: `libcuda.so` symlink 생성
   - `ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so`
2. **시뮬레이션 테스트 필요**
   - CUDA/PyTorch는 정상이지만 PhysX GPU 시뮬레이션은 별도 검증 필요

## 4. 컨테이너 내부 구조 — 실제 확인 결과

### 소스 디렉토리

```
/workspace/isaaclab/source/
├── isaaclab/              ← Isaac Lab 코어
├── isaaclab_assets/       ← 로봇/오브젝트 assets
├── isaaclab_contrib/      ← 커뮤니티 기여
├── isaaclab_mimic/        ← MimicGen (multi-EEF 지원!)
├── isaaclab_rl/           ← RL 훈련
└── isaaclab_tasks/        ← 환경 태스크
```

### G1 관련 파일 (이미 구현됨!)

| 파일 | 경로 | 설명 |
|------|------|------|
| G1 로봇 config | `isaaclab_assets/.../robots/unitree.py` | `G1_29DOF_CFG` (29 DoF) |
| G1 Pick-Place 환경 | `isaaclab_tasks/.../manipulation/pick_place/pickplace_unitree_g1_inspire_hand_env_cfg.py` | Inspire hand |
| G1 Locomanipulation 환경 | `isaaclab_tasks/.../locomanipulation/pick_place/locomanipulation_g1_env_cfg.py` | Pink IK 양팔 |
| G1 고정기반 상체 IK | `isaaclab_tasks/.../locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg.py` | 고정 기반 |
| **G1 MimicGen 환경 (cfg)** | `isaaclab_mimic/.../pinocchio_envs/locomanipulation_g1_mimic_env_cfg.py` | **듀얼 암 subtask** |
| **G1 MimicGen 환경 (impl)** | `isaaclab_mimic/.../pinocchio_envs/locomanipulation_g1_mimic_env.py` | **듀얼 암 MimicGen** |
| G1 SDG 환경 | `isaaclab_mimic/.../locomanipulation_sdg/envs/g1_locomanipulation_sdg_env.py` | 합성 데이터 생성 |

### Multi-EEF MimicGen — 코드 레벨 확인

**구 버전 (0.36.2) — single EEF 제한:**
```python
# data_generator.py:50-51
if len(subtask_configs) != 1:
    raise ValueError("Data generation currently supports only one end-effector.")
```

**신 버전 (1.0.16) — multi-EEF 완전 지원:**
```python
# data_generator.py — single EEF assertion 완전 제거
# 대신 모든 메서드가 eef_name 파라미터로 다중 EEF 처리
for eef_name in self.env_cfg.subtask_configs.keys():
    current_eef_selected_src_demo_indices[eef_name] = None
    current_eef_subtask_trajectories[eef_name] = []
    # ... 각 EEF별 독립적 trajectory 관리
```

주요 Multi-EEF API:
- `generate_eef_subtask_trajectory(eef_name, subtask_ind, ...)`
- `merge_eef_subtask_trajectory(eef_name, subtask_index, ...)`
- `select_source_demo(eef_name, eef_pose, ...)`
- `MultiWaypoint(eef_waypoint_dict)` — 여러 EEF waypoint 통합
- `SubTaskConstraintType.COORDINATION` — 양팔 동기화
- `SubTaskConstraintType.SEQUENTIAL` — 순차적 팔 전환

### G1 MimicGen 환경 — 실제 코드 구조

**`locomanipulation_g1_mimic_env_cfg.py`:**
```python
class LocomanipulationG1MimicEnvCfg(LocomanipulationG1EnvCfg, MimicEnvCfg):
    def __post_init__(self):
        # 오른팔 subtask (2단계: pick → place)
        self.subtask_configs["right"] = [
            SubTaskConfig(object_ref="object", subtask_term_signal="idle_right", ...),
            SubTaskConfig(object_ref="object", subtask_term_signal=None, ...),
        ]
        # 왼팔 subtask (1단계: 보조)
        self.subtask_configs["left"] = [
            SubTaskConfig(object_ref="object", subtask_term_signal=None, ...),
        ]
```

**`locomanipulation_g1_mimic_env.py`:**
```python
class LocomanipulationG1MimicEnv(ManagerBasedRLMimicEnv):
    def get_robot_eef_pose(self, eef_name, env_ids):
        # "left" → left_eef_pos/left_eef_quat
        # "right" → right_eef_pos/right_eef_quat

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, ...):
        # 양팔 pose → 통합 action tensor
        # [left_pos(3), left_quat(4), right_pos(3), right_quat(4), left_grip(7), right_grip(?)]

    def action_to_target_eef_pose(self, action):
        # action → {"left": pose_4x4, "right": pose_4x4}

    def actions_to_gripper_actions(self, actions):
        # actions → {"left": actions[:, 14:21], "right": actions[:, 21:]}
```

### SubTaskConstraintConfig — 양팔 동기화

```python
from isaaclab.envs.mimic_env_cfg import SubTaskConstraintConfig, SubTaskConstraintType

# 사용 예: 왼팔 subtask 1과 오른팔 subtask 1이 동시에 끝나도록
constraint = SubTaskConstraintConfig(
    eef_subtask_constraint_tuple=[("left", 1), ("right", 1)],
    constraint_type=SubTaskConstraintType.COORDINATION,
    coordination_scheme=SubTaskConstraintCoordinationScheme.REPLAY,
)

# 사용 예: 오른팔이 먼저 끝난 후 왼팔 시작
constraint = SubTaskConstraintConfig(
    eef_subtask_constraint_tuple=[("right", 0), ("left", 0)],
    constraint_type=SubTaskConstraintType.SEQUENTIAL,
)
```

### G1 Locomanipulation 기본 환경 설정

```python
# locomanipulation_g1_env_cfg.py
class LocomanipulationG1EnvCfg(ManagerBasedRLEnvCfg):
    scene = LocomanipulationG1SceneCfg  # G1_29DOF_CFG + PackingTable + Object
    actions:
        upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG  # Pink IK 양팔 제어
        lower_body_joint_pos = AgileBasedLowerBodyActionCfg  # RL locomotion policy
    observations:
        left_eef_pos = get_eef_pos("left_wrist_yaw_link")
        left_eef_quat = get_eef_quat("left_wrist_yaw_link")
        right_eef_pos = get_eef_pos("right_wrist_yaw_link")
        right_eef_quat = get_eef_quat("right_wrist_yaw_link")
        hand_joint_state = get_robot_joint_state(".*_hand.*")
    # URDF: g1_29dof_with_hand_only_kinematics.urdf
    # XR teleoperation: Apple Vision Pro 지원 (OpenXR)
```

## 5. 핵심 차이점: 헤드리스 전용

**가장 큰 차이점**: Isaac Lab 2.3.2 컨테이너는 **헤드리스 전용**입니다.

| 기능 | 현재 (gr00t-smmg-bp) | isaac-lab:2.3.2 |
|------|:-------------------:|:---------------:|
| Jupyter 웹 UI | O | 별도 설치 필요 |
| X11 GUI | O | X |
| 카메라 렌더링 (PNG) | O | O (headless 가능) |
| Cosmos 연동 | O | 별도 설정 필요 |
| notebook_utils.py (Warp) | O | 이식 필요 |

### 해결 방안: 커스텀 Docker 이미지 빌드 (권장)

```dockerfile
FROM nvcr.io/nvidia/isaac-lab:2.3.2

# Jupyter 및 필수 패키지 (Isaac Lab 내장 Python 사용)
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install \
    jupyterlab nest_asyncio toml flask requests

# 포트 노출
EXPOSE 8888
```

```bash
docker build -t isaac-lab-g1:2.3.2 .
```

## 6. 권장 사항 (업데이트)

### 검증 완료 항목

- [x] 이미지 Pull → `nvcr.io/nvidia/isaac-lab:2.3.2`
- [x] 내부 구조 확인 → isaaclab_mimic 1.0.16, G1 환경 모두 존재
- [x] GPU 테스트 → Blackwell 정상 인식, PyTorch 2.7.0+cu128
- [x] Multi-EEF 코드 확인 → single EEF 제한 완전 제거, 양팔 지원
- [x] G1 MimicGen 환경 확인 → locomanipulation_g1_mimic_env 완전 구현

### 남은 단계

- [ ] 커스텀 이미지 빌드 → Jupyter + Cosmos 패키지 추가
- [ ] docker-compose.g1.yml 작성 → 커스텀 이미지 기반
- [ ] PhysX GPU 시뮬레이션 테스트 → libcuda.so symlink 필요 여부 확인
- [ ] G1 pick-place 환경 실행 테스트

## 7. Franka 파이프라인에 대한 영향

**없음**. 기존 `docker-compose.yml`과 `gr00t-smmg-bp:1.0` 이미지는 **전혀 수정하지 않습니다**.
G1 전용 `docker-compose.g1.yml`에서만 새 이미지를 사용합니다.

---

## 참조

- Isaac Lab Docker Guide: https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html
- Isaac Lab pip install: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html
- Isaac Sim NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
- Blackwell GPU issue: https://github.com/isaac-sim/IsaacLab/discussions/3612
- Isaac Lab releases: https://github.com/isaac-sim/IsaacLab/releases
