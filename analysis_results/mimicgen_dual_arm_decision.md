# Phase 0-3: MimicGen 듀얼 암 지원 방식 결정

## 조사 일자: 2026-02-24

---

## 핵심 발견: Isaac Lab 2.3+에서 듀얼 암 공식 지원

### 결론: **옵션 A (Isaac Lab 업그레이드)가 명확한 최적 선택**

Isaac Lab 2.3+에서 다음이 모두 공식 지원됩니다:

| 기능 | 지원 여부 | 출처 |
|------|:--------:|------|
| **Multi-EEF MimicGen** | **O** | `isaaclab_mimic.datagen` API |
| **SubTaskConstraintConfig** | **O** | 듀얼 암 subtask 동기화/순서 제어 |
| **G1 locomanipulation 환경** | **O** | PR #3150, `locomanipulation_g1_env_cfg.py` |
| **G1 pick-place 환경** | **O** | `pickplace_unitree_g1_inspire_hand_env_cfg.py` |
| **Bimanual Pink IK** | **O** | 양팔 자연스러운 자세 유지 |
| **SkillGen** | **O** | MimicGen 대체/보완, cuRobo 통합 |
| **DexMimicGen** | **O** | 20,000+ 데모 자동 생성 (60개 소스에서) |
| **G1 Dex3 teleoperation** | **O** | Apple Vision Pro 지원 |

---

## 1. Multi-EEF MimicGen 아키텍처 (Isaac Lab 2.3+)

### 주요 API 변경사항

현재 버전 (0.36.2)의 제한:
```python
# data_generator.py:50-51
if len(subtask_configs) != 1:
    raise ValueError("Data generation currently supports only one end-effector.")
```

Isaac Lab 2.3+에서는:
```python
# 다중 EEF 지원 메서드
DataGenerator.generate_eef_subtask_trajectory(eef_name="left_wrist")
DataGenerator.merge_eef_subtask_trajectory(eef_name="right_wrist")
DataGenerator.select_source_demo(eef_name="left_wrist")
```

### SubTaskConstraintConfig (듀얼 암 동기화)

```python
from isaaclab_mimic.datagen import SubTaskConstraintConfig, SubTaskConstraintType

# 왼팔 2번째 subtask와 오른팔 2번째 subtask가 동시에 끝나도록 강제
constraint = SubTaskConstraintConfig(
    eef_subtask_constraint_tuple=(("left", 2), ("right", 2)),
    constraint_type=SubTaskConstraintType.COORDINATION
)
```

### 환경 인터페이스 (Custom Manipulation Env 필수 메서드)

```python
class G1ManipMimicEnv(ManagerBasedRLMimicEnv):
    def get_robot_eef_pose(self, eef_name, env_ids):
        """각 EEF의 현재 pose 반환"""
    def target_eef_pose_to_action(self, target_eef_pose, eef_name, env_ids):
        """목표 EEF pose를 IK 액션으로 변환"""
    def action_to_target_eef_pose(self, action, eef_name, env_ids):
        """액션에서 목표 EEF pose 역계산"""
    def actions_to_gripper_actions(self, actions, eef_name, env_ids):
        """그리퍼 액션 추출"""
    def get_object_poses(self, env_ids):
        """태스크 관련 물체 pose 반환"""
    def get_subtask_term_signals(self, env_ids):
        """각 subtask의 종료 신호 반환"""
    def get_subtask_start_signals(self, env_ids):
        """각 subtask의 시작 신호 반환"""
```

## 2. G1 Locomanipulation 환경 (이미 구현됨)

Isaac Lab 2.3+의 `locomanipulation_g1_env_cfg.py`:

| 항목 | 설정 |
|------|------|
| 로봇 | `G1_29DOF_CFG` (29 DoF) |
| 왼팔 EEF | `left_wrist_yaw_link` |
| 오른팔 EEF | `right_wrist_yaw_link` |
| IK 컨트롤러 | Pink IK (bimanual) |
| 하체 제어 | RL 기반 locomotion policy (사전 학습) |
| 손 관절 | `.*_hand.*` 패턴 |
| URDF | `g1_29dof_with_hand_only_kinematics.urdf` |

### Registered Environments

- `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` — G1 이동+조작
- `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` — G1 고정 기반 상체 IK

## 3. DexMimicGen 논문 핵심 결과

| 메트릭 | 값 |
|--------|-----|
| 소스 데모 수 | **60개** (인간 수집) |
| 생성된 데모 수 | **20,000+** |
| 대상 로봇 | Bimanual dexterous (양팔 민첩 손) |
| 태스크 수 | 9개 |
| 시뮬레이터 | 복수 (실제 로봇 포함) |

DexMimicGen은 MimicGen의 확장으로, 단일 팔 + 평행 조 그리퍼에서 **양팔 + 민첩 손**으로 적용 범위를 넓혔습니다.

## 4. 옵션별 최종 비교

| 옵션 | 실현 가능성 | 개발 비용 | 기능 완성도 | **권장** |
|------|:----------:|:--------:|:----------:|:--------:|
| **A. Isaac Lab 2.3 업그레이드** | **높음** | **낮음** (이미 구현됨) | **높음** (공식 지원) | **강력 권장** |
| B. 순차적 단일 암 | 높음 | 중 | 낮음 | Task 3,4 한정 |
| C. MimicGen 직접 확장 | 중 | **높음** | 중 | 불필요 (A 가능 시) |
| D. 팔별 독립 파이프라인 | 중 | 중 | 낮음 | 불필요 |

## 5. 결정: Isaac Lab 2.3+ 업그레이드

### 업그레이드 전략

1. **새 Docker 이미지 확보**: Isaac Lab 2.3+ 기반 이미지
   - 공식: `nvcr.io/nvidia/isaac-lab:2.3.x` (존재 확인 필요)
   - 또는: Isaac Sim 5.x pip 설치 + Isaac Lab 2.3 소스 빌드
2. **기존 Franka 파이프라인 보존**: 기존 Docker 이미지(`gr00t-smmg-bp:1.0`)는 그대로 유지
3. **G1 전용 환경**: Isaac Lab 2.3+의 기존 G1 환경을 커스터마이즈

### 업그레이드 시 해결되는 문제들

| 문제 | 해결 방법 |
|------|----------|
| MimicGen single-EEF 제한 | 2.3+에서 multi-EEF 지원 |
| G1 IK 제어 | Pink IK bimanual 지원 |
| Subtask 동기화 | SubTaskConstraintConfig |
| G1 manipulation 환경 | 이미 구현됨 (`locomanipulation_g1_env_cfg.py`) |
| G1 USD 관절 이름 차이 | 2.3+에서 최신 URDF 기반 config 사용 |

### Franka ↔ G1 전환 시 Docker 이미지 차이

```bash
# Franka (기존 - 변경 없음)
# 이미지: nvcr.io/nvidia/gr00t-smmg-bp:1.0
docker compose -f docker-compose.yml up -d

# G1 (새로운 - Isaac Lab 2.3+ 이미지)
# 이미지: nvcr.io/nvidia/isaac-lab:2.3.x (또는 커스텀 빌드)
docker compose -f docker-compose.g1.yml up -d
```

## 6. 다음 단계

1. **Isaac Lab 2.3+ Docker 이미지 확인/확보**
   - `nvcr.io/nvidia/isaac-lab` 태그 목록 확인
   - 또는 Isaac Sim 5.x + Isaac Lab 2.3 소스로 커스텀 이미지 빌드
2. **G1 locomanipulation 환경 커스터마이즈**
   - 기존 pick-place 환경을 4가지 주방 태스크에 맞게 수정
   - 물체 asset 추가 (과일, 바구니, 냄비, 인덕션)
3. **SubTaskConstraintConfig 설계**
   - Task 1 (handover): left pick → handover (COORDINATION) → right place
   - Task 2 (wipe): right pick → right place → left pick → left wipe
4. **Franka 호환성 검증**
   - 기존 이미지에서 Franka 파이프라인이 정상 동작하는지 확인

---

## 참조

- Isaac Lab 2.3 Blog: https://developer.nvidia.com/blog/streamline-robot-learning-with-whole-body-control-and-enhanced-teleoperation-in-nvidia-isaac-lab-2-3/
- DexMimicGen Paper: https://arxiv.org/abs/2410.24185
- DexMimicGen Website: https://dexmimicgen.github.io/
- Isaac Lab Mimic API: https://isaac-sim.github.io/IsaacLab/main/source/api/lab_mimic/isaaclab_mimic.datagen.html
- G1 locomanipulation PR: https://github.com/isaac-sim/IsaacLab/pull/3150
- G1 env config: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py
- SubTaskConstraintConfig discussion: https://github.com/isaac-sim/IsaacLab/discussions/3126
